"""
hierarchical_framework/hierarchical_team_wrapper.py

Purpose of this module:
- Wrap the base Pyquaticus environment so the blue team is controlled by:
    * a scripted commander baseline internally, and
    * a shared worker PPO policy externally.
- Append role one-hot to blue worker observations.
- Apply worker reward shaping for blue workers.
- Expose role assignment and commander reason through infos for debugging.

Why this file is the center of the current baseline:
- This is where the role hierarchy becomes real.
- The training script does not train a commander policy yet.
- Instead, this wrapper updates current_roles every role_period using the
  scripted commander.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class SharedEncoderRoleHeadModel(TorchModelV2, nn.Module):
    """
    One shared encoder plus three role-specific policy/value heads.

    Purpose:
    - Keep sample efficiency by sharing most of the network.
    - Reduce role interference by giving ATTACK / DEFEND / INTERCEPT
      their own decision heads.
    - Use the role one-hot already appended by the wrapper to select
      which head is active on each forward pass.

    Important assumption:
    - The last 3 entries of the blue observation are the role one-hot:
        [attack, defend, intercept]
    - This model is intended for the blue worker policy only.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_cfg = model_config.get("custom_model_config", {})
        hidden_dim = int(custom_cfg.get("hidden_dim", 256))
        head_dim = int(custom_cfg.get("head_dim", 128))
        self.role_dim = 3

        obs_dim = int(np.product(obs_space.shape))
        if obs_dim <= self.role_dim:
            raise ValueError(
                f"Observation dim {obs_dim} is too small for role_dim={self.role_dim}"
            )

        self.core_obs_dim = obs_dim - self.role_dim

        # Shared encoder for all roles.
        self.encoder = nn.Sequential(
            nn.Linear(self.core_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Role-specific policy heads.
        self.attack_policy_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_outputs),
        )
        self.defend_policy_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_outputs),
        )
        self.intercept_policy_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_outputs),
        )

        # Role-specific value heads.
        self.attack_value_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self.defend_value_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self.intercept_value_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass for PPO.

        Purpose:
        - Split observation into core features and role one-hot.
        - Compute one shared embedding.
        - Compute logits/value from all three heads.
        - Select the active head using the role one-hot.
        """
        obs = input_dict["obs_flat"].float()

        core_obs = obs[:, :self.core_obs_dim]
        role_one_hot = obs[:, self.core_obs_dim:self.core_obs_dim + self.role_dim]

        embedding = self.encoder(core_obs)

        attack_logits = self.attack_policy_head(embedding)
        defend_logits = self.defend_policy_head(embedding)
        intercept_logits = self.intercept_policy_head(embedding)

        attack_value = self.attack_value_head(embedding).squeeze(-1)
        defend_value = self.defend_value_head(embedding).squeeze(-1)
        intercept_value = self.intercept_value_head(embedding).squeeze(-1)

        # Stack outputs by role dimension.
        logits_stack = torch.stack(
            [attack_logits, defend_logits, intercept_logits],
            dim=1,
        )  # [B, 3, A]

        value_stack = torch.stack(
            [attack_value, defend_value, intercept_value],
            dim=1,
        )  # [B, 3]

        # Weighted selection via role one-hot.
        selected_logits = torch.sum(logits_stack * role_one_hot.unsqueeze(-1), dim=1)
        selected_value = torch.sum(value_stack * role_one_hot, dim=1)

        self._value_out = selected_value
        return selected_logits, state

    def value_function(self):
        """
        Return the value prediction associated with the selected role head.
        """
        if self._value_out is None:
            raise ValueError("value_function() called before forward()")
        return self._value_out

from hierarchical_framework.commander_action import (
    ATTACK,
    DEFEND,
    INTERCEPT,
    ROLE_NAMES,
    ScriptedCommander,
)
from hierarchical_framework.commander_observation import (
    build_commander_observation,
    infer_commander_observation_length,
)
from hierarchical_framework.commander_rewards import (
    compute_macro_reward,
    extract_macro_stats_from_state,
)
from hierarchical_framework.worker_rewards import shaped_worker_reward


def _deepcopy_state(state: dict) -> dict:
    """
    Make a safe copy of the environment state for reward delta calculations.

    Why this method exists:
    - Worker reward shaping compares current and previous state.
    - A shallow reference to mutable arrays would break those comparisons.
    """
    try:
        return deepcopy(state)
    except Exception:
        copied = {}
        for k, v in state.items():
            try:
                copied[k] = np.copy(v)
            except Exception:
                copied[k] = v
        return copied


def _role_one_hot(role_id: int) -> np.ndarray:
    """
    Convert a role ID into a one-hot vector.

    Why this method exists:
    - The shared worker PPO gets role-conditioning through appended
      one-hot role features.
    """
    vec = np.zeros(3, dtype=np.float32)
    if role_id in (ATTACK, DEFEND, INTERCEPT):
        vec[role_id] = 1.0
    return vec


class HierarchicalTeamWrapper:
    """
    Parallel-env style wrapper implementing scripted commander + shared workers.

    What this wrapper does:
    - Uses a scripted commander internally for blue role assignment.
    - Leaves red team as ordinary primitive agents.
    - Appends current blue role one-hot to each blue worker observation.
    - Shapes blue worker rewards using role-specific reward functions.
    - Logs commander reason, config index, and current role assignments.
    """

    def __init__(
        self,
        base_env,
        blue_team: Tuple[str, str, str] = ("agent_0", "agent_1", "agent_2"),
        red_team: Tuple[str, str, str] = ("agent_3", "agent_4", "agent_5"),
        role_period: int = 10,
        max_time: int = 300,
        shape_blue_worker_rewards: bool = True,
    ):
        """
        Initialize wrapper state and spaces.

        Why this method exists:
        - It stores team membership, refresh interval, scripted commander,
          reward-shaping settings, and observation/action spaces.
        """
        self.base_env = base_env
        self.blue_team = list(blue_team)
        self.red_team = list(red_team)
        self.role_period = int(role_period)
        self.max_time = int(max_time)
        self.shape_blue_worker_rewards = bool(shape_blue_worker_rewards)

        #self.scripted_commander = ScriptedCommander(self.blue_team, self.red_team)
        self.blue_commander = ScriptedCommander(self.blue_team, self.red_team, home_side_idx=0)
        self.red_commander = ScriptedCommander(self.red_team, self.blue_team, home_side_idx=1)

        self.current_roles: Dict[str, int] = {}
        for aid in self.blue_team:
            self.current_roles[aid] = ATTACK
        for aid in self.red_team:
            self.current_roles[aid] = ATTACK

        self.last_blue_config_index = None
        self.last_blue_reason = "RESET_DEFAULT"
        self.last_red_config_index = None
        self.last_red_reason = "RESET_DEFAULT"

        self.possible_agents = list(getattr(base_env, "possible_agents", []))
        self.agents: List[str] = []

        # Blue roles are assigned dynamically by the scripted commander.
        # Red roles are fixed for now so a frozen checkpoint can run on red too.
        self.static_red_roles: Dict[str, int] = {
            self.red_team[0]: ATTACK,
            self.red_team[1]: DEFEND,
            self.red_team[2]: INTERCEPT,
        }

        self.current_roles: Dict[str, int] = {aid: ATTACK for aid in self.blue_team}
        self.current_roles.update(self.static_red_roles)


        self.last_config_index = None
        self.last_reason = "RESET_DEFAULT"
        self.steps_since_role_update = 0
        self.elapsed_steps = 0

        self.prev_state = None
        self.prev_macro_stats = {}
        self.curr_macro_stats = {}
        self.last_macro_reward = 0.0

        # Infer base spaces from the base env.
        self._base_obs_space = self._infer_base_observation_space()
        self._base_act_space = self._infer_base_action_space()

        # Build observation and action space dictionaries for the wrapped env.
        self.observation_spaces = {}
        self.action_spaces = {}

        """
        for aid in self.possible_agents:
            if aid in self.blue_team:
                low = np.full((self._base_obs_space.shape[0] + 3,), -np.inf, dtype=np.float32)
                high = np.full((self._base_obs_space.shape[0] + 3,), np.inf, dtype=np.float32)
                self.observation_spaces[aid] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            else:
                self.observation_spaces[aid] = self._base_obs_space

            self.action_spaces[aid] = self._base_act_space
        """

        for aid in self.possible_agents:
            low = np.full((self._base_obs_space.shape[0] + 3,), -np.inf, dtype=np.float32)
            high = np.full((self._base_obs_space.shape[0] + 3,), np.inf, dtype=np.float32)
            self.observation_spaces[aid] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            self.action_spaces[aid] = self._base_act_space

        # Commander-style observation is not exposed as an RL agent in this baseline,
        # but we still store its length for debug/logging/future learned commander use.
        self.commander_observation_length = infer_commander_observation_length(
            blue_team_size=len(self.blue_team),
            red_team_size=len(self.red_team),
        )

    def _infer_base_observation_space(self):
        """
        Read the worker observation space from the base env.

        Why this method exists:
        - The wrapper needs to extend blue worker observations by +3 role features.
        """
        if hasattr(self.base_env, "observation_spaces") and "agent_0" in self.base_env.observation_spaces:
            return self.base_env.observation_spaces["agent_0"]
        raise RuntimeError("Could not infer base observation space from wrapped environment")

    def _infer_base_action_space(self):
        """
        Read the worker action space from the base env.

        Why this method exists:
        - Blue and red primitive agents still act in the normal Pyquaticus action space.
        """
        if hasattr(self.base_env, "action_spaces") and "agent_0" in self.base_env.action_spaces:
            return self.base_env.action_spaces["agent_0"]
        raise RuntimeError("Could not infer base action space from wrapped environment")

    def _get_state(self) -> dict:
        """
        Access the underlying Pyquaticus state dictionary.

        Why this method exists:
        - Commander logic and reward shaping both need direct state access.
        - This method keeps state lookup in one place in case your local env
          exposes it slightly differently.
        """
        if hasattr(self.base_env, "state"):
            return self.base_env.state
        if hasattr(self.base_env, "par_env") and hasattr(self.base_env.par_env, "state"):
            return self.base_env.par_env.state
        if hasattr(self.base_env, "aec_env") and hasattr(self.base_env.aec_env, "state"):
            return self.base_env.aec_env.state
        raise RuntimeError("Could not locate underlying environment state")

    def _apply_scripted_roles(self):
        """
        Refresh both blue and red role assignments using mirrored scripted commanders.
        """
        state = self._get_state()
        agents = list(getattr(self.base_env, "agents", self.blue_team + self.red_team))

        blue_assignments, blue_config_index, blue_reason = self.blue_commander.assign_roles(state, agents)
        red_assignments, red_config_index, red_reason = self.red_commander.assign_roles(state, agents)

        prev_blue_roles = {aid: self.current_roles.get(aid, ATTACK) for aid in self.blue_team}
        role_changed = blue_assignments != prev_blue_roles

        for aid in self.blue_team:
            self.current_roles[aid] = int(blue_assignments[aid])
        for aid in self.red_team:
            self.current_roles[aid] = int(red_assignments[aid])

        self.last_blue_config_index = blue_config_index
        self.last_blue_reason = blue_reason
        self.last_red_config_index = red_config_index
        self.last_red_reason = red_reason
        self.steps_since_role_update = 0

        self.curr_macro_stats = extract_macro_stats_from_state(state)
        if self.prev_macro_stats:
            self.last_macro_reward = compute_macro_reward(
                previous_stats=self.prev_macro_stats,
                current_stats=self.curr_macro_stats,
                role_assignment_changed=role_changed,
            )
        else:
            self.last_macro_reward = 0.0
        self.prev_macro_stats = dict(self.curr_macro_stats)

    def _augment_worker_observation(self, agent_id: str, obs: np.ndarray) -> np.ndarray:
        """
        Append role one-hot to all agent observations.

        Why this change exists:
        - Blue uses dynamic roles from the scripted commander.
        - Red uses static fixed roles for frozen-checkpoint opponents.
        - This makes both sides compatible with the same role-head worker model.
        """
        obs = np.asarray(obs, dtype=np.float32)
        role_vec = _role_one_hot(self.current_roles.get(agent_id, ATTACK))
        return np.concatenate([obs, role_vec], axis=0)

    def _build_debug_commander_observation(self) -> np.ndarray:
        """
        Build a commander-style observation vector for logging or future use.

        Why this method exists:
        - The scripted commander itself does not need this vector right now,
          but it is useful for debugging and future learned-commander work.
        """
        state = self._get_state()
        agents = list(getattr(self.base_env, "agents", self.blue_team + self.red_team))
        return build_commander_observation(
            state=state,
            agents=agents,
            blue_team=self.blue_team,
            red_team=self.red_team,
            current_roles=self.current_roles,
            previous_config_index=self.last_config_index,
            elapsed_steps=self.elapsed_steps,
            max_time=self.max_time,
        )

    def reset(self, seed=None, options=None):
        """
        Reset the base environment and initialize role assignments.

        Why this method exists:
        - A new episode should start from a valid commander-controlled role setup.
        - We also reset state tracking for reward shaping and commander logs.
        """
        obs, info = self.base_env.reset(seed=seed, options=options)
        """
        self.current_roles = {aid: ATTACK for aid in self.blue_team}
        self.current_roles.update(self.static_red_roles)
        self.last_config_index = None
        self.last_reason = "RESET_DEFAULT"
        """
        self.current_roles = {}
        for aid in self.blue_team:
            self.current_roles[aid] = ATTACK
        for aid in self.red_team:
            self.current_roles[aid] = ATTACK

        self.last_blue_config_index = None
        self.last_blue_reason = "RESET_DEFAULT"
        self.last_red_config_index = None
        self.last_red_reason = "RESET_DEFAULT"
        
        self.steps_since_role_update = 0
        self.elapsed_steps = 0

        state = self._get_state()
        self.prev_state = _deepcopy_state(state)
        self.prev_macro_stats = extract_macro_stats_from_state(state)
        self.curr_macro_stats = dict(self.prev_macro_stats)
        self.last_macro_reward = 0.0

        # Apply initial scripted roles immediately so workers start with meaningful assignments.
        self._apply_scripted_roles()

        wrapped_obs = {}
        wrapped_info = dict(info)

        for aid, aobs in obs.items():
            wrapped_obs[aid] = self._augment_worker_observation(aid, aobs)
            wrapped_info.setdefault(aid, {})
            if aid in self.blue_team:
                wrapped_info[aid]["role_id"] = self.current_roles[aid]
                wrapped_info[aid]["role_name"] = ROLE_NAMES[self.current_roles[aid]]
                wrapped_info[aid]["commander_reason"] = self.last_reason
                wrapped_info[aid]["commander_config_index"] = self.last_config_index

        self.agents = list(obs.keys())
        return wrapped_obs, wrapped_info

    def step(self, actions: Dict[str, int]):
        """
        Advance the environment one primitive step.

        Why this method exists:
        - It injects the commander refresh logic on a slower timescale than workers.
        - It applies reward shaping after the base env step.
        - It augments blue observations with role one-hot.
        """
        # Refresh blue roles at macro intervals before stepping workers.
        if self.steps_since_role_update % self.role_period == 0:
            self._apply_scripted_roles()

        obs, rewards, terminated, truncated, info = self.base_env.step(actions)

        self.elapsed_steps += 1
        self.steps_since_role_update += 1

        state = self._get_state()

        wrapped_obs = {}
        wrapped_rewards = {}
        wrapped_terminated = {}
        wrapped_truncated = {}
        wrapped_info = {}

        for aid, aobs in obs.items():
            wrapped_obs[aid] = self._augment_worker_observation(aid, aobs)

            base_reward = rewards.get(aid, 0.0)
            shaped_reward = float(base_reward)

            # Only blue workers get hierarchical role shaping in this baseline.
            if self.shape_blue_worker_rewards and aid in self.blue_team and self.prev_state is not None:
                role_id = self.current_roles[aid]
                team_index = 0  # blue
                agents = list(getattr(self.base_env, "agents", self.blue_team + self.red_team))
                shaped_reward = shaped_worker_reward(
                    role_id=role_id,
                    base_reward=base_reward,
                    agent_id=aid,
                    team=team_index,
                    agents=agents,
                    state=state,
                    prev_state=self.prev_state,
                )

            wrapped_rewards[aid] = shaped_reward
            wrapped_terminated[aid] = terminated.get(aid, False)
            wrapped_truncated[aid] = truncated.get(aid, False)
            wrapped_info[aid] = info.get(aid, {})

            if aid in self.blue_team:
                wrapped_info[aid]["role_id"] = self.current_roles[aid]
                wrapped_info[aid]["role_name"] = ROLE_NAMES[self.current_roles[aid]]
                wrapped_info[aid]["commander_reason"] = self.last_reason
                wrapped_info[aid]["commander_config_index"] = self.last_config_index
                wrapped_info[aid]["commander_macro_reward"] = self.last_macro_reward

            if aid in self.red_team:
                wrapped_info[aid]["role_id"] = self.current_roles[aid]
                wrapped_info[aid]["role_name"] = ROLE_NAMES[self.current_roles[aid]]
                wrapped_info[aid]["commander_reason"] = self.last_red_reason
                wrapped_info[aid]["commander_config_index"] = self.last_red_config_index

        self.prev_state = _deepcopy_state(state)
        self.agents = list(obs.keys())

        return wrapped_obs, wrapped_rewards, wrapped_terminated, wrapped_truncated, wrapped_info
    
    def observation_space(self, agent):
        """
        Return the observation space for a specific agent.

        Why this method exists:
        - RLlib's PettingZoo wrapper expects ParallelEnv-style space accessors
        named observation_space(agent) and action_space(agent).
        - The wrapper already stores observation_spaces as a dict, so this
        method simply exposes the expected interface.
        """
        return self.observation_spaces[agent]


    def action_space(self, agent):
        """
        Return the action space for a specific agent.

        Why this method exists:
        - RLlib's PettingZoo wrapper expects this accessor for each agent.
        - Blue and red workers both use the primitive action space from the
        wrapped Pyquaticus environment.
        """
        return self.action_spaces[agent]

    def render(self):
        """
        Forward render calls to the base environment.

        Why this method exists:
        - The training/eval scripts can optionally render without needing
          to know about the internal wrapper structure.
        """
        return self.base_env.render()

    def close(self):
        """
        Forward close calls to the base environment.

        Why this method exists:
        - Clean shutdown should be handled through the wrapper the same way
          as reset/step/render.
        """
        return self.base_env.close()