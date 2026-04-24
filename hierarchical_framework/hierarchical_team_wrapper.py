from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from hierarchical_framework.commander_action import (
    ATTACK,
    DEFEND,
    INTERCEPT,
    ROLE_NAMES,
    ROLE_CONFIGS,
    ScriptedCommander,
)
from hierarchical_framework.commander_observation import (
    build_commander_observation,
    infer_commander_observation_length,
)
from hierarchical_framework.commander_rewards import (
    compute_commander_reward,
    compute_macro_reward,
    extract_macro_stats_from_state,
)
from hierarchical_framework.worker_rewards import shaped_worker_reward


class SharedEncoderRoleHeadModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_cfg = model_config.get("custom_model_config", {})
        hidden_dim = int(custom_cfg.get("hidden_dim", 256))
        head_dim = int(custom_cfg.get("head_dim", 128))
        self.role_dim = 3

        obs_dim = int(np.product(obs_space.shape))
        if obs_dim <= self.role_dim:
            raise ValueError(f"Observation dim {obs_dim} is too small for role_dim={self.role_dim}")
        self.core_obs_dim = obs_dim - self.role_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.core_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attack_policy_head = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, num_outputs))
        self.defend_policy_head = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, num_outputs))
        self.intercept_policy_head = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, num_outputs))
        self.attack_value_head = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))
        self.defend_value_head = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))
        self.intercept_value_head = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
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
        logits_stack = torch.stack([attack_logits, defend_logits, intercept_logits], dim=1)
        value_stack = torch.stack([attack_value, defend_value, intercept_value], dim=1)
        selected_logits = torch.sum(logits_stack * role_one_hot.unsqueeze(-1), dim=1)
        self._value_out = torch.sum(value_stack * role_one_hot, dim=1)
        return selected_logits, state

    def value_function(self):
        if self._value_out is None:
            raise ValueError("value_function() called before forward()")
        return self._value_out


def _deepcopy_state(state: dict) -> dict:
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
    vec = np.zeros(3, dtype=np.float32)
    if role_id in (ATTACK, DEFEND, INTERCEPT):
        vec[role_id] = 1.0
    return vec


class HierarchicalTeamWrapper:
    def __init__(
        self,
        base_env,
        blue_team: Tuple[str, str, str] = ("agent_0", "agent_1", "agent_2"),
        red_team: Tuple[str, str, str] = ("agent_3", "agent_4", "agent_5"),
        role_period: int = 10,
        max_time: int = 300,
        shape_blue_worker_rewards: bool = True,
        env_config: dict | None = None,
    ):
        self.base_env = base_env
        self.blue_team = list(blue_team)
        self.red_team = list(red_team)
        self.role_period = int(role_period)
        self.max_time = int(max_time)
        self.shape_blue_worker_rewards = bool(shape_blue_worker_rewards)
        self.enable_boundary_guard = True
        self.boundary_margin = 0.10
        self.boundary_hard_margin = 0.04
        self.carrier_boundary_margin = 0.14

        # Pair each faster action with its slower counterpart.
        self.action_slowdown_map = {
            0: 8,  1: 9,  2: 10, 3: 11,
            4: 12, 5: 13, 6: 14, 7: 15,
            8: 16, 9: 16, 10: 16, 11: 16,
            12: 16, 13: 16, 14: 16, 15: 16,
            16: 16,
        }

        # Bias steering downward in world-frame y (use when too close to top edge).
        self.turn_away_from_top_map = {
            5: 11,
            6: 10,
            7: 8,
            13: 10,
            14: 8,
            15: 8,
            4: 10,
            12: 16,
        }

        # Bias steering upward in world-frame y (use when too close to bottom edge).
        self.turn_away_from_bottom_map = {
            3: 13,
            2: 14,
            0: 15,
            1: 15,
            11: 14,
            10: 15,
            8: 15,
            4: 14,
            12: 16,
        }

        
        self.env_config = env_config or {}

        self.use_learned_commander = bool(self.env_config.get("use_learned_commander", False))
        self.commander_team = self.env_config.get("commander_team", "blue")
        self.commander_agent_id = "blue_commander"

        self.blue_commander = ScriptedCommander(self.blue_team, self.red_team, home_side_idx=0)
        self.red_commander = ScriptedCommander(self.red_team, self.blue_team, home_side_idx=1)

        self.current_roles: Dict[str, int] = {aid: ATTACK for aid in (self.blue_team + self.red_team)}
        self.last_blue_config_index = None
        self.last_blue_reason = "RESET_DEFAULT"
        self.last_red_config_index = None
        self.last_red_reason = "RESET_DEFAULT"
        self.steps_since_role_update = 0
        self.elapsed_steps = 0
        self.prev_state = None
        self.prev_macro_stats = {}
        self.curr_macro_stats = {}
        self.last_macro_reward = 0.0

        self.possible_agents = list(getattr(base_env, "possible_agents", []))
        self.agents: List[str] = []
        self._base_obs_space = self._infer_base_observation_space()
        self._base_act_space = self._infer_base_action_space()

        self.observation_spaces = {}
        self.action_spaces = {}
        for aid in self.possible_agents:
            low = np.full((self._base_obs_space.shape[0] + 3,), -np.inf, dtype=np.float32)
            high = np.full((self._base_obs_space.shape[0] + 3,), np.inf, dtype=np.float32)
            self.observation_spaces[aid] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            self.action_spaces[aid] = self._base_act_space

        self.commander_observation_length = infer_commander_observation_length(len(self.blue_team), len(self.red_team))
        if self.use_learned_commander and self.commander_team == "blue":
            self.observation_spaces[self.commander_agent_id] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.commander_observation_length,),
                dtype=np.float32,
            )
            self.action_spaces[self.commander_agent_id] = gym.spaces.Discrete(len(ROLE_CONFIGS))

    def _infer_base_observation_space(self):
        if hasattr(self.base_env, "observation_spaces") and "agent_0" in self.base_env.observation_spaces:
            return self.base_env.observation_spaces["agent_0"]
        raise RuntimeError("Could not infer base observation space from wrapped environment")

    def _infer_base_action_space(self):
        if hasattr(self.base_env, "action_spaces") and "agent_0" in self.base_env.action_spaces:
            return self.base_env.action_spaces["agent_0"]
        raise RuntimeError("Could not infer base action space from wrapped environment")

    def _get_state(self) -> dict:
        if hasattr(self.base_env, "state"):
            return self.base_env.state
        if hasattr(self.base_env, "par_env") and hasattr(self.base_env.par_env, "state"):
            return self.base_env.par_env.state
        if hasattr(self.base_env, "aec_env") and hasattr(self.base_env.aec_env, "state"):
            return self.base_env.aec_env.state
        raise RuntimeError("Could not locate underlying environment state")

    def _apply_boundary_guard(self, action_dict: dict, state: dict) -> dict:
        if not self.enable_boundary_guard or "agent_position" not in state:
            return action_dict

        guarded = dict(action_dict)

        for aid, action in list(action_dict.items()):
            if aid not in getattr(self.base_env, "agents", []):
                continue

            if not isinstance(action, (int, np.integer)):
                # Continuous fallback.
                try:
                    i = self.base_env.agents.index(aid)
                    x, y = map(float, state["agent_position"][i])

                    has_flag = bool(state["agent_has_flag"][i]) if "agent_has_flag" in state else False
                    soft_margin = self.carrier_boundary_margin if has_flag else self.boundary_margin

                    a = np.array(action, dtype=np.float32).copy()
                    if a.shape[0] >= 2:
                        if x < soft_margin:
                            a[0] = abs(a[0])
                        if x > 1.0 - soft_margin:
                            a[0] = -abs(a[0])
                        if y < soft_margin:
                            a[1] = abs(a[1])
                        if y > 1.0 - soft_margin:
                            a[1] = -abs(a[1])
                        guarded[aid] = a
                except Exception:
                    pass
                continue

            a = int(action)
            i = self.base_env.agents.index(aid)
            x, y = map(float, state["agent_position"][i])

            has_flag = bool(state["agent_has_flag"][i]) if "agent_has_flag" in state else False
            soft_margin = self.carrier_boundary_margin if has_flag else self.boundary_margin
            hard_margin = self.boundary_hard_margin

            # Estimate recent motion direction from prev_state -> state.
            dx = 0.0
            dy = 0.0
            if self.prev_state is not None and "agent_position" in self.prev_state:
                try:
                    prev_x, prev_y = map(float, self.prev_state["agent_position"][i])
                    dx = x - prev_x
                    dy = y - prev_y
                except Exception:
                    pass

            near_left_soft = x < soft_margin
            near_right_soft = x > 1.0 - soft_margin
            near_bottom_soft = y < soft_margin
            near_top_soft = y > 1.0 - soft_margin

            near_left_hard = x < hard_margin
            near_right_hard = x > 1.0 - hard_margin
            near_bottom_hard = y < hard_margin
            near_top_hard = y > 1.0 - hard_margin

            # Soft zone: if motion is outward, reduce thrust.
            if near_left_soft and dx < -0.001:
                guarded[aid] = self.action_slowdown_map.get(a, 16)
                continue
            if near_right_soft and dx > 0.001:
                guarded[aid] = self.action_slowdown_map.get(a, 16)
                continue
            if near_bottom_soft and dy < -0.001:
                guarded[aid] = self.action_slowdown_map.get(a, 16)
                continue
            if near_top_soft and dy > 0.001:
                guarded[aid] = self.action_slowdown_map.get(a, 16)
                continue

            # Hard zone: stronger intervention.
            if near_left_hard and dx < -0.001:
                guarded[aid] = 16
                continue
            if near_right_hard and dx > 0.001:
                guarded[aid] = 16
                continue
            if near_bottom_hard and dy < -0.001:
                guarded[aid] = self.turn_away_from_bottom_map.get(a, 16)
                continue
            if near_top_hard and dy > 0.001:
                guarded[aid] = self.turn_away_from_top_map.get(a, 16)
                continue

        return guarded

    def _apply_scripted_roles(self):
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
            self.last_macro_reward = compute_macro_reward(self.prev_macro_stats, self.curr_macro_stats, role_changed)
        else:
            self.last_macro_reward = 0.0
        self.prev_macro_stats = dict(self.curr_macro_stats)

    def _apply_learned_blue_roles(self, commander_action: int):
        state = self._get_state()
        agents = list(getattr(self.base_env, "agents", self.blue_team + self.red_team))
        if hasattr(self.blue_commander, "assign_roles_from_config_index"):
            blue_assignments, blue_reason = self.blue_commander.assign_roles_from_config_index(state, agents, int(commander_action))
            blue_config_index = int(commander_action)
        else:
            blue_assignments, blue_config_index, blue_reason = self.blue_commander.assign_roles(state, agents)
        red_assignments, red_config_index, red_reason = self.red_commander.assign_roles(state, agents)
        prev_blue_roles = {aid: self.current_roles.get(aid, ATTACK) for aid in self.blue_team}
        role_changed = blue_assignments != prev_blue_roles
        for aid in self.blue_team:
            self.current_roles[aid] = int(blue_assignments[aid])
        for aid in self.red_team:
            self.current_roles[aid] = int(red_assignments[aid])
        self.last_blue_config_index = blue_config_index
        self.last_blue_reason = blue_reason if blue_reason else "LEARNED_POLICY"
        self.last_red_config_index = red_config_index
        self.last_red_reason = red_reason
        self.steps_since_role_update = 0
        self.curr_macro_stats = extract_macro_stats_from_state(state)
        if self.prev_macro_stats:
            self.last_macro_reward = compute_macro_reward(self.prev_macro_stats, self.curr_macro_stats, role_changed)
        else:
            self.last_macro_reward = 0.0
        self.prev_macro_stats = dict(self.curr_macro_stats)

    def _augment_worker_observation(self, agent_id: str, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        role_vec = _role_one_hot(self.current_roles.get(agent_id, ATTACK))
        return np.concatenate([obs, role_vec], axis=0)

    def _build_debug_commander_observation(self) -> np.ndarray:
        state = self._get_state()
        agents = list(getattr(self.base_env, "agents", self.blue_team + self.red_team))
        return build_commander_observation(state=state, agents=agents, team_idx=0, max_time=self.max_time)

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.current_roles = {aid: ATTACK for aid in (self.blue_team + self.red_team)}
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
        self._apply_scripted_roles()

        wrapped_obs = {}
        wrapped_info = {k: dict(v) if isinstance(v, dict) else {} for k, v in info.items()} if isinstance(info, dict) else {}
        for aid, aobs in obs.items():
            wrapped_obs[aid] = self._augment_worker_observation(aid, aobs)
            wrapped_info.setdefault(aid, {})
            wrapped_info[aid]["role_id"] = self.current_roles[aid]
            wrapped_info[aid]["role_name"] = ROLE_NAMES[self.current_roles[aid]]
            if aid in self.blue_team:
                wrapped_info[aid]["commander_reason"] = self.last_blue_reason
                wrapped_info[aid]["commander_config_index"] = self.last_blue_config_index
            else:
                wrapped_info[aid]["commander_reason"] = self.last_red_reason
                wrapped_info[aid]["commander_config_index"] = self.last_red_config_index

        if self.use_learned_commander and self.commander_team == "blue":
            wrapped_obs[self.commander_agent_id] = build_commander_observation(self._get_state(), self.base_env.agents, 0, max_time=self.max_time)
            wrapped_info[self.commander_agent_id] = {}

        self.agents = list(wrapped_obs.keys())
        return wrapped_obs, wrapped_info

    def step(self, actions: Dict[str, int]):
        action_dict = dict(actions)
        commander_action = None
        if self.use_learned_commander and self.commander_agent_id in action_dict:
            commander_action = int(action_dict.pop(self.commander_agent_id))

        if self.steps_since_role_update % self.role_period == 0:
            if self.use_learned_commander and self.commander_team == "blue" and commander_action is not None:
                self._apply_learned_blue_roles(commander_action)
            else:
                self._apply_scripted_roles()

        pre_step_state = _deepcopy_state(self._get_state())
        action_dict = self._apply_boundary_guard(action_dict, pre_step_state)
        obs, rewards, terminated, truncated, info = self.base_env.step(action_dict)
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
            base_reward = float(rewards.get(aid, 0.0))
            shaped_reward = base_reward
            if self.shape_blue_worker_rewards and aid in self.blue_team and self.prev_state is not None:
                shaped_reward = shaped_worker_reward(
                    role_id=self.current_roles[aid],
                    base_reward=base_reward,
                    agent_id=aid,
                    team=0,
                    agents=list(getattr(self.base_env, "agents", self.blue_team + self.red_team)),
                    state=state,
                    prev_state=self.prev_state,
                )
            wrapped_rewards[aid] = shaped_reward
            wrapped_terminated[aid] = terminated.get(aid, False)
            wrapped_truncated[aid] = truncated.get(aid, False)
            wrapped_info[aid] = info.get(aid, {}) if isinstance(info, dict) else {}
            wrapped_info[aid]["role_id"] = self.current_roles[aid]
            wrapped_info[aid]["role_name"] = ROLE_NAMES[self.current_roles[aid]]
            if aid in self.blue_team:
                wrapped_info[aid]["commander_reason"] = self.last_blue_reason
                wrapped_info[aid]["commander_config_index"] = self.last_blue_config_index
                wrapped_info[aid]["commander_macro_reward"] = self.last_macro_reward
            else:
                wrapped_info[aid]["commander_reason"] = self.last_red_reason
                wrapped_info[aid]["commander_config_index"] = self.last_red_config_index

        if self.use_learned_commander and self.commander_team == "blue":
            wrapped_obs[self.commander_agent_id] = build_commander_observation(state, self.base_env.agents, 0, max_time=self.max_time)
            wrapped_rewards[self.commander_agent_id] = compute_commander_reward(self.prev_state, state, 0, self.last_blue_config_index or 0)
            wrapped_terminated[self.commander_agent_id] = bool(terminated.get("__all__", False))
            wrapped_truncated[self.commander_agent_id] = bool(truncated.get("__all__", False))
            wrapped_info[self.commander_agent_id] = {
                "commander_config_index": self.last_blue_config_index,
                "commander_reason": self.last_blue_reason,
            }

        self.prev_state = _deepcopy_state(state)
        self.agents = list(wrapped_obs.keys())
        return wrapped_obs, wrapped_rewards, wrapped_terminated, wrapped_truncated, wrapped_info

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()
