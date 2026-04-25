import os
import numpy as np
from ray.rllib.policy.policy import Policy

from commander_action import ATTACK, DEFEND, INTERCEPT, ScriptedCommander
from commander_observation import build_commander_observation


ROLE_PERIOD = 5
TAG_COOLDOWN_LIMIT = 45.0


def _role_one_hot(role_id: int) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float32)
    if role_id in (ATTACK, DEFEND, INTERCEPT):
        vec[role_id] = 1.0
    return vec


class solution:
    """
    Competition submission wrapper for the hierarchical checkpoint.

    Loads:
      - checkpoints/policies/worker_policy
      - checkpoints/policies/commander_policy

    Uses the learned commander to pick a role configuration every ROLE_PERIOD
    leader-agent calls, then appends the corresponding role one-hot vector to the
    normalized worker observation before querying the learned worker policy.

    Falls back to the scripted commander if commander_policy is unavailable.
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.worker_policy = Policy.from_checkpoint(
            os.path.join(base_dir, "checkpoints", "policies", "worker_policy")
        )

        commander_path = os.path.join(base_dir, "checkpoints", "policies", "commander_policy")
        self.commander_policy = None
        if os.path.exists(commander_path):
            try:
                self.commander_policy = Policy.from_checkpoint(commander_path)
            except Exception:
                self.commander_policy = None

        self.team_agents = {
            0: ["agent_0", "agent_1", "agent_2"],
            1: ["agent_3", "agent_4", "agent_5"],
        }
        self.enemy_agents = {
            0: ["agent_3", "agent_4", "agent_5"],
            1: ["agent_0", "agent_1", "agent_2"],
        }
        self.leader_agent = {0: "agent_0", 1: "agent_3"}

        self.commanders = {
            0: ScriptedCommander(self.team_agents[0], self.enemy_agents[0], home_side_idx=0),
            1: ScriptedCommander(self.team_agents[1], self.enemy_agents[1], home_side_idx=1),
        }
        self.current_roles = {
            0: {aid: ATTACK for aid in self.team_agents[0]},
            1: {aid: ATTACK for aid in self.team_agents[1]},
        }
        self.commander_call_count = {0: 0, 1: 0}

    def _compute_team_idx(self, agent_id: str) -> int:
        idx = int(agent_id.split("_")[1])
        return 0 if idx < 3 else 1

    def _compute_commander_action(self, team_idx: int, global_state: dict):
        if self.commander_policy is None:
            return None
        agents = self.team_agents[0] + self.team_agents[1]
        obs = build_commander_observation(global_state, agents, team_idx, max_time=600.0, tag_cd_limit=TAG_COOLDOWN_LIMIT)
        act = self.commander_policy.compute_single_action(obs, explore=False)
        return act[0] if isinstance(act, tuple) else act

    def _maybe_refresh_roles(self, team_idx: int, agent_id: str, global_state: dict):
        # Refresh only on leader-agent calls to approximate one commander update per env step.
        if agent_id != self.leader_agent[team_idx]:
            return

        self.commander_call_count[team_idx] += 1
        if self.commander_call_count[team_idx] == 1 or (self.commander_call_count[team_idx] - 1) % ROLE_PERIOD == 0:
            agents = self.team_agents[0] + self.team_agents[1]
            commander = self.commanders[team_idx]
            learned_action = self._compute_commander_action(team_idx, global_state)

            if learned_action is not None and hasattr(commander, "assign_roles_from_config_index"):
                assignments, _reason = commander.assign_roles_from_config_index(global_state, agents, int(learned_action))
            else:
                assignments, _cfg_idx, _reason = commander.assign_roles(global_state, agents)
            self.current_roles[team_idx] = assignments

    def compute_action(self, agent_id: str, full_obs_normalized: dict, full_obs: dict, global_state: dict):
        team_idx = self._compute_team_idx(agent_id)
        self._maybe_refresh_roles(team_idx, agent_id, global_state)

        obs = np.asarray(full_obs_normalized[agent_id], dtype=np.float32)
        role_id = self.current_roles[team_idx].get(agent_id, ATTACK)
        worker_obs = np.concatenate([obs, _role_one_hot(role_id)], axis=0)

        act = self.worker_policy.compute_single_action(worker_obs, explore=False)
        return act[0] if isinstance(act, tuple) else act
