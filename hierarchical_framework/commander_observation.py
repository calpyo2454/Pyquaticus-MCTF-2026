"""
hierarchical_framework/commander_observation.py

Purpose of this module:
- Build a compact commander-style observation from full environment state.

Why this file exists during scripted baseline:
- The scripted commander does not strictly need a learned observation vector,
  but having a consistent commander observation builder now helps with:
    1) debugging,
    2) future learned commander training,
    3) logging and evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


def build_commander_observation(
    state: dict,
    agents: List[str],
    blue_team: List[str],
    red_team: List[str],
    current_roles: Dict[str, int],
    previous_config_index: Optional[int],
    elapsed_steps: int,
    max_time: int,
) -> np.ndarray:
    """
    Build a compact numeric commander observation vector.

    Why this method exists:
    - A commander needs a tactical summary of the game, not the full
      low-level worker observation for each agent.
    - This same builder can later be reused when you switch from the
      scripted commander to a learned PPO commander.

    Contents included here:
    - blue and red positions
    - who has a flag
    - who is tagged
    - side and OOB state
    - team/global flags, grabs, captures, tags
    - current blue role assignments
    - previous commander config index
    - normalized time progress
    """
    obs = []

    def idx(agent_id: str) -> int:
        return agents.index(agent_id)

    # Blue-team tactical state.
    for aid in blue_team:
        i = idx(aid)
        pos = state["agent_position"][i]
        obs.extend([float(pos[0]), float(pos[1])])
        obs.append(float(state["agent_has_flag"][i]))
        obs.append(float(state["agent_is_tagged"][i]))
        obs.append(float(state["agent_on_sides"][i]))
        obs.append(float(state["agent_oob"][i]))

    # Red-team tactical state.
    for aid in red_team:
        i = idx(aid)
        pos = state["agent_position"][i]
        obs.extend([float(pos[0]), float(pos[1])])
        obs.append(float(state["agent_has_flag"][i]))
        obs.append(float(state["agent_is_tagged"][i]))
        obs.append(float(state["agent_on_sides"][i]))
        obs.append(float(state["agent_oob"][i]))

    # Team/global state.
    obs.extend([float(x) for x in state["team_has_flag"]])
    obs.extend([float(x) for x in state["flag_taken"]])
    obs.extend([float(x) for x in state["captures"]])
    obs.extend([float(x) for x in state["grabs"]])
    obs.extend([float(x) for x in state["tags"]])

    # Current blue-team role IDs.
    for aid in blue_team:
        obs.append(float(current_roles[aid]))

    # Previous commander config index.
    obs.append(float(-1 if previous_config_index is None else previous_config_index))

    # Normalized elapsed time.
    denom = max(float(max_time), 1.0)
    obs.append(float(elapsed_steps) / denom)

    return np.asarray(obs, dtype=np.float32)


def infer_commander_observation_length(
    blue_team_size: int = 3,
    red_team_size: int = 3,
) -> int:
    """
    Infer the commander observation vector length.

    Why this method exists:
    - The wrapper needs an observation space for commander-style logging
      and future learned commander support.
    - This helper keeps the length calculation in one place.
    """
    per_agent_features = 6  # x, y, has_flag, tagged, side, oob
    global_features = 2 + 2 + 2 + 2 + 2  # team_has_flag, flag_taken, captures, grabs, tags
    role_features = blue_team_size
    prev_action_features = 1
    time_features = 1

    return (
        blue_team_size * per_agent_features
        + red_team_size * per_agent_features
        + global_features
        + role_features
        + prev_action_features
        + time_features
    )