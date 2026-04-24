"""
hierarchical_framework/commander_observation.py

Purpose of this module:
- Build a compact commander-style observation from full environment state.
"""

from __future__ import annotations

import numpy as np


def build_commander_observation(
    state: dict,
    agents: list[str],
    team_idx: int,
    max_time: float = 300.0,
    tag_cd_limit: float = 45.0,
) -> np.ndarray:
    """
    Build a compact commander observation.

    Per-agent features for all agents:
    - x, y
    - agent_on_sides value
    - has_flag
    - is_tagged
    - tagging cooldown fraction

    Global features:
    - own/enemy grabs
    - own/enemy captures
    - own/enemy flag_taken
    - enemy pressure on our home side
    - our pressure on enemy home side
    """
    feats: list[float] = []

    for idx, _aid in enumerate(agents):
        pos = state["agent_position"][idx]
        side_val = float(state["agent_on_sides"][idx])
        has_flag = float(state["agent_has_flag"][idx])
        is_tagged = float(state["agent_is_tagged"][idx])
        cd_frac = float(state["agent_tagging_cooldown"][idx]) / max(float(tag_cd_limit), 1.0)
        feats.extend([
            float(pos[0]), float(pos[1]),
            side_val,
            has_flag,
            is_tagged,
            cd_frac,
        ])

    feats.extend([
        float(state["grabs"][team_idx]),
        float(state["grabs"][1 - team_idx]),
        float(state["captures"][team_idx]),
        float(state["captures"][1 - team_idx]),
        float(state["flag_taken"][team_idx]),
        float(state["flag_taken"][1 - team_idx]),
    ])

    my_home_pressure = 0
    enemy_home_pressure = 0
    half = len(agents) // 2
    my_idxs = range(0, half) if team_idx == 0 else range(half, len(agents))
    enemy_idxs = range(half, len(agents)) if team_idx == 0 else range(0, half)

    for idx in enemy_idxs:
        if int(state["agent_on_sides"][idx]) == team_idx:
            my_home_pressure += 1
    for idx in my_idxs:
        if int(state["agent_on_sides"][idx]) == (1 - team_idx):
            enemy_home_pressure += 1

    feats.extend([float(my_home_pressure), float(enemy_home_pressure)])
    return np.asarray(feats, dtype=np.float32)


def infer_commander_observation_length(
    blue_team_size: int = 3,
    red_team_size: int = 3,
) -> int:
    """
    Match build_commander_observation exactly.
    """
    total_agents = blue_team_size + red_team_size
    per_agent_features = 6
    global_features = 6  # own/enemy grabs, captures, flag_taken
    pressure_features = 2
    return total_agents * per_agent_features + global_features + pressure_features
