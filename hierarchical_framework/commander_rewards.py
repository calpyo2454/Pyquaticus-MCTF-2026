"""
hierarchical_framework/commander_rewards.py
"""

from __future__ import annotations

from typing import Dict

from hierarchical_framework.commander_action import ROLE_CONFIGS


def extract_macro_stats_from_state(state: dict) -> Dict[str, float]:
    return {
        "blue_captures": float(state["captures"][0]),
        "red_captures": float(state["captures"][1]),
        "blue_grabs": float(state["grabs"][0]),
        "red_grabs": float(state["grabs"][1]),
        "blue_tags": float(state["tags"][0]),
        "red_tags": float(state["tags"][1]),
    }


def compute_macro_reward(
    previous_stats: Dict[str, float],
    current_stats: Dict[str, float],
    role_assignment_changed: bool,
    switch_penalty: float = 0.02,
) -> float:
    r = 0.0
    r += 1.00 * (current_stats["blue_captures"] - previous_stats["blue_captures"])
    r -= 1.00 * (current_stats["red_captures"] - previous_stats["red_captures"])
    r += 0.30 * (current_stats["blue_grabs"] - previous_stats["blue_grabs"])
    r -= 0.30 * (current_stats["red_grabs"] - previous_stats["red_grabs"])
    r += 0.05 * (current_stats["blue_tags"] - previous_stats["blue_tags"])
    r -= 0.05 * (current_stats["red_tags"] - previous_stats["red_tags"])
    if role_assignment_changed:
        r -= switch_penalty
    return float(r)


def compute_commander_reward(prev_state: dict, state: dict, team_idx: int, chosen_config_idx: int) -> float:
    r = 0.0
    opp = 1 - team_idx
    config = ROLE_CONFIGS[int(chosen_config_idx)]

    own_attackers = sum(1 for x in config if x == 0)
    own_defenders = sum(1 for x in config if x == 1)
    own_interceptors = sum(1 for x in config if x == 2)

    if state["captures"][team_idx] > prev_state["captures"][team_idx]:
        r += 20.0
    if state["captures"][opp] > prev_state["captures"][opp]:
        r -= 20.0

    if state["grabs"][team_idx] > prev_state["grabs"][team_idx]:
        r += 6.0
    if state["grabs"][opp] > prev_state["grabs"][opp]:
        r -= 6.0

    enemy_pressure = 0
    own_pressure = 0
    half = len(state["agent_on_sides"]) // 2
    my_idxs = range(0, half) if team_idx == 0 else range(half, len(state["agent_on_sides"]))
    enemy_idxs = range(half, len(state["agent_on_sides"])) if team_idx == 0 else range(0, half)

    for idx in enemy_idxs:
        if int(state["agent_on_sides"][idx]) == team_idx:
            enemy_pressure += 1
    for idx in my_idxs:
        if int(state["agent_on_sides"][idx]) == opp:
            own_pressure += 1

    if enemy_pressure >= 2:
        if own_defenders + own_interceptors >= 2:
            r += 1.0
        else:
            r -= 1.0

    if own_pressure >= 2 and enemy_pressure == 0:
        if own_attackers >= 1:
            r += 0.6

    enemy_has_flag = any(state["agent_has_flag"][idx] for idx in enemy_idxs)
    if enemy_has_flag:
        if own_interceptors >= 1:
            r += 0.8
        else:
            r -= 0.8

    return float(r)
