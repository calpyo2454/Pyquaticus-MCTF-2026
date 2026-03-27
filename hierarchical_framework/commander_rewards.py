"""
hierarchical_framework/commander_rewards.py

Purpose of this module:
- Provide commander-oriented reward/statistic helpers.

Why this file still matters during scripted baseline:
- The scripted commander is not trainable yet, but you still want to:
    1) track commander-level team progress,
    2) log macro metrics,
    3) reuse the same reward/stat logic later for a learned commander.
"""

from __future__ import annotations

from typing import Dict


def extract_macro_stats_from_state(state: dict) -> Dict[str, float]:
    """
    Extract commander-level macro statistics from environment state.

    Why this method exists:
    - The commander cares about team-level outcomes like captures and grabs,
      not low-level primitive movement.
    - The wrapper can call this each step and compare deltas over time.
    """
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
    """
    Compute a commander-style macro reward from stat deltas.

    Why this method exists:
    - Even during scripted baseline, a macro reward is useful for logs and
      later learned-commander work.
    - The reward focuses on team outcomes:
        + captures and grabs for blue
        - captures and grabs for red
        + tags for blue
        - tags for red
        - mild penalty when role assignment changes too often
    """
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