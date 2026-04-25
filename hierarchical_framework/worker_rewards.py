"""
hierarchical_framework/worker_rewards.py
"""

from __future__ import annotations

from typing import List

import numpy as np

from hierarchical_framework.commander_action import ATTACK, DEFEND, INTERCEPT

TAG_COOLDOWN_LIMIT = 45.0


def _team_index(team) -> int:
    try:
        return int(team.value)
    except Exception:
        return int(team)


def _agent_index(agents: List[str], agent_id: str) -> int:
    return agents.index(agent_id)


def _safe_get_array_value(state: dict, key: str, index: int, default=None):
    if key not in state:
        return default
    try:
        return state[key][index]
    except Exception:
        return default


def _get_agent_position(state: dict, agent_index: int):
    if "agent_position" not in state:
        return None
    try:
        return np.asarray(state["agent_position"][agent_index], dtype=np.float32)
    except Exception:
        return None


def _get_team_flag_position(state: dict, team_idx: int):
    if "flag_position" not in state:
        return None
    try:
        return np.asarray(state["flag_position"][team_idx], dtype=np.float32)
    except Exception:
        return None


def _mid_x_from_flags(state: dict, team_idx: int):
    own_flag = _get_team_flag_position(state, team_idx)
    enemy_flag = _get_team_flag_position(state, 1 - team_idx)
    if own_flag is None or enemy_flag is None:
        return None
    return float(0.5 * (own_flag[0] + enemy_flag[0]))


def _attacks_right(state: dict, team_idx: int) -> bool:
    own_flag = _get_team_flag_position(state, team_idx)
    enemy_flag = _get_team_flag_position(state, 1 - team_idx)
    if own_flag is None or enemy_flag is None:
        return team_idx == 0
    return float(enemy_flag[0]) > float(own_flag[0])


def _crossed_midfield(prev_x: float, x: float, state: dict, team_idx: int) -> bool:
    mid_x = _mid_x_from_flags(state, team_idx)
    if mid_x is None:
        return False
    if _attacks_right(state, team_idx):
        return prev_x <= mid_x and x > mid_x
    return prev_x >= mid_x and x < mid_x


def _on_enemy_side(x: float, state: dict, team_idx: int) -> bool:
    mid_x = _mid_x_from_flags(state, team_idx)
    if mid_x is None:
        return False
    if _attacks_right(state, team_idx):
        return x > mid_x
    return x < mid_x


def _on_home_side(x: float, state: dict, team_idx: int) -> bool:
    return not _on_enemy_side(x, state, team_idx)


def _team_agent_indices(team_idx: int, num_agents: int) -> List[int]:
    half = num_agents // 2
    if team_idx == 0:
        return list(range(0, half))
    return list(range(half, num_agents))


def _find_team_flag_carrier_index(team_idx: int, agents: List[str], state: dict):
    for idx in _team_agent_indices(team_idx, len(agents)):
        has_flag = _safe_get_array_value(state, "agent_has_flag", idx, False)
        if bool(has_flag):
            return idx
    return None


def _distance(pos_a, pos_b) -> float:
    if pos_a is None or pos_b is None:
        return 0.0
    return float(np.linalg.norm(pos_a - pos_b))


def _distance_progress(prev_pos, curr_pos, target_pos) -> float:
    if prev_pos is None or curr_pos is None or target_pos is None:
        return 0.0
    return float(_distance(prev_pos, target_pos) - _distance(curr_pos, target_pos))


def _tag_cd(state: dict, agent_idx: int) -> float:
    return float(_safe_get_array_value(state, "agent_tagging_cooldown", agent_idx, 0.0) or 0.0)


def _tag_ready_frac(state: dict, agent_idx: int) -> float:
    return max(0.0, min(1.0, _tag_cd(state, agent_idx) / TAG_COOLDOWN_LIMIT))


def _nearest_enemy_idx(team_idx: int, agents: List[str], state: dict, from_pos) -> int | None:
    opp = 1 - team_idx
    best_idx, best_d = None, float("inf")
    for idx in _team_agent_indices(opp, len(agents)):
        pos = _get_agent_position(state, idx)
        d = _distance(from_pos, pos)
        if d < best_d:
            best_idx, best_d = idx, d
    return best_idx


def _find_nearest_opponent_to_position(team_idx: int, agents: List[str], state: dict, target_pos):
    return _nearest_enemy_idx(team_idx, agents, state, target_pos)


def spacing_reward(agent_id, team, agents, state) -> float:
    i = _agent_index(agents, agent_id)
    t = _team_index(team)
    pos = _get_agent_position(state, i)
    if pos is None:
        return 0.0

    teammate_idxs = [j for j in _team_agent_indices(t, len(agents)) if j != i]
    dists = []
    for j in teammate_idxs:
        p = _get_agent_position(state, j)
        if p is not None:
            dists.append(_distance(pos, p))

    if not dists:
        return 0.0

    nearest = min(dists)
    if nearest < 0.12:
        return -0.15
    if 0.20 <= nearest <= 0.45:
        return 0.04
    if nearest > 0.95:
        return -0.05
    return 0.0


def team_reward(agent_id, team, agents, state, prev_state) -> float:
    t = _team_index(team)
    r = 0.0
    if "grabs" in state and "grabs" in prev_state:
        for idx in range(len(state["grabs"])):
            if state["grabs"][idx] > prev_state["grabs"][idx]:
                r += 2.0 if idx == t else -2.0
    if "captures" in state and "captures" in prev_state:
        for idx in range(len(state["captures"])):
            if state["captures"][idx] > prev_state["captures"][idx]:
                r += 12.0 if idx == t else -12.0
    return float(r)


def attacker_reward(agent_id, team, agents, state, prev_state) -> float:
    reward = 0.0
    i = _agent_index(agents, agent_id)
    t = _team_index(team)

    enemy_flag_pos = _get_team_flag_position(state, 1 - t)
    own_flag_pos = _get_team_flag_position(state, t)
    current_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)
    if current_pos is None or prev_pos is None:
        return reward

    x = float(current_pos[0])
    prev_x = float(prev_pos[0])
    has_flag = bool(_safe_get_array_value(state, "agent_has_flag", i, False))
    prev_has_flag = bool(_safe_get_array_value(prev_state, "agent_has_flag", i, False))

    if _crossed_midfield(prev_x, x, state, t):
        reward += 0.75
    if _on_enemy_side(x, state, t) and not _on_enemy_side(prev_x, prev_state, t):
        reward += 0.40

    dist_to_enemy_flag = _distance(current_pos, enemy_flag_pos)
    prev_dist_to_enemy_flag = _distance(prev_pos, enemy_flag_pos)
    dist_to_own_flag = _distance(current_pos, own_flag_pos)
    prev_dist_to_own_flag = _distance(prev_pos, own_flag_pos)

    if dist_to_enemy_flag < 0.60 and prev_dist_to_enemy_flag >= 0.60:
        reward += 0.35
    if dist_to_enemy_flag < 0.40 and prev_dist_to_enemy_flag >= 0.40:
        reward += 0.75
    if dist_to_enemy_flag < 0.25 and prev_dist_to_enemy_flag >= 0.25:
        reward += 1.75

    if has_flag and not prev_has_flag:
        reward += 12.0

    if has_flag and _on_home_side(x, state, t) and _on_enemy_side(prev_x, prev_state, t):
        reward += 12.0

    if (not has_flag) and _on_home_side(x, state, t) and _on_enemy_side(prev_x, prev_state, t):
        reward -= 1.5

    # Very close pickup-zone commitment.
    if dist_to_enemy_flag < 0.12 and prev_dist_to_enemy_flag >= 0.12:
        reward += 2.5

    # Punish backing out of the pickup zone without grabbing.
    if (not has_flag) and prev_dist_to_enemy_flag < 0.12 and dist_to_enemy_flag >= 0.12:
        reward -= 2.0

    # Mild bonus to keep pressing if already very near the flag.
    if (not has_flag) and dist_to_enemy_flag < 0.12:
        reward += 0.15

    nearest_enemy_idx = _nearest_enemy_idx(t, agents, state, current_pos)
    if nearest_enemy_idx is not None:
        enemy_pos = _get_agent_position(state, nearest_enemy_idx)
        enemy_dist = _distance(current_pos, enemy_pos)
        enemy_ready = _tag_ready_frac(state, nearest_enemy_idx)

        if enemy_dist < 0.35:
            if has_flag:
                if enemy_ready < 0.35:
                    reward += 0.60
                elif enemy_ready > 0.80:
                    reward -= 0.80
            else:
                if enemy_ready < 0.35:
                    reward += 0.75
                elif enemy_ready > 0.80:
                    reward -= 0.60

        # If the flag is effectively unprotected, strongly encourage the grab.
        if (not has_flag) and dist_to_enemy_flag < 0.15 and enemy_dist > 0.30:
            reward += 1.5

    if has_flag:
        home_progress = _distance_progress(prev_pos, current_pos, own_flag_pos)

        # Make return-home progress dominant.
        reward += 1.60 * home_progress

        # Strongly punish moving away from home while carrying.
        if home_progress < 0:
            reward += 2.50 * home_progress

        # Penalize making no meaningful homeward progress, even if moving in circles.
        if abs(home_progress) < 0.01:
            reward -= 0.20

        # Strong anti-spin / anti-hover penalty while carrying.
        if _distance(current_pos, prev_pos) < 0.025:
            reward -= 0.45

        # Carrying milestones.
        if dist_to_own_flag < 0.60 and prev_dist_to_own_flag >= 0.60:
            reward += 2.0
        if dist_to_own_flag < 0.40 and prev_dist_to_own_flag >= 0.40:
            reward += 3.0
        if dist_to_own_flag < 0.25 and prev_dist_to_own_flag >= 0.25:
            reward += 4.0

    if _distance(current_pos, prev_pos) < 0.015:
        reward -= 0.03
    if bool(_safe_get_array_value(state, "agent_is_tagged", i, False)) and not bool(_safe_get_array_value(prev_state, "agent_is_tagged", i, False)):
        reward -= 0.40
    if bool(_safe_get_array_value(state, "agent_oob", i, False)):
        reward -= 0.15
    if bool(_safe_get_array_value(state, "agent_oob", i, False)) and not bool(_safe_get_array_value(prev_state, "agent_oob", i, False)):
        reward -= 1.25

    # Local capture reward for the carrier role.
    if "captures" in state and "captures" in prev_state:
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 15.0

    return float(reward)


def defender_reward(agent_id, team, agents, state, prev_state) -> float:
    reward = 0.0
    i = _agent_index(agents, agent_id)
    t = _team_index(team)
    opp = 1 - t

    own_flag_pos = _get_team_flag_position(state, t)
    current_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)
    if current_pos is None or prev_pos is None:
        return reward

    x = float(current_pos[0])
    dist_to_own_flag = _distance(current_pos, own_flag_pos)

    if dist_to_own_flag < 0.20:
        reward += 0.25
    elif dist_to_own_flag < 0.35:
        reward += 0.12
    elif dist_to_own_flag > 0.60:
        reward -= 0.15
    elif dist_to_own_flag > 0.80:
        reward -= 0.30

    if _on_enemy_side(x, state, t):
        reward -= 0.30

    threat_idx = _find_nearest_opponent_to_position(t, agents, state, own_flag_pos)
    if threat_idx is not None:
        threat_pos = _get_agent_position(state, threat_idx)
        threat_dist = _distance(current_pos, threat_pos)
        ready_frac = _tag_ready_frac(state, i)
        if threat_dist < 0.50:
            reward += 0.15 + 0.20 * ready_frac
        reward += 0.12 * _distance_progress(prev_pos, current_pos, threat_pos)

    made_tag = _safe_get_array_value(state, "agent_made_tag", i, None)
    if made_tag is not None:
        reward += 4.0
        if _on_home_side(x, state, t):
            reward += 2.0

    if "grabs" in state and "grabs" in prev_state and state["grabs"][opp] > prev_state["grabs"][opp]:
        reward -= 4.0
    if "captures" in state and "captures" in prev_state and state["captures"][opp] > prev_state["captures"][opp]:
        reward -= 8.0
    if _distance(current_pos, prev_pos) < 0.015:
        reward -= 0.03
    if bool(_safe_get_array_value(state, "agent_oob", i, False)):
        reward -= 0.15
    if bool(_safe_get_array_value(state, "agent_oob", i, False)) and not bool(_safe_get_array_value(prev_state, "agent_oob", i, False)):
        reward -= 1.25
    return float(reward)


def interceptor_reward(agent_id, team, agents, state, prev_state) -> float:
    reward = 0.0
    i = _agent_index(agents, agent_id)
    t = _team_index(team)
    opp = 1 - t

    current_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)
    own_flag_pos = _get_team_flag_position(state, t)
    if current_pos is None or prev_pos is None:
        return reward

    enemy_carrier_idx = _find_team_flag_carrier_index(opp, agents, state)
    if enemy_carrier_idx is not None:
        carrier_pos = _get_agent_position(state, enemy_carrier_idx)
        progress = _distance_progress(prev_pos, current_pos, carrier_pos)
        reward += 2.0 * progress
        if progress < 0:
            reward += 1.0 * progress
        made_tag = _safe_get_array_value(state, "agent_made_tag", i, None)
        if made_tag is not None and int(made_tag) == int(enemy_carrier_idx):
            reward += 8.0
    else:
        threat_idx = _find_nearest_opponent_to_position(t, agents, state, own_flag_pos)
        if threat_idx is not None:
            threat_pos = _get_agent_position(state, threat_idx)
            reward += 0.12 * _distance_progress(prev_pos, current_pos, threat_pos)

    if _distance(current_pos, prev_pos) < 0.015:
        reward -= 0.03
    if bool(_safe_get_array_value(state, "agent_oob", i, False)):
        reward -= 0.15
    if bool(_safe_get_array_value(state, "agent_oob", i, False)) and not bool(_safe_get_array_value(prev_state, "agent_oob", i, False)):
        reward -= 1.25
    return float(reward)


def safety_reward(agent_id, team, agents, state, prev_state) -> float:
    i = _agent_index(agents, agent_id)
    reward = 0.0
    current_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)
    if current_pos is not None and prev_pos is not None and _distance(current_pos, prev_pos) < 0.01:
        reward -= 0.01
    if float(_safe_get_array_value(state, "agent_oob", i, 0.0)) > float(_safe_get_array_value(prev_state, "agent_oob", i, 0.0)):
        reward -= 0.5
    if bool(_safe_get_array_value(state, "agent_oob", i, False)):
        reward -= 0.1
    return float(reward)


def shaped_worker_reward(
    role_id: int,
    base_reward: float,
    agent_id,
    team,
    agents,
    state,
    prev_state,
    alpha: float = 1.00,
    beta: float = 0.20,
    gamma: float = 0.08,
) -> float:
    role_r = attacker_reward(agent_id, team, agents, state, prev_state) if role_id == ATTACK else defender_reward(agent_id, team, agents, state, prev_state) if role_id == DEFEND else interceptor_reward(agent_id, team, agents, state, prev_state)
    total = (
        float(base_reward)
        + alpha * role_r
        + beta * team_reward(agent_id, team, agents, state, prev_state)
        + gamma * safety_reward(agent_id, team, agents, state, prev_state)
        + 0.25 * spacing_reward(agent_id, team, agents, state)
    )
    return float(total)
