"""
hierarchical_framework/worker_rewards.py

Purpose of this module:
- Provide role-specific worker reward shaping for the shared PPO worker policy.

This version merges in the teammate's flat PPO reward ideas:
- stronger sparse team reward from caps_and_grabs
- milestone-style attacker / support / defender shaping
- anti-collapse and OOB penalties
- milestone bonuses for flag progress and territory crossing

Role mapping into this hierarchy:
- ATTACK     <- teammate attacker_reward
- INTERCEPT  <- teammate support_reward + small enemy-carrier chase bonus
- DEFEND     <- teammate defender_reward

Why the merge is structured this way:
- It preserves the existing wrapper API and training flow.
- It lets you test the stronger reward ideas without refactoring your
  commander or environment code.
"""

from __future__ import annotations

from typing import List

import numpy as np

from hierarchical_framework.commander_action import ATTACK, DEFEND, INTERCEPT


def _team_index(team) -> int:
    """
    Convert team enum/int into a plain integer index.

    Why this method exists:
    - Different branches sometimes expose team values differently.
    - Reward code is simpler if everything uses 0 for blue and 1 for red.
    """
    try:
        return int(team.value)
    except Exception:
        return int(team)


def _agent_index(agents: List[str], agent_id: str) -> int:
    """
    Convert an agent ID into its index in the state arrays.

    Why this method exists:
    - State entries are usually indexed by position in the master agent list.
    """
    return agents.index(agent_id)


def _safe_get_array_value(state: dict, key: str, index: int, default=None):
    """
    Safely read a value from a state array-like entry.

    Why this method exists:
    - Reward shaping should fail soft instead of crashing training if a key
      is absent or shaped unexpectedly in a rollout.
    """
    if key not in state:
        return default
    try:
        return state[key][index]
    except Exception:
        return default


def _get_agent_position(state: dict, agent_index: int):
    """
    Return an agent's 2D position from state, or None if unavailable.

    Why this method exists:
    - Milestone and spatial shaping depend on position access.
    """
    if "agent_position" not in state:
        return None
    try:
        pos = state["agent_position"][agent_index]
        return np.asarray(pos, dtype=np.float32)
    except Exception:
        return None


def _get_team_flag_position(state: dict, team_idx: int):
    """
    Return a team's own flag position, or None if unavailable.

    Why this method exists:
    - ATTACK, INTERCEPT, and DEFEND all reference flag-relative geometry.
    """
    if "flag_position" not in state:
        return None
    try:
        pos = state["flag_position"][team_idx]
        return np.asarray(pos, dtype=np.float32)
    except Exception:
        return None


def _mid_x_from_flags(state: dict, team_idx: int):
    """
    Infer midfield x-coordinate from the two flag x-positions.

    Why this method exists:
    - The teammate reward file uses scrimmage/midfield milestones.
    - Our current reward API does not receive scrimmage_coords directly,
      so we infer the midpoint from flag positions.
    """
    own_flag = _get_team_flag_position(state, team_idx)
    enemy_flag = _get_team_flag_position(state, 1 - team_idx)
    if own_flag is None or enemy_flag is None:
        return None
    return float(0.5 * (own_flag[0] + enemy_flag[0]))


def _attacks_right(state: dict, team_idx: int) -> bool:
    """
    Determine whether this team attacks toward increasing x.

    Why this method exists:
    - The teammate milestone rewards assume blue attacks right and red attacks left.
    - This helper makes that directional logic robust to whichever side the
      flag positions imply.
    """
    own_flag = _get_team_flag_position(state, team_idx)
    enemy_flag = _get_team_flag_position(state, 1 - team_idx)
    if own_flag is None or enemy_flag is None:
        return team_idx == 0
    return float(enemy_flag[0]) > float(own_flag[0])


def _crossed_midfield(prev_x: float, x: float, state: dict, team_idx: int) -> bool:
    """
    Check whether the agent crossed the inferred midfield line this step.

    Why this method exists:
    - The teammate's attacker/support rewards use one-time midfield bonuses.
    """
    mid_x = _mid_x_from_flags(state, team_idx)
    if mid_x is None:
        return False

    if _attacks_right(state, team_idx):
        return prev_x <= mid_x and x > mid_x
    return prev_x >= mid_x and x < mid_x


def _on_enemy_side(x: float, state: dict, team_idx: int) -> bool:
    """
    Check whether the agent is on the enemy side of the field.

    Why this method exists:
    - Several milestone rewards depend on being in enemy territory.
    """
    mid_x = _mid_x_from_flags(state, team_idx)
    if mid_x is None:
        return False

    if _attacks_right(state, team_idx):
        return x > mid_x
    return x < mid_x


def _on_home_side(x: float, state: dict, team_idx: int) -> bool:
    """
    Check whether the agent is on its own side of the field.

    Why this method exists:
    - ATTACK return-home milestone depends on re-crossing midfield.
    """
    return not _on_enemy_side(x, state, team_idx)


def _team_agent_indices(team_idx: int, num_agents: int) -> List[int]:
    """
    Return the contiguous block of agent indices belonging to a team.

    Why this method exists:
    - In the current 3v3 setup, blue is the first half and red the second half.
    - This helper makes it easier to scan teammates or opponents.
    """
    half = num_agents // 2
    if team_idx == 0:
        return list(range(0, half))
    return list(range(half, num_agents))


def _find_team_flag_carrier_index(team_idx: int, agents: List[str], state: dict):
    """
    Find which agent index on a team is carrying the enemy flag.

    Why this method exists:
    - INTERCEPT should react to enemy carriers when they exist.
    """
    for idx in _team_agent_indices(team_idx, len(agents)):
        has_flag = _safe_get_array_value(state, "agent_has_flag", idx, False)
        if bool(has_flag):
            return idx
    return None


def _distance(pos_a, pos_b) -> float:
    """
    Compute Euclidean distance between two 2D points.

    Why this method exists:
    - Milestone shaping needs flag-zone and carrier-zone proximity checks.
    """
    if pos_a is None or pos_b is None:
        return 0.0
    return float(np.linalg.norm(pos_a - pos_b))


def _distance_progress(prev_pos, curr_pos, target_pos) -> float:
    """
    Return positive value when the agent moved closer to target.

    Why this method exists:
    - INTERCEPT keeps a small chase term even though the teammate file is
      mostly milestone-based.
    """
    if prev_pos is None or curr_pos is None or target_pos is None:
        return 0.0
    prev_d = _distance(prev_pos, target_pos)
    curr_d = _distance(curr_pos, target_pos)
    return float(prev_d - curr_d)


def team_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Stronger sparse team reward adapted from the teammate's caps_and_grabs.

    Why this method exists:
    - The teammate reward file made team-level grab/capture events much more
      important than in the earlier hierarchy version.
    """
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
    """
    Milestone-heavy ATTACK reward adapted from the teammate's attacker_reward.

    Ported milestones:
    - cross midfield
    - enter enemy territory
    - reach flag pickup zone
    - grab flag
    - return to home side with flag
    - capture
    - anti-collapse and OOB penalties
    """
    reward = 0.0
    i = _agent_index(agents, agent_id)
    t = _team_index(team)

    enemy_flag_pos = _get_team_flag_position(state, 1 - t)
    own_flag_pos = _get_team_flag_position(state, t)

    has_flag = bool(_safe_get_array_value(state, "agent_has_flag", i, False))
    prev_has_flag = bool(_safe_get_array_value(prev_state, "agent_has_flag", i, False))

    current_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)
    if current_pos is None or prev_pos is None:
        return reward

    x = float(current_pos[0])
    prev_x = float(prev_pos[0])

    # MILESTONE 1: Cross midfield.
    if _crossed_midfield(prev_x, x, state, t):
        reward += 2.0
    
    # Penalty for retreating back home without the flag after entering enemy side.
    if (not has_flag) and _on_home_side(x, state, t) and _on_enemy_side(prev_x, prev_state, t):
        reward -= 1.5

    # MILESTONE 2: Enter enemy territory.
    if _on_enemy_side(x, state, t) and not _on_enemy_side(prev_x, state, t):
        reward += 0.5

    # MILESTONE 3: Reach flag pickup zone.
    dist_to_enemy_flag = _distance(current_pos, enemy_flag_pos)
    prev_dist_to_enemy_flag = _distance(prev_pos, enemy_flag_pos)
    if dist_to_enemy_flag < 0.60 and prev_dist_to_enemy_flag >= 0.60:
        reward += 0.4
    if dist_to_enemy_flag < 0.40 and prev_dist_to_enemy_flag >= 0.40:
        reward += 0.8
    if dist_to_enemy_flag < 0.25 and prev_dist_to_enemy_flag >= 0.25:
        reward += 1.5
    if dist_to_enemy_flag < 0.2 and prev_dist_to_enemy_flag >= 0.2:
        reward += 2.0

    # MILESTONE 4: Grab flag.
    has_flag = bool(_safe_get_array_value(state, "agent_has_flag", i, False))
    prev_has_flag = bool(_safe_get_array_value(prev_state, "agent_has_flag", i, False))
    if has_flag and not prev_has_flag:
        reward += 8.0

    # MILESTONE 5: Return to home side while carrying.
    if has_flag and _on_home_side(x, state, t) and _on_enemy_side(prev_x, prev_state, t):
        reward += 5.0

    # MILESTONE 6: Capture.
    if "captures" in state and "captures" in prev_state:
        for idx in range(len(state["captures"])):
            if state["captures"][idx] > prev_state["captures"][idx]:
                reward += 15.0 if idx == t else -15.0

    # Anti-collapse: discourage spinning / not moving.
    if _distance(current_pos, prev_pos) < 0.015:
        reward -= 0.03

    # OOB penalty.
    if float(_safe_get_array_value(state, "agent_oob", i, 0.0)) > float(_safe_get_array_value(prev_state, "agent_oob", i, 0.0)):
        reward -= 2.0

    return float(reward)


def defender_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Milestone-heavy DEFEND reward adapted from the teammate's defender_reward.

    Ported milestones:
    - stay near own flag
    - tag enemy intruder
    - prevent enemy grabs
    - team capture contribution
    - penalty for wandering onto enemy side
    - anti-collapse and OOB penalties
    """
    reward = 0.0
    i = _agent_index(agents, agent_id)
    t = _team_index(team)

    own_flag_pos = _get_team_flag_position(state, t)
    current_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)
    if current_pos is None or prev_pos is None:
        return reward

    x = float(current_pos[0])

    # MILESTONE 1: Stay near own flag.
    dist_to_own_flag = _distance(current_pos, own_flag_pos)
    if dist_to_own_flag < 0.20:
        reward += 0.25
    elif dist_to_own_flag < 0.35:
        reward += 0.12
    elif dist_to_own_flag > 0.60:
        reward -= 0.15
    elif dist_to_own_flag > 0.80:
        reward -= 0.30

    # MILESTONE 2: Tag enemy intruder.
    made_tag = _safe_get_array_value(state, "agent_made_tag", i, None)
    if made_tag is not None:
        reward += 2.0
    elif "tags" in state and "tags" in prev_state:
        # Fallback if per-agent tag bookkeeping differs.
        if state["tags"][t] > prev_state["tags"][t]:
            reward += 2.0

    # MILESTONE 3: Prevent enemy grab.
    if "grabs" in state and "grabs" in prev_state:
        for idx in range(len(state["grabs"])):
            if state["grabs"][idx] > prev_state["grabs"][idx] and idx != t:
                reward -= 4.0

    # MILESTONE 5: Team capture contribution.
    if "captures" in state and "captures" in prev_state:
        for idx in range(len(state["captures"])):
            if state["captures"][idx] > prev_state["captures"][idx]:
                reward += 8.0 if idx == t else -8.0

    # Penalty for wandering too far into enemy territory.
    if _on_enemy_side(x, state, t):
        reward -= 0.3

    # Anti-collapse and OOB.
    if _distance(current_pos, prev_pos) < 0.015:
        reward -= 0.03

    if float(_safe_get_array_value(state, "agent_oob", i, 0.0)) > float(_safe_get_array_value(prev_state, "agent_oob", i, 0.0)):
        reward -= 2.0

    return float(reward)


def interceptor_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    INTERCEPT reward built from the teammate's support_reward, with a small
    carrier-chase bonus to preserve this hierarchy's intercept semantics.

    Ported support-style milestones:
    - cross midfield
    - remain in enemy territory
    - approach enemy flag
    - backup flag grab
    - tag bonus
    - capture bonus
    - anti-collapse and OOB penalties

    Added hierarchy-specific term:
    - if enemy carrier exists, reward closing on that carrier
    """
    reward = 0.0
    i = _agent_index(agents, agent_id)
    t = _team_index(team)
    opp = 1 - t

    enemy_flag_pos = _get_team_flag_position(state, opp)
    current_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)
    if current_pos is None or prev_pos is None:
        return reward

    x = float(current_pos[0])
    prev_x = float(prev_pos[0])

    # SUPPORT-LIKE MILESTONE 1: Cross midfield.
    if _crossed_midfield(prev_x, x, state, t):
        reward += 1.5

    # SUPPORT-LIKE MILESTONE 2: Stay in enemy territory.
    if _on_enemy_side(x, state, t):
        reward += 0.1

    # SUPPORT-LIKE MILESTONE 3: Approach enemy flag.
    dist_to_flag = _distance(current_pos, enemy_flag_pos)
    if dist_to_flag < 0.3:
        reward += 0.5

    # SUPPORT-LIKE MILESTONE 4: Backup flag grab.
    has_flag = bool(_safe_get_array_value(state, "agent_has_flag", i, False))
    prev_has_flag = bool(_safe_get_array_value(prev_state, "agent_has_flag", i, False))
    if has_flag and not prev_has_flag:
        reward += 4.0

    # SUPPORT-LIKE MILESTONE 5: Tag bonus.
    made_tag = _safe_get_array_value(state, "agent_made_tag", i, None)
    if made_tag is not None:
        reward += 8.0
    elif "tags" in state and "tags" in prev_state:
        if state["tags"][t] > prev_state["tags"][t]:
            reward += 1.5

    # SUPPORT-LIKE capture contribution.
    if "captures" in state and "captures" in prev_state:
        for idx in range(len(state["captures"])):
            if state["captures"][idx] > prev_state["captures"][idx]:
                reward += 10.0 if idx == t else -10.0

    # Added small intercept-specific carrier chase shaping.
    enemy_carrier_idx = _find_team_flag_carrier_index(opp, agents, state)
    if enemy_carrier_idx is not None:
        carrier_pos = _get_agent_position(state, enemy_carrier_idx)
        progress_to_carrier = _distance_progress(prev_pos, current_pos, carrier_pos)
        reward += 2.0 * progress_to_carrier

        if progress_to_carrier < 0:
            reward += 1.0 * progress_to_carrier   # extra penalty for moving away

    # Anti-collapse and OOB.
    if _distance(current_pos, prev_pos) < 0.015:
        reward -= 0.03

    if float(_safe_get_array_value(state, "agent_oob", i, 0.0)) > float(_safe_get_array_value(prev_state, "agent_oob", i, 0.0)):
        reward -= 2.0

    return float(reward)


def safety_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Small generic safety/stability penalty.

    Why this helper still exists:
    - The hierarchy wrapper expects a separate safety term in the blended reward.
    - We keep it modest to avoid double-counting OOB and anti-collapse too hard.
    """
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


def spacing_reward(agent_id, team, agents, state) -> float:
    i = _agent_index(agents, agent_id)
    t = _team_index(team)

    num_agents = len(agents)
    half = num_agents // 2
    if t == 0:
        teammate_indices = [j for j in range(0, half) if j != i]
    else:
        teammate_indices = [j for j in range(half, num_agents) if j != i]

    current_pos = _get_agent_position(state, i)
    if current_pos is None:
        return 0.0

    dists = []
    for j in teammate_indices:
        teammate_pos = _get_agent_position(state, j)
        if teammate_pos is not None:
            dists.append(_distance(current_pos, teammate_pos))

    if not dists:
        return 0.0

    nearest = min(dists)
    reward = 0.0

    if nearest < 0.12:
        reward -= 0.12
    elif 0.18 <= nearest <= 0.45:
        reward += 0.03
    elif nearest > 0.90:
        reward -= 0.04

    return reward

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
    """
    Blend base environment reward with the merged flat-PPO-inspired role rewards.

    Why these weights:
    - The teammate's rewards are stronger and more milestone-oriented.
    - We want them to matter during training, but we still preserve the base
      env reward and a team-level term.
    """

    delta = spacing_reward(agent_id, team, agents, state)

    if role_id == ATTACK:
        role_r = attacker_reward(agent_id, team, agents, state, prev_state)
    elif role_id == DEFEND:
        role_r = defender_reward(agent_id, team, agents, state, prev_state)
    else:
        role_r = interceptor_reward(agent_id, team, agents, state, prev_state)

    total = (
        float(base_reward)
        + alpha * role_r
        + beta * team_reward(agent_id, team, agents, state, prev_state)
        + gamma * safety_reward(agent_id, team, agents, state, prev_state)
        + 0.25 * delta
    )
    return float(total)

