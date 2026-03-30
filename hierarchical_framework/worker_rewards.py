"""
hierarchical_framework/worker_rewards.py

Purpose of this module:
- Provide role-specific worker reward shaping for the shared PPO worker policy.

Why this file exists:
- All blue workers share one PPO policy, so the role one-hot appended to
  observation must be meaningful.
- Reward shaping helps the shared worker learn distinct ATTACK, DEFEND,
  and INTERCEPT behavior while still caring about team success.
"""

from __future__ import annotations

from typing import List

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

def _team_agent_indices(team_idx: int, num_agents: int) -> List[int]:
    """
    Return the contiguous block of agent indices belonging to a team.

    Why this method exists:
    - The current baseline assumes the 3v3 ordering:
        blue = first half of agents
        red  = second half of agents
    - This helper makes it easier to scan teammates or opponents.
    """
    half = num_agents // 2
    if team_idx == 0:
        return list(range(0, half))
    return list(range(half, num_agents))


def _safe_get_array_value(state: dict, key: str, index: int, default=None):
    """
    Safely read a value from a state array-like entry.

    Why this method exists:
    - Different branches or moments in rollout may omit a key or expose it
      in a shape we do not expect.
    - Reward shaping should fail soft instead of crashing training.
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
    - Spatial shaping depends on distance to flags, carriers, and threats.
    """
    if "agent_position" not in state:
        return None
    try:
        pos = state["agent_position"][agent_index]
        return (float(pos[0]), float(pos[1]))
    except Exception:
        return None


def _get_team_flag_position(state: dict, team_idx: int):
    """
    Return the position of a team's own flag, or None if unavailable.

    Why this method exists:
    - ATTACK needs enemy flag position.
    - DEFEND needs own flag position.
    - INTERCEPT may need own-flag-relative threat geometry.
    """
    if "flag_position" not in state:
        return None
    try:
        pos = state["flag_position"][team_idx]
        return (float(pos[0]), float(pos[1]))
    except Exception:
        return None


def _distance(pos_a, pos_b) -> float:
    """
    Compute Euclidean distance between two 2D points.

    Why this method exists:
    - Spatial shaping uses distance changes as dense reward.
    """
    if pos_a is None or pos_b is None:
        return 0.0
    dx = float(pos_a[0]) - float(pos_b[0])
    dy = float(pos_a[1]) - float(pos_b[1])
    return (dx * dx + dy * dy) ** 0.5


def _distance_progress(prev_pos, curr_pos, target_pos) -> float:
    """
    Return positive value when the agent moved closer to target.

    Why this method exists:
    - Reward shaping should be dense:
      * positive if an ATTACK agent moves toward enemy flag
      * positive if a DEFEND agent returns toward own flag
      * positive if an INTERCEPT agent closes on a threat
    """
    if prev_pos is None or curr_pos is None or target_pos is None:
        return 0.0
    prev_d = _distance(prev_pos, target_pos)
    curr_d = _distance(curr_pos, target_pos)
    return float(prev_d - curr_d)


def _find_team_flag_carrier_index(team_idx: int, agents: List[str], state: dict):
    """
    Find which agent index on a team is carrying the enemy flag.

    Why this method exists:
    - INTERCEPT should close on the opposing carrier.
    - ATTACK may be rewarded differently if it is the carrier.
    """
    for idx in _team_agent_indices(team_idx, len(agents)):
        has_flag = _safe_get_array_value(state, "agent_has_flag", idx, False)
        if bool(has_flag):
            return idx
    return None


def _find_nearest_opponent_to_position(team_idx: int, agents: List[str], state: dict, target_pos):
    """
    Find the nearest opposing agent to a given position.

    Why this method exists:
    - When no enemy carrier exists, INTERCEPT can still react to the most
      threatening nearby opponent relative to own flag.
    """
    opp = 1 - team_idx
    best_idx = None
    best_d = float("inf")

    for idx in _team_agent_indices(opp, len(agents)):
        pos = _get_agent_position(state, idx)
        d = _distance(pos, target_pos)
        if d < best_d:
            best_d = d
            best_idx = idx

    return best_idx


def _agent_on_team_side(team_idx: int, agent_index: int, state: dict):
    """
    Return whether the agent is on its own side, if that state key exists.

    Why this method exists:
    - DEFEND should prefer being home-side.
    - ATTACK can receive a small penalty for idling on home-side too long.

    Important note:
    - This assumes agent_on_sides uses team-index semantics where:
        0 = blue side
        1 = red side
    - If your branch differs, this reward term can be disabled later.
    """
    if "agent_on_sides" not in state:
        return None
    side = _safe_get_array_value(state, "agent_on_sides", agent_index, None)
    if side is None:
        return None
    return int(side) == int(team_idx)


def team_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Shared team reward component for a worker.

    Why this method exists:
    - Workers should not be trained in isolation.
    - This term ties each blue worker partly to team success.
    """
    t = _team_index(team)
    opp = 1 - t

    r = 0.0
    if state["grabs"][t] > prev_state["grabs"][t]:
        r += 0.25
    if state["captures"][t] > prev_state["captures"][t]:
        r += 1.00
    if state["grabs"][opp] > prev_state["grabs"][opp]:
        r -= 0.25
    if state["captures"][opp] > prev_state["captures"][opp]:
        r -= 1.00
    return r


def attacker_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Role-specific reward shaping for ATTACK.

    What this version adds:
    - Dense reward for moving toward the enemy flag when not carrying.
    - Dense reward for moving toward home flag while carrying.
    - Small penalty for idling on home side when not carrying.
    - Existing event shaping for flag gain/loss, OOB, and tagging.
    """
    i = _agent_index(agents, agent_id)
    t = _team_index(team)
    opp = 1 - t
    r = 0.0

    curr_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)

    own_flag_pos = _get_team_flag_position(state, t)
    enemy_flag_pos = _get_team_flag_position(state, opp)

    has_flag_now = bool(_safe_get_array_value(state, "agent_has_flag", i, False))
    had_flag_prev = bool(_safe_get_array_value(prev_state, "agent_has_flag", i, False))

    # Event shaping: gaining or losing flag possession matters a lot.
    if has_flag_now and not had_flag_prev:
        r += 1.0

    if had_flag_prev and not has_flag_now:
        r -= 0.5

    # Dense spatial shaping:
    # - if not carrying, reward progress toward enemy flag
    # - if carrying, reward progress toward home flag
    if has_flag_now:
        r += 0.08 * _distance_progress(prev_pos, curr_pos, own_flag_pos)
    else:
        r += 0.05 * _distance_progress(prev_pos, curr_pos, enemy_flag_pos)

    # Mild penalty for sitting on own side while not carrying.
    on_team_side = _agent_on_team_side(t, i, state)
    if on_team_side is True and not has_flag_now:
        r -= 0.01

    # Safety/event penalties.
    if float(_safe_get_array_value(state, "agent_oob", i, 0.0)) > float(_safe_get_array_value(prev_state, "agent_oob", i, 0.0)):
        r -= 0.5

    if bool(_safe_get_array_value(state, "agent_is_tagged", i, False)) and not bool(_safe_get_array_value(prev_state, "agent_is_tagged", i, False)):
        r -= 0.2

    return r


def defender_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Role-specific reward shaping for DEFEND.

    What this version adds:
    - Dense reward for staying near own flag / returning toward it.
    - Bonus for remaining on home side when possible.
    - Extra incentive to close on enemy carrier if one exists.
    - Existing tag bonus and penalties when opponent grabs/captures.
    """
    i = _agent_index(agents, agent_id)
    t = _team_index(team)
    opp = 1 - t
    r = 0.0

    curr_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)
    own_flag_pos = _get_team_flag_position(state, t)

    # Anchor defenders near home flag.
    if own_flag_pos is not None:
        curr_d = _distance(curr_pos, own_flag_pos)
        prev_d = _distance(prev_pos, own_flag_pos)

        # Reward moving back toward own flag if drifting.
        r += 0.04 * (prev_d - curr_d)

        # Small station-keeping bonus for being reasonably close.
        if curr_d < 0.35:
            r += 0.02

    # Bonus for staying on own side if available.
    on_team_side = _agent_on_team_side(t, i, state)
    if on_team_side is True:
        r += 0.01
    elif on_team_side is False:
        r -= 0.01

    # If enemy has the flag, defender should also collapse toward carrier.
    enemy_carrier_idx = _find_team_flag_carrier_index(opp, agents, state)
    if enemy_carrier_idx is not None:
        enemy_carrier_pos = _get_agent_position(state, enemy_carrier_idx)
        prev_enemy_carrier_pos = _get_agent_position(prev_state, enemy_carrier_idx)
        # Use current carrier position for a simple close-in shaping signal.
        r += 0.05 * _distance_progress(prev_pos, curr_pos, enemy_carrier_pos)

    # Tagging nearby threats is good defender behavior.
    tagged_idx = _safe_get_array_value(state, "agent_made_tag", i, None)
    if tagged_idx is not None:
        r += 0.4

    # Opponent success hurts defenders.
    if "grabs" in state and "grabs" in prev_state and state["grabs"][opp] > prev_state["grabs"][opp]:
        r -= 0.5

    if "captures" in state and "captures" in prev_state and state["captures"][opp] > prev_state["captures"][opp]:
        r -= 1.0

    return r


def interceptor_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Role-specific reward shaping for INTERCEPT.

    What this version adds:
    - Dense reward for closing on the enemy carrier if one exists.
    - If no carrier exists, close on the nearest enemy threat to own flag.
    - Existing tag bonus and OOB penalty.
    """
    i = _agent_index(agents, agent_id)
    t = _team_index(team)
    opp = 1 - t
    r = 0.0

    curr_pos = _get_agent_position(state, i)
    prev_pos = _get_agent_position(prev_state, i)
    own_flag_pos = _get_team_flag_position(state, t)

    enemy_carrier_idx = _find_team_flag_carrier_index(opp, agents, state)

    if enemy_carrier_idx is not None:
        carrier_pos = _get_agent_position(state, enemy_carrier_idx)
        r += 0.08 * _distance_progress(prev_pos, curr_pos, carrier_pos)
    else:
        # No enemy carrier yet: shadow the nearest likely threat to own flag.
        if own_flag_pos is not None:
            threat_idx = _find_nearest_opponent_to_position(t, agents, state, own_flag_pos)
            if threat_idx is not None:
                threat_pos = _get_agent_position(state, threat_idx)
                r += 0.04 * _distance_progress(prev_pos, curr_pos, threat_pos)

    tagged_idx = _safe_get_array_value(state, "agent_made_tag", i, None)
    if tagged_idx is not None:
        r += 0.5

    if float(_safe_get_array_value(state, "agent_oob", i, 0.0)) > float(_safe_get_array_value(prev_state, "agent_oob", i, 0.0)):
        r -= 0.5

    return r


def safety_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Small shared penalty term for unsafe behavior.

    Why this method exists:
    - Regardless of role, workers should avoid obvious mistakes like
      going OOB or getting tagged carelessly.
    """
    i = _agent_index(agents, agent_id)
    r = 0.0

    if float(state["agent_oob"][i]) > float(prev_state["agent_oob"][i]):
        r -= 0.5

    if bool(state["agent_is_tagged"][i]) and not bool(prev_state["agent_is_tagged"][i]):
        r -= 0.2

    return r


def shaped_worker_reward(
    role_id: int,
    base_reward: float,
    agent_id,
    team,
    agents,
    state,
    prev_state,
    alpha: float = 0.35,
    beta: float = 0.20,
    gamma: float = 0.10,
) -> float:
    """
    Blend base environment reward with role-specific shaping and team reward.

    Why this method exists:
    - For a first scripted-commander baseline, replacing the entire env reward
      is usually too aggressive.
    - This function keeps base env reward intact, then adds modest shaping so
      the shared worker learns role-conditioned behavior.

    Formula used:
    total = base_reward + alpha * role_reward + beta * team_reward + gamma * safety_reward
    """
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
    )
    return float(total)