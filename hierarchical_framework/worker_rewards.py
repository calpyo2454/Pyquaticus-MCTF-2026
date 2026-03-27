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

    Why this method exists:
    - The ATTACK role should learn to grab the enemy flag, keep possession,
      and avoid throwing away scoring opportunities.
    """
    i = _agent_index(agents, agent_id)
    r = 0.0

    if bool(state["agent_has_flag"][i]) and not bool(prev_state["agent_has_flag"][i]):
        r += 1.0

    if bool(prev_state["agent_has_flag"][i]) and not bool(state["agent_has_flag"][i]):
        r -= 0.5

    if float(state["agent_oob"][i]) > float(prev_state["agent_oob"][i]):
        r -= 0.5

    if bool(state["agent_is_tagged"][i]) and not bool(prev_state["agent_is_tagged"][i]):
        r -= 0.2

    return r


def defender_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Role-specific reward shaping for DEFEND.

    Why this method exists:
    - The DEFEND role should prioritize stopping enemy pressure and tagging
      intruders rather than wandering into offense all the time.
    """
    i = _agent_index(agents, agent_id)
    t = _team_index(team)
    opp = 1 - t
    r = 0.0

    tagged_idx = state["agent_made_tag"][i]
    if tagged_idx is not None:
        r += 0.4

    if state["grabs"][opp] > prev_state["grabs"][opp]:
        r -= 0.5

    if state["captures"][opp] > prev_state["captures"][opp]:
        r -= 1.0

    return r


def interceptor_reward(agent_id, team, agents, state, prev_state) -> float:
    """
    Role-specific reward shaping for INTERCEPT.

    Why this method exists:
    - The INTERCEPT role should prioritize chasing and disrupting enemy carriers
      or likely return paths.
    """
    i = _agent_index(agents, agent_id)
    r = 0.0

    tagged_idx = state["agent_made_tag"][i]
    if tagged_idx is not None:
        r += 0.5

    if float(state["agent_oob"][i]) > float(prev_state["agent_oob"][i]):
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
    alpha: float = 0.25,
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