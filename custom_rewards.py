import math
import numpy as np
from pyquaticus.structs import Team
from pyquaticus.utils.utils import *

### Enhanced Reward Functions for Better Training ###

def combined_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float,
):
    reward = 0.0
    agent_idx = agents.index(agent_id)

    # Base sparse reward
    reward += caps_and_grabs(
        agent_id,
        team,
        agents,
        agent_inds_of_team,
        state,
        prev_state,
        env_size,
        agent_radius,
        catch_radius,
        scrimmage_coords,
        max_speeds,
        tagging_cooldown,
    )

    # Flag positions
    if int(team) == 0:   # blue
        enemy_flag_pos = state["flag_position"][1]
        own_flag_pos   = state["flag_position"][0]
    else:                # red
        enemy_flag_pos = state["flag_position"][0]
        own_flag_pos   = state["flag_position"][1]

    current_pos = state["agent_position"][agent_idx]
    prev_pos    = prev_state["agent_position"][agent_idx]

    has_flag      = state["agent_has_flag"][agent_idx]
    prev_has_flag = prev_state["agent_has_flag"][agent_idx]

    # Small progress reward only. No free movement reward.
    target = own_flag_pos if has_flag else enemy_flag_pos
    current_dist = np.linalg.norm(current_pos - target)
    prev_dist    = np.linalg.norm(prev_pos - target)
    progress = prev_dist - current_dist

    # Reward real progress, lightly punish moving away.
    reward += 0.02 * progress

    # Tiny border penalty only.
    if state["agent_oob"][agent_idx]:
        reward -= 0.05

    # Tiny anti-stall penalty only if agent barely moved.
    step_move = np.linalg.norm(current_pos - prev_pos)
    if step_move < 0.03:
        reward -= 0.005

    # Bonus for actually taking the flag
    if (not prev_has_flag) and has_flag:
        reward += 3.0

    # Bonus for successfully returning toward home while carrying
    if has_flag and progress > 0:
        reward += 0.03 * progress

    # Reward crossing midfield / entering enemy territory
    x = current_pos[0]
    prev_x = prev_pos[0]
    
    mid_x = scrimmage_coords[0] if np.ndim(scrimmage_coords) > 0 else scrimmage_coords
    
    if int(team) == 0:  # blue attacks to the right
        if prev_x <= mid_x and x > mid_x:
            reward += 1.5   # one-time midfield crossing bonus
        if x > mid_x:
            reward += 0.01  # small per-step bonus for being in enemy territory
    else:  # red attacks to the left
        if prev_x >= mid_x and x < mid_x:
            reward += 1.5
        if x < mid_x:
            reward += 0.01

    # Bonus for getting close to enemy flag (encourages flag-seeking)
    dist_to_enemy_flag = np.linalg.norm(current_pos - enemy_flag_pos)
    if dist_to_enemy_flag < 0.3 and not has_flag:
        reward += 0.05

    # Small penalty for loitering near own flag when not carrying (encourages movement)
    dist_to_own_flag = np.linalg.norm(current_pos - own_flag_pos)
    if dist_to_own_flag < 0.2 and not has_flag:
        reward -= 0.02

    # Strong anti-spin penalty to prevent collapse
    if np.linalg.norm(current_pos - prev_pos) < 0.05:
        reward -= 0.1

    return reward

def tactical_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    """Tactical reward focusing on positioning and team coordination"""
    reward = 0.0
    
    agent_idx = agents.index(agent_id)
    current_pos = state['agent_position'][agent_idx]
    
    # Basic caps and grabs
    reward += caps_and_grabs(agent_id, team, agents, agent_inds_of_team, state, prev_state,
                           env_size, agent_radius, catch_radius, scrimmage_coords, 
                           max_speeds, tagging_cooldown)
    
    # Role-based rewards
    if int(team) == 0:  # Blue team
        enemy_flag_pos = state['flag_position'][1]  # Red flag (index 1)
        own_flag_pos = state['flag_position'][0]  # Blue flag (index 0)
    else:  # Red team
        enemy_flag_pos = state['flag_position'][0]  # Blue flag (index 0)
        own_flag_pos = state['flag_position'][1]  # Red flag (index 1)
    
    # Check if agent has flag (attacker role)
    if state['agent_has_flag'][agent_idx]:
        # Reward for moving toward home base when carrying flag
        dist_to_home = np.linalg.norm(current_pos - own_flag_pos)
        prev_dist_to_home = np.linalg.norm(prev_state['agent_position'][agent_idx] - own_flag_pos)
        
        if dist_to_home < prev_dist_to_home:
            reward += 0.05 * (prev_dist_to_home - dist_to_home)
        
        # Large reward for scoring (increased)
        if prev_state['agent_has_flag'][agent_idx] and not state['agent_has_flag'][agent_idx]:
            # Agent scored (lost flag near home base)
            if dist_to_home < 10:  # Near home base
                reward += 10.0
    else:
        # No flag - offensive positioning
        dist_to_enemy_flag = np.linalg.norm(current_pos - enemy_flag_pos)
        prev_dist_to_enemy_flag = np.linalg.norm(prev_state['agent_position'][agent_idx] - enemy_flag_pos)
        
        # Reward moving toward enemy flag
        if dist_to_enemy_flag < prev_dist_to_enemy_flag:
            reward += 0.02 * (prev_dist_to_enemy_flag - dist_to_enemy_flag)
        
        # Defensive positioning - reward being near own flag
        dist_to_own_flag = np.linalg.norm(current_pos - own_flag_pos)
        if dist_to_own_flag < 20:  # Defensive radius
            reward += 0.01
    
    # Team coordination rewards
    teammate_indices = [i for i in agent_inds_of_team[team] if i != agent_idx]
    for teammate_idx in teammate_indices:
        teammate_pos = state['agent_position'][teammate_idx]
        dist_to_teammate = np.linalg.norm(current_pos - teammate_pos)
        
        # Reward for maintaining reasonable team spacing
        if 10 < dist_to_teammate < 40:  # Good spacing
            reward += 0.005
        # Penalty for being too close or too far
        elif dist_to_teammate < 5:
            reward -= 0.01
    
    return reward

def caps_and_grabs(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float,
):
    reward = 0.0
    idx = agents.index(agent_id)

    # OOB should hurt, but not dominate everything
    if state["agent_oob"][idx] > prev_state["agent_oob"][idx]:
        reward -= 0.5

    prev_has_flag = prev_state["agent_has_flag"][idx]
    has_flag = state["agent_has_flag"][idx]

    # Dropping/losing the flag should matter
    if prev_has_flag and not has_flag:
        reward -= 1.0

    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            reward += 2.0 if t == int(team) else -2.0

        if state["captures"][t] > prev_state["captures"][t]:
            reward += 12.0 if t == int(team) else -12.0

    return reward


### Role-Based Milestone Rewards (No Dense Shaping) ###

def attacker_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float,
):
    """Milestone rewards for primary flag attacker - NO dense shaping"""
    reward = 0.0
    idx = agents.index(agent_id)
    
    # Flag positions
    if int(team) == 0:  # blue attacks right
        enemy_flag_pos = state["flag_position"][1]
        own_flag_pos = state["flag_position"][0]
    else:  # red attacks left
        enemy_flag_pos = state["flag_position"][0]
        own_flag_pos = state["flag_position"][1]
    
    current_pos = state["agent_position"][idx]
    prev_pos = prev_state["agent_position"][idx]
    x, prev_x = current_pos[0], prev_pos[0]
    mid_x = scrimmage_coords[0] if np.ndim(scrimmage_coords) > 0 else scrimmage_coords
    if isinstance(mid_x, np.ndarray):
        mid_x = mid_x[0]
    
    # MILESTONE 1: Cross midfield (+2.0) - one-time bonus
    if int(team) == 0:
        if prev_x <= mid_x and x > mid_x:
            reward += 2.0
    else:
        if prev_x >= mid_x and x < mid_x:
            reward += 2.0
    
    # MILESTONE 2: Enter enemy territory (+0.5)
    in_enemy_territory = (int(team) == 0 and x > mid_x) or (int(team) == 1 and x < mid_x)
    was_in_enemy = (int(team) == 0 and prev_x > mid_x) or (int(team) == 1 and prev_x < mid_x)
    if in_enemy_territory and not was_in_enemy:
        reward += 0.5
    
    # MILESTONE 3: Reach flag pickup zone (+1.0)
    dist_to_enemy_flag = np.linalg.norm(current_pos - enemy_flag_pos)
    prev_dist = np.linalg.norm(prev_pos - enemy_flag_pos)
    if dist_to_enemy_flag < 0.2 and prev_dist >= 0.2:
        reward += 1.0
    
    # MILESTONE 4: Grab flag (+5.0)
    has_flag = state["agent_has_flag"][idx]
    prev_has_flag = prev_state["agent_has_flag"][idx]
    if has_flag and not prev_has_flag:
        reward += 5.0
    
    # MILESTONE 5: Return to home side with flag (+3.0)
    if has_flag:
        on_home_side = (int(team) == 0 and x < mid_x) or (int(team) == 1 and x > mid_x)
        was_on_enemy = (int(team) == 0 and prev_x > mid_x) or (int(team) == 1 and prev_x < mid_x)
        if on_home_side and was_on_enemy:
            reward += 3.0
    
    # MILESTONE 6: Capture (+15.0)
    for t in range(len(state["captures"])):
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 15.0 if t == int(team) else -15.0
    
    # Anti-collapse: spinning penalty
    if np.linalg.norm(current_pos - prev_pos) < 0.03:
        reward -= 0.2
    
    # OOB penalty
    if state["agent_oob"][idx] > prev_state["agent_oob"][idx]:
        reward -= 1.0
    
    return reward


def support_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float,
):
    """Milestone rewards for support role - enters enemy territory, screens"""
    reward = 0.0
    idx = agents.index(agent_id)
    
    # Flag positions
    if int(team) == 0:
        enemy_flag_pos = state["flag_position"][1]
        own_flag_pos = state["flag_position"][0]
    else:
        enemy_flag_pos = state["flag_position"][0]
        own_flag_pos = state["flag_position"][1]
    
    current_pos = state["agent_position"][idx]
    prev_pos = prev_state["agent_position"][idx]
    x, prev_x = current_pos[0], prev_pos[0]
    mid_x = scrimmage_coords[0] if np.ndim(scrimmage_coords) > 0 else scrimmage_coords
    if isinstance(mid_x, np.ndarray):
        mid_x = mid_x[0]
    
    # MILESTONE 1: Cross midfield (+1.5)
    if int(team) == 0:
        if prev_x <= mid_x and x > mid_x:
            reward += 1.5
    else:
        if prev_x >= mid_x and x < mid_x:
            reward += 1.5
    
    # MILESTONE 2: Stay in enemy territory (+0.1 per step, capped)
    in_enemy = (int(team) == 0 and x > mid_x) or (int(team) == 1 and x < mid_x)
    if in_enemy:
        reward += 0.1
    
    # MILESTONE 3: Approach enemy flag (+0.5 for getting close)
    dist_to_flag = np.linalg.norm(current_pos - enemy_flag_pos)
    if dist_to_flag < 0.3:
        reward += 0.5
    
    # MILESTONE 4: Grab flag (backup attacker) (+4.0)
    has_flag = state["agent_has_flag"][idx]
    prev_has_flag = prev_state["agent_has_flag"][idx]
    if has_flag and not prev_has_flag:
        reward += 4.0
    
    # MILESTONE 5: Tag enemy (+2.0) - support defends attacker
    for t in range(len(state["tags"])):
        if state["tags"][t] > prev_state["tags"][t]:
            if t == int(team):
                reward += 2.0
    
    # Capture bonus
    for t in range(len(state["captures"])):
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 10.0 if t == int(team) else -10.0
    
    # Anti-collapse
    if np.linalg.norm(current_pos - prev_pos) < 0.03:
        reward -= 0.2
    
    if state["agent_oob"][idx] > prev_state["agent_oob"][idx]:
        reward -= 1.0
    
    return reward


def defender_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float,
):
    """Milestone rewards for defender - protects own flag"""
    reward = 0.0
    idx = agents.index(agent_id)
    
    # Flag positions
    if int(team) == 0:
        enemy_flag_pos = state["flag_position"][1]
        own_flag_pos = state["flag_position"][0]
    else:
        enemy_flag_pos = state["flag_position"][0]
        own_flag_pos = state["flag_position"][1]
    
    current_pos = state["agent_position"][idx]
    prev_pos = prev_state["agent_position"][idx]
    x = current_pos[0]
    mid_x = scrimmage_coords[0] if np.ndim(scrimmage_coords) > 0 else scrimmage_coords
    if isinstance(mid_x, np.ndarray):
        mid_x = mid_x[0]
    
    # MILESTONE 1: Stay near own flag (+0.2 per step when close)
    dist_to_own = np.linalg.norm(current_pos - own_flag_pos)
    if dist_to_own < 0.4:
        reward += 0.2
    
    # MILESTONE 2: Tag enemy intruder (+3.0)
    for t in range(len(state["tags"])):
        if state["tags"][t] > prev_state["tags"][t]:
            if t == int(team):
                reward += 3.0
    
    # MILESTONE 3: Prevent enemy grab (-2.0 if enemy grabs)
    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            if t != int(team):  # enemy grabbed
                reward -= 2.0
    
    # MILESTONE 4: Recover flag (+4.0) - enemy dropped it
    # This is implicit in the game mechanics
    
    # MILESTONE 5: Capture (defender contributes) (+8.0)
    for t in range(len(state["captures"])):
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 8.0 if t == int(team) else -8.0
    
    # Penalty for wandering too far from own flag
    on_enemy_side = (int(team) == 0 and x > mid_x) or (int(team) == 1 and x < mid_x)
    if on_enemy_side:
        reward -= 0.1
    
    # Anti-collapse
    if np.linalg.norm(current_pos - prev_pos) < 0.03:
        reward -= 0.2
    
    if state["agent_oob"][idx] > prev_state["agent_oob"][idx]:
        reward -= 1.0
    
    return reward
