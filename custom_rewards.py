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

    # Base sparse reward (captures/grabs)
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

    # Progress toward objective (flag or home)
    target = own_flag_pos if has_flag else enemy_flag_pos
    current_dist = np.linalg.norm(current_pos - target)
    prev_dist    = np.linalg.norm(prev_pos - target)
    progress = prev_dist - current_dist

    # Strong progress reward to guide learning
    reward += 0.5 * progress

    # OOB penalty
    if state["agent_oob"][agent_idx]:
        reward -= 0.3

    # Anti-stall penalty
    step_move = np.linalg.norm(current_pos - prev_pos)
    if step_move < 0.01:
        reward -= 0.05

    # Bonus for taking flag
    if (not prev_has_flag) and has_flag:
        reward += 5.0

    # Bonus for returning home with flag
    if has_flag and progress > 0:
        reward += 0.3 * progress

    # Midfield crossing milestone
    x = current_pos[0]
    prev_x = prev_pos[0]
    
    mid_x = scrimmage_coords[0] if np.ndim(scrimmage_coords) > 0 else scrimmage_coords
    if isinstance(mid_x, np.ndarray):
        mid_x = mid_x.item() if mid_x.ndim == 0 else mid_x[0]
    
    if int(team) == 0:
        if prev_x <= mid_x and x > mid_x:
            reward += 3.0
    else:
        if prev_x >= mid_x and x < mid_x:
            reward += 3.0

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


### Dense Shaping Rewards with Curriculum Learning ###

def dense_flag_reward(
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
    """Dense shaping reward - strong continuous feedback for moving toward objectives"""
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
    
    has_flag = state["agent_has_flag"][idx]
    prev_has_flag = prev_state["agent_has_flag"][idx]
    
    # DENSE REWARD 1: Move toward target (enemy flag or home)
    if has_flag:
        target = own_flag_pos
        progress_weight = 2.0  # Stronger incentive when carrying flag
    else:
        target = enemy_flag_pos
        progress_weight = 1.5  # Strong incentive to approach flag
    
    current_dist = np.linalg.norm(current_pos - target)
    prev_dist = np.linalg.norm(prev_pos - target)
    progress = prev_dist - current_dist
    reward += progress_weight * progress
    
    # DENSE REWARD 2: Bonus for being close to target
    if not has_flag:
        dist_to_flag = np.linalg.norm(current_pos - enemy_flag_pos)
        if dist_to_flag < 0.3:
            reward += 0.5
        elif dist_to_flag < 0.5:
            reward += 0.2
    else:
        dist_to_home = np.linalg.norm(current_pos - own_flag_pos)
        if dist_to_home < 0.3:
            reward += 1.0
        elif dist_to_home < 0.5:
            reward += 0.5
    
    # SPARSE REWARDS: Key events
    # Grab flag
    if has_flag and not prev_has_flag:
        reward += 10.0
    
    # Capture
    for t in range(len(state["captures"])):
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 20.0 if t == int(team) else -20.0
    
    # Grab (team grabbed enemy flag)
    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            reward += 5.0 if t == int(team) else -5.0
    
    # PENALTIES
    # OOB
    if state["agent_oob"][idx] > prev_state["agent_oob"][idx]:
        reward -= 2.0
    
    # Anti-stall (prevent spinning in place)
    movement = np.linalg.norm(current_pos - prev_pos)
    if movement < 0.02:
        reward -= 0.3
    
    return reward


def curriculum_reward(
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
    curriculum_stage: int = 3,
):
    """
    Curriculum learning reward with progressive difficulty:
    Stage 1: Just move forward (simple movement)
    Stage 2: Move toward enemy flag
    Stage 3: Full flag capture task (default)
    """
    reward = 0.0
    idx = agents.index(agent_id)
    
    # Flag positions
    if int(team) == 0:
        enemy_flag_pos = state["flag_position"][1]
        own_flag_pos = state["flag_position"][0]
        attack_direction = 1  # blue attacks right
    else:
        enemy_flag_pos = state["flag_position"][0]
        own_flag_pos = state["flag_position"][1]
        attack_direction = -1  # red attacks left
    
    current_pos = state["agent_position"][idx]
    prev_pos = prev_state["agent_position"][idx]
    x, prev_x = current_pos[0], prev_pos[0]
    
    has_flag = state["agent_has_flag"][idx]
    prev_has_flag = prev_state["agent_has_flag"][idx]
    
    # STAGE 1: Basic forward movement
    if curriculum_stage >= 1:
        forward_progress = (x - prev_x) * attack_direction
        reward += 2.0 * forward_progress
    
    # STAGE 2: Move toward enemy flag
    if curriculum_stage >= 2:
        if not has_flag:
            dist_to_flag = np.linalg.norm(current_pos - enemy_flag_pos)
            prev_dist = np.linalg.norm(prev_pos - enemy_flag_pos)
            progress = prev_dist - dist_to_flag
            reward += 1.5 * progress
        else:
            dist_to_home = np.linalg.norm(current_pos - own_flag_pos)
            prev_dist = np.linalg.norm(prev_pos - own_flag_pos)
            progress = prev_dist - dist_to_home
            reward += 2.0 * progress
    
    # STAGE 3: Full task with bonuses
    if curriculum_stage >= 3:
        # Proximity bonuses
        if not has_flag:
            dist_to_flag = np.linalg.norm(current_pos - enemy_flag_pos)
            if dist_to_flag < 0.3:
                reward += 0.5
        else:
            dist_to_home = np.linalg.norm(current_pos - own_flag_pos)
            if dist_to_home < 0.3:
                reward += 1.0
    
    # Sparse rewards (all stages)
    if has_flag and not prev_has_flag:
        reward += 10.0
    
    for t in range(len(state["captures"])):
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 20.0 if t == int(team) else -20.0
    
    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            reward += 5.0 if t == int(team) else -5.0
    
    # Penalties
    if state["agent_oob"][idx] > prev_state["agent_oob"][idx]:
        reward -= 2.0
    
    movement = np.linalg.norm(current_pos - prev_pos)
    if movement < 0.02:
        reward -= 0.3
    
    return reward


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
    """Attacker uses dense shaping rewards for aggressive flag chasing"""
    return dense_flag_reward(
        agent_id, team, agents, agent_inds_of_team, state, prev_state,
        env_size, agent_radius, catch_radius, scrimmage_coords, max_speeds, tagging_cooldown
    )


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
    """Support uses dense shaping - secondary flag chaser with team coordination"""
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
    
    has_flag = state["agent_has_flag"][idx]
    prev_has_flag = prev_state["agent_has_flag"][idx]
    
    # Move toward target (slightly weaker than attacker)
    if has_flag:
        target = own_flag_pos
        progress_weight = 1.5
    else:
        target = enemy_flag_pos
        progress_weight = 1.2
    
    current_dist = np.linalg.norm(current_pos - target)
    prev_dist = np.linalg.norm(prev_pos - target)
    progress = prev_dist - current_dist
    reward += progress_weight * progress
    
    # Proximity bonus
    if not has_flag and current_dist < 0.4:
        reward += 0.3
    
    # Sparse rewards
    if has_flag and not prev_has_flag:
        reward += 8.0
    
    for t in range(len(state["captures"])):
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 15.0 if t == int(team) else -15.0
    
    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            reward += 4.0 if t == int(team) else -4.0
    
    # Penalties
    if state["agent_oob"][idx] > prev_state["agent_oob"][idx]:
        reward -= 2.0
    
    movement = np.linalg.norm(current_pos - prev_pos)
    if movement < 0.02:
        reward -= 0.3
    
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
    """Defender uses dense shaping - stay near own flag, intercept enemies"""
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
    
    # Find nearest enemy
    enemy_indices = [i for i in range(len(agents)) if i not in agent_inds_of_team[team]]
    nearest_enemy_dist = float('inf')
    nearest_enemy_pos = None
    
    for enemy_idx in enemy_indices:
        enemy_pos = state["agent_position"][enemy_idx]
        dist = np.linalg.norm(current_pos - enemy_pos)
        if dist < nearest_enemy_dist:
            nearest_enemy_dist = dist
            nearest_enemy_pos = enemy_pos
    
    dist_to_own_flag = np.linalg.norm(current_pos - own_flag_pos)
    
    # DENSE REWARD 1: Stay near own flag (but not too close)
    if dist_to_own_flag < 0.5:
        reward += 0.3
    elif dist_to_own_flag > 1.0:
        reward -= 0.1  # Penalty for wandering too far
    
    # DENSE REWARD 2: Move toward enemies that are near our flag
    if nearest_enemy_pos is not None:
        enemy_dist_to_our_flag = np.linalg.norm(nearest_enemy_pos - own_flag_pos)
        if enemy_dist_to_our_flag < 0.6:  # Enemy is threatening
            prev_enemy_dist = np.linalg.norm(prev_pos - nearest_enemy_pos)
            intercept_progress = prev_enemy_dist - nearest_enemy_dist
            reward += 1.5 * intercept_progress
    
    # Sparse rewards
    for t in range(len(state["captures"])):
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 12.0 if t == int(team) else -12.0
    
    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            if t != int(team):  # Enemy grabbed - defender failed
                reward -= 3.0
    
    # Penalties
    if state["agent_oob"][idx] > prev_state["agent_oob"][idx]:
        reward -= 2.0
    
    movement = np.linalg.norm(current_pos - prev_pos)
    if movement < 0.02:
        reward -= 0.3
    
    return reward
