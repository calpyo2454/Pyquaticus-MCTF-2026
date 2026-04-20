# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

# Add parent directory for imports
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import pyquaticus.utils.rewards as rew


class RandomPolicy:
    """Simple random opponent policy for deployment."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def compute_action(self, obs):
        return self.action_space.sample()


def build_env_config():
    """Match the exact env config from training."""
    config_dict = config_dict_std.copy()
    config_dict.update({
        "sim_speedup_factor": 20,
        "max_score": 3,
        "max_time": 240,
        "tagging_cooldown": 60,
        "tag_on_oob": True,
    })
    
    reward_config = {
        "agent_0": rew.caps_and_grabs,
        "agent_1": rew.caps_and_grabs,
        "agent_2": rew.caps_and_grabs,
        "agent_3": None,
        "agent_4": None,
        "agent_5": None,
    }
    
    return {
        "config_dict": config_dict,
        "render_mode": None,
        "reward_config": reward_config,
        "team_size": 3,
    }


def make_env_creator(base_env_kwargs):
    def env_creator(_config):
        return pyquaticus_v0.PyQuaticusEnv(**base_env_kwargs)
    return env_creator


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Match the policy mapping from training (shared-policy for dense rewards)."""
    if agent_id in ['agent_0', 'agent_1', 'agent_2']:
        return "shared-policy"
    return "random-policy"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy trained 3v3 PPO policies')
    parser.add_argument('checkpoint', help='Path to PPO algorithm checkpoint (e.g., ./ray_test/run_YYYYMMDD_HHMMSS/optimized_final)')
    parser.add_argument('--env-name', type=str, default="pyquaticus_optimized_3v3", help='Registered environment name')
    args = parser.parse_args()
    
    # Build environment config (same as training)
    base_env_kwargs = build_env_config()
    env_creator = make_env_creator(base_env_kwargs)
    
    # Register environment with same wrapper as training
    register_env(args.env_name, lambda config: ParallelPettingZooWrapper(env_creator(config)))
    
    # Create temp env to get spaces
    temp_env = ParallelPettingZooWrapper(env_creator({}))
    obs_space = temp_env.observation_space["agent_0"]
    act_space = temp_env.action_space["agent_0"]
    temp_env.close()
    
    # Restore the full PPO algorithm
    print(f"Restoring checkpoint from: {args.checkpoint}")
    algo = PPO.from_checkpoint(os.path.abspath(args.checkpoint))
    print(f"Checkpoint loaded. Algorithm type: {type(algo)}")
    
    # Create wrapped environment for deployment
    env = ParallelPettingZooWrapper(env_creator({}))
    
    # Create random policy for opponents
    random_policy = RandomPolicy(act_space)
    
    obs, _ = env.reset()
    step = 0
    max_step = 2500
    
    print("Starting deployment...")
    print(f"Agents in environment: {list(obs.keys())}")
    
    while True:
        actions = {}
        
        # Get actions for each agent using the trained policies
        for agent_id in obs.keys():
            if agent_id in ['agent_0', 'agent_1', 'agent_2']:
                # Use trained shared policy
                if step == 0:
                    print(f"Agent {agent_id} using shared-policy")
                actions[agent_id] = algo.compute_single_action(
                    obs[agent_id],
                    policy_id="shared-policy"
                )
            else:
                # Use random policy for opponents
                if step == 0:
                    print(f"Agent {agent_id} using random policy")
                actions[agent_id] = random_policy.compute_action(obs[agent_id])
        
        # Step environment
        obs, reward, term, trunc, info = env.step(actions)
        
        step += 1
        if step >= max_step:
            break
        
        # Check if episode is done
        if any(term.values()) or any(trunc.values()):
            obs, _ = env.reset()
            print(f"Episode ended at step {step}")
    
    env.close()
    print("Deployment completed.")


