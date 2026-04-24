from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper

from hierarchical_framework.commander_action import ROLE_CONFIGS
from hierarchical_framework.hierarchical_team_wrapper import HierarchicalTeamWrapper, SharedEncoderRoleHeadModel

USE_FROZEN_OPPONENT = False
USE_LEARNED_COMMANDER = False


class RandPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

    def learn_on_batch(self, samples):
        return {}


def make_raw_hierarchical_env(env_config):
    base_env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=env_config["config_dict"],
        render_mode=env_config.get("render_mode", None),
        reward_config=env_config.get("reward_config", None),
        team_size=env_config.get("team_size", 3),
    )
    return HierarchicalTeamWrapper(
        base_env=base_env,
        role_period=env_config.get("role_period", 5),
        max_time=env_config["config_dict"].get("max_time", 240),
        shape_blue_worker_rewards=env_config.get("shape_blue_worker_rewards", True),
        env_config=env_config,
    )


def env_creator(env_config):
    raw_env = make_raw_hierarchical_env(env_config)
    return ParallelPettingZooWrapper(raw_env)


def policy_mapping_fn(agent_id, episode, **kwargs):
    if USE_LEARNED_COMMANDER and agent_id == "blue_commander":
        return "commander_policy"
    if agent_id in ["agent_0", "agent_1", "agent_2"]:
        return "worker_policy"
    return "opponent_policy" if USE_FROZEN_OPPONENT else "random_policy"


def build_env_config(role_period: int, render_mode: str, use_learned_commander: bool):
    config_dict = config_dict_std.copy()
    config_dict["sim_speedup_factor"] = 6
    config_dict["max_score"] = 3
    config_dict["max_time"] = 240
    config_dict["tagging_cooldown"] = 45
    config_dict["tag_on_oob"] = True
    return {
        "config_dict": config_dict,
        "render_mode": render_mode,
        "team_size": 3,
        "role_period": role_period,
        "shape_blue_worker_rewards": True,
        "use_learned_commander": use_learned_commander,
        "commander_team": "blue",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy trained hierarchical PPO in rendered GUI")
    parser.add_argument("checkpoint", help="Path to PPO algorithm checkpoint directory")
    parser.add_argument("--role-period", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--env-name", type=str, default="pyquaticus_hierarchical_roles_3v3")
    parser.add_argument("--frozen-opponent-checkpoint", type=str, default=None)
    parser.add_argument("--use-learned-commander", action="store_true")
    args = parser.parse_args()

    USE_FROZEN_OPPONENT = args.frozen_opponent_checkpoint is not None
    USE_LEARNED_COMMANDER = args.use_learned_commander

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    render_env_config = build_env_config(args.role_period, "human", args.use_learned_commander)
    register_env(args.env_name, lambda config: env_creator(render_env_config))
    ModelCatalog.register_custom_model("shared_encoder_role_heads", SharedEncoderRoleHeadModel)

    temp_env_config = build_env_config(args.role_period, None, args.use_learned_commander)
    temp_env = env_creator(temp_env_config)
    worker_obs_space = temp_env.observation_space["agent_0"]
    worker_act_space = temp_env.action_space["agent_0"]
    red_obs_space = temp_env.observation_space["agent_3"]
    red_act_space = temp_env.action_space["agent_3"]
    commander_obs_space = temp_env.observation_space.get("blue_commander")
    temp_env.close()

    policies = {
        "worker_policy": (
            None,
            worker_obs_space,
            worker_act_space,
            {"model": {"custom_model": "shared_encoder_role_heads", "custom_model_config": {"hidden_dim": 256, "head_dim": 128}}},
        ),
        "opponent_policy": (
            None,
            red_obs_space,
            red_act_space,
            {"model": {"custom_model": "shared_encoder_role_heads", "custom_model_config": {"hidden_dim": 256, "head_dim": 128}}},
        ),
        "random_policy": (RandPolicy, red_obs_space, red_act_space, {"no_checkpoint": True}),
    }
    if args.use_learned_commander:
        policies["commander_policy"] = (
            None,
            commander_obs_space,
            gym.spaces.Discrete(len(ROLE_CONFIGS)),
            {"model": {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}},
        )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    algo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=args.env_name, env_config=render_env_config)
        .env_runners(num_env_runners=0)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[],
        )
    )

    print(f"Restoring checkpoint from: {checkpoint_path}")
    algo = algo_config.build()
    algo.restore(checkpoint_path)

    if args.frozen_opponent_checkpoint:
        frozen_path = os.path.abspath(args.frozen_opponent_checkpoint)
        if not os.path.exists(frozen_path):
            raise FileNotFoundError(frozen_path)
        print(f"Loading frozen opponent from: {frozen_path}")
        frozen_algo = algo_config.build()
        frozen_algo.restore(frozen_path)
        frozen_weights = frozen_algo.get_policy("worker_policy").get_weights()
        algo.get_policy("opponent_policy").set_weights(frozen_weights)
        frozen_algo.stop()

    env = env_creator(render_env_config)
    obs, _ = env.reset()
    step = 0
    print("Starting deployment...")

    while True:
        actions = {}
        for agent_id in obs.keys():
            if USE_LEARNED_COMMANDER and agent_id == "blue_commander":
                actions[agent_id] = algo.compute_single_action(obs[agent_id], policy_id="commander_policy")
            elif agent_id in ["agent_0", "agent_1", "agent_2"]:
                actions[agent_id] = algo.compute_single_action(obs[agent_id], policy_id="worker_policy")
            else:
                if USE_FROZEN_OPPONENT:
                    actions[agent_id] = algo.compute_single_action(obs[agent_id], policy_id="opponent_policy")
                else:
                    actions[agent_id] = env.par_env.action_space(agent_id).sample()

        obs, reward, term, trunc, info = env.step(actions)
        step += 1

        if step % args.print_every == 0:
            print(f"\n[step {step}]")
            for agent_id in ["agent_0", "agent_1", "agent_2"]:
                role_name = info.get(agent_id, {}).get("role_name", "UNKNOWN")
                reason = info.get(agent_id, {}).get("commander_reason", "UNKNOWN")
                config_idx = info.get(agent_id, {}).get("commander_config_index", None)
                print(f"  {agent_id}: role={role_name} reason={reason} config={config_idx}")
            if args.use_learned_commander and "blue_commander" in info:
                print(f"  blue_commander: {info['blue_commander']}")

        if step >= args.max_steps:
            break
        if any(term.values()) or any(trunc.values()):
            obs, _ = env.reset()
            print(f"Episode ended at total step {step}")

    env.close()
    algo.stop()
    ray.shutdown()
    print("Deployment completed.")
