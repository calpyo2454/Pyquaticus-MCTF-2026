from __future__ import annotations

import argparse
import logging
import os
import time

import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper

from hierarchical_framework.commander_action import ROLE_CONFIGS
from hierarchical_framework.hierarchical_team_wrapper import (
    HierarchicalTeamWrapper,
    SharedEncoderRoleHeadModel,
)

USE_FROZEN_OPPONENT = False


class RandPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                        prev_reward_batch=None, info_batch=None, episodes=None,
                        explore=None, timestep=None, **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

    def learn_on_batch(self, samples):
        return {}


def make_raw_hierarchical_env(env_config):
    config_dict = env_config["config_dict"]
    render_mode = env_config.get("render_mode", None)
    base_env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=config_dict,
        render_mode=render_mode,
        reward_config=env_config.get("reward_config", None),
        team_size=3,
    )
    return HierarchicalTeamWrapper(
        base_env=base_env,
        role_period=env_config.get("role_period", 10),
        max_time=config_dict.get("max_time", 300),
        shape_blue_worker_rewards=env_config.get("shape_blue_worker_rewards", True),
        env_config=env_config,
    )


def env_creator(env_config):
    raw_env = make_raw_hierarchical_env(env_config)
    return ParallelPettingZooWrapper(raw_env)


def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "blue_commander":
        return "commander_policy"
    if agent_id in ["agent_0", "agent_1", "agent_2"]:
        return "worker_policy"
    return "opponent_policy" if USE_FROZEN_OPPONENT else "random_policy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--save-dir", type=str, default="./hierarchical_checkpoints/baseline_checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--role-period", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--train-mode", choices=["worker", "commander", "joint"], default="worker")
    parser.add_argument("--frozen-worker-checkpoint", type=str, default="./hierarchical_checkpoints_vs_frozen")
    parser.add_argument("--frozen-opponent-checkpoint", type=str, default=None)
    args = parser.parse_args()

    global USE_FROZEN_OPPONENT
    USE_FROZEN_OPPONENT = args.frozen_opponent_checkpoint is not None

    logging.basicConfig(level=logging.ERROR)
    os.makedirs(args.save_dir, exist_ok=True)

    config_dict = config_dict_std.copy()
    config_dict["sim_speedup_factor"] = 8
    config_dict["max_score"] = 5
    config_dict["max_time"] = 300
    config_dict["tagging_cooldown"] = 45
    config_dict["tag_on_oob"] = True

    env_config = {
        "config_dict": config_dict,
        "render_mode": "human" if args.render else None,
        "team_size": 3,
        "role_period": args.role_period,
        "shape_blue_worker_rewards": True,
        "use_learned_commander": args.train_mode in {"commander", "joint"},
        "commander_team": "blue",
    }

    register_env("pyquaticus_hierarchical_roles_3v3", env_creator)
    ModelCatalog.register_custom_model("shared_encoder_role_heads", SharedEncoderRoleHeadModel)

    raw_env = make_raw_hierarchical_env(env_config)
    worker_obs_space = raw_env.observation_spaces["agent_0"]
    worker_act_space = raw_env.action_spaces["agent_0"]
    red_obs_space = raw_env.observation_spaces["agent_3"]
    red_act_space = raw_env.action_spaces["agent_3"]
    commander_obs_space = raw_env.observation_spaces.get("blue_commander")
    raw_env.close()

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
        "random_policy": (
            RandPolicy,
            red_obs_space,
            red_act_space,
            {"no_checkpoint": True},
        ),
    }
    if env_config["use_learned_commander"]:
        policies["commander_policy"] = (
            None,
            commander_obs_space,
            gym.spaces.Discrete(len(ROLE_CONFIGS)),
            {"model": {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}},
        )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    if args.train_mode == "worker":
        policies_to_train = ["worker_policy"]
    elif args.train_mode == "commander":
        policies_to_train = ["commander_policy"]
    else:
        policies_to_train = ["worker_policy", "commander_policy"]

    algo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="pyquaticus_hierarchical_roles_3v3", env_config=env_config)
        .env_runners(num_env_runners=3, num_cpus_per_env_runner=1, num_envs_per_env_runner=1, rollout_fragment_length=300)
        .resources(num_gpus=0, num_cpus_for_main_process=1)
        .framework("torch")
        .debugging(log_level="ERROR")
        .training(train_batch_size=4500, minibatch_size=512, num_epochs=8, lr=3e-4,
                  gamma=0.995, lambda_=0.98, clip_param=0.2, entropy_coeff=0.003, vf_loss_coeff=1.0)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=policies_to_train)
    )

    algo = algo_config.build()

    # Resume whole run first, then re-freeze worker/opponent as needed.
    if args.checkpoint:
        print(f"Restoring from checkpoint: {args.checkpoint}")
        algo.restore(args.checkpoint)

    if args.train_mode == "commander":
        frozen_worker_path = os.path.abspath(args.frozen_worker_checkpoint)
        if not os.path.exists(frozen_worker_path):
            raise FileNotFoundError(frozen_worker_path)
        print(f"Loading frozen worker from: {frozen_worker_path}")
        frozen_algo = algo_config.build()
        frozen_algo.restore(frozen_worker_path)
        algo.get_policy("worker_policy").set_weights(frozen_algo.get_policy("worker_policy").get_weights())
        frozen_algo.stop()

    if args.frozen_opponent_checkpoint:
        frozen_path = os.path.abspath(args.frozen_opponent_checkpoint)
        if not os.path.exists(frozen_path):
            raise FileNotFoundError(frozen_path)
        print(f"Loading frozen opponent from: {frozen_path}")
        frozen_algo = algo_config.build()
        frozen_algo.restore(frozen_path)
        algo.get_policy("opponent_policy").set_weights(frozen_algo.get_policy("worker_policy").get_weights())
        frozen_algo.stop()

    training_loop_start = time.perf_counter()
    iteration_times = []
    for i in range(args.iterations):
        iter_start = time.perf_counter()
        result = algo.train()
        iter_end = time.perf_counter()
        iteration_times.append(iter_end - iter_start)

        if i == 0:
            print("top-level result keys:", sorted(result.keys()))
            print("env_runners keys:", sorted(result.get("env_runners", {}).keys()))

        if i % 10 == 0:
            env_runner_metrics = result.get("env_runners", {})
            reward_mean = env_runner_metrics.get("episode_reward_mean", env_runner_metrics.get("episode_return_mean", None))
            reward_min = env_runner_metrics.get("episode_reward_min", None)
            reward_max = env_runner_metrics.get("episode_reward_max", None)
            ep_len_mean = env_runner_metrics.get("episode_len_mean", None)
            episodes_this_iter = result.get("episodes_this_iter", None)
            print(f"[iter {i}] reward_mean={reward_mean} reward_min={reward_min} reward_max={reward_max} ep_len_mean={ep_len_mean} episodes_this_iter={episodes_this_iter}")

        if i > 0 and i % args.checkpoint_every == 0:
            save_result = algo.save(args.save_dir)
            print(f"Saved checkpoint at iter {i}: {save_result.checkpoint.path}")

    total_training_time = time.perf_counter() - training_loop_start
    avg_iteration_time = total_training_time / max(len(iteration_times), 1)
    print("\n=== TRAINING TIME SUMMARY ===")
    print(f"Train mode: {args.train_mode}")
    print(f"Total training iteration wall time: {total_training_time:.2f} seconds")
    print(f"Average wall time per iteration: {avg_iteration_time:.2f} seconds")
    print(f"Completed iterations: {len(iteration_times)}")

    final_save_result = algo.save(args.save_dir)
    print(f"Final checkpoint: {final_save_result.checkpoint.path}")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
