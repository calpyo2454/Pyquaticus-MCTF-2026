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
USE_LEARNED_OPPONENT_COMMANDER = False


import math


def _find_entropy_by_policy(obj, prefix=""):
    """
    Recursively find numeric entropy fields and bucket them by policy name
    if the path contains worker_policy or commander_policy.
    """
    found = {"worker_policy": [], "commander_policy": [], "other": []}

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                child = f"{path}.{k}" if path else str(k)
                key_lower = str(k).lower()

                if "entropy" in key_lower and isinstance(v, (int, float)):
                    if "worker_policy" in child:
                        found["worker_policy"].append((child, float(v)))
                    elif "commander_policy" in child:
                        found["commander_policy"].append((child, float(v)))
                    else:
                        found["other"].append((child, float(v)))

                walk(v, child)

        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj, prefix)
    return found


def _normalized_entropy(entropy_value: float | None, action_space_n: int | None):
    if entropy_value is None or action_space_n is None or action_space_n <= 1:
        return None
    return float(entropy_value / math.log(action_space_n))

def _find_first_metric(obj, target_keys, prefix=""):
    """
    Recursively search a nested dict/list structure for the first numeric value
    whose key matches one of target_keys.
    Returns (path, value) or (None, None).
    """
    target_keys = {str(k).lower() for k in target_keys}

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                child = f"{path}.{k}" if path else str(k)
                key_lower = str(k).lower()

                if key_lower in target_keys and isinstance(v, (int, float)):
                    return child, float(v)

                found_path, found_value = walk(v, child)
                if found_path is not None:
                    return found_path, found_value

        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                child = f"{path}[{i}]"
                found_path, found_value = walk(v, child)
                if found_path is not None:
                    return found_path, found_value

        return None, None

    return walk(obj, prefix)


def _extract_training_metrics(result):
    """
    Robustly pull reward/return/episode metrics from an RLlib result dict.
    """
    reward_mean_path, reward_mean = _find_first_metric(
        result, {"episode_reward_mean", "episode_return_mean"}
    )
    reward_min_path, reward_min = _find_first_metric(
        result, {"episode_reward_min", "episode_return_min"}
    )
    reward_max_path, reward_max = _find_first_metric(
        result, {"episode_reward_max", "episode_return_max"}
    )
    ep_len_mean_path, ep_len_mean = _find_first_metric(
        result, {"episode_len_mean"}
    )
    episodes_this_iter_path, episodes_this_iter = _find_first_metric(
        result, {"episodes_this_iter", "num_episodes"}
    )

    return {
        "reward_mean": reward_mean,
        "reward_mean_path": reward_mean_path,
        "reward_min": reward_min,
        "reward_min_path": reward_min_path,
        "reward_max": reward_max,
        "reward_max_path": reward_max_path,
        "ep_len_mean": ep_len_mean,
        "ep_len_mean_path": ep_len_mean_path,
        "episodes_this_iter": episodes_this_iter,
        "episodes_this_iter_path": episodes_this_iter_path,
    }


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
    if agent_id == "red_commander" and USE_LEARNED_OPPONENT_COMMANDER:
        return "opponent_commander_policy"
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
    parser.add_argument("--use-learned-opponent-commander", action="store_true")
    parser.add_argument("--entropy-coeff", type=float, default=0.003)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--worker-only-finetune", action="store_true")
    
    args = parser.parse_args()

    global USE_FROZEN_OPPONENT, USE_LEARNED_OPPONENT_COMMANDER
    USE_FROZEN_OPPONENT = args.frozen_opponent_checkpoint is not None
    USE_LEARNED_OPPONENT_COMMANDER = args.use_learned_opponent_commander

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
        "use_learned_commander": (args.train_mode in {"commander", "joint"}) or args.worker_only_finetune,
        "use_learned_opponent_commander": args.use_learned_opponent_commander,
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
    opponent_commander_obs_space = raw_env.observation_spaces.get("red_commander")
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

    if env_config.get("use_learned_opponent_commander", False):
        policies["opponent_commander_policy"] = (
            None,
            opponent_commander_obs_space,
            gym.spaces.Discrete(len(ROLE_CONFIGS)),
            {"model": {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}},
        )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    if args.worker_only_finetune:
        policies_to_train = ["worker_policy"]
    elif args.train_mode == "worker":
        policies_to_train = ["worker_policy"]
    elif args.train_mode == "commander":
        policies_to_train = ["commander_policy"]
    else:
        policies_to_train = ["worker_policy", "commander_policy"]

    algo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="pyquaticus_hierarchical_roles_3v3", env_config=env_config)
        .env_runners(num_env_runners=4, num_cpus_per_env_runner=1, num_envs_per_env_runner=1, rollout_fragment_length=300)
        .resources(num_gpus=0, num_cpus_for_main_process=1)
        .framework("torch")
        .debugging(log_level="ERROR")
        .training(train_batch_size=4500, minibatch_size=512, num_epochs=8, lr=args.lr,
                  gamma=0.995, lambda_=0.98, clip_param=0.2, entropy_coeff=args.entropy_coeff, vf_loss_coeff=1.0)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=policies_to_train)
    )

    algo = algo_config.build()

    # Resume whole run first, then re-freeze worker/opponent as needed.
    if args.checkpoint:
        checkpoint_path = os.path.abspath(args.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        print(f"Restoring from checkpoint: {checkpoint_path}")
        algo.restore(checkpoint_path)

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

        # Freeze red worker from the checkpoint's worker policy.
        algo.get_policy("opponent_policy").set_weights(
            frozen_algo.get_policy("worker_policy").get_weights()
        )

        # Optionally freeze red commander from the checkpoint's commander policy.
        if args.use_learned_opponent_commander:
            try:
                algo.get_policy("opponent_commander_policy").set_weights(
                    frozen_algo.get_policy("commander_policy").get_weights()
                )
                print("Loaded frozen opponent commander from checkpoint commander_policy")
            except Exception as e:
                print(f"Warning: could not load opponent commander weights: {e}")

        frozen_algo.stop()

    training_loop_start = time.perf_counter()
    iteration_times = []
    for i in range(args.iterations):
        iter_start = time.perf_counter()
        result = algo.train()
        
        entropy_by_policy = _find_entropy_by_policy(result)

        worker_entropy = (
            entropy_by_policy["worker_policy"][0][1]
            if entropy_by_policy["worker_policy"] else None
        )
        commander_entropy = (
            entropy_by_policy["commander_policy"][0][1]
            if entropy_by_policy["commander_policy"] else None
        )

        worker_action_n = getattr(worker_act_space, "n", None)
        commander_action_n = len(ROLE_CONFIGS) if args.train_mode in {"commander", "joint"} else None

        worker_entropy_norm = _normalized_entropy(worker_entropy, worker_action_n)
        commander_entropy_norm = _normalized_entropy(commander_entropy, commander_action_n)
        
        iter_end = time.perf_counter()
        iteration_times.append(iter_end - iter_start)

        metrics = _extract_training_metrics(result)

        if i == 0:
            print("top-level result keys:", sorted(result.keys()))
            print("env_runners keys:", sorted(result.get("env_runners", {}).keys()))
            print(
                "metric paths found:",
                {
                    "reward_mean": metrics["reward_mean_path"],
                    "reward_min": metrics["reward_min_path"],
                    "reward_max": metrics["reward_max_path"],
                    "ep_len_mean": metrics["ep_len_mean_path"],
                    "episodes_this_iter": metrics["episodes_this_iter_path"],
                },
            )
            print("worker entropy keys:", entropy_by_policy["worker_policy"])
            print("commander entropy keys:", entropy_by_policy["commander_policy"])
            print("other entropy keys:", entropy_by_policy["other"])

        if i % 10 == 0:
            print(
                f"[iter {i}] "
                f"reward_mean={metrics['reward_mean']} "
                f"reward_min={metrics['reward_min']} "
                f"reward_max={metrics['reward_max']} "
                f"ep_len_mean={metrics['ep_len_mean']} "
                f"episodes_this_iter={metrics['episodes_this_iter']}"
            )

            print(
                f"[iter {i}] "
                f"worker_entropy={worker_entropy} "
                f"worker_entropy_norm={worker_entropy_norm} "
                f"commander_entropy={commander_entropy} "
                f"commander_entropy_norm={commander_entropy_norm}"
            )

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
