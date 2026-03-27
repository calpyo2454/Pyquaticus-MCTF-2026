"""
rl_test/eval_hierarchical_roles_3v3.py

Purpose of this module:
- Evaluate a trained shared worker PPO policy under the scripted commander baseline.
- Report win rate, role usage, and commander reasons.

What this script evaluates:
- worker_policy for blue team
- random_policy for red team

Why this script matters now:
- It tells you whether the shared worker is actually responding to
  the role assignments produced by the scripted commander.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper

from hierarchical_framework.commander_action import ATTACK, DEFEND, INTERCEPT, ROLE_NAMES
from hierarchical_framework.hierarchical_team_wrapper import HierarchicalTeamWrapper


class RandPolicy(Policy):
    """
    Random opponent policy used for evaluation.

    Why this class exists:
    - Evaluation should use the same red-team baseline behavior as training
      unless you intentionally switch opponents later.
    """

    def __init__(self, observation_space, action_space, config):
        """Store action/observation spaces for random action sampling."""
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
        """Sample one random primitive action per red-agent observation."""
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        """Return empty weights because this policy is not trainable."""
        return {}

    def set_weights(self, weights):
        """Accept no-op weight updates because this policy is random."""
        pass

    def learn_on_batch(self, samples):
        """Return empty training stats because this policy does not learn."""
        return {}


def make_raw_hierarchical_env(env_config):
    """
    Construct the raw hierarchical wrapper before RLlib's PettingZoo wrapper.

    Why this method exists:
    - The evaluation script needs direct access to wrapped spaces to define
      the worker and random policy specs.
    """
    config_dict = env_config["config_dict"]
    render_mode = env_config.get("render_mode", None)

    base_env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=config_dict,
        render_mode=render_mode,
        reward_config=env_config.get("reward_config", None),
        team_size=env_config.get("team_size", 3),
    )

    return HierarchicalTeamWrapper(
        base_env=base_env,
        role_period=env_config.get("role_period", 10),
        max_time=config_dict.get("max_time", 300),
        shape_blue_worker_rewards=env_config.get("shape_blue_worker_rewards", True),
    )


def env_creator(env_config):
    """
    Create the RLlib-ready wrapped environment.

    Why this method exists:
    - RLlib expects the environment returned here when restoring the trained PPO.
    """
    raw_env = make_raw_hierarchical_env(env_config)
    return ParallelPettingZooWrapper(raw_env)


def policy_mapping_fn(agent_id, episode, **kwargs):
    """
    Match blue workers to the trained worker policy and red workers to random.

    Why this method exists:
    - Evaluation must mirror the same role of each policy used during training.
    """
    if agent_id in ["agent_0", "agent_1", "agent_2"]:
        return "worker_policy"
    return "random_policy"


def evaluate_episode(algo, env, render: bool = False):
    """
    Run one full episode and collect role and commander diagnostics.

    Why this method exists:
    - A simple average reward is not enough for hierarchical debugging.
    - We also want to know which roles were used and why the commander chose them.
    """
    obs, info = env.reset()

    # Use agents returned by reset.
    active_agents = list(obs.keys())

    episode_rewards = {agent: 0.0 for agent in active_agents}
    role_counts = {aid: {ATTACK: 0, DEFEND: 0, INTERCEPT: 0} for aid in ["agent_0", "agent_1", "agent_2"]}
    commander_reasons = {}
    steps = 0

    # Safety cap so eval cannot hang forever if done signals behave unexpectedly.
    max_eval_steps = 1000

    while active_agents and steps < max_eval_steps:
        actions = {}

        for aid in active_agents:
            if aid in ["agent_0", "agent_1", "agent_2"]:
                action = algo.compute_single_action(obs[aid], policy_id="worker_policy")
                actions[aid] = action
            else:
                # env is the RLlib wrapper; sample from the underlying parallel env.
                action = env.par_env.action_space(aid).sample()
                actions[aid] = action

        obs, rewards, terminated, truncated, info = env.step(actions)

        for aid, r in rewards.items():
            episode_rewards[aid] = episode_rewards.get(aid, 0.0) + r

        for aid in ["agent_0", "agent_1", "agent_2"]:
            if aid in info:
                role_id = info.get(aid, {}).get("role_id", ATTACK)
                role_counts[aid][role_id] += 1

        # Count commander reason once per step, using agent_0 as representative.
        reason = info.get("agent_0", {}).get("commander_reason", None)
        if reason is not None:
            commander_reasons[reason] = commander_reasons.get(reason, 0) + 1

        steps += 1

        if steps % 50 == 0:
            print(f"  eval progress: step {steps}, active_agents={list(obs.keys())}")

        if render:
            try:
                env.render()
            except Exception:
                pass

        # If all agents are done, end the episode cleanly.
        if terminated and truncated:
            all_done = True
            all_agents_seen = set(terminated.keys()) | set(truncated.keys())
            for aid in all_agents_seen:
                if not terminated.get(aid, False) and not truncated.get(aid, False):
                    all_done = False
                    break
            if all_done:
                break

        # Refresh active agents from the latest observation dict.
        active_agents = list(obs.keys())

    if steps >= max_eval_steps:
        print(f"  warning: hit max_eval_steps={max_eval_steps}, forcing episode stop")

    # Best-effort attempt to access final score/capture stats through nested wrappers.
    try:
        inner_env = env.par_env
        if hasattr(inner_env, "base_env"):
            state = inner_env.base_env.state
        elif hasattr(inner_env, "state"):
            state = inner_env.state
        else:
            state = None

        if state is not None:
            final_blue_captures = int(state["captures"][0])
            final_red_captures = int(state["captures"][1])
    except Exception:
        pass

    return {
        "episode_rewards": episode_rewards,
        "steps": steps,
        "role_counts": role_counts,
        "commander_reasons": commander_reasons,
        "blue_captures": final_blue_captures,
        "red_captures": final_red_captures,
    }


def main():
    """
    Parse CLI args, restore a checkpoint, run evaluation, and print summary stats.

    Why this method exists:
    - It ties together checkpoint loading, rollout execution, and the key
      diagnostics needed to see whether the scripted-commander baseline is working.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--role-period", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    logging.basicConfig(level=logging.ERROR)

    config_dict = config_dict_std.copy()
    config_dict["sim_speedup_factor"] = 4
    config_dict["max_score"] = 3
    config_dict["max_time"] = 240
    config_dict["tagging_cooldown"] = 60
    config_dict["tag_on_oob"] = True
    config_dict["team_size"] = 3

    env_config = {
        "config_dict": config_dict,
        "render_mode": "human" if args.render else None,
        "team_size": 3,
        "role_period": args.role_period,
        "shape_blue_worker_rewards": True,
    }

    register_env("pyquaticus_hierarchical_roles_3v3", env_creator)

    raw_env = make_raw_hierarchical_env(env_config)

    # Blue workers get role-augmented observations.
    worker_obs_space = raw_env.observation_spaces["agent_0"]
    worker_act_space = raw_env.action_spaces["agent_0"]

    # Red workers keep the base observation shape.
    red_obs_space = raw_env.observation_spaces["agent_3"]
    red_act_space = raw_env.action_spaces["agent_3"]

    raw_env.close()

    policies = {
        "worker_policy": (None, worker_obs_space, worker_act_space, {}),
        "random_policy": (RandPolicy, red_obs_space, red_act_space, {"no_checkpoint": True}),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    algo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="pyquaticus_hierarchical_roles_3v3", env_config=env_config)
        .env_runners(num_env_runners=0)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[],
        )
    )

    algo = algo_config.build()
    #algo.restore(args.checkpoint)
    algo.restore(checkpoint_path)

    env = env_creator(env_config)

    total_blue_wins = 0
    aggregate_role_counts = {aid: {ATTACK: 0, DEFEND: 0, INTERCEPT: 0} for aid in ["agent_0", "agent_1", "agent_2"]}
    aggregate_reasons = {}

    for ep in range(args.episodes):
        stats = evaluate_episode(algo, env, render=args.render)

        blue_caps = stats["blue_captures"]
        red_caps = stats["red_captures"]
        if blue_caps is not None and red_caps is not None and blue_caps > red_caps:
            total_blue_wins += 1

        for aid in ["agent_0", "agent_1", "agent_2"]:
            for role_id in [ATTACK, DEFEND, INTERCEPT]:
                aggregate_role_counts[aid][role_id] += stats["role_counts"][aid][role_id]

        for reason, count in stats["commander_reasons"].items():
            aggregate_reasons[reason] = aggregate_reasons.get(reason, 0) + count

        print(f"[episode {ep + 1}/{args.episodes}] "
              f"blue_caps={blue_caps} red_caps={red_caps} steps={stats['steps']}")

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Blue win count: {total_blue_wins} / {args.episodes}")

    print("\nRole usage by blue worker:")
    for aid in ["agent_0", "agent_1", "agent_2"]:
        total = sum(aggregate_role_counts[aid].values())
        print(f"  {aid}:")
        for role_id in [ATTACK, DEFEND, INTERCEPT]:
            pct = 100.0 * aggregate_role_counts[aid][role_id] / max(total, 1)
            print(f"    {ROLE_NAMES[role_id]} = {aggregate_role_counts[aid][role_id]} ({pct:.1f}%)")

    print("\nCommander reasons:")
    for reason, count in sorted(aggregate_reasons.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {reason} = {count}")

    env.close()
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()