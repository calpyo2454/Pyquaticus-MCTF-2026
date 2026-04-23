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
from ray.rllib.models import ModelCatalog

from hierarchical_framework.hierarchical_team_wrapper import (
    HierarchicalTeamWrapper,
    SharedEncoderRoleHeadModel,
)

USE_FROZEN_OPPONENT = False


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
    if agent_id in ["agent_0", "agent_1", "agent_2"]:
        return "worker_policy"
    return "opponent_policy" if USE_FROZEN_OPPONENT else "random_policy"

def _agent_index_from_id(agent_id: str) -> int:
    """
    Convert an agent name like 'agent_3' into integer index 3.

    Why this helper exists:
    - The underlying Pyquaticus state arrays are indexed by integer agent index.
    - The evaluation script receives string IDs, so we need a robust conversion.
    """
    return int(agent_id.split("_")[1])


def _safe_state_value(state: dict, key: str, index: int, default=None):
    """
    Safely read a per-agent value from environment state.

    Why this helper exists:
    - Evaluation should not crash if a state key is missing or shaped
      unexpectedly in a particular rollout.
    """
    if state is None or key not in state:
        return default
    try:
        return state[key][index]
    except Exception:
        return default


def _get_inner_state(env):
    """
    Best-effort access to the wrapped Pyquaticus state.

    Why this helper exists:
    - The RLlib wrapper does not expose raw state directly.
    - We need raw state to compute side occupancy, flag possession,
      and tag-by-role metrics.
    """
    try:
        if hasattr(env, "par_env"):
            par_env = env.par_env

            if hasattr(par_env, "_get_state"):
                return par_env._get_state()

            if hasattr(par_env, "base_env") and hasattr(par_env.base_env, "state"):
                return par_env.base_env.state

            if hasattr(par_env, "state"):
                return par_env.state
    except Exception:
        pass

    return None


def _bucket_side_for_agent(agent_id: str, state: dict) -> str:
    """
    Classify an agent's current side as home_side, enemy_side, or unknown_side.

    Why this helper exists:
    - We want to measure whether ATTACK spends more time away from home,
      DEFEND spends more time at home, and INTERCEPT shifts appropriately.

    Important assumption:
    - This uses agent_on_sides and assumes side index matches team index
      (blue=0, red=1). If your branch differs, this metric can still be
      kept as relative debug information but may need semantic adjustment later.
    """
    if state is None or "agent_on_sides" not in state:
        return "unknown_side"

    idx = _agent_index_from_id(agent_id)

    try:
        side_value = int(state["agent_on_sides"][idx])
    except Exception:
        return "unknown_side"

    num_agents = len(state["agent_on_sides"])
    half = num_agents // 2
    team_idx = 0 if idx < half else 1

    if side_value == team_idx:
        return "home_side"
    return "enemy_side"

def evaluate_episode(algo, env, render: bool = False):
    """
    Run one full evaluation episode and collect role-aware diagnostics.

    Metrics collected:
    - episode rewards
    - role occupancy by blue worker
    - commander reasons and config usage
    - side occupancy by blue worker and by role
    - tags made by current role
    - flag-carry steps by current role
    - role switches per blue worker

    Why this function exists:
    - Win/loss alone is too weak for hierarchical debugging.
    - We need to see whether ATTACK/DEFEND/INTERCEPT produce distinct behavior.
    """
    obs, info = env.reset()

    # Use the actually returned active agents from reset.
    active_agents = list(obs.keys())
    blue_agents = ["agent_0", "agent_1", "agent_2"]

    episode_rewards = {agent: 0.0 for agent in active_agents}
    role_counts = {aid: {ATTACK: 0, DEFEND: 0, INTERCEPT: 0} for aid in blue_agents}

    # Track territory occupancy per blue worker.
    side_steps_by_agent = {
        aid: {"home_side": 0, "enemy_side": 0, "unknown_side": 0}
        for aid in blue_agents
    }

    # Track territory occupancy aggregated by current role.
    side_steps_by_role = {
        ATTACK: {"home_side": 0, "enemy_side": 0, "unknown_side": 0},
        DEFEND: {"home_side": 0, "enemy_side": 0, "unknown_side": 0},
        INTERCEPT: {"home_side": 0, "enemy_side": 0, "unknown_side": 0},
    }

    # Track dense role-behavior metrics.
    tags_by_role = {ATTACK: 0, DEFEND: 0, INTERCEPT: 0}
    flag_carry_steps_by_role = {ATTACK: 0, DEFEND: 0, INTERCEPT: 0}

    # Track how often each blue worker changes assigned role.
    role_switches = {aid: 0 for aid in blue_agents}
    last_role_by_agent = {aid: None for aid in blue_agents}

    # Track commander-level usage.
    commander_reasons = {}
    commander_configs = {}

    steps = 0

    # Safety cap to avoid hanging forever if done handling is imperfect.
    max_eval_steps = 650

    while active_agents and steps < max_eval_steps:
        actions = {}

        for aid in active_agents:
            if aid in ["agent_0", "agent_1", "agent_2"]:
                action = algo.compute_single_action(obs[aid], policy_id="worker_policy")
                actions[aid] = action
            else:
                if USE_FROZEN_OPPONENT:
                    action = algo.compute_single_action(obs[aid], policy_id="opponent_policy")
                    actions[aid] = action
                else:
                    action = env.par_env.action_space(aid).sample()
                    actions[aid] = action

        obs, rewards, terminated, truncated, info = env.step(actions)

        # Best-effort raw state access for richer metrics.
        state = _get_inner_state(env)

        for aid, r in rewards.items():
            episode_rewards[aid] = episode_rewards.get(aid, 0.0) + r

        # Update blue-role diagnostics.
        for aid in blue_agents:
            if aid not in info:
                continue

            role_id = info.get(aid, {}).get("role_id", ATTACK)
            role_counts[aid][role_id] += 1

            # Count role switches after initial assignment is established.
            if last_role_by_agent[aid] is None:
                last_role_by_agent[aid] = role_id
            elif last_role_by_agent[aid] != role_id:
                role_switches[aid] += 1
                last_role_by_agent[aid] = role_id

            # Territory occupancy by agent and by role.
            side_bucket = _bucket_side_for_agent(aid, state)
            side_steps_by_agent[aid][side_bucket] += 1
            side_steps_by_role[role_id][side_bucket] += 1

            # Tags made while in current role.
            agent_index = _agent_index_from_id(aid)
            made_tag = _safe_state_value(state, "agent_made_tag", agent_index, None)
            if made_tag is not None:
                tags_by_role[role_id] += 1

            # Steps carrying enemy flag while in current role.
            has_flag = bool(_safe_state_value(state, "agent_has_flag", agent_index, False))
            if has_flag:
                flag_carry_steps_by_role[role_id] += 1

        # Count commander reason/config once per step using agent_0 as representative.
        step_reason = info.get("agent_0", {}).get("commander_reason", None)
        if step_reason is not None:
            commander_reasons[step_reason] = commander_reasons.get(step_reason, 0) + 1

        step_config = info.get("agent_0", {}).get("commander_config_index", None)
        if step_config is not None:
            commander_configs[step_config] = commander_configs.get(step_config, 0) + 1

        steps += 1

        if steps % 200 == 0:
            print(f"  eval progress: step {steps}, active_agents={list(obs.keys())}")

        if render:
            try:
                env.render()
            except Exception:
                pass

        # Stop if all agents are terminated or truncated.
        if terminated and truncated:
            all_done = True
            all_agents_seen = set(terminated.keys()) | set(truncated.keys())
            for aid in all_agents_seen:
                if not terminated.get(aid, False) and not truncated.get(aid, False):
                    all_done = False
                    break
            if all_done:
                break

        active_agents = list(obs.keys())

    if steps >= max_eval_steps:
        print(f"  warning: hit max_eval_steps={max_eval_steps}, forcing episode stop")

    final_blue_captures = None
    final_red_captures = None

    # Best-effort final capture readout.
    try:
        state = _get_inner_state(env)
        if state is not None and "captures" in state:
            final_blue_captures = int(state["captures"][0])
            final_red_captures = int(state["captures"][1])
    except Exception:
        pass

    return {
        "episode_rewards": episode_rewards,
        "steps": steps,
        "role_counts": role_counts,
        "side_steps_by_agent": side_steps_by_agent,
        "side_steps_by_role": side_steps_by_role,
        "tags_by_role": tags_by_role,
        "flag_carry_steps_by_role": flag_carry_steps_by_role,
        "role_switches": role_switches,
        "commander_reasons": commander_reasons,
        "commander_configs": commander_configs,
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
    parser.add_argument("--frozen-opponent-checkpoint", type=str, default=None)
    args = parser.parse_args()

    global USE_FROZEN_OPPONENT
    USE_FROZEN_OPPONENT = args.frozen_opponent_checkpoint is not None

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

    ModelCatalog.register_custom_model(
        "shared_encoder_role_heads",
        SharedEncoderRoleHeadModel,
    )

    raw_env = make_raw_hierarchical_env(env_config)

    # Blue workers get role-augmented observations.
    worker_obs_space = raw_env.observation_spaces["agent_0"]
    worker_act_space = raw_env.action_spaces["agent_0"]

    # Red workers keep the base observation shape.
    red_obs_space = raw_env.observation_spaces["agent_3"]
    red_act_space = raw_env.action_spaces["agent_3"]

    raw_env.close()

    policies = {
        "worker_policy": (
            None,
            worker_obs_space,
            worker_act_space,
            {
                "model": {
                    "custom_model": "shared_encoder_role_heads",
                    "custom_model_config": {
                        "hidden_dim": 256,
                        "head_dim": 128,
                    },
                }
            },
        ),
        "opponent_policy": (
            None,
            red_obs_space,
            red_act_space,
            {
                "model": {
                    "custom_model": "shared_encoder_role_heads",
                    "custom_model_config": {
                        "hidden_dim": 256,
                        "head_dim": 128,
                    },
                }
            },
        ),
        "random_policy": (
            RandPolicy,
            red_obs_space,
            red_act_space,
            {"no_checkpoint": True},
        ),
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

    env = env_creator(env_config)

    total_blue_wins = 0
    blue_agents = ["agent_0", "agent_1", "agent_2"]

    aggregate_role_counts = {
        aid: {ATTACK: 0, DEFEND: 0, INTERCEPT: 0}
        for aid in blue_agents
    }

    aggregate_side_steps_by_agent = {
        aid: {"home_side": 0, "enemy_side": 0, "unknown_side": 0}
        for aid in blue_agents
    }

    aggregate_side_steps_by_role = {
        ATTACK: {"home_side": 0, "enemy_side": 0, "unknown_side": 0},
        DEFEND: {"home_side": 0, "enemy_side": 0, "unknown_side": 0},
        INTERCEPT: {"home_side": 0, "enemy_side": 0, "unknown_side": 0},
    }

    aggregate_tags_by_role = {ATTACK: 0, DEFEND: 0, INTERCEPT: 0}
    aggregate_flag_carry_steps_by_role = {ATTACK: 0, DEFEND: 0, INTERCEPT: 0}
    aggregate_role_switches = {aid: 0 for aid in blue_agents}

    aggregate_reasons = {}
    aggregate_configs = {}

    for ep in range(args.episodes):
        stats = evaluate_episode(algo, env, render=args.render)

        blue_caps = stats["blue_captures"]
        red_caps = stats["red_captures"]
        if blue_caps is not None and red_caps is not None and blue_caps > red_caps:
            total_blue_wins += 1

        for aid in blue_agents:
            for role_id in [ATTACK, DEFEND, INTERCEPT]:
                aggregate_role_counts[aid][role_id] += stats["role_counts"][aid][role_id]

            for bucket in ["home_side", "enemy_side", "unknown_side"]:
                aggregate_side_steps_by_agent[aid][bucket] += stats["side_steps_by_agent"][aid][bucket]

        aggregate_role_switches[aid] += stats["role_switches"][aid]

    for role_id in [ATTACK, DEFEND, INTERCEPT]:
        for bucket in ["home_side", "enemy_side", "unknown_side"]:
            aggregate_side_steps_by_role[role_id][bucket] += stats["side_steps_by_role"][role_id][bucket]

        aggregate_tags_by_role[role_id] += stats["tags_by_role"][role_id]
        aggregate_flag_carry_steps_by_role[role_id] += stats["flag_carry_steps_by_role"][role_id]

    for reason, count in stats["commander_reasons"].items():
        aggregate_reasons[reason] = aggregate_reasons.get(reason, 0) + count

    for config_idx, count in stats["commander_configs"].items():
        aggregate_configs[config_idx] = aggregate_configs.get(config_idx, 0) + count

    print(
        f"[episode {ep + 1}/{args.episodes}] "
        f"blue_caps={blue_caps} red_caps={red_caps} "
        f"steps={stats['steps']} "
        f"role_switches={stats['role_switches']}"
    )

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Blue win count: {total_blue_wins} / {args.episodes}")

    print("\nRole usage by blue worker:")
    for aid in blue_agents:
        total = sum(aggregate_role_counts[aid].values())
        print(f"  {aid}:")
        for role_id in [ATTACK, DEFEND, INTERCEPT]:
            pct = 100.0 * aggregate_role_counts[aid][role_id] / max(total, 1)
            print(f"    {ROLE_NAMES[role_id]} = {aggregate_role_counts[aid][role_id]} ({pct:.1f}%)")

    print("\nTerritory occupancy by blue worker:")
    for aid in blue_agents:
        total = sum(aggregate_side_steps_by_agent[aid].values())
        print(f"  {aid}:")
        for bucket in ["home_side", "enemy_side", "unknown_side"]:
            pct = 100.0 * aggregate_side_steps_by_agent[aid][bucket] / max(total, 1)
            print(f"    {bucket} = {aggregate_side_steps_by_agent[aid][bucket]} ({pct:.1f}%)")

    print("\nTerritory occupancy by current role:")
    for role_id in [ATTACK, DEFEND, INTERCEPT]:
        total = sum(aggregate_side_steps_by_role[role_id].values())
        print(f"  {ROLE_NAMES[role_id]}:")
        for bucket in ["home_side", "enemy_side", "unknown_side"]:
            pct = 100.0 * aggregate_side_steps_by_role[role_id][bucket] / max(total, 1)
            print(f"    {bucket} = {aggregate_side_steps_by_role[role_id][bucket]} ({pct:.1f}%)")

    print("\nTags made by current role:")
    for role_id in [ATTACK, DEFEND, INTERCEPT]:
        print(f"  {ROLE_NAMES[role_id]} = {aggregate_tags_by_role[role_id]}")

    print("\nFlag-carry steps by current role:")
    for role_id in [ATTACK, DEFEND, INTERCEPT]:
        print(f"  {ROLE_NAMES[role_id]} = {aggregate_flag_carry_steps_by_role[role_id]}")

    print("\nRole switches by blue worker:")
    for aid in blue_agents:
        print(f"  {aid} = {aggregate_role_switches[aid]}")

    print("\nCommander reasons:")
    for reason, count in sorted(aggregate_reasons.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {reason} = {count}")

    print("\nCommander config usage:")
    for config_idx, count in sorted(aggregate_configs.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  config {config_idx} = {count}")

    env.close()
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()