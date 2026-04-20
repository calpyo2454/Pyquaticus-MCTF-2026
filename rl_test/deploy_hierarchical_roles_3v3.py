import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper

from hierarchical_framework.hierarchical_team_wrapper import HierarchicalTeamWrapper


class RandPolicy(Policy):
    """
    Random baseline policy used only so the restored PPO algorithm has the same
    multi-agent policy layout as training/evaluation.
    """

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
    """
    Build the raw hierarchical env exactly like eval/training, except render_mode
    can be set independently for deployment.
    """
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
    )


def env_creator(env_config):
    """
    RLlib-compatible environment creator.
    """
    raw_env = make_raw_hierarchical_env(env_config)
    return ParallelPettingZooWrapper(raw_env)


def policy_mapping_fn(agent_id, episode, **kwargs):
    """
    Match the current hierarchical shared-worker layout.
    """
    if agent_id in ["agent_0", "agent_1", "agent_2"]:
        return "worker_policy"
    return "random_policy"


def build_env_config(role_period: int, render_mode: str):
    """
    Match evaluation/training closely, but use a visible render mode for deployment.
    """
    config_dict = config_dict_std.copy()
    config_dict["sim_speedup_factor"] = 1
    config_dict["max_score"] = 3
    config_dict["max_time"] = 240
    config_dict["tagging_cooldown"] = 60
    config_dict["tag_on_oob"] = True

    return {
        "config_dict": config_dict,
        "render_mode": render_mode,
        "team_size": 3,
        "role_period": role_period,
        "shape_blue_worker_rewards": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy trained hierarchical 3v3 PPO worker in rendered GUI")
    parser.add_argument("checkpoint", help="Path to PPO algorithm checkpoint directory")
    parser.add_argument("--role-period", type=int, default=5, help="Commander role refresh period")
    parser.add_argument("--max-steps", type=int, default=2500, help="Maximum total steps before exit")
    parser.add_argument("--print-every", type=int, default=50, help="How often to print blue-role debug info")
    parser.add_argument("--env-name", type=str, default="pyquaticus_hierarchical_roles_3v3", help="Registered env name")
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    # Register the exact env name used by the hierarchical setup.
    # This also avoids mismatch if anything in restore references the original env string.
    render_env_config = build_env_config(role_period=args.role_period, render_mode="human")
    register_env(args.env_name, lambda config: env_creator(render_env_config))

    # Build a non-render temp env to infer spaces cleanly without flashing a GUI.
    temp_env_config = build_env_config(role_period=args.role_period, render_mode=None)
    temp_env = env_creator(temp_env_config)
    worker_obs_space = temp_env.observation_space["agent_0"]
    worker_act_space = temp_env.action_space["agent_0"]
    red_obs_space = temp_env.observation_space["agent_3"]
    red_act_space = temp_env.action_space["agent_3"]
    temp_env.close()

    policies = {
        "worker_policy": (None, worker_obs_space, worker_act_space, {}),
        "random_policy": (RandPolicy, red_obs_space, red_act_space, {"no_checkpoint": True}),
    }

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

    env = env_creator(render_env_config)
    obs, _ = env.reset()
    step = 0

    print("Starting deployment...")

    while True:
        actions = {}

        for agent_id in obs.keys():
            if agent_id in ["agent_0", "agent_1", "agent_2"]:
                actions[agent_id] = algo.compute_single_action(
                    obs[agent_id],
                    policy_id="worker_policy",
                )
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

        if step >= args.max_steps:
            break

        if any(term.values()) or any(trunc.values()):
            obs, _ = env.reset()
            print(f"Episode ended at total step {step}")

    env.close()
    algo.stop()
    ray.shutdown()
    print("Deployment completed.")