import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

# Add parent directory for optional custom rewards import.
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
import pyquaticus.utils.rewards as rew


LOGGER = logging.getLogger("pyquaticus_train")


try:
    from custom_rewards import combined_reward, tactical_reward, attacker_reward, support_reward, defender_reward
    CUSTOM_REWARDS_AVAILABLE = True
except ImportError:
    CUSTOM_REWARDS_AVAILABLE = False
    combined_reward = None
    tactical_reward = None
    attacker_reward = None
    support_reward = None
    defender_reward = None


class RandomPolicy(Policy):
    """Simple frozen random opponent policy."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        return None

    def learn_on_batch(self, samples):
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PyQuaticus 3v3 PPO agents with stable defaults."
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--opponents",
        choices=["random", "selfplay"],
        default="random",
        help="Opponent policy setup",
    )
    parser.add_argument(
        "--reward",
        choices=["basic", "combined", "tactical", "roles"],
        default="combined",
        help="Reward function selection",
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Training iterations"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./ray_test",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="pyquaticus_optimized_3v3",
        help="Registered Ray environment name",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=1,
        help="Number of Ray env runners",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to allocate",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4000,
        help="PPO train batch size",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=512,
        help="PPO minibatch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed",
    )
    return parser.parse_args()


def build_reward_config(reward_name: str) -> Dict[str, object]:
    if reward_name == "basic":
        selected_reward = rew.caps_and_grabs
        LOGGER.info("Using basic reward function")
        return {
            "agent_0": selected_reward,
            "agent_1": selected_reward,
            "agent_2": selected_reward,
            "agent_3": None,
            "agent_4": None,
            "agent_5": None,
        }
    elif reward_name == "combined" and CUSTOM_REWARDS_AVAILABLE:
        selected_reward = combined_reward
        LOGGER.info("Using combined reward function")
        return {
            "agent_0": selected_reward,
            "agent_1": selected_reward,
            "agent_2": selected_reward,
            "agent_3": None,
            "agent_4": None,
            "agent_5": None,
        }
    elif reward_name == "tactical" and CUSTOM_REWARDS_AVAILABLE:
        selected_reward = tactical_reward
        LOGGER.info("Using tactical reward function")
        return {
            "agent_0": selected_reward,
            "agent_1": selected_reward,
            "agent_2": selected_reward,
            "agent_3": None,
            "agent_4": None,
            "agent_5": None,
        }
    elif reward_name == "roles" and CUSTOM_REWARDS_AVAILABLE:
        LOGGER.info("Using role-based milestone rewards: attacker/support/defender")
        return {
            "agent_0": attacker_reward,   # Primary flag chaser
            "agent_1": support_reward,    # Screens and assists
            "agent_2": defender_reward,   # Protects own flag
            "agent_3": None,
            "agent_4": None,
            "agent_5": None,
        }
    else:
        selected_reward = rew.caps_and_grabs
        if reward_name != "basic":
            LOGGER.warning(
                "Custom reward '%s' unavailable; falling back to basic rewards.",
                reward_name,
            )
        return {
            "agent_0": selected_reward,
            "agent_1": selected_reward,
            "agent_2": selected_reward,
            "agent_3": None,
            "agent_4": None,
            "agent_5": None,
        }


def build_base_env_config(render: bool, reward_config: Dict[str, object]) -> Dict[str, object]:
    config_dict = config_dict_std.copy()
    config_dict.update(
        {
            "sim_speedup_factor": 4,
            "max_score": 3,
            "max_time": 240,
            "tagging_cooldown": 60,
            "tag_on_oob": True,
        }
    )

    return {
        "config_dict": config_dict,
        "render_mode": "human" if render else None,
        "reward_config": reward_config,
        "team_size": 3,
    }


def make_env_creator(base_env_kwargs: Dict[str, object]):
    def env_creator(_config):
        return pyquaticus_v0.PyQuaticusEnv(**base_env_kwargs)

    return env_creator


def build_policy_mapping(opponents: str):
    """Map agents to role-based policies: attacker, support, defender"""
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 'agent_0':
            return "attacker-policy"  # Primary flag chaser
        if agent_id == 'agent_1':
            return "support-policy"   # Screens/trails attacker
        if agent_id == 'agent_2':
            return "defender-policy"  # Protects own flag
        # Red agents use random policy
        return "random-policy"

    return policy_mapping_fn


def build_multiagent_config(opponents: str, policy_mapping_fn, obs_space, act_space):
    """3 role-based policies: attacker, support, defender + random for opponents"""
    policies = {
        "attacker-policy": (None, obs_space, act_space, {"model": {"fcnet_hiddens": [256, 256, 128], "fcnet_activation": "relu"}}),
        "support-policy": (None, obs_space, act_space, {"model": {"fcnet_hiddens": [256, 256, 128], "fcnet_activation": "relu"}}),
        "defender-policy": (None, obs_space, act_space, {"model": {"fcnet_hiddens": [256, 256, 128], "fcnet_activation": "relu"}}),
        "random-policy": (RandomPolicy, obs_space, act_space, {"no_checkpoint": True}),
    }
    
    if opponents == "selfplay":
        policies["red-policy"] = (None, obs_space, act_space, {"model": {"fcnet_hiddens": [256, 256, 128], "fcnet_activation": "relu"}})
        policies_to_train = ["attacker-policy", "support-policy", "defender-policy", "red-policy"]
    else:
        policies_to_train = ["attacker-policy", "support-policy", "defender-policy"]
    
    return policies, policies_to_train


def build_algorithm(args: argparse.Namespace):
    reward_config = build_reward_config(args.reward)
    base_env_kwargs = build_base_env_config(args.render, reward_config)
    env_creator = make_env_creator(base_env_kwargs)

    temp_env = ParallelPettingZooWrapper(env_creator({}))
    obs_space = temp_env.observation_space["agent_0"]
    act_space = temp_env.action_space["agent_0"]
    temp_env.close()

    register_env(args.env_name, lambda config: ParallelPettingZooWrapper(env_creator(config)))

    policy_mapping_fn = build_policy_mapping(args.opponents)
    policies, policies_to_train = build_multiagent_config(
        args.opponents, policy_mapping_fn, obs_space, act_space
    )

    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=args.env_name)
        .env_runners(
            num_env_runners=max(0, args.num_env_runners),
            num_cpus_per_env_runner=1,
        )
        .resources(num_gpus=args.num_gpus)
        .framework("torch")
        .debugging(seed=args.seed)
        .training(
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=10,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            grad_clip=0.5,
            vf_clip_param=10.0,
            entropy_coeff=0.03,
            model={
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "relu",
            },
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
    )

    algo = ppo_config.build_algo()
    return algo, policies_to_train


def format_metric(result: dict, key_path: Tuple[str, ...], default="N/A"):
    current = result
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _first_present(*values):
    for value in values:
        if value is None:
            continue
        if value == "N/A":
            continue
        return value
    return None


def extract_reward_mean(result: dict, policies_to_train: list):
    episode_reward = _first_present(
        result.get("episode_reward_mean"),
        format_metric(result, ("env_runners", "episode_reward_mean"), None),
        format_metric(result, ("sampler_results", "episode_reward_mean"), None),
        format_metric(result, ("evaluation", "episode_reward_mean"), None),
    )
    if episode_reward is not None:
        return episode_reward

    policy_reward = _first_present(
        result.get("policy_reward_mean"),
        format_metric(result, ("env_runners", "policy_reward_mean"), None),
        format_metric(result, ("sampler_results", "policy_reward_mean"), None),
    )

    if isinstance(policy_reward, dict):
        trained = [policy_reward.get(pid) for pid in policies_to_train if pid in policy_reward]
        trained = [v for v in trained if isinstance(v, (int, float))]
        if trained:
            return sum(trained) / len(trained)
        return policy_reward

    if isinstance(policy_reward, (int, float)):
        return policy_reward

    return "N/A"


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    checkpoint_root = Path(args.checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = checkpoint_root / f"run_{run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ray.init(ignore_reinit_error=True, include_dashboard=False)  # Skip Ray init to avoid Windows issues

    LOGGER.info("Starting training")
    LOGGER.info("Opponents: %s", args.opponents)
    LOGGER.info("Reward: %s", args.reward)
    LOGGER.info("Iterations: %d", args.iterations)

    algo = None
    try:
        algo, policies_to_train = build_algorithm(args)
        LOGGER.info("Training policies: %s", policies_to_train)

        dumped_result = False

        for iteration in range(1, args.iterations + 1):
            result = algo.train()

            reward_mean = extract_reward_mean(result, policies_to_train)
            
            env_steps = format_metric(result, ("num_env_steps_sampled_lifetime",))
            learner_info = result.get("info", {})

            if reward_mean == "N/A" and not dumped_result:
                dumped_result = True
                print("DEBUG - Full train() result dict (first time reward_mean is N/A):")
                print(result)

            LOGGER.info(
                "Iter %d/%d | reward_mean=%s | env_steps=%s",
                iteration,
                args.iterations,
                reward_mean,
                env_steps,
            )

            if iteration % args.checkpoint_every == 0:
                save_dir = checkpoint_dir / f"optimized_iter_{iteration}"
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_result = algo.save(str(save_dir))
                LOGGER.info("Checkpoint saved to %s", checkpoint_result.checkpoint.path)

            # Optional concise learner diagnostics when available.
            if learner_info:
                LOGGER.debug("Learner info keys: %s", list(learner_info.keys()))

        final_dir = checkpoint_dir / "optimized_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_result = algo.save(str(final_dir))
        LOGGER.info("Final checkpoint saved to %s", checkpoint_result.checkpoint.path)

    finally:
        # if algo is not None:
        #     algo.stop()
        # ray.shutdown()  # Skip Ray shutdown since we didn't initialize Ray
        LOGGER.info("Training finished")


if __name__ == "__main__":
    main()
