import argparse
import numpy as np
import gym
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import (DummyVecEnv, VecCheckNan, VecNormalize)

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env

import gym_socialgame.envs.utils as env_utils

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorboard_logger import (  # pylint: disable=import-error, no-name-in-module
    configure as tb_configure,
)
from tensorboard_logger import (  # pylint: disable=import-error, no-name-in-module
    log_value as tb_log_value,
)

import utils
import wandb

import os

import datetime as dt
import wandb

def train(agent, num_steps, tb_log_name):
    """
    Purpose: Train agent in env, and then call eval function to evaluate policy
    """
    # Train agent

    agent.learn(
        total_timesteps=num_steps,
        log_interval=10,
        tb_log_name=tb_log_name
    )


def eval_policy(model, env, num_eval_episodes: int, list_reward_per_episode=False):
    """
    Purpose: Evaluate policy on environment over num_eval_episodes and print results

    Args:
        Model: Stable baselines model
        Env: Gym environment for evaluation
        num_eval_episodes: (Int) number of episodes to evaluate policy
        list_reward_per_episode: (Boolean) Whether or not to return a list containing rewards per episode (instead of mean reward over all episodes)

    """
    mean_reward, std_reward = evaluate_policy(
        model, env, num_eval_episodes, return_episode_rewards=list_reward_per_episode
    )

    print("Test Results: ")
    print("Mean Reward: {:.3f}".format(mean_reward))
    print("Std Reward: {:.3f}".format(std_reward))


def get_agent(env, args, non_vec_env=None):
    """
    Purpose: Import algo, policy and create agent

    Returns: Agent

    Exceptions: Raises exception if args.algo unknown (not needed b/c we filter in the parser, but I added it for modularity)
    """
    if args.algo == "sac":
        return SAC(
            policy=MlpPolicy,
            env=env,
            batch_size=args.batch_size,
            learning_starts=30,
            verbose=0,
            tensorboard_log=args.rl_log_path,
            learning_rate=args.learning_rate
        )

    elif args.algo == "ppo":
        from stable_baselines import PPO2

        if args.policy_type == "mlp":
            from stable_baselines.common.policies import MlpPolicy as policy

        elif args.policy_type == "lstm":
            from stable_baselines.common.policies import MlpLstmPolicy as policy

        return PPO2(policy, env, verbose=0, tensorboard_log=args.rl_log_path)

    else:
        raise NotImplementedError("Algorithm {} not supported. :( ".format(args.algo))


def args_convert_bool(args):
    """
    Purpose: Convert args which are specified as strings (e.g. energy/price_in_state) into boolean to work with environment
    """
    if not isinstance(args.energy_in_state, (bool)):
        args.energy_in_state = utils.string2bool(args.energy_in_state)
    if not isinstance(args.price_in_state, (bool)):
        args.price_in_state = utils.string2bool(args.price_in_state)
    if not isinstance(args.test_planning_env, (bool)):
        args.test_planning_env = utils.string2bool(args.test_planning_env)
    if not isinstance(args.bin_observation_space, (bool)):
        args.bin_observation_space = utils.string2bool(args.bin_observation_space)

def get_environment(args, include_non_vec_env=False):
    """
    Purpose: Create environment for algorithm given by args. algo

    Args:
        args

    Returns: Environment with action space compatible with algo
    """
    # Convert string args (which are supposed to be bool) into actual boolean values
    args_convert_bool(args)
    
    # SAC only works in continuous environment
    if args.algo == "sac":
        if args.action_space == "fourier":
            action_space_string = "fourier"
        elif args.action_space == "c_norm":
            action_space_string = "continuous_normalized"
        else:
            action_space_string = "continuous"
    # For algos (e.g. ppo) which can handle discrete or continuous case
    # Note: PPO typically uses normalized environment (#TODO)
    else:
        convert_action_space_str = (
            lambda s: "continuous" if s == "c" else "multidiscrete"
        )
        action_space_string = convert_action_space_str(args.action_space)

    if args.env_id == "hourly":
        env_id = "_hourly-v0"
    elif args.env_id == "monthly":
        env_id = "_monthly-v0"
    else:
        env_id = "-v0"

    if args.reward_function == "lcr":
        reward_function = "log_cost_regularized"
    elif args.reward_function == "scd":
        reward_function = "scaled_cost_distance"
    elif args.reward_function == "lc":
        reward_function = "log_cost"
    else:
        reward_function = args.reward_function

    socialgame_env = gym.make(
        "gym_socialgame:socialgame{}".format(env_id),
        action_space_string=action_space_string,
        response_type_string=args.response,
        one_day=args.one_day,
        number_of_participants=args.num_players,
        price_in_state = args.price_in_state,
        energy_in_state=args.energy_in_state,
        pricing_type=args.pricing_type,
        reward_function=reward_function,
        bin_observation_space = args.bin_observation_space,
        manual_tou_magnitude=args.manual_tou_magnitude
        )


    # Check to make sure any new changes to environment follow OpenAI Gym API
    check_env(socialgame_env)

    # temp_step_fnc = socialgame_env.step

    # Using env_fn so we can create vectorized environment.
    env_fn = lambda: socialgame_env
    venv = DummyVecEnv([env_fn])
    env = VecNormalize(venv)

    # env.step = temp_step_fnc
    if not include_non_vec_env:
        return env
    else:
        return env, socialgame_env


def parse_args():
    """
    Purpose: Parse arguments to run script
    """

    parser = argparse.ArgumentParser(
        description="Arguments for running Stable Baseline RL Algorithms on SocialGameEnv"
    )
    parser.add_argument(
        "-w",
        "--wandb",
        help="Whether to run wandb",
        action="store_true"
    )
    parser.add_argument(
        "--env_id",
        help="Environment ID for Gym Environment",
        type=str,
        choices=["v0", "monthly"],
        default="v0",
    )
    parser.add_argument(
        "--algo", help="Stable Baselines Algorithm", type=str, choices=["sac", "ppo"]
    )
    parser.add_argument(
        "--base_log_dir",
        help="Base directory for tensorboard logs",
        type=str,
        default="./logs/"
    )

    parser.add_argument(
        "--batch_size",
        help="Batch Size for sampling from replay buffer",
        type=int,
        default=5,
        choices=[i for i in range(1, 30)],
    )
    parser.add_argument(
        "--num_steps",
        help="Number of timesteps to train algo",
        type=int,
        default=50000,
    )
    # Note: only some algos (e.g. PPO) can use LSTM Policy the feature below is for future testing
    parser.add_argument(
        "--policy_type",
        help="Type of Policy (e.g. MLP, LSTM) for algo",
        default="mlp",
        choices=["mlp", "lstm"],
    )
    parser.add_argument(
        "--action_space",
        help="Action Space for Algo (only used for algos that are compatable with both discrete & cont",
        default="c",
        choices=["c", "c_norm", "d", "fourier"],
    )
    parser.add_argument(
        "--response",
        help="Player response function (l = linear, t = threshold_exponential, s = sinusoidal",
        type=str,
        default="l",
        choices=["l", "t", "s"],
    )
    parser.add_argument(
        "--one_day",
        help="Specific Day of the year to Train on (default = 15, train on day 15)",
        type=int,
        default=15,
        choices=[i for i in range(365)],
    )
    parser.add_argument(
        "--manual_tou_magnitude",
        help="Magnitude of the TOU during hours 5,6,7. Sets price in normal hours to 0.103.",
        type=float,
        default=.4
    )
    parser.add_argument(
        "--num_players",
        help="Number of players ([1, 20]) in social game",
        type=int,
        default=10,
        choices=[i for i in range(1, 21)],
    )
    parser.add_argument(
        "--price_in_state",
        help = "Is price in the state",
        type = str,
        default = "T",
        choices = ["T", "F"]
    )
    parser.add_argument(
        "--energy_in_state",
        help="Whether to include energy in state (default = F)",
        type=str,
        default="F",
        choices=["T", "F"],
    )
    parser.add_argument(
        "--exp_name", 
        help="experiment_name", 
        type=str,
        default=str(dt.datetime.today())
    )
    parser.add_argument(
        "--planning_steps",
        help="How many planning iterations to partake in",
        type=int,
        default=0,
        choices=[i for i in range(0, 100)],
    )
    parser.add_argument(
        "--planning_model",
        help="Which planning model to use",
        type=str,
        default="Oracle",
        choices=["Oracle", "Baseline", "LSTM", "OLS"],
    )
    parser.add_argument(
        "--pricing_type",
        help="time of use or real time pricing",
        type=str,
        choices=["TOU", "RTP"],
        default="TOU",
    )
    parser.add_argument(
        "--test_planning_env",
        help="flag if you want to test vanilla planning",
        type=str,
        default='F',
        choices=['T', 'F'],
    )
    parser.add_argument(
        "--reward_function",
        help="reward function to test",
        type=str,
        default="lcr",
        choices=["scaled_cost_distance", "log_cost_regularized", "log_cost", "scd", "lcr", "lc"],
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate of the the agent",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--bin_observation_space",
        help = "Bin the observations",
        type = str,
        default = "F",
        choices = ["T", "F"]
    )


    args = parser.parse_args()

    args.log_path = os.path.join(args.base_log_dir, args.exp_name + "/")
    args.rl_log_path = os.path.join(args.log_path, "rl/")

    return args


def main():


    # Get args
    args = parse_args()

    # Print args for reference
    print(args)
    args_convert_bool(args)

    if args.wandb:
        wandb.init(project="energy-demand-response-game", entity="social-game-rl", sync_tensorboard=True)
        wandb.config.update(args)

    # Create environments

    if os.path.exists(args.log_path):
        print("Choose a new name for the experiment, log dir already exists")
        raise ValueError

    env, socialgame_env = get_environment(
        args, include_non_vec_env=True
    )
    print("Got environment, getting agent")

    # Create Agent
    model = get_agent(env, args, non_vec_env=socialgame_env)
    print("Got agent")

    # Train algo, (logging through Tensorboard)
    print("Beginning Testing!")
    r_real = train(
        model,
        args.num_steps,
        tb_log_name=args.exp_name
    )

    print("Training Completed! View TensorBoard logs at " + args.log_path)

    # Print evaluation of policy
    print("Beginning Evaluation")

    eval_env = get_environment(args)
    eval_policy(model, eval_env, num_eval_episodes=10)

    print(
        "If there was no planning model involved, remember that the output will be in the log dir"
    )


if __name__ == "__main__":
    main()
