import argparse
import numpy as np
import gym
import utils
from custom_callbacks import CustomCallbacks
import wandb
import os
import datetime as dt
import random

from stable_baselines3 import SAC, PPO
from stable_baselines3.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines3.ppo.policies import MlpPolicy as PPOMlpPolicy
from stable_baselines3.common.vec_env import (DummyVecEnv, VecCheckNan, VecNormalize)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import gym_socialgame.envs.utils as env_utils
from gym_socialgame.envs.socialgame_env import (SocialGameEnvRLLib, SocialGameMetaEnv)

import gym_microgrid.envs.utils as env_utils
from gym_microgrid.envs.microgrid_env import MicrogridEnvRLLib

import ray
import ray.rllib.agents.ppo as ray_ppo
import ray.rllib.agents.maml as ray_maml
from ray import tune
from ray.tune.integration.wandb import (wandb_mixin, WandbLoggerCallback)
from ray.tune.logger import (DEFAULT_LOGGERS, pretty_print, UnifiedLogger)
from ray.tune.integration.wandb import WandbLogger

from ray.rllib.contrib.bandits.agents.lin_ucb import UCB_CONFIG
from ray.rllib.contrib.bandits.agents.lin_ucb import LinUCBTrainer
from ordinal_action_overwrite2 import (OrdinalStochasticSampler, OrdinalStochasticSamplerGPU)

import IPython

def train(agent, num_steps, tb_log_name, args = None, library="sb3"):
    """
    Purpose: Train agent in env, and then call eval function to evaluate policy
    """
    # Train agent
    if library=="sb3":
        agent.learn(
            total_timesteps=num_steps,
            log_interval=10,
            tb_log_name=tb_log_name
        )

    elif library=="tune":

        ray.init()

        if args.algo=="ppo":
            config = ray_ppo.DEFAULT_CONFIG.copy()
            config["framework"] = "torch"
            config["env"] = SocialGameEnvRLLib
            config["callbacks"] = CustomCallbacks
            config["num_gpus"] = 0
            config["num_workers"] = 4
            config["env_config"] = vars(args)

            config["lr"] = tune.uniform(0.003, 5e-6)
            config["train_batch_size"] = tune.choice([4, 64, 256])
            config["sgd_minibatch_size"] = tune.sample_from(lambda spec: random.choice([x for x in [2, 4, 16, 32] if 2*x <= spec.config.train_batch_size]))
            config["clip_param"] = tune.choice([0.1, 0.2, 0.3])

            def stopper(_, result):
                return result["timesteps_total"] > num_steps

            exp_dict = {
                    'name': args.exp_name,
                    'run_or_experiment': ray_ppo.PPOTrainer,
                    'config': config,
                    'num_samples': 12,
                    'stop': stopper,
                    'local_dir': os.path.abspath(args.base_log_dir)
                }

            analysis = tune.run(**exp_dict)
            analysis.results_df.to_csv("POC results.csv")

        elif args.algo=="uc_bandit":
            config = UCB_CONFIG
            config["env"] = SocialGameEnvRLLib
            config["env_config"] = vars(args)

            analysis = tune.run(
                "contrib/LinUCB",
                config = config,
                stop = {"training_iteration":20},
                num_samples = 5
            )


    elif library=="rllib":

        ray.init(local_mode=True)

        if args.algo=="ppo":
            print("Entering ppo config setting")
            train_batch_size = 256
            config = ray_ppo.DEFAULT_CONFIG.copy()
            config["framework"] = "torch"
            config["train_batch_size"] = train_batch_size
            config["sgd_minibatch_size"] = 16
            config["lr"] = 0.0002
            config["clip_param"] = 0.3
            config["num_gpus"] =  1
            if not args.gpu:
                config["num_gpus"] = 0
            config["num_workers"] = 1
            if args.action_space == "ordinal":
                if not args.gpu:
                    config["exploration_config"] = {
                        "type":OrdinalStochasticSampler
                    }
                else: 
                    config["exploration_config"] = {
                        "type":OrdinalStochasticSamplerGPU
                    }
            print("CONFIG")
            print(config["exploration_config"])
            if args.gym_env == "socialgame":
                config["env"] = SocialGameEnvRLLib
                obs_dim = 10 * np.sum([args.energy_in_state, args.price_in_state])
            elif args.gym_env == "microgrid":
                config["env"] = MicrogridEnvRLLib
                obs_dim = 72 * np.sum([args.energy_in_state, args.price_in_state])

            out_path = os.path.join(args.log_path, "bulk_data.h5")
            callbacks = CustomCallbacks(log_path=out_path, save_interval=args.bulk_log_interval, obs_dim=obs_dim)
            config["callbacks"] = lambda: callbacks
            config["env_config"] = vars(args)
            logger_creator = utils.custom_logger_creator(args.log_path)

            # question for Tarang, what are callbacks for? 

            callbacks.save()
            if args.wandb:
                wandb.save(out_path)

            if args.gym_env == "socialgame":
                updated_agent = ray_ppo.PPOTrainer(config=config, env=SocialGameEnvRLLib, logger_creator=logger_creator)
            elif args.gym_env == "microgrid":
                updated_agent = ray_ppo.PPOTrainer(config=config, env=MicrogridEnvRLLib, logger_creator=logger_creator)

            to_log = ["episode_reward_mean"]
            timesteps_total = 0
            while timesteps_total < num_steps:
                result = updated_agent.train()
                timesteps_total = result["timesteps_total"]
                log = {name: result[name] for name in to_log}
                if args.wandb:
                    wandb.log(log)
                else:
                    print(log)

            callbacks.save()

        elif args.algo=="maml":
            config = ray_maml.DEFAULT_CONFIG.copy()
            config["num_gpus"] = 1
            config["train_batch_size"] = train_batch_size
            config["num_workers"] = 4
            config["env"] = SocialGameMetaEnv
            config["env_config"] = vars(args)
            config["normalize_actions"] = True
            config["log_save_interval"] = 10
            updated_agent = ray_maml.MAMLTrainer(config=config, env = SocialGameMetaEnv)
            to_log = ["episode_reward_mean", "episode_reward_mean_adapt_1", "adaptation_delta"]

            for i in range(num_steps):
                result = updated_agent.train()
                log = {name: result[name] for name in to_log}
                if args.wandb:
                    wandb.log(log)
                    wandb.log({"total_loss": result["info"]["learner"]["default_policy"]["total_loss"]})
                else:
                    print(log)

        # Trying bandit without tuning 
        elif args.algo=="uc_bandit":
            config = UCB_CONFIG
            config["env"] = SocialGameEnvRLLib
            config["env_config"] = vars(args)
            updated_agent = LinUCBTrainer(
                config = config
            )
    
            for i in range(num_steps):
                result = updated_agent.train()
                mean_reward = np.mean(result["hist_stats"]["episode_reward"])
                if args.wandb:
                    wandb.log({"episode_reward_mean" : mean_reward})
                else:
                    print(mean_reward)



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

    if args.library=="sb3":
        if args.algo == "sac":
            return SAC(
                policy=SACMlpPolicy,
                env=env,
                batch_size=args.batch_size,
                learning_starts=30,
                verbose=0,
                tensorboard_log=args.log_path,
                learning_rate=args.learning_rate)

        elif args.algo == "ppo":
            return PPO(
                    policy=PPOMlpPolicy,
                    env=env,
                    verbose=2,
                    n_steps=128,
                    tensorboard_log=args.log_path)

    elif args.library=="rllib" or args.library=="tune":

        if args.algo == "ppo":
            trainer = ray_ppo.PPOTrainer
            return trainer

        elif args.algo == "maml":
            trainer = ray_maml.MAMLTrainer
            return trainer

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

def get_environment(args):
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
        if args.action_space == "c_norm":
            args.action_space_string = "continuous_normalized"
        else:
            args.action_space_string = "continuous"
    # For algos (e.g. ppo) which can handle discrete or continuous case
    # Note: PPO typically uses normalized environment (#TODO)
    else:
        
        if args.action_space =="c" or args.action_space =="continuous":
            args.action_space_string = "continuous"
        elif args.action_space=="d" or args.action_space =="multidiscrete":
            args.action_space_string="multidiscrete"
        elif args.action_space=="ordinal":
            args.action_space_string ="multidiscrete"
        else:
            print("Wrong Action Space string")
            raise AssertionError 

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

    if args.gym_env == "socialgame":
        gym_env = gym.make(
            "gym_socialgame:socialgame{}".format(env_id),
            action_space_string=args.action_space_string,
            response_type_string=args.response_type_string,
            one_day=args.one_day,
            number_of_participants=args.number_of_participants,
            price_in_state = args.price_in_state,
            energy_in_state=args.energy_in_state,
            pricing_type=args.pricing_type,
            reward_function=reward_function,
            bin_observation_space = args.bin_observation_space,
            manual_tou_magnitude=args.manual_tou_magnitude,
            smirl_weight=args.smirl_weight
        )
    elif args.gym_env == "microgrid":
        gym_env = gym.make(
            "gym_microgrid:microgrid{}".format(env_id),
            action_space_string=action_space_string,
            response_type_string=args.response_type_string,
            one_day=args.one_day,
            number_of_participants=args.number_of_participants,
            energy_in_state=args.energy_in_state,
            pricing_type=args.pricing_type,
            reward_function=reward_function,
            manual_tou_magnitude=args.manual_tou_magnitude,
            smirl_weight=args.smirl_weight # NOTE: Complex Batt PV and two price state default values used
        )

    # Check to make sure any new changes to environment follow OpenAI Gym API
    check_env(gym_env)
    return gym_env

def vectorize_environment(env, args, include_non_vec_env=False):

    # temp_step_fnc = socialgame_env.step

    if args.library=="sb3":

        # Using env_fn so we can create vectorized environment for stable baselines.
        env_fn = lambda: Monitor(env)
        venv = DummyVecEnv([env_fn])
        env = VecNormalize(venv)

        # env.step = temp_step_fnc
        if not include_non_vec_env:
            return env
        else:
            return env, socialgame_env

    elif args.library=="rllib" or args.library == "tune":
        #RL lib auto-vectorizes them, sweet

        if include_non_vec_env==False:
            return env
        else:
            return env, env

    else:
        print("Wrong library!")
        raise AssertionError


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
        "--gym_env", 
        help="Which Gym Environment you wihs to use",
        type=str,
        choices=["socialgame", "microgrid"],
        default="socialgame"
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        type=str,
        default="sac",
        choices=["sac", "ppo", "maml", "uc_bandit"]
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
        choices=["c", "c_norm", "d", "fourier", "ordinal"],
    )
    parser.add_argument(
        "--action_space_string",
        help="action space string expanded (use this instead of action_space for RLLib)",
        default="continuous",
        )
    parser.add_argument(
        "--response_type_string",
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
        "--number_of_participants",
        help="Number of players ([1, 20]) in social game",
        type=int,
        default=10,
        choices=[i for i in range(1, 21)],
    )
    parser.add_argument(
        "--energy_in_state",
        help="Whether to include energy in state (default = F)",
        type=str,
        default="F",
        choices=["T", "F"],
    )
    parser.add_argument(
        "--price_in_state",
        help="Whether to include price in state (default = F)",
        type=str,
        default="T",
        choices=["T", "F"],
    )
    parser.add_argument(
        "--exp_name",
        help="experiment_name",
        type=str,
        default="experiment"
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
        default="log_cost_regularized",
        choices=["scaled_cost_distance", "log_cost_regularized", "log_cost", "scd", "lcr", "lc", "market_solving", "profit_maximizing"],
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate of the the agent",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--pb_scenario",
        type = int,
        default = 1,
        help = "1 is for repeated PV, 2 for small, 3 or medium scenario, 4 no batt, 5 no solar, 6 nothing",
        choices = [ 1, 2, 3, 4, 5, 6 ]),
    parser.add_argument(
        "--two_price_state",
        help="Whether to include buy and sell price in state (default = F)",
        type=str,
        default="F",
        choices=["T", "F"],
    )
    parser.add_argument(
        "--bin_observation_space",
        help = "Bin the observations",
        type = str,
        default = "F",
        choices = ["T", "F"]
   )
    parser.add_argument(
        "--library",
        help = "What RL Library backend is in use",
        type = str,
        default = "sb3",
        choices = ["sb3", "rllib", "tune"]
    )
    parser.add_argument(
        "--smirl_weight",
        help="Whether to run with SMiRL. When using SMiRL you must specify a weight.",
        type = float,
        default=None,
    )
    parser.add_argument(
        "--circ_buffer_size",
        help="Size of circular smirl buffer to use. Will use an unlimited size buffer in None",
        type = float,
        default=None,
    )
    parser.add_argument(
        "--bulk_log_interval",
        help="Interval at which to save bulk log information",
        type=int,
        default=100
    )

    parser.add_argument(
        "--gpu",
        help="whether we are requesting GPUs"
        type=str,
        default="T"
    )

    args = parser.parse_args()

    args.log_path = os.path.join(os.path.abspath(args.base_log_dir), "{}_{}".format(args.exp_name, str(dt.datetime.today())))

    os.makedirs(args.log_path, exist_ok=True)

    return args


def main():

    # Get args
    args = parse_args()

    # Print args for reference
    args_convert_bool(args)

    if args.wandb:
        wandb.init(project="energy-demand-response-game", entity="social-game-rl")
        wandb.tensorboard.patch(root_logdir=args.log_path) # patching the logdir directly seems to work
        wandb.config.update(args)

    # Create environments

    env = get_environment(
        args,
    )

    # if you need to modify to bring in non vectorized env, you need to modify function returns
    vec_env = vectorize_environment(
        env,
        args,
        )

    print("Got vectorized environment, getting agent")

    # Create Agent
    model = get_agent(vec_env, args, non_vec_env=None)
    print("Got agent")

    # Train algo, (logging through Tensorboard)
    print("Beginning Testing!")
    r_real = train(
        agent = model,
        num_steps = args.num_steps,
        tb_log_name=args.exp_name,
        args = args,
        library=args.library
    )

    print("Training Completed! View TensorBoard logs at " + args.log_path)

    # Print evaluation of policy
    print("Beginning Evaluation")

    print(
        "If there was no planning model involved, remember that the output will be in the log dir"
    )


if __name__ == "__main__":
    main()
