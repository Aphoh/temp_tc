import argparse
import numpy as np
import gym
import utils
import wandb
import os
import datetime as dt

from stable_baselines3 import SAC, PPO
from stable_baselines3.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines3.ppo.policies import MlpPolicy as PPOMlpPolicy
from stable_baselines3.common.vec_env import (DummyVecEnv, VecCheckNan, VecNormalize)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import gym_socialgame.envs.utils as env_utils
from gym_socialgame.envs.socialgame_env import (SocialGameEnvRLLib, SocialGameMetaEnv)

import ray
import ray.rllib.agents.ppo as ray_ppo
import ray.rllib.agents.maml as ray_maml
from ray import tune
from ray.tune.integration.wandb import (wandb_mixin, WandbLoggerCallback)
from ray.tune.logger import (DEFAULT_LOGGERS, pretty_print)
from ray.tune.integration.wandb import WandbLogger

from ray.rllib.agents.maml.maml_tf_policy import MAMLTFPolicy
from ray.rllib.agents.maml.maml_torch_policy import MAMLTorchPolicy
from ray.rllib.agents.trainer_template import build_trainer

from pprint import pprint
import pdb

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

    elif library=="rllib":

        #ray.init()

        if args.algo=="ppo":
            
            to_log = ["episode_reward_mean"]
            for i in range(int(np.ceil(num_steps/train_batch_size))):
                result = agent.train()
                log = {name: result[name] for name in to_log}
                if args.wandb:
                    wandb.log(log)
                else:
                    print(log)

        elif args.algo=="maml":
            
            to_log = ["episode_reward_mean", "episode_reward_mean_adapt_5", "adaptation_delta"]

            for i in range(num_steps):
                result = agent.train()
                log = {name: result[name] for name in to_log}
                if args.wandb:
                    wandb.log(log)
                    wandb.log({"total_loss": result["info"]["learner"]["default_policy"]["total_loss"]})
                else:
                    print(log)
                if i % args.checkpoint_interval == 0:
                    agent.save_checkpoint("./maml_ckpts")

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


def maml_eval_fn(trainer, args):
    """
    Purpose: Evaluate policy on environment over num_eval_episodes and print results

    Args:
        Model: Stable baselines model
        Env: Gym environment for evaluation
        num_eval_episodes: (Int) number of episodes to evaluate policy
        list_reward_per_episode: (Boolean) Whether or not to return a list containing rewards per episode (instead of mean reward over all episodes)

    """
    # def dummy_validate(config):
    #     pass
    # MAMLValTrainer = build
    config = ray_maml.DEFAULT_CONFIG.copy()
    config["framework"] = "tf1"
    config["num_gpus"] = 1
    config["num_envs_per_worker"] = 2
    config["num_workers"] = args.maml_num_workers
    
    config["inner_adaptation_steps"] = 10
    config["env"] = SocialGameMetaEnv
    config["env_config"] = vars(args)
    config["env_config"]["mode"] = "test"
    config["normalize_actions"] = True
    config["clip_actions"] = True
    config["inner_lr"] = args.maml_inner_lr
    config["lr"] = 0
    config["vf_clip_param"] = args.maml_vf_clip_param
    config["maml_optimizer_steps"] = 1
    #config["reuse_actors"] = True
    # config["evaluation_config"] = config["env_config"]
    # config["evaluation_interval"] = 10
    # config["evaluation_num_episodes"] = 10
    # config["evaluation_num_workers"] = 1
    model_weights = trainer.get_weights()
    print("got model weight")
    trainer.stop()
    print("trainer stopped")
    updated_agent = ray_maml.MAMLTrainer(config=config, env = SocialGameMetaEnv)
    print("started new trainer")
    updated_agent.set_weights(model_weights)
    print("set new weights")
    # import pprint; pp = pprint.PrettyPrinter(indent=4); print(pp.pprint(updated_agent.__dict__))
    # import pdb; pdb.set_trace()
    to_log = ["episode_reward_mean", "episode_reward_mean_adapt_5", "adaptation_delta"]

    for i in range(10):
        print("step {}".format(i))
        result = updated_agent.train()
        log = {"val_maml_" + name: result[name] for name in to_log}
        if args.wandb:
            wandb.log({'validation_step': i})
            wandb.log(log)
            wandb.log({"val_maml_total_loss": result["info"]["learner"]["default_policy"]["total_loss"]})
        else:
            print(log)
    model_weights2 = updated_agent.get_weights()
    np.testing.assert_equal(model_weights2, model_weights, err_msg="Weights were changed during validation")
    if str(model_weights) != str(model_weights2):
        print("Weights were changed")
    else:
        print("Weights were not changed")
    

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

    elif args.library=="rllib":

        if args.algo == "ppo":
            train_batch_size = 4000
            config = ray_ppo.DEFAULT_CONFIG.copy()
            config["framework"] = "torch"
            config["train_batch_size"] = train_batch_size
            config["num_gpus"] = 0
            config["num_workers"] = 1
            config["env"] = SocialGameEnvRLLib
            config["env_config"] = vars(args)
            trainer = ray_ppo.PPOTrainer(config=config, env= SocialGameEnvRLLib)
            return trainer

        elif args.algo == "maml":
            config = ray_maml.DEFAULT_CONFIG.copy()
            config["framework"] = "tf1"
            config["num_gpus"] = 1
            config["num_envs_per_worker"] = 2
            config["num_workers"] = args.maml_num_workers
            
            config["inner_adaptation_steps"] = args.maml_inner_adaptation_steps
            config["env"] = SocialGameMetaEnv
            config["env_config"] = vars(args)
            config["env_config"]["mode"] = "train"
            config["normalize_actions"] = True
            config["clip_actions"] = True
            config["inner_lr"] = args.maml_inner_lr
            config["lr"] = args.maml_outer_lr
            config["vf_clip_param"] = args.maml_vf_clip_param
            trainer = ray_maml.MAMLTrainer(config=config, env = SocialGameMetaEnv)
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
        response_type_string=args.response_type_string,
        one_day=args.one_day,
        number_of_participants=args.number_of_participants,
        price_in_state = args.price_in_state,
        energy_in_state=args.energy_in_state,
        pricing_type=args.pricing_type,
        reward_function=reward_function,
        bin_observation_space = args.bin_observation_space,
        manual_tou_magnitude=args.manual_tou_magnitude,
        use_smirl=args.smirl
        )


    # Check to make sure any new changes to environment follow OpenAI Gym API
    check_env(socialgame_env)

    return socialgame_env

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

    elif args.library=="rllib":
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
        "--algo",
        help="RL Algorithm",
        type=str,
        default="sac",
        choices=["sac", "ppo", "maml"]
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
        default="log_cost_regularized",
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
    parser.add_argument(
        "--library",
        help = "What RL Library backend is in use",
        type = str,
        default = "sb3",
        choices = ["sb3", "rllib"]
    )
    parser.add_argument(
        "--smirl",
        help="Whether to run with SMiRL",
        action="store_true"
    )
    parser.add_argument(
        "--maml_num_workers",
        help = "Number of workers to use with MAML",
        type = int,
        default = 1
    )
    parser.add_argument(
        "--maml_inner_adaptation_steps",
        help = "Number of inner adaptation steps for MAML",
        type = int,
        default = 1
    )
    parser.add_argument(
        "--maml_inner_lr",
        help = "Learning rate for inner adaptation training",
        type = float,
        default = 0.1
    )
    parser.add_argument(
        "--maml_vf_clip_param",
        help = "Clip param for the value function. Note that this is sensitive to the \
                scale of the rewards. If your expected V is large, increase this.",
        type = float,
        default=10.0
    )
    parser.add_argument(
        "--maml_outer_lr",
        help = "Learning rate for meta adaptation training",
        type = float,
        default=0.001
    )
    parser.add_argument(
        "--checkpoint_interval",
        help = "How many training iterations between checkpoints (only implemented for MAML)",
        type = int
    )

    args = parser.parse_args()

    args.log_path = os.path.join(args.base_log_dir, args.exp_name + "/")
    args.rl_log_path = os.path.join(args.log_path, "rl/")

    return args


def main():
    # Get args
    args = parse_args()

    # Print args for reference
    args_convert_bool(args)

    if args.wandb:
        wandb.init(project="energy-demand-response-game", entity="social-game-rl", sync_tensorboard=True)
        wandb.config.update(args)

    # Create environments

    if os.path.exists(args.log_path):
        print("Choose a new name for the experiment, log dir already exists")
        raise ValueError

    if args.library == "rllib" :
        ray.init()

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
    if args.algo == "maml":
        maml_eval_fn(model, args)
    print(
        "If there was no planning model involved, remember that the output will be in the log dir"
    )


if __name__ == "__main__":
    #Workaround for pdb to work with multiprocessing
    __spec__=None
    main()
