"""Example of using RLlib's debug callbacks.

Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict
import IPython
import numpy as np
import pandas as pd
import pdb

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

class CustomCallbacks(DefaultCallbacks):

    ## Note: (Lucas) I started to combine agent and multiagent into one class.
    #               Then, I decided to make the on_episode_start, step, and end
    #               different for different types. So the __init__ still has both,
    #               but the rest of the functions here are _not_ for multiagent.

    def __init__(self, log_path, save_interval, obs_dim=10, energy_consumption_dim = 24, multiagent = False):
        super().__init__()
        self.log_path = log_path + ".h5"
        self.save_interval = save_interval
        self.cols = ["step", "energy_reward", "smirl_reward", "energy_cost"]
        for i in range(obs_dim):
            self.cols.append("utility_price_hour_" + str(i))
        
        ## we're gonna have to change this if doing people agents
        for i in range(24):
            self.cols.append("agent_buy_price_hour_" + str(i))
            self.cols.append("agent_sell_price_hour_" + str(i))

        for i in range(energy_consumption_dim):
            self.cols.append("prosumer_response_hour_" + str(i))

        self.energy_consumption_dim = energy_consumption_dim
        self.cols.append("pv_size")
        self.cols.append("battery_size")
        self.cols.append("sample_user")
        self.obs_dim = obs_dim
        self.multiagent = multiagent

        self.log_vals = {k: [] for k in self.cols}
        print("initialized Custom Callbacks")

    def save(self):

        IPython.embed()

        log_df=pd.DataFrame(data=self.log_vals)
        log_df.to_hdf(self.log_path, "metrics", append=True, format="table")
        for v in self.log_vals.values():
            v.clear()


    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):

        socialgame_env = base_env.get_unwrapped()[0]
        if socialgame_env.use_smirl:
            episode.user_data["smirl_reward"] = []
            episode.hist_data["smirl_reward"] = []

        episode.user_data["energy_reward"] = []
        episode.hist_data["energy_reward"] = []

        episode.user_data["energy_cost"] = []
        episode.hist_data["energy_cost"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):

        socialgame_env = base_env.get_unwrapped()[0]

        step_i = socialgame_env.total_iter

        self.log_vals["step"].append(step_i)   

        # TODO: Implement logging for planning_env 
        if hasattr(socialgame_env, "planning_steps") and socialgame_env.planning_steps > 0: 
            if socialgame_env.is_step_in_real:
                if socialgame_env.use_smirl and socialgame_env.last_smirl_reward:
                    smirl_rew = socialgame_env.last_smirl_reward
                    episode.user_data["smirl_reward"].append(smirl_rew)
                    episode.hist_data["smirl_reward"].append(smirl_rew)
                    self.log_vals["smirl_reward"].append(smirl_rew)
                else:
                    self.log_vals["smirl_reward"].append(np.nan)

                if socialgame_env.last_energy_reward:
                    energy_rew = socialgame_env.last_energy_reward
                    episode.user_data["energy_reward"].append(energy_rew)
                    episode.hist_data["energy_reward"].append(energy_rew)
                    self.log_vals["energy_reward"].append(energy_rew)
                else:
                    self.log_vals["energy_reward"].append(np.nan)

                if socialgame_env.last_energy_cost:
                    energy_cost = socialgame_env.last_energy_cost
                    episode.user_data["energy_cost"].append(energy_cost)
                    episode.hist_data["energy_cost"].append(energy_cost)
                    self.log_vals["energy_cost"].append(energy_cost)
                else:
                    self.log_vals["energy_cost"].append(np.nan)

                obs = socialgame_env._get_observation()
                if obs is not None:
                    for i, k in enumerate(obs.flatten()):
                        self.log_vals["utility_price_hour_" + str(i)].append(k)
                else:
                    for i in range(self.obs_dim):
                        self.log_vals["utility_price_hour_" + str(i)].append(np.nan)

                # log the agent's prices 
                for i, k in enumerate(socialgame_env.action.flatten()):
                    if i < 24:
                        self.log_vals["agent_buy_price_hour_" + str(i)].append(k)
                    else:
                        self.log_vals["agent_sell_price_hour_" + str(i - 24)].append(k)
          
                for i in range(self.energy_consumption_dim):
                    self.log_vals["prosumer_response_hour_" + str(i)] = (
                        socialgame_env.sample_user_response["prosumer_response_hour_" + str(i)]
                    )
                self.log_vals["pv_size"] = socialgame_env.sample_user_response["pv_size"]
                self.log_vals["battery_size"] = socialgame_env.sample_user_response["battery_size"]
                self.log_vals["sample_user"] = socialgame_env.sample_user_response["sample_user"]


                self.steps_since_save += 1
                if self.steps_since_save == self.save_interval:
                    self.save()
        else:
            if socialgame_env.use_smirl and socialgame_env.last_smirl_reward:
                smirl_rew = socialgame_env.last_smirl_reward
                episode.user_data["smirl_reward"].append(smirl_rew)
                episode.hist_data["smirl_reward"].append(smirl_rew)
                self.log_vals["smirl_reward"].append(smirl_rew)
            else:
                self.log_vals["smirl_reward"].append(np.nan)

            if socialgame_env.last_energy_reward:
                energy_rew = socialgame_env.last_energy_reward
                episode.user_data["energy_reward"].append(energy_rew)
                episode.hist_data["energy_reward"].append(energy_rew)
                self.log_vals["energy_reward"].append(energy_rew)
            else:
                self.log_vals["energy_reward"].append(np.nan)

            if socialgame_env.last_energy_cost:
                energy_cost = socialgame_env.last_energy_cost
                episode.user_data["energy_cost"].append(energy_cost)
                episode.hist_data["energy_cost"].append(energy_cost)
                self.log_vals["energy_cost"].append(energy_cost)
            else:
                self.log_vals["energy_cost"].append(np.nan)

            obs = socialgame_env._get_observation()

            if obs is not None:
                for i, k in enumerate(obs.flatten()):
                    self.log_vals["utility_price_hour_" + str(i)].append(k)
            else:
                for i in range(self.obs_dim):
                    self.log_vals["utility_price_hour_" + str(i)].append(np.nan)

            for i, k in enumerate(socialgame_env.action.flatten()):
                if i < 24:
                    self.log_vals["agent_buy_price_hour_" + str(i)].append(k)
                else:
                    self.log_vals["agent_sell_price_hour_" + str(i - 24)].append(k)

            for i in range(self.energy_consumption_dim):
                self.log_vals["prosumer_response_hour_" + str(i)] = (
                    socialgame_env.sample_user_response["prosumer_response_hour_" + str(i)]
                )
            self.log_vals["pv_size"] = socialgame_env.sample_user_response["pv_size"]
            self.log_vals["battery_size"] = socialgame_env.sample_user_response["battery_size"]
            self.log_vals["sample_user"] = socialgame_env.sample_user_response["sample_user"]


            self.steps_since_save += 1
            if self.steps_since_save == self.save_interval:
                self.save()

        return

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        socialgame_env = base_env.get_unwrapped()[0]
        episode.custom_metrics["energy_reward"] = np.mean(episode.user_data["energy_reward"])
        episode.custom_metrics["energy_cost"] = np.mean(episode.user_data["energy_cost"])
        if socialgame_env.use_smirl:
            episode.custom_metrics["smirl_reward"] = np.mean(episode.user_data["smirl_reward"])

        return

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):

        return

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["callback_ok"] = True


    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0

        episode.custom_metrics["num_batches"] += 1
        return

class CustomCallbacksMultiagent(CustomCallbacks):
    def __init__(self, log_path, save_interval, obs_dim=10, multiagent = True):
        super().__init__(
            log_path=log_path, 
            save_interval=save_interval, 
            obs_dim=obs_dim,
            multiagent=True)

        # TODO: generalize this for arbitrary # of multiagents
        
        ## we're gonna have to change this if doing people agents
        for i in range(24):
            self.cols.append("agent_buy_price_hour_" + str(i))
            self.cols.append("agent_sell_price_hour_" + str(i))
        
        self.log_vals = {
            "real": {k: [] for k in self.cols},
            "shadow": {k: [] for k in self.cols},
            }
        self.log_path_real = self.log_path + "real.h5"
        self.log_path_shadow = self.log_path + "shadow.h5"
        print("initialized Custom Callbacks w real and shadow agents")


    def save(self):

        # TODO: generalize this to an arbitrary number of agents. 
        # will probably need to instantiate a self object with agent ids 

        # IPython.embed()

        log_df_real = pd.DataFrame(data=self.log_vals["real"])
        log_df_shadow = pd.DataFrame(data=self.log_vals["shadow"])
        log_df_real.to_hdf(self.log_path_real, "metrics", append=True, format="table")
        log_df_shadow.to_hdf(self.log_path_shadow, "metrics", append=True, format="table")
        for v in self.log_vals["real"].values():
            v.clear()
        for v in self.log_vals["shadow"].values():
            v.clear()
                                          
        self.steps_since_save=0

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):

        socialgame_env = base_env.get_unwrapped()[0]
        obs = socialgame_env._get_observation()

        for key, value in obs.items():
            if socialgame_env.use_smirl:
                episode.user_data["smirl_reward_" + key] = []
                episode.hist_data["smirl_reward_" + key] = []

            episode.user_data["energy_reward_" + key] = []
            episode.hist_data["energy_reward_" + key] = []

            episode.user_data["energy_cost_" + key] = []
            episode.hist_data["energy_cost_" + key] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):

        socialgame_env = base_env.get_unwrapped()[0]
        obs = socialgame_env._get_observation()
        agent_ids  = obs.keys()

        step_i = socialgame_env.total_iter

        for key, value in obs.items(): 
            self.log_vals[key]["step"].append(step_i)
            
        if socialgame_env.use_smirl and socialgame_env.last_smirl_reward:
            for key, value in socialgame_env.last_smirl_reward.items():
                episode.user_data["smirl_reward_" + key].append(value)
                episode.hist_data["smirl_reward_" + key].append(value)
                self.log_vals[key]["smirl_reward"].append(value)
        else:
            ## TODO: loop this over keys when you make a self.agents_id var 
            # r.n. it doesn't really matter
            for key in agent_ids:
                self.log_vals[key]["smirl_reward"].append(np.nan)

        if socialgame_env.last_energy_reward:
            for key, energy_rew in socialgame_env.last_energy_reward.items():
                episode.user_data["energy_reward_" + key].append(energy_rew)
                episode.hist_data["energy_reward_" + key].append(energy_rew)
                self.log_vals[key]["energy_reward"].append(energy_rew)
        else:
            for key in agent_ids:
                self.log_vals[key]["energy_reward"].append(np.nan)

        if socialgame_env.last_energy_cost.items():
            for key, energy_cost in socialgame_env.last_energy_cost.items():
                episode.user_data["energy_cost_" + key].append(energy_cost)
                episode.hist_data["energy_cost_" + key].append(energy_cost)
                self.log_vals[key]["energy_cost"].append(energy_cost)
        else:
            for key in agent_ids:
                self.log_vals[key]["energy_cost"].append(np.nan)

        # utility prices 
        if obs is not None:
            for key, value in obs.items():
                for i, k in enumerate(value.flatten()):
                    self.log_vals[key]["utility_price_hour_" + str(i)].append(k)
        else:
            for key in self.log_vals.keys():
                for i in range(self.obs_dim):
                    self.log_vals[key]["utility_price_hour_" + str(i)].append(np.nan)

        # agent buy and sell prices
        for key, val in socialgame_env.action_dict.items():
            for i, k in enumerate(val.flatten()):
                if i < 24:
                    self.log_vals[key]["agent_buy_price_hour_" + str(i)].append(k)
                else:
                    self.log_vals[key]["agent_sell_price_hour_" + str(i - 24)].append(k)

        # prosumer responses
        for key, val in self.log_vals.items():
            for i, k in enumerate(socialgame_env.sample_user_response[key]):
                self.log_vals[key]["prosumer_response_hour_" + str(i)].append(k)

        # various metadata

        for key, val in self.log_vals.items():
            self.log_vals[key]["pv_size"] = socialgame_env.sample_user_response["pv_size"]
            self.log_vals[key]["battery_size"] = socialgame_env.sample_user_response["battery_size"]
            self.log_vals[key]["sample_user"] = socialgame_env.sample_user_response["sample_user"]



        self.steps_since_save += 1
        if self.steps_since_save == self.save_interval:
            self.save()

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        socialgame_env = base_env.get_unwrapped()[0]
        obs = socialgame_env._get_observation()
        for key, value in obs.items():
            episode.custom_metrics["energy_reward_" + key] = (
                np.mean(episode.user_data["energy_reward_" + key]))
            episode.custom_metrics["energy_cost_" + key] = (
                np.mean(episode.user_data["energy_cost_" + key]))
            if socialgame_env.use_smirl:
                episode.custom_metrics["smirl_reward_" + key] = (
                    np.mean(episode.user_data["smirl_reward_" + key]))
        return