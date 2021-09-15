"""Example of using RLlib's debug callbacks.

Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict
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

    def __init__(self, log_path, save_interval, obs_dim=10):
        super().__init__()
        self.log_path=log_path
        self.save_interval=save_interval
        self.cols = ["step", "energy_reward", "smirl_reward", "energy_cost"]
        for i in range(obs_dim):
            self.cols.append("observation_" + str(i))
        self.obs_dim = obs_dim
        self.log_vals = {k: [] for k in self.cols}
        print("initialized Custom Callbacks")

    def save(self):
        log_df=pd.DataFrame(data=self.log_vals)
        log_df.to_hdf(self.log_path, "metrics", append=True, format="table")
        for v in self.log_vals.values():
            v.clear()

        self.steps_since_save=0


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
                        self.log_vals["observation_" + str(i)].append(k)
                else:
                    for i in range(obs_dim):
                        self.log_vals["observation_" + str(i)].append(np.nan)

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
                    self.log_vals["observation_" + str(i)].append(k)
            else:
                for i in range(obs_dim):
                    self.log_vals["observation_" + str(i)].append(np.nan)

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

