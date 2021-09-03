"""Example of using RLlib's debug callbacks.

Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from pdb import set_trace
from typing import Dict
import numpy as np
import pandas as pd

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.env.normalize_actions import NormalizeActionWrapper
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

class CustomCallbacks(DefaultCallbacks):

    def __init__(self, log_path, save_interval, obs_dim=10, planning_model=None, env=None, args=None):
        super().__init__()
        self.log_path=log_path
        self.save_interval=save_interval
        self.cols = ["step", "energy_reward", "smirl_reward", "energy_cost"]
        for i in range(obs_dim):
            self.cols.append("observation_" + str(i))
        self.obs_dim = obs_dim
        self.log_vals = {k: [] for k in self.cols}
        self.log_path=log_path
        self.planning_model=planning_model
        self.env = env
        self.args = args

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

        episode.user_data["real_step"] = []
        episode.hist_data["real_step"] = []

        episode.user_data["predicted_costs"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        socialgame_env = base_env.get_unwrapped()[0]

        step_i = socialgame_env.total_iter
        if not hasattr(socialgame_env, "planning_steps"):
            self.log_vals["step"].append(step_i)        

        # TODO: Implement logging for planning_env 
        if hasattr(socialgame_env, "is_step_in_real"):
            if socialgame_env.is_step_in_real:
                print("Logging real step: ", self.steps_since_save)
                self.log_vals["step"].append(socialgame_env.num_real_steps)
                episode.user_data["real_step"].append(socialgame_env.num_real_steps)
                episode.hist_data["real_step"].append(socialgame_env.num_real_steps)
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
                if isinstance(socialgame_env, NormalizeActionWrapper):
                    obs = socialgame_env.env._get_observation()
                else:
                    obs = socialgame_env._get_observation()
                if obs is not None:
                    for i, k in enumerate(obs.flatten()):
                        self.log_vals["observation_" + str(i)].append(k)
                else:
                    for i in range(self.obs_dim):
                        self.log_vals["observation_" + str(i)].append(np.nan)

                self.steps_since_save += 1
                
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

            if isinstance(socialgame_env, NormalizeActionWrapper):
                obs = socialgame_env.env._get_observation()
            else:
                obs = socialgame_env._get_observation()
            if obs is not None:
                for i, k in enumerate(obs.flatten()):
                    self.log_vals["observation_" + str(i)].append(k)
            else:
                for i in range(self.obs_dim):
                    self.log_vals["observation_" + str(i)].append(np.nan)

            self.steps_since_save += 1
        return

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        socialgame_env = base_env.get_unwrapped()[0]
        episode.custom_metrics["real_step"] = np.mean(episode.user_data["real_step"])
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
        if len(postprocessed_batch) > 1:
            print("Error: trajectories larger than 1")
            import pdb; pdb.set_trace()
        if self.planning_model != None:
            # Assumes length of trajectory is 1
            actions = postprocessed_batch["actions"]
            rewards = postprocessed_batch["rewards"]
            predicted_consumption, _ = self.planning_model(actions) # TODO, SEE WHAT SAMPLEBATCH ACTUALLY IS
            tou = 0.103 * np.ones((10), dtype=np.float32)
            tou[5:8] = self.args.manual_tou_magnitude
            _, tou_rewards, _, _ = self.env.step(tou)
            tou_cost = 75#self.env.last_energy_cost
            predicted_costs = (np.dot(predicted_consumption, self.env.prices[self.env.day].squeeze()) / self.env.number_of_participants) * 500 * 0.001
            episode.custom_metrics["predicted_costs"] = predicted_costs
            if predicted_costs > tou_cost:
            #if True:
                print("Replacing...")
                postprocessed_batch["rewards"] = tou_rewards.reshape(rewards.shape)
                postprocessed_batch["actions"] = tou.reshape(actions.shape)
                actions[:] = tou
                #Revise the logs if the intervention is necessary
                #WARNING: WILL NOT CORRECTLY LOG SMIRL REWARDS
                if "energy_reward" in self.log_vals and len(self.log_vals["energy_reward"]) > 0:
                    episode.user_data["energy_reward"][-1] = tou_rewards
                    episode.hist_data["energy_reward"][-1] = tou_rewards
                    self.log_vals["energy_reward"][-1] = tou_rewards

                if "energy_cost" in self.log_vals and len(self.log_vals["energy_cost"]) > 0:
                    episode.user_data["energy_cost"][-1] = tou_cost
                    episode.hist_data["energy_cost"][-1] = tou_cost
                    self.log_vals["energy_cost"][-1] = tou_cost
        if self.steps_since_save >= self.save_interval:
                self.save()
        
        return

