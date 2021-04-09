"""Example of using RLlib's debug callbacks.

Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict
import numpy as np
import wandb
import csv

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch



class CustomCallbacks(DefaultCallbacks):
    out_name=""
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):

        socialgame_env = base_env.get_unwrapped()[0]
        if socialgame_env.use_smirl:
            episode.user_data["smirl_reward"] = []
            episode.hist_data["smirl_reward"] = []

        episode.user_data["energy_reward"] = []
        episode.hist_data["energy_reward"] = []
        episode.user_data["observations"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        socialgame_env = base_env.get_unwrapped()[0]
        if socialgame_env.use_smirl and socialgame_env.last_smirl_reward:
            episode.user_data["smirl_reward"].append(socialgame_env.last_smirl_reward)
            episode.hist_data["smirl_reward"].append(socialgame_env.last_smirl_reward)

        if socialgame_env.last_energy_reward:
            episode.user_data["energy_reward"].append(base_env.get_unwrapped()[0].last_energy_reward)
            episode.hist_data["energy_reward"].append(base_env.get_unwrapped()[0].last_energy_reward)

        obs = socialgame_env._get_observation()
        if obs is not None:
            episode.user_data["observations"].append(obs)

        return

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        socialgame_env = base_env.get_unwrapped()[0]
        episode.custom_metrics["energy_reward"] = np.mean(episode.user_data["energy_reward"])

        if socialgame_env.use_smirl:
            episode.custom_metrics["smirl_reward"] = np.mean(episode.user_data["smirl_reward"])

        with open(CustomCallbacks.out_name, 'a') as fout:
            w = csv.writer(fout, delimiter=",")
            w.writerows(episode.user_data["observations"])

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

