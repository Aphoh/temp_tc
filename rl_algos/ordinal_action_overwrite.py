import gym
import numpy as np
import tree  # pip install dm_tree
from typing import Union

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import get_variable, try_import_tf, \
    try_import_torch, TensorType

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

from ray.rllib.utils.exploration.stochastic_sampling import StochasticSampling
from ray.rllib.utils.annotations import override
import IPython
from scipy.special import expit

class OrdinalStochasticSampler(StochasticSampling):
    """A modification of the rllib stochastic sampling to allow for ordinal interpretation 
    of discrete logits
    
    This should pair with multi-discrete action environment in base RLLib / gym

    """


    @override(StochasticSampling)
    def get_exploration_action(self,
                                *,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True):
        if self.framework == "torch":
            print("ENTERING TORCH EXPLORATION")
            return self._get_torch_exploration_action(action_distribution,
                                                      timestep, explore)
        else:
            print("ENTERING TF EXPLORATION")
            return self._get_tf_exploration_action_op(action_distribution,
                                                      timestep, explore)

    def _get_tf_exploration_action_op(self, action_dist, timestep, explore):
        ts = timestep if timestep is not None else self.last_timestep + 1

        IPython.embed()

        logits = action_dist.inputs
        s = expit(logits)
        action_dist.input = [np.sum(np.log(s[:i+1])) + np.sum(1-np.log(s[i+1:])) for i in range(len(s))]

        stochastic_actions = tf.cond(
            pred=tf.convert_to_tensor(ts < self.random_timesteps),
            true_fn=lambda: (
                self.random_exploration.get_tf_exploration_action_op(
                    action_dist,
                    explore=True)[0]),
            false_fn=lambda: action_dist.sample(),
        )
        deterministic_actions = action_dist.deterministic_sample()

        action = tf.cond(
            tf.constant(explore) if isinstance(explore, bool) else explore,
            true_fn=lambda: stochastic_actions,
            false_fn=lambda: deterministic_actions)

        def logp_false_fn():
            batch_size = tf.shape(tree.flatten(action)[0])[0]
            return tf.zeros(shape=(batch_size, ), dtype=tf.float32)

        logp = tf.cond(
            tf.math.logical_and(
                explore, tf.convert_to_tensor(ts >= self.random_timesteps)),
            true_fn=lambda: action_dist.sampled_action_logp(),
            false_fn=logp_false_fn)

        # Increment `last_timestep` by 1 (or set to `timestep`).
        if self.framework in ["tf2", "tfe"]:
            if timestep is None:
                self.last_timestep.assign_add(1)
            else:
                self.last_timestep.assign(timestep)
            return action, logp
        else:
            assign_op = (tf1.assign_add(self.last_timestep, 1)
                         if timestep is None else tf1.assign(
                             self.last_timestep, timestep))
            with tf1.control_dependencies([assign_op]):
                return action, logp

    def _get_torch_exploration_action(self, action_dist: ActionDistribution,
                                      timestep: Union[TensorType, int],
                                      explore: Union[TensorType, bool]):
        # Set last timestep or (if not given) increase by one.
        self.last_timestep = timestep if timestep is not None else \
            self.last_timestep + 1

        logits = action_dist.inputs.cpu()
        s = expit(logits)

        print("--"*10)

        print(action_dist.inputs)

        print("--"*10)
        print("--"*10)

        ss_output = []
        for ss in s:
            row = []
            for i in range(len(ss)):
                if i == (len(ss)-1):
                    elem = sum(np.log(ss))
                else:
                    elem = sum(np.log(ss[:i+1])) + sum(1-np.log(ss[i+1:]))
                row.append(elem.item())
            ss_output.append(row)

        action_dist.inputs = torch.tensor(ss_output)

        print(action_dist.inputs)

        # Apply exploration.
        if explore:
            # Random exploration phase.
            if self.last_timestep < self.random_timesteps:
                action, logp = \
                    self.random_exploration.get_torch_exploration_action(
                        action_dist, explore=True)
            # Take a sample from our distribution.
            else:
                action = action_dist.sample()
                logp = action_dist.sampled_action_logp()

        # No exploration -> Return deterministic actions.
        else:
            action = action_dist.deterministic_sample()
            logp = torch.zeros_like(action_dist.sampled_action_logp())

        return action, logp
