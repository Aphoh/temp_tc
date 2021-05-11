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

        logits = action_distribution.inputs.cpu()
        s = expit(logits)

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

        action_distribution.inputs = torch.tensor(ss_output)

        print("--"*10)

        print(action_distribution.inputs)

        print("--"*10)
        print("--"*10)

        if self.framework == "torch":
            return self._get_torch_exploration_action(action_distribution,
                                                      timestep, explore)
        else:
            return self._get_tf_exploration_action_op(action_distribution,
                                                      timestep, explore)

