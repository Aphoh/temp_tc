import gym
from gym import spaces

import numpy as np

from gym_socialgame.envs.socialgame_env import SocialGameEnv
from gym_socialgame.envs.utils import price_signal
from gym_socialgame.envs.agents import *
from gym_socialgame.envs.reward import Reward


class SocialGameEnvHourly(SocialGameEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        action_space_string="continuous",
        response_type_string="l",
        number_of_participants=10,
        one_price=0,
        energy_in_state=True,
        yesterday_in_state=False,
    ):
        """
        SocialGameEnv for an agent determining incentives in a social game. 
        
        Note: One-step trajectory (i.e. agent submits a 10-dim vector containing incentives for each hour (8AM - 5PM) each day. 
            Agent provides incentives per hour until the end-of-the-day, when episode is finished.

        Args: (same as SocialGameEnv)
            action_space_string: (String) either "continuous", or "multidiscrete"
            response_type_string: (String) either "t", "s", "l" , denoting whether the office's response function is threshold, sinusoidal, or linear
            number_of_participants: (Int) denoting the number of players in the social game (must be > 0 and < 20)
            one_price: (Int) in range [-1,365] denoting which fixed day to train on . 
                    Note: -1 = Random Day, 0 = Train over entire Yr, [1,365] = Day of the Year
            energy_in_state: (Boolean) denoting whether (or not) to include the previous day's energy consumption within the state
            yesterday_in_state: (Boolean) denoting whether (or not) to append yesterday's price signal to the state

        """
        super(SocialGameEnvHourly, self).__init__(
            action_space_string=action_space_string,
            response_type_string=response_type_string,
            number_of_participants=number_of_participants,
            one_price=one_price,
            energy_in_state=energy_in_state,
            yesterday_in_state=yesterday_in_state,
        )

        self.action_length = 1
        self.prev_energy = np.zeros(10) #Stores full energy consumption vector from previous day
        self.hour = 0 #Goes from [0,10] (indicating 8AM - 5PM)
        self.points = np.zeros(10) #Array which accumulates points for the entire day



    def _points_from_action(self, action):
        """
        Purpose: Convert agent actions into incentives (conversion is for multidiscrete setting)

        Args:
            Action: 10-dim vector corresponding to action for each hour 8AM - 5PM
        
        Returns: Points: 10-dim vector of incentives for game (same incentive for each player)
        """
        if self.action_space_string == "multidiscrete":
            # Mapping 0 -> 0.0, 1 -> 5.0, 2 -> 10.0
            points = action

        elif self.action_space_string == "continuous":
            # Continuous space is symmetric [-1,1], we map to -> [0,10] by adding 1 and multiplying by 5
            points = 5 * (action + np.ones_like(action))

        return points
    
    def _create_action_space(self):
        """
        Purpose: Return action space of type specified by self.action_space_string

        Args:
            None
        
        Returns:
            Action Space for environment based on action_space_str 
        
        Note: Multidiscrete refers to a 10-dim vector where each action {0,1,2} represents Low, Medium, High points respectively.
        We pose this option to test whether simplifying the action-space helps the agent. 
        """

        #Making a symmetric, continuous space to help learning for continuous control (suggested in StableBaselines doc.)
        if self.action_space_string == "continuous":
            return spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        elif self.action_space_string == "multidiscrete":
            discrete_space = [self.action_subspace] * 1
            return spaces.MultiDiscrete(discrete_space)


    def _get_observation(self):
        """ Returns observation for current hour """

        # Observations are per hour now
        prev_price = np.array([self.prices[self.day][(self.hour - 1) % 10]])

        next_observation = np.array([self.prices[self.day][self.hour]])

        prev_energy = np.array([self.prev_energy[self.hour]]) #Using previous day's energy from current hour (e.g. if its 8AM Tuesday, prev_enery = energy used on 8AM on Monday)
        
        if self.yesterday_in_state:
            if self.energy_in_state:
                return np.concatenate(
                    (next_observation, np.concatenate((prev_price, prev_energy)))
                )
            else:
                return np.concatenate((next_observation, prev_price))

        elif self.energy_in_state:
            return np.concatenate((next_observation, prev_energy))

        else:
            return next_observation

    def step(self, action):
        """
        Purpose: Takes a step in the environment 

        Args:
            Action: 1-dim vector detailing player incentive for a given hour
        
        Returns: 
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)

        """
        # Checking that action is valid; If not, we clip (OpenAI algos don't take into account action space limits so we must do it ourselves)
        if not self.action_space.contains(action):
            action = np.asarray(action)
            if self.action_space_string == "continuous":
                action = np.clip(action, 0, 10)

            elif self.action_space_string == "multidiscrete":
                action = np.clip(action, 0, 2)


        point = self._points_from_action(action)[0]

        # Advancing hour
        self.hour += 1

        # Setting done and return reward
        if self.hour == 10:
            #Calculate energy used in the day
            energy_consumptions = self._simulate_humans(self.points)
            self.prev_energy = energy_consumptions["avg"]

            # Reset hour
            self.hour = 0

            # Advance one day
            self.day = (self.day + 1) % 365

            #Reset saved points
            self.points

            # Finish episode
            done = True
            reward = self._get_reward(self.prices[self.day], energy_consumptions)

        else:
            #Save points
            self.points[self.hour] = point

            #Update done, reward (False, 0.0 b/c episode still in progress!)
            done = False
            reward = 0.0

        # Setting info for baselines compatibility
        info = {}
        
        # Getting next observation
        observation = self._get_observation()
        
        return observation, reward, done, info

    # Keeping reset, render, close for clarity sake
    def reset(self):
        """ Resets the environment on the current day """
        # Currently resetting based on current day to work with StableBaselines

        return self._get_observation()

    def render(self, mode="human"):
        pass

    def close(self):
        pass
