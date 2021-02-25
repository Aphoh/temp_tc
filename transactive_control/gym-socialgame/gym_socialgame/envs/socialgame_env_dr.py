import gym
from gym import spaces

import numpy as np

from gym_socialgame.envs.socialgame_env import SocialGameEnv
from gym_socialgame.envs.utils import price_signal
from gym_socialgame.envs.agents import *
from gym_socialgame.envs.reward import Reward

class SocialGameEnvDR(SocialGameEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space_string = "continuous", response_type_string = "l", number_of_participants = 10,
                one_price = 0, low = 0, high = 50, distr = 'U', energy_in_state = True, yesterday_in_state = False):
        """
        SocialGameEnv for an agent determining incentives in a social game. 
        
        Note: One-step trajectory (i.e. agent submits a 10-dim vector containing incentives for each hour (8AM - 5PM) each day. 
            Then, environment advances one-day and agent is told that the episode has finished.)

        Args:
            action_space_string: (String) either "continuous", or "multidiscrete"
            response_type_string: (String) either "t", "s", "l" , denoting whether the office's response function is threshold, sinusoidal, or linear
            number_of_participants: (Int) denoting the number of players in the social game (must be > 0 and < 20)
            one_price: (Int) in range [-1,365] denoting which fixed day to train on . 
                    Note: -1 = Random Day, 0 = Train over entire Yr, [1,365] = Day of the Year
            Low: Lower bound for DR distribution
            High: Upper bound for DR distribution
            Distribution: U for Uniform, G for Gaussian
            energy_in_state: (Boolean) denoting whether (or not) to include the previous day's energy consumption within the state
            yesterday_in_state: (Boolean) denoting whether (or not) to append yesterday's price signal to the state

        """
        #Checking that random and corresp. param are valid
        assert isinstance(low, int), "Variable low is not an integer. Got type {}".format(type(low))
        assert isinstance(high, int), "Variable high is not an integer. Got type {}".format(type(high))
        assert isinstance(distr, str), "Variable distr is not a String. Got type {}".format(type(distr))
        assert distr.upper() in ['G', 'U'], "Distr not either G or U. Got {}".format(distr.upper())

        #Set Randomization param
        self.low = low
        self.high = high
        self.distr = distr.upper()

        super().__init__(action_space_string=action_space_string,
                                            response_type_string=response_type_string,
                                            number_of_participants=number_of_participants,
                                            one_price = one_price,
                                            energy_in_state=energy_in_state,
                                            yesterday_in_state=yesterday_in_state)    

    def _create_agents(self):
        """
        Purpose: Create the participants of the social game. We create a game with n players, where n = self.number_of_participants

        Args:
            None

        Returns:
              agent_dict: Dictionary of players, each with response function based on self.response_type_string

        """

        player_dict = {}

        #Sample Energy from average energy in the office (pre-treatment) from the last experiment 
        #Reference: Lucas Spangher, et al. Engineering  vs.  ambient  typevisualizations:  Quantifying effects of different data visualizations on energy consumption. 2019
        sample_energy = np.array([ 0.28,  11.9,   16.34,  16.8,  17.43,  16.15,  16.23,  15.88,  15.09,  35.6, 
                                123.5,  148.7,  158.49, 149.13, 159.32, 157.62, 158.8,  156.49, 147.04,  70.76,
                                42.87,  23.13,  22.52,  16.8 ])

        #only grab working hours (8am - 5pm)
        working_hour_energy = sample_energy[8:18]

        my_baseline_energy = pd.DataFrame(data={"net_energy_use": working_hour_energy})

        for i in range(self.number_of_participants):
            player = RandomizedFunctionPerson(my_baseline_energy, points_multiplier=10, response = self.response_type_string, 
                                            low = self.low, high = self.high, distr = self.distr)
            
            player_dict['player_{}'.format(i)] = player

        return player_dict



    def _simulate_humans(self, action):
        """
        Purpose: Gets energy consumption from players given action from agent

        Args:
            Action: 10-dim vector corresponding to action for each hour 8AM - 5PM
        
        Returns: 
            Energy_consumption: Dictionary containing the energy usage by player and the average energy used in the office (key = "avg")
        """

        energy_consumptions = {}
        total_consumption = np.zeros(10)

        for player_name in self.player_dict:

            #Get players response to agent's actions
            player = self.player_dict[player_name]
            player_energy = player.get_response(action)
            
            #Once energy is given, update their random noise for next episode
            player.update_noise()

            #Calculate energy consumption by player and in total (over the office)
            energy_consumptions[player_name] = player_energy
            total_consumption += player_energy

        energy_consumptions["avg"] = total_consumption / self.number_of_participants
        return energy_consumptions

    def step(self, action):
        """
        Purpose: Takes a step in the environment 

        Args:
            Action: 10-dim vector detailing player incentive for each hour (8AM - 5PM)
        
        Returns: 
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)
        
        Exceptions:
            raises AssertionError if action is not in the action space
        """
        #Checking that action is valid; If not, we clip (OpenAI algos don't take into account action space limits so we must do it ourselves)
        if(not self.action_space.contains(action)):
            action = np.asarray(action)
            if(self.action_space_string == 'continuous'):
                action = np.clip(action, 0, 10)

            elif(self.action_space_string == 'multidiscrete'):
                action = np.clip(action, 0, 2) 

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.cur_iter += 1
        if self.cur_iter > 0:
            done = True
        else:
            done = False

        points = self._points_from_action(action)

        energy_consumptions = self._simulate_humans(points)
        
        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same
        self.prev_energy = energy_consumptions["avg"]
        
        observation = self._get_observation()
        reward = self._get_reward(prev_price, energy_consumptions)
        info = {}
        return observation, reward, done, info


    def reset(self):
        """ Resets the environment on the current day """ 
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass



