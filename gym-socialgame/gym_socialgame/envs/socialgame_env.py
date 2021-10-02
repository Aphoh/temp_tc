from math import isnan
import gym
from gym import spaces

import numpy as np
import random

from gym_socialgame.envs.utils import price_signal
from gym_socialgame.envs.agents import *
from gym_socialgame.envs.reward import Reward
from gym_socialgame.envs.buffers import (GaussianBuffer, GaussianCircularBuffer)

from gym_socialgame.envs.planning_net import EnsembleModule, Net
from numpy.linalg import multi_dot

from sklearn.preprocessing import MinMaxScaler
import IPython
import torch
class SocialGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
        action_space_string = "continuous",
        response_type_string = "l",
        number_of_participants = 10,
        one_day = 0,
        price_in_state= False,
        energy_in_state = False,
        day_of_week = False,
        pricing_type="TOU",
        reward_function = "log_cost_regularized",
        bin_observation_space=False,
        manual_tou_magnitude=.3,
        person_type_string="c",
        points_multiplier=10,
        smirl_weight=None,
        circ_buffer_size=None):

        """
        SocialGameEnv for an agent determining incentives in a social game.

        Note: One-step trajectory (i.e. agent submits a 10-dim vector containing incentives for each hour (8AM - 5PM) each day.
            Then, environment advances one-day and agent is told that the episode has finished.)

        Args:
            action_space_string: (String) either "continuous", "continuous_normalized", "multidiscrete"
            response_type_string: (String) either "t", "s", "l" , denoting whether the office's response function is threshold, sinusoidal, or linear
            number_of_participants: (Int) denoting the number of players in the social game (must be > 0 and < 20)
            one_day: (Int) in range [-1,365] denoting which fixed day to train on .
                    Note: -1 = Random Day, 0 = Train over entire Yr, [1,365] = Day of the Year
            price_in_state: (Boolean) denoting whether (or not) to include the current grid price in the state
            energy_in_state: (Boolean) denoting whether (or not) to append yesterday's grid price to the state
            manual_tou_magnitude: (Float>1) The relative magnitude of the TOU pricing to the regular pricing

        """
        super(SocialGameEnv, self).__init__()

        #Verify that inputs are valid
        self.check_valid_init_inputs(
            action_space_string,
            response_type_string,
            number_of_participants,
            one_day,
            price_in_state,
            energy_in_state
        )

        if action_space_string == "continuous_normalized" and reward_function == "log_cost_regularized":
            print("WARNING: You should probably be using log_cost with the c_norm action space")

        #Assigning Instance Variables
        self.action_space_string = action_space_string
        self.response_type_string = response_type_string
        self.number_of_participants = number_of_participants
        self.one_day = self._find_one_day(one_day)
        self.price_in_state = price_in_state
        self.energy_in_state = energy_in_state
        self.reward_function = reward_function
        self.bin_observation_space = bin_observation_space
        self.manual_tou_magnitude = manual_tou_magnitude
        self.smirl_weight = smirl_weight
        self.hours_in_day = 10
        self.last_smirl_reward = None
        self.last_energy_reward = None
        self.person_type_string = person_type_string
        self.points_multiplier = points_multiplier
        self.last_energy_cost = None
        self.add_last_cost = True

        self.day = 0
        self.days_of_week = [0, 1, 2, 3, 4]
        self.day_of_week_flag = day_of_week
        self.day_of_week = self.days_of_week[self.day % 5]

        #Create Observation Space (aka State Space)
        self.observation_space = self._create_observation_space()

        self.pricing_type = "real_time_pricing" if pricing_type.upper() == "RTP" else "time_of_use"

        self.prices = self._get_prices()
        #Day corresponds to day # of the yr

        #Cur_iter counts length of trajectory for current step (i.e. cur_iter = i^th hour in a 10-hour trajectory)
        #For our case cur_iter just flips between 0-1 (b/c 1-step trajectory)
        #TODO: this above comment ^ is wrong
        self.curr_iter = 0
        self.total_iter = 0

        #Create Action Space
        self.action_length = 10 # number of hours in a day
        self.action_subspace = 3
        self.action_subspace = 11
        self.action_space = self._create_action_space()

        #Create Players
        self.player_dict = self._create_agents()

        #TODO: Check initialization of prev_energy
        self.prev_energy = np.zeros(10)

        self.use_smirl = smirl_weight > 0 if smirl_weight else False
        if self.use_smirl:
            if circ_buffer_size and circ_buffer_size > 0:
                print("Using circular gaussian buffer")
                self.buffer = GaussianCircularBuffer(self.action_length, circ_buffer_size)
            else:
                print("Using standard gaussian buffer")
                self.buffer = GaussianBuffer(self.action_length)

        self.costs = []

        print("\n Social Game Environment Initialized! Have Fun! \n")

    def _find_one_day(self, one_day: int):
        """
        Purpose: Helper function to find one_day to train on (if applicable)

        Args:
            One_day: (Int) in range [-1,365]

        Returns:
            0 if one_day = 0
            one_day if one_day in range [1,365]
            random_number(1,365) if one_day = -1
        """

        return one_day if one_day != -1 else np.random.randint(0, high=365)

    def _create_observation_space(self):
        """
        Purpose: Returns the observation space.
        dim is 10 for previous days energy usage, 10 for prices

        Args:
            None

        Returns:
            Action Space for environment based on action_space_str
        """

        dim = self.hours_in_day*np.sum([self.price_in_state, self.energy_in_state])
        #TODO: Normalize obs_space !
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


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
            return spaces.Box(low=-1, high=1, shape=(self.action_length,), dtype=np.float32)

        elif self.action_space_string == "continuous_normalized":
            return spaces.Box(low = 0, high = np.inf, shape = (self.action_length,), dtype=np.float32)

        elif self.action_space_string == "multidiscrete":
            discrete_space = [self.action_subspace] * self.action_length  # num of actions times the length of the action space. [a1, a2, a3], [a1, a2, a3]
            return spaces.MultiDiscrete(discrete_space)

    def _create_agents(self):
        """
        Purpose: Create the participants of the social game. We create a game with n players, where n = self.number_of_participants

        Args:
            None

        Returns:
              agent_dict: Dictionary of players, each with response function based on self.response_type_string

        """

        player_dict = {}

        # Sample Energy from average energy in the office (pre-treatment) from the last experiment
        # Reference: Lucas Spangher, et al. Engineering  vs.  ambient  typevisualizations:  Quantifying effects of different data visualizations on energy consumption. 2019

        sample_energy = np.array([ 0.28,  11.9,   16.34,  16.8,  17.43,  16.15,  16.23,  15.88,  15.09,  35.6,
                                123.5,  148.7,  158.49, 149.13, 159.32, 157.62, 158.8,  156.49, 147.04,  70.76,
                                42.87,  23.13,  22.52,  16.8 ])

        #only grab working hours (8am - 5pm)
        working_hour_energy = sample_energy[8:18]

        my_baseline_energy = pd.DataFrame(data = {"net_energy_use" : working_hour_energy})

        for i in range(self.number_of_participants):
            if self.person_type_string=="c":
                print("using curtail and shift")
                player = CurtailAndShiftPerson(my_baseline_energy, points_multiplier = 10, response = 'l')
            elif self.person_type_string=="d":
                print("using deterministic with type {}".format(self.response_type_string))
                player = DeterministicFunctionPerson(my_baseline_energy, response=self.response_type_string, points_multiplier = self.points_multiplier)
            player_dict['player_{}'.format(i)] = player

        return player_dict


    def _get_prices(self):
        """
        Purpose: Get grid price signals for the entire year (using past data from a building in Los Angeles as reference)

        Args:
            None

        Returns: Array containing 365 price signals, where array[day_number] = grid_price for day_number from 8AM - 5PM

        """
        all_prices = []
        print("--" * 10)
        print("One day is: ", self.one_day)
        print("--" * 10)

        type_of_DR = self.pricing_type

        if self.manual_tou_magnitude:
            price = 0.103 * np.ones((365, 10))
            price[:,5:8] = self.manual_tou_magnitude
            print("Using manual tou pricing", price[0])
            return price

        if self.one_day is not None:
            print("Single Day")
            # If one_day we repeat the price signals from a fixed day
            # Tweak One_Day Price Signal HERE
            price = price_signal(self.one_day, type_of_DR=type_of_DR)
            price = np.array(price[8:18])
            if np.mean(price) == price[2]:
                print("Given constant price signal")
                price[3:6] += 0.3
            price = np.maximum(0.01 * np.ones_like(price), price)
            for i in range(365):
                all_prices.append(price)
        else:
            print("All days")
            for day in range(1, 366):
                price = price_signal(day, type_of_DR=type_of_DR)
                price = np.array(price[8:18])
                # put a floor on the prices so we don't have negative prices
                if np.mean(price) == price[2]:
                    print("Given constant price signal")
                    price[3:6] += 0.3
                price = np.maximum(0.01 * np.ones_like(price), price)
                all_prices.append(price)

        return np.array(all_prices)

    def _points_from_action(self, action):
        """
        Purpose: Convert agent actions into incentives (conversion is for multidiscrete setting)

        Args:
            Action: 10-dim vector corresponding to action for each hour 8AM - 5PM

        Returns: Points: 10-dim vector of incentives for game (same incentive for each player)
        """
        if self.action_space_string == "multidiscrete":
            #Mapping 0 -> 0.0, 1 -> 5.0, 2 -> 10.0
            points = action * (10 / (self.action_subspace -1))
        elif self.action_space_string == 'continuous':
            #Continuous space is symmetric [-1,1], we map to -> [0,10] by adding 1 and multiplying by 5
            points = 5 * (action + np.ones_like(action))

        elif self.action_space_string == "continuous_normalized":
            points = 10 * (action / np.sum(action))

        return points

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

            if (self.day_of_week_flag):
                player_energy = player.get_response(action, day_of_week = self.day_of_week)
            else:
                player_energy = player.get_response(action, day_of_week = None)

            #Calculate energy consumption by player and in total (over the office)
            energy_consumptions[player_name] = player_energy
            total_consumption += player_energy

        energy_consumptions["avg"] = total_consumption / self.number_of_participants
        return energy_consumptions

    def _get_reward(self, price, energy_consumptions, reward_function = "log_cost_regularized"):
        """
        Purpose: Compute reward given price signal and energy consumption of the office

        Args:
            Price: Price signal vector (10-dim)
            Energy_consumption: Dictionary containing energy usage by player in the office and the average office energy usage

        Returns:
            Energy_consumption: Dictionary containing the energy usage by player and the average energy used in the office (key = "avg")
            TODO: Does it actually return that?
        """

        total_energy_reward = 0
        total_smirl_reward = 0
        total_energy_cost = 0
        for player_name in energy_consumptions:
            if player_name != "avg":
                # get the points output from players
                player = self.player_dict[player_name]

                # get the reward from the player's output
                player_min_demand = player.get_min_demand()
                player_max_demand = player.get_max_demand()
                player_energy = energy_consumptions[player_name]
                player_energy_cost = np.dot(player_energy, price)
                player_reward = Reward(player_energy, price, player_min_demand, player_max_demand)
                if reward_function == "scaled_cost_distance":
                   player_ideal_demands = player_reward.ideal_use_calculation()
                   reward = player_reward.scaled_cost_distance(player_ideal_demands)

                elif reward_function == "log_cost_regularized":
                   reward = player_reward.log_cost_regulariczed()

                elif reward_function == "log_cost":
                    reward = player_reward.log_cost()

                else:
                    print("Reward function not recognized")
                    raise AssertionError

                total_energy_reward += reward
                total_energy_cost += player_energy_cost
        total_energy_reward = total_energy_reward / self.number_of_participants

        if self.use_smirl:
            smirl_rew = self.buffer.logprob(self._get_observation())
            total_smirl_reward = self.smirl_weight * np.clip(smirl_rew, -300, 300)

        self.last_smirl_reward = total_smirl_reward
        self.last_energy_reward = total_energy_reward
        self.last_energy_cost = 500 * (total_energy_cost / self.number_of_participants) * 0.001 # 500 hardcoded for now
        if self.add_last_cost:
            self.costs.append(self.last_energy_cost)
        return total_energy_reward + total_smirl_reward

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
        self.action = action

        if not self.action_space.contains(action):
            print("made it within the if statement in SG_E that tests if the the action space doesn't have the action")
            action = np.asarray(action)
            if self.action_space_string == 'continuous':
                action = np.clip(action, -1, 1) #TODO: check if correct

            elif self.action_space_string == 'multidiscrete':
                action = np.clip(action, 0, self.action_subspace - 1)

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1
        self.total_iter +=1
        print(self.total_iter)
        done = self.curr_iter > 0

        points = self._points_from_action(action)

        energy_consumptions = self._simulate_humans(points)

        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same

        self.prev_energy = energy_consumptions["avg"]

        observation = self._get_observation()
        reward = self._get_reward(prev_price, energy_consumptions, reward_function = self.reward_function)

        if self.use_smirl:
            self.buffer.add(observation)

        info = {}
        return observation, reward, done, info

    def _get_observation(self):
        prev_price = self.prices[ (self.day - 1) % 365]
        next_price = self.prices[self.day]

        if self.bin_observation_space:
            self.prev_energy = np.round(self.prev_energy, -1)

        next_observation = np.array([])

        if self.price_in_state:
            next_observation = np.concatenate((next_observation, next_price))

        if self.energy_in_state:
            next_observation = np.concatenate((next_observation, self.prev_energy))

        return next_observation

    def reset(self):
        """ Resets the environment on the current day """
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass


    def check_valid_init_inputs(self, action_space_string: str, response_type_string: str, number_of_participants = 10,
                one_day = False, price_in_state = False, energy_in_state = False):

        """
        Purpose: Verify that all initialization variables are valid

        Args (from initialization):
            action_space_string: String either "continuous" or "discrete" ; Denotes the type of action space
            response_type_string: String either "t", "s", "l" , denoting whether the office's response function is threshold, sinusoidal, or linear
            number_of_participants: Int denoting the number of players in the social game (must be > 0 and < 20)
            one_day: Boolean denoting whether (or not) the environment is FIXED on ONE price signal
            price_in_state: (Boolean) denoting whether (or not) to include the current grid price in the state
            energy_in_state: (Boolean) denoting whether (or not) to include the energy usage in the state

        Exceptions:
            Raises AssertionError if action_space_string is not a String or if it is not either "continuous", or "multidiscrete"
            Raises AssertionError if response_type_string is not a String or it is is not either "t","s","l"
            Raises AssertionError if number_of_participants is not an integer, is less than 1,  or greater than 20 (upper bound set arbitrarily for comp. purposes).
            Raises AssertionError if any of {one_day, price_in_state, energy_in_state} is not a Boolean
        """

        #Checking that action_space_string is valid
        assert isinstance(action_space_string, str), "action_space_str is not of type String. Instead got type {}".format(type(action_space_string))
        action_space_string = action_space_string.lower()
        assert action_space_string in ["continuous", "multidiscrete", "continuous_normalized"], "action_space_str is not continuous or discrete. Instead got value {}".format(action_space_string)

        #Checking that response_type_string is valid
        assert isinstance(response_type_string, str), "Variable response_type_string should be of type String. Instead got type {}".format(type(response_type_string))
        response_type_string = response_type_string.lower()
        assert response_type_string in ["t", "s", "l"], "Variable response_type_string should be either t, s, l. Instead got value {}".format(response_type_string)


        #Checking that number_of_participants is valid
        assert isinstance(number_of_participants, int), "Variable number_of_participants is not of type Integer. Instead got type {}".format(type(number_of_participants))
        assert number_of_participants > 0, "Variable number_of_participants should be atleast 1, got number_of_participants = {}".format(number_of_participants)
        assert number_of_participants <= 20, "Variable number_of_participants should not be greater than 20, got number_of_participants = {}".format(number_of_participants)

        #Checking that one_day is valid
        assert isinstance(one_day, int), "Variable one_day is not of type Int. Instead got type {}".format(type(one_day))
        assert 366 > one_day and one_day > -2, "Variable one_day out of range [-1,365]. Got one_day = {}".format(one_day)

        #Checking that price_in_state is valid
        assert isinstance(price_in_state, bool), "Variable one_day is not of type Boolean. Instead got type {}".format(type(price_in_state))

        #Checking that yesterday_in_state is valid
        assert isinstance(energy_in_state, bool), "Variable one_day is not of type Boolean. Instead got type {}".format(type(energy_in_state))
        print("all inputs valid")

class SocialGameEnvRLLib(SocialGameEnv):
    def __init__(self, env_config):
        super().__init__(
            action_space_string = env_config["action_space_string"],
            response_type_string = env_config["response_type_string"],
            number_of_participants = env_config["number_of_participants"],
            one_day = env_config["one_day"],
            price_in_state= env_config["price_in_state"],
            energy_in_state = env_config["energy_in_state"],
            pricing_type=env_config["pricing_type"],
            reward_function = env_config["reward_function"],
            bin_observation_space=env_config["bin_observation_space"],
            manual_tou_magnitude=env_config["manual_tou_magnitude"],
            person_type_string=env_config["person_type_string"],
            smirl_weight=env_config["smirl_weight"],
            circ_buffer_size=env_config["circ_buffer_size"]
        )
        print("Initialized RLLib child class")

class SocialGameMetaEnv(SocialGameEnvRLLib):

    def __init__(self,
        env_config,
        task = None):

#        self.goal_direction = goal_direction if goal_direction else 1.0

        self.task = (task if task else {
            "person_type":np.random.choice([DeterministicFunctionPerson, CurtailAndShiftPerson]),
            "points_multiplier":np.random.choice(range(20)),
            "response":np.random.choice(['t','l', 's']),
            "shiftable_load_frac":np.random.uniform(0, 1),
            "curtailable_load_frac":np.random.uniform(0, 1),
            "shiftByHours":np.random.choice(range(8), ),
            "maxCurtailHours":np.random.choice(range(8),)
        })

        super().__init__(
            env_config=env_config,
        )

        self.hours_in_day = 10

    def sample_tasks(self, n_tasks):
        """
        n_tasks will be passed in as a hyperparameter
        """

        person_type = np.random.choice([DeterministicFunctionPerson, CurtailAndShiftPerson], size = (n_tasks, ))
        points_multiplier = np.random.choice(range(20), size = (n_tasks, ))
        response = np.random.choice(['t','l', 's'], size = (n_tasks, ))
        shiftable_load_frac = np.random.uniform(0, 1, size = (n_tasks, ))
        curtailable_load_frac = np.random.uniform(0, 1, size = (n_tasks, ))
        shiftByHours = np.random.choice(range(8), (n_tasks, ))
        maxCurtailHours=np.random.choice(range(8), (n_tasks, ))

        task_parameters = {
            "person_type":person_type,
            "points_multiplier":points_multiplier,
            "response":response,
            "shiftable_load_frac":shiftable_load_frac,
            "curtailable_load_frac":curtailable_load_frac,
            "shiftByHours":shiftByHours,
            "maxCurtailHours":maxCurtailHours
        }

        tasks_dicts = []
        for i in range(n_tasks):
            temp_dict = {k: v[i] for k, v in task_parameters.items()}
            tasks_dicts.append(temp_dict)

        return tasks_dicts


    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.task=task
        self._create_agents()

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.task


    def _create_agents(self):
        """
        Purpose: Create the participants of the social game. We create a game with n players, where n = self.number_of_participants
        This function has been modified to create a variety of people environments to work with MAML

        Args:
            None

        Returns:
              agent_dict: Dictionary of players, each with response function based on self.response_type_string

        """

        player_dict = {}

        # Sample Energy from average energy in the office (pre-treatment) from the last experiment
        # Reference: Lucas Spangher, et al. Engineering  vs.  ambient  typevisualizations:  Quantifying effects of different data visualizations on energy consumption. 2019

        sample_energy = np.array([ 0.28,  11.9,   16.34,  16.8,  17.43,  16.15,  16.23,  15.88,  15.09,  35.6,
                                123.5,  148.7,  158.49, 149.13, 159.32, 157.62, 158.8,  156.49, 147.04,  70.76,
                                42.87,  23.13,  22.52,  16.8 ])

        #only grab working hours (8am - 5pm)
        working_hour_energy = sample_energy[8:18] ### this needs to be changed for 18 hours!!!!!
        my_baseline_energy = pd.DataFrame(data = {"net_energy_use" : working_hour_energy})
        for i in range(self.number_of_participants):
            player = self.task["person_type"](baseline_energy_df = my_baseline_energy, **self.task)
            player_dict['player_{}'.format(i)] = player

        return player_dict


class SocialGameEnvRLLibPlanning(SocialGameEnvRLLib):
    def __init__(self, env_config):
        self.planning_steps = env_config["planning_steps"]
        self.decay = env_config["dagger_decay"]
        self.is_step_in_real=True
        self.planning_step_cnt = 0
        self.num_real_steps = 0
        self.orig_planning_steps = self.planning_steps
        self.oracle_noise = env_config["oracle_noise"]
        self.oracle_noise_type = env_config["oracle_noise_type"]
        self.planning_model_path = env_config["planning_ckpt"]
        self.planning_type=env_config["planning_model"]
        self.planning_delay = env_config["planning_delay"]
        self.intrinsic_reward = env_config["intrinsic_reward"]
        self.predicted_costs = []
        if self.planning_type == "OLS" or self.num_real_steps < self.planning_delay:
            print("Using OLS planning model")
            self.planning_model = self.OLS_model
        elif self.planning_type == "ANN":
            self.swap_to_ANN()
        elif self.planning_type == "Noisy_Oracle":
            self.planning_model = self.NoisyOracle
        super().__init__(
            env_config=env_config,
        )


    def swap_to_ANN(self):
        print("Using ANN planning model")
        #self.planning_net = Net(10, 32, 10)
        #self.planning_net.load_state_dict(torch.load(self.planning_model_path))
        self.planning_net = EnsembleModule.load_from_checkpoint(self.planning_model_path, n_feature=10, n_output=10, n_hidden=64, n_layers=5, n_networks=24, lr=3e-4).model
        # Put loaded planning model into evaluation mode
        self.planning_net.eval()
        def model_wrapper_fn(action):
            out, std = self.planning_net(
                torch.tensor(
                    action.reshape([-1, 10])))
            out = out.detach().numpy().flatten()
            return out, std
        self.planning_model = model_wrapper_fn

    def OLS_model(self, action):
        return 86 + (5 * (action - 5))

    def Oracle(self, action):
        return self._simulate_humans(action)
    
    def NoisyOracle(self, action):
        orig_consumptions = self.Oracle(action)
        if self.oracle_noise_type=="normal":
            output = orig_consumptions["avg"] + np.random.randn(10) * self.oracle_noise
        elif self.oracle_noise_type=="uniform":
            output = orig_consumptions["avg"] + np.random.uniform(low=-1.0, high=1.0, size=(10,)) * self.oracle_noise
        output[output <= 0] = 1e-8 + np.random.uniform(low=1e-8, high=1e-9, size=output[output <= 0].shape)
        return output
        
    def _simulate_humans_planning_model(self, action):
        """
        Purpose: A planning model to wrap simulate_humans. 

        Args:
            Action: 10-dim vector corresponding to action for each hour 

        Returns:
            Energy_consumption: Dictionary containing the energy usage by player and the average energy 
        """
        print("using planning model")
        energy_consumptions = {}
        total_consumption = np.zeros(10)

        for player_name in self.player_dict:
            #Get players response to agent's actions
            player = self.player_dict[player_name]
            if self.planning_type == "ANN":
                player_energy, std = self.planning_model(action)
                self.last_std = std
            else:
                player_energy = self.planning_model(action)
 
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
        
        self.action = action

        if not self.action_space.contains(action):
            print("made it within the if statement in SG_E that tests if the action space doesn't have the action")
            action = np.asarray(action)
            if self.action_space_string == 'continuous':
                action = np.clip(action, -1, 1) #TODO: check if correct

            elif self.action_space_string == 'multidiscrete':
                action = np.clip(action, 0, self.action_subspace - 1)

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1
        self.total_iter +=1

        done = self.curr_iter > 0

        points = self._points_from_action(action)

        if any([self.planning_steps < 1, self.planning_step_cnt > self.planning_steps, self.num_real_steps <= self.planning_delay]):
            # take a step in real
            self.is_step_in_real = True
            self.num_real_steps += 1
            self.planning_step_cnt = 0
            energy_consumptions = self._simulate_humans(points)
            print("using real steps")
            if self.num_real_steps == self.planning_delay:
                self.swap_to_ANN()
                self.planning_steps = self.orig_planning_steps
        else: 
            self.planning_step_cnt += 1
            # take a step in planning
            self.is_step_in_real = False
            energy_consumptions = self._simulate_humans_planning_model(points)

        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same

        self.prev_energy = energy_consumptions["avg"]

        observation = self._get_observation()
        reward = self._get_reward(prev_price, energy_consumptions, reward_function = self.reward_function)

        if self.use_smirl:
            self.buffer.add(observation)

        info = {}
        return observation, reward, done, info

    def decay_ratio(self):
        """Decays the ratio of planning steps to target steps, meant to be used between calls to train()"""
        self.planning_steps *= self.decay

class SocialGameEnvRLLibExtreme(SocialGameEnvRLLibPlanning):
    def __init__(self, env_config):
        self.ma_smoothing = env_config["MA_smoothing"]
        self.extreme_intervention_rarity = env_config["extreme_intervention_rarity"]
        super().__init__(env_config)

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
        
        self.action = action

        if not self.action_space.contains(action):
            print("made it within the if statement in SG_E that tests if the action space doesn't have the action")
            action = np.asarray(action)
            if self.action_space_string == 'continuous':
                action = np.clip(action, -1, 1) #TODO: check if correct

            elif self.action_space_string == 'multidiscrete':
                action = np.clip(action, 0, self.action_subspace - 1)

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1
        self.total_iter +=1

        done = self.curr_iter > 0

        points = self._points_from_action(action)
        predicted_energy_consumptions = self._simulate_humans_planning_model(points)
        predicted_energy_cost = np.dot(predicted_energy_consumptions["avg"], prev_price) * 0.001 * 500
        self.predicted_costs.append(predicted_energy_cost)
        smooth_param = min(self.ma_smoothing, len(self.costs))
        #avg_energy_costs = np.array(self.costs[-smooth_param:])
        avg_energy_costs = np.array(self.predicted_costs[-smooth_param:])
        avg_energy_cost = avg_energy_costs.mean()
        std_energy_cost = avg_energy_costs.std()
        tou = 0.103 * np.ones((10))
        tou[5:8] = self.manual_tou_magnitude
        tou_points = self._points_from_action(tou)
        self.record_last_cost = True
        if any([self.planning_steps < 1, self.planning_step_cnt > self.planning_steps, self.num_real_steps <= self.planning_delay]):
            # take a step in real
            self.is_step_in_real = True
            self.num_real_steps += 1
            self.planning_step_cnt = 0
            if self.num_real_steps > max(self.ma_smoothing, 5) and predicted_energy_cost > avg_energy_cost + self.extreme_intervention_rarity * std_energy_cost:
                self.is_step_in_real = False
                self.add_last_cost = False
                #import pdb; pdb.set_trace()
                energy_consumptions = predicted_energy_consumptions
                # Send TOU pricing to target environment instead
                tou_energy_consumptions = self._simulate_humans(tou_points)
                reward = self._get_reward(prev_price, tou_energy_consumptions, reward_function=self.reward_function)
                self.record_last_cost = False # make this cost get logged instead of the one from the planning model
                print("intervention activated. Predicted: {} vs Avg: {}".format(predicted_energy_cost, avg_energy_cost))
            else:
                self.add_last_cost = True
                energy_consumptions = self._simulate_humans(points)
                print("using real steps")
            if self.num_real_steps == self.planning_delay:
                self.swap_to_ANN()
                self.planning_steps = self.orig_planning_steps
        else: 
            self.planning_step_cnt += 1
            # take a step in planning
            self.is_step_in_real = False
            energy_consumptions = self._simulate_humans_planning_model(points)

        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same

        self.prev_energy = energy_consumptions["avg"]

        observation = self._get_observation()
        reward = self._get_reward(prev_price, energy_consumptions, reward_function = self.reward_function)
        if self.use_smirl:
            self.buffer.add(observation)

        info = {}
        return observation, reward, done, info

    def _get_reward(self, price, energy_consumptions, reward_function = "log_cost_regularized"):
        """
        Purpose: Compute reward given price signal and energy consumption of the office

        Args:
            Price: Price signal vector (10-dim)
            Energy_consumption: Dictionary containing energy usage by player in the office and the average office energy usage

        Returns:
            Energy_consumption: Dictionary containing the energy usage by player and the average energy used in the office (key = "avg")
            TODO: Does it actually return that?
        """

        total_energy_reward = 0
        total_smirl_reward = 0
        total_energy_cost = 0
        for player_name in energy_consumptions:
            if player_name != "avg":
                # get the points output from players
                player = self.player_dict[player_name]

                # get the reward from the player's output
                player_min_demand = player.get_min_demand()
                player_max_demand = player.get_max_demand()
                player_energy = energy_consumptions[player_name]
                player_energy_cost = np.dot(player_energy, price)
                player_reward = Reward(player_energy, price, player_min_demand, player_max_demand)
                if reward_function == "scaled_cost_distance":
                   player_ideal_demands = player_reward.ideal_use_calculation()
                   reward = player_reward.scaled_cost_distance(player_ideal_demands)

                elif reward_function == "log_cost_regularized":
                   reward = player_reward.log_cost_regularized()

                elif reward_function == "log_cost":
                    reward = player_reward.log_cost()

                else:
                    print("Reward function not recognized")
                    raise AssertionError

                total_energy_reward += reward
                total_energy_cost += player_energy_cost
        total_energy_reward = total_energy_reward / self.number_of_participants

        if self.use_smirl:
            smirl_rew = self.buffer.logprob(self._get_observation())
            total_smirl_reward = self.smirl_weight * np.clip(smirl_rew, -300, 300)

        
        self.last_smirl_reward = total_smirl_reward
        self.last_energy_reward = total_energy_reward
        if self.record_last_cost:
            self.last_energy_cost = 500 * (total_energy_cost / self.number_of_participants) * 0.001 # 500 hardcoded for now
        if self.add_last_cost:
            self.costs.append(self.last_energy_cost)
        return total_energy_reward + total_smirl_reward


class SocialGameEnvRLLibGuardRail(SocialGameEnvRLLibPlanning):
    def __init__(self, env_config):
        print("Using Guardrails")
        self.predicted_costs = []
        self.last_predicted_cost = 0
        super().__init__(env_config)

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
        
        self.action = action

        if not self.action_space.contains(action):
            print("made it within the if statement in SG_E that tests if the action space doesn't have the action")
            action = np.asarray(action)
            if self.action_space_string == 'continuous':
                action = np.clip(action, -1, 1) #TODO: check if correct

            elif self.action_space_string == 'multidiscrete':
                action = np.clip(action, 0, self.action_subspace - 1)

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1
        self.total_iter +=1

        done = self.curr_iter > 0

        points = self._points_from_action(action)
        predicted_energy_consumptions = self._simulate_humans_planning_model(points)
        predicted_energy_cost = np.dot(predicted_energy_consumptions["avg"], prev_price) * 0.001 * 500
        self.last_predicted_cost = predicted_energy_cost
        self.predicted_costs.append(predicted_energy_cost)
        tou = 0.103 * np.ones((10))
        tou[5:8] = self.manual_tou_magnitude
        tou_points = self._points_from_action(tou)
        tou_cost = 75
        if predicted_energy_cost < tou_cost:
            # take a step in real
            self.is_step_in_real = True
            self.num_real_steps += 1
            self.planning_step_cnt = 0
            energy_consumptions = self._simulate_humans(points)
            print("using real steps")
            if self.num_real_steps == self.planning_delay:
                self.swap_to_ANN()
                self.planning_steps = self.orig_planning_steps
        else: 
            self.planning_step_cnt += 1
            # take a step in planning
            self.is_step_in_real = False
            energy_consumptions = self._simulate_humans_planning_model(points)

        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same

        self.prev_energy = energy_consumptions["avg"]

        observation = self._get_observation()
        reward = self._get_reward(prev_price, energy_consumptions, reward_function = self.reward_function)

        if self.use_smirl:
            self.buffer.add(observation)

        info = {}
        return observation, reward, done, info


class SocialGameEnvRLLibIntrinsicMotivation(SocialGameEnvRLLibPlanning):
    def __init__(self, env_config):
        self.planning_type = "ANN"
        super().__init__(
            env_config=env_config
        )
        self.stds = []
        self.intrinsic_motivation_step = 0
        self.total_instrinsic_steps = env_config["total_intrinsic_steps"] ## need to set this
        #self.last_predicted_cost = 1
        #IPython.embed()

    def _torch_mean_to_numpy(self, torch_array):
        """ take the mean of a torch tensor and return as np array

        Args:
            torch_array: 2-d torch array
        
        Returns:
            mu: mean
        """
        mu = torch.mean(torch_array[0]).detach().numpy()
        if isnan(mu):
            print("---------------------")
            print("planning model mu is nan")
            # IPython.embed()
            mu = 0
        return mu
    
    def _simulate_humans_planning_model(self, action):
        """
        Purpose: A planning model to wrap simulate_humans. 

        Args:
            Action: 10-dim vector corresponding to action for each hour 

        Returns:
            Energy_consumption: Dictionary containing the energy usage by player and the average energy 
        """
        print("using planning model")
        energy_consumptions = {}
        total_consumption = np.zeros(10)

        for player_name in self.player_dict:
            #Get players response to agent's actions
            player = self.player_dict[player_name]
            if self.planning_type == "ANN":
                player_energy, std = self.planning_model(action)
                self.last_std = std
                self.stds.append(self._torch_mean_to_numpy(std))
            else:
                player_energy = self.planning_model(action)
 
            #Calculate energy consumption by player and in total (over the office)
            energy_consumptions[player_name] = player_energy
            total_consumption += player_energy

        if self.intrinsic_reward=="curiosity_mean" or self.intrinsic_reward == "apt":
            intrinsic_reward = np.mean(self.stds)
        elif self.intrinsic_reward =="curiosity_max":
            intrinsic_reward = np.max(self.stds)
        elif self.intrinsic_reward == "curiosity_l2_norm":
            intrinsic_reward = np.linalg.norm(self.stds, ord = 2)
        else:
            print("Wrong intrinsic reward")
            raise AssertionError

        energy_consumptions["avg"] = total_consumption / self.number_of_participants
        return energy_consumptions, intrinsic_reward


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
        
        self.action = action

        if not self.action_space.contains(action):
            print("made it within the if statement in SG_E that tests if the action space doesn't have the action")
            action = np.asarray(action)
            if self.action_space_string == 'continuous':
                action = np.clip(action, -1, 1) #TODO: check if correct

            elif self.action_space_string == 'multidiscrete':
                action = np.clip(action, 0, self.action_subspace - 1)

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1
        self.total_iter +=1

        done = self.curr_iter > 0

        points = self._points_from_action(action)

        if self.intrinsic_motivation_step > self.total_instrinsic_steps:
            # take a step in real
            self.is_step_in_real = True
            self.num_real_steps += 1
            self.planning_step_cnt = 0
            energy_consumptions = self._simulate_humans(points)
            print("using real steps")
            reward = self._get_reward(prev_price, energy_consumptions, reward_function = self.reward_function)

        else: 
            self.intrinsic_motivation_step += 1
            # take a step in planning
            self.is_step_in_real = False
            energy_consumptions, intrinsic_reward = self._simulate_humans_planning_model(points)
            reward = intrinsic_reward

            if self.intrinsic_reward=="apt":
                intrinsic_rewards = [intrinsic_reward]
                for i in range(10):
                    temp_points = points + np.random.uniform(
                        low = -.01,
                        high = .01,
                        size = 10)
                    temp_points = np.clip(temp_points, 0, 10)
                    energy_consumptions, intrinsic_reward = self._simulate_humans_planning_model(temp_points)
                    intrinsic_rewards.append(intrinsic_reward)
                
                reward = np.mean(intrinsic_rewards)

            print("std reward mean")
            print(reward)

        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same

        self.prev_energy = energy_consumptions["avg"]

        observation = self._get_observation()
        
        if self.use_smirl:
            self.buffer.add(observation)

        info = {}

        return observation, reward, done, info
