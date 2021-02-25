# transactive_control
Code meant to support and simulate the Social Game that will be launched in 2020. Elements of transactive control and behavioral engineering will be tested and designed here 

# 12/20/2020

This repository has been cleaned and updated for use. It contains: (1) The OpenAI gym environment "OfficeLearn", in the "gym-socialgame" folder, and (2) implementations of Reinforcement learning algorithms in "rl_algos" folder. In the "simulations" folder are various datasets for setting up and training models associated with the gym simulation. 

# 9/1/2020

Here is how to instantite, train, and execute the RL agent. (0) Make sure you have python3.6 installed, and (1) Clone this repo, (2) Navigate to the rl_algos/ directory, then run the python command for the vanilla version: 

python StableBaselines.py sac

Of course, you will need to install python packages required. Adding in the planning model can be done with the following flags:

python StableBaselines.py sac --planning_steps=10 --planning_model=Oracle --num_steps=10000

Please see transactive_control/gym-socialgame/gym_socialgame/envs for files pertaining to the setup of the environment. The socialgame_env.py contains a lot of the information necessary for understanding how the agent steps through the environment. The reward.py file contains information on the variety of reward types available for testing. agents.py contains information on the deterministic people that we created for our simulation. 

-->  gym-socialgame
OpenAI Gym environment for a social game.

--> Usage
This environment can be directly plugged into existing OpenAI-Gym compatible RL libraries (e.g. [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html)).
For this case, try the following:
    
    pip install -e .

From the gym-socialgame/gym-socialgame directory. Otherwise, this package can be used as a standalone environment for other RL algorithms.
