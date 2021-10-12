import os
import argparse
import IPython

parser = argparse.ArgumentParser()

parser.add_argument('--intrinsic_rew', 
                    type=int,
                    default=0,
                    help='Rarity of extreme intervention (# of stds above rolling average)')

parser.add_argument('--steps', 
                    type=int,
                    default=1000,
                    help='Rarity of extreme intervention (# of stds above rolling average)')

args = parser.parse_args()

rew = {0: "curiosity_mean",
    1: "curiosity_l2_norm",
    2: "apt",
    3: "curiosity_max",
    4: "control",
    5: "higher_percentile",
    6: "intr_extr"
    }

# IPython.embed()

command1 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --reward_function=profit_maximizing --exp_name=gym_microgrid &" 

Command = command1 

os.system(Command)