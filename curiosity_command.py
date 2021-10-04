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
    3: "curiosity_max"
    }

# IPython.embed()

command = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --checkpoint_interval=50 --gym_env=curiosity --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame".format(rew[args.intrinsic_rew], args.steps)

os.system(command)