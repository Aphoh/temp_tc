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

command1 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[0], 1000) 
command2 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[1], 1000)
command3 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[2], 1000)
command4 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[3], 1000)
command5 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[4], 1000)
command6 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[5], 1000)
command7 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[6], 1000)


command11 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[0], 5000)
command12 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[1], 5000)
command13 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[2], 5000)
command14 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[3], 5000)
command15 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[4], 5000)
command16 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame &".format(rew[5], 5000)
command17 = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=50000 --algo=ppo --gym_env=curiosity --energy_in_state=T --base_log_dir=./logs/ --planning_steps=0  --planning_model=ANN --planning_ckpt=planning_ckpts_diverse2/100000/epoch=100-step=4949.ckpt --intrinsic_rew={} --total_intrinsic_steps={} --exp_name=socialgame ".format(rew[6], 5000)


os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)
os.system(command6)
os.system(command7)

os.system(command11)
os.system(command12)
os.system(command13)
os.system(command14)
os.system(command15)
os.system(command16)
os.system(command17)