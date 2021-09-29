import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--extreme_intervention_rarity', 
                    type=float,
                    default=0,
                    help='Rarity of extreme intervention (# of stds above rolling average)')
args = parser.parse_args()
ckpt_folders = "planning_ckpts_diverse2/"
num_datas = os.listdir(ckpt_folders)
base_command = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=10000 --algo=sac --checkpoint_interval=50 --gym_env=curiosity --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0 --dagger_decay=1 --planning_model=ANN --planning_ckpt={} --exp_name=curiosity_5000steps_{}planning --planning_num_data={} --tou_replacement --total_intrinsic_steps=5000"
base_command = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=10000 --algo=sac --checkpoint_interval=50 --gym_env=curiosity --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0 --dagger_decay=1 --planning_model=ANN --planning_ckpt={} --exp_name=curiosity_15000steps_{}planning --planning_num_data={} --tou_replacement --total_intrinsic_steps=15000"
base_command = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=10000 --algo=sac --checkpoint_interval=50 --gym_env=curiosity --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0 --dagger_decay=1 --planning_model=ANN --planning_ckpt={} --exp_name=curiosity_25000steps_{}planning --planning_num_data={} --tou_replacement --total_intrinsic_steps=25000"
base_command = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=10000 --algo=sac --checkpoint_interval=50 --gym_env=curiosity --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0 --dagger_decay=1 --planning_model=ANN --planning_ckpt={} --exp_name=curiosity_35000steps_{}planning --planning_num_data={} --tou_replacement --total_intrinsic_steps=35000"
for num_data in num_datas:
    ckpt_folder = os.path.join(ckpt_folders, num_data)
    ckpt_name = os.listdir(ckpt_folder)[0]
    ckpt_path = os.path.join(ckpt_folder, ckpt_name)
    command = base_command.format(ckpt_path, num_data, num_data)
    os.system(command)