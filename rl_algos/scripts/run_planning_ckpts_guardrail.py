import os
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
ckpt_folders = "planning_ckpts_diverse3/"
num_datas = os.listdir(ckpt_folders)
base_command = "python rl_algos/StableBaselines.py -w --num_gpus=1 --library=rllib --num_steps=300 --algo=sac --checkpoint_interval=50 --gym_env=planning_guardrails --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0 --dagger_decay=1 --planning_model=ANN --planning_ckpt={} --exp_name=guardrails_nn4 --planning_num_data={}"
for num_data in num_datas:
    ckpt_folder = os.path.join(ckpt_folders, num_data)
    ckpt_name = [f for f in os.listdir(ckpt_folder) if os.path.isfile(os.path.join(ckpt_folder, f))][0]
    ckpt_path = os.path.join(ckpt_folder, ckpt_name)
    command = base_command.format(ckpt_path, num_data)
    os.system(command)