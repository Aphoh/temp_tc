import os
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
ckpt_folders = "planning_ckpts_diverse3/"
num_datas = os.listdir(ckpt_folders)
base_command = "python rl_algos/StableBaselines.py -w --num_gpus=1 --library=rllib --num_steps=300 --algo=sac --checkpoint_interval=50 --gym_env=planning_guardrails --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0 --dagger_decay=1 --planning_model=Noisy_Oracle --exp_name=guardrails_oracle4 --oracle_noise={}"
oracle_noise = [20, 60, 90, 120, 140, 140, 160]
for noise in oracle_noise:
    command = base_command.format(noise)
    os.system(command)