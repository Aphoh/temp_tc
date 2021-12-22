import os
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
ckpt_folders = "planning_ckpts_diverse3/"
num_datas = os.listdir(ckpt_folders)
base_command = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=300 --algo=sac --checkpoint_interval=50 --num_gpus=1 --gym_env=planning_guardrails --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0 --dagger_decay=1 --planning_model=Noisy_Oracle --exp_name=guardrails_oracle_s --oracle_noise={} --threshold_type=s --thresh_scale={} --thresh_offset={:d}"
n_train_datas = [500, 1000, 10000, 30000, 2000, 7500, 15000, 20000, 75000, 100000]
offsets = [75, 80, 90]
scales = [2, 5, 8]
oracle_noise = [20, 60, 90, 120, 140, 140, 160]
for offset in offsets:
    for scale in scales:
        for noise in oracle_noise:
            command = base_command.format(noise, scale, offset)
            os.system(command)