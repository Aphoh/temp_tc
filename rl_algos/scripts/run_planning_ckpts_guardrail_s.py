import os
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
ckpt_folders = "planning_ckpts_diverse3/"
num_datas = os.listdir(ckpt_folders)
base_command = "python rl_algos/StableBaselines.py -w --library=rllib --num_steps=300 --algo=sac --checkpoint_interval=50 --num_gpus=1 --gym_env=planning_guardrails --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0 --dagger_decay=1 --planning_model=ANN --planning_ckpt={} --exp_name=guardrails_nn_s --planning_num_data={} --threshold_type=s --thresh_scale={} --thresh_offset={:d}"
n_train_datas = [500, 1000, 10000, 30000, 2000, 7500, 15000, 20000, 75000, 100000]
offsets = [75, 80, 90]
scales = [2, 5, 8]
for offset in offsets:
    for scale in scales:
        for num_data in n_train_datas:
            ckpt_folder = os.path.join(ckpt_folders, str((num_data)))
            ckpt_name = [f for f in os.listdir(ckpt_folder) if os.path.isfile(os.path.join(ckpt_folder, f))][0]
            ckpt_path = os.path.join(ckpt_folder, ckpt_name)
            command = base_command.format(ckpt_path, num_data, scale, offset)
            os.system(command)