import os
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
ckpt_folders = "planning_ckpts_diverse3/"
num_datas = os.listdir(ckpt_folders)
base_command = "python rl_algos/StableBaselines.py -w --library=rllib --num_gpus=1 --num_steps=300 --algo=sac --checkpoint_interval=50 --gym_env=planning_guardrails --bulk_log_interval=50 --ignore_warnings --base_log_dir=./logs/ --planning_steps=0 --dagger_decay=1 --planning_model=ANN --planning_ckpt={} --exp_name=guardrails_nn_quantile --planning_num_data={} --threshold_type={}"
n_train_datas = [500, 1000, 10000, 30000, 2000, 7500, 15000, 20000, 75000, 100000]
num_nets = [1, 3, 5, 7, 10]
for num_data in n_train_datas:
    for num_net in num_nets:
        ckpt_folder = os.path.join(ckpt_folders, str(num_data))
        ckpt_name = [f for f in os.listdir(ckpt_folder) if os.path.isfile(os.path.join(ckpt_folder, f))][0]
        ckpt_path = os.path.join(ckpt_folder, ckpt_name)
        command = base_command.format(ckpt_path, num_data, num_net)
        os.system(command)