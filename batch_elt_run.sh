#!/bin/bash

export WANDB_RUN_GROUP="batch_run"

python3 rl_algos/StableBaselines.py -w --num_steps=300000 --exp_name=smirl_$1_ppo --algo=ppo --library=rllib --one_day=15 --smirl_weight=$1 --energy_in_state=T --price_in_state=F --base_log_dir=$2

exit

