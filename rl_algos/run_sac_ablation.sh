#!/bin/bash
for A in 1 2 4 6 8 10  # used to be 200
do
    B=$(($A * 1))
    rl_algos/build_dataset.sh $B # test in increments of 25
    python rl_algos/StableBaselines.py -w --num_steps=250 --exp_name=sac_offline.1.0_nosmirl --algo=sac --library=rllib --offline_sampling_prop=1.0 --offline_data_path="rl_algos/offline_data/sac_ablation_output_sim_data_redo_sin$B" --checkpoint_interval=50 --wandb_run_name="sac_ablation_redo_sin$B"
    rl_algos/run_sac.sh $B "sac_ablation_redo_sin$B"

done