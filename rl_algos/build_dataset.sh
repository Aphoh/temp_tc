#!/bin/bash
RESPONSE_TYPES='s'

for RESPONSE in $RESPONSE_TYPES
do 
    for A in {1..$1} # used to be 200
    do
        PM=$(($RANDOM % 10 + 10))
        echo $PM $RESPONSE
        python rl_algos/StableBaselines.py --algo=sac --library=rllib  --num_steps=1 --person_type_string=d --response_type_string=$RESPONSE --points_multiplier=$PM --checkpoint_interval=-1 --save_transitions --ignore_warnings --offline_data_path="rl_algos/offline_data/sac_ablation_output_sim_data_redo_sin$1" --rllib_train_batch_size=1
    done
done
# for I in {1..3}
# do
#     echo $I
#     python StableBaselines.py --algo=sac --library=rllib -w --num_steps=100 --points_multiplier=$PM --person_type_string=c
# done
