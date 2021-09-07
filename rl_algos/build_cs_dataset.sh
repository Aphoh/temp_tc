#!/bin/bash
RESPONSE_TYPES='s'

python rl_algos/StableBaselines.py --algo=sac --library=rllib  --num_steps=200 --checkpoint_interval=-1 --save_transitions --ignore_warnings --offline_data_path="rl_algos/offline_data/sac_cs_data" 

# for I in {1..3}
# do
#     echo $I
#     python StableBaselines.py --algo=sac --library=rllib -w --num_steps=100 --points_multiplier=$PM --person_type_string=c
# done
