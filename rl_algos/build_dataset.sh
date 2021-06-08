#!/bin/bash
RESPONSE_TYPES='l t s'

for RESPONSE in $RESPONSE_TYPES
do 
    for A in {1..200}
    do
        PM=($RANDOM % 10 + 10)
        echo $PM $RESPONSE
        python StableBaselines.py --algo=sac --library=rllib  --num_steps=5 --person_type_string=d --response_type_string=$RESPONSE --points_multiplier=$PM --checkpoint_interval=-1
    done
done
# for I in {1..3}
# do
#     echo $I
#     python StableBaselines.py --algo=sac --library=rllib -w --num_steps=100 --points_multiplier=$PM --person_type_string=c
# done