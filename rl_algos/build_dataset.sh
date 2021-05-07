#!/bin/bash
RESPONSE_TYPES='l t s'

for RESPONSE in $RESPONSE_TYPES
do 
    for PM in 10 15 20
    do
        echo $PM $RESPONSE
        python StableBaselines.py --algo=sac --library=rllib -w --num_steps=100 --person_type_string=d --response_type_string=$RESPONSE --points_multiplier=$PM
    done
done
for I in {1..3}
do
    echo $I
    python StableBaselines.py --algo=sac --library=rllib -w --num_steps=100 --points_multiplier=$PM --person_type_string=c
done