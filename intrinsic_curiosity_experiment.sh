for steps in 1500 5000 
do
for rew in 0 1 2 3 4 5 6
do
    python curiosity_command.py --intrinsic_rew=$rew --steps=$steps
done
done
