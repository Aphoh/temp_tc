for steps in 50 5000 
do
for rew in 3 5 
do
    python curiosity_command.py --intrinsic_rew=$rew --steps=$steps
done
done
