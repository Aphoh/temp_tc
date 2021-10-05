for steps in 50 5000 
do
for rew in 4 
do
    python curiosity_command.py --intrinsic_rew=$rew --steps=$steps
done
done
