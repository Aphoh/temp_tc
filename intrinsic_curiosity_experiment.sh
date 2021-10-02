for steps in 5000 10000 15000 20000 25000
do
for rew in 0 1 2 3 
do
    python curiosity_command.py --intrinsic_rew=$rew --steps=$steps
done
done