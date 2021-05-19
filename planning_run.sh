export WANDB_RUN_GROUP=testing

python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=planning_test --algo=ppo --library=rllib --one_day=15 --energy_in_state=T --price_in_state=F
