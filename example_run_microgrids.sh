python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=pm_net_exports --algo=ppo --library=rllib --one_day=15 --energy_in_state=T --price_in_state=F --gym_env=microgrid -w --reward_function=profit_maximizing --two_price_state=T

python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=ms_net_exports --algo=ppo --library=rllib --one_day=15 --energy_in_state=T --price_in_state=F --gym_env=microgrid -w --reward_function=market_solving --two_price_state=T
