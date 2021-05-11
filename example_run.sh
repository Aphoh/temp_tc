python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=ordinal_test --algo=ppo --library=rllib --one_day=15 --energy_in_state=T --price_in_state=F --action_space=ordinal -w 

python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=multidiscrete_test --algo=ppo --library=rllib --one_day=15 --energy_in_state=T --price_in_state=F -w --action_space_string=d

python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=continuous_test --algo=ppo --library=rllib --one_day=15 --energy_in_state=T --price_in_state=F --action_space_string=c






