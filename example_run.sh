#python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=baseline_ppo --algo=ppo --library=rllib --one_day=15 --energy_in_state=T --price_in_state=F
#python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=smirl_050_ppo --algo=ppo --library=rllib --one_day=15 --smirl_weight=0.50 --energy_in_state=T --price_in_state=F &
#python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=smirl_010_ppo --algo=ppo --library=rllib --one_day=15 --smirl_weight=0.10 --energy_in_state=T --price_in_state=F &
#python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=smirl_005_ppo --algo=ppo --library=rllib --one_day=15 --smirl_weight=0.05 --energy_in_state=T --price_in_state=F &
#python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=smirl_003_ppo --algo=ppo --library=rllib --one_day=15 --smirl_weight=0.03 --energy_in_state=T --price_in_state=F &
#python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=smirl_0003_ppo --algo=ppo --library=rllib --one_day=15 --smirl_weight=0.003 --energy_in_state=T --price_in_state=F &
python3 rl_algos/StableBaselines.py --exp_name=test --algo=ppo --library=tune --one_day=15 --energy_in_state=T --price_in_state=F



