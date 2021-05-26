python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=maml_smirl_ppo --algo=ppo --library=rllib --one_day=15 --smirl_weight=0.10 --energy_in_state=T --price_in_state=F --checkpoint=rl_algos/maml_ckpts/vivid-pyramid-501100.ckpt
python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=baseline_ppo --algo=ppo --library=rllib --one_day=15 --energy_in_state=T --price_in_state=F
python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=smirl_10_ppo --algo=ppo --library=rllib --one_day=15 --smirl_weight=0.10 --energy_in_state=T --price_in_state=F

#python3 rl_algos/StableBaselines.py --exp_name=test --algo=ppo --library=tune --one_day=15 --energy_in_state=T --price_in_state=F



