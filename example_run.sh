python3 rl_algos/StableBaselines.py --num_steps=300000 --exp_name=baseline_ppo --algo=ppo --library=rllib --one_day=15 --energy_in_state=T --price_in_state=F
python3 rl_algos/StableBaselines.py --algo=maml --library=rllib -w --maml_inner_lr=0.01 --maml_inner_adaptation_steps=50 --maml_num_workers=4 --maml_vf_clip_param=100.0 --maml_outer_lr=1e-4 --checkpoint_interval=10 --num_steps=500 --maml_optimizer_steps=1
