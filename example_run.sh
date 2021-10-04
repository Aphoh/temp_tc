export WANDB_API_KEY="3cc9567683fdfd761e3a28516c67a053e4b76331"
export WANDB_RUN_GROUP="smirl_run_v2"

python3 rl_algos/StableBaselines.py -w --num_steps=300000 --exp_name=smirl_010_ppo --algo=ppo --library=rllib --one_day=15 --smirl_weight=0.10 --energy_in_state=T --price_in_state=F



