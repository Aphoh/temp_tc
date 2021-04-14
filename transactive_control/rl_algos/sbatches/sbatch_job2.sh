#!/bin/bash
# Job name:
#SBATCH --job-name=benchmark_sac_planning
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2
#
# Request one node:
#SBATCH --nodes=1
#
# Request cores (24, for example)
#SBATCH --ntasks-per-node=2
#
#Request GPUs
#SBATCH --gres=gpu:0
#
#Request CPU
#SBATCH --cpus-per-task=8
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lucas_spangher@berkeley.edu
## Command(s) to run (example):
module load python/3.6
source /global/home/users/lucas_spangher/transactive_control/auto_keras_env/bin/activate
## OLS
python StableBaselines.py sac --own_tb_log=20200914_SAC_OLS_RTP_one_day --planning_steps=10 --planning_model=OLS --one_day=15 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_OLS_RTP_one_day_2 --planning_steps=10 --planning_model=OLS --one_day=15 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_OLS_RTP_multi_day --planning_steps=10 --planning_model=OLS --one_day=-1 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_OLS_RTP_multi_day_2 --planning_steps=10 --planning_model=OLS --one_day=-1 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_OLS_TOU_one_day --planning_steps=10 --planning_model=OLS --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_OLS_TOU_one_day_2 --planning_steps=10 --planning_model=OLS --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_OLS_TOU_multi_day --planning_steps=10 --planning_model=OLS --one_day=-1 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_OLS_TOU_multi_day_2 --planning_steps=10 --planning_model=OLS --one_day=-1 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
## Baseline
python StableBaselines.py sac --own_tb_log=20200914_SAC_Baseline_RTP_one_day --planning_steps=10 --planning_model=Baseline --one_day=15 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_Baseline_RTP_one_day_2 --planning_steps=10 --planning_model=Baseline --one_day=15 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_Baseline_RTP_multi_day --planning_steps=10 --planning_model=Baseline --one_day=-1 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_Baseline_RTP_multi_day_2 --planning_steps=10 --planning_model=Baseline --one_day=-1 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_Baseline_TOU_one_day --planning_steps=10 --planning_model=Baseline --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_Baseline_TOU_one_day_2 --planning_steps=10 --planning_model=Baseline --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_Baseline_TOU_multi_day --planning_steps=10 --planning_model=Baseline --one_day=-1 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_Baseline_TOU_multi_day_2 --planning_steps=10 --planning_model=Baseline --one_day=-1 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
## LSTM
python StableBaselines.py sac --own_tb_log=20200914_SAC_LSTM_RTP_one_day --planning_steps=10 --planning_model=LSTM --one_day=15 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_LSTM_RTP_one_day_2 --planning_steps=10 --planning_model=LSTM --one_day=15 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_LSTM_RTP_multi_day --planning_steps=10 --planning_model=LSTM --one_day=-1 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_LSTM_RTP_multi_day_2 --planning_steps=10 --planning_model=LSTM --one_day=-1 --num_players=10 --pricing_type=RTP --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_LSTM_TOU_one_day --planning_steps=10 --planning_model=LSTM --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_LSTM_TOU_one_day_2 --planning_steps=10 --planning_model=LSTM --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_LSTM_TOU_multi_day --planning_steps=10 --planning_model=LSTM --one_day=-1 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
python StableBaselines.py sac --own_tb_log=20200914_SAC_LSTM_TOU_multi_day_2 --planning_steps=10 --planning_model=LSTM --one_day=-1 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=T
