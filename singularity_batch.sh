#!/bin/bash
# Job name:
#SBATCH --job-name=socialgame_train
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit (8hrs):
#SBATCH --time=08:00:00
#
# Run 8 examples concurrently
#SBATCH --array=0-7

VALS=(3.00 0.50 0.10 0.05 0.03 0.003 0.001 0)
SMIRL_VAL=${VALS[$SLURM_ARRAY_TASK_ID]}

singularity exec --nv --workdir ./tmp --bind $(pwd):$HOME library://aphoh/default/sg-k80-env:v1 \
  sh -c './singularity_preamble.sh && ./batch_elt_run.sh $SMIRL_VAL'
