#!/bin/bash
# Job name:
#SBATCH --job-name=socialgame_train
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2
#
# Number of nodes:
#SBATCH --nodes=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
# Wall clock limit (2hrs):
#SBATCH --time=01:00:00

singularity exec --nv --workdir ./tmp --bind $(pwd):$HOME library://aphoh/default/sg-k80-env:v1 sh -c './singularity_preamble.sh && ./planning_run.sh'
