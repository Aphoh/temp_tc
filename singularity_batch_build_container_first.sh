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
#SBATCH --array=0-2

VALS=(0 0.10 0.125)
SMIRL_VAL=${VALS[$SLURM_ARRAY_TASK_ID]}

BASE_DIR=/global/scratch/users/$USER

LDIR=$BASE_DIR/.local$SLURM_ARRAY_TASK_ID
LOGDIR_BASE=$BASE_DIR/logs


rm -rf $LDIR
mkdir -p $LDIR

rm -rf /global/home/users/lucas_spangher/.cache/pip

SINGULARITY_IMAGE_LOCATION=/global/scratch/users/$USER
SINGULARITY_CACHEDIR=$BASE_DIR/.singularity/cache
export SINGULARITY_CACHEDIR=$BASE_DIR/.singularity/cache

SINGULARITY_TEMPDIR=$BASE_DIR/tmp
export SINGULARITY_TEMPDIR=$BASEDIR/tmp

if test -f "$BASE_DIR/lucas_test.sif"; then
    echo "image exists"
else
    singularity build /global/scratch/users/$USER/lucas_test.sif docker://lucasspangher/socialgame_v1
fi
    
singularity exec --nv --workdir ./tmp --bind $(pwd):$HOME \
  --bind "$LDIR:$HOME/.local" \
  /global/scratch/users/$USER/lucas_test.sif \
  sh -c 'bash singularity_preamble.sh && bash intrinsic_curiosity_experiment.sh'
