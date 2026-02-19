#!/bin/bash -l

GPUS=1
set -euo pipefail

cd /mnt/home/mgoldstein/athenak_flatiron

# Ensure logs directory exists
mkdir -p logs

JOBID="${SLURM_JOB_ID:-0}"
MASTER_PORT=$(( 12000 + (JOBID % 20000) ))

# Examples:
#   sbatch train_ddp.sbatch --no-overfit --channels velx
#   sbatch train_ddp.sbatch --no-overfit --local-batch-size 16 --microbatch-size 4

torchrun --standalone --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
    simple_diffusion/trainer_ddp.py \
    "$@"
