#!/bin/bash -l

#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --job-name=mri_sr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --constraint="h100|a100-80gb"
#SBATCH --cpus-per-task=16
#SBATCH --exclude=workergpu027,workergpu047

GPUS=4
set -euo pipefail

cd /mnt/home/mgoldstein/athenak_flatiron
module load python/3.11.11

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
