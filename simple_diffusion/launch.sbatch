#!/bin/bash -l

#SBATCH --time=2-00:00:00
#SBATCH --mem=100G
#SBATCH --job-name=mri_sr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --constraint="h100|a100-80gb"
#SBATCH --cpus-per-task=20
#SBATCH --exclude=workergpu027,workergpu047

GPUS=4
set -euo pipefail

cd /mnt/home/mgoldstein/athenak_flatiron

module load python/3.12.9 cuda/12 cudnn
source /mnt/home/mgoldstein/gameflow/.venv/bin/activate

# Ensure logs directory exists
mkdir -p logs

JOBID="${SLURM_JOB_ID:-0}"
MASTER_PORT=$(( 12000 + (JOBID % 20000) ))

# Examples:
#   sbatch simple_diffusion/launch.sbatch
#   sbatch simple_diffusion/launch.sbatch --config simple_diffusion/configs/overfit.yaml
#   sbatch simple_diffusion/launch.sbatch --resume-from /path/to/ckpt_dir

CONFIG="${CONFIG:-simple_diffusion/configs/full_experiment.yaml}"

torchrun --standalone --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
    simple_diffusion/trainer_ddp.py \
    --config "${CONFIG}" \
    "$@"
