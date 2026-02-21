#!/bin/bash -l

GPUS=1
set -euo pipefail

cd /mnt/home/mgoldstein/athenak_flatiron

# Ensure logs directory exists
mkdir -p logs

JOBID="${SLURM_JOB_ID:-0}"
MASTER_PORT=$(( 12000 + (JOBID % 20000) ))

# Examples:
#   ./simple_diffusion/run.sh
#   ./simple_diffusion/run.sh --config simple_diffusion/configs/overfit.yaml
#   ./simple_diffusion/run.sh --resume-from /path/to/ckpt_dir

CONFIG="${CONFIG:-simple_diffusion/configs/full_experiment.yaml}"

torchrun --standalone --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
    simple_diffusion/trainer_ddp.py \
    --config "${CONFIG}" \
    "$@"
