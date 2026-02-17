#!/bin/bash -l
#
# Launch MRI 2D 256x256 sims with different random seeds.
# Each SLURM array task runs one seed.
#
# Usage:
#   sbatch --array=0-9 launch256_seeds.sbatch     # seeds 0-9 (10 sims)
#   sbatch --array=0-49 launch256_seeds.sbatch     # seeds 0-49 (50 sims)
#   sbatch --array=5,10,42 launch256_seeds.sbatch  # specific seeds
#

#SBATCH --partition gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:4
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-task 1
#SBATCH --time 2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --job-name=mri256
#SBATCH --exclude=workergpu027,workergpu047
#SBATCH --mem=100G

set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID}
EXE="/mnt/home/mgoldstein/athenak_flatiron/build_mri2d/src/athena"
DECK="/mnt/home/mgoldstein/athenak_flatiron/decks/mri2d.athinput.256"
OUTBASE="/mnt/home/mgoldstein/ceph/athenak/mri256_seeds"
OUTDIR="${OUTBASE}/seed_$(printf '%04d' ${SEED})"

mkdir -p "${OUTDIR}" /mnt/home/mgoldstein/athenak_flatiron/logs

echo "=== MRI 2D 256x256, seed=${SEED}, output=${OUTDIR} ==="

module purge
module load slurm gcc cuda openmpi/cuda-4.1.8

export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

srun --cpus-per-task=${SLURM_CPUS_PER_TASK:-16} --cpu-bind=cores \
  bash -c "unset CUDA_VISIBLE_DEVICES; \
  ${EXE} -i ${DECK} -d ${OUTDIR}/ problem/rand_seed=${SEED}"
