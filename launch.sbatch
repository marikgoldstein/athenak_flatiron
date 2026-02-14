#!/bin/bash -l

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
#SBATCH --job-name=athenak
#SBATCH --exclude=workergpu027,workergpu047
#SBATCH --mem=100G

cd /mnt/home/mgoldstein/athenak_flatiron
EXE="/mnt/home/mgoldstein/athenak_flatiron/build_mri2d/src/athena"
DECK="/mnt/home/mgoldstein/athenak_flatiron/decks/mri2d.athinput.2048"
OUT="/mnt/home/mgoldstein/ceph/athenak/feb13/"
set -euo pipefail

module purge
module load slurm gcc cuda openmpi/cuda-4.1.8

export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

srun --cpus-per-task=${SLURM_CPUS_PER_TASK:-16} --cpu-bind=cores \
  bash -c "unset CUDA_VISIBLE_DEVICES; \
  ${EXE} -i ${DECK} -d ${OUT}"
