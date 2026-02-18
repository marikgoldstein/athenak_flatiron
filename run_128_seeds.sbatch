#!/bin/bash -l
#SBATCH --partition gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-task 1
#SBATCH --time 8:00:00
#SBATCH --mem=80G
#SBATCH --job-name=mri128_3e-5
#SBATCH --output=logs/mri128_seed%a_%j.out
#SBATCH --error=logs/mri128_seed%a_%j.err
#SBATCH --exclude=workergpu027,workergpu047
#SBATCH --array=0-99

# Launch 100 seeds of 128x128 MRI at nu=eta=3e-5 (50 orbits, snaps 0-500)
# Usage: sbatch run_128_seeds.sbatch

set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID}
EXE="/mnt/home/mgoldstein/athenak_flatiron/build_mri2d/src/athena"
DECK="/mnt/home/mgoldstein/athenak_flatiron/decks/mri2d.athinput.128"
OUTDIR="/mnt/home/mgoldstein/ceph/athenak/mri128_nu3e-5/seed_$(printf '%04d' ${SEED})"

mkdir -p "${OUTDIR}" /mnt/home/mgoldstein/athenak_flatiron/logs

echo "=== MRI 2D 128x128, nu=eta=3e-5, seed=${SEED}, output=${OUTDIR} ==="
echo "Start: $(date)"

module purge
module load slurm gcc cuda openmpi/cuda-4.1.8

export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

srun --cpus-per-task=${SLURM_CPUS_PER_TASK:-8} --cpu-bind=cores \
  bash -c "unset CUDA_VISIBLE_DEVICES; \
  ${EXE} -i ${DECK} -d ${OUTDIR}/ \
  problem/rand_seed=${SEED} \
  mhd/viscosity=3e-5 \
  mhd/ohmic_resistivity=3e-5"

echo "Done: $(date)"
