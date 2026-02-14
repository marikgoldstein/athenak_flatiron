#!/bin/bash -l
cd /mnt/home/mgoldstein/athenak_flatiron
EXE="/mnt/home/mgoldstein/athenak_flatiron/build_mri2d/src/athena"
DECK="/mnt/home/mgoldstein/athenak_flatiron/decks/mri2d.athinput.2048"
OUT="/mnt/home/mgoldstein/ceph/athenak/feb13_test_fresh/"
set -euo pipefail

module purge
module load slurm gcc cuda openmpi/cuda-4.1.8

export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

srun --cpus-per-task=${SLURM_CPUS_PER_TASK:-16} --cpu-bind=cores \
  bash -c "unset CUDA_VISIBLE_DEVICES; \
  ${EXE} -i ${DECK} -d ${OUT}"
