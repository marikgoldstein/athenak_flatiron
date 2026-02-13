#!/bin/bash -l
cd /mnt/home/mgoldstein/athenak_flatiron
EXE="/mnt/home/mgoldstein/athenak_flatiron/build_mri2d/src/athena"
LD="/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so"
DECK="/mnt/home/mgoldstein/athenak_flatiron/decks/mri2d.athinput.2048"
OUT="/mnt/home/mgoldstein/ceph/athenak/feb13_res2048/"
set -euo pipefail                                                                                                 
module purge
module load slurm cuda openmpi/cuda-4.1.8

export LD_PRELOAD=${LD}
export CEPHTWEAKS_LAZYIO=1

mpirun -np 4 \
    --bind-to core \
    --map-by ppr:4:node \
    -x LD_PRELOAD \
    -x CEPHTWEAKS_LAZYIO \
    bash -c 'export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}; '"${EXE}"' -i '"${DECK}"' -d '"${OUT}"
