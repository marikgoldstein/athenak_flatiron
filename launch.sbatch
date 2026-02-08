#!/bin/bash -l                                                                                                                                                                                

#SBATCH --partition gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:4
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-task 1
#SBATCH --time 1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --job-name=athenak
#SBATCH --exclude=workergpu027,workergpu047

# --constraint="h100|a100-80gb"
EXE="/mnt/home/mgoldstein/athenak_flatiron/build_mri2d/src/athena"
LD="/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so"
DECK="/mnt/home/mgoldstein/athenak_flatiron/mri2d.athinput.2048"
OUT="/mnt/home/mgoldstein/ceph/athenak/"
set -euo pipefail
module purge
#module load slurm cuda/11.8.0 openmpi/cuda-4.0.7
module load slurm cuda openmpi

export LD_PRELOAD=${LD}
export CEPHTWEAKS_LAZYIO=1

srun --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores --gpu-bind=single:2 \
  bash -c "unset CUDA_VISIBLE_DEVICES; \
  ${EXE} -i ${DECK} -d ${OUT}"

#Notes
#There are 4 GPUs per node, 
# so generally <tasks> = <nodes> * 4. 
# The code does not necessarily use all 16 CPUs per task, 
# but this forces the tasks to be bound appropriately to NUMA nodes. 

#The --gpu-bind=single:2 statement may appear odd 
# (we want 1 task per GPU, not 2), 
# but this is a workaround for a bug in the current Slurm version installed.
#Scratch space appropriate for large I/O is under /mnt/ceph/users/.


#$ athena -h
#Athena v0.1
#Usage: athena [options] [block/par=value ...]
#Options:
#  -i <file>       specify input file [athinput]
#  -r <file>       restart with this file
#  -d <directory>  specify run dir [current dir]
#  -n              parse input file and quit
#  -c              show configuration and quit
#  -m              output mesh structure and quit
#  -t hh:mm:ss     wall time limit for final output
#  -h              this help

