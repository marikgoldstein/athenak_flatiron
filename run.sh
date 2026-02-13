EXE="/mnt/home/mgoldstein/athenak_flatiron/build_mri2d/src/athena"
LD="/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so"
DECK="/mnt/home/mgoldstein/athenak_flatiron/decks/mri2d.athinput.test"
OUT="/mnt/home/mgoldstein/ceph/athenak/test/"
set -euo pipefail
module purge
module load slurm cuda openmpi
export LD_PRELOAD=${LD}
export CEPHTWEAKS_LAZYIO=1
eval "${EXE} -i ${DECK} -d ${OUT}"

