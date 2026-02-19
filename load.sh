module load python/3.12.9 cuda/12 cudnn
source /mnt/home/mgoldstein/gameflow/.venv/bin/activate

# Store wandb logs outside project root so the wandb/ directory
# doesn't shadow the real wandb Python package
export WANDB_DIR=/mnt/home/mgoldstein/ceph/wandb_logs
