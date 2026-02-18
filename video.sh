module load python/3.11.11 ffmpeg                                                                                                                                                                                                                     
RES="256"
SEED="0008"

python plotting/make_video_from_npy.py \
  --npy /mnt/home/mgoldstein/ceph/athenak/mri${RES}_nu3e-5/seed_${SEED}/npy/all_fields_${RES}x${RES}.npy \
  --meta /mnt/home/mgoldstein/ceph/athenak/mri${RES}_nu3e-5/seed_${SEED}/npy/metadata.npz \
  --field jz --vmin -10 --vmax 10 --fps 22 \
  --outdir /mnt/home/mgoldstein/ceph/athenak/mri${RES}_nu3e-5/seed_${SEED}/videos

