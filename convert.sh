module load python/3.11.11
SEED="0008"
RES="256"
python processing/convert_to_npy.py \
	--bindir /mnt/home/mgoldstein/ceph/athenak/mri${RES}_nu3e-5/seed_${SEED}/bin \
	--outdir /mnt/home/mgoldstein/ceph/athenak/mri${RES}_nu3e-5/seed_${SEED}/npy \
	--snap-start 0 --snap-end 199
