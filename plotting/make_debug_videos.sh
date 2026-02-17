#!/bin/bash
# Make debug videos for the nu=3e-5 runs at 128 and 256.
# Run on a compute node: ssh ccmlin077 'bash /mnt/home/mgoldstein/athenak_flatiron/plotting/make_debug_videos.sh'

set -euo pipefail

module load ffmpeg
source /mnt/home/mgoldstein/athenak_flatiron/load.sh

SCRIPT="/mnt/home/mgoldstein/athenak_flatiron/plotting/make_video.py"
BASE="/mnt/home/mgoldstein/ceph/athenak"

# --- 128x128 nu=3e-5, seed 0 ---
echo "=== 128x128 nu=3e-5 jz video ==="
python3 "${SCRIPT}" \
    --bindir "${BASE}/mri128_nu3e-5/seed_0000/bin" \
    --outdir "${BASE}/mri128_nu3e-5/seed_0000/videos" \
    --variables mhd_jz --fields jz \
    --vmin -5 --vmax 5 --vscale linear \
    --workers 4

echo "=== 128x128 nu=3e-5 bcc1 video ==="
python3 "${SCRIPT}" \
    --bindir "${BASE}/mri128_nu3e-5/seed_0000/bin" \
    --outdir "${BASE}/mri128_nu3e-5/seed_0000/videos" \
    --variables mhd_bcc --fields bcc1 \
    --vscale symmetric \
    --workers 4

echo "=== 128x128 nu=3e-5 velx video ==="
python3 "${SCRIPT}" \
    --bindir "${BASE}/mri128_nu3e-5/seed_0000/bin" \
    --outdir "${BASE}/mri128_nu3e-5/seed_0000/videos" \
    --variables mhd_w --fields velx \
    --vscale symmetric \
    --workers 4

# --- 256x256 nu=3e-5, seed 0 ---
echo "=== 256x256 nu=3e-5 jz video ==="
python3 "${SCRIPT}" \
    --bindir "${BASE}/mri256_nu3e-5/seed_0000/bin" \
    --outdir "${BASE}/mri256_nu3e-5/seed_0000/videos" \
    --variables mhd_jz --fields jz \
    --vmin -5 --vmax 5 --vscale linear \
    --workers 4

echo "=== 256x256 nu=3e-5 bcc1 video ==="
python3 "${SCRIPT}" \
    --bindir "${BASE}/mri256_nu3e-5/seed_0000/bin" \
    --outdir "${BASE}/mri256_nu3e-5/seed_0000/videos" \
    --variables mhd_bcc --fields bcc1 \
    --vscale symmetric \
    --workers 4

echo "=== 256x256 nu=3e-5 velx video ==="
python3 "${SCRIPT}" \
    --bindir "${BASE}/mri256_nu3e-5/seed_0000/bin" \
    --outdir "${BASE}/mri256_nu3e-5/seed_0000/videos" \
    --variables mhd_w --fields velx \
    --vscale symmetric \
    --workers 4

echo "=== All videos done! ==="
echo "128 videos: ${BASE}/mri128_nu3e-5/seed_0000/videos/"
echo "256 videos: ${BASE}/mri256_nu3e-5/seed_0000/videos/"
