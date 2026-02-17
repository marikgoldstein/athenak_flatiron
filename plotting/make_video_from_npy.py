#!/usr/bin/env python
"""
Render an mp4 video of a single field from the pre-converted npy file.

Usage:
    module load python/3.11.11 ffmpeg
    python plotting/make_video_from_npy.py \
        --npy /mnt/home/mgoldstein/ceph/athenak/feb13/npy/all_fields_256x256.npy \
        --meta /mnt/home/mgoldstein/ceph/athenak/feb13/npy/metadata.npz \
        --field jz --vmin -20 --vmax 20

Optional flags:
    --outdir DIR     Output directory (default: same as npy dir)
    --fps 30         Framerate (default: 30)
    --dpi 150        Frame DPI (default: 150)
    --cmap CMAP      Colormap (default: RdBu_r)
    --snap-start N   First snapshot index (default: 0)
    --snap-end N     Last snapshot index exclusive (default: all)
    --snap-stride N  Stride (default: 1)
    --workers N      Parallel frame rendering (default: 4)
"""

import sys
import os
import argparse
import subprocess
import tempfile
import shutil
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def render_frame(args_tuple):
    """Render a single frame."""
    (data2d, sim_time, field, frame_idx, total_frames,
     framedir, dpi, vmin, vmax, cmap_name,
     x1min, x1max, x2min, x2max) = args_tuple

    if vmin is not None and vmax is not None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        halfrange = max(abs(data2d.min()), abs(data2d.max()))
        norm = mcolors.CenteredNorm(vcenter=0, halfrange=halfrange)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        data2d,
        extent=(x1min, x1max, x2min, x2max),
        origin="lower",
        cmap=cmap_name,
        norm=norm,
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{field}, t = {sim_time:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    frame_path = os.path.join(framedir, f"frame_{frame_idx:05d}.png")
    plt.savefig(frame_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
        print(f"  [{field}] frame {frame_idx + 1:04d}/{total_frames}")
    return frame_path


def stitch_video(framedir, outpath, fps):
    """Use ffmpeg to stitch PNGs into an mp4."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(framedir, "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        outpath,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg stderr:\n{result.stderr}")
        result.check_returncode()
    print(f"  Saved video: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Render mp4 video from pre-converted npy data."
    )
    parser.add_argument(
        "--npy", required=True,
        help="Path to all_fields_NxN.npy file",
    )
    parser.add_argument(
        "--meta", required=True,
        help="Path to metadata.npz file",
    )
    parser.add_argument(
        "--field", required=True,
        help="Field name to render (e.g., jz, density, velx)",
    )
    parser.add_argument("--vmin", type=float, default=None, help="Color scale min")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale max")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Video framerate")
    parser.add_argument("--dpi", type=int, default=150, help="Frame DPI")
    parser.add_argument("--cmap", default="RdBu_r", help="Colormap")
    parser.add_argument("--snap-start", type=int, default=0, help="First snapshot index")
    parser.add_argument("--snap-end", type=int, default=None, help="Last snapshot index (exclusive)")
    parser.add_argument("--snap-stride", type=int, default=1, help="Snapshot stride")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.npy} ...")
    data = np.load(args.npy, mmap_mode="r")
    meta = np.load(args.meta, allow_pickle=True)

    field_names = list(meta["field_names"])
    if args.field not in field_names:
        print(f"ERROR: Field '{args.field}' not found. Available: {field_names}")
        sys.exit(1)

    field_idx = field_names.index(args.field)
    times = meta["times"]
    x1min, x1max = float(meta["x1min"]), float(meta["x1max"])
    x2min, x2max = float(meta["x2min"]), float(meta["x2max"])

    n_snaps = data.shape[0]
    snap_end = args.snap_end if args.snap_end is not None else n_snaps
    snap_indices = list(range(args.snap_start, snap_end, args.snap_stride))
    n_frames = len(snap_indices)

    print(f"Field: {args.field} (index {field_idx})")
    print(f"Resolution: {data.shape[1]}x{data.shape[2]}")
    print(f"Frames: {n_frames} (snapshots {snap_indices[0]}..{snap_indices[-1]})")
    if args.vmin is not None and args.vmax is not None:
        print(f"Color range: [{args.vmin}, {args.vmax}]")

    if args.outdir is None:
        args.outdir = os.path.dirname(args.npy)
    os.makedirs(args.outdir, exist_ok=True)

    framedir = tempfile.mkdtemp(prefix=f"video_{args.field}_")
    try:
        # Build argument tuples
        render_args = []
        for i, si in enumerate(snap_indices):
            frame_data = np.array(data[si, :, :, field_idx])  # copy from mmap
            render_args.append((
                frame_data, times[si], args.field, i, n_frames,
                framedir, args.dpi, args.vmin, args.vmax, args.cmap,
                x1min, x1max, x2min, x2max
            ))

        if args.workers > 1:
            with Pool(args.workers) as pool:
                pool.map(render_frame, render_args)
        else:
            for ra in render_args:
                render_frame(ra)

        # Stitch
        video_name = f"{args.field}_video.mp4"
        video_path = os.path.join(args.outdir, video_name)
        stitch_video(framedir, video_path, args.fps)

    finally:
        shutil.rmtree(framedir, ignore_errors=True)

    print(f"\nDone! Video saved to: {video_path}")


if __name__ == "__main__":
    main()
