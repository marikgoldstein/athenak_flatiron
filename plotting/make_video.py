#!/usr/bin/env python
"""
Render mp4 videos from per-rank AthenaK binary snapshots.

Usage:
    source load.sh && module load ffmpeg
    python plotting/make_video.py --bindir /mnt/home/mgoldstein/ceph/athenak/feb13/bin

    # Quick test with one field and a few snapshots:
    python plotting/make_video.py \
        --bindir /mnt/home/mgoldstein/ceph/athenak/feb13/bin \
        --fields jz --snapshots 0:5

Optional flags:
    --outdir DIR          Output directory for videos (default: <bindir>/../videos/)
    --variables V [V..]   Subset of variable groups (default: all discovered)
    --fields F [F..]      Subset of fields (default: all in each variable group)
    --snapshots START:STOP  Snapshot range, python-style slice (default: all)
    --fps 30              Video framerate (default: 30)
    --dpi 150             Frame resolution (default: 150)
    --resolution N        Downsample data to NxN before plotting (default: native)
    --cmap CMAP           Colormap override
    --vscale MODE         Force linear/log/symmetric (default: auto per field)
    --workers N           Parallel frame rendering (default: 1)
"""

import sys
import os
import re
import argparse
import subprocess
import tempfile
import shutil
from functools import partial
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "athenak", "vis", "python")
)
import bin_convert

from postprocess_bins import assemble_2d


# Fields that should use log scale by default
LOG_FIELDS = {"dens"}
# Everything else gets symmetric (CenteredNorm) by default


def discover_variable_groups(bindir):
    """Scan rank_00000000/ for variable groups and snapshot numbers."""
    rank0 = os.path.join(bindir, "rank_00000000")
    pattern = re.compile(r"^HB3\.(.+)\.(\d+)\.bin$")
    groups = {}  # vargroup -> sorted list of snapshot strings
    for fname in os.listdir(rank0):
        m = pattern.match(fname)
        if m:
            vg, snap = m.group(1), m.group(2)
            groups.setdefault(vg, set()).add(snap)
    # Sort snapshot numbers
    for vg in groups:
        groups[vg] = sorted(groups[vg])
    return groups


def get_field_names(bindir, vargroup, snapshot):
    """Read one snapshot to discover field names in a variable group."""
    rank0_path = os.path.join(
        bindir, "rank_00000000", f"HB3.{vargroup}.{snapshot}.bin"
    )
    filedata = bin_convert.read_all_ranks_binary(rank0_path)
    return filedata["var_names"]


def compute_global_range(bindir, vargroup, field, snapshots, resolution=None):
    """Pre-scan all snapshots to find global min/max for a field."""
    global_min = np.inf
    global_max = -np.inf
    global_pos_min = np.inf  # smallest positive value, for log scale

    for snap in snapshots:
        rank0_path = os.path.join(
            bindir, "rank_00000000", f"HB3.{vargroup}.{snap}.bin"
        )
        filedata = bin_convert.read_all_ranks_binary(rank0_path)
        data2d = assemble_2d(filedata, field)

        if resolution is not None and resolution < data2d.shape[0]:
            from scipy.ndimage import zoom
            factor = resolution / data2d.shape[0]
            data2d = zoom(data2d, factor, order=1)

        global_min = min(global_min, data2d.min())
        global_max = max(global_max, data2d.max())
        pos = data2d[data2d > 0]
        if len(pos) > 0:
            global_pos_min = min(global_pos_min, pos.min())

    return global_min, global_max, global_pos_min


def choose_norm_and_cmap(field, vmin, vmax, vpos_min,
                         vscale_override=None, cmap_override=None):
    """Pick norm and colormap for a field based on global data range."""
    if vscale_override:
        vscale = vscale_override
    elif field in LOG_FIELDS:
        vscale = "log"
    else:
        vscale = "symmetric"

    if cmap_override:
        cmap = cmap_override
    elif vscale == "log":
        cmap = "viridis"
    else:
        cmap = "RdBu_r"

    if vscale == "symmetric":
        halfrange = max(abs(vmin), abs(vmax))
        norm = mcolors.CenteredNorm(vcenter=0, halfrange=halfrange)
    elif vscale == "log":
        if np.isinf(vpos_min):
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.LogNorm(vmin=vpos_min, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    return norm, cmap


def render_frame(args_tuple):
    """Render a single frame. Designed to work with multiprocessing."""
    (bindir, vargroup, field, snap, frame_idx, total_frames,
     framedir, dpi, resolution, vscale, cmap_name,
     global_min, global_max, global_pos_min) = args_tuple

    rank0_path = os.path.join(
        bindir, "rank_00000000", f"HB3.{vargroup}.{snap}.bin"
    )
    filedata = bin_convert.read_all_ranks_binary(rank0_path)
    data2d = assemble_2d(filedata, field)

    # Downsample if requested
    if resolution is not None and resolution < data2d.shape[0]:
        from scipy.ndimage import zoom
        factor = resolution / data2d.shape[0]
        data2d = zoom(data2d, factor, order=1)

    x1min, x1max = filedata["x1min"], filedata["x1max"]
    x2min, x2max = filedata["x2min"], filedata["x2max"]

    norm, cmap_used = choose_norm_and_cmap(
        field, global_min, global_max, global_pos_min, vscale, cmap_name
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        data2d,
        extent=(x1min, x1max, x2min, x2max),
        origin="lower",
        cmap=cmap_used,
        norm=norm,
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{field}, t = {filedata['time']:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    frame_path = os.path.join(framedir, f"frame_{frame_idx:05d}.png")
    plt.savefig(frame_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    print(f"  [{vargroup}/{field}] frame {frame_idx + 1:04d}/{total_frames}")
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


def parse_snapshots_slice(s):
    """Parse a string like '0:5' or '10:100:2' into a Python slice."""
    parts = s.split(":")
    parts = [int(p) if p else None for p in parts]
    return slice(*parts)


def main():
    parser = argparse.ArgumentParser(
        description="Render mp4 videos from AthenaK binary snapshots."
    )
    parser.add_argument(
        "--bindir", required=True,
        help="Path to bin/ directory containing rank_* subdirs",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory for videos (default: <bindir>/../videos/)",
    )
    parser.add_argument(
        "--variables", nargs="+", default=None,
        help="Subset of variable groups (default: all discovered)",
    )
    parser.add_argument(
        "--fields", nargs="+", default=None,
        help="Subset of fields (default: all in each variable group)",
    )
    parser.add_argument(
        "--snapshots", default=None,
        help="Snapshot range as python slice, e.g. 0:5 or 10:100:2 (default: all)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video framerate (default: 30)")
    parser.add_argument("--dpi", type=int, default=150, help="Frame DPI (default: 150)")
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Downsample data to NxN before plotting (default: native resolution)",
    )
    parser.add_argument("--cmap", default=None, help="Colormap override")
    parser.add_argument(
        "--vscale", default=None, choices=["linear", "log", "symmetric"],
        help="Force color scale mode (default: auto per field)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for frame rendering (default: 1)",
    )
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(
            os.path.dirname(args.bindir.rstrip("/")), "videos"
        )
    os.makedirs(args.outdir, exist_ok=True)

    # Step 1: Discover variable groups and snapshots
    print("Scanning for variable groups and snapshots...")
    all_groups = discover_variable_groups(args.bindir)
    if not all_groups:
        print("ERROR: No HB3.*.bin files found in rank_00000000/")
        sys.exit(1)

    vargroups = args.variables if args.variables else sorted(all_groups.keys())
    for vg in vargroups:
        if vg not in all_groups:
            print(f"ERROR: Variable group '{vg}' not found. Available: {sorted(all_groups.keys())}")
            sys.exit(1)

    print(f"Variable groups: {vargroups}")

    # Step 2: Process each variable group
    for vg in vargroups:
        snapshots = all_groups[vg]
        if args.snapshots:
            sl = parse_snapshots_slice(args.snapshots)
            snapshots = snapshots[sl]

        print(f"\n{'='*60}")
        print(f"Variable group: {vg} ({len(snapshots)} snapshots)")
        print(f"{'='*60}")

        # Read one snapshot to discover fields
        field_names = get_field_names(args.bindir, vg, snapshots[0])
        if args.fields:
            field_names = [f for f in field_names if f in args.fields]
            if not field_names:
                print(f"  No matching fields in {vg}. Available: {get_field_names(args.bindir, vg, snapshots[0])}")
                continue

        print(f"  Fields: {field_names}")

        # Step 3: Render each field
        for field in field_names:
            print(f"\n  --- {vg}/{field}: {len(snapshots)} frames ---")

            # Pre-scan to fix the colorbar across all frames
            print(f"  Pre-scanning {len(snapshots)} snapshots for global color range...")
            gmin, gmax, gpos_min = compute_global_range(
                args.bindir, vg, field, snapshots, args.resolution
            )
            print(f"  Global range: [{gmin:.4e}, {gmax:.4e}]")

            framedir = tempfile.mkdtemp(prefix=f"video_{vg}_{field}_")
            try:
                # Build argument tuples for render_frame
                render_args = [
                    (args.bindir, vg, field, snap, i, len(snapshots),
                     framedir, args.dpi, args.resolution, args.vscale, args.cmap,
                     gmin, gmax, gpos_min)
                    for i, snap in enumerate(snapshots)
                ]

                if args.workers > 1:
                    with Pool(args.workers) as pool:
                        pool.map(render_frame, render_args)
                else:
                    for ra in render_args:
                        render_frame(ra)

                # Stitch into video
                video_name = f"{vg}_{field}.mp4"
                video_path = os.path.join(args.outdir, video_name)
                stitch_video(framedir, video_path, args.fps)

            finally:
                shutil.rmtree(framedir, ignore_errors=True)

    print(f"\nDone! Videos saved to: {args.outdir}")


if __name__ == "__main__":
    main()
