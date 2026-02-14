#!/usr/bin/env python
"""
Merge per-rank AthenaK bin files, save merged athdf, and plot a 2D field.

Usage:
    source load.sh
    python plotting/postprocess_bins.py \
        --bindir /mnt/home/mgoldstein/ceph/athenak/feb13_512/bin \
        --variable mhd_bcc --field bcc1 --snapshot 00002

Defaults produce:
  - Merged athdf+xdmf in <bindir>/../processed/
  - Plot saved as plotting/mri_512.png
"""

import sys
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "athenak", "vis", "python")
)
import bin_convert


def assemble_2d(filedata, varname):
    """Assemble meshblock data into a full 2D array.

    Uses mb_logical (i, j, k, level) to place each meshblock at its
    correct global position.  mb_index contains *local* indices within
    each meshblock, so we need the logical coordinates instead.
    """
    Nx1 = filedata["Nx1"]
    Nx2 = filedata["Nx2"]
    nx1_out = filedata["nx1_out_mb"]
    nx2_out = filedata["nx2_out_mb"]
    full = np.zeros((Nx2, Nx1))

    for imb in range(filedata["n_mbs"]):
        logical = filedata["mb_logical"][imb]
        li, lj = int(logical[0]), int(logical[1])

        i_s = li * nx1_out
        j_s = lj * nx2_out

        block = filedata["mb_data"][varname][imb]
        if block.ndim == 3:
            block = block[0]  # squeeze k dimension for 2D

        full[j_s : j_s + nx2_out, i_s : i_s + nx1_out] = block

    return full


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-rank AthenaK bin files and plot a 2D field."
    )
    parser.add_argument(
        "--bindir",
        required=True,
        help="Path to bin/ directory containing rank_* subdirs",
    )
    parser.add_argument(
        "--variable",
        default="mhd_bcc",
        help="Variable group: mhd_w, mhd_bcc, mhd_jz (default: mhd_bcc)",
    )
    parser.add_argument(
        "--field",
        default="bcc1",
        help="Field name within variable group (default: bcc1)",
    )
    parser.add_argument(
        "--snapshot", default="00002", help="Snapshot number (default: 00002)"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory for merged athdf output (default: <bindir>/../processed)",
    )
    parser.add_argument(
        "--imagename",
        default=None,
        help="Output image path (default: plotting/mri_512.png)",
    )
    parser.add_argument(
        "--vscale",
        default="symmetric",
        choices=["linear", "log", "symmetric"],
        help="Color scale (default: symmetric)",
    )
    parser.add_argument("--cmap", default="RdBu_r", help="Colormap (default: RdBu_r)")
    parser.add_argument(
        "--no-save-athdf",
        action="store_true",
        help="Skip saving merged athdf files",
    )
    args = parser.parse_args()

    # Resolve defaults
    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(args.bindir.rstrip("/")), "processed")
    if args.imagename is None:
        args.imagename = os.path.join(os.path.dirname(__file__), "mri_512.png")

    # Build path to rank 0 file
    basename = f"HB3.{args.variable}.{args.snapshot}.bin"
    rank0_path = os.path.join(args.bindir, "rank_00000000", basename)

    if not os.path.exists(rank0_path):
        print(f"ERROR: {rank0_path} not found")
        sys.exit(1)

    # --- Step 1: Read and merge all ranks ---
    print(f"Reading all ranks for: {basename}")
    filedata = bin_convert.read_all_ranks_binary(rank0_path)

    print(f"  Time:       {filedata['time']:.6f}")
    print(f"  Cycle:      {filedata['cycle']}")
    print(f"  Grid:       {filedata['Nx1']} x {filedata['Nx2']}")
    print(f"  MeshBlocks: {filedata['n_mbs']}")
    print(f"  Variables:  {filedata['var_names']}")

    # --- Step 2: Save merged athdf ---
    if not args.no_save_athdf:
        os.makedirs(args.outdir, exist_ok=True)
        athdf_name = os.path.join(args.outdir, basename.replace(".bin", ".athdf"))
        print(f"  Writing merged athdf: {athdf_name}")
        bin_convert.write_athdf(athdf_name, filedata)
        xdmf_name = athdf_name + ".xdmf"
        bin_convert.write_xdmf_for(xdmf_name, os.path.basename(athdf_name), filedata)
        print(f"  Writing xdmf:         {xdmf_name}")

    # --- Step 3: Assemble 2D field and plot ---
    field = args.field
    if field not in filedata["var_names"]:
        print(f"  Field '{field}' not found. Available: {filedata['var_names']}")
        field = filedata["var_names"][0]
        print(f"  Using '{field}' instead.")

    data2d = assemble_2d(filedata, field)
    print(f"  Assembled {field}: shape={data2d.shape}, "
          f"min={data2d.min():.6e}, max={data2d.max():.6e}")

    x1min, x1max = filedata["x1min"], filedata["x1max"]
    x2min, x2max = filedata["x2min"], filedata["x2max"]

    fig, ax = plt.subplots(figsize=(8, 8))

    if args.vscale == "symmetric":
        norm = mcolors.CenteredNorm()
    elif args.vscale == "log":
        pos = data2d[data2d > 0]
        norm = mcolors.LogNorm(vmin=pos.min(), vmax=pos.max())
    else:
        norm = mcolors.Normalize(vmin=data2d.min(), vmax=data2d.max())

    im = ax.imshow(
        data2d,
        extent=(x1min, x1max, x2min, x2max),
        origin="lower",
        cmap=args.cmap,
        norm=norm,
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{field}, t = {filedata['time']:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    plt.savefig(args.imagename, bbox_inches="tight", dpi=200)
    print(f"  Saved plot: {args.imagename}")


if __name__ == "__main__":
    main()
