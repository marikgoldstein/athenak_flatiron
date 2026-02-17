#!/usr/bin/env python
"""
Convert AthenaK per-rank binary snapshots to a single numpy array.

Reads all snapshots, merges ranks, assembles 2D fields, downsamples,
and saves as a combined .npy file plus a metadata .npz.

Usage:
    python plotting/convert_to_npy.py \
        --bindir /mnt/home/mgoldstein/ceph/athenak/feb13/bin \
        --outdir /mnt/home/mgoldstein/ceph/athenak/feb13/npy \
        --resolution 256

Output:
    all_fields_{RES}x{RES}.npy   shape (N_snapshots, RES, RES, N_fields), float32
    metadata.npz                 field_names, times, x_coords, y_coords, sim_params
"""

import sys
import os
import argparse
import time as timer

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "athenak", "vis", "python")
)
import bin_convert


# ---- Field definitions --------------------------------------------------------
# (variable_group, field_name, human_label)
FIELD_DEFS = [
    ("mhd_w", "dens", "density"),
    ("mhd_w", "velx", "velx"),
    ("mhd_w", "vely", "vely"),
    ("mhd_w", "velz", "velz"),
    ("mhd_bcc", "bcc1", "Bx"),
    ("mhd_bcc", "bcc2", "By"),
    ("mhd_bcc", "bcc3", "Bz"),
    ("mhd_jz", "jz", "jz"),
]


def assemble_2d(filedata, varname):
    """Assemble meshblock data into a full 2D array."""
    Nx1 = filedata["Nx1"]
    Nx2 = filedata["Nx2"]
    nx1_out = filedata["nx1_out_mb"]
    nx2_out = filedata["nx2_out_mb"]
    full = np.zeros((Nx2, Nx1), dtype=np.float32)

    for imb in range(filedata["n_mbs"]):
        logical = filedata["mb_logical"][imb]
        li, lj = int(logical[0]), int(logical[1])
        i_s = li * nx1_out
        j_s = lj * nx2_out
        block = filedata["mb_data"][varname][imb]
        if block.ndim == 3:
            block = block[0]
        full[j_s : j_s + nx2_out, i_s : i_s + nx1_out] = block

    return full


def downsample(field, target_res, method="antialias"):
    """Downsample a 2D field to target_res x target_res.

    Args:
        field: (Ny, Nx) array
        target_res: target side length
        method: "antialias" for scipy zoom, "stride" for strided indexing
    """
    ny, nx = field.shape
    if ny == target_res and nx == target_res:
        return field

    if method == "stride":
        sy = ny // target_res
        sx = nx // target_res
        return field[::sy, ::sx][:target_res, :target_res].copy()
    else:
        # Anti-aliased via scipy zoom (order=3 = cubic)
        factor_y = target_res / ny
        factor_x = target_res / nx
        return zoom(field, (factor_y, factor_x), order=3).astype(np.float32)


def discover_snapshots(bindir, vargroup="mhd_w"):
    """Find all snapshot numbers available in rank 0."""
    rank0 = os.path.join(bindir, "rank_00000000")
    prefix = f"HB3.{vargroup}."
    suffix = ".bin"
    snaps = []
    for fname in os.listdir(rank0):
        if fname.startswith(prefix) and fname.endswith(suffix):
            num_str = fname[len(prefix) : -len(suffix)]
            snaps.append(int(num_str))
    snaps.sort()
    return snaps


def read_snapshot_fields(bindir, snap_num, field_defs):
    """Read all requested fields for one snapshot, return dict of 2D arrays.

    Groups reads by variable group to avoid redundant I/O.
    """
    snap_str = f"{snap_num:05d}"

    # Group field_defs by vargroup
    groups = {}
    for vargroup, fieldname, label in field_defs:
        groups.setdefault(vargroup, []).append((fieldname, label))

    fields = {}
    sim_time = None
    for vargroup, field_list in groups.items():
        rank0_path = os.path.join(
            bindir, "rank_00000000", f"HB3.{vargroup}.{snap_str}.bin"
        )
        if not os.path.exists(rank0_path):
            raise FileNotFoundError(f"Missing: {rank0_path}")

        filedata = bin_convert.read_all_ranks_binary(rank0_path)
        if sim_time is None:
            sim_time = filedata["time"]

        for fieldname, label in field_list:
            if fieldname not in filedata["var_names"]:
                print(f"  WARNING: {fieldname} not in {vargroup} var_names "
                      f"{filedata['var_names']}, skipping")
                continue
            fields[label] = assemble_2d(filedata, fieldname)

    return fields, sim_time


def main():
    parser = argparse.ArgumentParser(
        description="Convert AthenaK binary snapshots to numpy arrays."
    )
    parser.add_argument(
        "--bindir",
        default="/mnt/home/mgoldstein/ceph/athenak/feb13/bin",
        help="Path to bin/ directory containing rank_* subdirs",
    )
    parser.add_argument(
        "--outdir",
        default="/mnt/home/mgoldstein/ceph/athenak/feb13/npy",
        help="Output directory for npy files",
    )
    parser.add_argument(
        "--resolution", type=int, default=256,
        help="Target spatial resolution (default: 256)",
    )
    parser.add_argument(
        "--snap-start", type=int, default=None,
        help="First snapshot index (default: auto-detect min)",
    )
    parser.add_argument(
        "--snap-end", type=int, default=None,
        help="Last snapshot index inclusive (default: auto-detect max)",
    )
    parser.add_argument(
        "--snap-stride", type=int, default=1,
        help="Snapshot stride (default: 1 = every snapshot)",
    )
    parser.add_argument(
        "--downsample-method", default="antialias",
        choices=["antialias", "stride"],
        help="Downsampling method (default: antialias)",
    )
    parser.add_argument(
        "--fields", default=None,
        help="Comma-separated field labels to include "
             "(default: all 8 fields). E.g., 'velx,vely,Bx'",
    )
    args = parser.parse_args()

    # Resolve which fields to extract
    if args.fields is not None:
        requested = [f.strip() for f in args.fields.split(",")]
        field_defs = [fd for fd in FIELD_DEFS if fd[2] in requested]
        if not field_defs:
            print(f"ERROR: None of the requested fields {requested} found. "
                  f"Available: {[fd[2] for fd in FIELD_DEFS]}")
            sys.exit(1)
        missing = set(requested) - {fd[2] for fd in field_defs}
        if missing:
            print(f"WARNING: Fields not found: {missing}")
    else:
        field_defs = FIELD_DEFS

    field_labels = [fd[2] for fd in field_defs]
    n_fields = len(field_labels)
    res = args.resolution
    print(f"Fields to extract ({n_fields}): {field_labels}")
    print(f"Target resolution: {res}x{res}")
    print(f"Downsample method: {args.downsample_method}")

    # Discover snapshots
    all_snaps = discover_snapshots(args.bindir)
    print(f"Found {len(all_snaps)} snapshots: {all_snaps[0]:05d}..{all_snaps[-1]:05d}")

    snap_start = args.snap_start if args.snap_start is not None else all_snaps[0]
    snap_end = args.snap_end if args.snap_end is not None else all_snaps[-1]
    snaps = [s for s in all_snaps
             if snap_start <= s <= snap_end and (s - snap_start) % args.snap_stride == 0]
    n_snaps = len(snaps)
    print(f"Processing {n_snaps} snapshots: {snaps[0]:05d}..{snaps[-1]:05d} "
          f"(stride={args.snap_stride})")

    # Pre-allocate output array
    data = np.zeros((n_snaps, res, res, n_fields), dtype=np.float32)
    times = np.zeros(n_snaps, dtype=np.float64)

    os.makedirs(args.outdir, exist_ok=True)

    t_start = timer.time()
    for i, snap in enumerate(snaps):
        t0 = timer.time()
        fields, sim_time = read_snapshot_fields(args.bindir, snap, field_defs)
        times[i] = sim_time

        for j, label in enumerate(field_labels):
            if label in fields:
                data[i, :, :, j] = downsample(
                    fields[label], res, method=args.downsample_method
                )

        elapsed = timer.time() - t0
        if i % 50 == 0 or i == n_snaps - 1:
            total_elapsed = timer.time() - t_start
            rate = (i + 1) / total_elapsed
            eta = (n_snaps - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{n_snaps}] snap {snap:05d}  t={sim_time:10.4f}  "
                  f"({elapsed:.1f}s)  ETA: {eta/60:.1f}min")

    # Save data
    outname = os.path.join(args.outdir, f"all_fields_{res}x{res}.npy")
    print(f"\nSaving {outname}")
    print(f"  shape: {data.shape}, dtype: {data.dtype}")
    print(f"  size: {data.nbytes / 1e9:.2f} GB")
    np.save(outname, data)

    # Save metadata
    meta_path = os.path.join(args.outdir, "metadata.npz")
    # Read domain info from first snapshot
    rank0_path = os.path.join(
        args.bindir, "rank_00000000", f"HB3.mhd_w.{snaps[0]:05d}.bin"
    )
    fd0 = bin_convert.read_all_ranks_binary(rank0_path)

    x_coords = np.linspace(fd0["x1min"], fd0["x1max"], res, endpoint=False)
    y_coords = np.linspace(fd0["x2min"], fd0["x2max"], res, endpoint=False)

    np.savez(
        meta_path,
        field_names=np.array(field_labels),
        times=times,
        x_coords=x_coords,
        y_coords=y_coords,
        x1min=fd0["x1min"],
        x1max=fd0["x1max"],
        x2min=fd0["x2min"],
        x2max=fd0["x2max"],
        original_resolution=fd0["Nx1"],
        target_resolution=res,
        snap_indices=np.array(snaps),
    )
    print(f"Saved metadata: {meta_path}")

    total = timer.time() - t_start
    print(f"\nDone! Total time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
