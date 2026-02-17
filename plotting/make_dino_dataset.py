#!/usr/bin/env python
"""
Convert all_fields npy into DINO-compatible dataset files.

Takes the combined numpy array from convert_to_npy.py and produces:
  - Sliding-window samples for neural operator training
  - Diffusion format with low-res inputs and high-res targets
  - Normalization statistics (.npz)

Usage:
    python plotting/make_dino_dataset.py \
        --input /mnt/home/mgoldstein/ceph/athenak/feb13/npy/all_fields_256x256.npy \
        --metadata /mnt/home/mgoldstein/ceph/athenak/feb13/npy/metadata.npz \
        --outdir /mnt/home/mgoldstein/ceph/athenak/feb13/dino \
        --channels velx,vely,Bx \
        --format both

See --help for all options.
"""

import os
import sys
import argparse

import numpy as np
from scipy.ndimage import zoom


def sliding_window_samples(data, times, window, stride):
    """Create sliding-window samples from a time series.

    Args:
        data: (N_time, Ny, Nx, C) array
        times: (N_time,) array of simulation times
        window: number of timesteps per sample
        stride: step between consecutive windows

    Returns:
        samples: (N_samples, window, Ny, Nx, C)
        sample_times: (N_samples, window)
    """
    n_time = data.shape[0]
    starts = list(range(0, n_time - window + 1, stride))
    n_samples = len(starts)

    samples = np.zeros(
        (n_samples, window, data.shape[1], data.shape[2], data.shape[3]),
        dtype=data.dtype,
    )
    sample_times = np.zeros((n_samples, window), dtype=times.dtype)

    for i, s in enumerate(starts):
        samples[i] = data[s : s + window]
        sample_times[i] = times[s : s + window]

    return samples, sample_times


def split_train_val_test(n_samples, train_frac, val_frac, seed=42):
    """Split sample indices into train/val/test.

    Uses contiguous blocks (not random) since samples from one simulation
    are temporally correlated. This avoids data leakage from overlapping
    windows appearing in different splits.

    Args:
        n_samples: total number of samples
        train_frac: fraction for training
        val_frac: fraction for validation
        seed: unused (kept for API compatibility)

    Returns:
        train_idx, val_idx, test_idx: arrays of indices
    """
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_samples)

    return train_idx, val_idx, test_idx


def create_lowres_inputs(data, downsample_factor, method="cubic"):
    """Create low-resolution inputs by downsampling then upsampling.

    Args:
        data: (N_samples, C, T, Ny, Nx) array
        downsample_factor: spatial downsampling factor (e.g., 4 means 256->64->256)
        method: interpolation method for upsampling

    Returns:
        lowres: same shape as data, but blurred/degraded
    """
    n, c, t, ny, nx = data.shape
    lo_ny = ny // downsample_factor
    lo_nx = nx // downsample_factor
    lowres = np.zeros_like(data)

    order = 3 if method == "cubic" else 1  # 3=cubic, 1=linear

    for i in range(n):
        for ci in range(c):
            for ti in range(t):
                frame = data[i, ci, ti]
                # Downsample
                small = zoom(frame, (lo_ny / ny, lo_nx / nx), order=order)
                # Upsample back
                lowres[i, ci, ti] = zoom(small, (ny / lo_ny, nx / lo_nx), order=order)

    return lowres


def compute_stats(data):
    """Compute per-channel normalization statistics.

    Args:
        data: (N, C, T, Ny, Nx) or (N*T, C, Ny, Nx) array

    Returns:
        dict with mean, std, min_val, max_val each shape (C,)
    """
    if data.ndim == 5:
        # (N, C, T, Ny, Nx) -> reduce over N, T, Ny, Nx
        axes = (0, 2, 3, 4)
    elif data.ndim == 4:
        # (N, C, Ny, Nx) -> reduce over N, Ny, Nx
        axes = (0, 2, 3)
    else:
        raise ValueError(f"Expected 4D or 5D data, got {data.ndim}D")

    return {
        "mean": data.mean(axis=axes).astype(np.float32),
        "std": data.std(axis=axes).astype(np.float32),
        "min_val": data.min(axis=axes).astype(np.float32),
        "max_val": data.max(axis=axes).astype(np.float32),
    }


def resolve_channels(channel_str, field_names):
    """Parse channel selection string into indices.

    Args:
        channel_str: comma-separated field names (e.g., "velx,vely,Bx")
        field_names: list of all available field names

    Returns:
        indices: list of integer indices
        names: list of selected field names
    """
    requested = [s.strip() for s in channel_str.split(",")]
    field_list = list(field_names)
    indices = []
    names = []
    for name in requested:
        if name in field_list:
            indices.append(field_list.index(name))
            names.append(name)
        else:
            print(f"WARNING: Field '{name}' not found in {field_list}, skipping")
    if not indices:
        print(f"ERROR: No valid fields selected from {requested}")
        print(f"Available: {field_list}")
        sys.exit(1)
    return indices, names


def main():
    parser = argparse.ArgumentParser(
        description="Create DINO-compatible datasets from converted MRI simulation data."
    )
    parser.add_argument(
        "--input",
        default="/mnt/home/mgoldstein/ceph/athenak/feb13/npy/all_fields_256x256.npy",
        help="Path to all_fields npy from convert_to_npy.py",
    )
    parser.add_argument(
        "--metadata",
        default="/mnt/home/mgoldstein/ceph/athenak/feb13/npy/metadata.npz",
        help="Path to metadata.npz from convert_to_npy.py",
    )
    parser.add_argument(
        "--outdir",
        default="/mnt/home/mgoldstein/ceph/athenak/feb13/dino",
        help="Output directory for DINO datasets",
    )
    parser.add_argument(
        "--channels", default="velx,vely,Bx",
        help="Comma-separated field names to include (default: velx,vely,Bx)",
    )
    parser.add_argument(
        "--window", type=int, default=50,
        help="Sliding window length in timesteps (default: 50)",
    )
    parser.add_argument(
        "--stride", type=int, default=10,
        help="Sliding window stride (default: 10)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.7,
        help="Fraction of samples for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15,
        help="Fraction of samples for validation (default: 0.15)",
    )
    parser.add_argument(
        "--format", default="both",
        choices=["neurops", "diffusion", "both"],
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--downsample-factor", type=int, default=4,
        help="Spatial downsampling factor for diffusion low-res inputs (default: 4)",
    )
    parser.add_argument(
        "--target-resolution", type=int, default=None,
        help="Spatially downsample data to this resolution before creating datasets. "
             "E.g., --target-resolution 128 to go from 256x256 to 128x128. "
             "Default: None (use input resolution as-is)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # ---- Load data ----
    print(f"Loading {args.input}")
    data = np.load(args.input)  # (N_time, Ny, Nx, N_fields)
    meta = np.load(args.metadata, allow_pickle=True)
    field_names = meta["field_names"]
    times = meta["times"]
    print(f"  shape: {data.shape}, fields: {list(field_names)}")
    print(f"  time range: [{times[0]:.4f}, {times[-1]:.4f}]")

    # ---- Select channels ----
    ch_indices, ch_names = resolve_channels(args.channels, field_names)
    n_channels = len(ch_indices)
    print(f"\nSelected channels ({n_channels}): {ch_names} (indices {ch_indices})")

    data_selected = data[:, :, :, ch_indices]  # (N_time, Ny, Nx, C)
    print(f"  selected data shape: {data_selected.shape}")

    # ---- Optional spatial downsampling ----
    if args.target_resolution is not None:
        src_ny, src_nx = data_selected.shape[1], data_selected.shape[2]
        tgt = args.target_resolution
        if tgt != src_ny or tgt != src_nx:
            print(f"\nDownsampling spatially: {src_ny}x{src_nx} -> {tgt}x{tgt}")
            n_time_ds = data_selected.shape[0]
            n_ch = data_selected.shape[3]
            ds = np.zeros((n_time_ds, tgt, tgt, n_ch), dtype=data_selected.dtype)
            fy, fx = tgt / src_ny, tgt / src_nx
            for t_i in range(n_time_ds):
                for c_i in range(n_ch):
                    ds[t_i, :, :, c_i] = zoom(
                        data_selected[t_i, :, :, c_i], (fy, fx), order=3
                    ).astype(data_selected.dtype)
                if t_i % 200 == 0:
                    print(f"  [{t_i+1}/{n_time_ds}]")
            data_selected = ds
            print(f"  downsampled shape: {data_selected.shape}")

    # ---- Create sliding window samples ----
    print(f"\nCreating sliding window samples: window={args.window}, stride={args.stride}")
    samples, sample_times = sliding_window_samples(
        data_selected, times, args.window, args.stride
    )
    n_samples = samples.shape[0]
    print(f"  {n_samples} samples, shape: {samples.shape}")

    # ---- Train/val/test split ----
    train_idx, val_idx, test_idx = split_train_val_test(
        n_samples, args.train_frac, args.val_frac, args.seed
    )
    print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ---- Create output directory ----
    os.makedirs(args.outdir, exist_ok=True)
    stats_dir = os.path.join(args.outdir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Save metadata about this dataset
    np.savez(
        os.path.join(args.outdir, "dataset_info.npz"),
        channel_names=np.array(ch_names),
        channel_indices=np.array(ch_indices),
        window=args.window,
        stride=args.stride,
        n_samples=n_samples,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        sample_times=sample_times,
        downsample_factor=args.downsample_factor,
    )

    # ---- Neural operator format ----
    if args.format in ("neurops", "both"):
        print("\n--- Neural Operator Format ---")
        # Shape: (N_samples, T_window, Ny, Nx, C)
        # This is what get_dataloaders() expects from np.load()

        train_data = samples[train_idx]
        val_data = samples[val_idx]
        test_data = samples[test_idx]

        # Save combined (all splits in one file, DINO does its own splitting internally)
        all_path = os.path.join(args.outdir, "neurops_all.npy")
        print(f"  Saving {all_path}  shape={samples.shape}")
        np.save(all_path, samples)

        # Also save per-split for flexibility
        for name, d in [("train", train_data), ("val", val_data), ("test", test_data)]:
            path = os.path.join(args.outdir, f"neurops_{name}.npy")
            print(f"  Saving {path}  shape={d.shape}")
            np.save(path, d)

        # Compute stats on training data
        # Reshape to (N, C, T, Ny, Nx) for stats
        train_5d = np.transpose(train_data, (0, 4, 1, 2, 3))
        stats = compute_stats(train_5d)
        stats_path = os.path.join(stats_dir, "train_neurops_stats.npz")
        np.savez(stats_path, **stats)
        print(f"  Saved stats: {stats_path}")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    # ---- Diffusion format ----
    if args.format in ("diffusion", "both"):
        print("\n--- Diffusion Format ---")
        # DINO expects: .npy containing dict with "diff_inputs" and "diff_targets"
        # Shape: (N_samples, C, T_window, Ny, Nx)

        ny, nx = samples.shape[2], samples.shape[3]
        factor = args.downsample_factor
        print(f"  Spatial downsampling factor: {factor}x "
              f"({nx}x{ny} -> {nx//factor}x{ny//factor} -> {nx}x{ny})")

        for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_data = samples[idx]  # (N, T, Ny, Nx, C)

            # Transpose to (N, C, T, Ny, Nx) -- DINO's expected layout
            targets = np.transpose(split_data, (0, 4, 1, 2, 3)).astype(np.float32)

            # Create low-res inputs
            print(f"  Creating low-res inputs for {split_name} "
                  f"({targets.shape[0]} samples)...")
            inputs = create_lowres_inputs(targets, factor)

            # Save as dict in npy
            out_path = os.path.join(args.outdir, f"diffusion_{split_name}.npy")
            diff_dict = {"diff_inputs": inputs, "diff_targets": targets}
            np.save(out_path, diff_dict)
            print(f"  Saved {out_path}")
            print(f"    inputs:  shape={inputs.shape}, "
                  f"min={inputs.min():.5e}, max={inputs.max():.5e}")
            print(f"    targets: shape={targets.shape}, "
                  f"min={targets.min():.5e}, max={targets.max():.5e}")

        # Compute stats on training data
        train_targets = np.transpose(
            samples[train_idx], (0, 4, 1, 2, 3)
        ).astype(np.float32)
        train_inputs = create_lowres_inputs(train_targets, factor)

        input_stats = compute_stats(train_inputs)
        target_stats = compute_stats(train_targets)

        input_stats_path = os.path.join(stats_dir, "train_diffusion_inputs_stats.npz")
        target_stats_path = os.path.join(stats_dir, "train_diffusion_targets_stats.npz")
        np.savez(input_stats_path, **input_stats)
        np.savez(target_stats_path, **target_stats)
        print(f"\n  Saved input stats:  {input_stats_path}")
        for k, v in input_stats.items():
            print(f"    {k}: {v}")
        print(f"  Saved target stats: {target_stats_path}")
        for k, v in target_stats.items():
            print(f"    {k}: {v}")

    print(f"\nDone! All files saved to {args.outdir}")


if __name__ == "__main__":
    main()
