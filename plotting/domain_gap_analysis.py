#!/usr/bin/env python
"""
Domain gap analysis: compare native low-res vs downsampled high-res.

Produces:
  1. Power spectra comparison (native 128, native 256, downsampled 256->128)
  2. PDFs of jz
  3. Visual side-by-side (native 128 vs downsampled 256)
  4. Time evolution of jz std (to check if MRI sustains)

Usage:
    module load python/3.11.11
    python plotting/domain_gap_analysis.py \
        --dir128 /mnt/home/mgoldstein/ceph/athenak/mri128_check \
        --dir256 /mnt/home/mgoldstein/ceph/athenak/mri256_check \
        --seeds 0 1 \
        --snap 500 \
        --outdir /mnt/home/mgoldstein/ceph/athenak/

    # For the nu sweep:
    python plotting/domain_gap_analysis.py \
        --dir128 /mnt/home/mgoldstein/ceph/athenak/mri128_nu3e-5 \
        --dir256 /mnt/home/mgoldstein/ceph/athenak/mri256_nu3e-5 \
        --seeds 0 \
        --snap 500 \
        --outdir /mnt/home/mgoldstein/ceph/athenak/ \
        --label nu3e-5
"""

import sys
import os
import argparse

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "athenak", "vis", "python")
)
import bin_convert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def assemble_2d(filedata, varname):
    Nx1, Nx2 = filedata["Nx1"], filedata["Nx2"]
    nx1, nx2 = filedata["nx1_out_mb"], filedata["nx2_out_mb"]
    full = np.zeros((Nx2, Nx1), dtype=np.float32)
    for imb in range(filedata["n_mbs"]):
        lo = filedata["mb_logical"][imb]
        li, lj = int(lo[0]), int(lo[1])
        blk = filedata["mb_data"][varname][imb]
        if blk.ndim == 3:
            blk = blk[0]
        full[lj * nx2 : (lj + 1) * nx2, li * nx1 : (li + 1) * nx1] = blk
    return full


def power_spectrum_2d(field):
    """Azimuthally-averaged 2D power spectrum."""
    ft = np.fft.fft2(field)
    ps = np.abs(ft) ** 2
    N = field.shape[0]
    kx = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_bins = np.arange(0.5, N // 2 + 1, 1.0)
    k_vals = 0.5 * (k_bins[:-1] + k_bins[1:])
    ps_avg = np.zeros(len(k_vals))
    for i in range(len(k_vals)):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if mask.sum() > 0:
            ps_avg[i] = ps[mask].mean()
    return k_vals, ps_avg


def load_field(basedir, seed, vargroup, field, snap):
    """Load a 2D field from AthenaK binary output."""
    seed_str = f"seed_{seed:04d}"
    path = os.path.join(
        basedir, seed_str, "bin", "rank_00000000",
        f"HB3.{vargroup}.{snap:05d}.bin",
    )
    fd = bin_convert.read_all_ranks_binary(path)
    return assemble_2d(fd, field), fd["time"]


def track_jz_evolution(basedir, seed, snaps=None):
    """Track jz std over time for one seed."""
    if snaps is None:
        snaps = [0, 50, 100, 200, 300, 500, 700, 1000]
    times, stds = [], []
    for snap in snaps:
        try:
            jz, t = load_field(basedir, seed, "mhd_jz", "jz", snap)
            times.append(t)
            stds.append(jz.std())
        except FileNotFoundError:
            pass
    return np.array(times), np.array(stds)


def main():
    parser = argparse.ArgumentParser(
        description="Domain gap analysis: native low-res vs downsampled high-res."
    )
    parser.add_argument("--dir128", required=True, help="Base dir for 128 runs")
    parser.add_argument("--dir256", required=True, help="Base dir for 256 runs")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="Seed indices to compare (default: 0)",
    )
    parser.add_argument(
        "--snap", type=int, default=500,
        help="Snapshot number for spectra/PDF/visual comparison (default: 500)",
    )
    parser.add_argument(
        "--field", default="jz",
        help="Field to analyze (default: jz)",
    )
    parser.add_argument(
        "--vargroup", default="mhd_jz",
        help="Variable group for the field (default: mhd_jz)",
    )
    parser.add_argument("--outdir", default=".", help="Output directory for plots")
    parser.add_argument("--label", default="", help="Label for output filenames")
    parser.add_argument(
        "--vrange", type=float, default=2.0,
        help="Symmetric color range for visual comparison (default: 2.0)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""
    n_seeds = len(args.seeds)

    # =========================================================================
    # Figure 1: Power spectra + PDFs + visual comparison (per seed)
    # =========================================================================
    fig1, axes1 = plt.subplots(n_seeds, 3, figsize=(18, 6 * n_seeds), squeeze=False)

    for row, seed in enumerate(args.seeds):
        snap = args.snap
        print(f"\n=== seed {seed}, snap {snap:05d} ===")

        # Load fields
        jz128, t128 = load_field(args.dir128, seed, args.vargroup, args.field, snap)
        jz256, t256 = load_field(args.dir256, seed, args.vargroup, args.field, snap)

        # Downsample 256 -> 128
        factor = jz128.shape[0] / jz256.shape[0]
        jz256_ds = zoom(jz256, factor, order=3)

        print(f"  128 native:      std={jz128.std():.4e}  range=[{jz128.min():.4f}, {jz128.max():.4f}]")
        print(f"  256 native:      std={jz256.std():.4e}  range=[{jz256.min():.4f}, {jz256.max():.4f}]")
        print(f"  256 downsampled: std={jz256_ds.std():.4e}  range=[{jz256_ds.min():.4f}, {jz256_ds.max():.4f}]")
        print(f"  L2 gap (128 vs ds256): {np.sqrt(((jz128 - jz256_ds)**2).mean()):.4e}")

        # Power spectra
        ax = axes1[row, 0]
        k128, ps128 = power_spectrum_2d(jz128)
        k256, ps256 = power_spectrum_2d(jz256)
        k256ds, ps256ds = power_spectrum_2d(jz256_ds)
        ax.loglog(k128, ps128, "b-", label="native 128", linewidth=2)
        ax.loglog(k256, ps256, "r-", label="native 256", linewidth=2)
        ax.loglog(k256ds, ps256ds, "g--", label="256→128", linewidth=2)
        ax.set_xlabel("k")
        ax.set_ylabel("P(k)")
        ax.set_title(f"seed {seed}, snap {snap}: Power Spectra ({args.field})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # PDFs
        ax = axes1[row, 1]
        vmax_pdf = max(abs(jz128).max(), abs(jz256).max(), abs(jz256_ds).max())
        bins = np.linspace(-vmax_pdf, vmax_pdf, 100)
        ax.hist(jz128.ravel(), bins=bins, density=True, alpha=0.5, label="native 128", color="blue")
        ax.hist(jz256_ds.ravel(), bins=bins, density=True, alpha=0.5, label="256→128", color="green")
        ax.hist(jz256.ravel(), bins=bins, density=True, alpha=0.5, label="native 256", color="red")
        ax.set_xlabel(args.field)
        ax.set_ylabel("PDF")
        ax.set_title(f"seed {seed}: PDF of {args.field}")
        ax.legend()
        ax.set_yscale("log")

        # Visual comparison
        ax = axes1[row, 2]
        im = ax.imshow(
            np.concatenate([jz128, jz256_ds], axis=1),
            extent=[0, 2, 0, 1], origin="lower", cmap="RdBu_r",
            vmin=-args.vrange, vmax=args.vrange,
        )
        ax.axvline(1.0, color="black", linewidth=2)
        ax.set_title(f"seed {seed}: native 128 (left) vs 256→128 (right)")
        ax.text(0.25, 0.95, "native 128", ha="center", va="top",
                transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.text(0.75, 0.95, "256→128", ha="center", va="top",
                transform=ax.transAxes, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig1.tight_layout()
    path1 = os.path.join(args.outdir, f"domain_gap_comparison{suffix}.png")
    fig1.savefig(path1, dpi=150)
    print(f"\nSaved: {path1}")

    # =========================================================================
    # Figure 2: jz std time evolution (check if MRI sustains)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for seed in args.seeds:
        t128, std128 = track_jz_evolution(args.dir128, seed)
        t256, std256 = track_jz_evolution(args.dir256, seed)
        ax2.semilogy(t128 / (2 * np.pi), std128, "b-o", label=f"128 seed {seed}", markersize=4)
        ax2.semilogy(t256 / (2 * np.pi), std256, "r-s", label=f"256 seed {seed}", markersize=4)

    ax2.set_xlabel("Time (orbits)")
    ax2.set_ylabel(f"std({args.field})")
    ax2.set_title(f"{args.field} amplitude evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    path2 = os.path.join(args.outdir, f"jz_evolution{suffix}.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2}")

    plt.close("all")
    print("\nDone!")


if __name__ == "__main__":
    main()
