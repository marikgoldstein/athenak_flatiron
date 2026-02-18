"""
Paired MRI super-resolution dataset.

Returns (native_128, native_256) pairs from memory-mapped npy files.
Use prepare_batch() to get the GPU-side triplet:
    up_128:      native 128 bilinearly upsampled to 256x256  (inference input)
    x_256:       native 256x256                               (high-res target)
    up_down_256: 256 avg-pooled to 128 then upsampled to 256  (training input)

Memory management: all npy files are memory-mapped (mmap_mode='r'), so the OS
lazily pages in only the frames you access. Total data is ~52 GB but actual RAM
usage depends on working set, not total size.

Split by seed (not time) to avoid leakage.

Example usage:
    from dataset import MRIPairedDataset, prepare_batch

    train_seeds = list(range(0, 80))
    val_seeds   = list(range(80, 90))
    test_seeds  = list(range(90, 100))

    train_ds = MRIPairedDataset(train_seeds)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True)

    for x_128, x_256 in train_loader:
        up_128, x_256, up_down_256 = prepare_batch(x_128, x_256)
        # up_128:      (B, 8, 256, 256) - what model sees at inference
        # x_256:       (B, 8, 256, 256) - ground truth target
        # up_down_256: (B, 8, 256, 256) - what model sees at training
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Default paths
BASE_256 = "/mnt/home/mgoldstein/ceph/athenak/mri256_nu3e-5"
BASE_128 = "/mnt/home/mgoldstein/ceph/athenak/mri128_nu3e-5"

FIELD_NAMES = ["density", "velx", "vely", "velz", "Bx", "By", "Bz", "jz"]


class MRIPairedDataset(Dataset):
    """
    Paired (128, 256) dataset from memory-mapped per-seed npy files.

    Each item is one frame from one seed, returned at native resolution.
    Indices map as: idx = seed_position * n_frames + frame_idx.
    """

    def __init__(
        self,
        seed_ids,
        base_dir_256=BASE_256,
        base_dir_128=BASE_128,
        n_frames=200,
    ):
        self.seed_ids = list(seed_ids)
        self.n_frames = n_frames

        # Memory-map all seed files (lazy, no RAM cost until accessed)
        self.data_256 = []
        self.data_128 = []
        for s in self.seed_ids:
            path_256 = f"{base_dir_256}/seed_{s:04d}/npy/all_fields_256x256.npy"
            path_128 = f"{base_dir_128}/seed_{s:04d}/npy/all_fields_128x128.npy"
            self.data_256.append(np.load(path_256, mmap_mode="r"))
            self.data_128.append(np.load(path_128, mmap_mode="r"))

        print(
            f"MRIPairedDataset: {len(self.seed_ids)} seeds x {n_frames} frames "
            f"= {len(self)} samples"
        )

    def __len__(self):
        return len(self.seed_ids) * self.n_frames

    def __getitem__(self, idx):
        seed_pos = idx // self.n_frames
        frame = idx % self.n_frames

        # Copy from mmap into contiguous numpy array, then to tensor
        x_256 = np.array(self.data_256[seed_pos][frame])  # (256, 256, 8)
        x_128 = np.array(self.data_128[seed_pos][frame])  # (128, 128, 8)

        # channels-last -> channels-first
        x_256 = torch.from_numpy(x_256).permute(2, 0, 1)  # (8, 256, 256)
        x_128 = torch.from_numpy(x_128).permute(2, 0, 1)  # (8, 128, 128)

        return x_128, x_256


def prepare_batch(x_128, x_256, device="cuda"):
    """
    Move to GPU and compute the three views.

    Args:
        x_128: (B, 8, 128, 128) native low-res
        x_256: (B, 8, 256, 256) native high-res

    Returns:
        up_128:      (B, 8, 256, 256) native 128 upsampled to 256
        x_256:       (B, 8, 256, 256) native 256 (target)
        up_down_256: (B, 8, 256, 256) 256 -> avg_pool -> 128 -> upsample -> 256
    """
    x_128 = x_128.to(device)
    x_256 = x_256.to(device)

    # Native 128 upsampled to 256 (what the model gets at inference)
    up_128 = F.interpolate(x_128, size=256, mode="bilinear", align_corners=False)

    # Synthetic low-res from 256: downsample then upsample (training input)
    down_256 = F.avg_pool2d(x_256, kernel_size=2)  # (B, 8, 128, 128)
    up_down_256 = F.interpolate(
        down_256, size=256, mode="bilinear", align_corners=False
    )

    return up_128, x_256, up_down_256


# ---------------------------------------------------------------------------
# Convenience: build train/val/test loaders with default 80/10/10 seed split
# ---------------------------------------------------------------------------

def make_loaders(
    batch_size=16,
    num_workers=4,
    train_seeds=None,
    val_seeds=None,
    test_seeds=None,
    n_frames=200,
    base_dir_256=BASE_256,
    base_dir_128=BASE_128,
):
    """
    Returns (train_loader, val_loader, test_loader).

    Default split: seeds 0-79 train, 80-89 val, 90-99 test.
    """
    if train_seeds is None:
        train_seeds = list(range(0, 80))
    if val_seeds is None:
        val_seeds = list(range(80, 90))
    if test_seeds is None:
        test_seeds = list(range(90, 100))

    kw = dict(n_frames=n_frames, base_dir_256=base_dir_256, base_dir_128=base_dir_128)

    train_ds = MRIPairedDataset(train_seeds, **kw)
    val_ds = MRIPairedDataset(val_seeds, **kw)
    test_ds = MRIPairedDataset(test_seeds, **kw)

    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kw
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kw
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo_seeds = [0, 1, 2]
    n_frames = 200

    # --- 1. Dataset + DataLoader basics ---
    print(f"=== Demo: {len(demo_seeds)} seeds, {n_frames} frames, device={device} ===\n")
    ds = MRIPairedDataset(demo_seeds, n_frames=n_frames)

    x_128, x_256 = ds[0]
    print(f"Single item:  x_128={x_128.shape}  x_256={x_256.shape}")

    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    # --- 2. Iterate a few batches, time it ---
    print(f"\nDataLoader: {len(loader)} batches of 8")
    t0 = time.time()
    for i, (x_128_b, x_256_b) in enumerate(loader):
        up_128, x_256_b, up_down_256 = prepare_batch(x_128_b, x_256_b, device)
        if i == 0:
            print(f"  First batch shapes after prepare_batch:")
            print(f"    up_128      = {up_128.shape}")
            print(f"    x_256       = {x_256_b.shape}")
            print(f"    up_down_256 = {up_down_256.shape}")
        if i >= 9:
            break
    elapsed = time.time() - t0
    print(f"  10 batches in {elapsed:.2f}s ({elapsed / 10 * 1000:.0f} ms/batch)")

    # --- 3. Per-field statistics across a few batches ---
    print(f"\nPer-field stats (first 5 batches, native 256):")
    sums = torch.zeros(8, device=device)
    sq_sums = torch.zeros(8, device=device)
    mins = torch.full((8,), float("inf"), device=device)
    maxs = torch.full((8,), float("-inf"), device=device)
    n = 0
    for i, (x_128_b, x_256_b) in enumerate(loader):
        x = x_256_b.to(device)  # (B, 8, 256, 256)
        B = x.shape[0]
        sums += x.sum(dim=(0, 2, 3))
        sq_sums += (x ** 2).sum(dim=(0, 2, 3))
        mins = torch.min(mins, x.reshape(B, 8, -1).min(dim=2).values.min(dim=0).values)
        maxs = torch.max(maxs, x.reshape(B, 8, -1).max(dim=2).values.max(dim=0).values)
        n += B * 256 * 256
        if i >= 4:
            break
    means = sums / n
    stds = ((sq_sums / n) - means ** 2).clamp(min=0).sqrt()
    print(f"  {'field':<10s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s}")
    for j, name in enumerate(FIELD_NAMES):
        print(f"  {name:<10s} {means[j].item():12.4e} {stds[j].item():12.4e} "
              f"{mins[j].item():12.4e} {maxs[j].item():12.4e}")

    # --- 4. Comparison figure: the three views for 4 fields ---
    print("\nGenerating comparison figure...")
    x_128_b, x_256_b = next(iter(loader))
    up_128, x_256_b, up_down_256 = prepare_batch(x_128_b, x_256_b, device)

    show_fields = [1, 4, 7]  # velx, Bx, jz
    sample_idx = 0
    views = [
        ("up(128) [inference input]", up_128),
        ("up(down(256)) [train input]", up_down_256),
        ("native 256 [target]", x_256_b),
    ]

    fig, axes = plt.subplots(
        len(show_fields), len(views) + 1,
        figsize=(4 * (len(views) + 1), 4 * len(show_fields)),
    )
    for row, ch in enumerate(show_fields):
        imgs = [v[1][sample_idx, ch].cpu().numpy() for v in views]
        vmax = max(abs(im.min()) for im in imgs + [imgs[0]])
        vmax = max(vmax, max(abs(im.max()) for im in imgs))

        for col, (label, _) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(imgs[col], origin="lower", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, aspect="equal")
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(FIELD_NAMES[ch], fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        # Difference: up(down(256)) vs native 256
        diff = imgs[1] - imgs[2]
        ax = axes[row, len(views)]
        ax.imshow(diff, origin="lower", cmap="RdBu_r",
                  vmin=-vmax * 0.3, vmax=vmax * 0.3, aspect="equal")
        if row == 0:
            ax.set_title("train_input - target", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Three views + residual (one sample)", fontsize=14, y=1.01)
    fig.tight_layout()
    out_path = "simple_diffusion/demo_three_views.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # --- 5. Domain gap: up(128) vs up(down(256)) ---
    gap = (up_128 - up_down_256).abs().mean(dim=(0, 2, 3))  # per-channel
    print(f"\nDomain gap  mean|up(128) - up(down(256))| per field:")
    for j, name in enumerate(FIELD_NAMES):
        print(f"  {name:<10s} {gap[j].item():.6f}")

    # --- 6. Power spectra: all three views ---
    print("\nComputing power spectra (averaged over batch)...")

    def radial_power_spectrum(field_2d):
        """Radially averaged 2D power spectrum. Input: (H, W) numpy array."""
        N = field_2d.shape[0]
        fft2 = np.fft.fft2(field_2d)
        ps2d = np.abs(fft2) ** 2 / N ** 4
        # Radial binning
        kx = np.fft.fftfreq(N, d=1.0 / N)
        ky = np.fft.fftfreq(N, d=1.0 / N)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX ** 2 + KY ** 2)
        k_max = N // 2
        k_bins = np.arange(1, k_max + 1)
        ps1d = np.zeros(len(k_bins))
        for i, k in enumerate(k_bins):
            mask = (K >= k - 0.5) & (K < k + 0.5)
            if mask.any():
                ps1d[i] = ps2d[mask].mean()
        return k_bins, ps1d

    # Average spectra over the batch for each view
    show_spec = [1, 4, 7]  # velx, Bx, jz
    B_size = up_128.shape[0]

    fig, axes = plt.subplots(1, len(show_spec), figsize=(6 * len(show_spec), 5))
    for col, ch in enumerate(show_spec):
        ax = axes[col]
        spectra = {}
        for label, tensor in [
            ("up(128)", up_128),
            ("up(down(256))", up_down_256),
            ("native 256", x_256_b),
        ]:
            ps_sum = None
            for b in range(B_size):
                k_bins, ps = radial_power_spectrum(tensor[b, ch].cpu().numpy())
                if ps_sum is None:
                    ps_sum = ps
                else:
                    ps_sum += ps
            spectra[label] = (k_bins, ps_sum / B_size)

        colors = {"up(128)": "C0", "up(down(256))": "C1", "native 256": "C2"}
        styles = {"up(128)": "--", "up(down(256))": "-.", "native 256": "-"}
        for label, (k, ps) in spectra.items():
            ax.loglog(k, ps, styles[label], color=colors[label], label=label, lw=1.5)

        # Mark the Nyquist wavenumber for 128 grid
        ax.axvline(64, color="gray", ls=":", lw=1, alpha=0.7)
        ax.text(64, ax.get_ylim()[0] * 5, "k=64\n(128 Nyquist)", fontsize=8,
                ha="center", va="bottom", color="gray")

        ax.set_xlabel("wavenumber k")
        ax.set_ylabel("P(k)")
        ax.set_title(FIELD_NAMES[ch])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Radially averaged power spectra (batch-averaged)", fontsize=14)
    fig.tight_layout()
    spec_path = "simple_diffusion/demo_power_spectra.png"
    fig.savefig(spec_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {spec_path}")

    print("\nDone.")
