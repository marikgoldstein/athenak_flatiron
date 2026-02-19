"""
Simple flow-matching super-resolution trainer for MHD data.

Trains on paired (up_down_256, native_256) data from multi-seed MRI simulations.
At inference, the model receives up(native_128) instead — the domain gap is
monitored during training via wandb visualizations.

Usage:
    source load.sh
    python3 simple_diffusion/trainer.py
    python3 simple_diffusion/trainer.py --channels Bx       # single channel
    python3 simple_diffusion/trainer.py --channels Bx,jz    # subset
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from model import UNetModel
from dataset import (
    MRIPairedDataset, prepare_batch, make_loaders,
    normalize, denormalize, FIELD_NAMES,
)

# ---- Config (defaults, overridable via CLI args) ----
DEFAULTS = dict(
    batch_size=4,
    lr=1e-4,
    total_steps=50_000,
    print_every=100,
    log_every=1000,
    sample_steps=50,
    base_dist="gaussian",   # or "x_lo_plus_noise"
    overfit=True,
    num_workers=4,
    channels=None,          # None = all 8; or e.g. "Bx" / "Bx,jz"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=str, default=DEFAULTS["channels"],
                   help="Comma-separated field names (default: all 8)")
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--total-steps", type=int, default=DEFAULTS["total_steps"])
    p.add_argument("--print-every", type=int, default=DEFAULTS["print_every"])
    p.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    p.add_argument("--sample-steps", type=int, default=DEFAULTS["sample_steps"])
    p.add_argument("--base-dist", type=str, default=DEFAULTS["base_dist"])
    p.add_argument("--overfit", action="store_true", default=DEFAULTS["overfit"])
    p.add_argument("--no-overfit", dest="overfit", action="store_false")
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    return p.parse_args()


# ---- Sampling ----

@torch.no_grad()
def euler_sample(model, x_lo, num_steps=50, device='cuda', base_dist="gaussian"):
    """Euler ODE integration from t=0 to t=1."""
    B = x_lo.shape[0]
    dt = 1.0 / num_steps

    if base_dist == "gaussian":
        x = torch.randn_like(x_lo)
    else:  # x_lo_plus_noise
        x = x_lo + torch.randn_like(x_lo)

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((B,), t_val, device=device)
        model_input = torch.cat([x, x_lo], dim=1)
        v = model(t, model_input)
        x = x + dt * v

    return x


# ---- Visualization ----

def make_comparison_figure(x_lo, x_hi_true, x_hi_gen, channel_names,
                           channels_to_show=None, x_lo_128=None):
    """
    Create a comparison figure (in physical units, already denormalized).
    Rows: up(128) / up(down(256)) / native 256 / generated.
    Columns: samples. One figure per channel.
    """
    if channels_to_show is None:
        channels_to_show = list(range(len(channel_names)))

    figs = {}
    n_samples = x_lo.shape[0]
    n_rows = 4 if x_lo_128 is not None else 3

    for ch_idx in channels_to_show:
        fig, axes = plt.subplots(n_rows, n_samples, figsize=(3 * n_samples, 3 * n_rows))
        if n_samples == 1:
            axes = axes[:, None]

        row_labels = []
        tensors = []
        if x_lo_128 is not None:
            row_labels.append('up(128) [inference input]')
            tensors.append(x_lo_128)
        row_labels += ['up(down(256)) [train input]', 'native 256 [target]', 'generated']
        tensors += [x_lo, x_hi_true, x_hi_gen]

        for row, (label, data) in enumerate(zip(row_labels, tensors)):
            for col in range(n_samples):
                ax = axes[row, col]
                img = data[col, ch_idx].cpu().numpy()
                vmax = max(abs(img.min()), abs(img.max()))
                ax.imshow(img, origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, aspect='equal')
                if col == 0:
                    ax.set_ylabel(label, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f'{channel_names[ch_idx]}', fontsize=14)
        fig.tight_layout()
        figs[channel_names[ch_idx]] = fig
    return figs


# ---- Power spectra ----

def radial_power_spectrum(field_2d):
    """Radially averaged 2D power spectrum. Input: (H, W) numpy array."""
    N = field_2d.shape[0]
    fft2 = np.fft.fft2(field_2d)
    ps2d = np.abs(fft2) ** 2 / N ** 4
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


def make_spectra_figure(tensors_dict, channel_names, channels_to_show=None):
    """Power spectra comparison. Tensors should be in physical units."""
    if channels_to_show is None:
        channels_to_show = list(range(len(channel_names)))

    colors = ["C0", "C1", "C2", "C3", "C4"]
    styles = ["--", "-.", "-", "-", ":"]

    fig, axes = plt.subplots(1, len(channels_to_show),
                             figsize=(6 * len(channels_to_show), 5))
    if len(channels_to_show) == 1:
        axes = [axes]

    for col, ch in enumerate(channels_to_show):
        ax = axes[col]
        for i, (label, tensor) in enumerate(tensors_dict.items()):
            B = tensor.shape[0]
            ps_sum = None
            for b in range(B):
                k_bins, ps = radial_power_spectrum(tensor[b, ch].cpu().numpy())
                ps_sum = ps if ps_sum is None else ps_sum + ps
            ax.loglog(k_bins, ps_sum / B,
                      styles[i % len(styles)], color=colors[i % len(colors)],
                      label=label, lw=1.5)

        ax.axvline(64, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("wavenumber k")
        ax.set_ylabel("P(k)")
        ax.set_title(channel_names[ch])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Power spectra", fontsize=14)
    fig.tight_layout()
    return fig


# ---- Training ----

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Parse channels
    channels = None
    if args.channels is not None:
        channels = [c.strip() for c in args.channels.split(",")]
    print(f"Channels: {channels or 'all 8'}")

    train_seeds = list(range(0, 80))
    val_seeds = list(range(80, 90))
    test_seeds = list(range(90, 100))

    # Data
    if args.overfit:
        train_ds = MRIPairedDataset([0], n_frames=200, channels=channels)
        stats = train_ds.compute_stats()
        from torch.utils.data import DataLoader
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)
    else:
        loader, val_loader, test_loader, stats = make_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers,
            train_seeds=train_seeds, val_seeds=val_seeds, test_seeds=test_seeds,
            channels=channels,
        )

    n_channels = loader.dataset.n_channels
    channel_names = loader.dataset.channel_names
    print(f"Using {n_channels} channels: {channel_names}")

    # Model — in_channels = 2*C (xt + x_lo concat), out_channels = C
    model = UNetModel(
        in_channels=2 * n_channels,
        out_channels=n_channels,
        model_channels=128,
        image_size=256,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(4,),
        num_res_blocks=2,
        dropout=0.0,
        num_heads=4,
        num_head_channels=64,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_name = f'fm_sr_{"overfit" if args.overfit else "full"}_{"_".join(channel_names)}'

    wandb_config = dict(
        batch_size=args.batch_size, lr=args.lr, total_steps=args.total_steps,
        sample_steps=args.sample_steps, base_dist=args.base_dist,
        overfit=args.overfit, n_params=f"{n_params:.1f}M",
        channels=channel_names, n_channels=n_channels,
        train_seeds=str(train_seeds[:5]) + "...",
        num_train_seeds=len(train_seeds),
    )
    wandb.init(entity='marikgoldstein', project='mri', name=run_name, config=wandb_config)

    # Grab a fixed batch for visualization (and for overfit mode)
    data_iter = iter(loader)
    x_128_fix, x_256_fix = next(data_iter)
    up_128_fix, x_256_fix, up_down_256_fix = prepare_batch(x_128_fix, x_256_fix, device)

    # Keep physical-unit copies for visualization
    vis_lo_phys = up_down_256_fix[:4]
    vis_hi_phys = x_256_fix[:4]
    vis_lo_128_phys = up_128_fix[:4]

    # Normalized copies for model input
    vis_lo_norm = normalize(vis_lo_phys, stats)
    vis_hi_norm = normalize(vis_hi_phys, stats)
    vis_lo_128_norm = normalize(vis_lo_128_phys, stats)

    if args.overfit:
        fixed_lo = normalize(up_down_256_fix, stats)
        fixed_hi = normalize(x_256_fix, stats)
        print(f"Overfit mode: training on single batch of {fixed_lo.shape[0]} samples")

    # Which channels to show in figures (all of them)
    show_ch = list(range(n_channels))

    # Training loop
    step = 0
    if not args.overfit:
        data_iter = iter(loader)
    while step < args.total_steps:
        if args.overfit:
            x_lo, x_hi = fixed_lo, fixed_hi
        else:
            try:
                x_128_b, x_256_b = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x_128_b, x_256_b = next(data_iter)
            _, x_hi, x_lo = prepare_batch(x_128_b, x_256_b, device)
            x_lo = normalize(x_lo, stats)
            x_hi = normalize(x_hi, stats)

        B = x_lo.shape[0]

        # Flow matching: x1 = x_hi, x0 = noise (or x_lo + noise)
        x1 = x_hi
        noise = torch.randn_like(x_hi)
        if args.base_dist == "gaussian":
            x0 = noise
        else:
            x0 = x_lo + noise

        eps = 1e-4
        t = torch.rand(B, device=device) * (1 - 2 * eps) + eps
        t_expand = t[:, None, None, None]
        xt = (1 - t_expand) * x0 + t_expand * x1
        target = x1 - x0  # velocity field

        model_input = torch.cat([xt, x_lo], dim=1)  # (B, 2C, H, W)
        pred = model(t, model_input)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % args.print_every == 0:
            print(f"step {step}/{args.total_steps}  loss={loss.item():.6f}")
            wandb.log({'train/loss': loss.item()}, step=step)

        if step % args.log_every == 0:
            model.eval()
            # Generate from training input (normalized) then denormalize
            x_hi_gen_norm = euler_sample(model, vis_lo_norm,
                                         num_steps=args.sample_steps,
                                         device=device, base_dist=args.base_dist)
            x_hi_gen = denormalize(x_hi_gen_norm, stats)

            # Generate from inference input (native 128)
            x_hi_gen_128_norm = euler_sample(model, vis_lo_128_norm,
                                              num_steps=args.sample_steps,
                                              device=device, base_dist=args.base_dist)
            x_hi_gen_128 = denormalize(x_hi_gen_128_norm, stats)

            # Comparison figures (physical units)
            figs = make_comparison_figure(
                vis_lo_phys, vis_hi_phys, x_hi_gen,
                channel_names=channel_names,
                channels_to_show=show_ch,
                x_lo_128=x_hi_gen_128,
            )
            log_dict = {}
            for name, fig in figs.items():
                log_dict[f'samples/{name}'] = wandb.Image(fig)
                plt.close(fig)

            # Power spectra (physical units)
            spec_fig = make_spectra_figure({
                "up(128)": vis_lo_128_phys,
                "up(down(256))": vis_lo_phys,
                "native 256": vis_hi_phys,
                "generated (train)": x_hi_gen,
                "generated (128)": x_hi_gen_128,
            }, channel_names=channel_names, channels_to_show=show_ch)
            log_dict['spectra'] = wandb.Image(spec_fig)
            plt.close(spec_fig)

            # Domain gap metrics (physical units)
            gap_train = F.mse_loss(x_hi_gen, vis_hi_phys).item()
            gap_128 = F.mse_loss(x_hi_gen_128, vis_hi_phys).item()
            log_dict['val/mse_from_train_input'] = gap_train
            log_dict['val/mse_from_128_input'] = gap_128
            wandb.log(log_dict, step=step)
            model.train()
            print(f"  [logged samples at step {step}]  "
                  f"mse_train={gap_train:.6f}  mse_128={gap_128:.6f}")

    # Save final checkpoint
    ckpt = {
        'model_state_dict': model.state_dict(),
        'stats': stats,
        'channel_names': channel_names,
        'step': step,
    }
    torch.save(ckpt, 'simple_diffusion/checkpoint.pt')
    print("Saved checkpoint.")
    wandb.finish()


if __name__ == '__main__':
    main()
