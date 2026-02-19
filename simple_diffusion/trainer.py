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
import copy
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from unet import UNetModel
from edm2_unet import EDM2UNet
from dataset import (
    MRIPairedDataset, prepare_batch, make_loaders,
    normalize, denormalize, FIELD_NAMES,
)

# ---- Config (defaults, overridable via CLI args) ----
DEFAULTS = dict(
    batch_size=4,
    lr=2e-4,
    weight_decay=0.0,
    total_steps=50_000,
    print_every=100,
    log_every=100,
    sample_steps=100,
    base_dist="x_lo_plus_noise",  # or "gaussian"
    base_noise_scale=0.1,         # noise scale for x_lo_plus_noise base dist
    t_min_train=1e-4,             # avoid t=0 during training
    t_max_train=1 - 1e-4,         # avoid t=1 during training
    t_min_sample=1e-4,            # start of Euler integration
    t_max_sample=1 - 1e-4,        # end of Euler integration
    loss_scale=100.0,             # scale loss to avoid Adam epsilon regime
    grad_clip=1.0,                # max grad norm (0 = no clipping)
    ema_decay=0.9999,             # EMA decay rate
    warmup_steps=10_000,          # linear LR warmup (0 in overfit mode)
    use_bf16=True,                # bf16 autocast for forward/loss
    sde_delta=0.1,                # SDE diffusion coeff (0 = no SDE sampling)
    overfit=True,
    num_workers=4,
    channels="velx",              # None = all 8; or e.g. "Bx" / "Bx,jz"
    time_sampler="uniform",       # "uniform" or "logit_normal"
    logit_normal_mean=0.0,        # mu for logit-normal (0 = symmetric)
    logit_normal_std=1.0,         # sigma for logit-normal (1=mid-peak, 2+=U-shape)
    model="unet",                 # "unet" or "edm2"
    dropout=0.0,                  # ResBlock dropout (0.1 is reasonable)
    seed=42,                      # random seed for reproducibility
    save_most_recent_every=1000,  # overwrite latest.pt
    save_periodic_every=5000,     # step-specific checkpoint (no overwrite)
    ckpt_dir="simple_diffusion/checkpoints",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=str, default=DEFAULTS["channels"],
                   help="Comma-separated field names (default: velx)")
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--total-steps", type=int, default=DEFAULTS["total_steps"])
    p.add_argument("--print-every", type=int, default=DEFAULTS["print_every"])
    p.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    p.add_argument("--sample-steps", type=int, default=DEFAULTS["sample_steps"])
    p.add_argument("--base-dist", type=str, default=DEFAULTS["base_dist"])
    p.add_argument("--base-noise-scale", type=float, default=DEFAULTS["base_noise_scale"],
                   help="Noise scale for x_lo_plus_noise base dist (default: 0.1)")
    p.add_argument("--t-min-train", type=float, default=DEFAULTS["t_min_train"],
                   help="Min t during training (default: 1e-4)")
    p.add_argument("--t-max-train", type=float, default=DEFAULTS["t_max_train"],
                   help="Max t during training (default: 1-1e-4)")
    p.add_argument("--t-min-sample", type=float, default=DEFAULTS["t_min_sample"],
                   help="Start t for Euler sampling (default: 1e-4)")
    p.add_argument("--t-max-sample", type=float, default=DEFAULTS["t_max_sample"],
                   help="End t for Euler sampling (default: 1-1e-4)")
    p.add_argument("--loss-scale", type=float, default=DEFAULTS["loss_scale"],
                   help="Multiply loss before backward (default: 100)")
    p.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"],
                   help="Max grad norm for clipping (0=off, default: 1.0)")
    p.add_argument("--ema-decay", type=float, default=DEFAULTS["ema_decay"],
                   help="EMA decay rate (default: 0.9999)")
    p.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"],
                   help="Linear LR warmup steps (default: 10000, 0 in overfit)")
    p.add_argument("--use-bf16", action="store_true", default=DEFAULTS["use_bf16"])
    p.add_argument("--no-bf16", dest="use_bf16", action="store_false")
    p.add_argument("--sde-delta", type=float, default=DEFAULTS["sde_delta"],
                   help="SDE diffusion coeff delta (0=skip SDE sampling, default: 0.1)")
    p.add_argument("--overfit", action="store_true", default=DEFAULTS["overfit"])
    p.add_argument("--no-overfit", dest="overfit", action="store_false")
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--time-sampler", type=str, default=DEFAULTS["time_sampler"],
                   choices=["uniform", "logit_normal"],
                   help="Timestep sampling: uniform or logit_normal (default: uniform)")
    p.add_argument("--logit-normal-mean", type=float,
                   default=DEFAULTS["logit_normal_mean"],
                   help="Logit-normal mu (default: 0.0)")
    p.add_argument("--logit-normal-std", type=float,
                   default=DEFAULTS["logit_normal_std"],
                   help="Logit-normal sigma: 1=mid-peak, 2+=U-shape (default: 1.0)")
    p.add_argument("--model", type=str, default=DEFAULTS["model"],
                   choices=["unet", "edm2"],
                   help="Model architecture (default: unet)")
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"],
                   help="ResBlock dropout rate (default: 0.0)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                   help="Random seed (default: 42)")
    p.add_argument("--save-most-recent-every", type=int,
                   default=DEFAULTS["save_most_recent_every"],
                   help="Save latest.pt every N steps (default: 1000)")
    p.add_argument("--save-periodic-every", type=int,
                   default=DEFAULTS["save_periodic_every"],
                   help="Save step-specific checkpoint every N steps (default: 5000)")
    p.add_argument("--ckpt-dir", type=str, default=DEFAULTS["ckpt_dir"],
                   help="Checkpoint directory (default: simple_diffusion/checkpoints)")
    return p.parse_args()


# ---- Interpolation schedule ----
# x_t = alpha(t) * x_0 + sigma(t) * x_1
# v(t,x) = E[dot_alpha(t)*x_0 + dot_sigma(t)*x_1 | x_t = x]
# To change the schedule, replace these four functions.

def alpha(t):
    """Coefficient on x_0 (base). Linear: alpha(t) = 1 - t."""
    return 1 - t

def sigma(t):
    """Coefficient on x_1 (data). Linear: sigma(t) = t."""
    return t

def dot_alpha(t):
    """d/dt alpha(t). Linear: -1."""
    return -1.0

def dot_sigma(t):
    """d/dt sigma(t). Linear: 1."""
    return 1.0


def sample_timesteps(B, device, t_min, t_max, method="uniform",
                     logit_normal_mean=0.0, logit_normal_std=1.0):
    """Sample training timesteps.

    Args:
        method: "uniform" -> U(t_min, t_max)
                "logit_normal" -> sigmoid(N(mu, sigma^2)), clamped to [t_min, t_max]
                    sigma ~ 1: peaks in middle (SD3 style)
                    sigma ~ 2+: U-shaped, mass near 0 and 1
    """
    if method == "uniform":
        return torch.rand(B, device=device) * (t_max - t_min) + t_min
    elif method == "logit_normal":
        z = torch.randn(B, device=device) * logit_normal_std + logit_normal_mean
        t = torch.sigmoid(z)
        return t.clamp(t_min, t_max)
    else:
        raise ValueError(f"Unknown time sampler: {method}")


def v_to_score(v, x, t, x_lo=None, noise_scale=1.0, base_dist="gaussian"):
    """Convert velocity v(t,x) to score s(t,x) = nabla_x log p_t(x).

    Derivation (general alpha/sigma):
        hat_x0 = E[x_0 | x_t=x] = (dot_sigma(t)*x - sigma(t)*v) / D(t)
        where D(t) = alpha(t)*dot_sigma(t) - sigma(t)*dot_alpha(t)  (Wronskian)

        For gaussian base (x_0 ~ N(0,I)):
            s = -hat_x0 / alpha(t)
        For x_lo_plus_noise base (x_0 ~ N(x_lo, noise_scale^2 I)):
            s = -(hat_x0 - x_lo) / (alpha(t) * noise_scale^2)
    """
    a = alpha(t)
    s = sigma(t)
    da = dot_alpha(t)
    ds = dot_sigma(t)
    D = a * ds - s * da  # Wronskian; = 1 for linear schedule
    hat_x0 = (ds * x - s * v) / D

    if base_dist == "gaussian":
        return -hat_x0 / a
    else:  # x_lo_plus_noise
        return -(hat_x0 - x_lo) / (a * noise_scale ** 2)


# ---- Checkpointing ----

def save_checkpoint(path, model, ema, optimizer, stats, channel_names, step,
                    scheduler=None):
    """Save training checkpoint (model, EMA, optimizer, scheduler, metadata)."""
    ckpt = {
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.shadow.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'channel_names': channel_names,
        'step': step,
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(ckpt, path)


# ---- Sampling ----

@torch.no_grad()
def euler_sample(model, x_lo, num_steps=50, device='cuda',
                 base_dist="x_lo_plus_noise", base_noise_scale=0.1,
                 t_min=1e-4, t_max=1 - 1e-4):
    """Euler ODE integration from t_min to t_max."""
    B = x_lo.shape[0]
    times = torch.linspace(t_min, t_max, num_steps, device=device)
    dt = times[1] - times[0]

    if base_dist == "gaussian":
        x = torch.randn_like(x_lo)
    else:  # x_lo_plus_noise
        x = x_lo + base_noise_scale * torch.randn_like(x_lo)

    for t_val in times:
        t = t_val.expand(B)
        model_input = torch.cat([x, x_lo], dim=1)
        v = model(t, model_input)
        x = x + dt * v

    return x


@torch.no_grad()
def sde_sample(model, x_lo, num_steps=50, device='cuda',
               base_dist="x_lo_plus_noise", base_noise_scale=0.1,
               t_min=1e-4, t_max=1 - 1e-4, delta=0.1):
    """Euler-Maruyama SDE integration from t_min to t_max.

    Uses the SDE with matching marginals to the ODE:
        dx = [v + delta * s] dt + sqrt(2*delta) dW

    where the score s(t,x) = nabla_x log p_t(x) is derived from v(t,x)
    via v_to_score() using the general alpha/sigma schedule.
    """
    B = x_lo.shape[0]
    times = torch.linspace(t_min, t_max, num_steps, device=device)
    dt = times[1] - times[0]
    g = (2 * delta) ** 0.5

    if base_dist == "gaussian":
        x = torch.randn_like(x_lo)
    else:  # x_lo_plus_noise
        x = x_lo + base_noise_scale * torch.randn_like(x_lo)

    for t_val in times:
        t = t_val.expand(B)
        model_input = torch.cat([x, x_lo], dim=1)
        v = model(t, model_input)

        # Convert v -> score via general alpha/sigma schedule
        score = v_to_score(v, x, t_val, x_lo=x_lo,
                           noise_scale=base_noise_scale, base_dist=base_dist)

        drift = v + delta * score
        noise = torch.randn_like(x)
        x = x + dt * drift + g * dt ** 0.5 * noise

    return x


# ---- Visualization ----

def make_comparison_figure(rows, channel_names, channels_to_show=None):
    """
    Create a comparison figure (in physical units, already denormalized).

    Args:
        rows: list of (label, tensor) tuples. Each tensor is (B, C, H, W).
        channel_names: list of channel name strings.
        channels_to_show: which channel indices to plot.

    Returns dict of {channel_name: figure}.
    Columns = batch samples, rows = the provided (label, tensor) pairs.
    """
    if channels_to_show is None:
        channels_to_show = list(range(len(channel_names)))

    figs = {}
    n_rows = len(rows)
    n_samples = rows[0][1].shape[0]

    for ch_idx in channels_to_show:
        fig, axes = plt.subplots(n_rows, n_samples,
                                 figsize=(3 * n_samples, 3 * n_rows))
        if n_samples == 1:
            axes = axes[:, None]

        for row, (label, data) in enumerate(rows):
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


def make_diff_figure(rows, channel_names, channels_to_show=None):
    """
    Difference images (physical units, already denormalized).

    Args:
        rows: list of (label, diff_tensor) tuples. Each tensor is (B, C, H, W).
        channel_names: list of channel name strings.
        channels_to_show: which channel indices to plot.

    Returns dict of {channel_name: figure}.
    """
    if channels_to_show is None:
        channels_to_show = list(range(len(channel_names)))

    figs = {}
    n_rows = len(rows)
    n_samples = rows[0][1].shape[0]

    for ch_idx in channels_to_show:
        fig, axes = plt.subplots(n_rows, n_samples,
                                 figsize=(3 * n_samples, 3 * n_rows))
        if n_samples == 1:
            axes = axes[:, None]

        for row, (label, diff) in enumerate(rows):
            for col in range(n_samples):
                ax = axes[row, col]
                img = diff[col, ch_idx].cpu().numpy()
                vmax = max(abs(img.min()), abs(img.max()))
                if vmax == 0:
                    vmax = 1e-8
                ax.imshow(img, origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, aspect='equal')
                if col == 0:
                    ax.set_ylabel(label, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f'{channel_names[ch_idx]} (differences)', fontsize=14)
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


def make_spectra_figure(spectra_list, channel_names, channels_to_show=None):
    """Power spectra comparison. Handles mixed resolutions on same plot.

    Args:
        spectra_list: list of (label, tensor, color, style) tuples.
            tensor is (B, C, H, W) — H can differ between entries.
        channel_names: list of channel name strings.
        channels_to_show: which channel indices to plot.
    """
    if channels_to_show is None:
        channels_to_show = list(range(len(channel_names)))

    fig, axes = plt.subplots(1, len(channels_to_show),
                             figsize=(6 * len(channels_to_show), 5))
    if len(channels_to_show) == 1:
        axes = [axes]

    for col, ch in enumerate(channels_to_show):
        ax = axes[col]
        for label, tensor, color, style in spectra_list:
            B = tensor.shape[0]
            ps_sum = None
            for b in range(B):
                k_bins, ps = radial_power_spectrum(tensor[b, ch].cpu().numpy())
                ps_sum = ps if ps_sum is None else ps_sum + ps
            ax.loglog(k_bins, ps_sum / B, style, color=color, label=label, lw=1.5)

        ax.axvline(64, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("wavenumber k")
        ax.set_ylabel("P(k)")
        ax.set_title(channel_names[ch])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Power spectra", fontsize=14)
    fig.tight_layout()
    return fig


# ---- Sample logging ----

@torch.no_grad()
def generate_sample_logs(model, ema_shadow, lo_norm, lo_phys, hi_phys, native_128,
                         channel_names, show_ch, stats, sample_kwargs,
                         use_sde, sde_kwargs, prefix, input_label):
    """Generate ODE/SDE samples and build wandb log dict.

    Args:
        lo_norm: (B, C, 256, 256) normalized low-res input for the model.
        lo_phys: (B, C, 256, 256) physical-unit low-res for plots.
        hi_phys: (B, C, 256, 256) physical-unit high-res target.
        native_128: (B, C, 128, 128) native 128 for power spectra.
        prefix: wandb key prefix, e.g. "" or "test/".
        input_label: label for the input row, e.g. "up(down(256))" or "up(128)".

    Returns:
        (log_dict, mse_msg) where mse_msg is a printable summary string.
    """
    log_dict = {}
    p = prefix  # shorthand

    x_hi_gen = denormalize(euler_sample(model, lo_norm, **sample_kwargs), stats)
    x_hi_ema = denormalize(euler_sample(ema_shadow, lo_norm, **sample_kwargs), stats)
    if use_sde:
        x_hi_sde = denormalize(sde_sample(ema_shadow, lo_norm, **sde_kwargs), stats)

    # Raw model: comparison + diffs
    comp_rows = [
        (f'input [{input_label}]', lo_phys),
        ('target [native 256]', hi_phys),
        ('ODE', x_hi_gen),
    ]
    diff_rows = [
        ('target - input', hi_phys - lo_phys),
        ('target - ODE', hi_phys - x_hi_gen),
    ]
    if use_sde:
        comp_rows.append(('SDE (EMA)', x_hi_sde))
        diff_rows.append(('target - SDE (EMA)', hi_phys - x_hi_sde))

    for name, fig in make_comparison_figure(comp_rows, channel_names, show_ch).items():
        log_dict[f'{p}samples/{name}'] = wandb.Image(fig)
        plt.close(fig)
    for name, fig in make_diff_figure(diff_rows, channel_names, show_ch).items():
        log_dict[f'{p}diffs/{name}'] = wandb.Image(fig)
        plt.close(fig)

    # EMA model: comparison + diffs
    ema_comp_rows = [
        (f'input [{input_label}]', lo_phys),
        ('target [native 256]', hi_phys),
        ('ODE (EMA)', x_hi_ema),
    ]
    ema_diff_rows = [
        ('target - input', hi_phys - lo_phys),
        ('target - ODE (EMA)', hi_phys - x_hi_ema),
    ]
    if use_sde:
        ema_comp_rows.append(('SDE (EMA)', x_hi_sde))
        ema_diff_rows.append(('target - SDE (EMA)', hi_phys - x_hi_sde))

    for name, fig in make_comparison_figure(ema_comp_rows, channel_names, show_ch).items():
        log_dict[f'{p}samples_ema/{name}'] = wandb.Image(fig)
        plt.close(fig)
    for name, fig in make_diff_figure(ema_diff_rows, channel_names, show_ch).items():
        log_dict[f'{p}diffs_ema/{name}'] = wandb.Image(fig)
        plt.close(fig)

    # Power spectra
    spectra_list = [
        ("truth (256)", hi_phys, "C0", "-"),
        ("ODE", x_hi_gen, "C1", "--"),
        ("ODE (EMA)", x_hi_ema, "C2", "--"),
    ]
    if use_sde:
        spectra_list.append(("SDE (EMA)", x_hi_sde, "C4", "-."))
    spectra_list.append(("low resolution (128)", native_128, "C3", "-"))
    spec_fig = make_spectra_figure(spectra_list, channel_names, show_ch)
    log_dict[f'{p}spectra'] = wandb.Image(spec_fig)
    plt.close(spec_fig)

    # MSE metrics (physical units)
    mse = F.mse_loss(x_hi_gen, hi_phys).item()
    mse_ema = F.mse_loss(x_hi_ema, hi_phys).item()
    log_dict[f'{p}mse'] = mse
    log_dict[f'{p}mse_ema'] = mse_ema
    msg = f"mse={mse:.6f}  ema={mse_ema:.6f}"
    if use_sde:
        mse_sde = F.mse_loss(x_hi_sde, hi_phys).item()
        log_dict[f'{p}mse_sde'] = mse_sde
        msg += f"  sde={mse_sde:.6f}"
    return log_dict, msg


# ---- EMA ----

class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# ---- Training ----

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Seed: {args.seed}")

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
    if args.model == "unet":
        model = UNetModel(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            model_channels=128,
            image_size=256,
            channel_mult=(1, 2, 2, 2),
            attention_resolutions=(4,),     # attention at ds=4 (64x64)
            num_res_blocks=2,
            dropout=args.dropout,
            num_heads=4,
            num_head_channels=64,
        ).to(device)
    elif args.model == "edm2":
        model = EDM2UNet(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            img_resolution=256,
            model_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_blocks=2,
            attn_resolutions=[64],          # attention at 64x64 spatial res
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    ema = EMA(model, decay=args.ema_decay)
    warmup_steps = 0 if args.overfit else args.warmup_steps

    run_name = f'fm_sr_{"overfit" if args.overfit else "full"}_{"_".join(channel_names)}'

    wandb_config = dict(
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        total_steps=args.total_steps, sample_steps=args.sample_steps,
        base_dist=args.base_dist, base_noise_scale=args.base_noise_scale,
        t_min_train=args.t_min_train, t_max_train=args.t_max_train,
        t_min_sample=args.t_min_sample, t_max_sample=args.t_max_sample,
        loss_scale=args.loss_scale, grad_clip=args.grad_clip,
        ema_decay=args.ema_decay, warmup_steps=warmup_steps,
        use_bf16=args.use_bf16, sde_delta=args.sde_delta,
        overfit=args.overfit, n_params=f"{n_params:.1f}M",
        channels=channel_names, n_channels=n_channels,
        train_seeds=str(train_seeds[:5]) + "...",
        num_train_seeds=len(train_seeds),
        save_most_recent_every=args.save_most_recent_every,
        save_periodic_every=args.save_periodic_every,
        model=args.model, dropout=args.dropout, seed=args.seed,
        time_sampler=args.time_sampler,
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
    )
    wandb.init(entity='marikgoldstein', project='mri', name=run_name, config=wandb_config)

    # Grab a fixed batch for train visualization (and for overfit mode)
    data_iter = iter(loader)
    x_128_fix, x_256_fix = next(data_iter)
    _, x_256_fix, up_down_256_fix = prepare_batch(x_128_fix, x_256_fix, device)

    # Train vis: model input is up(down(256)), target is native 256
    n_vis = 4
    train_vis = dict(
        lo_phys=up_down_256_fix[:n_vis],            # up(down(256)) at 256x256
        hi_phys=x_256_fix[:n_vis],                  # native 256
        native_128=F.avg_pool2d(x_256_fix[:n_vis], kernel_size=2),  # for spectrum
        lo_norm=normalize(up_down_256_fix[:n_vis], stats),
        input_label="up(down(256))",
    )

    if args.overfit:
        fixed_lo = normalize(up_down_256_fix, stats)
        fixed_hi = normalize(x_256_fix, stats)
        print(f"Overfit mode: training on single batch of {fixed_lo.shape[0]} samples")

    # Test vis: model input is up(native_128) — the real deployment scenario
    # Only set up when not overfitting; grab a batch then discard the loader
    test_vis = None
    if not args.overfit:
        from torch.utils.data import DataLoader
        test_ds = MRIPairedDataset(test_seeds[:3], n_frames=200, channels=channels)
        test_tmp = DataLoader(test_ds, batch_size=n_vis, shuffle=True,
                              num_workers=0, pin_memory=True)
        x_128_test, x_256_test = next(iter(test_tmp))
        up_128_test, x_256_test, _ = prepare_batch(x_128_test, x_256_test, device)
        test_vis = dict(
            lo_phys=up_128_test[:n_vis],             # up(native_128) at 256x256
            hi_phys=x_256_test[:n_vis],              # native 256 from test seed
            native_128=x_128_test[:n_vis].to(device), # native 128x128 for spectrum
            lo_norm=normalize(up_128_test[:n_vis], stats),
            input_label="up(native 128)",
        )
        del test_tmp, test_ds
        print(f"Test vis: {len(test_seeds[:3])} seeds, {n_vis} samples")

    # Which channels to show in figures (all of them)
    show_ch = list(range(n_channels))

    # LR schedule: linear warmup then cosine decay to 0
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, args.total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpointing setup
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    save_checkpoint(f'{ckpt_dir}/init.pt', model, ema, optimizer, stats,
                    channel_names, step=0, scheduler=scheduler)
    print(f"Saved init checkpoint to {ckpt_dir}/init.pt")

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

        # Flow matching: x1 = x_hi, x0 = noise (or x_lo + scaled noise)
        x1 = x_hi
        noise = torch.randn_like(x_hi)
        if args.base_dist == "gaussian":
            x0 = noise
        else:
            x0 = x_lo + args.base_noise_scale * noise

        t = sample_timesteps(B, device, args.t_min_train, args.t_max_train,
                             method=args.time_sampler,
                             logit_normal_mean=args.logit_normal_mean,
                             logit_normal_std=args.logit_normal_std)
        t_expand = t[:, None, None, None]
        xt = alpha(t_expand) * x0 + sigma(t_expand) * x1
        target = dot_alpha(t_expand) * x0 + dot_sigma(t_expand) * x1  # velocity

        model_input = torch.cat([xt, x_lo], dim=1)  # (B, 2C, H, W)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.use_bf16):
            pred = model(t, model_input)
            loss = F.mse_loss(pred, target) * args.loss_scale
        optimizer.zero_grad()
        loss.backward()
        max_norm = args.grad_clip if args.grad_clip > 0 else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        scheduler.step()
        ema.update(model)
        step += 1

        if step % args.print_every == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"step {step}/{args.total_steps}  loss={loss.item():.6f}  grad_norm={grad_norm.item():.4f}  lr={cur_lr:.2e}")
            wandb.log({'train/loss': loss.item(),
                       'train/grad_norm': grad_norm.item(),
                       'train/lr': cur_lr}, step=step)

        if step % args.log_every == 0:
            model.eval()
            sample_kwargs = dict(num_steps=args.sample_steps, device=device,
                                 base_dist=args.base_dist,
                                 base_noise_scale=args.base_noise_scale,
                                 t_min=args.t_min_sample, t_max=args.t_max_sample)
            use_sde = args.sde_delta > 0
            sde_kw = dict(**sample_kwargs, delta=args.sde_delta) if use_sde else {}

            # Train samples
            log_dict, train_msg = generate_sample_logs(
                model, ema.shadow, train_vis['lo_norm'], train_vis['lo_phys'],
                train_vis['hi_phys'], train_vis['native_128'],
                channel_names, show_ch, stats, sample_kwargs,
                use_sde, sde_kw, prefix="train/", input_label=train_vis['input_label'])
            msg = f"  [step {step}]  train: {train_msg}"

            # Test samples (only when not overfitting)
            if test_vis is not None:
                test_logs, test_msg = generate_sample_logs(
                    model, ema.shadow, test_vis['lo_norm'], test_vis['lo_phys'],
                    test_vis['hi_phys'], test_vis['native_128'],
                    channel_names, show_ch, stats, sample_kwargs,
                    use_sde, sde_kw, prefix="test/", input_label=test_vis['input_label'])
                log_dict.update(test_logs)
                msg += f"  |  test: {test_msg}"

            wandb.log(log_dict, step=step)
            model.train()
            print(msg)

        # Checkpointing during training
        if step % args.save_most_recent_every == 0:
            save_checkpoint(f'{ckpt_dir}/latest.pt', model, ema, optimizer,
                            stats, channel_names, step, scheduler)
        if step % args.save_periodic_every == 0:
            save_checkpoint(f'{ckpt_dir}/step_{step:07d}.pt', model, ema,
                            optimizer, stats, channel_names, step, scheduler)
            print(f"Saved periodic checkpoint at step {step}")

    # Save final checkpoint
    save_checkpoint(f'{ckpt_dir}/final.pt', model, ema, optimizer, stats,
                    channel_names, step, scheduler)
    print(f"Saved final checkpoint to {ckpt_dir}/final.pt")
    wandb.finish()


if __name__ == '__main__':
    main()
