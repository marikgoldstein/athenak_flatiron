"""
DDP flow-matching super-resolution trainer for MHD data.

===============================================================================
PLAN (dumped here so it survives session interrupts)
===============================================================================

1. REMAINING OVERFIT SANITY CHECKS (on existing trainer.py or this file):
   a) Multi-channel:  --channels velx,Bx,jz  (3 channels, loss -> 0?)
   b) EDM2 arch:      --model edm2            (comparable to UNet?)
   c) Logit-normal t:  --time-sampler logit_normal  (no breakage?)
   d) Multi-GPU overfit: torchrun --nproc_per_node=2 ... --overfit  (DDP sanity)

2. THIS FILE — DDP-ified trainer (single-node, multi-GPU via torchrun):
   - Imports pure functions from trainer.py (alpha/sigma, sampling, viz, EMA)
   - Adds distributed helpers (setup/rank/barrier/cleanup)
   - Restructures into Trainer class with clear methods
   - Microbatch gradient accumulation:
       DataLoader delivers local_batch_size items per GPU.
       train_step() splits into microbatches of microbatch_size.
       Gradients accumulate in a loop; DDP sync only on last microstep
       via model.no_sync(). This is transparent to the optimizer —
       one optimizer.step() per local_batch_size, not per microbatch.
       effective_batch_size = local_batch_size * world_size
       (microbatch_size does NOT multiply into effective_batch_size,
       it's just a memory-saving trick to fit local_batch_size on GPU.)

   DDP specifics (matching gf3 patterns):
   - torchrun --standalone --nproc_per_node=GPUS (sbatch sets MASTER_PORT)
   - NCCL backend, rank/world_size from env vars (LOCAL_RANK, RANK, WORLD_SIZE)
   - DDP(model, device_ids=[local_rank], find_unused_parameters=False,
         static_graph=True, gradient_as_bucket_view=True)
   - Seed offset: seed + rank for different data per GPU
   - DistributedSampler + set_epoch() for proper reshuffling
   - is_main() guards: wandb, logging, checkpoints, sample generation
   - barrier() after every rank-0-only operation

3. SBATCH SCRIPT — train_ddp.sbatch:
   - torchrun --standalone --nproc_per_node=${GPUS}
   - Dynamic MASTER_PORT from SLURM_JOB_ID
   - module load python/3.11.11
   - Passes "$@" for CLI overrides

4. FULL TRAINING RUN (after DDP + overfit verified):
   - All 8 channels, 80 train / 10 val / 10 test seeds
   - 4-8 GPUs, ~50k-100k steps, warmup + cosine decay
   - Periodic checkpoints + wandb monitoring

Batch size naming convention:
   --local-batch-size  = samples per GPU per optimizer step (DataLoader batch_size)
   --microbatch-size   = samples per forward/backward pass (memory limit)
   effective_batch_size = local_batch_size * world_size
   n_microsteps        = local_batch_size // microbatch_size  (transparent)
===============================================================================

Usage:
    # Single GPU (debug):
    torchrun --standalone --nproc_per_node=1 simple_diffusion/trainer_ddp.py --overfit

    # Multi-GPU:
    torchrun --standalone --nproc_per_node=4 simple_diffusion/trainer_ddp.py --no-overfit

    # With gradient accumulation (16 samples/GPU, 4 at a time in memory):
    torchrun --standalone --nproc_per_node=4 simple_diffusion/trainer_ddp.py \\
        --local-batch-size 16 --microbatch-size 4 --no-overfit

    # Via SLURM:
    sbatch simple_diffusion/train_ddp.sbatch --no-overfit --channels velx
"""

import argparse
import copy
import math
import os
import uuid
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from unet import UNetModel
from edm2_unet import EDM2UNet, MPConv
from dataset import (
    MRIPairedDataset, prepare_batch,
    normalize, denormalize, FIELD_NAMES,
)


# ---------------------------------------------------------------------------
# Interpolation schedule
# ---------------------------------------------------------------------------
# x_t = alpha(t) * x_0 + sigma(t) * x_1
# v(t,x) = E[dot_alpha(t)*x_0 + dot_sigma(t)*x_1 | x_t = x]

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
    """Convert velocity v(t,x) to score s(t,x) = nabla_x log p_t(x)."""
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


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def make_sample_times(num_steps, t_min, t_max, schedule="linear", device="cuda"):
    """Build integration time grid.

    Args:
        schedule: "linear"    -> uniform spacing
                  "quadratic" -> denser near t_max (data end)
    Returns:
        times: (num_steps,) tensor of integration times.
    """
    if schedule == "linear":
        return torch.linspace(t_min, t_max, num_steps, device=device)
    elif schedule == "quadratic":
        # u in [0,1] uniform, t = t_min + (t_max - t_min) * u^2
        # squares concentrate points near t_min; we flip so density is near t_max
        u = torch.linspace(1.0, 0.0, num_steps, device=device)
        return t_max - (t_max - t_min) * u ** 2
    else:
        raise ValueError(f"Unknown sample schedule: {schedule}")


@torch.no_grad()
def euler_sample(model, x_lo, num_steps=50, device='cuda',
                 base_dist="x_lo_plus_noise", base_noise_scale=0.1,
                 t_min=1e-4, t_max=1 - 1e-4, schedule="linear"):
    """Euler ODE integration from t_min to t_max."""
    B = x_lo.shape[0]
    times = make_sample_times(num_steps, t_min, t_max, schedule, device)

    if base_dist == "gaussian":
        x = torch.randn_like(x_lo)
    else:  # x_lo_plus_noise
        x = x_lo + base_noise_scale * torch.randn_like(x_lo)

    for i, t_val in enumerate(times):
        dt = times[i + 1] - t_val if i < num_steps - 1 else t_max - t_val
        t = t_val.expand(B)
        model_input = torch.cat([x, x_lo], dim=1)
        v = model(t, model_input)
        x = x + dt * v

    return x


@torch.no_grad()
def sde_sample(model, x_lo, num_steps=50, device='cuda',
               base_dist="x_lo_plus_noise", base_noise_scale=0.1,
               t_min=1e-4, t_max=1 - 1e-4, sde_base_delta=0.1, schedule="linear"):
    """Euler-Maruyama SDE integration from t_min to t_max.

    Delta is annealed linearly from `delta` at t_min down to
    `delta * base_noise_scale**2` at t_max.  This compensates for the
    1/(alpha * noise_scale**2) divergence of the score near t_max.
    """
    B = x_lo.shape[0]
    times = make_sample_times(num_steps, t_min, t_max, schedule, device)
    ns2 = base_noise_scale ** 2  # noise_scale^2 for annealing floor

    if base_dist == "gaussian":
        x = torch.randn_like(x_lo)
    else:  # x_lo_plus_noise
        x = x_lo + base_noise_scale * torch.randn_like(x_lo)

    for i, t_val in enumerate(times):
        dt = times[i + 1] - t_val if i < num_steps - 1 else t_max - t_val
        t = t_val.expand(B)
        model_input = torch.cat([x, x_lo], dim=1)
        v = model(t, model_input)

        score = v_to_score(
            v, x, t_val, 
            x_lo=x_lo, 
            noise_scale=base_noise_scale, 
            base_dist=base_dist
        )

        # Linear anneal: delta at t_min -> delta*noise_scale^2 at t_max
        progress = (t_val - t_min) / (t_max - t_min)
        delta_t = sde_base_delta * (1.0 - progress * (1.0 - ns2))
        g_t = (2 * delta_t) ** 0.5

        score_correction = (delta_t * score).clamp(-1e3, 1e3)
        drift = v + score_correction
        if i < num_steps - 1:
            noise = torch.randn_like(x)
            x = x + dt * drift + g_t * dt ** 0.5 * noise
        else:
            x = x + dt * drift

    return x


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_comparison_figure(rows, channel_names, channels_to_show=None):
    """Create a comparison figure (physical units, already denormalized).

    Args:
        rows: list of (label, tensor) tuples. Each tensor is (B, C, H, W).
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
    """Difference images (physical units, already denormalized).

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


# ---------------------------------------------------------------------------
# Power spectra
# ---------------------------------------------------------------------------

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
    """Power spectra comparison.

    Args:
        spectra_list: list of (label, tensor, color, style) tuples.
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


# ---------------------------------------------------------------------------
# Sample logging
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample_logs(model, ema_shadow, lo_norm, lo_phys, hi_phys, native_128,
                         channel_names, show_ch, stats, sample_kwargs,
                         use_sde, sde_kwargs, prefix, input_label):
    """Generate ODE/SDE samples and build wandb log dict.

    Returns:
        (log_dict, mse_msg) where mse_msg is a printable summary string.
    """
    log_dict = {}
    p = prefix

    x_hi_gen = denormalize(euler_sample(model, lo_norm, **sample_kwargs), stats)
    x_hi_ema = denormalize(euler_sample(ema_shadow, lo_norm, **sample_kwargs), stats)
    if use_sde:
        x_hi_sde = denormalize(sde_sample(model, lo_norm, **sde_kwargs), stats)
        x_hi_sde_ema = denormalize(sde_sample(ema_shadow, lo_norm, **sde_kwargs), stats)

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
        comp_rows.append(('SDE', x_hi_sde))
        diff_rows.append(('target - SDE', hi_phys - x_hi_sde))

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
        ema_comp_rows.append(('SDE (EMA)', x_hi_sde_ema))
        ema_diff_rows.append(('target - SDE (EMA)', hi_phys - x_hi_sde_ema))

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
        spectra_list.append(("SDE", x_hi_sde, "C4", "-."))
        spectra_list.append(("SDE (EMA)", x_hi_sde_ema, "C5", "-."))
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
        mse_sde_ema = F.mse_loss(x_hi_sde_ema, hi_phys).item()
        log_dict[f'{p}mse_sde'] = mse_sde
        log_dict[f'{p}mse_sde_ema'] = mse_sde_ema
        msg += f"  sde={mse_sde:.6f}  sde_ema={mse_sde_ema:.6f}"
    return log_dict, msg


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize single-node DDP from torchrun env vars."""
    if "RANK" not in os.environ:
        # Not launched with torchrun — single-GPU fallback
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.barrier()
    return rank, local_rank, world_size


def is_main():
    """True on rank 0 (or when not using DDP)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    """Synchronize all ranks (no-op if not distributed)."""
    if dist.is_initialized():
        dist.barrier()


def cleanup():
    """Destroy process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# EDM2 forced weight normalization (outside the network, after gradient step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _force_normalize_mpconv_weights(model):
    """Project all MPConv weight params back to unit norm (per-filter)."""
    for module in model.modules():
        if isinstance(module, MPConv):
            w = module.mpconv_weight
            dims = tuple(range(1, w.ndim))
            norm = w.to(torch.float32).pow(2).sum(dim=dims, keepdim=True).sqrt()
            norm = norm.clamp(min=1e-8)
            w.data.copy_((w / norm.to(w.dtype)).data)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # OPTIMIZATION
    local_batch_size=4,
    microbatch_size=4,
    base_lr=2e-4,
    lr_schedule="cosine",
    min_lr=1e-7,
    grad_clip=1.0,
    weight_decay=0.0,
    loss_scale=100.0,
    time_sampler="logit_normal",
    logit_normal_mean=0.0,
    logit_normal_std=1.0,
    dropout=0.1,
    total_steps=200_000,
    warmup_steps=10_000,
    force_weight_norm=True,
    t_min_train=1e-4,
    t_max_train=1 - 1e-4,
    # PROBLEM
    overfit=True,
    arch_name="edm2",
    channels="velx",
    base_dist="x_lo_plus_noise",
    base_noise_scale=0.1,
    # SAMPLING
    t_min_sample=1e-4,
    t_max_sample=1 - 1e-4,
    sde_base_delta=0.01,
    sample_steps=100,
    sample_schedule="linear",
    # SYSTEM
    use_bf16=True,
    use_compile=True,
    num_workers=4,
    seed=42,
    # LOGGING 
    log_every=100,
    sample_every=500,
    save_most_recent_every=1000,
    save_periodic_every=5000,
    ckpt_dir="/mnt/home/mgoldstein/ceph/mri",
    # EMA 
    ema_decay=0.9999,
    ema_start_step=10_000,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=str, default=DEFAULTS["channels"])
    p.add_argument("--local-batch-size", type=int,
                   default=DEFAULTS["local_batch_size"],
                   help="Samples per GPU per optimizer step (default: 4)")
    p.add_argument("--microbatch-size", type=int,
                   default=DEFAULTS["microbatch_size"],
                   help="Samples per fwd/bwd pass; must divide local-batch-size "
                        "(default: same as local-batch-size, i.e. no accumulation)")
    p.add_argument("--base-lr", type=float, default=DEFAULTS["base_lr"])
    p.add_argument("--lr-schedule", type=str, default=DEFAULTS["lr_schedule"],
                   choices=["const", "cosine"],
                   help="LR schedule after warmup: 'const' or 'cosine' (default: cosine)")
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--total-steps", type=int, default=DEFAULTS["total_steps"])
    p.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    p.add_argument("--sample-every", type=int, default=DEFAULTS["sample_every"])
    p.add_argument("--sample-steps", type=int, default=DEFAULTS["sample_steps"])
    p.add_argument("--sample-schedule", type=str,
                   default=DEFAULTS["sample_schedule"],
                   choices=["linear", "quadratic"])
    p.add_argument("--base-dist", type=str, default=DEFAULTS["base_dist"])
    p.add_argument("--base-noise-scale", type=float, default=DEFAULTS["base_noise_scale"])
    p.add_argument("--t-min-train", type=float, default=DEFAULTS["t_min_train"])
    p.add_argument("--t-max-train", type=float, default=DEFAULTS["t_max_train"])
    p.add_argument("--t-min-sample", type=float, default=DEFAULTS["t_min_sample"])
    p.add_argument("--t-max-sample", type=float, default=DEFAULTS["t_max_sample"])
    p.add_argument("--loss-scale", type=float, default=DEFAULTS["loss_scale"])
    p.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"])
    p.add_argument("--ema-decay", type=float, default=DEFAULTS["ema_decay"])
    p.add_argument("--ema-start-step", type=int, default=DEFAULTS["ema_start_step"],
                   help="Step at which to start EMA tracking (full copy-in at this step). "
                        "Set to 0 in overfit mode automatically.")
    p.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    p.add_argument("--min-lr", type=float, default=DEFAULTS["min_lr"])
    p.add_argument("--sde-base-delta", type=float, default=DEFAULTS["sde_base_delta"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--time-sampler", type=str, default=DEFAULTS["time_sampler"],
                   choices=["uniform", "logit_normal"])
    p.add_argument("--logit-normal-mean", type=float,
                   default=DEFAULTS["logit_normal_mean"])
    p.add_argument("--logit-normal-std", type=float,
                   default=DEFAULTS["logit_normal_std"])
    p.add_argument("--arch_name", type=str, default=DEFAULTS["arch_name"],
                   choices=["unet", "edm2"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--save-most-recent-every", type=int,
                   default=DEFAULTS["save_most_recent_every"])
    p.add_argument("--save-periodic-every", type=int,
                   default=DEFAULTS["save_periodic_every"])
    p.add_argument("--ckpt-dir", type=str, default=DEFAULTS["ckpt_dir"])
    
    
    p.add_argument("--overfit", action="store_true", default=DEFAULTS["overfit"])
    p.add_argument("--no-overfit", dest="overfit", action="store_false")
    p.add_argument("--use-bf16", action="store_true", default=DEFAULTS["use_bf16"])
    p.add_argument("--no-bf16", dest="use_bf16", action="store_false")
    p.add_argument("--use-compile", action="store_true", default=DEFAULTS["use_compile"])
    p.add_argument("--no-compile", dest="use_compile", action="store_false")
    p.add_argument("--force-weight-norm", action="store_true",
                   default=DEFAULTS["force_weight_norm"])
    p.add_argument("--no-force-weight-norm", dest="force_weight_norm",
                   action="store_false")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, args, rank, local_rank, world_size):
        self.args = args
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        # Batch size bookkeeping
        self.local_batch_size = args.local_batch_size
        self.microbatch_size = args.microbatch_size
        self.effective_batch_size = self.local_batch_size * world_size
        assert self.local_batch_size % self.microbatch_size == 0, (
            f"local_batch_size ({self.local_batch_size}) must be divisible by "
            f"microbatch_size ({self.microbatch_size})"
        )
        self.n_microsteps = self.local_batch_size // self.microbatch_size

        # Seed: offset by rank so each GPU sees different data
        seed = args.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Parse channels
        self.channels = None
        if args.channels is not None:
            self.channels = [c.strip() for c in args.channels.split(",")]

        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_logging()

        self.print(
            f"Rank {rank}/{world_size}, device={self.device}, seed={seed}\n"
            f"  local_batch_size={self.local_batch_size}, "
            f"microbatch_size={self.microbatch_size}, "
            f"n_microsteps={self.n_microsteps}\n"
            f"  effective_batch_size={self.effective_batch_size} "
            f"(= {self.local_batch_size} x {world_size} GPUs)"
        )

    # -- Printing (rank 0 only) --

    def print(self, *a, **kw):
        if is_main():
            print(*a, **kw)

    # -- Data --

    def setup_data(self):
        args = self.args
        train_seeds = list(range(0, 80))
        val_seeds = list(range(80, 90))
        test_seeds = list(range(90, 100))

        if args.overfit:
            train_ds = MRIPairedDataset([0], n_frames=200,
                                        channels=self.channels)
            self.stats = train_ds.compute_stats()
            # No DistributedSampler in overfit: each rank gets the same data
            self.train_loader = DataLoader(
                train_ds, batch_size=self.local_batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True,
            )
            self.val_loader = None
            self.test_loader = None
            self.train_sampler = None
        else:
            train_ds = MRIPairedDataset(train_seeds, n_frames=200,
                                        channels=self.channels)
            val_ds = MRIPairedDataset(val_seeds, n_frames=200,
                                      channels=self.channels)
            test_ds = MRIPairedDataset(test_seeds, n_frames=200,
                                       channels=self.channels)
            self.stats = train_ds.compute_stats()

            if self.world_size > 1:
                self.train_sampler = DistributedSampler(
                    train_ds, num_replicas=self.world_size,
                    rank=self.rank, shuffle=True,
                )
                val_sampler = DistributedSampler(
                    val_ds, num_replicas=self.world_size,
                    rank=self.rank, shuffle=False,
                )
            else:
                self.train_sampler = None
                val_sampler = None

            self.train_loader = DataLoader(
                train_ds, batch_size=self.local_batch_size,
                sampler=self.train_sampler,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_workers, pin_memory=True,
                drop_last=True, persistent_workers=(args.num_workers > 0),
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=self.local_batch_size,
                sampler=val_sampler,
                shuffle=(val_sampler is None),
                num_workers=args.num_workers, pin_memory=True,
                drop_last=True, persistent_workers=(args.num_workers > 0),
            )
            self.test_loader = DataLoader(
                test_ds, batch_size=self.local_batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True,
            )

        self.n_channels = self.train_loader.dataset.n_channels
        self.channel_names = self.train_loader.dataset.channel_names
        self.show_ch = list(range(self.n_channels))
        self.print(f"Channels ({self.n_channels}): {self.channel_names}")

    # -- Model --

    def setup_model(self):
        args = self.args
        dropout = 0.0 if args.overfit else args.dropout

        if args.arch_name == "unet":
            model = UNetModel(
                in_channels=2 * self.n_channels,
                out_channels=self.n_channels,
                model_channels=128, image_size=256,
                channel_mult=(1, 2, 2, 2),
                attention_resolutions=(4,),
                num_res_blocks=2, dropout=dropout,
                num_heads=4, num_head_channels=64,
            )
        elif args.arch_name == "edm2":
            model = EDM2UNet(
                in_channels=2 * self.n_channels,
                out_channels=self.n_channels,
                img_resolution=256, model_channels=128,
                channel_mult=[1, 2, 2, 2], num_blocks=2,
                attn_resolutions=[64], dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown arch_name: {args.arch_name}")

        model = model.to(self.device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        self.print(f"Model: {args.arch_name}, {n_params:.1f}M params")

        # Compile before DDP wrap
        if args.use_compile:
            model = torch.compile(model)

        # EMA tracks the unwrapped model (delayed start: no updates until ema_start_step)
        self.ema = EMA(model, decay=args.ema_decay)
        self.ema_start_step = 0 if args.overfit else args.ema_start_step
        self.ema_started = (self.ema_start_step == 0)

        # Wrap with DDP
        if self.world_size > 1:
            self.model = DDP(
                model, device_ids=[self.local_rank],
                find_unused_parameters=False,
                static_graph=True,
                gradient_as_bucket_view=True,
            )
            self.model_without_ddp = self.model.module
            self.no_sync_ctx = self.model.no_sync
        else:
            self.model = model
            self.model_without_ddp = model
            self.no_sync_ctx = nullcontext

    # -- Optimizer / scheduler --

    def setup_optimizer(self):
        args = self.args
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.base_lr,
            weight_decay=args.weight_decay,
        )
        self.warmup_steps = 0 if args.overfit else args.warmup_steps
        min_ratio = args.min_lr / args.base_lr
        schedule = args.lr_schedule

        def lr_lambda(step):
            # Linear warmup
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return (step + 1) / self.warmup_steps
            # After warmup
            if schedule == "const":
                return 1.0
            # cosine decay from base_lr to min_lr
            progress = (step - self.warmup_steps) / max(
                1, args.total_steps - self.warmup_steps
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda,
        )

    @torch.no_grad()
    def _ema_copy_in(self):
        """Full copy of current model weights into EMA shadow."""
        for s, p in zip(self.ema.shadow.parameters(),
                        self.model_without_ddp.parameters()):
            s.data.copy_(p.data)
        self.print(f"EMA: full copy-in at step {self.ema_start_step}")

    # -- Logging / visualization setup (rank 0) --

    def setup_logging(self):
        args = self.args
        n_vis = 4

        # Grab a fixed batch for visualization
        data_iter = iter(self.train_loader)
        x_128_fix, x_256_fix = next(data_iter)
        _, x_256_fix, up_down_256_fix = prepare_batch(
            x_128_fix, x_256_fix, self.device,
        )

        self.train_vis = dict(
            lo_phys=up_down_256_fix[:n_vis],
            hi_phys=x_256_fix[:n_vis],
            native_128=F.avg_pool2d(x_256_fix[:n_vis], kernel_size=2),
            lo_norm=normalize(up_down_256_fix[:n_vis], self.stats),
            input_label="up(down(256))",
        )

        # Overfit mode: keep fixed normalized batch for training
        if args.overfit:
            self.fixed_lo = normalize(up_down_256_fix, self.stats)
            self.fixed_hi = normalize(x_256_fix, self.stats)
            self.print(f"Overfit mode: fixed batch of {self.fixed_lo.shape[0]}")

        # Test vis (non-overfit only, rank 0)
        self.test_vis = None
        if not args.overfit and is_main():
            test_ds = MRIPairedDataset(
                list(range(90, 93)), n_frames=200, channels=self.channels,
            )
            test_tmp = DataLoader(
                test_ds, batch_size=n_vis, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            x_128_t, x_256_t = next(iter(test_tmp))
            _, x_256_t, up_down_t = prepare_batch(
                x_128_t, x_256_t, self.device,
            )
            self.test_vis = dict(
                lo_phys=up_down_t[:n_vis],
                hi_phys=x_256_t[:n_vis],
                native_128=F.avg_pool2d(x_256_t[:n_vis], kernel_size=2),
                lo_norm=normalize(up_down_t[:n_vis], self.stats),
                input_label="up(down(256))",
            )
            del test_tmp, test_ds

        # wandb (rank 0 only)
        if is_main():
            run_name = (
                f'fm_sr_{"overfit" if args.overfit else "full"}'
                f'_{"_".join(self.channel_names)}'
                f'_gpus{self.world_size}'
            )
            n_params = sum(
                p.numel() for p in self.model_without_ddp.parameters()
            ) / 1e6
            wandb_config = dict(
                local_batch_size=self.local_batch_size,
                microbatch_size=self.microbatch_size,
                n_microsteps=self.n_microsteps,
                effective_batch_size=self.effective_batch_size,
                base_lr=args.base_lr, lr_schedule=args.lr_schedule,
                weight_decay=args.weight_decay,
                total_steps=args.total_steps,
                log_every=args.log_every, sample_every=args.sample_every,
                sample_steps=args.sample_steps,
                sample_schedule=args.sample_schedule,
                base_dist=args.base_dist,
                base_noise_scale=args.base_noise_scale,
                t_min_train=args.t_min_train, t_max_train=args.t_max_train,
                t_min_sample=args.t_min_sample, t_max_sample=args.t_max_sample,
                loss_scale=args.loss_scale, grad_clip=args.grad_clip,
                ema_decay=args.ema_decay, ema_start_step=self.ema_start_step,
                warmup_steps=self.warmup_steps, min_lr=args.min_lr,
                force_weight_norm=args.force_weight_norm,
                use_bf16=args.use_bf16, sde_base_delta=args.sde_base_delta,
                overfit=args.overfit, n_params=f"{n_params:.1f}M",
                channels=self.channel_names, n_channels=self.n_channels,
                world_size=self.world_size,
                arch_name=args.arch_name, dropout=args.dropout, seed=args.seed,
                time_sampler=args.time_sampler,
                logit_normal_mean=args.logit_normal_mean,
                logit_normal_std=args.logit_normal_std,
            )
            wandb.init(
                entity="marikgoldstein", project="mri",
                name=run_name, config=wandb_config,
            )

        # Checkpointing dir: base/YYYY-MM-DD_<random>_overfit|full
        overfit_tag = "overfit" if args.overfit else "full"
        date_str = datetime.now().strftime("%Y-%m-%d")
        short_id = uuid.uuid4().hex[:8]
        self.ckpt_dir = os.path.join(args.ckpt_dir, f"{date_str}_{short_id}_{overfit_tag}")
        if is_main():
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.print(f"Checkpoints -> {self.ckpt_dir}")
        barrier()

    # -- Single training step (with microbatch gradient accumulation) --

    def train_step(self, x_lo, x_hi, step):
        """Forward + loss + backward (microbatched) + clip + step + EMA.

        The full local batch (x_lo, x_hi) of local_batch_size samples is
        split into n_microsteps chunks of microbatch_size.  Each chunk does
        a forward + backward; DDP gradient sync is suppressed on all but the
        last chunk via model.no_sync().  The optimizer steps once at the end.

        Returns (loss_value, grad_norm_value).
        """
        args = self.args
        mb = self.microbatch_size
        n_micro = self.n_microsteps

        self.optimizer.zero_grad()
        total_loss = 0.0

        for i in range(n_micro):
            s = i * mb
            e = s + mb
            mb_lo = x_lo[s:e]
            mb_hi = x_hi[s:e]

            # Flow matching: sample t, build x_t and velocity target
            noise = torch.randn_like(mb_hi)
            if args.base_dist == "gaussian":
                x0 = noise
            else:
                x0 = mb_lo + args.base_noise_scale * noise

            t = sample_timesteps(
                mb, self.device, args.t_min_train, args.t_max_train,
                method=args.time_sampler,
                logit_normal_mean=args.logit_normal_mean,
                logit_normal_std=args.logit_normal_std,
            )
            t_expand = t[:, None, None, None]
            xt = alpha(t_expand) * x0 + sigma(t_expand) * mb_hi
            target = dot_alpha(t_expand) * x0 + dot_sigma(t_expand) * mb_hi

            model_input = torch.cat([xt, mb_lo], dim=1)

            # Suppress DDP gradient sync on all but the last microstep
            sync_ctx = (
                self.no_sync_ctx() if (i < n_micro - 1) else nullcontext()
            )
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=args.use_bf16):
                    pred = self.model(t, model_input)
                    loss = (
                        F.mse_loss(pred, target) * args.loss_scale / n_micro
                    )
                loss.backward()

            total_loss += loss.item()

        # Clip + step (once per full local batch)
        max_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm,
        )
        self.optimizer.step()
        self.scheduler.step()

        # EDM2 forced weight normalization (after gradient step)
        if args.force_weight_norm:
            _force_normalize_mpconv_weights(self.model_without_ddp)

        # EMA: delayed start with full copy-in
        if not self.ema_started and step >= self.ema_start_step:
            self._ema_copy_in()
            self.ema_started = True
        if self.ema_started:
            self.ema.update(self.model_without_ddp)

        return total_loss, grad_norm.item()

    # -- Sample generation + wandb logging (rank 0) --

    def generate_samples(self, step):
        """Generate ODE/SDE samples and log to wandb. Rank 0 only."""
        if not is_main():
            return
        args = self.args

        self.model_without_ddp.eval()
        sample_kwargs = dict(
            num_steps=args.sample_steps, device=self.device,
            base_dist=args.base_dist,
            base_noise_scale=args.base_noise_scale,
            t_min=args.t_min_sample, t_max=args.t_max_sample,
            schedule=args.sample_schedule,
        )
        use_sde = args.sde_base_delta > 0
        sde_kw = dict(**sample_kwargs, sde_base_delta=args.sde_base_delta) if use_sde else {}

        # Train samples
        log_dict, train_msg = generate_sample_logs(
            self.model_without_ddp, self.ema.shadow,
            self.train_vis["lo_norm"], self.train_vis["lo_phys"],
            self.train_vis["hi_phys"], self.train_vis["native_128"],
            self.channel_names, self.show_ch, self.stats,
            sample_kwargs, use_sde, sde_kw,
            prefix="train/", input_label=self.train_vis["input_label"],
        )
        msg = f"  [step {step}]  train: {train_msg}"

        # Test samples
        if self.test_vis is not None:
            test_logs, test_msg = generate_sample_logs(
                self.model_without_ddp, self.ema.shadow,
                self.test_vis["lo_norm"], self.test_vis["lo_phys"],
                self.test_vis["hi_phys"], self.test_vis["native_128"],
                self.channel_names, self.show_ch, self.stats,
                sample_kwargs, use_sde, sde_kw,
                prefix="test/", input_label=self.test_vis["input_label"],
            )
            log_dict.update(test_logs)
            msg += f"  |  test: {test_msg}"

        wandb.log(log_dict, step=step)
        self.model_without_ddp.train()
        print(msg)

    # -- Checkpoint saving --

    def save_ckpt(self, path, step):
        """Save checkpoint from rank 0, then barrier."""
        if is_main():
            ckpt = {
                "model_state_dict": self.model_without_ddp.state_dict(),
                "ema_state_dict": self.ema.shadow.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "stats": self.stats,
                "channel_names": self.channel_names,
                "step": step,
            }
            torch.save(ckpt, path)
        barrier()

    # -- Main training loop --

    def train(self):
        args = self.args
        ckpt_dir = self.ckpt_dir

        # Save init checkpoint
        self.save_ckpt(f"{ckpt_dir}/init.pt", step=0)
        self.print(f"Saved init checkpoint to {ckpt_dir}/init.pt")

        # Log samples at step 0 (init spectrum, etc.)
        self.generate_samples(step=0)
        barrier()

        step = 0
        epoch = 0

        if args.overfit:
            # Fixed batch, no iterator management
            while step < args.total_steps:
                loss, gnorm = self.train_step(self.fixed_lo, self.fixed_hi, step)
                step += 1
                self._maybe_log_and_save(step, loss, gnorm)
        else:
            # Step-based loop; increment epoch when dataloader exhausts
            while step < args.total_steps:
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)

                for x_128_b, x_256_b in self.train_loader:
                    if step >= args.total_steps:
                        break

                    _, x_hi, x_lo = prepare_batch(
                        x_128_b, x_256_b, self.device,
                    )
                    x_lo = normalize(x_lo, self.stats)
                    x_hi = normalize(x_hi, self.stats)

                    loss, gnorm = self.train_step(x_lo, x_hi, step)
                    step += 1
                    self._maybe_log_and_save(step, loss, gnorm)

                epoch += 1

        # Final checkpoint
        self.save_ckpt(f"{ckpt_dir}/final.pt", step=step)
        self.print(f"Saved final checkpoint to {ckpt_dir}/final.pt")

        if is_main():
            wandb.finish()

    def _maybe_log_and_save(self, step, loss, gnorm):
        """Periodic printing, wandb logging, sample generation, checkpointing."""
        args = self.args

        if step % args.log_every == 0:
            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.print(
                f"step {step}/{args.total_steps}  loss={loss:.6f}  "
                f"grad_norm={gnorm:.4f}  lr={cur_lr:.2e}"
            )
            if is_main():
                wandb.log(
                    {"train/loss": loss, "train/grad_norm": gnorm,
                     "train/lr": cur_lr},
                    step=step,
                )

        if step % args.sample_every == 0:
            self.generate_samples(step)
            barrier()  # other ranks wait for rank 0 to finish sampling

        if step % args.save_most_recent_every == 0:
            self.save_ckpt(f"{self.ckpt_dir}/latest.pt", step)

        if step % args.save_periodic_every == 0:
            self.save_ckpt(f"{self.ckpt_dir}/step_{step:07d}.pt", step)
            self.print(f"Saved periodic checkpoint at step {step}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    
    # TODO: other things like this?
    torch.set_float32_matmul_precision('high')


    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    trainer = Trainer(args, rank, local_rank, world_size)
    try:
        trainer.train()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
