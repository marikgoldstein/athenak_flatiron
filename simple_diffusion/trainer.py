"""
Simple flow-matching super-resolution trainer for MHD data.

Usage:
    source load.sh
    python3 simple_diffusion/trainer.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from model import UNetModel

# ---- Config ----
data_path = '/mnt/home/mgoldstein/ceph/athenak/feb13/npy/all_fields_256x256.npy'
batch_size = 4
lr = 1e-4
total_steps = 50_000
print_every = 100
log_every = 1000
sample_steps = 50
base_dist = "gaussian"  # or "x_lo_plus_noise"
overfit = True # False          # if True, train on a single fixed batch

FIELD_NAMES = ['density', 'velx', 'vely', 'velz', 'Bx', 'By', 'Bz', 'jz']


# ---- Dataset ----

class MRISuperResDataset(Dataset):
    def __init__(self, data_path):
        # data is (N, 256, 256, 8) channels-last float32
        raw = np.load(data_path)
        # -> (N, 8, 256, 256) channels-first
        self.data = torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous()
        print(f"Dataset: {self.data.shape}, dtype={self.data.dtype}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x_hi = self.data[idx]  # (8, 256, 256)
        # Create low-res: avg_pool to 128x128, then bilinear upsample to 256x256
        x_lo = F.avg_pool2d(x_hi.unsqueeze(0), kernel_size=2).squeeze(0)  # (8, 128, 128)
        x_lo = F.interpolate(x_lo.unsqueeze(0), size=256, mode='bilinear',
                             align_corners=False).squeeze(0)  # (8, 256, 256)
        return x_lo, x_hi


# ---- Sampling ----

@torch.no_grad()
def euler_sample(model, x_lo, num_steps=50, device='cuda'):
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
        model_input = torch.cat([x, x_lo], dim=1)  # (B, 16, 256, 256)
        v = model(t, model_input)
        x = x + dt * v

    return x


# ---- Visualization ----

def make_comparison_figure(x_lo, x_hi_true, x_hi_gen, channels_to_show=(1, 4)):
    """
    Create a comparison figure: rows = lo / true hi / generated hi,
    columns = samples, one figure per channel.
    """
    figs = {}
    n_samples = x_lo.shape[0]
    for ch_idx in channels_to_show:
        fig, axes = plt.subplots(3, n_samples, figsize=(3 * n_samples, 9))
        if n_samples == 1:
            axes = axes[:, None]
        row_labels = ['LR input', 'HR true', 'HR generated']
        tensors = [x_lo, x_hi_true, x_hi_gen]
        for row, (label, data) in enumerate(zip(row_labels, tensors)):
            for col in range(n_samples):
                ax = axes[row, col]
                img = data[col, ch_idx].cpu().numpy()
                vmax = max(abs(img.min()), abs(img.max()))
                ax.imshow(img, origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, aspect='equal')
                if col == 0:
                    ax.set_ylabel(label, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f'{FIELD_NAMES[ch_idx]}', fontsize=14)
        fig.tight_layout()
        figs[FIELD_NAMES[ch_idx]] = fig
    return figs


# ---- Training ----

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    dataset = MRISuperResDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = UNetModel(
        in_channels=16,
        out_channels=8,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    run_name = 'flow_matching_sr_overfit' if overfit else 'flow_matching_sr'
    wandb.init(entity='marikgoldstein', project='mri', name=run_name,
               config=dict(batch_size=batch_size, lr=lr, total_steps=total_steps,
                           sample_steps=sample_steps, base_dist=base_dist,
                           overfit=overfit, n_params=f"{n_params:.1f}M"))

    if overfit:
        # Grab one fixed batch for both training and visualization
        data_iter = iter(loader)
        fixed_lo, fixed_hi = next(data_iter)
        fixed_lo = fixed_lo.to(device)
        fixed_hi = fixed_hi.to(device)
        vis_lo = fixed_lo[:4]
        vis_hi = fixed_hi[:4]
        print(f"Overfit mode: training on single batch of {fixed_lo.shape[0]} samples")
    else:
        # Fixed samples for visualization (first 4)
        vis_lo, vis_hi = [], []
        for i in range(4):
            lo, hi = dataset[i]
            vis_lo.append(lo)
            vis_hi.append(hi)
        vis_lo = torch.stack(vis_lo).to(device)
        vis_hi = torch.stack(vis_hi).to(device)

    # Training loop
    step = 0
    if not overfit:
        data_iter = iter(loader)
    while step < total_steps:
        if overfit:
            x_lo, x_hi = fixed_lo, fixed_hi
        else:
            try:
                x_lo, x_hi = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x_lo, x_hi = next(data_iter)
            x_lo = x_lo.to(device)
            x_hi = x_hi.to(device)

        # Flow matching: x1 = x_hi, x0 = noise (or x_lo + noise)
        x1 = x_hi
        noise = torch.randn_like(x_hi)
        if base_dist == "gaussian":
            x0 = noise
        else:
            x0 = x_lo + noise

        t = torch.rand(batch_size, device=device)
        # xt = (1-t)*x0 + t*x1
        t_expand = t[:, None, None, None]
        xt = (1 - t_expand) * x0 + t_expand * x1
        target = x1 - x0  # velocity field

        # (B, 8, Nx, Ny)
        # (B, 8, Nx, Ny)
        # --------------
        # (B, 16, Nx, Ny)
        model_input = torch.cat([xt, x_lo], dim=1)  # (B, 16, H, W)
        pred = model(t, model_input)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % print_every == 0:
            print(f"step {step}/{total_steps}  loss={loss.item():.6f}")
            wandb.log({'train/loss': loss.item()}, step=step)

        if step % log_every == 0:
            model.eval()
            x_hi_gen = euler_sample(model, vis_lo, num_steps=sample_steps, device=device)
            figs = make_comparison_figure(vis_lo, vis_hi, x_hi_gen,
                                          channels_to_show=(1, 4))
            log_dict = {}
            for name, fig in figs.items():
                log_dict[f'samples/{name}'] = wandb.Image(fig)
                plt.close(fig)
            wandb.log(log_dict, step=step)
            model.train()
            print(f"  [logged samples at step {step}]")

    # Save final checkpoint
    torch.save(model.state_dict(), 'simple_diffusion/checkpoint.pt')
    print("Saved checkpoint.")
    wandb.finish()


if __name__ == '__main__':
    main()
