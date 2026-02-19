# simple_diffusion — Flow Matching Super-Resolution for MHD

Standalone flow matching super-resolution trainer. Takes 2x-downsampled MHD fields as input and learns to generate the high-resolution version conditioned on the low-res input.

## Method

**Flow matching** with linear interpolation paths:
- `x0` = base distribution sample, `x1` = high-res target (normalized)
- `xt = (1-t)*x0 + t*x1`, with `t ~ U(t_min, t_max)`
- Model predicts the velocity field `v = x1 - x0`
- Loss: `loss_scale * MSE(v_pred, v_true)`
- Sampling: Euler ODE integration from `t_min` to `t_max`

The model is conditioned on the low-res input by channel-concatenation: the UNet receives `cat(xt, x_lo)` and outputs the velocity prediction. Channel count adapts to the selected fields.

### Base distribution

`base_dist` controls what `x0` is (default: `x_lo_plus_noise`):
- `"gaussian"`: `x0 ~ N(0, I)` — standard flow matching
- `"x_lo_plus_noise"`: `x0 = x_lo + sigma * N(0, I)` where `sigma = base_noise_scale` — starts closer to the target, easier transport

### SDE sampler

In addition to the ODE sampler (`dx = v dt`), there is an SDE sampler with matching marginals:

```
dx = [v + delta * s] dt + sqrt(2*delta) dW
```

The score `s(t,x)` is derived analytically from the learned velocity `v(t,x)` — no separate score model needed:
- **x_lo_plus_noise base**: `s(t,x) = -(x - t*v - x_lo) / ((1-t) * sigma^2)`
- **gaussian base**: `s(t,x) = -(x - t*v) / (1-t)`

`delta=0` recovers the ODE. Set `--sde-delta 0.1` (or similar) to enable.

### Per-channel normalization

Training data is standardized per-channel (zero mean, unit variance) using statistics computed from the training set (Welford online algorithm). Model operates in normalized space; outputs are denormalized for visualization and metrics.

## Data

Multi-seed paired data from AthenaK MRI simulations:
- **256x256**: `mri256_nu3e-5/seed_NNNN/npy/all_fields_256x256.npy` — shape `(200, 256, 256, 8)`
- **128x128**: `mri128_nu3e-5/seed_NNNN/npy/all_fields_128x128.npy` — shape `(200, 128, 128, 8)`
- 100 seeds (0-99) for each resolution, frames 0-199 (~20 orbits)

8 MHD fields (channels-last in npy, channels-first in PyTorch):

| Index | Field   | Description        |
|-------|---------|--------------------|
| 0     | density | mass density       |
| 1     | velx    | x-velocity         |
| 2     | vely    | y-velocity         |
| 3     | velz    | z-velocity         |
| 4     | Bx      | x magnetic field   |
| 5     | By      | y magnetic field   |
| 6     | Bz      | z magnetic field   |
| 7     | jz      | z current density  |

Low-res inputs created on-the-fly: `avg_pool2d(256->128)` then `bilinear upsample(128->256)`.

Seed split: 80 train / 10 val / 10 test.

## Model

UNet adapted from `gf3/models/mg_unet.py`:

- **Architecture**: Encoder-decoder with skip connections, ResBlock down/up
- **Conditioning**: Fourier time embedding -> MLP -> scale-shift in each ResBlock (FiLM)
- **Channels**: `model_channels=128`, `channel_mult=(1,2,2,2)` -> 128, 256, 256, 256
- **Attention**: Self-attention at 64x64 resolution (ds=4), 4 heads with 64 channels each
- **ResBlocks**: 2 per level, GroupNorm + scale-shift, zero-init output conv
- **Input/Output**: `2*C` channels in (C for xt + C for x_lo), C channels out (velocity)
- **Parameters**: ~43M (all 8 channels), ~40M (1 channel)

## Files

```
simple_diffusion/
  model.py    — UNet definition (ResBlock, AttentionBlock, Fourier embeddings)
  dataset.py  — MRIPairedDataset, prepare_batch, make_loaders, normalize/denormalize
  trainer.py  — Flow matching training loop, ODE/SDE sampling, EMA, wandb viz
  README.md   — this file
```

## Config

All hyperparameters are CLI args with defaults in `DEFAULTS` dict at top of `trainer.py`:

```python
DEFAULTS = dict(
    batch_size=4,
    lr=2e-4,
    weight_decay=0.0,
    total_steps=50_000,
    print_every=100,
    log_every=100,
    sample_steps=100,
    base_dist="x_lo_plus_noise",  # or "gaussian"
    base_noise_scale=0.1,
    t_min_train=1e-4,
    t_max_train=1 - 1e-4,
    t_min_sample=1e-4,
    t_max_sample=1 - 1e-4,
    loss_scale=100.0,       # scale loss to avoid Adam epsilon regime
    grad_clip=1.0,          # max grad norm
    ema_decay=0.9999,
    warmup_steps=10_000,    # linear LR warmup (0 in overfit mode)
    use_bf16=True,
    sde_delta=0.0,          # 0 = ODE only
    overfit=True,
    channels="velx",        # or None for all 8, or "Bx,jz" etc.
)
```

### Optimization details

- **AdamW** with explicit `weight_decay` (default 0)
- **Loss scaling**: MSE multiplied by `loss_scale` (default 100) before backward to keep gradients above Adam's epsilon floor — important with x_lo_plus_noise base where the velocity targets are small
- **Gradient clipping**: `clip_grad_norm_` with max norm `grad_clip` (default 1.0)
- **LR warmup**: Linear ramp from 0 to `lr` over `warmup_steps` (default 10k). Skipped in overfit mode
- **EMA**: Exponential moving average of model weights (decay 0.9999). Samples from both raw and EMA models are logged
- **bf16**: `torch.autocast` with bfloat16 for forward + loss. No grad scaler needed (bf16 has same exponent range as fp32). Disable with `--no-bf16`
- **t_min/t_max**: Separate for training and sampling. Avoids singularities at t=0 (pure noise) and t=1 (pure data)

## Usage

```bash
source load.sh

# Overfit on single seed, single channel (default)
python3 simple_diffusion/trainer.py

# Single channel, custom settings
python3 simple_diffusion/trainer.py --channels Bx --lr 1e-4

# All 8 channels, full training
python3 simple_diffusion/trainer.py --channels density,velx,vely,velz,Bx,By,Bz,jz --no-overfit

# Enable SDE sampling during eval
python3 simple_diffusion/trainer.py --sde-delta 0.1
```

### Wandb visualizations

Every `log_every` steps, logs to wandb project `mri`:

- **samples/{field}**: 3-row comparison (input / target / generated) — raw model
- **samples_ema/{field}**: Same but from EMA model
- **diffs/{field}**: Difference images (target-input, target-generated) — raw model
- **diffs_ema/{field}**: Same but from EMA model
- **samples_sde/{field}**: SDE samples (EMA model, if `sde_delta > 0`)
- **diffs_sde/{field}**: SDE difference images (if `sde_delta > 0`)
- **spectra**: Power spectra — truth (256), ODE generated, ODE EMA, SDE EMA (if enabled), low-res (128)
- **train/loss**: Scaled MSE loss
- **val/mse**, **val/mse_ema**, **val/mse_sde**: Physical-unit MSE metrics

### Checkpoint

Saves to `simple_diffusion/checkpoint.pt`:
```python
{
    'model_state_dict': ...,
    'ema_state_dict': ...,
    'stats': {'mean': ..., 'std': ...},  # per-channel normalization
    'channel_names': [...],
    'step': ...,
}
```
