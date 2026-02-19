"""
UNet for flow-matching super-resolution, adapted from gf3/models/mg_unet.py.

Changes from gf3:
  - in_channels, out_channels, model_channels, image_size are constructor args
  - Single time embedding (OneTimeEmbedder) instead of two-time
  - No augmentation conditioning
  - No assert out.shape == x.shape (in_ch != out_ch)
  - forward(t, x) instead of forward(t, u, x, aug_cond)

Everything else (ResBlock with GroupNorm + scale-shift, AttentionBlock,
Downsample/Upsample, Fourier time embeddings) is the same as gf3.
"""

from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ---------- init helpers ----------

def default_edm2_init(module):
    if hasattr(module, 'weight') and module.weight is not None:
        if module.weight.ndim >= 2:
            th.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if hasattr(module, 'bias') and module.bias is not None:
        th.nn.init.zeros_(module.bias)

def scale_shift_init(module):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, 0)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        net = nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        net = nn.Conv2d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")
    net.apply(default_edm2_init)
    return net

def linear(*args, **kwargs):
    m = nn.Linear(*args, **kwargs)
    m.apply(default_edm2_init)
    return m

def scale_shift_linear(*args, **kwargs):
    m = nn.Linear(*args, **kwargs)
    m.apply(scale_shift_init)
    return m

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


# ---------- normalization ----------

class CustomGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, affine=False, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(th.ones(num_channels))
            self.bias = nn.Parameter(th.zeros(num_channels))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.group_norm(x, num_groups=self.num_groups, weight=w, bias=b, eps=self.eps)

def get_norm(ch, scale=True):
    return CustomGroupNorm(num_groups=32, num_channels=ch, affine=scale, eps=1e-4)


# ---------- timestep modules ----------

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ---------- up/down sampling ----------

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


# ---------- ResBlock ----------

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_conv=False, dims=2, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            get_norm(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            scale_shift_linear(emb_channels, 2 * self.out_channels),
        )

        self.out_layers = nn.Sequential(
            get_norm(self.out_channels, scale=False),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # scale-shift norm
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = th.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
        return self.skip_connection(x) + h


# ---------- Attention ----------

class OldQKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight, dim=-1)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm = get_norm(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = OldQKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_flat))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, *spatial)


# ---------- Fourier time embedding ----------

class FourierPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        half_dim = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(half_dim) / half_dim)
        self.register_buffer("freqs", freqs.to(th.get_default_dtype()))

    def forward(self, t):
        t_work = t[:, None]
        f = self.freqs.clone().to(t.dtype)
        tf = t_work * f[None, :]
        return th.cat([tf.cos(), tf.sin(), t_work], dim=-1)

class OneTimeEmbedder(nn.Module):
    def __init__(self, fourier_dim, hidden_dim, out_dim):
        super().__init__()
        self.fourier = FourierPositionalEmbedding(dim=fourier_dim)
        net_in_dim = fourier_dim + 1  # cos + sin + t
        self.net = nn.Sequential(
            linear(net_in_dim, hidden_dim),
            nn.SiLU(),
            linear(hidden_dim, out_dim),
        )

    def forward(self, t):
        return self.net(self.fourier(t))


# ---------- UNet ----------

class UNetModel(nn.Module):
    def __init__(
        self,
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
    ):
        super().__init__()
        dims = 2
        time_embed_dim = model_channels * 4

        self.time_embed = OneTimeEmbedder(
            fourier_dim=model_channels,
            hidden_dim=time_embed_dim,
            out_dim=time_embed_dim,
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout,
                             out_channels=int(mult * model_channels), dims=dims)
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads,
                                       num_head_channels=num_head_channels)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout,
                                 out_channels=out_ch, dims=dims, down=True)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels),
            ResBlock(ch, time_embed_dim, dropout, dims=dims),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout,
                             out_channels=int(model_channels * mult), dims=dims)
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads,
                                       num_head_channels=num_head_channels)
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout,
                                 out_channels=ch, dims=dims, up=True)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            get_norm(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, t, x):
        """
        t: (B,) flow time in [0, 1]
        x: (B, in_channels, H, W) — concatenation of xt and x_lo
        """
        emb = self.time_embed(t)
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
