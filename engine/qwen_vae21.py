"""Qwen-Image 2.1 VAE.

A Wan 2.2-style causal 3D autoencoder (``z_dim=64``, ``dim=96``, separate
``dec_dim=144``, ``dim_mult=[1,2,4,8,8]``, temporal kernel 1, no patchify, RGBA
output) as configured for ``qwen_image_2.1_vae``. Re-implemented in plain
``torch`` from ComfyUI's ``comfy/ldm/wan/vae2_2.py`` (Alibaba Wan Team, Apache
2.0). Reuses the causal-3D primitives already in :mod:`engine.vae`.

Images are encoded/decoded as RGBA; RGB input is padded with opaque alpha and
RGB output drops the alpha channel.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import AttentionBlock, CausalConv3d, RMS_norm

CACHE_T = 2
STRIP_ELEMS = 2 ** 24


def strip_apply(fn, x, scale=1, halo=1, out=None):
    """Run ``fn`` in row strips to bound conv workspace (a halo keeps it exact)."""
    n = -(-x.numel() * scale * scale // STRIP_ELEMS)
    if n <= 1 and out is None:
        return fn(x)
    add = out is not None
    size = x.shape[-2]
    step = -(-size // n)
    for a in range(0, size, step):
        b = min(size, a + step)
        lo = max(0, a - halo)
        y = fn(x.narrow(-2, lo, min(size, b + halo) - lo)).narrow(-2, (a - lo) * scale, (b - a) * scale)
        if out is None:
            out = y.new_empty(*y.shape[:-2], size * scale, y.shape[-1])
        dst = out.narrow(-2, a * scale, (b - a) * scale)
        if add:
            dst.add_(y)
        else:
            dst.copy_(y)
    return out


def conv3x3(in_dim, out_dim, temporal_kernel=3):
    return CausalConv3d(in_dim, out_dim, (temporal_kernel, 3, 3), padding=(temporal_kernel // 2, 1, 1))


class Resample(nn.Module):
    def __init__(self, dim, mode, temporal_kernel=3):
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (temporal_kernel, 1, 1), padding=(temporal_kernel // 2, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (temporal_kernel, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        b, c, t, h, w = x.size()
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if self.mode in ("upsample2d", "upsample3d"):
            x = strip_apply(self.resample, x, scale=2)
        else:
            x = self.resample(x)
        x = x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, temporal_kernel=3):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            conv3x3(in_dim, out_dim, temporal_kernel),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            conv3x3(out_dim, out_dim, temporal_kernel),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return strip_apply(lambda s: self.residual(s).add_(self.shortcut(s)), x, halo=2)


def _patchify(x, patch_size):
    if patch_size == 1:
        return x
    b, c, f, h, w = x.shape
    x = x.reshape(b, c, f, h // patch_size, patch_size, w // patch_size, patch_size)
    x = x.permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * patch_size * patch_size, f, h // patch_size, w // patch_size)
    return x


def _unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    b, crq, f, h, w = x.shape
    c = crq // (patch_size * patch_size)
    x = x.reshape(b, c, patch_size, patch_size, f, h, w).permute(0, 1, 4, 5, 2, 6, 3)
    return x.reshape(b, c, f, h * patch_size, w * patch_size)


class AvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels
        self.out_channels = out_channels

    def forward(self, x):
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.view(b, c, t // self.factor_t, self.factor_t, h // self.factor_s, self.factor_s, w // self.factor_s, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * self.factor, t // self.factor_t, h // self.factor_s, w // self.factor_s)
        x = x.view(b, self.out_channels, self.group_size, t // self.factor_t, h // self.factor_s, w // self.factor_s)
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        self.out_channels = out_channels
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x, first_chunk=False):
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(x.size(0), self.out_channels, self.factor_t, self.factor_s, self.factor_s, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * self.factor_t, x.size(4) * self.factor_s, x.size(6) * self.factor_s)
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False, temporal_kernel=3):
        super().__init__()
        self.avg_shortcut = AvgDown3D(in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1)
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout, temporal_kernel=temporal_kernel))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode, temporal_kernel=temporal_kernel))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x):
        return self.downsamples(x) + self.avg_shortcut(x)


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False, temporal_kernel=3):
        super().__init__()
        self.avg_shortcut = DupUp3D(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2 if up_flag else 1) if up_flag else None
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout, temporal_kernel=temporal_kernel))
            in_dim = out_dim
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode, temporal_kernel=temporal_kernel))
        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, first_chunk=False):
        x_main = self.upsamples(x)
        if self.avg_shortcut is not None:
            return strip_apply(lambda s: self.avg_shortcut(s, first_chunk), x, scale=self.avg_shortcut.factor_s, halo=0, out=x_main)
        return x_main


class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=(),
                 temperal_downsample=(True, True, False), dropout=0.0, in_channels=12, temporal_kernel=3):
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]
        self.conv1 = conv3x3(in_channels, dims[0], temporal_kernel)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(Down_ResidualBlock(in_dim, out_dim, dropout, num_res_blocks,
                                                  temperal_downsample=t_down, down_flag=i != len(dim_mult) - 1,
                                                  temporal_kernel=temporal_kernel))
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout, temporal_kernel=temporal_kernel),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout, temporal_kernel=temporal_kernel),
        )
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), conv3x3(out_dim, z_dim, temporal_kernel))

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=(),
                 temperal_upsample=(False, True, True), dropout=0.0, out_channels=12, temporal_kernel=3):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult)[::-1]]
        self.conv1 = conv3x3(z_dim, dims[0], temporal_kernel)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout, temporal_kernel=temporal_kernel),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout, temporal_kernel=temporal_kernel),
        )
        upsamples = []
        out_dim = dims[0]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(Up_ResidualBlock(in_dim, out_dim, dropout, num_res_blocks + 1,
                                              temperal_upsample=t_up, up_flag=i != len(dim_mult) - 1,
                                              temporal_kernel=temporal_kernel))
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), conv3x3(out_dim, out_channels, temporal_kernel))

    def forward(self, x, first_chunk=False):
        x = self.conv1(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.upsamples:
            x = layer(x, first_chunk=first_chunk)
        for layer in self.head:
            x = layer(x)
        return x


class WanVAE22(nn.Module):
    def __init__(self, dim=96, dec_dim=144, z_dim=64, dim_mult=(1, 2, 4, 8, 8), num_res_blocks=2,
                 attn_scales=(), temperal_downsample=(False, True, True, True), dropout=0.0,
                 image_channels=4, patch_size=1, temporal_kernel=1):
        super().__init__()
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.image_channels = image_channels
        self.temperal_downsample = list(temperal_downsample)
        self.temperal_upsample = list(temperal_downsample)[::-1]
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales,
                                 self.temperal_downsample, dropout,
                                 in_channels=image_channels * patch_size * patch_size, temporal_kernel=temporal_kernel)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dec_dim, z_dim, dim_mult, num_res_blocks, attn_scales,
                                 self.temperal_upsample, dropout,
                                 out_channels=image_channels * patch_size * patch_size, temporal_kernel=temporal_kernel)

    def encode(self, x):
        if x.ndim == 4:
            x = _patchify(x.unsqueeze(2), self.patch_size)
            return self.conv1(self.encoder(x)).chunk(2, dim=1)[0].squeeze(2)
        x = _patchify(x, self.patch_size)
        raise NotImplementedError("temporal encoding not supported in this engine")

    def decode(self, z):
        if z.ndim == 4:
            out = self.decoder(self.conv2(z.unsqueeze(2)), first_chunk=True)
            return _unpatchify(out, self.patch_size).squeeze(2)
        raise NotImplementedError("temporal decoding not supported in this engine")


DEFAULT_CONFIG = dict(dim=96, dec_dim=144, z_dim=64, dim_mult=(1, 2, 4, 8, 8), num_res_blocks=2,
                      attn_scales=(), temperal_downsample=(False, True, True, True), dropout=0.0,
                      image_channels=4, patch_size=1, temporal_kernel=1)


def _check_vae_signature(path, state):
    """Fail early when ``state`` is not the Qwen-Image 2.1 VAE.

    The 2.1 VAE (``qwen_image_2.1_vae_bf16``) is a 64-channel RGBA autoencoder;
    the older Qwen-Image / Wan 2.1 VAE (``qwen_image_vae``) reuses the same key
    names with different shapes (16 latent channels, 3 colour channels, temporal
    kernel 3). Loading the wrong one raises a wall of ``size mismatch`` entries
    from ``load_state_dict`` that never names the file, so check the two
    discriminating tensors first and raise a clear, actionable error.
    """
    conv2 = state.get("conv2.weight")
    enc_conv1 = state.get("encoder.conv1.weight")
    z_dim = int(conv2.shape[0]) if conv2 is not None else None
    in_channels = int(enc_conv1.shape[1]) if enc_conv1 is not None else None
    if z_dim is None and in_channels is None:
        return
    expected_z = DEFAULT_CONFIG["z_dim"]
    expected_c = DEFAULT_CONFIG["image_channels"]
    if (z_dim is not None and z_dim != expected_z) or (
        in_channels is not None and in_channels != expected_c
    ):
        raise RuntimeError(
            f"{os.path.basename(path)} is not the Qwen-Image 2.1 VAE: it has "
            f"z_dim={z_dim} / image_channels={in_channels}, but the 2.1 VAE "
            f"needs z_dim={expected_z} / image_channels={expected_c}. Assign the "
            f"Qwen-Image 2.1 VAE (qwen_image_2.1_vae_bf16.safetensors) to this "
            f"model's VAE slot."
        )


def load_vae(path, device="cpu", dtype=torch.float32):
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    _check_vae_signature(path, state)
    vae = WanVAE22(**DEFAULT_CONFIG)
    missing, unexpected = vae.load_state_dict(state, strict=False)
    vae = vae.to(device=device, dtype=dtype).eval()
    from .precision import compile_vae_decoder

    sample = torch.zeros(1, vae.z_dim, 16, 16, device=device, dtype=dtype)
    compile_vae_decoder(vae, sample)
    return vae, missing, unexpected
