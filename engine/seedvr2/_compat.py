"""Small compatibility shims for the ported SeedVR2 modules.

The SeedVR2 DiT and VAE in this package are ports of ComfyUI's
``comfy/ldm/seedvr`` (GPL-3.0, see ``CREDITS.md``). The originals lean on a few
ComfyUI runtime helpers -- its operation wrappers (``comfy.ops``), attention
kernels and ``comfy.model_management`` -- which this project does not depend on.
This module reimplements exactly the pieces those ports use, on top of plain
``torch``, so the ported code can keep its original structure.

Nothing here downloads or imports ComfyUI.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import attention

# The ported code reads ``comfy.model_management.in_training`` in a couple of
# places to choose a grad-friendly path. This app only ever infers.
in_training = False


# ---------------------------------------------------------------------------
# Timestep / rotary embeddings
# ---------------------------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False,
                           downscale_freq_shift=1):
    """The sinusoidal timestep embedding the SeedVR2 DiT was trained with.

    Matches ComfyUI's ``get_timestep_embedding``; SeedVR2 calls it with
    ``flip_sin_to_cos=False`` and ``downscale_freq_shift=0``.
    """
    if timesteps.ndim != 1:
        raise ValueError("get_timestep_embedding expects a 1-D timestep tensor")
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - downscale_freq_shift)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rope1(x, freqs_cis):
    """Apply a precomputed rotary embedding pair to ``x`` (Flux convention)."""
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if (x_.shape[2] != 1 and freqs_cis.shape[2] != 1
            and x_.shape[2] != freqs_cis.shape[2]):
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]
    out = freqs_cis[..., 0] * x_[..., 0]
    out = out.addcmul(freqs_cis[..., 1], x_[..., 1])
    return out.reshape(*x.shape).type_as(x)


# ---------------------------------------------------------------------------
# Attention (plain PyTorch SDPA)
# ---------------------------------------------------------------------------
def optimized_attention(q, k, v, heads, mask=None, skip_reshape=False,
                        skip_output_reshape=False, **kwargs):
    """Multi-head scaled dot-product attention.

    With ``skip_reshape`` the inputs are already ``(B, heads, L, D)``; otherwise
    ``(B, L, heads*D)``. This mirrors the subset of ComfyUI's
    ``optimized_attention`` the SeedVR2 port uses.
    """
    if skip_reshape:
        batch = q.shape[0]
    else:
        batch, _length, dim = q.shape
        head_dim = dim // heads
        q = q.view(batch, -1, heads, head_dim).transpose(1, 2)
        if k.dim() == 3:
            k = k.view(k.shape[0], k.shape[1], heads, head_dim).transpose(1, 2)
            v = v.view(v.shape[0], v.shape[1], heads, head_dim).transpose(1, 2)
    out = attention(q, k, v, attn_mask=mask)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(batch, -1, heads * out.shape[-1])
    return out


def _spatial_attention(q, k, v):
    """Single-head attention over a ``(B, C, H, W)`` feature map (VAE path)."""
    b, c, h, w = q.shape
    q = q.reshape(b, c, h * w).permute(0, 2, 1)
    k = k.reshape(b, c, h * w).permute(0, 2, 1)
    v = v.reshape(b, c, h * w).permute(0, 2, 1)
    out = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))
    out = out.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
    return out


def vae_attention():
    """Return the single-head spatial attention used by the SeedVR2 VAE."""
    return _spatial_attention


# ---------------------------------------------------------------------------
# Operation wrappers (a stand-in for comfy.ops.disable_weight_init)
# ---------------------------------------------------------------------------
class _NoInit:
    """Skip random initialisation; every weight is filled from a checkpoint."""

    def reset_parameters(self):
        pass


class Linear(_NoInit, nn.Linear):
    pass


class Conv2d(_NoInit, nn.Conv2d):
    pass


class Conv3d(_NoInit, nn.Conv3d):
    pass


class GroupNorm(_NoInit, nn.GroupNorm):
    pass


class LayerNorm(_NoInit, nn.LayerNorm):
    pass


class RMSNorm(_NoInit, nn.RMSNorm):
    pass


def cast_to_input(weight, input, non_blocking=False, copy=True):
    """Cast ``weight`` to ``input``'s dtype/device (ComfyUI's helper)."""
    if weight is None:
        return None
    if weight.dtype == input.dtype and weight.device == input.device:
        return weight
    return weight.to(dtype=input.dtype, device=input.device, non_blocking=non_blocking,
                     copy=copy)


class _Ops:
    """Namespace mirroring ``comfy.ops.disable_weight_init``."""

    Linear = Linear
    Conv2d = Conv2d
    Conv3d = Conv3d
    GroupNorm = GroupNorm
    LayerNorm = LayerNorm
    RMSNorm = RMSNorm
    cast_to_input = staticmethod(cast_to_input)


ops = _Ops()


class _ModelManagement:
    """Namespace mirroring the ``comfy.model_management`` bits the ports touch."""

    in_training = False


model_management = _ModelManagement()


# ---------------------------------------------------------------------------
# Progress bar (only the VAE's optional tiling path calls it)
# ---------------------------------------------------------------------------
class ProgressBar:
    def __init__(self, total, node_id=None):
        self.total = total
        self.count = 0

    def update(self, value):
        self.count = value

    def update_absolute(self, value, total=None):
        self.count = value
