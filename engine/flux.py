"""Flux (``flux``) DiT in plain torch.

The Black-Forest-Labs Flux transformer: ``depth`` double-stream blocks (separate
image/text streams with joint attention) followed by ``depth_single_blocks``
single-stream blocks, RoPE over a 3-axis latent grid, and adaLN modulation.
Ported from ``comfy/ldm/flux/{model,layers,math}.py`` (Apache-2.0), rewritten to
depend only on ``torch``. Weights stream from a GGUF state dict (see
:mod:`engine.gguf`); text2image only (no reference/Kontext latents).

The module attribute names match the GGUF tensor names exactly, so
:func:`load_flux` copies tensors straight in.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gguf as gguf_reader
from .attention import attention
from .precision import build_on_meta


# ---------------------------------------------------------------------------
# RoPE / timestep helpers (from comfy/ldm/flux).
# ---------------------------------------------------------------------------

def _rope(pos, dim, theta):
    assert dim % 2 == 0
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=pos.device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).to(torch.float32)


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n = ids.shape[-1]
        emb = torch.cat([_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n)], dim=-3)
        return emb.unsqueeze(1)


def apply_rope1(x, freqs_cis):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]
    out = freqs_cis[..., 0] * x_[..., 0]
    out = out.addcmul(freqs_cis[..., 1], x_[..., 1])
    return out.reshape(*x.shape).type_as(x)


def apply_rope(q, k, freqs_cis):
    return apply_rope1(q, freqs_cis), apply_rope1(k, freqs_cis)


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        emb = emb.to(t)
    return emb


# ---------------------------------------------------------------------------
# Building blocks.
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm with a ``scale`` parameter (matches the Flux weight layout)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=True)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        return self.out_layer(self.silu(self.in_layer(x)))


class ModulationOut:
    __slots__ = ("shift", "scale", "gate")

    def __init__(self, shift, scale, gate):
        self.shift = shift
        self.scale = scale
        self.gate = gate


class Modulation(nn.Module):
    def __init__(self, dim, double):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        if vec.ndim == 2:
            vec = vec[:, None, :]
        out = self.lin(F.silu(vec)).chunk(self.multiplier, dim=-1)
        second = ModulationOut(*out[3:]) if self.is_double else None
        return ModulationOut(*out[:3]), second


def _modulated_norm(x, norm, scale, shift):
    return (1 + scale) * norm(x) + shift


def _split_qkv(qkv, heads):
    b, n, _ = qkv.shape
    qkv = qkv.view(b, n, 3, heads, -1).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def _attention(q, k, v, pe, mask=None):
    if pe is not None:
        q, k = apply_rope(q, k, pe)
    b, h, length, d = q.shape
    # A regional mask is an additive bias and must share the query dtype.
    if mask is not None and mask.dtype != torch.bool and mask.dtype != q.dtype:
        mask = mask.to(q.dtype)
    out = attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(b, length, h * d)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )

    def forward(self, img, txt, vec, pe, mask=None):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = _modulated_norm(img, self.img_norm1, img_mod1.scale, img_mod1.shift)
        img_q, img_k, img_v = _split_qkv(self.img_attn.qkv(img_modulated), self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = _modulated_norm(txt, self.txt_norm1, txt_mod1.scale, txt_mod1.shift)
        txt_q, txt_k, txt_v = _split_qkv(self.txt_attn.qkv(txt_modulated), self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = _attention(q, k, v, pe, mask)
        txt_attn = attn[:, :txt.shape[1]]
        img_attn = attn[:, txt.shape[1]:]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            _modulated_norm(img, self.img_norm2, img_mod2.scale, img_mod2.shift)
        )
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            _modulated_norm(txt, self.txt_norm2, txt_mod2.scale, txt_mod2.shift)
        )
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=True)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x, vec, pe, mask=None):
        mod, _ = self.modulation(vec)
        x_mod = _modulated_norm(x, self.pre_norm, mod.scale, mod.shift)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )
        q, k, v = _split_qkv(qkv, self.num_heads)
        q, k = self.norm(q, k, v)
        attn = _attention(q, k, v, pe, mask)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, vec):
        if vec.ndim == 2:
            vec = vec[:, None, :]
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = _modulated_norm(x, self.norm_final, scale, shift)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------

def _patchify(x, p):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // p) * (w // p), c * p * p)


def _unpatchify(x, h_len, w_len, p, c):
    b = x.shape[0]
    x = x.reshape(b, h_len, w_len, c, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, h_len * p, w_len * p)


class Flux(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, vec_in_dim=768, context_in_dim=4096,
                 hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19,
                 depth_single_blocks=38, axes_dim=(16, 56, 56), theta=10000,
                 patch_size=2, guidance_embed=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels * patch_size * patch_size
        self.out_channels = out_channels * patch_size * patch_size
        self.latent_channels = out_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        pe_dim = hidden_size // num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=list(axes_dim))
        self.img_in = nn.Linear(self.in_channels, hidden_size, bias=True)
        self.time_in = MLPEmbedder(256, hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size) if vec_in_dim is not None else None
        self.guidance_in = MLPEmbedder(256, hidden_size) if guidance_embed else None
        self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=True)
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth_single_blocks)
        ])
        self.final_layer = LastLayer(hidden_size, 1, self.out_channels)
        self._block_storage = None
        self._block_compute = None

    def enable_block_offload(self, storage_device, compute_device):
        """Keep the transformer blocks on ``storage_device`` and stream them
        through ``compute_device`` one at a time (bf16 is larger than VRAM)."""
        self._block_storage = torch.device(storage_device)
        self._block_compute = torch.device(compute_device)

    def place_head(self, device):
        """Move everything except the streamed blocks to a device (dtype unchanged)."""
        for mod in (self.img_in, self.time_in, self.vector_in, self.guidance_in,
                    self.txt_in, self.final_layer, self.pe_embedder):
            if mod is not None:
                mod.to(device=device)
        storage = self._block_storage if self._block_storage is not None else device
        self.double_blocks.to(storage)
        self.single_blocks.to(storage)

    def _img_ids(self, h_len, w_len, device):
        ids = torch.zeros((h_len, w_len, len(self.pe_embedder.axes_dim)),
                          device=device, dtype=torch.float32)
        ids[:, :, 1] = torch.linspace(0, h_len - 1, steps=h_len, device=device).unsqueeze(1)
        ids[:, :, 2] = torch.linspace(0, w_len - 1, steps=w_len, device=device).unsqueeze(0)
        return ids.reshape(1, h_len * w_len, 3)

    def _stream(self, block, *args, **kwargs):
        if self._block_compute is not None:
            block.to(device=self._block_compute)
        out = block(*args, **kwargs)
        if self._block_compute is not None:
            block.to(device=self._block_storage)
        return out

    def forward(self, x, timestep, context, y=None, guidance=None,
                cross_attn_mask=None):
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p or w % p:
            x = F.pad(x, (0, (-w) % p, 0, (-h) % p))
            h, w = x.shape[-2:]
        h_len, w_len = h // p, w // p

        img = self.img_in(_patchify(x, p))
        img_ids = self._img_ids(h_len, w_len, x.device)

        vec = self.time_in(timestep_embedding(timestep, 256).to(img.dtype))
        if self.guidance_in is not None and guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        if self.vector_in is not None and y is not None:
            vec = vec + self.vector_in(y.to(img.dtype))

        txt = self.txt_in(context)
        txt_ids = torch.zeros((b, context.shape[1], 3), device=x.device, dtype=torch.float32)
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))

        for block in self.double_blocks:
            img, txt = self._stream(block, img, txt, vec, pe, mask=cross_attn_mask)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = self._stream(block, img, vec, pe, mask=cross_attn_mask)

        img = img[:, txt.shape[1]:]
        img = self.final_layer(img, vec)
        return _unpatchify(img, h_len, w_len, p, self.latent_channels)


DEFAULT_CONFIG = dict(
    in_channels=16, out_channels=16, vec_in_dim=768, context_in_dim=4096,
    hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19,
    depth_single_blocks=38, axes_dim=(16, 56, 56), theta=10000, patch_size=2,
    guidance_embed=True,
)


def load_flux(gguf_path, device="cpu", dtype=torch.bfloat16, stream=False,
              stream_device=None, ram_budget=0, prefetch_depth=0, pin_blocks=True):
    """Build the Flux DiT and load weights straight from a GGUF file.

    With ``stream`` the model is built on the meta device and its blocks are
    read from the GGUF on demand (see :mod:`engine.streaming`), so system RAM no
    longer has to hold the whole transformer; otherwise every weight is
    dequantized into the module up front and the model is moved to ``device``.
    """
    reader = gguf_reader.gguf.GGUFReader(gguf_path)
    names = {t.name for t in reader.tensors}
    config = dict(DEFAULT_CONFIG)
    config["guidance_embed"] = "guidance_in.in_layer.weight" in names
    config["depth"] = 1 + max(int(n.split(".")[1]) for n in names if n.startswith("double_blocks."))
    config["depth_single_blocks"] = 1 + max(
        int(n.split(".")[1]) for n in names if n.startswith("single_blocks.")
    )

    factory = lambda: Flux(**config)

    if stream and str(gguf_path).lower().endswith(".gguf"):
        from . import streaming as streaming_module

        source = streaming_module.GgufSource(gguf_path)
        model, _streamer = streaming_module.build_streaming_model(
            factory, source, device=stream_device or device, dtype=dtype,
            ram_budget=ram_budget, prefetch_depth=prefetch_depth, pin_blocks=pin_blocks,
        )
        missing = [n for n in dict(model.named_parameters()) if source.source_for(n) is None]
        return model, missing, []

    # Build on the ``meta`` device so no ~24 GB bf16 parameter storage is
    # allocated and no random initialiser runs over the ~12B parameters (they
    # cost minutes); every weight is read from the GGUF and *assigned* in place
    # (``assign=True``).
    with build_on_meta(dtype, factory) as model:
        pass
    params = dict(model.named_parameters())
    state = {}
    loaded = set()
    for tensor in reader.tensors:
        dst = params.get(tensor.name)
        if dst is None:
            continue
        state[tensor.name] = gguf_reader.dequantize(tensor, dtype=dtype).reshape(dst.shape)
        loaded.add(tensor.name)
    model.load_state_dict(state, strict=False, assign=True)
    # Materialise the loaded parameters in the target dtype/device; params the
    # file did not carry stay on ``meta`` and are skipped.
    for p in model.parameters():
        if p.device.type != "meta" and (p.dtype != dtype or str(p.device) != str(device)):
            p.data = p.to(device=device, dtype=dtype)
    model.eval()
    missing = [n for n in params if n not in loaded]
    for p in model.parameters():
        p.requires_grad_(False)
    return model, missing, []
