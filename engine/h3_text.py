"""MiniMax H3 text conditioning: Qwen3-VL-32B truncated to 50 layers.

H3 conditions its DiT on the **unnormalized hidden state after LM layer 50**
(ComfyUI ``comfy/text_encoders/minimax.py``: the converted checkpoint is
truncated at layer 50, so this is the last decoder layer's output with **no
final norm**). Unlike Qwen-Image's encoder (:mod:`engine.qwen_text`) the H3
presentation is **not chat-templated**: for plain text-to-image the conditioning
is just the raw prompt text (no ``<|im_start|>`` turns, no special tokens).

The DiT's per-token adaLN tag is returned alongside the embeddings
(``minimax_token_tags``): text positions carry tag 1, so a text-only prompt is
all ones. Matches ComfyUI's ``comfy/text_encoders/minimax.py``.

Memory
------
The released ``qwen3vl_32b_minimax_h3_*`` file is ~25 GB (int8) and would be
~50 GB rebuilt dense in bf16 -- too large to hold alongside the H3 DiT in a
64 GB machine (that co-residence was the preload-then-generate OOM). So the
encoder is **not** built as a stock ``transformers`` model. Instead the
safetensors file is memory-mapped and one decoder layer is dequantized, run and
freed at a time, bounding the working set to a single layer (~1 GB) plus the
dequant transient (~128 MB). The 25 GB on disk stays reclaimable file cache,
not committed RAM, so the DiT can remain resident.

The forward is the stock Qwen3-VL text decoder (RMSNorm, GQA self-attention with
per-head q/k norm, RoPE, SwiGLU MLP). For a text-only prompt all three mRoPE
grids are the same ``arange`` positions, so the 3-axis interleave collapses to
ordinary RoPE over the head dim. Weights load from the ComfyUI-layout safetensors
(optionally int8/convrot, via :func:`engine.moss_text._copy_weight`); the
tokenizer reuses the Qwen2 BPE data shipped for Anima (:mod:`engine.tokenizer_data`),
whose vocab matches Qwen3-VL's 151936 entries for ordinary prompt text.
"""

import os

import torch
import torch.nn.functional as F

from .moss_text import _copy_weight
from .safetensors_io import open_safetensors

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOKENIZER_DATA = os.path.join(_HERE, "tokenizer_data")

PAD = 151643


def _default_config():
    """The Qwen3-VL-32B text config, with the H3 checkpoint's layer truncation."""
    return {
        "vocab": 151936,
        "hidden": 5120,
        "intermediate": 25600,
        "num_layers": 50,
        "heads": 64,
        "kv_heads": 8,
        "head_dim": 128,
        "eps": 1e-6,
        "rope_theta": 5_000_000.0,
    }


# Per-layer parameter layout in ComfyUI's ``model.layers.N.*`` namespace: the
# short name used by the forward, the on-disk suffix and the ``(out, in)`` shape.
_LAYER_SPEC = (
    ("in_ln", "input_layernorm.weight", ("hidden",)),
    ("post_ln", "post_attention_layernorm.weight", ("hidden",)),
    ("q_proj", "self_attn.q_proj.weight", ("q_dim", "hidden")),
    ("k_proj", "self_attn.k_proj.weight", ("kv_dim", "hidden")),
    ("v_proj", "self_attn.v_proj.weight", ("kv_dim", "hidden")),
    ("o_proj", "self_attn.o_proj.weight", ("hidden", "q_dim")),
    ("q_norm", "self_attn.q_norm.weight", ("head_dim",)),
    ("k_norm", "self_attn.k_norm.weight", ("head_dim",)),
    ("gate", "mlp.gate_proj.weight", ("intermediate", "hidden")),
    ("up", "mlp.up_proj.weight", ("intermediate", "hidden")),
    ("down", "mlp.down_proj.weight", ("hidden", "intermediate")),
)

_EMBED_KEY = "model.embed_tokens.weight"


def _rms(x, weight, eps):
    """``Qwen3VLTextRMSNorm``: fp32 variance, weight applied in the input dtype."""
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_tables(inv_freq, length, dtype):
    """Cos/sin ``[length, head_dim]`` for text-only mRoPE.

    All three mRoPE grids are the same ``arange`` positions, so the axis
    interleave is a no-op and the table is standard RoPE: ``head_dim//2``
    frequencies duplicated to fill ``head_dim``.
    """
    positions = torch.arange(length, dtype=torch.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)
    return cos.to(dtype), sin.to(dtype)


def _load_layer(handle, names, index, dtype, dims, scratch):
    """Dequantize one decoder layer into reusable ``scratch`` buffers.

    Every layer has identical shapes, so the buffers are allocated once and
    overwritten per layer: the working set stays at one layer's weights (~1 GB)
    with no per-layer allocation churn.
    """
    for name, suffix, shape_spec in _LAYER_SPEC:
        key = f"model.layers.{index}.{suffix}"
        if key not in names:
            raise RuntimeError(f"MiniMax H3 text encoder is missing {key}")
        dst = scratch.get(name)
        if dst is None:
            dst = torch.empty(tuple(dims[part] for part in shape_spec), dtype=dtype)
            scratch[name] = dst
        _copy_weight(handle, names, key, dst, dtype)
    return scratch


def _embed_rows(handle, ids, dtype):
    """Gather the prompt's embedding rows as a fresh, owned tensor.

    ``slice[int]`` returns a *zero-copy* view over safetensors' shared buffer, and
    ``.to(dtype)`` on a matching dtype is a no-op -- so the row stayed a view over
    memory whose lifetime is not guaranteed, and gathering it later could fault
    (an access violation, seen only inside the app). Read one contiguous slice
    and ``clone`` it into memory we own before indexing.
    """
    low, high = int(min(ids)), int(max(ids))
    rows = handle.get_slice(_EMBED_KEY)[low:high + 1].clone()
    index = torch.as_tensor([int(i) - low for i in ids], dtype=torch.long)
    return rows.index_select(0, index).to(dtype)


def _attention(x, weights, cos, sin, dims):
    batch, length, _ = x.shape
    heads, kv_heads, head_dim = dims["heads"], dims["kv_heads"], dims["head_dim"]

    q = F.linear(x, weights["q_proj"]).view(batch, length, heads, head_dim)
    k = F.linear(x, weights["k_proj"]).view(batch, length, kv_heads, head_dim)
    v = F.linear(x, weights["v_proj"]).view(batch, length, kv_heads, head_dim)

    eps = dims["eps"]
    q = _rms(q, weights["q_norm"], eps)
    k = _rms(k, weights["k_norm"], eps)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    q = q * c + _rotate_half(q) * s
    k = k * c + _rotate_half(k) * s

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    out = out.transpose(1, 2).reshape(batch, length, -1)
    return F.linear(out, weights["o_proj"])


def _mlp(x, weights):
    return F.linear(F.silu(F.linear(x, weights["gate"])) * F.linear(x, weights["up"]),
                    weights["down"])


class Qwen3VLTextStreamEncoder:
    """Qwen3-VL-32B (50 layers) last-hidden-state encoder, streamed per layer."""

    def __init__(self, weights_path, device="cpu", dtype=torch.bfloat16, tick=None,
                 config=None):
        from transformers import Qwen2Tokenizer

        self.path = weights_path
        self.device = device
        self.dtype = dtype
        self.config = dict(_default_config())
        if config:
            self.config.update(config)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(_TOKENIZER_DATA)

        cfg = self.config
        cfg["q_dim"] = cfg["heads"] * cfg["head_dim"]
        cfg["kv_dim"] = cfg["kv_heads"] * cfg["head_dim"]
        self._inv_freq = 1.0 / (
            cfg["rope_theta"] ** (torch.arange(0, cfg["head_dim"], 2, dtype=torch.float32)
                                  / cfg["head_dim"])
        )
        # Reused per-layer scratch buffers (one layer's weights, ~1 GB), so a
        # generate does not churn 50 layer allocations.
        self._scratch = {}
        if tick is not None:
            tick(1.0)

    def _tokenize(self, prompt):
        # Raw prompt, no template: H3's presentation is the bare text.
        return self.tokenizer(prompt or " ", add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def _forward(self, ids):
        """Return ``[1, T, hidden]``: the layer-50 output, no final norm."""
        cfg = self.config
        ids = list(ids) or [PAD]
        with open_safetensors(self.path) as handle:
            names = set(handle.keys())
            x = _embed_rows(handle, ids, self.dtype).unsqueeze(0).to(self.device)
            cos, sin = _rope_tables(self._inv_freq, x.shape[1], self.dtype)
            cos = cos.unsqueeze(0).to(self.device)
            sin = sin.unsqueeze(0).to(self.device)
            for index in range(cfg["num_layers"]):
                weights = _load_layer(handle, names, index, self.dtype, cfg, self._scratch)
                residual = x
                x = residual + _attention(_rms(x, weights["in_ln"], cfg["eps"]),
                                          weights, cos, sin, cfg)
                residual = x
                x = residual + _mlp(_rms(x, weights["post_ln"], cfg["eps"]), weights)
        return x

    @torch.no_grad()
    def encode(self, prompt):
        """Return ``(hidden, tags)`` for ``prompt``.

        ``hidden`` is ``[1, T, 5120]`` (layer-50 output, before any final norm)
        and ``tags`` is ``[T]`` of ``1`` (text modality) -- so a text-only still is
        all ones.
        """
        hidden = self._forward(self._tokenize(prompt or ""))
        tags = torch.ones(hidden.shape[1], device=hidden.device, dtype=torch.long)
        return hidden, tags


def load_text_encoder(path, device="cpu", dtype=torch.bfloat16, tick=None):
    return Qwen3VLTextStreamEncoder(path, device=device, dtype=dtype, tick=tick)
