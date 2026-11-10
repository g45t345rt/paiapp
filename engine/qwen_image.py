"""Qwen-Image 2.1 MMDiT (``qwen_image``) in plain torch.

Single-stream MMDiT: text tokens and latent patches share one sequence and one
set of block weights. Ported from the architecture ComfyUI mirrors
(``comfy/ldm/qwen_image21/model.py``, itself from HF diffusers, Apache-2.0),
rewritten to depend only on ``torch``. Weights load from a GGUF state dict (see
:mod:`engine.gguf`) or a safetensors file, including ComfyUI's ``int8_tensorwise``
quantisation with the optional ``convrot`` rotation (``*_int8_convrot`` files,
dequantized via :mod:`engine.quant_formats`).

Text2image, img2img and reference-image conditioning: condition images are packed
ahead of the target latents and substituted into the vision slots the text
encoder reserved for them, in one block-causal joint sequence (see
:class:`QwenRef`). The prefix K/V cache is not implemented.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import attention
from .precision import build_on_meta


# ---------------------------------------------------------------------------
# RoPE / timestep helpers (from comfy/ldm/flux).
# ---------------------------------------------------------------------------

def _rope(pos, dim, theta):
    assert dim % 2 == 0
    dev = pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=dev)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=dev), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.to(torch.float32)


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        emb = torch.cat([_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])], dim=-3)
        return emb.unsqueeze(1)


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def apply_rope1(x, freqs_cis):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, : x_.shape[2]]
    out = freqs_cis[..., 0] * x_[..., 0]
    out = out.addcmul(freqs_cis[..., 1], x_[..., 1])
    return out.reshape(*x.shape).type_as(x)


# ---------------------------------------------------------------------------
# Building blocks.
# ---------------------------------------------------------------------------

class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm whose stored weight is ``scale - 1`` (applied in fp32)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        w = self.weight.float() + 1.0
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * w).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden_size, eps=1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=False)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)

    def forward(self, timestep):
        return self.timestep_embedder(timestep_embedding(timestep.float(), 256).to(self.timestep_embedder.linear_1.weight.dtype))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-6):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=False)])
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

    def forward(self, x, pe, attn_fn):
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.heads, -1)
        k = self.to_k(x).view(b, n, self.heads, -1)
        v = self.to_v(x).view(b, n, self.heads, -1)
        q = apply_rope1(self.norm_q(q), pe)
        k = apply_rope1(self.norm_k(k), pe)
        out = attn_fn(q, k, v, self.heads)
        return self.to_out[0](out.reshape(b, n, -1))


def _split_rows(p):
    return p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)


def _modulated_norm(norm, x, scale, prefix_len, eps):
    s_prefix, s_target = scale
    ln = norm(x)
    if prefix_len:
        scale_row = torch.cat([s_prefix.expand(1, prefix_len, -1), s_target.expand(1, x.shape[1] - prefix_len, -1)], dim=1)
    else:
        scale_row = s_target
    return ln * (1 + scale_row)


def _gated_residual(x, y, gate, prefix_len):
    g_prefix, g_target = gate
    if prefix_len:
        x[:, :prefix_len] = x[:, :prefix_len] + y[:, :prefix_len] * g_prefix
    x[:, prefix_len:] = x[:, prefix_len:] + y[:, prefix_len:] * g_target
    return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, mlp_ratio=3, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, num_heads, head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio)

    def forward(self, x, mod, pe, attn_fn, prefix_len):
        scale1, gate1, scale2, gate2, _zero = mod
        h = _modulated_norm(self.img_norm1, x, scale1, prefix_len, self.eps)
        h = self.attn(h, pe, attn_fn)
        x = _gated_residual(x, h, gate1, prefix_len)
        h = _modulated_norm(self.img_norm2, x, scale2, prefix_len, self.eps)
        h = self.img_mlp(h)
        x = _gated_residual(x, h, gate2, prefix_len)
        return x


class LastLayer(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, x, temb):
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        return self.norm(x) * (1 + scale)


def _make_attention(segments):
    """Block-causal attention: text segment masked, later segments full."""

    def attn(q, k, v, heads):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        outs = []
        for start, end, mask in segments:
            qs = q[:, :, start:end]
            ks = k[:, :, :end]
            vs = v[:, :, :end]
            # A regional mask is an additive bias and must share the query dtype.
            if mask is not None and mask.dtype != torch.bool and mask.dtype != qs.dtype:
                mask = mask.to(qs.dtype)
            outs.append(attention(qs, ks, vs, attn_mask=mask))
        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        return out.transpose(1, 2)

    return attn


# ---------------------------------------------------------------------------
# Joint-sequence conditioning (reference images). Ported from the reference
# ``QwenImage21Transformer2DModel`` / ``QwenImage21Rope``; the text encoder
# reserves one vision slot per 2x2 group of latent tokens.
# ---------------------------------------------------------------------------

# Each vision-language image slot stands for a 2x2 group of latent tokens.
IMG_TOKENS_PER_SLOT = 4


class QwenRef:
    """Reference-image conditioning for a Qwen-Image 2.1 run.

    ``latents`` is ``[B, S, 64]``: the normalized condition-image latents, one
    flattened ``h*w`` block per image, in the same order as ``shapes``.
    ``shapes`` is the list of ``(1, h, w)`` latent grids for the condition
    images (the target image's grid is appended by the model). ``image_pad_mask``
    is ``[B, T]`` bool over the text context, ``True`` at the vision slots the
    text encoder reserved and ``False`` at text positions; each slot holds 4
    latent tokens.
    """

    def __init__(self, latents, shapes, image_pad_mask):
        self.latents = latents
        self.shapes = list(shapes)
        self.image_pad_mask = image_pad_mask


def _rope_ids(image_pad_mask, img_shapes, device):
    """Frame/height/width rotary positions for the joint sequence.

    Text tokens advance one shared position on all three axes; every image block
    freezes the frame at the position the preceding text reached and lays its
    tokens on a zero-centred height/width grid. Matches ``QwenImage21Rope``.
    """
    is_img = image_pad_mask.tolist()
    total = len(is_img)
    frame, img_h, img_w = [], [], []
    cursor = position = 0
    for _frame, h, w in img_shapes:
        block_start = is_img.index(True, cursor)
        text_len = block_start - cursor
        frame.extend(range(position, position + text_len))
        position += text_len
        cursor = block_start + h * w
        frame.extend([position] * (h * w))
        position += max(h, w)
        img_h.extend([y for y in range(-(h - h // 2), h // 2) for _ in range(w)])
        img_w.extend([x for _ in range(h) for x in range(-(w - w // 2), w // 2)])
    if cursor < total:
        frame.extend(range(position, position + total - cursor))
    frame_index = torch.tensor(frame, dtype=torch.float32, device=device)
    height_index = frame_index.clone()
    width_index = frame_index.clone()
    height_index[image_pad_mask] = torch.tensor(img_h, dtype=torch.float32, device=device)
    width_index[image_pad_mask] = torch.tensor(img_w, dtype=torch.float32, device=device)
    return torch.stack([frame_index, height_index, width_index], dim=-1)


def _build_token_metadata(image_pad_mask, img_shapes):
    """Label every token with its image block and mark the target's tokens.

    Block boundaries come from the token counts in ``img_shapes``, not from runs
    of ``True``: two condition images with no text between them form one run but
    must stay separate blocks. Matches ``build_token_metadata``.
    """
    positions = image_pad_mask.nonzero(as_tuple=True)[0]
    lengths = [int(h) * int(w) for _frame, h, w in img_shapes]
    if sum(lengths) != positions.numel():
        raise ValueError(
            f"img_shapes accounts for {sum(lengths)} image tokens but image_pad_mask "
            f"marks {positions.numel()}."
        )
    image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
    block_ids = torch.tensor(
        [block for block, n in enumerate(lengths) for _ in range(n)],
        device=image_pad_mask.device,
    )
    image_ids[positions] = block_ids
    target_token_mask = torch.zeros_like(image_pad_mask)
    target_token_mask[positions[-lengths[-1]:]] = True
    return image_ids, target_token_mask


def _block_causal_segments(image_ids, prefix_len, total_len, device):
    """Split the prefix into ``(start, end, mask)`` block-causal segments.

    A text run attends causally to its own keys and fully to everything before
    it; an image block attends fully to everything before it and stays
    bidirectional within the block; the target (appended last) attends to the
    whole sequence. The exact multi-pass form of the reference's
    ``QwenImage21AttnProcessor``.
    """
    segments = []
    prefix = image_ids[:prefix_len].tolist()
    start = 0
    for index in range(1, prefix_len + 1):
        if index == prefix_len or prefix[index] != prefix[start]:
            if prefix[start] < 0:  # text run: causal over its own block
                n = index - start
                mask = torch.cat([
                    torch.ones(n, start, dtype=torch.bool, device=device),
                    torch.tril(torch.ones(n, n, dtype=torch.bool, device=device)),
                ], dim=1)
            else:  # image block: full over [0, end), bidirectional within
                mask = None
            segments.append((start, index, mask))
            start = index
    segments.append((prefix_len, total_len, None))
    return segments


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.heads = num_attention_heads

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, eps=eps)
            for _ in range(num_layers)
        ])
        self.norm_out = LastLayer(self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)
        self._block_storage = None
        self._block_compute = None
        self._resident_count = 0

    @property
    def blocks(self):
        """The streamed transformer stack, under the name ``place_dit`` looks for."""
        return self.transformer_blocks

    def enable_block_offload(self, storage_device, compute_device):
        """Keep ``transformer_blocks`` on ``storage_device`` and stream them through
        ``compute_device`` one at a time (the bf16 model is larger than 12 GB VRAM)."""
        self._block_storage = torch.device(storage_device)
        self._block_compute = torch.device(compute_device)

    def set_resident_blocks(self, count, device):
        """Keep the first ``count`` blocks on ``device``; stream the rest.

        With a card too small for the whole DiT, streaming all 32 blocks across
        PCIe every step dominates; holding as many as fit cuts that traffic.
        """
        self._resident_count = max(0, min(int(count), len(self.transformer_blocks)))
        for block in self.transformer_blocks[:self._resident_count]:
            block.to(device=device)

    def place_head(self, device):
        """Move everything except the streamed ``transformer_blocks`` to a device.

        The dtype is left as loaded (the model is already in the target dtype),
        so fp8 block/head weights chosen by :func:`engine.precision.place_dit`
        are not clobbered by a cast.
        """
        for mod in (self.time_text_embed, self.txt_in, self.img_in, self.modulation, self.norm_out, self.proj_out):
            mod.to(device=device)
        self.transformer_blocks.to(self._block_storage if self._block_storage is not None else device)

    def build_sequence(self, x, context, ref=None, cross_attn_mask=None):
        """Build the joint text/image sequence, its RoPE and attention segments.

        ``ref`` (a :class:`QwenRef`) packs the condition-image latents ahead of
        the target and drops them into the vision slots of ``context``; without
        it the sequence is text followed by the target image. Each vision slot
        holds a 2x2 group of latent tokens; the target contributes its latent
        tokens directly, so any latent size is fine.
        """
        h, w = x.shape[-2:]
        target = self.img_in(x.flatten(2).transpose(1, 2))
        txt = self.txt_in(context)

        if ref is not None:
            cond = self.img_in(ref.latents.to(target.dtype))
            vlm_img = ref.image_pad_mask[0].to(torch.bool)
            img_shapes = list(ref.shapes) + [(1, h, w)]
        else:
            cond = None
            vlm_img = torch.zeros(txt.shape[1], dtype=torch.bool, device=x.device)
            img_shapes = [(1, h, w)]

        # Expand only the text context's vision slots (text passes through once,
        # each slot expands to a 2x2 group); the target tokens are appended as-is.
        repeats = torch.where(vlm_img, IMG_TOKENS_PER_SLOT, 1)
        context_mask = torch.repeat_interleave(vlm_img, repeats)
        context_seq = txt.repeat_interleave(repeats, dim=1)
        if cond is not None:
            if cond.shape[1] != int(context_mask.sum()):
                raise ValueError(
                    f"conditioning carries {cond.shape[1]} image tokens but the vision "
                    f"slots expect {int(context_mask.sum())}."
                )
            context_seq[:, context_mask] = cond
        seq = torch.cat([context_seq, target], dim=1)
        image_pad_mask = torch.cat([context_mask, vlm_img.new_ones(target.shape[1])])

        ids = _rope_ids(image_pad_mask, img_shapes, x.device)
        pe = self.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()

        image_ids, target_token_mask = _build_token_metadata(image_pad_mask, img_shapes)
        prefix_len = int((~target_token_mask).sum())
        segments = _block_causal_segments(image_ids, prefix_len, seq.shape[1], x.device)
        if cross_attn_mask is not None:
            # A regional run restricts the target tokens to the base prompt plus
            # the token's own region block (see engine.regions).
            segments[-1] = (
                prefix_len, seq.shape[1],
                cross_attn_mask[0, 0, prefix_len:seq.shape[1], :seq.shape[1]],
            )
        return seq, pe, segments

    def forward(self, x, timestep, context, cross_attn_mask=None, ref=None):
        b, _c, h, w = x.shape
        dtype = x.dtype
        hidden, pe, segments = self.build_sequence(x, context, ref=ref, cross_attn_mask=cross_attn_mask)
        prefix_len = hidden.shape[1] - h * w

        t = ((timestep.to(dtype) * 1000.0).to(dtype) / 1000.0).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]))
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2), _split_rows(gate2.tanh()), None)

        attn_fn = _make_attention(segments)
        for index, block in enumerate(self.transformer_blocks):
            streamed = self._block_compute is not None and index >= self._resident_count
            if streamed:
                block.to(device=self._block_compute)
            hidden = block(hidden, mod, pe, attn_fn, prefix_len)
            if streamed:
                block.to(device=self._block_storage)
        hidden = self.norm_out(hidden[:, prefix_len:], temb[:-1])
        hidden = self.proj_out(hidden)
        return hidden.transpose(1, 2).reshape(b, self.out_channels, h, w)


DEFAULT_CONFIG = dict(
    in_channels=64, out_channels=64, num_layers=32, attention_head_dim=128,
    num_attention_heads=32, context_in_dim=4096, mlp_ratio=3, axes_dims_rope=(16, 56, 56),
)


# Container prefixes a converted GGUF's keys may carry (ComfyUI stores the DiT
# under ``model.diffusion_model.``). Longest first so the check is unambiguous.
_GGUF_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "model.")


def _gguf_prefix(names):
    """The container prefix a GGUF's keys carry (``""`` when they carry none)."""
    for prefix in _GGUF_PREFIXES:
        if prefix + "img_in.weight" in names:
            return prefix
    return ""


def _stream_name_map(prefix):
    """Map a Qwen-Image GGUF key to model parameter name(s) for disk streaming.

    Most keys map 1:1 after the container prefix is stripped. Conversions that
    fuse the SwiGLU MLP into ``img_mlp.gate_up.weight`` (gate rows first, then
    up) instead feed two parameters, so a fused key returns both names with the
    slice each needs.
    """
    fused_tail = ".img_mlp.gate_up.weight"

    def map_name(source_name):
        if not source_name.startswith(prefix):
            return None
        name = source_name[len(prefix):]
        if name.endswith(fused_tail):
            base = name[: -len(fused_tail)] + ".img_mlp."
            return [
                (base + "gate_layer.weight", lambda full: full[: full.shape[0] // 2]),
                (base + "proj.weight", lambda full: full[full.shape[0] // 2:]),
            ]
        return name

    return map_name


def _is_gguf(path):
    return str(path).lower().endswith(".gguf")


def _safetensors_weight(handle, names, key, dtype):
    """Read one safetensors tensor into a dense CPU tensor of ``dtype``.

    Plain bf16/f16 tensors are copied row-chunked; fp8 weights are dequantized
    with their scale side-car (see :func:`engine.krea2._fp8_reader`) and ComfyUI
    ``int8_tensorwise`` (+ convrot) weights are reconstructed (see
    :func:`engine.quant_formats._copy_weight`), so a plain ``F.linear`` reproduces
    ComfyUI's result.
    """
    slice_ = handle.get_slice(key)
    shape = tuple(slice_.get_shape())
    buf = torch.empty(shape, dtype=dtype)
    if slice_.get_dtype().startswith("F8"):
        from .krea2 import _fp8_reader

        buf.copy_(_fp8_reader(handle, names)(key).reshape(shape).to(dtype))
    else:
        from .quant_formats import _copy_weight

        _copy_weight(handle, names, key, buf, dtype)
    return buf


def _model_source(path, dtype):
    """A ``(names, get, raw, bytes_of, close)`` source for a DiT checkpoint.

    ``names`` is the set of model parameter names (the on-disk key minus its
    container prefix); ``get(name)`` returns that parameter dequantized to
    ``dtype``; ``raw(key)`` returns a (possibly fused) tensor by its stripped
    key, for splitting conversions that merge the SwiGLU MLP; ``bytes_of(name)``
    estimates the tensor's dense bf16 size from the header, for streaming
    budgets. GGUF is read via :mod:`engine.gguf`; safetensors may be bf16, fp8
    (with a scale side-car) or ComfyUI ``int8_tensorwise`` with the optional
    ``convrot`` rotation.
    """
    if _is_gguf(path):
        from . import gguf as gguf_reader

        reader = gguf_reader.gguf.GGUFReader(path)
        tensors = {t.name: t for t in reader.tensors}
        prefix = _gguf_prefix(tensors)

        def raw(key):
            return gguf_reader.dequantize(tensors[prefix + key], dtype=dtype)

        def bytes_of(name):
            tensor = tensors.get(prefix + name)
            if tensor is None:
                return 0
            count = 1
            for dim in tensor.shape:
                count *= int(dim)
            return int(count * 2)

        names = {k[len(prefix):] for k in tensors if k.startswith(prefix)}
        return names, raw, raw, bytes_of, (lambda: None)

    from safetensors import safe_open

    handle = safe_open(path, framework="pt", device="cpu")
    keys = list(handle.keys())
    name_set = set(keys)
    mapping = {}
    for key in keys:
        stripped = key
        for prefix in _GGUF_PREFIXES:
            if key.startswith(prefix):
                stripped = key[len(prefix):]
                break
        # Skip quantisation side-cars (``X.weight_scale`` / ``X.comfy_quant``):
        # they are not model parameters, only inputs to the dequantizer.
        if stripped.endswith("_scale") or stripped.endswith(".comfy_quant"):
            continue
        mapping.setdefault(stripped, key)

    def raw(key):
        return _safetensors_weight(handle, name_set, mapping.get(key, key), dtype)

    def bytes_of(name):
        key = mapping.get(name)
        if key is None:
            return 0
        count = 1
        for dim in handle.get_slice(key).get_shape():
            count *= int(dim)
        return int(count * 2)

    def close():
        try:
            handle.__exit__(None, None, None)
        except Exception:
            pass

    return set(mapping), raw, raw, bytes_of, close


def load_transformer(path, device="cpu", dtype=torch.bfloat16, stream=False,
                     stream_device=None, ram_budget=0, prefetch_depth=0, pin_blocks=True):
    """Build the MMDiT and load weights from a GGUF or safetensors checkpoint.

    Handles both key layouts seen in the wild: the bare ``img_in.weight`` names
    and the ComfyUI-style ``model.diffusion_model.`` prefix. Some conversions
    also fuse the SwiGLU MLP into a single ``img_mlp.gate_up.weight`` (gate rows
    first, then up); those are split back into ``gate_layer``/``proj`` here.
    Safetensors may carry plain bf16/fp8 weights or ComfyUI's
    ``int8_tensorwise`` quantisation with the optional ``convrot`` rotation
    (``*_int8_convrot`` files; see :func:`_safetensors_weight`).

    The model is built on the ``meta`` device (no parameter storage) and the
    dequantised weights are then assigned in place, so the ~15 GB of bf16
    parameters are never allocated twice. With ``stream`` the blocks are instead
    read from the checkpoint on demand (see :mod:`engine.streaming`), so the
    whole transformer never has to sit in system RAM.
    """
    model_factory = lambda: QwenImage21Transformer2DModel(**DEFAULT_CONFIG)

    if stream:
        from . import streaming as streaming_module

        if _is_gguf(path):
            from . import gguf as gguf_reader

            names = {t.name for t in gguf_reader.gguf.GGUFReader(path).tensors}
            source = streaming_module.GgufSource(
                path, name_map=_stream_name_map(_gguf_prefix(names))
            )
        else:
            names, get, _raw, bytes_of, close = _model_source(path, dtype)
            source = streaming_module.CallableSource(
                names, get, bytes_of=bytes_of, close=close,
            )
        model, _streamer = streaming_module.build_streaming_model(
            model_factory, source, device=stream_device or device, dtype=dtype,
            ram_budget=ram_budget, prefetch_depth=prefetch_depth, pin_blocks=pin_blocks,
        )
        missing = [n for n in dict(model.named_parameters()) if not source.has(n)]
        return model, missing, []

    with build_on_meta(dtype, model_factory) as model:
        pass
    params = dict(model.named_parameters())
    state = {}
    names, get, raw, _bytes_of, close = _model_source(path, dtype)

    try:
        fused_cache = {}
        for name, dst in params.items():
            if name in names:
                state[name] = get(name).reshape(dst.shape)
                continue

            # Split a fused ``img_mlp.gate_up.weight`` back into its two halves.
            if name.endswith(".img_mlp.gate_layer.weight"):
                fused_name = name[: -len("gate_layer.weight")] + "gate_up.weight"
                take_up = False
                sibling = name[: -len("gate_layer.weight")] + "proj.weight"
            elif name.endswith(".img_mlp.proj.weight"):
                fused_name = name[: -len("proj.weight")] + "gate_up.weight"
                take_up = True
                sibling = name[: -len("proj.weight")] + "gate_layer.weight"
            else:
                continue
            if fused_name not in names:
                continue
            full = fused_cache.get(fused_name)
            if full is None:
                full = raw(fused_name)
                fused_cache[fused_name] = full
            half = full.shape[0] // 2
            state[name] = (full[half:] if take_up else full[:half]).reshape(dst.shape).clone()
            if sibling in state:
                fused_cache.pop(fused_name, None)
    finally:
        close()

    model.load_state_dict(state, strict=False, assign=True)
    missing = [n for n in params if n not in state]
    # Move the loaded parameters to the target dtype/device, skipping any the file
    # did not carry (they stay on the meta device; a whole-module ``.to()`` would
    # raise on their missing storage). ``missing`` is reported and rejected upstream.
    for p in model.parameters():
        if p.device.type != "meta" and (p.dtype != dtype or str(p.device) != str(device)):
            p.data = p.to(device=device, dtype=dtype)
    model = model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, missing, []
