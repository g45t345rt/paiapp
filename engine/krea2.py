"""Krea 2 (``krea2``) single-stream MMDiT in plain torch.

Krea 2 (Krea.ai, 2026) is a 12B single-stream MMDiT: the text tokens produced by
a Qwen3-VL-4B 12-layer ``txtfusion`` adapter and the patchified image tokens are
concatenated into one sequence and run through ``layers`` shared transformer
blocks with AdaLN-single modulation, GQA + per-head QK-norm + sigmoid-gated
attention, a SwiGLU MLP and 3-axis RoPE.

Ported from ComfyUI's ``comfy/ldm/krea2/model.py`` (itself from the HF diffusers
``Krea2Transformer2DModel``, Apache-2.0), rewritten to depend only on ``torch``.
The module attribute names match the ComfyUI tensor names exactly, so weights
load straight from a ComfyUI-style GGUF or single-file safetensors (see
:func:`load_transformer`). The RoPE and timestep helpers are shared with
:mod:`engine.flux`.

Text2image / image2image, plus reference-image conditioning through cached
per-block reference keys/values (see :meth:`Krea2Transformer2DModel.precompute_ref_kv`
and ``forward(ref_kv_cache=...)``), which the Krea 2 functional adapters
(AnyPaint and friends) use for arbitrary-mask editing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gguf as gguf_reader
from .attention import attention
from .flux import EmbedND, apply_rope, timestep_embedding
from .precision import build_on_meta


# ---------------------------------------------------------------------------
# Building blocks.
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm with the Krea 2 ``(1 + scale)`` convention (scale stored zero-centered)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        w = self.scale.float() + 1.0
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * w).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.qnorm = RMSNorm(dim, eps)
        self.knorm = RMSNorm(dim, eps)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


def _swiglu_dim(features, multiplier, multiple=128):
    hidden = int(2 * features / 3) * multiplier
    return multiple * ((hidden + multiple - 1) // multiple)


class SwiGLU(nn.Module):
    def __init__(self, features, multiplier):
        super().__init__()
        hidden = _swiglu_dim(features, multiplier)
        self.gate = nn.Linear(features, hidden, bias=False)
        self.up = nn.Linear(features, hidden, bias=False)
        self.down = nn.Linear(hidden, features, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """GQA self-attention with per-head QK-norm, RoPE and a sigmoid output gate."""

    def __init__(self, dim, heads, kvheads=None, bias=False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, freqs=None, mask=None, kv_capture=None, kv_cache=None):
        b, length, _ = x.shape
        q = self.wq(x).view(b, length, self.heads, self.headdim).movedim(1, 2)
        k = self.wk(x).view(b, length, self.kvheads, self.headdim).movedim(1, 2)
        v = self.wv(x).view(b, length, self.kvheads, self.headdim).movedim(1, 2)
        gate = self.gate(x)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        # Reference-image conditioning: capture this pass's pre-GQA keys/values
        # (``kv_capture``) or reuse another pass's cached reference keys/values as
        # extra attention keys (``kv_cache``). Capture happens before GQA repeat so
        # the cache stays at the key/value head count.
        if kv_capture is not None:
            kv_capture.append((k, v))
        if kv_cache is not None:
            k = torch.cat([k, kv_cache[0].to(k.dtype)], dim=2)
            v = torch.cat([v, kv_cache[1].to(v.dtype)], dim=2)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # A regional mask is an additive bias and must share the query dtype.
        if mask is not None and mask.dtype != torch.bool and mask.dtype != q.dtype:
            mask = mask.to(q.dtype)
        out = attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, length, self.heads * self.headdim)
        return self.wo(out * torch.sigmoid(gate))


class SharedModulation(nn.Module):
    """The per-block additive table over the shared 6-way time modulation."""

    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec):
        return (vec + self.lin).chunk(6, dim=-1)


class SimpleModulation(nn.Module):
    """The final layer's 2-way (scale, shift) additive table."""

    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec):
        out = vec + self.lin
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class TextFusionBlock(nn.Module):
    """Pre-norm transformer block (no RoPE, no time modulation) for the text fusion stage."""

    def __init__(self, features, heads, multiplier, kvheads=None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier)

    def forward(self, x, mask=None):
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusion(nn.Module):
    """Fuses the tapped text-encoder hidden-state stack into one text sequence.

    Two ``layerwise_blocks`` attend across the tapped-layer axis independently
    for every token, the ``projector`` collapses that axis, and two
    ``refiner_blocks`` attend across the token sequence.
    """

    def __init__(self, num_text_layers, dim, heads, multiplier, kvheads=None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([
            TextFusionBlock(dim, heads, multiplier, kvheads=kvheads) for _ in range(2)
        ])
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([
            TextFusionBlock(dim, heads, multiplier, kvheads=kvheads) for _ in range(2)
        ])

    def forward(self, x, mask=None):
        b, length, num_layers, dim = x.shape
        h = x.reshape(b * length, num_layers, dim)
        for block in self.layerwise_blocks:
            h = block(h.contiguous())
        h = h.reshape(b, length, num_layers, dim).permute(0, 1, 3, 2)
        h = self.projector(h).squeeze(-1)
        for block in self.refiner_blocks:
            h = block(h, mask=mask)
        return h


class TransformerBlock(nn.Module):
    def __init__(self, features, heads, multiplier, kvheads=None):
        super().__init__()
        self.mod = SharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier)

    def forward(self, x, vec, freqs, mask=None, kv_capture=None, kv_cache=None):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        h = self.attn(
            (1 + prescale) * self.prenorm(x) + preshift, freqs, mask,
            kv_capture=kv_capture, kv_cache=kv_cache,
        )
        x = x + pregate * h
        h = self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        x = x + postgate * h
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, vec):
        scale, shift = self.modulation(vec)
        return self.linear((1 + scale) * self.norm(x) + shift)


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------

class Krea2Transformer2DModel(nn.Module):
    def __init__(self, features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12,
                 multiplier=4, layers=28, patch=2, channels=16, theta=1000.0,
                 txtlayers=12, txtheads=20, txtkvheads=20):
        super().__init__()
        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)

        self.first = nn.Linear(channels * patch * patch, features, bias=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(features, heads, multiplier, kvheads=kvheads)
            for _ in range(layers)
        ])
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.txtfusion = TextFusion(txtlayers, txtdim, txtheads, multiplier, kvheads=txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            nn.Linear(txtdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.last = LastLayer(features, patch, channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features * 6),
        )

        self._block_storage = None
        self._block_compute = None
        self._resident_count = 0

    def enable_block_offload(self, storage_device, compute_device):
        """Keep ``blocks`` on ``storage_device`` and stream them through
        ``compute_device`` one at a time (bf16 is larger than most VRAM)."""
        self._block_storage = torch.device(storage_device)
        self._block_compute = torch.device(compute_device)

    def set_resident_blocks(self, count, device):
        """Keep the first ``count`` blocks on ``device``; stream the rest.

        With a card too small for the whole DiT, streaming all 28 blocks across
        PCIe every step dominates. Holding as many as fit cuts that traffic.
        """
        self._resident_count = max(0, min(int(count), len(self.blocks)))
        for block in self.blocks[:self._resident_count]:
            block.to(device=device)

    def place_head(self, device):
        """Move everything except the streamed ``blocks`` to a device."""
        for mod in (self.first, self.tmlp, self.txtfusion, self.txtmlp, self.last,
                    self.tproj, self.pe_embedder):
            mod.to(device=device)
        self.blocks.to(self._block_storage if self._block_storage is not None else device)

    @torch.no_grad()
    def precompute_ref_kv(self, ref_tokens, ref_positions):
        """Per-block keys/values for clean reference tokens at t=0.

        The reference tokens are run through the blocks on their own, so their
        per-block K/V are independent of the timestep and of the rest of the
        sequence; the denoiser then appends them as extra attention keys (see
        ``forward(ref_kv_cache=...)``). ``ref_tokens`` is the packed reference
        latents ``(B, seq, C * patch * patch)`` (encoded like :meth:`_process_img`)
        and ``ref_positions`` their ``(seq, 3)`` rotary coordinates.

        Only correct for adapters trained with isolated reference attention
        (AI-Toolkit's ``kv_cache`` model kwarg), which is what the Krea 2
        functional adapters use.
        """
        first = self.first
        x = ref_tokens.to(getattr(first, "compute_dtype", None) or first.weight.dtype)
        t_zero = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        t = self.tmlp(timestep_embedding(t_zero, self.tdim).unsqueeze(1))
        tvec = self.tproj(t)
        h = self.first(x)
        freqs = self.pe_embedder(ref_positions.unsqueeze(0))
        caches = []
        for index, block in enumerate(self.blocks):
            streamed = self._block_compute is not None and index >= self._resident_count
            if streamed:
                block.to(device=self._block_compute)
            captured = []
            h = block(h, tvec, freqs, None, kv_capture=captured)
            if streamed:
                block.to(device=self._block_storage)
            caches.append(captured[0])
        return caches

    def _process_img(self, x):
        patch = self.patch
        if x.shape[-2] % patch or x.shape[-1] % patch:
            x = F.pad(x, (0, (-x.shape[-1]) % patch, 0, (-x.shape[-2]) % patch))
        h, w = x.shape[-2] // patch, x.shape[-1] // patch
        img = x.reshape(x.shape[0], self.channels, h, patch, w, patch)
        img = img.permute(0, 2, 4, 1, 3, 5).reshape(x.shape[0], h * w, self.channels * patch * patch)
        ids = torch.zeros(h, w, 3, device=x.device, dtype=torch.float32)
        ids[..., 1] = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
        return img, ids.reshape(1, h * w, 3).repeat(x.shape[0], 1, 1), h, w

    def _unpack_context(self, context):
        b, seq, fused = context.shape
        if fused != self.txtlayers * self.txtdim:
            raise ValueError(
                f"Krea 2 expects conditioning with {self.txtlayers}x{self.txtdim} "
                f"features (a {self.txtlayers}-layer Qwen3-VL stack) but got {fused}."
            )
        return context.reshape(b, seq, self.txtlayers, self.txtdim)

    def forward(self, x, timestep, context, attention_mask=None,
                cross_attn_mask=None, ref_kv_cache=None):
        temporal = x.ndim == 5
        if temporal:
            b5, c5, t5, h5, w5 = x.shape
            x = x.reshape(b5 * t5, c5, h5, w5)
        first = self.first
        x = x.to(getattr(first, "compute_dtype", None) or first.weight.dtype)
        batch, _c, _h, _w = x.shape

        context = self._unpack_context(context)
        t = self.tmlp(timestep_embedding(timestep.to(x.dtype), self.tdim).unsqueeze(1))
        tvec = self.tproj(t)

        img, img_pos, h_tok, w_tok = self._process_img(x)
        img = self.first(img)

        context = self.txtfusion(context, mask=attention_mask[:, None, None, :] if attention_mask is not None else None)
        context = self.txtmlp(context)
        txt_len = context.shape[1]

        mask = cross_attn_mask
        if mask is None and attention_mask is not None:
            ones = torch.ones(batch, img.shape[1], dtype=torch.bool, device=x.device)
            mask = torch.cat([attention_mask, ones], dim=1)[:, None, None, :]
        # Cached reference K/V (see :meth:`precompute_ref_kv`) are appended as extra
        # always-valid attention keys after the text and image tokens.
        if ref_kv_cache is not None and mask is not None:
            ref_keys = ref_kv_cache[0][0].shape[2]
            extra = torch.ones(
                mask.shape[0], 1, 1, ref_keys, dtype=torch.bool, device=x.device
            )
            mask = torch.cat([mask, extra], dim=-1)

        combined = torch.cat([context, img], dim=1)
        pos = torch.cat([torch.zeros(batch, txt_len, 3, device=x.device, dtype=torch.float32), img_pos], dim=1)
        freqs = self.pe_embedder(pos)

        for index, block in enumerate(self.blocks):
            streamed = self._block_compute is not None and index >= self._resident_count
            if streamed:
                block.to(device=self._block_compute)
            combined = block(
                combined, tvec, freqs, mask,
                kv_cache=ref_kv_cache[index] if ref_kv_cache is not None else None,
            )
            if streamed:
                block.to(device=self._block_storage)

        out = self.last(combined, t)[:, txt_len:txt_len + h_tok * w_tok]
        patch = self.patch
        out = out.reshape(batch, h_tok, w_tok, self.channels, patch, patch)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(batch, self.channels, h_tok * patch, w_tok * patch)
        out = out[:, :, :_h, :_w]
        if temporal:
            out = out.reshape(b5, t5, self.channels, _h, _w).movedim(1, 2)
        return out


DEFAULT_CONFIG = dict(
    features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4,
    layers=28, patch=2, channels=16, theta=1000.0, txtlayers=12,
    txtheads=20, txtkvheads=20,
)


_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "model.", "net.")


def _pick_prefix(keys):
    """The container prefix a checkpoint's keys carry (``""`` when none)."""
    for prefix in _PREFIXES:
        if any(
            key.startswith(prefix + "blocks.") or key.startswith(prefix + "first.")
            or key.startswith(prefix + "tproj.")
            for key in keys
        ):
            return prefix
    return ""


def _layer_ids(keys, base):
    """Sorted block indices from keys like ``"blocks.0."``."""
    ids = []
    for key in keys:
        if key.startswith(base + "."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                ids.append(int(parts[1]))
    return ids


def _detect_config(shape, keys):
    """Fill the Krea 2 config from the checkpoint's tensor shapes."""
    config = dict(DEFAULT_CONFIG)

    first = shape("first.weight")
    if first and len(first) >= 2:
        config["features"] = int(first[0])
        config["channels"] = int(first[1]) // (config["patch"] ** 2)
    else:
        tproj = shape("tproj.1.weight")
        if tproj and len(tproj) >= 2:
            config["features"] = int(tproj[1])

    block_ids = _layer_ids(keys, "blocks")
    if block_ids:
        config["layers"] = 1 + max(block_ids)

    tdim = shape("tmlp.0.weight")
    if tdim and len(tdim) >= 2:
        config["tdim"] = int(tdim[1])

    head_dim = shape("blocks.0.attn.qknorm.qnorm.scale")
    if head_dim:
        head_dim = int(head_dim[0])
        config["heads"] = config["features"] // head_dim
        wk = shape("blocks.0.attn.wk.weight")
        if wk:
            config["kvheads"] = int(wk[0]) // head_dim

    txt = shape("txtmlp.1.weight")
    if txt and len(txt) >= 2:
        config["txtdim"] = int(txt[1])
    projector = shape("txtfusion.projector.weight")
    if projector and len(projector) >= 2:
        config["txtlayers"] = int(projector[1])
    txt_head_dim = shape("txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale")
    if txt_head_dim:
        txt_head_dim = int(txt_head_dim[0])
        config["txtheads"] = config["txtdim"] // txt_head_dim
        txt_wk = shape("txtfusion.layerwise_blocks.0.attn.wk.weight")
        if txt_wk:
            config["txtkvheads"] = int(txt_wk[0]) // txt_head_dim
    return config


_FP8_DTYPES = {dtype for dtype in (
    getattr(torch, "float8_e4m3fn", None),
    getattr(torch, "float8_e4m3fnuz", None),
    getattr(torch, "float8_e5m2", None),
    getattr(torch, "float8_e5m2fnuz", None),
) if dtype is not None}


def _scale_value(tensor):
    """A per-tensor fp8 scale as float32 (some converters pack it as e4m3 bytes)."""
    if tensor.dtype == torch.uint8:
        tensor = tensor.view(torch.float8_e4m3fn)
    return tensor.to(torch.float32)


def _fp8_reader(handle, names):
    """A ``get(key)`` that dequantizes fp8 weights with their scale side-cars.

    Comfy-Org ``*_fp8_scaled`` files store ``X.weight`` as float8 plus a per-tensor
    ``X.weight_scale`` (occasionally a global ``scaled_fp8``). Reading the raw
    bytes and casting to bf16 silently drops the scale and yields garbage weights.
    """
    def get(key):
        tensor = handle.get_tensor(key)
        if tensor.dtype not in _FP8_DTYPES:
            return tensor
        value = tensor.to(torch.float32)
        if key + "_scale" in names:
            value = value * _scale_value(handle.get_tensor(key + "_scale"))
        elif "scaled_fp8" in names:
            value = value * _scale_value(handle.get_tensor("scaled_fp8"))
        return value
    return get


def _weights_source(path, dtype):
    """A ``(keys, shape_of, get)`` triple that reads one tensor at a time.

    ``get(name)`` returns a dense CPU tensor and ``shape_of(name)`` its logical
    shape, so the loader never has to materialise the whole checkpoint at once
    (Krea 2 is ~24 GB in bf16, and a second full copy will not fit). fp8 weights
    are dequantized with their scale side-cars (see :func:`_fp8_reader`).
    """
    if path.lower().endswith(".gguf"):
        reader = gguf_reader.gguf.GGUFReader(path)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        return (
            set(tensors),
            lambda name: tuple(int(v) for v in reversed(tensors[name].shape)),
            lambda name: gguf_reader.dequantize(tensors[name], dtype=dtype),
        )
    from safetensors import safe_open

    handle = safe_open(path, framework="pt", device="cpu")
    keys = set(handle.keys())
    return (
        keys,
        lambda name: tuple(handle.get_slice(name).get_shape()),
        _fp8_reader(handle, keys),
    )


def _numel(shape):
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _streaming_reader(path, dtype, keys, shape_of, get):
    """Wrap Krea 2's one-tensor ``get`` source as a streaming reader.

    The prefix is picked the same way the dense loader picks it, and each model
    parameter name is mapped back to ``prefix + name`` for the read.
    """
    from . import streaming as streaming_module

    prefix = _pick_prefix(keys)
    stripped = {key[len(prefix):] for key in keys if key.startswith(prefix)}

    def bytes_of(model_name):
        key = prefix + model_name
        return int(_numel(shape_of(key)) * 2) if key in keys else 0

    return streaming_module.CallableSource(
        stripped, lambda name: get(prefix + name), bytes_of=bytes_of,
    ), prefix, stripped


def load_transformer(path, device="cpu", dtype=torch.bfloat16, tick=None, stream=False,
                     stream_device=None, ram_budget=0, prefetch_depth=0, pin_blocks=True):
    """Build the Krea 2 MMDiT and load it from a GGUF or single-file safetensors.

    Weights are streamed one tensor at a time (see :func:`_weights_source`) and
    the model is built on the ``meta`` device (:func:`build_on_meta`), so
    peak CPU memory is roughly one copy of the checkpoint instead of three. GGUF
    weights are dequantized on the fly. Only a plain bf16/fp16 checkpoint or a
    GGUF is handled here -- the ComfyUI ``fp8_scaled`` / ``int8`` single files
    carry per-tensor scale side-cars and are not decoded (use the bf16 file or a
    GGUF, then let :func:`engine.precision.place_dit` choose residency).

    With ``stream`` the model is built on the meta device and its blocks are read
    on demand (see :mod:`engine.streaming`), so the checkpoint never has to be
    held in RAM at all.

    ``tick`` is an optional ``tick(fraction)`` progress callback called as
    tensors are copied (Krea 2's ~12B weights make the read itself take a while).
    """
    keys, shape_of, get = _weights_source(path, dtype)
    prefix = _pick_prefix(keys)
    stripped_keys = {key[len(prefix):] for key in keys if key.startswith(prefix)}

    def shape(name):
        key = prefix + name
        return shape_of(key) if key in keys else None

    if stream:
        from . import streaming as streaming_module

        reader, _prefix, _stripped = _streaming_reader(path, dtype, keys, shape_of, get)
        factory = lambda: Krea2Transformer2DModel(**_detect_config(shape, stripped_keys))
        model, _streamer = streaming_module.build_streaming_model(
            factory, reader, device=stream_device or device, dtype=dtype,
            ram_budget=ram_budget, prefetch_depth=prefetch_depth, pin_blocks=pin_blocks,
        )
        params = dict(model.named_parameters())
        missing = [n for n in params if not reader.has(n)]
        return model, missing, []

    # Build on the ``meta`` device: no ~24 GB bf16 parameter storage and no
    # random initialiser over the ~12B parameters. Each tensor is read one at a
    # time (see :func:`_weights_source`) and *assigned* in place (``assign=True``).
    with build_on_meta(
        dtype, lambda: Krea2Transformer2DModel(**_detect_config(shape, stripped_keys))
    ) as model:
        pass

    params = dict(model.named_parameters())
    total = max(1, len(params))
    state = {}
    loaded = set()
    for index, name in enumerate(params):
        if tick is not None and index % 16 == 0:
            tick(index / total)
        key = prefix + name
        if key not in keys:
            continue
        src = get(key)
        state[name] = src.reshape(params[name].shape).to(dtype)
        del src
        loaded.add(name)
    if tick is not None:
        tick(1.0)
    model.load_state_dict(state, strict=False, assign=True)
    missing = [n for n in params if n not in loaded]

    # Materialise the loaded parameters in the target dtype/device; params the
    # file did not carry stay on ``meta`` and are skipped.
    for p in model.parameters():
        if p.device.type != "meta" and (p.dtype != dtype or str(p.device) != str(device)):
            p.data = p.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, missing, []
