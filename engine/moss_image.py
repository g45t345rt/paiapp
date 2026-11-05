"""MOSS Image (OpenMOSS) DiT in plain torch.

MOSS Image is a 5B single-stream DiT (42 blocks, ``dim=3072``) that operates
directly on a 128-channel latent at 1/16 resolution (the FLUX.2 VAE's 32
channels, 2x2 patch-packed) -- there is no patchify step inside the model. It
conditions on the last hidden states of a Qwen3.5-9B text encoder
(:mod:`engine.moss_text`) and is sampled with flow-matching Euler at shift 5.

Ported from the reference Diffusers ``MOSSImageTransformer`` (Apache-2.0,
OpenMOSS-Team) and ComfyUI's ``comfy/ldm/moss_image/model.py``, rewritten to
depend only on ``torch``. Text2image / image2image only (no reference-image edit
LoRA, no control maps).

Effective config (detected from the checkpoint): ``dim=3072``, ``ffn_dim=8192``,
24 heads (``head_dim=128``), 42 layers, ``text_dim=4096`` (Qwen3.5-9B),
``latent_channels=128``, ``freq_dim=256``, RoPE base 10000 with four 32-wide
axes. The main blocks may store a fused ``proj.qkv`` (ComfyUI repack) or the
separate ``proj.q``/``proj.k``/``proj.v``; both are supported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gguf as gguf_reader
from .attention import attention
from .precision import build_on_dtype, skip_random_init


def _timestep_embedding(times, freq_dim, time_factor=1000.0):
    """Sinusoidal timestep embedding, computed in float64 like the reference.

    ``times`` is a ``[B]`` float tensor of flow sigmas in ``[0, 1]``; returns
    ``[B, freq_dim]`` in float32. The reference (ComfyUI's ``flux.layers``
    ``timestep_embedding``) scales the input by ``time_factor=1000`` before the
    sinusoid; MOSS is sampled with ``multiplier=1.0`` so the sigma is passed in
    unscaled and must be expanded here.
    """
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = (times.double() * time_factor)[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1).float()


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    """Soft-capped gated activation used by the DiT's feed-forward."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    """Rotary embedding on half-split pairs (the MOSS / reference convention).

    ``rope`` is ``[..., head_dim]`` holding ``[sin_bands, cos_bands]`` (the four
    axes' bands concatenated). Each head's ``head_dim`` splits into two halves
    rotated as ``(a, b) -> (a*cos - b*sin, b*cos + a*sin)``, matching the
    reference ``MOSSImageTransformer._rotate`` and ComfyUI's
    ``comfy/ldm/moss_image/model._rotate`` -- *not* flux's adjacent-pair rule.
    """
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads, mask=None):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    if mask is not None and mask.dtype != torch.bool and mask.dtype != q.dtype:
        mask = mask.to(q.dtype)
    out = attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    """One of the two blocks that refine the projected text tokens."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    """A joint image+text transformer block with shared adaLN modulation."""

    def __init__(self, dim, ffn_dim, heads, fused_qkv=False):
        super().__init__()
        self.heads = heads
        self.fused_qkv = fused_qkv
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        names = ("qkv",) if fused_qkv else ("q", "k", "v")
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, 3 * dim if name == "qkv" else dim, bias=False)
            for name in (*names, "o")
        })
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, mask=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["o"].weight.dtype
        y = y.to(dtype)
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, -1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, v, rope, self.heads, mask)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    """Four-axis (r, f, h, w) rotary embedding for the joint sequence.

    The four axes are packed axis-major, 16 bands each, into 64 frequencies and
    laid out as ``[sin (64), cos (64)]`` along the head dimension; :func:`_rotate`
    applies them as half-split pairs. The image tokens come first (rows/columns
    centred on zero), then the text tokens. Frequencies are held in fp32 and
    recomputed on placement (they must not be rounded by a bf16 cast).
    """

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")), persistent=False)

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.bands = self._default_bands(self.bands.device)
        return self

    def forward(self, height, width, text_length, device):
        if self.bands.dtype != torch.float32 or self.bands.device != torch.device(device):
            self.bands = self._default_bands(torch.device(device))
        h, w = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij",
        )
        image = torch.stack(
            (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
        ).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        ids = torch.cat((image, text), 0).unsqueeze(-1)
        # One frequency per (axis, band); the four axes are concatenated as
        # sin(axes) then cos(axes) -- see ``_rotate``.
        phase = ids * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImage(nn.Module):
    """The MOSS Image DiT (single-stream, 128-channel latent, no patchify)."""

    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42,
                 text_dim=4096, freq_dim=256, latent_channels=128,
                 rope_base=10000.0, fused_qkv=False):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, even freq_dim.")
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.latent_channels = latent_channels
        self.fused_qkv = fused_qkv

        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([
            _Block(dim, ffn_dim, num_heads, fused_qkv=fused_qkv) for _ in range(num_layers)
        ])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

        self._block_storage = None
        self._block_compute = None

    # ------------------------------------------------------------- placement
    def enable_block_offload(self, storage_device, compute_device):
        self._block_storage = torch.device(storage_device)
        self._block_compute = torch.device(compute_device)

    def place_head(self, device):
        for module in (self.video_embedder, self.text_embedder, self.rope, self.time_embedding,
                       self.shared_block_modulation, self.final_norm_video,
                       self.final_linear_video, self.video_head_modulation):
            module.to(device=device)
        storage = self._block_storage if self._block_storage is not None else device
        self.layers.to(storage)

    def _stream(self, module, *args, **kwargs):
        if self._block_compute is not None:
            module.to(device=self._block_compute)
        out = module(*args, **kwargs)
        if self._block_compute is not None:
            module.to(device=self._block_storage)
        return out

    # ---------------------------------------------------------------- forward
    def forward(self, x, timestep, context, cross_attn_mask=None, skip_layers=None):
        """Predict the flow velocity for latent ``x`` ``[B, C, H, W]``.

        ``context`` is the Qwen3.5 hidden states ``[B, S, text_dim]``.
        ``cross_attn_mask`` is an optional additive joint-attention mask (regional
        prompting). ``skip_layers`` is an optional iterable of zero-based block
        indices to skip (the reference CFG negative branch skips block 22).
        """
        skip = frozenset(skip_layers or ())
        batch, _, height, width = x.shape
        times = torch.as_tensor(timestep, device=x.device, dtype=torch.float32).reshape(-1).expand(batch)
        embedding = _timestep_embedding(times, self.freq_dim)
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(
            time.to(self.shared_block_modulation.weight.dtype)
        ).view(batch, 6, -1)
        head = self.video_head_modulation(
            time.to(self.video_head_modulation.weight.dtype)
        ).view(batch, 2, -1)

        outputs = []
        for index in range(batch):
            video = x[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            rope = self.rope(height, width, context.shape[1], x.device).to(video.dtype)
            text = self.text_embedder(
                context[index:index + 1].to(self.text_embedder.proj.weight.dtype),
                rope[:, height * width:],
            )
            tokens = torch.cat((video, text), 1)
            layer_mod = mod[index:index + 1]
            for layer_index, layer in enumerate(self.layers):
                if layer_index in skip:
                    continue
                tokens = self._stream(layer, tokens, rope, layer_mod, cross_attn_mask)
            out = self.final_norm_video(
                tokens[:, :height * width].to(self.final_norm_video.weight.dtype)
            )
            out = out * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            out = self.final_linear_video(out.to(self.final_linear_video.weight.dtype))
            outputs.append(out.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)


def _strip_prefix(shapes):
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if any(key.startswith(prefix) for key in shapes):
            return {key[len(prefix):]: value for key, value in shapes.items() if key.startswith(prefix)}
    return dict(shapes)


def _dim(entry, index):
    shape = getattr(entry, "shape", entry)
    return int(shape[index])


def _shape_numel(shape):
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _detect_config(shapes):
    """Detect the DiT config from a checkpoint's shapes (ComfyUI's signature rules)."""
    dim = _dim(shapes["shared_block_modulation.weight"], 0) // 6
    ffn_dim = _dim(shapes["layers.0.proj.ffn_up_gate.weight"], 0) // 2
    num_layers = 1 + max(
        int(key.split(".")[1]) for key in shapes if key.startswith("layers.")
    )
    text_dim = _dim(shapes["text_embedder.proj.weight"], 1)
    latent_channels = _dim(shapes["video_embedder.weight"], 1)
    return dict(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=dim // 128,
        num_layers=num_layers,
        text_dim=text_dim,
        latent_channels=latent_channels,
        fused_qkv="layers.0.proj.qkv.weight" in shapes,
    )


def _checkpoint_source(path, dtype):
    """A ``(shapes, fetch, close)`` triple that streams one tensor at a time.

    Nothing is materialised up front: the shapes are read from the header, and
    ``fetch(name)`` returns a single dequantized tensor. Holding the whole DiT
    (plus a second copy) at once is what overruns memory on a tight machine.
    """
    if path.lower().endswith(".gguf"):
        reader = gguf_reader.gguf.GGUFReader(path)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        shapes = {name: tuple(int(v) for v in reversed(t.shape)) for name, t in tensors.items()}
        return shapes, (lambda name: gguf_reader.dequantize(tensors[name], dtype=dtype)), (lambda: None)

    from safetensors import safe_open

    from .moss_text import _dequant

    handle = safe_open(path, framework="pt", device="cpu")
    keys = list(handle.keys())
    names = set(keys)
    # DiT files may carry the ComfyUI ``model.diffusion_model.`` prefix; map each
    # stripped parameter name back to its on-disk key so ``fetch`` can find it.
    prefixes = ("model.diffusion_model.", "diffusion_model.")
    mapping = {}
    for key in keys:
        stripped = key
        for prefix in prefixes:
            if key.startswith(prefix):
                stripped = key[len(prefix):]
                break
        mapping.setdefault(stripped, key)
    shapes = {name: tuple(handle.get_slice(key).get_shape()) for name, key in mapping.items()}

    def fetch(name):
        key = mapping.get(name, name)
        tensor = handle.get_tensor(key)
        if tensor.dtype == torch.int8 and (key + "_scale") in names:
            # ComfyUI int8 (optionally convrot) quantisation: reconstruct the
            # dense weight so a plain Linear reproduces ComfyUI's result.
            return _dequant(handle, names, key)
        return tensor

    return shapes, fetch, _close_handle(handle)


def _close_handle(handle):
    def close():
        try:
            handle.__exit__(None, None, None)
        except Exception:
            pass
    return close


def load_moss_image(path, device="cpu", dtype=torch.bfloat16, stream=False,
                    stream_device=None, ram_budget=0, prefetch_depth=0, pin_blocks=True):
    """Build the MOSS Image DiT and load weights from safetensors or GGUF.

    With ``stream`` the model is built on the meta device and its blocks are read
    on demand through the checkpoint's own ``fetch`` closure (see
    :mod:`engine.streaming`), so the DiT is never held in RAM.
    """
    shapes, fetch, close = _checkpoint_source(path, dtype)
    stripped = _strip_prefix(shapes)
    stripped.pop("rope.bands", None)

    if stream:
        from . import streaming as streaming_module

        def bytes_of(model_name):
            shape = stripped.get(model_name)
            return int(_shape_numel(shape) * 2) if shape is not None else 0

        reader = streaming_module.CallableSource(
            stripped.keys(), fetch, bytes_of=bytes_of, close=close,
        )
        factory = lambda: MOSSImage(**_detect_config(stripped))
        model, _streamer = streaming_module.build_streaming_model(
            factory, reader, device=stream_device or device, dtype=dtype,
            ram_budget=ram_budget, prefetch_depth=prefetch_depth, pin_blocks=pin_blocks,
        )
        params = dict(model.named_parameters())
        missing = [name for name in params if not reader.has(name)]
        unexpected = [name for name in stripped if name not in params]
        return model, missing, unexpected

    model = None
    try:
        # The RoPE frequencies are derived constants (recomputed on placement).
        # Skip the random initialisers: every weight is overwritten below, and
        # initialising billions of parameters in bf16 costs minutes.
        with skip_random_init():
            model = build_on_dtype(dtype, lambda: MOSSImage(**_detect_config(stripped)))
        params = dict(model.named_parameters())
        for parameter in params.values():
            parameter.requires_grad_(False)
        loaded = set()
        with torch.no_grad():
            for name, dst in params.items():
                if name not in stripped:
                    continue
                dst.copy_(fetch(name).reshape(dst.shape).to(dst.dtype))
                loaded.add(name)
        missing = [name for name in params if name not in loaded]
        unexpected = [name for name in stripped if name not in params]
    finally:
        close()
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, missing, unexpected

