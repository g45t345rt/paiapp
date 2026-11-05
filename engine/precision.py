"""Runtime placement for DiTs on consumer GPUs.

Three automatic behaviours that matter on a small card (e.g. a 12 GB RTX 4070):

* **resident** -- if the whole transformer fits in free VRAM (with headroom), put
  it there directly instead of streaming blocks through the GPU every step. This
  is what makes Z-Image-Turbo fast.
* **fp8** -- when a model does not fit in bf16 but would in fp8, store its
  ``nn.Linear`` weights as ``float8_e4m3fn`` and cast to the compute dtype on the
  fly. This halves the footprint at a small quality cost (the same thing
  ComfyUI's fp8 checkpoints do).
* **disk stream** -- when even that does not fit *and* system RAM is the
  constraint, stream the transformer blocks from disk via :mod:`engine.streaming`
  (see that module); the model is built on the meta device and every block is
  materialised on use.

The mode is chosen at load time and applies to :mod:`engine.flux`,
:mod:`engine.z_image` and :mod:`engine.krea2`, which only
need to expose ``to_fp8``-compatible ``nn.Linear`` layers plus
``place_head``/``enable_block_offload``.

Every policy and margin here is user-tunable through :mod:`engine.tuning`
(the Settings dialog's Generation tab). :func:`configure` is the single write
path; :data:`DEFAULT_RESERVE`/:data:`STREAM_RESERVE` are the shipped defaults
that :data:`_CONFIG` starts from, so an untouched install behaves exactly as
before these controls existed.
"""

import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Leave this much free VRAM for the VAE decode, activations and fragmentation.
DEFAULT_RESERVE = 2_500_000_000

# A model that has to stream its blocks needs a bigger margin than one that fits
# whole: the resident blocks stay on the GPU through the VAE decode, so the
# per-step activations, the head modules and the decode all have to fit in what
# is left. Too small a margin (2.5 GB) makes a 12B DiT OOM mid-generation on a
# 12 GB card, so keep a wider gap before holding blocks resident.
STREAM_RESERVE = 4_000_000_000

#: Live policy, overwritten by :func:`configure` (from :mod:`engine.tuning`).
_CONFIG = {
    "fp8_mode": "auto",             # auto | fp8 | full
    "vram_reserve": DEFAULT_RESERVE,
    "stream_reserve": STREAM_RESERVE,
    "keep_resident": True,
    "compile_vae": True,
    "tiled_decode": False,
    "tile_size": 512,
    "tile_overlap": 64,
    "text_encoder_device": "cpu",   # cpu | auto
    "disk_stream": False,
    "ram_budget": 0,
    "prefetch_depth": 0,
}


def configure(**values):
    """Update the live placement policy from :mod:`engine.tuning`.

    Only recognised keys are written so an older caller (or a test) cannot
    introduce a typo key that silently does nothing.
    """
    for key, value in values.items():
        if key in _CONFIG:
            _CONFIG[key] = value
    return dict(_CONFIG)


def config():
    """A copy of the live policy (for status text and the loaded-models panel)."""
    return dict(_CONFIG)


@contextlib.contextmanager
def skip_random_init():
    """No-op torch's random initialisers while a model is constructed.

    Initialising billions of parameters costs minutes in bf16 (its
    ``normal_``/``uniform_`` CPU kernels are far slower than fp32), and a loader
    that fills every weight from a checkpoint overwrites them immediately anyway.
    Buffers a model derives from its config (e.g. RoPE frequencies) are plain
    arithmetic and are still computed.
    """
    import torch.nn.init as init

    saved = {}
    for name in ("normal_", "uniform_", "kaiming_uniform_", "kaiming_normal_",
                 "xavier_uniform_", "xavier_normal_", "trunc_normal_"):
        if hasattr(init, name):
            saved[name] = getattr(init, name)
            setattr(init, name, lambda tensor, *args, **kwargs: tensor)
    try:
        yield
    finally:
        for name, function in saved.items():
            setattr(init, name, function)


def build_on_dtype(dtype, factory):
    """Build a module with ``dtype`` as torch's default so its parameters are
    allocated in it directly.

    ``Model(config)`` creates its parameters in the default dtype (float32), so a
    12B checkpoint costs 48 GB and the later ``.to(bf16)`` needs that much again
    -- a peak of roughly three copies while the fp32 one is still referenced.
    Setting the default dtype makes ``nn.Linear``/``nn.LayerNorm``/``torch.ones``
    land in ``dtype`` from the start, so peak stays at about one copy. The
    previous default is restored even when ``factory`` raises.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        return factory()
    finally:
        torch.set_default_dtype(previous)


@contextlib.contextmanager
def build_on_meta(dtype, factory):
    """Build a module on the ``meta`` device so no parameter storage is allocated.

    Used by the disk-streaming path and by the GGUF loaders that overwrite every
    weight from the file: materialising billions of bf16 parameters only to
    replace them costs minutes of CPU time and gigabytes of RAM for nothing.
    Falls back to :func:`build_on_dtype` when ``accelerate`` is unavailable.

    The built module is yielded, so callers may use ``with build_on_meta(...)
    as model:``. On the meta device its parameters have no storage; the caller
    must fill them (e.g. ``load_state_dict(assign=True)`` or the streamer).
    """
    try:
        from accelerate import init_empty_weights
    except Exception:
        yield build_on_dtype(dtype, factory)
        return
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        # include_buffers=False keeps small derived buffers (e.g. RoPE tables)
        # real, matching WeeLLM's meta init; only parameters go to meta.
        with init_empty_weights(include_buffers=False):
            yield factory()
    finally:
        torch.set_default_dtype(previous)


class FP8Linear(nn.Module):
    """A ``nn.Linear`` whose weight is stored in fp8 and cast per forward.

    ``weight.dtype`` is ``float8_e4m3fn``, so it must never be used to pick the
    activation dtype; the original compute dtype is kept in ``compute_dtype``.
    """

    def __init__(self, in_features, out_features, bias=True, compute_dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        dtype = self.compute_dtype
        weight = self.weight.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None
        return F.linear(x.to(dtype), weight, bias)


def to_fp8(module):
    """Swap every ``nn.Linear`` under ``module`` for an fp8-backed equivalent."""
    for name, child in list(module.named_children()):
        if isinstance(child, FP8Linear):
            continue
        if isinstance(child, nn.Linear):
            # Deliberate fp32 islands (e.g. MiniMax H3's patch / final / AdaLN
            # projections) must keep their checkpoint precision; re-quantizing
            # them to fp8 loses it.
            if child.weight.dtype == torch.float32:
                continue
            new = FP8Linear(
                child.in_features, child.out_features, child.bias is not None,
                compute_dtype=child.weight.dtype,
            )
            with torch.no_grad():
                # A ``meta``-built model has no weight storage: leave the new
                # fp8 weight empty for the loader to fill (see ``h3.load_dit``),
                # which avoids first materialising a full bf16 copy.
                if not child.weight.is_meta:
                    new.weight.copy_(child.weight.detach().to(torch.float8_e4m3fn))
                if child.bias is not None and not child.bias.is_meta:
                    new.bias.copy_(child.bias.detach())
            for p in new.parameters():
                p.requires_grad_(False)
            setattr(module, name, new)
        else:
            to_fp8(child)
    return module


def try_to_fp8(module):
    """Apply :func:`to_fp8`, returning whether the model ended up quantized.

    fp8 storage is not available on every device: some AMD/ROCm targets (and
    pre-Ada NVIDIA cards) cannot hold ``float8_e4m3fn`` tensors, so a conversion
    that raises is reported as "not quantized" and the caller keeps bf16 rather
    than failing the load.
    """
    try:
        to_fp8(module)
        return True
    except Exception:
        return False


def dit_bytes(model):
    """Bytes needed to hold every parameter of ``model``."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _linear_weight_elements(model):
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))


def fp8_bytes(model):
    """Bytes needed if every ``nn.Linear`` weight were stored in fp8."""
    # Only the Linear weights shrink (assume the current dtype is 2 bytes).
    return dit_bytes(model) - _linear_weight_elements(model)


def free_vram():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.mem_get_info()[0]


def _want_fp8(requested):
    """Whether fp8 should be used, given the caller's flag and the live mode."""
    if requested in (True, "on", "fp8"):
        return True
    if requested in (False, "off", "full"):
        return False
    # requested is "auto" (or unset): fall back to the user's precision policy.
    mode = _CONFIG.get("fp8_mode", "auto")
    if mode == "fp8":
        return True
    if mode == "full":
        return False
    return None  # "auto": decide from the actual fit


def _resident_block_count(model, reserve):
    """How many of ``model``'s streamed blocks fit on the GPU right now.

    ``model`` must expose ``blocks`` and ``set_resident_blocks`` (Krea 2 does);
    anything else keeps the old all-streamed behaviour.
    """
    if not _CONFIG.get("keep_resident", True):
        return 0
    blocks = getattr(model, "blocks", None)
    if blocks is None or not hasattr(model, "set_resident_blocks"):
        return 0
    budget = free_vram() - reserve
    used = 0
    count = 0
    for block in blocks:
        size = dit_bytes(block)
        if used + size > budget:
            break
        used += size
        count += 1
    return count


def place_dit(model, device, fp8="auto", reserve=None):
    """Place ``model`` resident on ``device`` if it fits, else block-stream it.

    ``fp8`` is ``"auto"`` (use fp8 whenever bf16 does not fit, halving both
    residency and, when it still does not fit, the bytes streamed across PCIe
    each step), ``"on"`` (always) or ``"off"``. Returns a short status string.

    A model already armed for disk streaming (it carries a ``_streaming_source``
    attribute, set by :mod:`engine.streaming`) is placed by that streamer and is
    left untouched here.
    """
    if getattr(model, "_streaming_source", None) is not None:
        return getattr(model, "_streaming_label", "disk-streamed")

    if reserve is None:
        reserve = _CONFIG.get("vram_reserve", DEFAULT_RESERVE)
    if not torch.cuda.is_available() or str(device) == "cpu":
        model.place_head(device)
        return "cpu"

    target = torch.device(device)
    fits = lambda size: size + reserve < free_vram()

    want_fp8 = _want_fp8(fp8)
    use_fp8 = False
    if want_fp8 is True:
        use_fp8 = try_to_fp8(model)
    elif want_fp8 is None and not fits(dit_bytes(model)):
        use_fp8 = try_to_fp8(model)

    if fits(dit_bytes(model)):
        model.to(target)
        return "fp8-resident" if use_fp8 else "resident"

    # Too large even in fp8: stream the blocks through the GPU, keeping as many
    # resident as fit so each step only moves the remainder across PCIe. A
    # streamed model keeps its resident blocks on the GPU through the VAE decode,
    # so use the wider streaming reserve to leave room for the decode.
    model.enable_block_offload("cpu", target)
    model.place_head(target)
    resident = _resident_block_count(model, _CONFIG.get("stream_reserve", STREAM_RESERVE))
    if resident:
        model.set_resident_blocks(resident, target)
    label = "fp8-streamed" if use_fp8 else "streamed"
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        label += f"({resident}/{len(blocks)})"
    return label


def place_text_encoder(encoder, target="cpu", dtype=None, reserve=None):
    """Move a text encoder to ``target`` (``"cpu"`` or ``"auto"``), returning where.

    ``"auto"`` keeps (or moves) it on the GPU only when the whole encoder plus
    the VRAM reserve still fits in the free VRAM *after* the DiT has been placed,
    so the common case in this app -- a text encoder that runs once and is much
    larger than the card (T5-XXL is ~22 GB in bf16) -- stays on CPU. The move is
    a no-op without CUDA, and any failure falls back to CPU rather than taking
    the job down.
    """
    if dtype is None:
        dtype = next(encoder.parameters()).dtype if any(True for _ in encoder.parameters()) else torch.bfloat16
    if target == "auto" and torch.cuda.is_available() and str(next(encoder.parameters()).device) != "cuda":
        if reserve is None:
            reserve = _CONFIG.get("vram_reserve", DEFAULT_RESERVE)
        try:
            needed = dit_bytes(encoder)
            if needed + reserve < free_vram():
                return str(encoder.to(device="cuda", dtype=dtype).eval())
        except Exception:
            pass
    device = next(encoder.parameters()).device
    if str(device) != "cpu":
        encoder.to(device="cpu")
    return "cpu"


# ---------------------------------------------------------------------------
# Tiled VAE decode
# ---------------------------------------------------------------------------

def _ramp(length, overlap, at_start, at_end):
    """A 1-D weight ramp of ``length`` samples feathering the ``overlap`` ends."""
    weight = torch.ones(length)
    if not at_start and overlap > 0:
        weight[:overlap] = torch.linspace(0.0, 1.0, overlap)
    if not at_end and overlap > 0:
        weight[-overlap:] = torch.linspace(1.0, 0.0, overlap)
    return weight


def decode_tiled(decode_fn, latent, tile_size=512, overlap=64, scale=None):
    """Decode ``latent`` in overlapping spatial tiles, feather-blending the seams.

    ``decode_fn`` takes the whole latent (a ``(B, C, h, w)`` tensor) and returns
    the image latent at some spatial scale. The full-latent path is used when the
    latent already fits in one tile or tiling is disabled, so this is a pure
    memory/time trade: the peak activation of a conv VAE decode is quadratic in
    the spatial side, so decoding 2x-smaller tiles roughly quarters it.

    ``scale`` is the output:latent side ratio; when ``None`` it is read from the
    first (full-width or full-height) tile. The overlap, in *latent* pixels, is
    blended linearly in the output, which hides the small boundary differences a
    padded conv produces.
    """
    if latent.ndim != 4:
        return decode_fn(latent)
    batch, _channels, height, width = latent.shape
    if tile_size <= 0 or overlap < 0 or (height <= tile_size and width <= tile_size):
        return decode_fn(latent)

    overlap = min(overlap, max(1, tile_size // 2 - 1))
    step = max(1, tile_size - overlap)

    def starts(total):
        values = list(range(0, max(1, total - tile_size) + 1, step))
        if not values or values[-1] != max(0, total - tile_size):
            values.append(max(0, total - tile_size))
        return sorted(set(values))

    accumulator = None
    weights = None
    for y0 in starts(height):
        y1 = min(height, y0 + tile_size)
        for x0 in starts(width):
            x1 = min(width, x0 + tile_size)
            decoded = decode_fn(latent[:, :, y0:y1, x0:x1]).float()
            if accumulator is None:
                if scale is None:
                    scale = max(1, round(decoded.shape[-2] / (y1 - y0)))
                full_h, full_w = height * scale, width * scale
                accumulator = torch.zeros(batch, decoded.shape[1], full_h, full_w,
                                          dtype=torch.float32, device=decoded.device)
                weights = torch.zeros(1, 1, full_h, full_w, dtype=torch.float32,
                                      device=decoded.device)
            oy0, ox0 = y0 * scale, x0 * scale
            oy1, ox1 = oy0 + decoded.shape[-2], ox0 + decoded.shape[-1]
            wy = _ramp(decoded.shape[-2], overlap * scale, y0 == 0,
                       y1 == height).to(decoded.device)
            wx = _ramp(decoded.shape[-1], overlap * scale, x0 == 0,
                       x1 == width).to(decoded.device)
            w2d = (wy[:, None] * wx[None, :])[None, None]
            accumulator[:, :, oy0:oy1, ox0:ox1] += decoded * w2d
            weights[:, :, oy0:oy1, ox0:ox1] += w2d
    weights = weights.clamp_min(1e-6)
    return (accumulator / weights).to(dtype=latent.dtype)


def compile_vae_decoder(vae, sample_latent):
    """Swap ``vae.decoder`` for a ``torch.compile``d version when it is safe to.

    The custom Wan-style VAEs are conv-bound and decode is a one-shot stage, so a
    compiled decoder is a straight latency win once warmed up. Compilation is
    skipped when the user disabled it (``compile_vae`` in
    :mod:`engine.tuning`), when ``PAI_COMPILE_VAE=0`` is set, without CUDA, or
    when the runtime rejects ``torch.compile`` (e.g. an unsupported interpreter).
    The warmup decode both builds the graph and checks it against the eager
    result, so a graph that disagrees (or a compile that raises) reverts to eager
    rather than shipping a wrong image.
    """
    if os.environ.get("PAI_COMPILE_VAE", "1") == "0":
        return vae
    if not _CONFIG.get("compile_vae", True) or not torch.cuda.is_available():
        return vae
    original = vae.decoder
    try:
        with torch.inference_mode():
            eager = vae.decode(sample_latent).float()
        vae.decoder = torch.compile(original, dynamic=True)
        with torch.inference_mode():
            got = vae.decode(sample_latent).float()
        if not torch.allclose(eager, got, atol=1e-2, rtol=1e-2):
            raise RuntimeError("torch.compile changed the VAE decode result")
    except Exception:
        vae.decoder = original
    return vae