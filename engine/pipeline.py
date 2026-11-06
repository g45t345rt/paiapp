"""End-to-end generation pipeline (no ComfyUI).

Two model families are supported behind one ``generate(prompt, config, image,
progress)`` entry point:

* ``anima`` -- the Cosmos-Predict2 DiT + Qwen3/T5 LLM adapter + Qwen-Image VAE
  (see :mod:`engine.anima`, :mod:`engine.text_encoder`); flow-matching sampler.
* ``stable_diffusion`` -- a single-file ``Load Checkpoint`` model (SD 1.x / SDXL
  and their finetunes): diffusers UNet + CLIP + VAE (see
  :mod:`engine.checkpoint`); epsilon-prediction sampler.
* ``qwen_image`` -- Qwen-Image 2.1: a GGUF MMDiT (:mod:`engine.qwen_image`),
  the Qwen3-VL-8B text encoder (:mod:`engine.qwen_text`) and the Wan-2.2-style
  2.1 VAE (:mod:`engine.qwen_vae21`); flux flow sampler with a token-count
  derived shift.
* ``krea2`` -- Krea 2: a single-stream MMDiT (:mod:`engine.krea2`), the
  Qwen3-VL-4B 12-layer hidden-state stack (:mod:`engine.krea2_text`) and the
  Qwen-Image VAE; flux flow sampler (fixed shift for the turbo checkpoint).
* ``moss_image`` -- MOSS Image: a 5B single-stream DiT (:mod:`engine.moss_image`),
  the Qwen3.5-9B text encoder (:mod:`engine.moss_text`) and the FLUX.2 VAE
  (:mod:`engine.moss_vae`); flow-matching Euler at shift 5.

The family is selected with ``config["family"]`` (default ``anima``).
"""

import functools
import gc
import os
import threading

import numpy as np
import torch
from PIL import Image

from . import paths
from . import precision
from . import preprocess
from . import preview as preview_mod
from . import tuning
from . import lllite
from . import regions
from .cancel import Cancelled
from .anima import load_anima, preprocess_text_embeds
from .checkpoint import (
    ClassicCheckpoint,
    clear_region_processors,
    install_region_processors,
)
from .flux import load_flux as load_flux_dit
from .flux_text import CLIPLEncoder, T5Encoder
from .flux_vae import load_vae as load_flux_vae
from .fibo import load_pipeline as load_fibo_pipeline
from .krea2 import load_transformer as load_krea2_dit
from .krea2_text import load_text_encoder as load_krea2_text
from .krea2_vl import load_vl_encoder
from . import krea2_anypaint as anypaint
from .latent import (
    ClassicLatent,
    FluxLatent,
    MossImageLatent,
    QwenImage21Latent,
    Wan21,
)
from .lora import apply_lora, is_dedistill, usage
from .loaded_models import record
from .h3 import load_dit as load_h3_dit
from .h3_layout import build_still_packed_sequence as h3_still_layout
from .h3_sampler import (
    AUDIO_SHIFT as H3_AUDIO_SHIFT,
    VIDEO_SHIFT as H3_VIDEO_SHIFT,
    MiniMaxH3Scheduler,
)
from .h3_text import load_text_encoder as load_h3_text
from .h3_vae import load_vae as load_h3_vae
from .moss_image import load_moss_image
from .moss_text import MossTextEncoder
from .moss_vae import (
    load_vae as load_moss_vae,
    pack_latents as moss_pack_latents,
    unpack_latents as moss_unpack_latents,
)
from .precision import dit_bytes, free_vram, place_dit
from .streaming import can_stream as _can_stream
from .qwen_image import QwenRef, load_transformer
from .qwen_text import load_text_encoder
from .qwen_vae21 import load_vae as load_qwen_vae
from .sampler import (
    AnimaApplyModel,
    AnimaRefApplyModel,
    FluxApplyModel,
    Krea2ApplyModel,
    ModelSamplingDiscrete,
    ModelSamplingFlow,
    ModelSamplingFlux,
    MossImageApplyModel,
    QwenConditioning,
    QwenImageApplyModel,
    UNetApplyModel,
    ZImageApplyModel,
    anima_ref_guider,
    calculate_shift,
    linear_sigmas,
    scheduler_sigmas,
    sample,
)
from .anima_ref import AnimaRefConditioning, RefAdapter
from .text_encoder import TextEncoder
from .vae import load_vae
from .z_image import load_z_image
from .z_image_text import ZImageTextEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIT_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
UNET_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
# diffusers ``AutoencoderKL`` VAEs (Flux / Z-Image): bf16 on CUDA is
# ~2x faster than fp32 (benchmarked 700ms -> 377ms at 1024px) and uses ~half the
# activation memory. bf16, not fp16, because its fp32 exponent range avoids the
# fp16 NaN/black-image failure mode. fp32 on CPU, where bf16 is emulated.
VAE_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# Wan-style 3D causal VAEs (Qwen-Image, Krea 2, Anima): bf16 on CUDA. fp32 is
# only neutral at small sizes (512px: fp32 203ms vs bf16 211ms) and ~2x slower at
# 1024px (fp32 1753ms vs bf16 922ms), where the doubled activation bytes push the
# convs off cuDNN's fast paths. bf16, not fp16, for its fp32 exponent range, and
# it matches ComfyUI's default for these VAEs. fp32 on CPU, where bf16 is emulated.
WAN_VAE_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# SD 1.x / SDXL classic VAE: fp32 even on CUDA. Its fp16 decode is the well-known
# source of black/NaN images (diffusers' ``force_upcast`` upcasts for exactly this).
CLASSIC_VAE_DTYPE = torch.float32

_lock = threading.Lock()
_base = None
_base_key = None
_base_error = None

# The Anima reference-adapter LoRAs installed on the cached DiT, keyed by
# (id(dit), path, strength). The hooks are gated (see :mod:`engine.anima_ref`),
# so the LoRA only fires while a reference frame is attached.
_ref_adapter_cache = {}

# The Anima ControlNet-LLLite adapters currently patched into the cached DiT, in
# apply order, plus a cache of built adapters keyed by (id(dit), path) so
# switching adapters does not reload weights. An adapter patches the DiT's
# linears in place, so several may be stacked (each wraps the previous one) and
# they must be detached in reverse order before their DiT is dropped.
_lllite_stack = []
_lllite_dit = None
_lllite_cache = {}

_classic = None
_classic_key = None
_classic_error = None

_qwen = None
_qwen_key = None
_qwen_error = None

_flux = None
_flux_key = None
_flux_error = None

_z = None
_z_key = None
_z_error = None

# The Z-Image ControlNet currently attached to the cached DiT, and the (dit, path)
# it belongs to. The control branch lives inside the DiT, so a new DiT invalidates
# it and it is rebuilt on the next generation.
_z_control = None
_z_control_key = None

_krea2 = None
_krea2_key = None
_krea2_error = None
_krea2_trace = None

_moss = None
_moss_key = None
_moss_error = None
_h3 = None
_h3_key = None
_h3_error = None

_fibo = None
_fibo_key = None
_fibo_error = None

# Cache slot name -> (engine, cache-key, error) globals; lets the UI unload one
# family without dropping the others.
_SLOTS = {
    "base": ("_base", "_base_key", "_base_error"),
    "classic": ("_classic", "_classic_key", "_classic_error"),
    "qwen": ("_qwen", "_qwen_key", "_qwen_error"),
    "flux": ("_flux", "_flux_key", "_flux_error"),
    "z": ("_z", "_z_key", "_z_error"),
    "krea2": ("_krea2", "_krea2_key", "_krea2_error"),
    "moss": ("_moss", "_moss_key", "_moss_error"),
    "h3": ("_h3", "_h3_key", "_h3_error"),
    "fibo": ("_fibo", "_fibo_key", "_fibo_error"),
}


def _report(progress, frac, label, sub=None):
    """Report ``frac`` overall progress with an optional within-step ``sub``.

    ``frac`` is the job's overall fraction and drives the overall meter; ``sub``
    is the current sub-process's own fraction (sampling step, loader tick) and
    drives the sub-process meter. ``sub`` is ``None`` for a sub-process that
    cannot report fine-grained progress.
    """
    if progress is not None:
        try:
            progress(
                min(max(frac, 0.0), 1.0),
                label,
                None if sub is None else min(max(sub, 0.0), 1.0),
            )
        except Exception:
            pass


def _check(cancel):
    """Raise ``Cancelled`` when the job's token has been tripped."""
    if cancel is not None:
        cancel.throw_if_cancelled()


def _stage(cancel, progress, frac, label, sub=None):
    """Check cancellation, then report a load stage's progress.

    Every model loader reports its phases through here instead of ``_report``
    directly, so a cancel requested while a model loads is honoured at the next
    phase boundary (or loader tick) rather than only after the whole load
    finishes. ``progress``/``frac``/``label``/``sub`` are passed through to
    :func:`_report`.
    """
    _check(cancel)
    _report(progress, frac, label, sub)


def _make_preview(cfg, preview, latent_format=None, key=None, out_size=None):
    """Build a per-step preview emitter for a family's sampler callback.

    ``preview`` is the UI callback (called with a PIL image) or ``None``; the
    decode mode comes from ``cfg["live_preview"]`` (``off`` / ``latent2rgb`` /
    ``taesd``). Returns ``emit(i, denoised, total)`` or ``None`` when preview is
    off. ``taesd`` (the expensive path) is throttled to a handful of frames
    while ``latent2rgb`` runs every step, and every error is swallowed so a
    preview can never break a run.
    """
    if preview is None:
        return None
    previewer = preview_mod.build(
        latent_format, key=key, mode=cfg.get("live_preview") or "off",
    )
    if previewer is None:
        return None
    every_step = previewer.mode == "latent2rgb"

    def emit(i, denoised, total):
        if not every_step:
            step = max(1, total // 8)
            if i % step != 0 and i + 1 != total:
                return
        try:
            image = previewer.decode(denoised, out_size=out_size)
        except Exception:
            image = None
        if image is not None:
            try:
                preview(image)
            except Exception:
                pass

    return emit


def _cfg_loras(cfg):
    """Resolve ``cfg``'s LoRA selection to ``[(path, strength), ...]``.

    A model has no LoRAs unless ``config`` lists them, so a missing ``loras``
    key (or an empty list) yields none. Each entry is a ``{"path", "strength"}``
    dict, or a bare path string that takes strength 1.0.
    """
    entries = cfg.get("loras")
    if not isinstance(entries, (list, tuple)):
        return []
    items = []
    for item in entries:
        if isinstance(item, dict):
            path = item.get("path")
            if path:
                items.append((os.fspath(path), float(item.get("strength", 1.0) or 1.0)))
        elif isinstance(item, str) and item:
            items.append((item, 1.0))
    return items


def _fraction(value, default):
    """Clamp a stored start/end fraction into ``[0, 1]``."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _as_image_list(control, count):
    """Normalize ``control`` to ``count`` hint images (or ``None`` placeholders)."""
    if control is None:
        return [None] * count
    if isinstance(control, (list, tuple)):
        images = list(control)
        if len(images) < count:
            images += [None] * (count - len(images))
        return images[:count]
    return [control] * count


def _cfg_controlnets(cfg, kind):
    """Resolve ``cfg``'s control adapters of ``kind`` to a list of specs.

    Each spec is ``(path, strength, preprocessor, start, end)``. Control adapters
    are per-family: ``"controlnet"`` is the classic SD/SDXL and Z-Image
    ControlNet, ``"lllite"`` the Anima ControlNet-LLLite. Entries are taken from
    ``cfg["controlnets"]`` (the panel's list), falling back to the legacy single
    ``controlnet``/``lllite`` keys so older projects still load.
    """
    specs = []
    entries = cfg.get("controlnets")
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            item_kind = item.get("kind")
            if item_kind is not None and item_kind != kind:
                continue
            path = item.get("model")
            if not path:
                continue
            specs.append((
                os.fspath(path),
                float(item.get("strength", 1.0) or 1.0),
                item.get("preprocessor") or "none",
                _fraction(item.get("start"), 0.0),
                _fraction(item.get("end"), 1.0),
            ))
    if specs:
        return specs
    legacy_key = "lllite" if kind == "lllite" else "controlnet"
    legacy = cfg.get(legacy_key)
    if legacy:
        specs.append((
            os.fspath(legacy),
            float(cfg.get("control_strength", 1.0) or 1.0),
            cfg.get("control_preprocessor") or "none",
            0.0,
            1.0,
        ))
    return specs


def _attach_controlnets(ckpt, specs, progress=None, cancel=None):
    """Load the classic ControlNets a checkpoint is used with.

    ``specs`` is a list of ``(path, strength, preprocessor, start, end)``. Every
    control model must match the checkpoint's architecture; a mismatch is reported
    rather than loaded into a broken network. Cached models are reused, so
    switching back to a ControlNet does not reload it.
    """
    models = []
    if specs:
        from . import controlnet as controlnet_module
    for path, strength, preprocessor, start, end in specs:
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"ControlNet model not found: {path}")
        arch = controlnet_module.detect_arch(path)
        if arch != ckpt.family:
            raise RuntimeError(
                f"ControlNet {os.path.basename(path)} is {arch.upper()} but the "
                f"checkpoint is {ckpt.family.upper()}."
            )
        _stage(cancel, progress, 0.85, "loading controlnet")
        model, _ = controlnet_module.load(
            path, arch=arch, device=DEVICE, dtype=UNET_DTYPE,
        )
        models.append((model, strength, preprocessor, start, end))
    ckpt.controlnets = models


def _control_tensor(image):
    """A prepared control hint as a ``[1, 3, H, W]`` pixel tensor in ``[0, 1]``."""
    arr = np.asarray(image.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr)[None].permute(0, 3, 1, 2).to(DEVICE, UNET_DTYPE)


def _lllite_cond_tensor(image, mask=None, masked_input=False):
    """A prepared control hint as a pixel tensor in ``[-1, 1]``.

    A plain adapter wants ``[1, 3, H, W]``. An inpainting adapter (whose
    ``cond_in_channels`` is 4) additionally wants the inpaint mask as a fourth
    channel, scaled from ``{0, 1}`` to ``[-1, 1]`` to match the RGB range; with
    ``masked_input`` the hint's RGB is zeroed where the mask is kept. This mirrors
    kohya's LLLite inpainting training and ``anima_minimal_inference_control_net_lllite``.
    """
    arr = np.asarray(image.convert("RGB")).astype("float32") / 127.5 - 1.0
    if mask is not None:
        grey = np.asarray(mask.convert("L")).astype("float32") / 255.0
        binary = (grey >= 0.5).astype("float32")
        if masked_input:
            arr = arr * (binary < 0.5).astype("float32")[..., None]
        arr = np.concatenate([arr, (binary * 2.0 - 1.0)[..., None]], axis=-1)
    return torch.from_numpy(arr)[None].permute(0, 3, 1, 2).to(DEVICE, DIT_DTYPE)


def _reset_zcontrol():
    """Forget the Z-Image ControlNet attached to a DiT that is being dropped."""
    global _z_control, _z_control_key
    _z_control = None
    _z_control_key = None


def _attach_zcontrol(dit, path, progress=None, cancel=None):
    """Load (or reuse) the Z-Image ControlNet attached to ``dit``.

    The control branch lives inside the DiT (so the DiT's own block offload covers
    it) and loading is cached per ``(dit, path)``, so re-selecting the same control
    model does not re-read its weights. Returns the load report, or ``None`` when
    no control model is set.
    """
    global _z_control, _z_control_key
    if not path or not os.path.isfile(path):
        _reset_zcontrol()
        return None
    key = (id(dit), os.path.abspath(path))
    if _z_control_key == key and _z_control is not None:
        return _z_control
    from . import z_controlnet

    _stage(cancel, progress, 0.9, "loading controlnet")
    _z_control = z_controlnet.load(dit, path, device=DEVICE, dtype=DIT_DTYPE)
    _z_control_key = key
    return _z_control


def _detach_lllite_stack():
    """Detach every applied LLLite adapter (reverse order) and drop the cache."""
    global _lllite_stack, _lllite_dit
    for adapter in reversed(_lllite_stack):
        adapter.remove_from()
    _lllite_stack = []
    _lllite_dit = None
    _lllite_cache.clear()


def _reset_lllite():
    """Detach the applied adapters and drop the build cache (see ``_load_base``)."""
    _detach_lllite_stack()


def _apply_lllite_stack(dit, paths, progress=None, cancel=None):
    """Patch the listed Anima LLLite adapters into ``dit``, in order.

    Each adapter rewrites the DiT's linear ``forward``s, so they are stacked: the
    second wraps the first, and so on, letting one run combine several adapters
    (a 3D-pose adapter plus a regional adapter, for example). They are always
    re-applied from scratch in order, and building is cached per ``(dit, path)``
    so re-selecting an adapter does not reload weights. Returns the applied
    adapters in the same order as ``paths``.
    """
    global _lllite_stack, _lllite_dit
    if _lllite_dit is not dit:
        _detach_lllite_stack()
    else:
        for adapter in reversed(_lllite_stack):
            adapter.remove_from()
        _lllite_stack = []
    _lllite_dit = dit
    adapters = []
    for path in paths:
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"Control adapter not found: {path}")
        key = (id(dit), os.path.abspath(path))
        adapter = _lllite_cache.get(key)
        if adapter is None:
            _stage(cancel, progress, 0.88, "loading control adapter")
            adapter, _ = lllite.load(dit, path, device=DEVICE, dtype=DIT_DTYPE)
            _lllite_cache[key] = adapter
        adapter.set_active(True)
        adapter.apply_to()
        _lllite_stack.append(adapter)
        adapters.append(adapter)
    return adapters


def _apply_loras(model, loras, progress=None, frac=0.9, cancel=None):
    """Apply each ``(path, strength)`` LoRA to ``model``. Returns modules patched."""
    if not loras:
        return 0
    from safetensors.torch import load_file

    applied = 0
    total = max(1, len(loras))
    for index, (path, strength) in enumerate(loras):
        if not (path and os.path.isfile(path)):
            continue
        _stage(cancel, progress, frac, "applying lora", (index + 1) / total)
        applied += apply_lora(
            model, load_file(path, device="cpu"), strength=strength, source=path,
        )
    return applied


def _loras(selection, *models):
    """LoRA rows for a family's cache: one per selected LoRA (see ``loaded_models``).

    ``selection`` is the ``[(path, strength), ...]`` the engine was built with
    (the last element of its cache key). Each row reports the strength declared
    in the selection and, from a model's :func:`engine.lora.usage`, the bytes its
    delta tensors hold and how many layers it patched -- totals merged when a
    LoRA was applied to more than one model (the Anima DiT and its adapter).
    """
    found = {}
    for model in models:
        for item in usage(model):
            merged = found.get(item["source"])
            if merged is None:
                found[item["source"]] = dict(item)
            else:
                merged["bytes"] += item["bytes"]
                merged["layers"] += item["layers"]
    rows = []
    seen = set()
    for path, strength in selection:
        seen.add(path)
        applied = found.get(path)
        rows.append({
            "source": path,
            "strength": strength,
            "bytes": applied["bytes"] if applied else 0,
            "layers": applied["layers"] if applied else 0,
        })
    for source, item in found.items():
        if source not in seen:
            rows.append(item)
    return rows


def _place_text_encoders(*encoders):
    """Honour the ``text_encoder_device`` setting for encoder wrappers.

    Only acts when the setting is ``"auto"``; each wrapper's ``.model`` (and its
    ``.device``, which its ``encode`` uses to place inputs) is moved to the GPU
    only when it fits beside the resident DiT. Failures are swallowed: the
    encoder stays on CPU and the run proceeds, exactly as the default does.
    """
    if precision.config().get("text_encoder_device", "cpu") != "auto":
        return
    for encoder in encoders:
        model = getattr(encoder, "model", None)
        if not isinstance(model, torch.nn.Module):
            continue
        try:
            where = precision.place_text_encoder(model, target="auto")
            if where.startswith("cuda"):
                encoder.device = "cuda"
        except Exception:
            pass


def _load_base(path, clip_path=None, vae_path=None, loras=(), progress=None,
               cancel=None):
    global _base, _base_key, _base_error
    # The cache key includes the components so a model that shares a DiT but
    # uses a different text encoder/VAE/LoRA is not served a stale engine.
    loras = list(loras)
    key = (path, clip_path, vae_path, tuple(loras))
    if _base is not None and _base_key == key:
        return _base
    _base = None
    _base_error = None
    _base_key = key
    # A reload replaces the DiT object, so any adapter patched into the old one
    # (and the adapters cached against it) must go.
    _reset_lllite()
    _ref_adapter_cache.clear()
    try:
        _stage(cancel, progress, 0.02, "loading engine")
        dit, adapter, _ = load_anima(path)
        dit = dit.to(DEVICE, DIT_DTYPE).eval()
        adapter = adapter.to(DEVICE, DIT_DTYPE).eval()
        for p in list(dit.parameters()) + list(adapter.parameters()):
            p.requires_grad_(False)

        _stage(cancel, progress, 0.45, "loading text encoder")
        clip = TextEncoder(
            clip_path or paths.DEFAULTS["clip"], device=DEVICE, dtype=DIT_DTYPE,
            tick=lambda f: _stage(cancel, progress, 0.45 + 0.23 * f, "loading text encoder", f),
        )

        _stage(cancel, progress, 0.7, "loading vae")
        vae, _, _ = load_vae(vae_path or paths.DEFAULTS["vae"], device=DEVICE, dtype=WAN_VAE_DTYPE)

        if loras:
            from safetensors.torch import load_file

            # Anima LoRAs may carry LLM-adapter weights under this prefix; the
            # rest target the DiT.
            adapter_prefix = "diffusion_model.llm_adapter."
            for lora_path, strength in loras:
                if not (lora_path and os.path.isfile(lora_path)):
                    continue
                _stage(cancel, progress, 0.85, "applying lora")
                lora_sd = load_file(lora_path, device="cpu")
                adapter_lora = {
                    k[len(adapter_prefix):]: v
                    for k, v in lora_sd.items() if k.startswith(adapter_prefix)
                }
                dit_lora = {
                    k: v for k, v in lora_sd.items() if not k.startswith(adapter_prefix)
                }
                apply_lora(dit, dit_lora, strength=strength, source=lora_path)
                if adapter_lora:
                    apply_lora(adapter, adapter_lora, strength=strength, source=lora_path)

        _base = {
            "dit": dit,
            "adapter": adapter,
            "clip": clip,
            "vae": vae,
            "ms": ModelSamplingFlow(),
            "latent": Wan21(),
        }
    except Cancelled:
        _base_key = None
        raise
    except Exception as exc:  # surfaced to the UI
        _base_error = exc
    return _base


def _load_classic(path, loras=(), progress=None, controlnet_specs=(), cancel=None):
    global _classic, _classic_key, _classic_error
    loras = list(loras)
    controlnet_specs = list(controlnet_specs)
    key = (path, tuple(loras))
    if _classic is not None and _classic_key == key:
        # The ControlNets are not part of the checkpoint key (they are cached on
        # their own), so they are refreshed on every call: switching control
        # models is cheap and never reloads the checkpoint.
        _classic_error = None
        try:
            _attach_controlnets(_classic, controlnet_specs, progress, cancel)
        except Cancelled:
            raise
        except Exception as exc:
            _classic_error = exc
        return _classic
    _classic = None
    _classic_error = None
    _classic_key = key
    try:
        _stage(cancel, progress, 0.05, "loading checkpoint")
        ckpt = ClassicCheckpoint(path, device="cpu", unet_dtype=UNET_DTYPE, vae_dtype=CLASSIC_VAE_DTYPE)
        _stage(cancel, progress, 0.7, "moving to device")
        ckpt.unet = ckpt.unet.to(DEVICE).eval()
        ckpt.vae = ckpt.vae.to(DEVICE).eval()
        ckpt.text_encoders = [e.to(DEVICE).eval() for e in ckpt.text_encoders]
        ckpt.device = DEVICE
        _apply_loras(ckpt.unet, loras, progress, 0.8, cancel)
        _attach_controlnets(ckpt, controlnet_specs, progress, cancel)
        _stage(cancel, progress, 0.9, "ready")
        _classic = ckpt
    except Cancelled:
        _classic_key = None
        raise
    except Exception as exc:
        _classic_error = exc
    return _classic


def _load_qwen(path, text_path=None, vae_path=None, loras=(), progress=None,
               cancel=None, fp8="auto"):
    global _qwen, _qwen_key, _qwen_error
    loras = list(loras)
    key = (path, text_path, vae_path, tuple(loras), fp8)
    if _qwen is not None and _qwen_key == key:
        return _qwen
    _qwen = None
    _qwen_error = None
    _qwen_key = key
    try:
        _stage(cancel, progress, 0.05, "loading text encoder")
        te = load_text_encoder(
            text_path or paths.QWEN_DEFAULTS["text"], device="cpu", dtype=DIT_DTYPE,
            tick=lambda f: _stage(cancel, progress, 0.05 + 0.35 * f, "loading text encoder", f),
        )
        if getattr(te, "missing", None):
            raise RuntimeError(
                f"text encoder weights missing ({len(te.missing)}): {te.missing[:8]}"
            )

        _stage(cancel, progress, 0.45, "loading transformer")
        tune = tuning.read()
        streaming_on = tune["disk_stream"] and _can_stream(path, loras)
        dit, missing, _ = load_transformer(
            path, device="cpu", dtype=DIT_DTYPE, stream=streaming_on,
            stream_device=DEVICE, ram_budget=int(tune["ram_budget_gb"] * 1e9),
            prefetch_depth=tune["prefetch_depth"], pin_blocks=tune["keep_resident"],
        )
        if missing:
            raise RuntimeError(f"transformer weights missing: {missing[:4]}")
        # Place through the shared policy: keep the DiT resident when it fits,
        # else stream its blocks with as many held on the GPU as fit. The old
        # path unconditionally block-offloaded, so every step (and CFG doubles
        # them) dragged the whole model across PCIe while the GPU sat nearly idle.
        # A disk-streamed (meta) model is already placed by its streamer and is
        # left untouched by place_dit.
        if not streaming_on:
            place_dit(dit, DEVICE, fp8=fp8)
        _place_text_encoders(te)
        _apply_loras(dit, loras, progress, 0.7, cancel)

        _stage(cancel, progress, 0.85, "loading vae")
        vae, _, _ = load_qwen_vae(vae_path or paths.QWEN_DEFAULTS["vae"], device=DEVICE, dtype=WAN_VAE_DTYPE)

        _qwen = {"dit": dit, "text": te, "vae": vae}
    except Cancelled:
        _qwen_key = None
        raise
    except Exception as exc:
        _qwen_error = exc
    return _qwen


def _load_flux(path, clip_path=None, t5_path=None, vae_path=None, loras=(), progress=None, fp8="auto", cancel=None):
    global _flux, _flux_key, _flux_error
    loras = list(loras)
    key = (path, clip_path, t5_path, vae_path, fp8, tuple(loras))
    if _flux is not None and _flux_key == key:
        return _flux
    _flux = None
    _flux_error = None
    _flux_key = key
    try:
        # The encoders run on CPU (T5-XXL is ~22 GB in bf16, far past the GPU),
        # matching the Qwen path; only the DiT blocks stream to the GPU.
        _stage(cancel, progress, 0.05, "loading text encoders")
        clip = CLIPLEncoder(
            clip_path or paths.FLUX_DEFAULTS["clip"], device="cpu", dtype=DIT_DTYPE,
            tick=lambda f: _stage(cancel, progress, 0.05 + 0.07 * f, "loading text encoders", f),
        )
        t5 = T5Encoder(
            t5_path or paths.FLUX_DEFAULTS["t5"], device="cpu", dtype=DIT_DTYPE,
            tick=lambda f: _stage(cancel, progress, 0.12 + 0.23 * f, "loading text encoders", f),
        )

        _stage(cancel, progress, 0.35, "loading transformer")
        tune = tuning.read()
        streaming_on = tune["disk_stream"] and _can_stream(path, loras)
        dit, missing, _ = load_flux_dit(
            path, device="cpu", dtype=DIT_DTYPE, stream=streaming_on,
            stream_device=DEVICE, ram_budget=int(tune["ram_budget_gb"] * 1e9),
            prefetch_depth=tune["prefetch_depth"], pin_blocks=tune["keep_resident"],
        )
        if missing:
            raise RuntimeError(f"transformer weights missing: {missing[:4]}")
        place_dit(dit, DEVICE, fp8=fp8)
        _place_text_encoders(clip, t5)
        _apply_loras(dit, loras, progress, 0.7, cancel)

        _stage(cancel, progress, 0.85, "loading vae")
        vae, _, _ = load_flux_vae(vae_path or paths.FLUX_DEFAULTS["vae"], device=DEVICE, dtype=VAE_DTYPE)

        _flux = {"dit": dit, "clip": clip, "t5": t5, "vae": vae}
    except Cancelled:
        _flux_key = None
        raise
    except Exception as exc:
        _flux_error = exc
    return _flux


def _flux_pil_to_latent(vae, image, lh, lw):
    img = image.convert("RGB").resize((lw * 8, lh * 8), Image.LANCZOS)
    arr = np.asarray(img).astype("float32") / 255.0
    px = torch.from_numpy(arr)[None].permute(0, 3, 1, 2).to(DEVICE, VAE_DTYPE) * 2.0 - 1.0
    with torch.no_grad():
        return vae.encode(px).latent_dist.mean.float()


def _vae_decode(vae, latent, sample=False):
    """Decode ``latent``, tiling it when the ``tiled_decode`` setting is on.

    ``sample=True`` unwraps the diffusers ``.sample`` (``AutoencoderKL``);
    otherwise the raw tensor is returned (the Wan-style VAEs). Tiling splits the
    decode into overlapping spatial chunks, trading a little time for a much
    smaller peak on a big canvas (a conv decode is quadratic in the spatial
    side); it is a no-op for non-4D (packed/video) latents.
    """

    def run(tile):
        result = vae.decode(tile)
        return result.sample if sample else result

    cfg = precision.config()
    if cfg.get("tiled_decode") and latent.ndim == 4:
        try:
            return precision.decode_tiled(
                run, latent, tile_size=int(cfg.get("tile_size", 512)),
                overlap=int(cfg.get("tile_overlap", 64)),
            )
        except Exception:
            pass
    return run(latent)


def _flux_latent_to_pil(vae, latent, height, width):
    with torch.inference_mode():
        out = _vae_decode(vae, latent.to(DEVICE, VAE_DTYPE), sample=True)
    out = torch.clamp((out.float() + 1.0) / 2.0, 0.0, 1.0)[0]
    out = out[:, :height, :width]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, "RGB")


def _region_blocks(prompt, region_entries, encode):
    """Encode the run prompt then each region's prompt into text blocks.

    Returns ``(blocks, lengths)``: the shared base block first, then one per
    region, each ``encode``'s output tensor, with their token lengths.
    """
    blocks = [encode(prompt or "")]
    for region in region_entries:
        blocks.append(encode(region.get("prompt") or ""))
    return blocks, [int(block.shape[1]) for block in blocks]


def _region_joint_mask(region_entries, grid_w, grid_h, lengths, image_len, device,
                       text_len=None, text_keep=None):
    """The joint-attention mask for a regional run (see :mod:`engine.regions`)."""
    table = regions.token_regions(region_entries, grid_w, grid_h)
    return regions.joint_attn_mask(
        table, lengths, image_len, device, text_len=text_len, text_keep=text_keep,
        biases=regions.region_biases(region_entries),
    )


# --------------------------------------------------------------- hires fix
# A second "hires fix" pass: the finished latent is enlarged by ``upscale_by``
# with one of the ``paths.UPSCALE_TYPES`` methods and re-denoised for a few
# steps at ``upscale_denoise`` strength, then decoded at the larger size. The
# latent is enlarged in the sampler's normalized space (what ``sample`` works
# in), so a family's mean/std or shift/scale normalization is never rescaled.
#
# The pass runs over the plain (non-regional) conditioning and drops any
# ControlNet / inpaint mask: those are built for the pass-1 token grid, so they
# cannot be reused at the larger size.
_LATENT_MODES = {
    "latent_nearest": "nearest",
    "latent_bilinear": "bilinear",
    "latent_bicubic": "bicubic",
    # ComfyUI's ``bislerp``; approximated here with bicubic resampling.
    "latent_bislerp": "bicubic",
}


def _upscale_spec(cfg):
    """The hires-fix settings from ``cfg``: ``(method, factor, denoise, steps)``.

    Returns ``None`` when the second pass is off or its values are degenerate
    (a factor at or below 1, or a denoise at or below 0).
    """
    method = cfg.get("upscale_type") or "none"
    if method == "none" or method not in paths.UPSCALE_TYPES:
        return None
    factor = cfg.get("upscale_by")
    try:
        factor = 2.0 if factor is None else float(factor)
    except (TypeError, ValueError):
        return None
    if factor <= 1.0:
        return None
    denoise = cfg.get("upscale_denoise")
    try:
        denoise = 0.5 if denoise is None else float(denoise)
    except (TypeError, ValueError):
        return None
    denoise = min(max(denoise, 0.0), 0.9999)
    if denoise <= 0.0:
        return None
    steps = cfg.get("upscale_steps")
    steps = cfg.get("steps") if steps is None else steps
    steps = int(steps if steps is not None else 10)
    return method, factor, denoise, max(1, steps)


def _resize_latent(latent, method, target_h, target_w):
    """Enlarge a model-space ``latent`` to ``(target_h, target_w)`` spatial size.

    Handles both 4-D ``[B, C, H, W]`` and 5-D ``[B, C, T, H, W]`` latents (the
    Wan-style VAEs carry a singleton time axis).
    """
    mode = _LATENT_MODES[method]
    squeeze = latent.dim() == 5
    x = latent.squeeze(2) if squeeze else latent
    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    out = torch.nn.functional.interpolate(
        x, size=(target_h, target_w), mode=mode, **kwargs
    )
    return out.unsqueeze(2) if squeeze else out


def _hires_fix(cfg, samples, ms, model, ctx_pos, ctx_neg, sampler, seed,
               progress, cancel, decode, encode, make_ms=None,
               base=0.82, span=0.13, guider=None):
    """Run the hires-fix second pass over ``samples`` (a model-space latent).

    ``decode(latent)`` returns the PIL image for a model-space latent (applying
    the family's ``process_out``) and ``encode(pil, lh, lw)`` returns a
    model-space latent (applying ``process_in``), for the ``pixel`` method.
    ``make_ms(lh, lw)`` rebuilds the model sampling for the larger token grid
    when a family derives its shift from the token count (Flux / Qwen / Krea 2).
    Returns the enlarged model-space samples.
    """
    method, factor, denoise, steps = _upscale_spec(cfg)
    lh, lw = int(samples.shape[-2]), int(samples.shape[-1])
    th = max(1, round(lh * factor))
    tw = max(1, round(lw * factor))

    if method == "pixel":
        _report(progress, base, "upscaling", 0.0)
        pil = decode(samples)
        size = (max(1, round(pil.width * factor)), max(1, round(pil.height * factor)))
        up = encode(pil.resize(size, Image.LANCZOS), th, tw).to(samples.dtype)
    else:
        up = _resize_latent(samples, method, th, tw)
    _check(cancel)

    ms2 = make_ms(th, tw) if make_ms is not None else ms
    sigmas = scheduler_sigmas(
        ms2, steps, denoise, cfg.get("scheduler", "simple"), cfg
    ).to(up.device)
    generator = torch.manual_seed(int(seed))
    noise = torch.randn(up.shape, generator=generator, dtype=torch.float32, device="cpu")
    total = max(1, len(sigmas) - 1)

    def on_step(i, *_):
        _check(cancel)
        _report(progress, base + span * (i / total), "upscaling", i / total)

    kwargs = {}
    if guider is not None:
        kwargs["guider"] = guider
    out = sample(
        ms2, model, noise.to(up.device), up, sigmas, ctx_pos, ctx_neg,
        float(cfg["cfg"]), sampler=sampler,
        callback=on_step if (progress is not None or cancel is not None) else None,
        generator=generator, mask=None, feather=0.0, cancel=cancel, **kwargs,
    )
    return out.to(torch.float32)


def _generate_flux(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
                   regions_list=None, preview=None):
    """Flux txt2img / img2img (GGUF DiT + CLIP-L/T5-XXL + 16-ch VAE)."""
    base = _load_flux(
        cfg.get("dit") or paths.FLUX_DEFAULTS["dit"],
        cfg.get("clip"), cfg.get("t5"), cfg.get("vae"), _cfg_loras(cfg), progress,
        cfg.get("fp8", "auto"), cancel,
    )
    _check(cancel)
    if _flux_error is not None:
        raise RuntimeError(f"Flux load failed: {_flux_error}")
    if base is None:
        raise RuntimeError("Flux load failed.")

    dit, clip, t5, vae = base["dit"], base["clip"], base["t5"], base["vae"]
    width, height = int(cfg["width"]), int(cfg["height"])
    lh, lw = height // 8, width // 8
    latent_format = FluxLatent()
    # Flux derives the sampling shift from the image-token count (2x2 patched).
    shift = calculate_shift(
        (lh // 2) * (lw // 2), base_seq_len=256, max_seq_len=4096,
        base_shift=0.5, max_shift=1.15,
    )
    ms = ModelSamplingFlux(shift=shift)
    guidance = torch.tensor([float(cfg.get("guidance", 3.5))], dtype=torch.float32, device=DEVICE)

    region_entries = [r for r in (regions_list or []) if r.get("mask") is not None]
    with _lock:
        _report(progress, 0.0, "encoding prompt")
        pooled_pos = clip.encode(prompt or "").to(DEVICE, DIT_DTYPE)
        negative = cfg.get("negative") or ""
        ctx_pos_plain = (t5.encode(prompt or "").to(DEVICE, DIT_DTYPE), pooled_pos)
        if region_entries:
            blocks, lengths = _region_blocks(prompt, region_entries, t5.encode)
            ctx_pos = torch.cat(blocks, dim=1).to(DEVICE, DIT_DTYPE)
            region_mask = _region_joint_mask(
                region_entries, lw // 2, lh // 2, lengths, (lh // 2) * (lw // 2),
                DEVICE,
            )
            ctx_pos = (ctx_pos, pooled_pos, region_mask)
        else:
            ctx_pos = ctx_pos_plain
        ctx_neg = (
            t5.encode(negative).to(DEVICE, DIT_DTYPE),
            clip.encode(negative).to(DEVICE, DIT_DTYPE),
        )

        _report(progress, 0.1, "preparing latent")
        generator = torch.manual_seed(int(cfg["seed"]))
        if image is not None:
            latent = latent_format.process_in(_flux_pil_to_latent(vae, image, lh, lw))
        else:
            latent = torch.zeros(1, 16, lh, lw)
        latent_in = latent.to(DEVICE)

        sigmas = scheduler_sigmas(
ms, int(cfg["steps"]), float(cfg["denoise"]), cfg.get("scheduler", "simple"), cfg
    ).to(DEVICE)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")

        model = FluxApplyModel(dit, DIT_DTYPE, guidance=guidance)
        total = max(1, len(sigmas) - 1)
        emit = _make_preview(cfg, preview, latent_format, out_size=(width, height))

        def on_step(i, denoised, *_):
            _check(cancel)
            _report(progress, 0.15 + 0.75 * (i / total), "sampling", i / total)
            if emit is not None:
                emit(i, denoised, total)

        _report(progress, 0.15, "sampling", 0.0)
        samples = sample(
            ms, model, noise.to(DEVICE), latent_in, sigmas, ctx_pos, ctx_neg,
            float(cfg["cfg"]), sampler=cfg.get("sampler", "euler"),
            callback=on_step if (progress is not None or cancel is not None
                                 or emit is not None) else None,
            mask=mask, feather=float(cfg.get("feather") or 0.0),
            cancel=cancel,
        )
        if _upscale_spec(cfg) is not None:
            samples = _hires_fix(
                cfg, samples, ms, model, ctx_pos_plain, ctx_neg,
                cfg.get("sampler", "euler"), int(cfg["seed"]), progress, cancel,
                lambda lat: _flux_latent_to_pil(
                    vae, latent_format.process_out(lat),
                    int(lat.shape[-2]) * 8, int(lat.shape[-1]) * 8,
                ),
                lambda pil, h, w: latent_format.process_in(_flux_pil_to_latent(vae, pil, h, w)),
                make_ms=lambda h, w: ModelSamplingFlux(shift=calculate_shift(
                    (h // 2) * (w // 2), base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15)),
            )
            _report(progress, 1.0, "done")
            return _flux_latent_to_pil(
                vae, latent_format.process_out(samples),
                int(samples.shape[-2]) * 8, int(samples.shape[-1]) * 8,
            )
        samples = latent_format.process_out(samples.to(torch.float32))

        _report(progress, 0.95, "decoding")
        pil = _flux_latent_to_pil(vae, samples, height, width)
        _report(progress, 1.0, "done")
        return pil


def _load_z(path, text_path=None, vae_path=None, loras=(), progress=None, fp8="auto",
            cancel=None):
    global _z, _z_key, _z_error
    loras = list(loras)
    key = (path, text_path, vae_path, fp8, tuple(loras))
    if _z is not None and _z_key == key:
        return _z
    _z = None
    _z_error = None
    _z_key = key
    # A reload replaces the DiT object, so the ControlNet attached to the old one
    # must be forgotten too.
    _reset_zcontrol()
    try:
        _stage(cancel, progress, 0.05, "loading text encoder")
        text = ZImageTextEncoder(
            text_path or paths.ZIMAGE_DEFAULTS["text"], device="cpu", dtype=DIT_DTYPE,
            tick=lambda f: _stage(cancel, progress, 0.05 + 0.33 * f, "loading text encoder", f),
        )

        _stage(cancel, progress, 0.4, "loading transformer")
        tune = tuning.read()
        streaming_on = tune["disk_stream"] and _can_stream(path, loras)
        dit, missing, _ = load_z_image(
            path, device="cpu", dtype=DIT_DTYPE, stream=streaming_on,
            stream_device=DEVICE, ram_budget=int(tune["ram_budget_gb"] * 1e9),
            prefetch_depth=tune["prefetch_depth"], pin_blocks=tune["keep_resident"],
        )
        if missing:
            raise RuntimeError(f"transformer weights missing: {missing[:4]}")
        place_dit(dit, DEVICE, fp8=fp8)
        _place_text_encoders(text)
        _apply_loras(dit, loras, progress, 0.75, cancel)

        _stage(cancel, progress, 0.85, "loading vae")
        vae, _, _ = load_flux_vae(vae_path or paths.ZIMAGE_DEFAULTS["vae"], device=DEVICE, dtype=VAE_DTYPE)

        _z = {"dit": dit, "text": text, "vae": vae}
    except Cancelled:
        _z_key = None
        raise
    except Exception as exc:
        _z_error = exc
    return _z


def _generate_z(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
                control=None, regions_list=None, preview=None):
    """Z-Image txt2img / img2img (Lumina2 DiT + Qwen3-4B + Flux VAE).

    With ``config["controlnet"]`` set, the Z-Image ControlNet-Union steers the run
    from the control image (VAE-encoded to a latent, see
    :mod:`engine.z_controlnet`).
    """
    base = _load_z(
        cfg.get("dit") or paths.ZIMAGE_DEFAULTS["dit"],
        cfg.get("text"), cfg.get("vae"), _cfg_loras(cfg), progress, cfg.get("fp8", "auto"),
        cancel,
    )
    _check(cancel)
    if _z_error is not None:
        raise RuntimeError(f"Z-Image load failed: {_z_error}")
    if base is None:
        raise RuntimeError("Z-Image load failed.")

    dit, text, vae = base["dit"], base["text"], base["vae"]
    dit_path = cfg.get("dit") or paths.ZIMAGE_DEFAULTS["dit"]
    nitro = paths.z_image_kind(dit_path) == "nitrofusion"
    width, height = int(cfg["width"]), int(cfg["height"])
    lh, lw = height // 8, width // 8
    latent_format = FluxLatent()
    ms = ModelSamplingFlow(shift=6.0 if nitro else 3.0)

    # NitroFusion2's de-distillation LoRA is applied on the first step only: it
    # takes the model back toward Z-Image-Turbo to remove artifacts, then the
    # second step runs NitroFusion2 alone. Schedule it by step, not uniformly.
    lora_schedule = []
    if nitro and int(cfg["steps"]) >= 2:
        for lora_path, _strength in _cfg_loras(cfg):
            if is_dedistill(lora_path):
                lora_schedule.append((lora_path, (1.0, 0.0)))

    specs = _cfg_controlnets(cfg, "controlnet")
    control_latent = None
    control_scale = 1.0
    control_start, control_end = 0.0, 1.0
    control_ctx = None
    if specs:
        # The Z-Image control branch is a single built-in DiT path, so only the
        # first adapter is used.
        path, strength, preprocessor, control_start, control_end = specs[0]
        if control is None:
            raise RuntimeError(
                "ControlNet needs a source image; set the panel's Source."
            )
        hint_image = _as_image_list(control, 1)[0]
        _attach_zcontrol(dit, path, progress, cancel)
        _report(progress, 0.08, "preparing control image")
        hint = preprocess.prepare(hint_image, preprocessor, (width, height))
        control_latent = latent_format.process_in(
            _flux_pil_to_latent(vae, hint, lh, lw)
        ).to(DEVICE, DIT_DTYPE)
        control_scale = strength
        control_ctx = {
            "image": hint_image, "preprocessor": preprocessor, "strength": strength,
            "start": control_start, "end": control_end,
        }

    region_entries = [r for r in (regions_list or []) if r.get("mask") is not None]
    with _lock:
        _report(progress, 0.0, "encoding prompt")
        base_ctx = text.encode(prompt or "")
        ctx_pos_plain = base_ctx.to(DEVICE, DIT_DTYPE)
        if region_entries:
            blocks = [base_ctx]
            for region in region_entries:
                blocks.append(text.encode(region.get("prompt") or ""))
            ctx = torch.cat(blocks, dim=1).to(DEVICE, DIT_DTYPE)
            # Z-Image pads the caption and image to a fixed multiple, so the mask
            # spans the padded text and the (padded) image tokens.
            mult = getattr(dit, "pad_tokens_multiple", None)
            image_len = regions.pad_length((lh // 2) * (lw // 2), mult)
            region_mask = _region_joint_mask(
                region_entries, lw // 2, lh // 2, [int(b.shape[1]) for b in blocks],
                image_len, DEVICE, text_len=regions.pad_length(ctx.shape[1], mult),
            )
            ctx_pos = (ctx, region_mask)
        else:
            ctx_pos = ctx_pos_plain
        ctx_neg = text.encode(cfg.get("negative") or "").to(DEVICE, DIT_DTYPE)

        _report(progress, 0.1, "preparing latent")
        generator = torch.manual_seed(int(cfg["seed"]))
        if image is not None:
            latent = latent_format.process_in(_flux_pil_to_latent(vae, image, lh, lw))
        else:
            latent = torch.zeros(1, 16, lh, lw)
        latent_in = latent.to(DEVICE)

        denoise = float(cfg["denoise"])
        if nitro and denoise > 0.9999:
            # NitroFusion2 samples on an explicit linear ramp (1 -> 0); a
            # partial-denoise img2img keeps the scheduler so denoise still bites.
            sigmas = linear_sigmas(int(cfg["steps"])).to(DEVICE)
        else:
            sigmas = scheduler_sigmas(
                ms, int(cfg["steps"]), denoise, cfg.get("scheduler", "simple"), cfg
            ).to(DEVICE)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")

        model = ZImageApplyModel(
            dit, DIT_DTYPE, control_latent=control_latent, control_scale=control_scale,
            control_start=control_start, control_end=control_end,
            lora_schedule=lora_schedule,
        )
        total = max(1, len(sigmas) - 1)
        emit = _make_preview(cfg, preview, latent_format, out_size=(width, height))

        def on_step(i, denoised, *_):
            _check(cancel)
            _report(progress, 0.15 + 0.75 * (i / total), "sampling", i / total)
            if emit is not None:
                emit(i, denoised, total)

        _report(progress, 0.15, "sampling", 0.0)
        samples = sample(
            ms, model, noise.to(DEVICE), latent_in, sigmas, ctx_pos, ctx_neg,
            float(cfg["cfg"]), sampler=cfg.get("sampler", "euler"),
            callback=on_step if (progress is not None or cancel is not None
                                 or emit is not None) else None,
            mask=mask, feather=float(cfg.get("feather") or 0.0),
            cancel=cancel,
        )
        if _upscale_spec(cfg) is not None:
            # The Z-Image control hint is a fixed-grid latent, so it is rebuilt
            # for the larger grid (or dropped when unchanged).
            model2 = model
            if control_ctx is not None:
                th, tw = int(samples.shape[-2]), int(samples.shape[-1])
                hint = preprocess.prepare(
                    control_ctx["image"], control_ctx["preprocessor"], (tw * 8, th * 8)
                )
                model2 = ZImageApplyModel(
                    dit, DIT_DTYPE,
                    control_latent=latent_format.process_in(
                        _flux_pil_to_latent(vae, hint, th, tw)
                    ).to(DEVICE, DIT_DTYPE),
                    control_scale=control_ctx["strength"],
                    control_start=control_ctx["start"], control_end=control_ctx["end"],
                    lora_schedule=lora_schedule,
                )
            samples = _hires_fix(
                cfg, samples, ms, model2, ctx_pos_plain, ctx_neg,
                cfg.get("sampler", "euler"), int(cfg["seed"]), progress, cancel,
                lambda lat: _flux_latent_to_pil(
                    vae, latent_format.process_out(lat),
                    int(lat.shape[-2]) * 8, int(lat.shape[-1]) * 8,
                ),
                lambda pil, h, w: latent_format.process_in(_flux_pil_to_latent(vae, pil, h, w)),
            )
            _report(progress, 1.0, "done")
            return _flux_latent_to_pil(
                vae, latent_format.process_out(samples),
                int(samples.shape[-2]) * 8, int(samples.shape[-1]) * 8,
            )
        samples = latent_format.process_out(samples.to(torch.float32))

        _report(progress, 0.95, "decoding")
        pil = _flux_latent_to_pil(vae, samples, height, width)
        _report(progress, 1.0, "done")
        return pil


def _qwen_pil_to_latent(vae, image, lh, lw):
    img = image.convert("RGB").resize((lw * 16, lh * 16), Image.LANCZOS)
    arr = np.asarray(img).astype("float32") / 255.0
    px = torch.from_numpy(arr)[None].permute(0, 3, 1, 2).to(DEVICE, WAN_VAE_DTYPE) * 2.0 - 1.0
    alpha = torch.ones_like(px[:, :1])
    px = torch.cat([px, alpha], dim=1)  # RGBA -> [1,4,H,W]
    with torch.no_grad():
        return vae.encode(px).float()


def _qwen_latent_to_pil(vae, latent, height, width):
    with torch.inference_mode():
        out = _vae_decode(vae, latent.to(DEVICE, WAN_VAE_DTYPE)).float()
    out = torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)[0, :3]
    out = out[:, :height, :width]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, "RGB")


def _qwen_condition_latents(vae, images, sizes, latent_format):
    """VAE-encode reference images at the vision tower's own size and pack them.

    ``sizes`` is the ``(w, h)`` each image was resized to for the vision tower, so
    the packed latents line up with the slots the text encoder reserved (each
    slot is a 2x2 group of latent tokens). Returns ``(latents [1, S, 64], shapes)``.
    """
    blocks, shapes = [], []
    for image, (pw, ph) in zip(images, sizes):
        img = image.convert("RGB").resize((pw, ph), Image.LANCZOS)
        lw, lh = pw // 16, ph // 16
        latent = latent_format.process_in(_qwen_pil_to_latent(vae, img, lh, lw))
        blocks.append(latent.flatten(2).transpose(1, 2))
        shapes.append((1, lh, lw))
    return torch.cat(blocks, dim=1), shapes


def _generate_qwen(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
                   regions_list=None, ref_images=None, preview=None):
    """Qwen-Image 2.1 txt2img / img2img / reference-edit (GGUF DiT + Qwen3-VL + VAE).

    ``ref_images`` is an optional list of PIL reference images for edit-style
    conditioning: the Qwen3-VL vision tower sees them and the DiT fills the
    vision slots with their latents (see :class:`engine.qwen_image.QwenRef`).
    Cannot be combined with ``regions_list``.
    """
    base = _load_qwen(
        cfg.get("dit") or paths.QWEN_DEFAULTS["dit"],
        cfg.get("text"), cfg.get("vae"), _cfg_loras(cfg), progress, cancel,
        fp8=cfg.get("fp8", "auto"),
    )
    _check(cancel)
    if _qwen_error is not None:
        raise RuntimeError(f"Qwen-Image load failed: {_qwen_error}")
    if base is None:
        raise RuntimeError("Qwen-Image load failed.")

    dit, te, vae = base["dit"], base["text"], base["vae"]
    width, height = int(cfg["width"]), int(cfg["height"])
    lh, lw = height // 16, width // 16
    latent_format = QwenImage21Latent()
    ms = ModelSamplingFlux(shift=calculate_shift(lh * lw))

    region_entries = [r for r in (regions_list or []) if r.get("mask") is not None]
    ref_images = [im for im in (ref_images or []) if im is not None]
    if ref_images and region_entries:
        raise RuntimeError("Reference images and regional prompts cannot be combined.")
    with _lock:
        _report(progress, 0.0, "encoding prompt")
        if ref_images:
            hidden, image_pad_mask, sizes = te.encode_with_images(
                prompt or "", ref_images, width * height,
            )
            ref_latents, ref_shapes = _qwen_condition_latents(
                vae, ref_images, sizes, latent_format,
            )
            ref = QwenRef(
                ref_latents.to(DEVICE), ref_shapes, image_pad_mask.to(DEVICE),
            )
            ctx_pos = ctx_pos_plain = QwenConditioning(
                hidden.to(DEVICE, DIT_DTYPE), ref=ref, mask=None,
            )
        else:
            base_ctx = te.encode(prompt or "")
            ctx_pos_plain = base_ctx.to(DEVICE, DIT_DTYPE)
            if region_entries:
                blocks = [base_ctx]
                for region in region_entries:
                    blocks.append(te.encode(region.get("prompt") or ""))
                ctx = torch.cat(blocks, dim=1).to(DEVICE, DIT_DTYPE)
                region_mask = _region_joint_mask(
                    region_entries, lw, lh, [int(b.shape[1]) for b in blocks], lh * lw, DEVICE,
                )
                ctx_pos = (ctx, region_mask)
            else:
                ctx_pos = ctx_pos_plain
        ctx_neg = te.encode(cfg.get("negative") or "").to(DEVICE, DIT_DTYPE)

        _report(progress, 0.1, "preparing latent")
        generator = torch.manual_seed(int(cfg["seed"]))
        if image is not None:
            latent = latent_format.process_in(_qwen_pil_to_latent(vae, image, lh, lw))
        else:
            latent = torch.zeros(1, 64, lh, lw)
        latent_in = latent.to(DEVICE)

        sigmas = scheduler_sigmas(
ms, int(cfg["steps"]), float(cfg["denoise"]), cfg.get("scheduler", "simple"), cfg
    ).to(DEVICE)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")

        model = QwenImageApplyModel(dit, DIT_DTYPE)
        total = max(1, len(sigmas) - 1)
        emit = _make_preview(cfg, preview, latent_format, out_size=(width, height))

        def on_step(i, denoised, *_):
            _check(cancel)
            _report(progress, 0.15 + 0.75 * (i / total), "sampling", i / total)
            if emit is not None:
                emit(i, denoised, total)

        _report(progress, 0.15, "sampling", 0.0)
        samples = sample(
            ms, model, noise.to(DEVICE), latent_in, sigmas, ctx_pos, ctx_neg,
            float(cfg["cfg"]), sampler=cfg.get("sampler", "euler"),
            callback=on_step if (progress is not None or cancel is not None
                                 or emit is not None) else None,
            mask=mask, feather=float(cfg.get("feather") or 0.0),
            cancel=cancel,
        )
        if _upscale_spec(cfg) is not None:
            samples = _hires_fix(
                cfg, samples, ms, model, ctx_pos_plain, ctx_neg,
                cfg.get("sampler", "euler"), int(cfg["seed"]), progress, cancel,
                lambda lat: _qwen_latent_to_pil(
                    vae, latent_format.process_out(lat),
                    int(lat.shape[-2]) * 16, int(lat.shape[-1]) * 16,
                ),
                lambda pil, h, w: latent_format.process_in(_qwen_pil_to_latent(vae, pil, h, w)),
                make_ms=lambda h, w: ModelSamplingFlux(shift=calculate_shift(h * w)),
            )
            _report(progress, 1.0, "done")
            return _qwen_latent_to_pil(
                vae, latent_format.process_out(samples),
                int(samples.shape[-2]) * 16, int(samples.shape[-1]) * 16,
            )
        samples = latent_format.process_out(samples.to(torch.float32))

        _report(progress, 0.95, "decoding")
        pil = _qwen_latent_to_pil(vae, samples, height, width)
        _report(progress, 1.0, "done")
        return pil


def _load_krea2(path, text_path=None, vae_path=None, loras=(), progress=None, fp8="auto",
                cancel=None):
    global _krea2, _krea2_key, _krea2_error, _krea2_trace
    import traceback

    loras = list(loras)
    key = (path, text_path, vae_path, fp8, tuple(loras))
    if _krea2 is not None and _krea2_key == key:
        return _krea2
    print(
        f"[krea2] loading model loras={[os.path.basename(p) for p, _ in loras] or 'none'}",
        flush=True,
    )
    # Replacing a differently-keyed engine (e.g. toggling the AnyPaint adapter
    # changes the LoRA list) must release the old DiT *before* the new one is read
    # from disk: a Krea 2 checkpoint is ~24 GB of bf16 and the GGUF read pages the
    # file back in, so holding both at once exhausts RAM and the read faults.
    if _krea2 is not None:
        _krea2 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _krea2 = None
    _krea2_error = None
    _krea2_trace = None
    _krea2_key = key
    try:
        _stage(cancel, progress, 0.05, "loading text encoder")
        text = load_krea2_text(
            text_path or paths.KREA2_DEFAULTS["text"], device="cpu", dtype=DIT_DTYPE,
            tick=lambda f: _stage(cancel, progress, 0.05 + 0.25 * f, "loading text encoder", f),
        )
        if getattr(text, "missing", None):
            raise RuntimeError(
                f"text encoder weights missing ({len(text.missing)}): {text.missing[:8]}"
            )

        _stage(cancel, progress, 0.35, "loading transformer")
        tune = tuning.read()
        streaming_on = tune["disk_stream"] and _can_stream(path, loras)
        dit, missing, _ = load_krea2_dit(
            path, device="cpu", dtype=DIT_DTYPE, stream=streaming_on,
            stream_device=DEVICE, ram_budget=int(tune["ram_budget_gb"] * 1e9),
            prefetch_depth=tune["prefetch_depth"], pin_blocks=tune["keep_resident"],
            tick=lambda f: _stage(cancel, progress, 0.35 + 0.35 * f, "loading transformer", f),
        )
        if missing:
            raise RuntimeError(f"transformer weights missing: {missing[:4]}")
        cfg_text = text.model.config
        print(
            f"[krea2] text: hidden={cfg_text.hidden_size} layers={cfg_text.num_hidden_layers} "
            f"heads={cfg_text.num_attention_heads}/{cfg_text.num_key_value_heads} "
            f"head_dim={cfg_text.head_dim} missing={len(text.missing)}",
            flush=True,
        )
        print(
            f"[krea2] dit: features={dit.first.weight.shape[0]} layers={len(dit.blocks)} "
            f"heads={dit.heads} txtdim={dit.txtdim} txtlayers={dit.txtlayers}",
            flush=True,
        )
        _stage(cancel, progress, 0.72, "placing transformer")
        status = place_dit(dit, DEVICE, fp8=fp8)
        _place_text_encoders(text)
        print(
            f"[krea2] placement={status} dit={dit_bytes(dit) / 1e9:.1f}GB "
            f"free_vram={free_vram() / 1e9:.1f}GB",
            flush=True,
        )
        patched = _apply_loras(dit, loras, progress, 0.8, cancel)
        if loras:
            print(f"[krea2] lora patched {patched} module layers", flush=True)

        _stage(cancel, progress, 0.85, "loading vae")
        vae, vae_missing, vae_unexpected = load_vae(
            vae_path or paths.KREA2_DEFAULTS["vae"], device=DEVICE, dtype=WAN_VAE_DTYPE
        )
        print(
            f"[krea2] vae missing={len(vae_missing)} unexpected={len(vae_unexpected)} "
            f"z_dim={getattr(vae, 'z_dim', '?')}",
            flush=True,
        )
        if len(vae_missing) > 8:
            raise RuntimeError(f"VAE weights missing ({len(vae_missing)}): {vae_missing[:4]}")

        _krea2 = {"dit": dit, "text": text, "vae": vae}
    except Cancelled:
        _krea2_key = None
        raise
    except Exception as exc:
        _krea2_error = exc
        _krea2_trace = traceback.format_exc()
        traceback.print_exc()
    return _krea2


def _krea2_loras(cfg):
    """The model-row LoRAs plus the AnyPaint adapter, if enabled.

    AnyPaint is a functional adapter that applies only to its own masked edit
    mode, so it rides on the run's LoRA stack and therefore the load cache key:
    the same list must be produced by both the preload path and the generate path,
    otherwise preloading loads one key and generating reloads a second. Also skip
    it when the file is already on the model row (applying it twice roughly squares
    its effect and breaks the edit).
    """
    loras = list(_cfg_loras(cfg))
    if not (cfg.get("anypaint") and cfg.get("anypaint_lora")):
        return loras
    strength = float(cfg.get("anypaint_strength", 1.0) or 1.0)
    adapter = os.fspath(cfg["anypaint_lora"])
    key = os.path.normcase(adapter)
    base = os.path.normcase(os.path.basename(adapter))
    # The model row may store the adapter as a bare file name while the References
    # picker holds its absolute path, so match on either the full path or the file
    # name. Applying the same functional adapter twice roughly squares its effect
    # (it collapses back toward the input) and splits the load cache key.
    if not any(
        os.path.normcase(os.fspath(path)) == key
        or os.path.normcase(os.path.basename(os.fspath(path))) == base
        for path, _ in loras
    ):
        loras.append((adapter, strength))
    return loras


def _generate_krea2(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
                    regions_list=None, preview=None):
    """Krea 2 txt2img / img2img (single-stream MMDiT + Qwen3-VL-4B stack + Qwen VAE)."""
    dit_path = cfg.get("dit") or paths.KREA2_DEFAULTS["dit"]
    loras = _krea2_loras(cfg)
    base = _load_krea2(
        dit_path, cfg.get("text"), cfg.get("vae"), loras, progress,
        cfg.get("fp8", "auto"), cancel,
    )
    _check(cancel)
    if _krea2_error is not None:
        raise RuntimeError(f"Krea 2 load failed: {_krea2_error}\n{_krea2_trace or ''}")
    if base is None:
        raise RuntimeError("Krea 2 load failed.")

    dit, text, vae = base["dit"], base["text"], base["vae"]
    if cfg.get("anypaint") and image is not None and mask is not None:
        return _krea2_anypaint(
            prompt, cfg, dit, text, vae, image, mask, progress, cancel, preview
        )
    width, height = int(cfg["width"]), int(cfg["height"])
    lh, lw = height // 8, width // 8
    latent_format = Wan21()

    # The distilled (turbo) checkpoint uses a fixed timestep shift; the base
    # (raw) checkpoint derives it from the image-token count (patch 2x2).
    distilled = "turbo" in os.path.basename(dit_path).lower()
    seq_len = (lh // 2) * (lw // 2)
    shift = 1.15 if distilled else calculate_shift(seq_len, 256, 6400, 0.5, 1.15)
    ms = ModelSamplingFlux(shift=shift)

    region_entries = [r for r in (regions_list or []) if r.get("mask") is not None]
    with _lock:
        _report(progress, 0.0, "encoding prompt")
        hidden, hmask = text.encode(prompt or "")
        ctx_pos_plain = (hidden.to(DEVICE, DIT_DTYPE), hmask.to(DEVICE))
        if region_entries:
            blocks, keeps = [hidden], [hmask]
            for region in region_entries:
                block, keep = text.encode(region.get("prompt") or "")
                blocks.append(block)
                keeps.append(keep)
            ctx = torch.cat(blocks, dim=1)
            keep = torch.cat(keeps, dim=1)
            region_mask = _region_joint_mask(
                region_entries, lw // 2, lh // 2, [int(b.shape[1]) for b in blocks],
                (lh // 2) * (lw // 2), DEVICE, text_keep=keep[0].cpu().numpy(),
            )
            ctx_pos = (ctx.to(DEVICE, DIT_DTYPE), keep.to(DEVICE), region_mask)
        else:
            ctx_pos = ctx_pos_plain
        hidden, hmask = text.encode(cfg.get("negative") or "")
        ctx_neg = (hidden.to(DEVICE, DIT_DTYPE), hmask.to(DEVICE))

        _report(progress, 0.1, "preparing latent")
        generator = torch.manual_seed(int(cfg["seed"]))
        if image is not None:
            latent = latent_format.process_in(_pil_to_latent(vae, image, lh, lw))
        else:
            latent = torch.zeros(1, 16, 1, lh, lw)
        latent_in = latent.to(DEVICE)

        sigmas = scheduler_sigmas(
ms, int(cfg["steps"]), float(cfg["denoise"]), cfg.get("scheduler", "simple"), cfg
    ).to(DEVICE)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")

        model = Krea2ApplyModel(dit, DIT_DTYPE)
        total = max(1, len(sigmas) - 1)
        emit = _make_preview(cfg, preview, latent_format, out_size=(width, height))

        def on_step(i, denoised, *_):
            _check(cancel)
            _report(progress, 0.15 + 0.75 * (i / total), "sampling", i / total)
            if emit is not None:
                emit(i, denoised, total)

        _report(progress, 0.15, "sampling", 0.0)
        samples = sample(
            ms, model, noise.to(DEVICE), latent_in, sigmas, ctx_pos, ctx_neg,
            float(cfg["cfg"]), sampler=cfg.get("sampler", "euler"),
            callback=on_step if (progress is not None or cancel is not None
                                 or emit is not None) else None,
            mask=mask, feather=float(cfg.get("feather") or 0.0),
            cancel=cancel,
        )
        if _upscale_spec(cfg) is not None:
            samples = _hires_fix(
                cfg, samples, ms, model, ctx_pos_plain, ctx_neg,
                cfg.get("sampler", "euler"), int(cfg["seed"]), progress, cancel,
                lambda lat: _latent_to_pil(
                    vae, latent_format.process_out(lat),
                    int(lat.shape[-2]) * 8, int(lat.shape[-1]) * 8,
                ),
                lambda pil, h, w: latent_format.process_in(_pil_to_latent(vae, pil, h, w)),
                make_ms=lambda h, w: ModelSamplingFlux(shift=(
                    1.15 if distilled else calculate_shift((h // 2) * (w // 2), 256, 6400, 0.5, 1.15)
                )),
            )
            _report(progress, 1.0, "done")
            return _latent_to_pil(
                vae, latent_format.process_out(samples),
                int(samples.shape[-2]) * 8, int(samples.shape[-1]) * 8,
            )
        samples = latent_format.process_out(samples.to(torch.float32))

        _report(progress, 0.95, "decoding")
        pil = _latent_to_pil(vae, samples, height, width)
        _report(progress, 1.0, "done")
        return pil


def _krea2_anypaint(prompt, cfg, dit, text, vae, image, mask, progress=None,
                    cancel=None, preview=None):
    """Krea 2 AnyPaint: arbitrary-mask inpainting/outpainting/image editing.

    Runs the Krea 2 functional adapter (see :mod:`engine.krea2_anypaint`): the
    source ``image`` is placed on the canvas, the white pixels of ``mask`` are
    (re)generated and everything outside the (dilated) mask is pinned to its clean
    latent after every flow step. The adapter conditions on a low-res reference of
    the known canvas through isolated cached reference keys/values.
    """
    width, height = int(cfg["width"]), int(cfg["height"])
    lh, lw = height // 8, width // 8
    latent_format = Wan21()

    # The app hands the source and mask at document size; every other family is
    # resized to the run size inside the engine. Do the same here so the helper's
    # placement checks see a source that already matches the canvas (otherwise a
    # canvas that differs from the run size raises on the aspect/bbox sanity check).
    image = image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    if mask is not None:
        mask = mask.convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.NEAREST)

    prepared = anypaint.prepare_anypaint(image, mask, (width, height), None)

    # Temporary diagnostics: dump the source and the derived masks so the actual
    # generation region can be compared against what the user selected.
    try:
        _dbg = os.path.join(
            os.environ.get("APPDATA", os.path.expanduser("~")), "pai", "anypaint_debug"
        )
        os.makedirs(_dbg, exist_ok=True)
        image.save(os.path.join(_dbg, "1_source.png"))
        prepared.generated_mask.save(os.path.join(_dbg, "2_generated_mask.png"))
        prepared.keep_mask.save(os.path.join(_dbg, "3_keep_mask.png"))
        prepared.condition.save(os.path.join(_dbg, "4_condition.png"))
        print(
            f"[krea2-anypaint] canvas={prepared.canvas_size} "
            f"mask_bbox={prepared.generated_mask.getbbox()} debug={_dbg}",
            flush=True,
        )
    except OSError:
        pass

    dit_path = cfg.get("dit") or paths.KREA2_DEFAULTS["dit"]
    distilled = "turbo" in os.path.basename(dit_path).lower()
    shift = 1.15 if distilled else calculate_shift((lh // 2) * (lw // 2), 256, 6400, 0.5, 1.15)
    ms = ModelSamplingFlux(shift=shift)

    with _lock:
        _report(progress, 0.02, "encoding prompt")
        # The adapter was trained with the reference image embedded in the prompt
        # through the Qwen3-VL vision tower ("reference in prompt"), not just the
        # high-res reference latents. Build that encoder; fall back to the plain
        # text tower if it is unavailable so the run still proceeds.
        vl = None
        try:
            vl = load_vl_encoder(
                cfg.get("text") or paths.KREA2_DEFAULTS["text"],
                device="cpu", dtype=DIT_DTYPE,
            )
        except Exception as exc:  # pragma: no cover - depends on the checkpoint
            print(
                f"[krea2-anypaint] VL reference encoder unavailable ({exc}); "
                f"prompting without the reference image",
                flush=True,
            )
        condition = prepared.condition
        if vl is not None:
            hidden, hmask = vl.encode(prompt or "", condition)
            ctx_pos = (hidden.to(DEVICE, DIT_DTYPE), hmask.to(DEVICE))
            hidden, hmask = vl.encode(cfg.get("negative") or "", condition)
            ctx_neg = (hidden.to(DEVICE, DIT_DTYPE), hmask.to(DEVICE))
        else:
            hidden, hmask = text.encode(prompt or "")
            ctx_pos = (hidden.to(DEVICE, DIT_DTYPE), hmask.to(DEVICE))
            hidden, hmask = text.encode(cfg.get("negative") or "")
            ctx_neg = (hidden.to(DEVICE, DIT_DTYPE), hmask.to(DEVICE))

        generator = torch.manual_seed(int(cfg["seed"]))

        _report(progress, 0.1, "encoding reference")
        ref_lh, ref_lw = condition.height // 8, condition.width // 8
        ref_latent = latent_format.process_in(
            _pil_to_latent(vae, condition, ref_lh, ref_lw)
        )
        ref_tokens = anypaint.pack_latents(ref_latent.to(DEVICE, DIT_DTYPE))
        ref_token_count = ref_tokens.shape[1]
        bbox = tuple(prepared.reference_placement["bbox_normalized"])
        ref_positions = anypaint.reference_positions(
            ref_lh, ref_lw, (lh // 2, lw // 2), bbox=bbox, frame=1, device=DEVICE,
        )
        _report(progress, 0.12, "caching reference")
        ref_kv = dit.precompute_ref_kv(ref_tokens, ref_positions)
        del ref_tokens

        _report(progress, 0.15, "encoding source")
        known_latent = latent_format.process_in(
            _pil_to_latent(vae, prepared.known_image, lh, lw)
        ).to(DEVICE, torch.float32)
        token_keep = anypaint.known_token_mask(prepared.keep_mask, lh, lw)
        keep = anypaint.expand_token_mask(token_keep, lh, lw).to(DEVICE).bool()
        print(
            f"[krea2-anypaint] canvas={width}x{height} condition={condition.size} "
            f"ref_tokens={ref_token_count} keep={keep.float().mean().item():.3f} "
            f"steps={int(cfg['steps'])} cfg={float(cfg['cfg'])} "
            f"vl={'on' if vl is not None else 'off'} "
            f"loras={[os.path.basename(p) for p, _ in _krea2_loras(cfg)] or 'none'}",
            flush=True,
        )

        latent = torch.zeros(1, 16, 1, lh, lw)
        noise = torch.randn(
            latent.shape, generator=generator, dtype=torch.float32, device="cpu"
        )
        initial_noise = noise.to(DEVICE)

        model = Krea2ApplyModel(dit, DIT_DTYPE, ref_kv_cache=ref_kv)

        def post_step(_i, x, sigma_next):
            return anypaint.restore_known(
                x, known_latent, keep, initial_noise, float(sigma_next)
            )

        sigmas = scheduler_sigmas(
            ms, int(cfg["steps"]), 1.0, cfg.get("scheduler", "simple"), cfg
        ).to(DEVICE)
        total = max(1, len(sigmas) - 1)
        emit = _make_preview(cfg, preview, latent_format, out_size=(width, height))

        def on_step(i, denoised, *_):
            _check(cancel)
            _report(progress, 0.15 + 0.75 * (i / total), "sampling", i / total)
            if emit is not None:
                emit(i, denoised, total)

        _report(progress, 0.15, "sampling", 0.0)
        samples = sample(
            ms, model, noise.to(DEVICE), latent.to(DEVICE), sigmas, ctx_pos, ctx_neg,
            float(cfg["cfg"]), sampler="euler",
            callback=on_step if (progress is not None or cancel is not None
                                 or emit is not None) else None,
            post_step=post_step, cancel=cancel,
        )
        free = ~keep
        delta = float(
            (samples - known_latent).abs().detach()[free.expand_as(samples)].mean()
        )
        print(
            f"[krea2-anypaint] done: free-region latent delta vs source = {delta:.3f} "
            f"(0 means the mask did nothing)",
            flush=True,
        )
        samples = latent_format.process_out(samples.to(torch.float32))
        _report(progress, 0.95, "decoding")
        pil = _latent_to_pil(vae, samples, height, width)
        try:
            pil.save(os.path.join(_dbg, "5_output.png"))
        except (OSError, NameError):
            pass
        _report(progress, 1.0, "done")
        return pil


def _pil_to_latent(vae, image, lh, lw):
    img = image.convert("RGB").resize((lw * 8, lh * 8), Image.LANCZOS)
    arr = np.asarray(img).astype("float32") / 255.0
    pixels = torch.from_numpy(arr)[None].permute(0, 3, 1, 2).to(DEVICE, WAN_VAE_DTYPE)
    pixels = pixels * 2.0 - 1.0
    pixels = pixels.movedim(1, 0).unsqueeze(0)  # [1,3,1,H,W]
    with torch.no_grad():
        return vae.encode(pixels).float()


def _latent_to_pil(vae, latent, height, width):
    with torch.inference_mode():
        out = _vae_decode(vae, latent.to(DEVICE, WAN_VAE_DTYPE)).float()
    out = torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)[0, :, 0]
    out = out[:, :height, :width]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, "RGB")


def _anima_ref_latent(vae, image, lh, lw, latent_format, fill=255):
    """VAE-encode ``image`` letterboxed to the run size as a reference latent.

    ``image`` is fitted into ``(lw * 8, lh * 8)`` preserving aspect ratio (padded
    with ``fill``), matching the ComfyUI "Anima Ref Latent" node, then encoded and
    normalised with ``latent_format.process_in``. The result is the clean second
    frame :meth:`engine.cosmos.MiniTrainDIT.forward_with_ref` reads.
    """
    target = (lw * 8, lh * 8)
    src = image.convert("RGB")
    scale = min(target[0] / src.width, target[1] / src.height)
    size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
    canvas = Image.new("RGB", target, (fill, fill, fill))
    canvas.paste(
        src.resize(size, Image.LANCZOS),
        ((target[0] - size[0]) // 2, (target[1] - size[1]) // 2),
    )
    return latent_format.process_in(_pil_to_latent(vae, canvas, lh, lw))


def _anima_ref_latent_area(vae, image, megapixels, latent_format):
    """VAE-encode the reference at about ``megapixels`` area, its own aspect ratio.

    The v2 (asymmetric) adapter reads the reference at any size, so it is not
    letterboxed to the run size: it is scaled to roughly ``megapixels`` (rounded
    to a multiple of 16) and encoded directly. The result is the clean latent
    :meth:`engine.anima_ref.RefAdapter.ref_kv` reads.
    """
    src = image.convert("RGB")
    scale = (max(0.01, float(megapixels)) * 1024 * 1024 / (src.width * src.height)) ** 0.5
    width = max(16, round(src.width * scale / 16) * 16)
    height = max(16, round(src.height * scale / 16) * 16)
    return latent_format.process_in(
        _pil_to_latent(vae, src, height // 8, width // 8)
    )


def _anima_ref_adapter(dit, path, strength, progress=None, cancel=None):
    """Load and install the reference adapter LoRA on ``dit`` (cached), or ``None``.

    Returns ``None`` when no path is set, so a run can still attach a reference
    frame without the LoRA (useful for testing the forward, though the adapter
    weights are what teach the DiT to read it).
    """
    if not path or not os.path.isfile(path):
        return None
    key = (id(dit), os.path.abspath(path), float(strength))
    adapter = _ref_adapter_cache.get(key)
    if adapter is None:
        _stage(cancel, progress, 0.82, "loading reference adapter")
        adapter = RefAdapter(path, strength)
        if adapter.install(dit) == 0:
            raise RuntimeError(
                f"The reference adapter {os.path.basename(path)} targets no DiT "
                "layers; check that it is an Anima reference adapter LoRA."
            )
        _ref_adapter_cache[key] = adapter
    return adapter


def _load_moss(path, text_path=None, vae_path=None, loras=(), progress=None,
               fp8="auto", text_repo=None, cancel=None):
    global _moss, _moss_key, _moss_error
    loras = list(loras)
    key = (path, text_path, vae_path, fp8, text_repo, tuple(loras))
    if _moss is not None and _moss_key == key:
        return _moss
    _moss = None
    _moss_error = None
    _moss_key = key
    try:
        # The Qwen3.5-9B encoder runs on CPU (like the Qwen / Flux paths); only
        # the DiT blocks stream to the GPU.
        _stage(cancel, progress, 0.05, "loading text encoder")
        text = MossTextEncoder(
            text_path or paths.MOSS_DEFAULTS["text"], device="cpu", dtype=DIT_DTYPE,
            repo=text_repo,
            tick=lambda f: _stage(cancel, progress, 0.05 + 0.38 * f, "loading text encoder", f),
        )

        _stage(cancel, progress, 0.45, "loading transformer")
        tune = tuning.read()
        streaming_on = tune["disk_stream"] and _can_stream(path, loras)
        dit, missing, _ = load_moss_image(
            path, device="cpu", dtype=DIT_DTYPE, stream=streaming_on,
            stream_device=DEVICE, ram_budget=int(tune["ram_budget_gb"] * 1e9),
            prefetch_depth=tune["prefetch_depth"], pin_blocks=tune["keep_resident"],
        )
        if missing:
            raise RuntimeError(f"transformer weights missing: {missing[:4]}")
        place_dit(dit, DEVICE, fp8=fp8)
        _place_text_encoders(text)
        _apply_loras(dit, loras, progress, 0.8, cancel)

        _stage(cancel, progress, 0.88, "loading vae")
        vae, vae_missing, _ = load_moss_vae(
            vae_path or paths.MOSS_DEFAULTS["vae"], device=DEVICE, dtype=VAE_DTYPE,
        )
        if len(vae_missing) > 8:
            raise RuntimeError(f"VAE weights missing ({len(vae_missing)}): {vae_missing[:4]}")

        _moss = {"dit": dit, "text": text, "vae": vae}
    except Cancelled:
        _moss_key = None
        raise
    except Exception as exc:
        _moss_error = exc
    return _moss


def _moss_pil_to_latent(vae, image, lh, lw):
    img = image.convert("RGB").resize((lw * 16, lh * 16), Image.LANCZOS)
    arr = np.asarray(img).astype("float32") / 255.0
    px = torch.from_numpy(arr)[None].permute(0, 3, 1, 2).to(DEVICE, VAE_DTYPE) * 2.0 - 1.0
    with torch.no_grad():
        z32 = vae.encode(px).latent_dist.mean.float()
    return moss_pack_latents(vae, z32)


def _moss_latent_to_pil(vae, latent, height, width):
    z32 = moss_unpack_latents(vae, latent.to(DEVICE, VAE_DTYPE))
    with torch.inference_mode():
        out = vae.decode(z32, return_dict=False)[0]
    out = torch.clamp((out.float() + 1.0) / 2.0, 0.0, 1.0)[0]
    out = out[:, :height, :width]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, "RGB")


def _generate_moss(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
                   regions_list=None, preview=None):
    """MOSS Image txt2img / img2img (5B DiT + Qwen3.5-9B + FLUX.2 VAE).

    Flow-matching Euler at shift 5 (the reference default); the CFG negative
    branch skips block 22 (``uncond_skip_layers``). The FLUX.2 VAE's 32-channel
    latent is 2x2 patch-packed and BatchNorm-normalised to the DiT's 128 channels
    by :mod:`engine.moss_vae`.
    """
    base = _load_moss(
        cfg.get("dit") or paths.MOSS_DEFAULTS["dit"],
        cfg.get("text"), cfg.get("vae"), _cfg_loras(cfg), progress,
        cfg.get("fp8", "auto"), cfg.get("text_repo"), cancel,
    )
    _check(cancel)
    if _moss_error is not None:
        raise RuntimeError(f"MOSS Image load failed: {_moss_error}")
    if base is None:
        raise RuntimeError("MOSS Image load failed.")

    dit, text, vae = base["dit"], base["text"], base["vae"]
    width, height = int(cfg["width"]), int(cfg["height"])
    lh, lw = height // 16, width // 16
    latent_format = MossImageLatent()
    ms = ModelSamplingFlow(shift=5.0)

    skip_layers = tuple(cfg.get("uncond_skip_layers") or (22,))
    region_entries = [r for r in (regions_list or []) if r.get("mask") is not None]
    with _lock:
        _report(progress, 0.0, "encoding prompt")
        base_ctx = text.encode(prompt or "").to(DEVICE, DIT_DTYPE)
        ctx_pos_plain = base_ctx
        if region_entries:
            blocks = [base_ctx]
            for region in region_entries:
                blocks.append(text.encode(region.get("prompt") or "").to(DEVICE, DIT_DTYPE))
            ctx = torch.cat(blocks, dim=1)
            # Named ``region_mask`` (not ``mask``) so it does not shadow the
            # inpaint ``mask`` argument that ``sample`` below still needs.
            region_mask = _region_joint_mask(
                region_entries, lw, lh, [int(b.shape[1]) for b in blocks], lh * lw, DEVICE,
            )
            # The shared mask is text-then-image; the MOSS sequence is
            # image-then-text, so reorder rows and columns to match. Rows are
            # axis -2 and columns axis -1, so permute each explicitly (a bare
            # ``mask[..., perm]`` would only touch the columns -- twice).
            text_len, image_len = int(ctx.shape[1]), lh * lw
            perm = torch.cat([
                torch.arange(text_len, text_len + image_len, device=DEVICE),
                torch.arange(text_len, device=DEVICE),
            ])
            reordered = region_mask[:, :, perm][:, :, :, perm]
            ctx_pos = (ctx, reordered)
        else:
            ctx_pos = ctx_pos_plain
        # The negative branch is the 3-tuple form (see ``_moss_context``): it
        # carries the blocks the uncond pass skips.
        neg = text.encode(cfg.get("negative") or "").to(DEVICE, DIT_DTYPE)
        ctx_neg = (neg, None, skip_layers)

        _report(progress, 0.1, "preparing latent")
        generator = torch.manual_seed(int(cfg["seed"]))
        if image is not None:
            latent = latent_format.process_in(_moss_pil_to_latent(vae, image, lh, lw))
        else:
            latent = torch.zeros(1, 128, lh, lw)
        latent_in = latent.to(DEVICE)

        sigmas = scheduler_sigmas(
ms, int(cfg["steps"]), float(cfg["denoise"]), cfg.get("scheduler", "simple"), cfg
    ).to(DEVICE)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")

        model = MossImageApplyModel(dit, DIT_DTYPE)
        total = max(1, len(sigmas) - 1)
        emit = _make_preview(cfg, preview, latent_format, out_size=(width, height))

        def on_step(i, denoised, *_):
            _check(cancel)
            _report(progress, 0.15 + 0.75 * (i / total), "sampling", i / total)
            if emit is not None:
                emit(i, denoised, total)

        _report(progress, 0.15, "sampling", 0.0)
        samples = sample(
            ms, model, noise.to(DEVICE), latent_in, sigmas, ctx_pos, ctx_neg,
            float(cfg["cfg"]), sampler=cfg.get("sampler", "euler"),
            callback=on_step if (progress is not None or cancel is not None
                                 or emit is not None) else None,
            mask=mask, feather=float(cfg.get("feather") or 0.0),
            cancel=cancel,
        )
        if _upscale_spec(cfg) is not None:
            # The regional mask is sized for the pass-1 token grid, so the second
            # pass runs plain (see :func:`_hires_fix`).
            samples = _hires_fix(
                cfg, samples, ms, model, ctx_pos_plain, ctx_neg,
                cfg.get("sampler", "euler"), int(cfg["seed"]), progress, cancel,
                lambda lat: _moss_latent_to_pil(
                    vae, latent_format.process_out(lat),
                    int(lat.shape[-2]) * 16, int(lat.shape[-1]) * 16,
                ),
                lambda pil, h, w: latent_format.process_in(_moss_pil_to_latent(vae, pil, h, w)),
            )
            _report(progress, 1.0, "done")
            return _moss_latent_to_pil(
                vae, latent_format.process_out(samples),
                int(samples.shape[-2]) * 16, int(samples.shape[-1]) * 16,
            )
        samples = latent_format.process_out(samples.to(torch.float32))

        _report(progress, 0.95, "decoding")
        pil = _moss_latent_to_pil(vae, samples, height, width)
        _report(progress, 1.0, "done")
        return pil


def _load_h3(path, vae_path=None, loras=(), progress=None, fp8="auto",
             cancel=None):
    """Load and cache the H3 DiT + video VAE.

    The Qwen3-VL-32B encoder is not cached here; it is streamed per generation
    (one decoder layer resident at a time, see :mod:`engine.h3_text`) so it
    coexists with the DiT instead of needing to be evicted.
    """
    global _h3, _h3_key, _h3_error
    loras = list(loras)
    key = (path, vae_path, fp8, tuple(loras))
    if _h3 is not None and _h3_key == key:
        return _h3
    _h3 = None
    _h3_error = None
    _h3_key = key
    try:
        _stage(cancel, progress, 0.4, "loading transformer")
        dit, missing = load_h3_dit(path, device="cpu", dtype=DIT_DTYPE, fp8=fp8)
        if missing:
            raise RuntimeError(f"transformer weights missing: {missing[:4]}")

        # Load the VAE *before* placing the DiT: the H3 video VAE is ~5 GB bf16
        # and stays on the GPU through sampling, so ``place_dit`` has to see it
        # to reserve room for it. Placing first fills the card with resident
        # blocks that the VAE then oversubscribes, and the next forward OOMs.
        _stage(cancel, progress, 0.8, "loading vae")
        vae, vae_missing = load_h3_vae(
            vae_path or paths.H3_DEFAULTS["vae"], device=DEVICE, dtype=VAE_DTYPE,
        )
        if len(vae_missing) > 8:
            raise RuntimeError(f"VAE weights missing ({len(vae_missing)}): {vae_missing[:4]}")

        place_dit(dit, DEVICE, fp8=fp8)
        _apply_loras(dit, loras, progress, 0.9, cancel)

        _h3 = {"dit": dit, "vae": vae}
    except Cancelled:
        _h3_key = None
        raise
    except Exception as exc:
        _h3_error = exc
    return _h3


def _h3_text_encode(text_path, prompt, progress, base=0.0, span=0.15, cancel=None):
    """Encode ``prompt`` with the streamed Qwen3-VL-32B encoder.

    The encoder is never materialized dense: it reads one decoder layer at a
    time from the mmap'd safetensors (~1 GB working set), so it runs alongside a
    resident DiT. The returned context is ``[T, 5120]`` on CPU.
    """
    _stage(cancel, progress, base, "loading text encoder")
    text = load_h3_text(
        text_path or paths.H3_DEFAULTS["text"], device="cpu", dtype=DIT_DTYPE,
        tick=lambda f: _stage(cancel, progress, base + span * f, "loading text encoder", f),
    )
    try:
        hidden, _tags = text.encode(prompt or "")
        return hidden[0].to("cpu")
    finally:
        del text
        gc.collect()



def _generate_h3(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
                 regions_list=None, preview=None):
    """MiniMax H3 text-to-image still (33B DiT + Qwen3-VL-32B + video VAE).

    One latent frame is packed as a still (``engine.h3_layout``), denoised with
    the data-ward rectified-flow Euler schedule (:mod:`engine.h3_sampler`, video
    shift 12 / audio shift 3) and decoded by the video VAE's single-frame path
    (repeat to a chunk, keep ``frame_pre_padding``). The audio stream is carried
    only to drive the joint attention; it is discarded.

    The Qwen3-VL-32B encoder is streamed (see :mod:`engine.h3_text`), so a
    preloaded DiT stays resident through encoding.
    """
    width, height = int(cfg["width"]), int(cfg["height"])
    lh, lw = height // 16, width // 16
    audio_channels, audio_t = 2, 2

    with _lock:
        # The Qwen3-VL encoder streams one layer at a time (~1 GB working set),
        # so it coexists with the DiT: a preloaded DiT is reused, not evicted.
        context = _h3_text_encode(cfg.get("text"), prompt, progress, base=0.0, span=0.15, cancel=cancel)
        _check(cancel)

        base = _load_h3(
            cfg.get("dit") or paths.H3_DEFAULTS["dit"],
            cfg.get("vae"), _cfg_loras(cfg), progress, cfg.get("fp8", "auto"), cancel,
        )
        if _h3_error is not None:
            raise RuntimeError(f"MiniMax H3 load failed: {_h3_error}")
        if base is None:
            raise RuntimeError("MiniMax H3 load failed.")
        dit, vae = base["dit"], base["vae"]

        context = context.to(DEVICE, DIT_DTYPE)
        num_text_tokens = int(context.shape[0])
        layout = tuple(
            t.to(DEVICE) for t in
            h3_still_layout(num_text_tokens, lh, lw, audio_t, audio_channels)
        )

        generator = torch.manual_seed(int(cfg["seed"]))
        video_lat = torch.randn(1, 24, 1, lh, lw, generator=generator).to(DEVICE)
        audio_lat = torch.randn(1, 32, 2, audio_t, generator=generator).to(DEVICE)

        sched_v = MiniMaxH3Scheduler(H3_VIDEO_SHIFT)
        sched_a = MiniMaxH3Scheduler(H3_AUDIO_SHIFT)
        sched_v.set_timesteps(int(cfg["steps"]), device=DEVICE)
        sched_a.set_timesteps(int(cfg["steps"]), device=DEVICE)
        total = max(1, len(sched_v.timesteps))
        emit = _make_preview(cfg, preview, key="h3", out_size=(width, height))

        _report(progress, 0.15, "sampling", 0.0)
        for i, t in enumerate(sched_v.timesteps):
            _check(cancel)
            sigma_v = float(1.0 - t)
            vvel, avel = dit.forward_still(
                video_lat.to(DIT_DTYPE), audio_lat.to(DIT_DTYPE), context, layout, sigma_v,
            )
            video_lat = sched_v.step(vvel.float(), t, video_lat)
            j = min(i, len(sched_a.timesteps) - 1)
            audio_lat = sched_a.step(avel.float(), sched_a.timesteps[j], audio_lat)
            _report(progress, 0.15 + 0.7 * ((i + 1) / total), "sampling", (i + 1) / total)
            if emit is not None:
                emit(i, video_lat, total)
            if cancel is not None and cancel.keep_requested:
                break

        _report(progress, 0.9, "decoding")
        with torch.inference_mode():
            px = vae.decode(video_lat.to(VAE_DTYPE))
        out = px[0, :, 0].float().clamp(0.0, 1.0)[:, :height, :width]
        arr = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
        pil = Image.fromarray(arr, "RGB")
        _report(progress, 1.0, "done")
        return pil


def _load_fibo(source, progress=None, cancel=None):
    """Load and cache the Fibo 2.0 diffusers pipeline (see :mod:`engine.fibo`)."""
    global _fibo, _fibo_key, _fibo_error
    path = os.fspath(source) if source else None
    key = (path,)
    if _fibo is not None and _fibo_key == key:
        return _fibo
    _fibo = None
    _fibo_error = None
    _fibo_key = key
    try:
        _stage(cancel, progress, 0.05, "loading pipeline")
        pipe = load_fibo_pipeline(
            source or paths.FIBO_DEFAULTS["model"], device=DEVICE, dtype=DIT_DTYPE,
            progress=lambda frac, label: _stage(
                cancel, progress, 0.05 + 0.9 * frac, label, frac,
            ),
            cancel=cancel,
        )
        _fibo = {"pipe": pipe}
    except Cancelled:
        _fibo_key = None
        raise
    except Exception as exc:
        _fibo_error = exc
    return _fibo


def _generate_fibo(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
                   regions_list=None, ref_images=None, preview=None):
    """Fibo 2.0 text-to-image / image edit / masked edit.

    Runs diffusers' :class:`BriaFibo2Pipeline` directly: ``image`` is the source
    to edit (with ``mask`` restricting the edit to the masked region) and
    ``ref_images`` are extra references for a multi-image edit (up to four in
    total, referred to as ``<image_1>`` ... in the prompt). The pipeline's own
    sampler/scheduler/CFG are used, so ``cfg["sampler"]``/``cfg["scheduler"]``
    do not apply; ``cfg["steps"]`` and ``cfg["cfg"]`` do. Live preview is not
    supported for this family.
    """
    base = _load_fibo(cfg.get("model") or paths.FIBO_DEFAULTS["model"], progress, cancel)
    _check(cancel)
    if _fibo_error is not None:
        raise RuntimeError(f"Fibo 2.0 load failed: {_fibo_error}")
    if base is None:
        raise RuntimeError("Fibo 2.0 load failed.")
    pipe = base["pipe"]

    # Fibo 2.0 is trained at multiples of 16, so snap the requested size.
    width = max(16, int(round(int(cfg["width"]) / 16)) * 16)
    height = max(16, int(round(int(cfg["height"]) / 16)) * 16)

    images = [image] if image is not None else []
    images += [im for im in (ref_images or []) if im is not None]
    images = images or None
    # A mask only applies to a single edited image (the pipeline rejects it with
    # several), so it is dropped when references are present.
    edit_mask = mask if (image is not None and len(images) == 1) else None

    steps = max(1, int(cfg["steps"]))
    negative = cfg.get("negative") or None
    generator = torch.manual_seed(int(cfg["seed"]))

    with _lock:
        _report(progress, 0.15, "sampling", 0.0)

        def on_step(_pipe, step, _timestep, kwargs):
            _check(cancel)
            frac = (step + 1) / steps
            _report(progress, 0.15 + 0.8 * min(frac, 1.0), "sampling", min(frac, 1.0))
            return kwargs

        try:
            result = pipe(
                prompt=prompt or "",
                image=images,
                mask=edit_mask,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=float(cfg.get("cfg", 5.0)),
                generator=generator,
                callback_on_step_end=(
                    on_step if (progress is not None or cancel is not None) else None
                ),
            )
        except Cancelled:
            # A cancel raised from the step callback leaves the pipeline's CPU
            # offload hooks attached; release them before the exception surfaces.
            try:
                pipe.maybe_free_model_hooks()
            except Exception:
                pass
            raise
        _report(progress, 1.0, "done")
        return result.images[0]

def generate(prompt, config=None, image=None, progress=None, cancel=None, mask=None,
             control=None, regions_list=None, ref_images=None, reference=None,
             preview=None):
    """Generate a PIL image for ``prompt`` (txt2img, or img2img when ``image``).

    ``mask`` is an optional document-sized 8-bit selection image; with it the run
    is a masked inpaint and the feather radius comes from ``config["feather"]``
    (document pixels, applied at latent scale). The mask is ignored without an
    ``image``, since there is no source latent to preserve.

    ``control`` is the control hint image (or a list of them, one per entry in
    ``config["controlnets"]``): ControlNet hints for the ``stable_diffusion`` and
    ``z_image`` families and Anima ControlNet-LLLite hints for ``anima``. Each is
    preprocessed into the hint the adapter reads.

    ``regions_list`` is an optional list of ``{"prompt", "mask", "strength"}``
    entries for regional prompting: each region's prompt is conditioned only on
    the image tokens inside its mask, scaled by ``strength`` (1.0 is neutral)
    relative to the base prompt (see :mod:`engine.regions`). Supported by every
    family.

    ``ref_images`` is an optional list of PIL reference images for the
    ``qwen_image`` family's edit-style conditioning (see :func:`_generate_qwen`);
    it is rejected for every other family.

    ``reference`` is an optional PIL reference image for the ``anima`` family's
    reference adapter (character/style transfer from one image, see
    :func:`_generate`); it is rejected for every other family. The adapter LoRA
    path and the guidance settings come from ``config`` (``anima_ref_lora``,
    ``anima_ref_strength``, ``ref_guidance``, ``ref_correction``,
    ``ref_rescale_phi``, ``ref_offset_h``). A v2 (asymmetric) adapter also reads
    ``ref_megapixels`` for the reference's encode size.

    ``cancel`` is an optional cancellation token; when tripped, the job raises
    :class:`engine.cancel.Cancelled` and no image is returned.
    """
    cfg = dict(paths.DEFAULTS)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    tuning.apply()
    family = cfg.get("family") or "anima"
    if image is None:
        mask = None
    _check(cancel)
    with torch.no_grad():
        if family == "stable_diffusion":
            _reject_ref_images(family, ref_images)
            _reject_reference(family, reference)
            return _generate_classic(prompt, cfg, image, progress, cancel, mask,
                                     control, regions_list, preview)
        if family == "qwen_image":
            _reject_reference(family, reference)
            return _generate_qwen(prompt, cfg, image, progress, cancel, mask,
                                  regions_list, ref_images, preview)
        if family == "flux":
            _reject_ref_images(family, ref_images)
            _reject_reference(family, reference)
            return _generate_flux(prompt, cfg, image, progress, cancel, mask,
                                  regions_list, preview)
        if family == "z_image":
            _reject_ref_images(family, ref_images)
            _reject_reference(family, reference)
            return _generate_z(prompt, cfg, image, progress, cancel, mask, control,
                               regions_list, preview)
        if family == "krea2":
            _reject_ref_images(family, ref_images)
            _reject_reference(family, reference)
            return _generate_krea2(prompt, cfg, image, progress, cancel, mask,
                                   regions_list, preview)
        if family == "moss_image":
            _reject_ref_images(family, ref_images)
            _reject_reference(family, reference)
            return _generate_moss(prompt, cfg, image, progress, cancel, mask,
                                  regions_list, preview)
        if family == "h3":
            _reject_ref_images(family, ref_images)
            _reject_reference(family, reference)
            return _generate_h3(prompt, cfg, image, progress, cancel, mask,
                                regions_list, preview)
        if family == "fibo":
            _reject_reference(family, reference)
            return _generate_fibo(prompt, cfg, image, progress, cancel, mask,
                                  regions_list, ref_images, preview)
        _reject_ref_images(family, ref_images)
        return _generate(prompt, cfg, image, progress, cancel, mask, control,
                         regions_list, reference, preview)


def _reject_ref_images(family, ref_images):
    """Guard: only ``qwen_image`` accepts reference images."""
    if ref_images:
        raise RuntimeError(
            f"Reference images are only supported by the qwen_image family, not {family}."
        )


def _reject_reference(family, reference):
    """Guard: only ``anima`` accepts a reference image."""
    if reference is not None:
        raise RuntimeError(
            f"The reference adapter is only supported by the anima family, not {family}."
        )


def preload(config=None, progress=None, cancel=None):
    """Load ``config``'s engine into the memory cache without generating.

    Returns a label for the loaded family. Raises on load failure the same way
    :func:`generate` does, so the caller can report it. A cancel requested while
    the load runs is honoured at the next phase boundary (or loader tick): the
    load raises ``Cancelled`` and no partially-loaded engine is cached.
    """
    cfg = dict(paths.DEFAULTS)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    tuning.apply()
    family = cfg.get("family") or "anima"
    _check(cancel)
    with torch.no_grad():
        label = _preload_family(family, cfg, progress, cancel)
    _check(cancel)
    return label


def _preload_family(family, cfg, progress, cancel=None):
    """Load one family's engine and return its label (see :func:`preload`)."""
    if family == "stable_diffusion":
        path = cfg.get("checkpoint")
        if not path:
            raise RuntimeError("No classic checkpoint selected.")
        _load_classic(
            path, _cfg_loras(cfg), progress, _cfg_controlnets(cfg, "controlnet"),
            cancel,
        )
        if _classic_error is not None:
            raise RuntimeError(f"Checkpoint load failed: {_classic_error}")
        return "Checkpoint"
    if family == "qwen_image":
        _load_qwen(
            cfg.get("dit") or paths.QWEN_DEFAULTS["dit"], cfg.get("text"),
            cfg.get("vae"), _cfg_loras(cfg), progress, cancel,
            fp8=cfg.get("fp8", "auto"),
        )
        if _qwen_error is not None:
            raise RuntimeError(f"Qwen-Image load failed: {_qwen_error}")
        return "Qwen-Image 2.1"
    if family == "flux":
        _load_flux(
            cfg.get("dit") or paths.FLUX_DEFAULTS["dit"], cfg.get("clip"),
            cfg.get("t5"), cfg.get("vae"), _cfg_loras(cfg), progress,
            cfg.get("fp8", "auto"), cancel,
        )
        if _flux_error is not None:
            raise RuntimeError(f"Flux load failed: {_flux_error}")
        return "Flux"
    if family == "z_image":
        _load_z(
            cfg.get("dit") or paths.ZIMAGE_DEFAULTS["dit"], cfg.get("text"),
            cfg.get("vae"), _cfg_loras(cfg), progress, cfg.get("fp8", "auto"), cancel,
        )
        if _z_error is not None:
            raise RuntimeError(f"Z-Image load failed: {_z_error}")
        specs = _cfg_controlnets(cfg, "controlnet")
        if specs and _z is not None:
            _attach_zcontrol(_z["dit"], specs[0][0], progress, cancel)
        return "Z-Image"
    if family == "krea2":
        _load_krea2(
            cfg.get("dit") or paths.KREA2_DEFAULTS["dit"], cfg.get("text"),
            cfg.get("vae"), _krea2_loras(cfg), progress, cfg.get("fp8", "auto"), cancel,
        )
        if _krea2_error is not None:
            raise RuntimeError(f"Krea 2 load failed: {_krea2_error}")
        return "Krea 2"
    if family == "moss_image":
        _load_moss(
            cfg.get("dit") or paths.MOSS_DEFAULTS["dit"], cfg.get("text"),
            cfg.get("vae"), _cfg_loras(cfg), progress, cfg.get("fp8", "auto"),
            cfg.get("text_repo"), cancel,
        )
        if _moss_error is not None:
            raise RuntimeError(f"MOSS Image load failed: {_moss_error}")
        return "MOSS Image"
    if family == "h3":
        # Only DiT + VAE are preloaded; the Qwen3-VL-32B encoder is streamed per
        # generation (see engine.h3_text), so a preloaded DiT stays resident.
        _load_h3(
            cfg.get("dit") or paths.H3_DEFAULTS["dit"],
            cfg.get("vae"), _cfg_loras(cfg), progress, cfg.get("fp8", "auto"), cancel,
        )
        if _h3_error is not None:
            raise RuntimeError(f"MiniMax H3 load failed: {_h3_error}")
        return "MiniMax H3"
    if family == "fibo":
        _load_fibo(cfg.get("model") or paths.FIBO_DEFAULTS["model"], progress, cancel)
        if _fibo_error is not None:
            raise RuntimeError(f"Fibo 2.0 load failed: {_fibo_error}")
        return "Fibo 2.0"
    _load_base(
        cfg.get("model") or paths.DEFAULTS["model"], cfg.get("clip"),
        cfg.get("vae"), _cfg_loras(cfg), progress, cancel,
    )
    if _base_error is not None:
        raise RuntimeError(f"Model load failed: {_base_error}")
    selection = _cfg_controlnets(cfg, "lllite")
    if selection and _base is not None:
        _apply_lllite_stack(_base["dit"], [spec[0] for spec in selection], progress, cancel)
    return "Anima"


def _anima_region_contexts(clip, adapter, base_prompt, region_entries, negative):
    """Context blocks for regional prompting: base prompt, then one per region.

    Returns ``(pos_blocks, neg_blocks)``; the negative side repeats the run's
    negative prompt for every block, since the regions share one negative.
    """
    def encode(text):
        qwen, t5, weights = clip.encode(text or "")
        return preprocess_text_embeds(adapter, qwen, t5, weights)

    pos_blocks = [encode(base_prompt)]
    for region in region_entries:
        pos_blocks.append(encode(region.get("prompt") or ""))
    neg_block = encode(negative or "")
    neg_blocks = [neg_block for _ in range(len(region_entries) + 1)]
    return pos_blocks, neg_blocks


def _generate(prompt, config=None, image=None, progress=None, cancel=None, mask=None,
              control=None, regions_list=None, reference=None, preview=None):
    cfg = dict(paths.DEFAULTS)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})

    base = _load_base(
        cfg.get("model") or paths.DEFAULTS["model"],
        cfg.get("clip"), cfg.get("vae"), _cfg_loras(cfg), progress, cancel,
    )
    _check(cancel)
    if _base_error is not None:
        raise RuntimeError(f"Model load failed: {_base_error}")

    dit, adapter = base["dit"], base["adapter"]
    clip, vae = base["clip"], base["vae"]
    ms, latent_format = base["ms"], base["latent"]

    width, height = int(cfg["width"]), int(cfg["height"])
    lh, lw = height // 8, width // 8
    ph, pw = lh + (lh % 2), lw + (lw % 2)

    # ControlNet-LLLite: patch the DiT with each adapter and hand it a hint
    # derived from its source image. The conditioning trunk downsamples by 16, so
    # the hint is fed at pixel resolution (latent x8) and lands exactly on the
    # token grid. Several adapters may be stacked and each is scheduled by its
    # own start/end fraction during sampling.
    specs = _cfg_controlnets(cfg, "lllite")
    lllite_entries = []
    if specs:
        if control is None:
            raise RuntimeError(
                "A control adapter needs a source image; set the panel's Source."
            )
        hints = _as_image_list(control, len(specs))
        adapters = _apply_lllite_stack(dit, [spec[0] for spec in specs], progress, cancel)
        for lllite_adapter, spec, hint_image in zip(adapters, specs, hints):
            if hint_image is None:
                raise RuntimeError(
                    "A control adapter needs a source image; set the panel's Source."
                )
            _report(progress, 0.12, "preparing control image")
            hint = preprocess.prepare(hint_image, spec[2], (pw * 8, ph * 8))
            cond_mask = None
            if getattr(lllite_adapter, "cond_in_channels", 3) == 4:
                # Inpainting adapters read RGB+mask; the painted inpaint mask is
                # the fourth channel, at the same resolution as the hint.
                if mask is None:
                    raise RuntimeError(
                        "This inpainting control adapter needs a painted inpaint "
                        "mask; paint one before generating."
                    )
                cond_mask = mask.convert("L").resize((pw * 8, ph * 8), Image.NEAREST)
            lllite_adapter.set_cond_image(_lllite_cond_tensor(
                hint, cond_mask,
                getattr(lllite_adapter, "inpaint_masked_input", False),
            ))
            lllite_adapter.set_multiplier(spec[1])
            lllite_entries.append(
                {"adapter": lllite_adapter, "start": spec[3], "end": spec[4]}
            )

    region_entries = [r for r in (regions_list or []) if r.get("mask") is not None]
    if reference is not None and region_entries:
        raise RuntimeError(
            "The Anima reference adapter cannot be combined with regional prompts."
        )
    region_mask = None
    with _lock:
        _report(progress, 0.0, "encoding prompt")
        qwen_pos, t5_pos, t5w_pos = clip.encode(prompt or "")
        ctx_pos_plain = preprocess_text_embeds(adapter, qwen_pos, t5_pos, t5w_pos)
        qwen_neg, t5_neg, t5w_neg = clip.encode(cfg.get("negative") or "")
        ctx_neg_plain = preprocess_text_embeds(adapter, qwen_neg, t5_neg, t5w_neg)
        if region_entries:
            pos_blocks, neg_blocks = _anima_region_contexts(
                clip, adapter, prompt, region_entries, cfg.get("negative"),
            )
            ctx_pos = torch.cat(pos_blocks, dim=1)
            ctx_neg = torch.cat(neg_blocks, dim=1)
            table = regions.token_regions(region_entries, pw // 2, ph // 2)
            region_mask = regions.cross_attn_mask(
                table, [block.shape[1] for block in pos_blocks], DEVICE,
                biases=regions.region_biases(region_entries),
            )
        else:
            ctx_pos = ctx_pos_plain
            ctx_neg = ctx_neg_plain

        _report(progress, 0.1, "preparing latent")
        if image is not None:
            latent = _pil_to_latent(vae, image, lh, lw)
        else:
            latent = torch.zeros(1, 16, 1, lh, lw)
        if ph != lh or pw != lw:
            latent = torch.nn.functional.pad(latent, (0, pw - lw, 0, ph - lh, 0, 0))

        latent_in = latent_format.process_in(latent) if torch.count_nonzero(latent) else latent
        latent_in = latent_in.to(DEVICE)

        sigmas = scheduler_sigmas(
ms, int(cfg["steps"]), float(cfg["denoise"]), cfg.get("scheduler", "simple"), cfg
    ).to(DEVICE)
        generator = torch.manual_seed(int(cfg["seed"]))
        noise = torch.randn(latent.shape, generator=generator, dtype=latent.dtype, device="cpu")

        # Anima reference adapter: VAE-encode the reference image as a clean
        # offset-RoPE reference frame and run the ref-aware guider. The adapter
        # LoRA is gated, so it only fires while the reference is attached.
        ref_guider = None
        ref_adapter = None
        if reference is not None:
            ref_adapter = _anima_ref_adapter(
                dit, cfg.get("anima_ref_lora"),
                float(cfg.get("anima_ref_strength", 1.0) or 1.0), progress, cancel,
            )
            # Free the previous run's reference keys/values before this one.
            if ref_adapter is not None:
                ref_adapter.ref_cache.clear()
            asym = ref_adapter is not None and ref_adapter.asym
            _report(progress, 0.12, "encoding reference")
            if asym:
                # v2 reads the reference at any size, so it is encoded at its own
                # aspect ratio to about ref_megapixels, not the run size.
                ref_latent = _anima_ref_latent_area(
                    vae, reference, float(cfg.get("ref_megapixels", 1.0) or 1.0),
                    latent_format,
                ).to(DEVICE)
            else:
                ref_latent = _anima_ref_latent(vae, reference, lh, lw, latent_format)
                if ph != lh or pw != lw:
                    ref_latent = torch.nn.functional.pad(
                        ref_latent, (0, pw - lw, 0, ph - lh, 0, 0)
                    )
                ref_latent = ref_latent.to(DEVICE)
            model = AnimaRefApplyModel(
                dit, DIT_DTYPE, lllite=lllite_entries, cross_attn_mask=region_mask,
                adapter=ref_adapter, offset=float(cfg.get("ref_offset_h", 1.0) or 1.0),
            )
            ctx_pos = AnimaRefConditioning(ctx_pos, ref_latent)
            ctx_neg = AnimaRefConditioning(ctx_neg, ref_latent)
            ref_guider = functools.partial(
                anima_ref_guider,
                ref_guidance=float(cfg.get("ref_guidance", 4.5)),
                correction=cfg.get("ref_correction", "apg"),
                rescale_phi=float(cfg.get("ref_rescale_phi", 0.5)),
            )
        else:
            model = AnimaApplyModel(
                dit, DIT_DTYPE, lllite=lllite_entries, cross_attn_mask=region_mask,
            )
        total = max(1, len(sigmas) - 1)
        emit = _make_preview(cfg, preview, latent_format, out_size=(width, height))

        def on_step(i, denoised, *_):
            _check(cancel)
            _report(progress, 0.15 + 0.75 * (i / total), "sampling", i / total)
            if emit is not None:
                emit(i, denoised, total)

        _report(progress, 0.15, "sampling", 0.0)
        try:
            kwargs = {}
            if ref_guider is not None:
                kwargs["guider"] = ref_guider
            samples = sample(
                ms, model, noise.to(DEVICE), latent_in, sigmas, ctx_pos, ctx_neg,
                float(cfg["cfg"]), sampler=cfg.get("sampler", "euler"),
                callback=on_step if (progress is not None or cancel is not None
                                     or emit is not None) else None,
                mask=mask, feather=float(cfg.get("feather") or 0.0),
                cancel=cancel, **kwargs,
            )
        finally:
            for entry in lllite_entries:
                entry["adapter"].clear_cond_image()
                entry["adapter"].set_active(True)
            if ref_adapter is not None and getattr(ref_adapter, "asym", False):
                ref_adapter.ref_cache.clear()
        if _upscale_spec(cfg) is not None:
            # The ControlNet-LLLite hints and the regional mask are sized for the
            # pass-1 token grid, so the second pass runs plain (see :func:`_hires_fix`).
            # The reference frame is resized to the new grid by the forward, so
            # the ref-aware model and contexts carry over.
            model2 = AnimaRefApplyModel(
                dit, DIT_DTYPE, adapter=ref_adapter,
                offset=float(cfg.get("ref_offset_h", 1.0) or 1.0),
            ) if reference is not None else AnimaApplyModel(dit, DIT_DTYPE)
            hires_ctx = (ctx_pos, ctx_neg) if reference is not None else (
                ctx_pos_plain, ctx_neg_plain
            )
            samples = _hires_fix(
                cfg, samples, ms, model2, hires_ctx[0], hires_ctx[1],
                cfg.get("sampler", "euler"), int(cfg["seed"]), progress, cancel,
                lambda lat: _latent_to_pil(
                    vae, latent_format.process_out(lat),
                    int(lat.shape[-2]) * 8, int(lat.shape[-1]) * 8,
                ),
                lambda pil, h, w: latent_format.process_in(_pil_to_latent(vae, pil, h, w)),
                guider=ref_guider,
            )
            if ref_adapter is not None and getattr(ref_adapter, "asym", False):
                ref_adapter.ref_cache.clear()
            _report(progress, 1.0, "done")
            return _latent_to_pil(
                vae, latent_format.process_out(samples),
                int(samples.shape[-2]) * 8, int(samples.shape[-1]) * 8,
            )
        samples = latent_format.process_out(samples.to(torch.float32))

        _report(progress, 0.95, "decoding")
        pil = _latent_to_pil(vae, samples, height, width)
        _report(progress, 1.0, "done")
        return pil


def _classic_pixels_to_latent(ckpt, image, lh, lw, generator):
    img = image.convert("RGB").resize((lw * 8, lh * 8), Image.LANCZOS)
    arr = np.asarray(img).astype("float32") / 255.0
    pixels = torch.from_numpy(arr)[None].permute(0, 3, 1, 2).to(DEVICE, CLASSIC_VAE_DTYPE)
    pixels = pixels * 2.0 - 1.0
    with torch.no_grad():
        latent = ckpt.vae.encode(pixels).latent_dist.mean
    return latent.float() * ckpt.scale_factor


def _classic_latent_to_pil(ckpt, latent, height, width):
    with torch.inference_mode():
        out = _vae_decode(ckpt.vae, (latent / ckpt.scale_factor).to(DEVICE, CLASSIC_VAE_DTYPE), sample=True)
    out = torch.clamp((out.float() + 1.0) / 2.0, 0.0, 1.0)[0]
    out = out[:, :height, :width]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, "RGB")


def _classic_region_embeds(ckpt, base_prompt, region_texts):
    """Concatenated (base + per-region) text conditioning for regional prompting.

    Each block is one 77-token SD prompt embedding; the pooled vector comes from
    the base prompt, since it is a single global conditioning.
    """
    base = ckpt.encode(base_prompt or "")
    embeds = [base["prompt_embeds"]]
    for text in region_texts:
        embeds.append(ckpt.encode(text or "")["prompt_embeds"])
    return {
        "prompt_embeds": torch.cat(embeds, dim=1),
        "pooled": base.get("pooled"),
    }


def _generate_classic(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
                      control=None, regions_list=None, preview=None):
    """SD 1.x / SDXL txt2img / img2img from a classic checkpoint.

    With ``config["controlnets"]`` set the run is guided by one or more
    ControlNets: each ``control`` image is preprocessed into its hint and the
    ControlNets' residuals are summed into the UNet at every step (see
    ``sampler.UNetApplyModel``).

    With ``regions_list`` set, each region's prompt is conditioned only on the
    latent tokens inside its mask (regional prompting): the region prompts are
    concatenated after the base prompt and a per-layer cross-attention mask
    routes each token to its region's block.
    """
    path = cfg.get("checkpoint")
    if not path:
        raise RuntimeError("No classic checkpoint selected.")

    specs = _cfg_controlnets(cfg, "controlnet")
    ckpt = _load_classic(path, _cfg_loras(cfg), progress, specs, cancel)
    _check(cancel)
    if _classic_error is not None:
        raise RuntimeError(f"Checkpoint load failed: {_classic_error}")
    if ckpt is None:
        raise RuntimeError("Checkpoint load failed.")

    width, height = int(cfg["width"]), int(cfg["height"])
    lh, lw = height // 8, width // 8
    steps = int(cfg["steps"])
    denoise = float(cfg["denoise"])
    ms = ModelSamplingDiscrete()
    latent_format = ClassicLatent(ckpt.scale_factor)

    control_entries = []
    if ckpt.controlnets:
        if control is None:
            raise RuntimeError(
                "ControlNet needs a source image; set the panel's Source."
            )
        hints = _as_image_list(control, len(ckpt.controlnets))
        for (control_model, strength, preprocessor, start, end), hint_image in zip(
            ckpt.controlnets, hints
        ):
            if hint_image is None:
                raise RuntimeError(
                    "ControlNet needs a source image; set the panel's Source."
                )
            _report(progress, 0.08, "preparing control image")
            hint = preprocess.prepare(hint_image, preprocessor, (width, height))
            control_entries.append({
                "model": control_model,
                "cond": _control_tensor(hint),
                "scale": strength,
                "start": start,
                "end": end,
            })

    region_entries = [r for r in (regions_list or []) if r.get("mask") is not None]
    previous_processors = None
    with _lock:
        _report(progress, 0.0, "encoding prompt")
        ctx_pos_plain = ckpt.encode(prompt or "")
        ctx_neg_plain = ckpt.encode(cfg.get("negative") or "")
        if region_entries:
            ctx_pos = _classic_region_embeds(
                ckpt, prompt, [r.get("prompt") or "" for r in region_entries],
            )
            negative = cfg.get("negative") or ""
            ctx_neg = _classic_region_embeds(
                ckpt, negative, [negative] * len(region_entries),
            )
            block_length = ctx_pos["prompt_embeds"].shape[1] // (len(region_entries) + 1)
            provider = regions.unet_mask_provider(
                region_entries, lw, lh, [block_length] * (len(region_entries) + 1),
                DEVICE,
            )
            previous_processors = install_region_processors(ckpt.unet, provider)
        else:
            ctx_pos = ctx_pos_plain
            ctx_neg = ctx_neg_plain
        time_ids = ckpt.time_ids(height, width)

        _report(progress, 0.1, "preparing latent")
        generator = torch.manual_seed(int(cfg["seed"]))
        if image is not None:
            latent = _classic_pixels_to_latent(ckpt, image, lh, lw, generator)
        else:
            latent = torch.zeros(1, 4, lh, lw)
        latent_in = latent_format.process_in(latent).to(DEVICE)

        sigmas = scheduler_sigmas(ms, steps, denoise, cfg.get("scheduler", "normal"), cfg).to(DEVICE)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")

        model = UNetApplyModel(
            ckpt.unet, ms, dtype=UNET_DTYPE, time_ids=time_ids,
            controlnets=control_entries,
        )
        total = max(1, len(sigmas) - 1)
        emit = _make_preview(cfg, preview, latent_format, out_size=(width, height))

        def on_step(i, denoised, *_):
            _check(cancel)
            _report(progress, 0.15 + 0.75 * (i / total), "sampling", i / total)
            if emit is not None:
                emit(i, denoised, total)

        _report(progress, 0.15, "sampling", 0.0)
        try:
            samples = sample(
                ms, model, noise.to(DEVICE), latent_in, sigmas, ctx_pos, ctx_neg,
                float(cfg["cfg"]), sampler=cfg.get("sampler", "euler"),
                callback=on_step if (progress is not None or cancel is not None
                                     or emit is not None) else None,
                generator=generator,
                mask=mask, feather=float(cfg.get("feather") or 0.0),
                cancel=cancel,
            )
        finally:
            if previous_processors is not None:
                clear_region_processors(ckpt.unet, previous_processors)
        if _upscale_spec(cfg) is not None:
            # The ControlNets and the regional mask are sized for the pass-1
            # token grid, so the second pass runs plain (see :func:`_hires_fix`).
            model2 = UNetApplyModel(
                ckpt.unet, ms, dtype=UNET_DTYPE, time_ids=time_ids,
            )
            samples = _hires_fix(
                cfg, samples, ms, model2, ctx_pos_plain, ctx_neg_plain,
                cfg.get("sampler", "euler"), int(cfg["seed"]), progress, cancel,
                lambda lat: _classic_latent_to_pil(
                    ckpt, latent_format.process_out(lat),
                    int(lat.shape[-2]) * 8, int(lat.shape[-1]) * 8,
                ),
                lambda pil, h, w: latent_format.process_in(
                    _classic_pixels_to_latent(ckpt, pil, h, w, torch.manual_seed(int(cfg["seed"])))
                ),
            )
            _report(progress, 1.0, "done")
            return _classic_latent_to_pil(
                ckpt, latent_format.process_out(samples),
                int(samples.shape[-2]) * 8, int(samples.shape[-1]) * 8,
            )
        samples = latent_format.process_out(samples.to(torch.float32))

        _report(progress, 0.95, "decoding")
        pil = _classic_latent_to_pil(ckpt, samples, height, width)
        _report(progress, 1.0, "done")
        return pil


def _clear(*slots):
    """Drop the named cache slots; returns how many engines went away."""
    dropped = 0
    for slot in slots:
        engine, key, error = _SLOTS[slot]
        if globals()[engine] is not None:
            dropped += 1
        globals()[engine] = None
        globals()[key] = None
        globals()[error] = None
    return dropped


def unload(key=None):
    """Drop cached generation engines; returns how many families went away.

    With no ``key`` every family is dropped. With a family key as published by
    :func:`loaded_models` (for example ``"classic"``) only that family's cache is
    dropped, leaving the others loaded.

    The caller must ensure no generation is running: a job reads the caches
    directly while it works, so clearing them under it would fail the job.
    """
    with _lock:
        if key is None:
            from . import controlnet as controlnet_module

            controlnet_module.clear()
            _reset_lllite()
            _reset_zcontrol()
            return _clear(*_SLOTS)
        if key in _SLOTS:
            if key == "classic":
                from . import controlnet as controlnet_module

                controlnet_module.clear()
            if key == "base":
                _reset_lllite()
            if key == "z":
                _reset_zcontrol()
            return _clear(key)
        return 0


def loaded_models():
    """Describe the generation engines currently cached in memory, for the UI.

    Returns one record per loaded family (see :mod:`engine.loaded_models`). The caches
    are read without ``_lock`` so the panel can refresh while a job runs: each
    cache is published as a finished object, so a reader sees either ``None`` or
    a complete engine.
    """
    records = []
    if _base is not None:
        base_components = [
            ("DiT", _base["dit"]),
            ("LLM adapter", _base["adapter"]),
            ("Text encoder", _base["clip"]),
            ("VAE", _base["vae"]),
        ]
        if _lllite_stack and _lllite_dit is _base["dit"]:
            for index, adapter in enumerate(_lllite_stack):
                label = ("ControlNet-LLLite" if len(_lllite_stack) == 1
                         else f"ControlNet-LLLite {index + 1}")
                base_components.append((label, adapter))
        records.append(record("Anima", _base_key[0] if _base_key else "", base_components,
            key="base", loras=_loras(
            _base_key[-1] if _base_key else (), _base["dit"], _base["adapter"],
        )))
    if _classic is not None:
        encoders = list(_classic.text_encoders)
        components = [("UNet", _classic.unet)]
        for index, encoder in enumerate(encoders):
            label = "Text encoder" if len(encoders) == 1 else f"Text encoder {index + 1}"
            components.append((label, encoder))
        components.append(("VAE", _classic.vae))
        for index, entry in enumerate(getattr(_classic, "controlnets", None) or []):
            label = ("ControlNet" if len(_classic.controlnets) == 1
                     else f"ControlNet {index + 1}")
            components.append((label, entry[0]))
        records.append(record(
            f"Checkpoint ({_classic.family.upper()})", _classic.path, components,
            key="classic",
            loras=_loras(_classic_key[-1] if _classic_key else (), _classic.unet),
        ))
    if _qwen is not None:
        records.append(record("Qwen-Image 2.1", _qwen_key[0] if _qwen_key else "", [
            ("Transformer", _qwen["dit"]),
            ("Text encoder", _qwen["text"]),
            ("VAE", _qwen["vae"]),
        ], key="qwen",
            loras=_loras(_qwen_key[-1] if _qwen_key else (), _qwen["dit"])))
    if _flux is not None:
        records.append(record("Flux", _flux_key[0] if _flux_key else "", [
            ("Transformer", _flux["dit"]),
            ("CLIP-L", _flux["clip"]),
            ("T5-XXL", _flux["t5"]),
            ("VAE", _flux["vae"]),
        ], key="flux",
            loras=_loras(_flux_key[-1] if _flux_key else (), _flux["dit"])))
    if _z is not None:
        z_components = [
            ("Transformer", _z["dit"]),
            ("Text encoder", _z["text"]),
            ("VAE", _z["vae"]),
        ]
        control_branch = _z["dit"].control_modules()
        if control_branch is not None:
            z_components.append(("ControlNet", control_branch))
        records.append(record("Z-Image", _z_key[0] if _z_key else "", z_components, key="z",
            loras=_loras(_z_key[-1] if _z_key else (), _z["dit"])))
    if _krea2 is not None:
        records.append(record("Krea 2", _krea2_key[0] if _krea2_key else "", [
            ("Transformer", _krea2["dit"]),
            ("Text encoder", _krea2["text"]),
            ("VAE", _krea2["vae"]),
        ], key="krea2",
            loras=_loras(_krea2_key[-1] if _krea2_key else (), _krea2["dit"])))
    if _moss is not None:
        records.append(record("MOSS Image", _moss_key[0] if _moss_key else "", [
            ("Transformer", _moss["dit"]),
            ("Text encoder", _moss["text"]),
            ("VAE", _moss["vae"]),
        ], key="moss",
            loras=_loras(_moss_key[-1] if _moss_key else (), _moss["dit"])))
    if _fibo is not None:
        pipe = _fibo["pipe"]
        records.append(record("Fibo 2.0", _fibo_key[0] if _fibo_key else "", [
            ("Transformer", getattr(pipe, "transformer", None)),
            ("Text encoder", getattr(pipe, "text_encoder", None)),
            ("VAE", getattr(pipe, "vae", None)),
        ], key="fibo", loras=()))
    return [item for item in records if item is not None]
