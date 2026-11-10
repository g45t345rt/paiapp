"""SeedVR2 one-step diffusion restoration, ported for local image upscaling.

This package is a port of ComfyUI's native SeedVR2 support
(``comfy/ldm/seedvr``, GPL-3.0 -- see ``CREDITS.md``): the ``NaDiT`` diffusion
transformer (:mod:`engine.seedvr2.model`), its causal video VAE
(:mod:`engine.seedvr2.vae`) and the CIELAB / wavelet / AdaIN colour-correction
helpers (:mod:`engine.seedvr2.color_fix`). Nothing here imports ComfyUI; the few
runtime helpers it needs are reimplemented in :mod:`engine.seedvr2._compat`.

SeedVR2 is a *one-step* flow-matching restorer: it is run at ``sigma = 1``
(timestep 1000) for a single Euler step, so the restored latent is simply
``noise - velocity``. The degraded image enters only through the ``condition``
latent (the VAE-encoded, padded input plus a ones mask); the text conditioning is
the fixed ``positive_conditioning`` buffer baked into the checkpoint, and CFG is
not used.

Weights are never bundled: the user downloads the diffusion model (3B or 7B, in
fp16 / fp8 / int8-convrot) and the ``seedvr2_ema_vae`` file, and they are
discovered under the configured model folders (see :mod:`engine.paths`).
"""

import math
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .. import paths
from . import color_fix
from ._compat import ops as _ops
from .constants import (
    BYTEDANCE_VAE_SCALING_FACTOR,
    SEEDVR2_LATENT_CHANNELS,
)

# Spatial padding used before VAE encoding (multiple of the 8x spatial
# downsample times the 2x DiT patch; 16 keeps the latent/2 exact).
_PAD_MULTIPLE = 16

# Upper bound on the restored (pixel) area before we shrink the resize factor.
# Keeps a huge source from asking for an unbounded single-pass diffusion run.
_MAX_PIXELS = 4096 * 4096

_lock = threading.Lock()
_models = {}

# A checkpoint's diffusion-model key prefix (ComfyUI's "model.diffusion_model.").
_KEY_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "",
)
_VAE_PREFIXES = (
    "first_stage_model.",
    "vae.",
    "model.",
    "",
)

_DIT_DTYPE = torch.bfloat16
_VAE_DTYPE = torch.float16


# --------------------------------------------------------------------- detection
def _strip(name, prefixes=_KEY_PREFIXES):
    for prefix in prefixes:
        if prefix and name.startswith(prefix):
            return name[len(prefix):]
    return name


def dit_config_from_keys(keys):
    """Return the ``NaDiT`` kwargs for a stripped key set, or ``None``.

    Mirrors ComfyUI's model detection: the 7B checkpoints are told apart by
    their shared (``mlp.all``) or separate (``mlp.vid``) block keys, and the 3B
    by its final shared block.
    """
    keys = set(keys)
    if "blocks.35.mlp.vid.proj_out.weight" in keys:  # SeedVR2 7B, separate keys
        return {
            "image_model": "seedvr2", "vid_dim": 3072, "heads": 24, "num_layers": 36,
            "mm_layers": 36, "norm_eps": 1e-5, "rope_type": "rope3d", "rope_dim": 64,
            "mlp_type": "normal",
        }
    if "blocks.35.mlp.all.proj_in_gate.weight" in keys:  # SeedVR2 7B, shared keys
        return {
            "image_model": "seedvr2", "vid_dim": 3072, "heads": 24, "num_layers": 36,
            "mm_layers": 10, "norm_eps": 1e-5, "rope_type": "rope3d", "rope_dim": 64,
            "mlp_type": "swiglu",
        }
    if "blocks.31.mlp.all.proj_in_gate.weight" in keys:  # SeedVR2 3B
        return {
            "image_model": "seedvr2", "vid_dim": 2560, "heads": 20, "num_layers": 32,
            "norm_eps": 1e-5, "mlp_type": "swiglu", "vid_out_norm": True,
        }
    return None


def is_seedvr2_file(path):
    """Whether ``path`` looks like a SeedVR2 *diffusion* file (by name).

    A cheap filename test used for catalogue scanning; the VAE file shares the
    ``seedvr2`` token, so ``vae`` is excluded.
    """
    name = os.path.basename(path or "").lower()
    if not name.endswith(".safetensors"):
        return False
    stem = name.rsplit(".", 1)[0]
    return "seedvr2" in stem and "vae" not in stem


def list_dit_files():
    """SeedVR2 diffusion files discovered under the diffusion-model folder."""
    return [p for p in paths._list_files(paths.diffusion_dir(), (".safetensors",))
            if is_seedvr2_file(p)]


# ----------------------------------------------------------------------- loading
def _safetensors_state(path, dtype, prefixes):
    """Build a ``{stripped_name: tensor}`` state dict, dequantising as needed.

    Handles plain fp16/bf16/fp32, fp8 (cast up), and ComfyUI ``int8_tensorwise``
    (optionally ``convrot``) via :func:`engine.quant_formats._dequant`.
    """
    from safetensors import safe_open
    from ..quant_formats import _dequant

    with safe_open(path, framework="pt", device="cpu") as handle:
        names = handle.keys()
        name_set = set(names)
        # Drop quant side-cars: ``X.comfy_quant`` config and an ``X.weight_scale``
        # that belongs to an int8 ``X.weight``. A real parameter that merely ends
        # in ``_scale`` (the DiT's ``attn_scale``/``out_scale``) has no base
        # tensor and must be kept.
        side_cars = set()
        for name in names:
            if name.endswith(".comfy_quant"):
                side_cars.add(name)
            elif name.endswith("_scale") and name[:-len("_scale")] in name_set:
                side_cars.add(name)
        state = {}
        for name in names:
            if name in side_cars:
                continue
            tensor = _dequant(handle, names, name)
            if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                tensor = tensor.to(dtype)
            state[_strip(name, prefixes)] = tensor
    return state


def load_dit(path, device="cpu", dtype=_DIT_DTYPE, config=None):
    """Load a SeedVR2 ``NaDiT`` from a ``.safetensors`` file.

    The architecture is detected from the checkpoint keys; ``config`` overrides
    it when supplied (used by tests). Returns ``(model, config)``.
    """
    from safetensors import safe_open
    from .model import NaDiT

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = [_strip(k) for k in handle.keys()]

    if config is None:
        config = dit_config_from_keys(keys)
        if config is None:
            raise ValueError(
                f"{os.path.basename(path)} is not a recognised SeedVR2 "
                "diffusion model."
            )

    model = NaDiT(operations=_ops, device="meta", dtype=dtype, **config)
    state = _safetensors_state(path, dtype, _KEY_PREFIXES)
    info = model.load_state_dict(state, strict=False, assign=True)
    del state
    for required in ("positive_conditioning", "negative_conditioning"):
        if required in info.missing_keys:
            raise ValueError(
                f"{os.path.basename(path)} is missing the SeedVR2 text "
                f"conditioning buffer {required!r}; it is not a complete "
                "SeedVR2 diffusion model."
            )
    model = model.to(device=device, dtype=dtype).eval()
    return model, config, info


def load_vae(path, device="cpu", dtype=_VAE_DTYPE):
    """Load the SeedVR2 causal video VAE from a ``.safetensors`` file."""
    from .vae import VideoAutoencoderKL

    model = VideoAutoencoderKL()
    state = _safetensors_state(path, dtype, _VAE_PREFIXES)
    model.load_state_dict(state, strict=False, assign=True)
    del state
    model = model.to(device=device, dtype=dtype).eval()
    model.device = torch.device(device)
    return model


# ------------------------------------------------------------------- run helpers
def _report(progress, frac, label):
    if progress is not None:
        try:
            progress(min(max(frac, 0.0), 1.0), label)
        except Exception:
            pass


def _check(cancel):
    if cancel is not None:
        cancel.throw_if_cancelled()


def _resize_lanczos(arr, size):
    """Resize an ``H, W, 3`` float ``0..1`` array with Lanczos."""
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).round().astype("uint8"), "RGB")
    img = img.resize(size, Image.LANCZOS)
    return np.asarray(img).astype("float32") / 255.0


def _pad_to_multiple(t, multiple):
    h, w = t.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t


class SeedVR2Upscaler:
    """A loaded SeedVR2 diffusion model + VAE, ready to restore an image."""

    def __init__(self, dit, vae, device, dit_config, load_info=None):
        self.dit = dit
        self.vae = vae
        self.device = torch.device(device)
        self.config = dit_config
        self.load_info = load_info

    @property
    def label(self):
        return "SeedVR2 7B" if self.config.get("vid_dim") == 3072 else "SeedVR2 3B"

    def upscale(self, arr, scale=4.0, seed=0, color="lab", progress=None, cancel=None):
        """Restore ``arr`` (``H, W, 3`` float ``0..1``); returns the same layout.

        ``scale`` is the Lanczos resize factor applied before restoration (the
        model itself is resolution-agnostic and outputs at the resized size).
        """
        _check(cancel)
        src = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float()
        h, w = src.shape[-2:]
        target_w = max(1, int(round(w * scale)))
        target_h = max(1, int(round(h * scale)))
        if target_w * target_h > _MAX_PIXELS:
            shrink = math.sqrt(_MAX_PIXELS / (target_w * target_h))
            target_w = max(1, int(round(target_w * shrink)))
            target_h = max(1, int(round(target_h * shrink)))
            _report(progress, 0.02, "capping size")

        _report(progress, 0.05, "resizing")
        resized = _resize_lanczos(arr, (target_w, target_h))
        ref = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        padded = _pad_to_multiple(ref, _PAD_MULTIPLE)
        _check(cancel)

        with torch.no_grad():
            _report(progress, 0.2, "encoding")
            img5d = padded.unsqueeze(2)  # (1, 3, 1, H, W)
            raw = self.vae.encode(img5d.to(next(self.vae.parameters()).dtype))
            latent = (raw * BYTEDANCE_VAE_SCALING_FACTOR).to(next(self.dit.parameters()).dtype)
            condition = torch.cat([latent, torch.ones_like(latent[:, :1])], dim=1)

            _check(cancel)
            _report(progress, 0.45, "restoring")
            generator = torch.Generator(device=self.device).manual_seed(int(seed) & 0x7FFFFFFF)
            x = torch.randn(latent.shape, generator=generator, device=self.device,
                            dtype=latent.dtype)
            timestep = torch.tensor([1000.0], device=self.device, dtype=torch.float32)
            velocity = self.dit(
                x, timestep, None,
                condition=condition,
                transformer_options={"cond_or_uncond": [0]},
            )
            restored = x - velocity

            _check(cancel)
            _report(progress, 0.75, "decoding")
            vae_dtype = next(self.vae.parameters()).dtype
            decoded = self.vae.decode_(
                (restored / BYTEDANCE_VAE_SCALING_FACTOR).to(vae_dtype)
            )
            decoded = decoded.to(torch.float32)

        # Decoded is (1, 3, 1, H, W) in -1..1; crop to the resized reference.
        frame = decoded[0, :, 0]  # (3, H, W)
        ref_frame = ref[0]  # (3, target_h, target_w)
        fh = min(frame.shape[-2], ref_frame.shape[-2])
        fw = min(frame.shape[-1], ref_frame.shape[-1])
        frame = frame[:, :fh, :fw]
        ref_frame = ref_frame[:, :fh, :fw]

        frame01 = frame.add(1.0).mul(0.5).clamp(0.0, 1.0)
        if color in ("lab", "wavelet", "adain"):
            _report(progress, 0.9, "color correction")
            raw_frame = frame01.mul(2.0).sub(1.0).unsqueeze(0)
            raw_ref = ref_frame.mul(2.0).sub(1.0).unsqueeze(0)
            transfer = {
                "lab": color_fix.lab_color_transfer,
                "wavelet": color_fix.wavelet_color_transfer,
                "adain": color_fix.adain_color_transfer,
            }[color]
            with torch.no_grad():
                corrected = transfer(raw_frame, raw_ref)
            frame01 = corrected.squeeze(0).add(1.0).mul(0.5).clamp(0.0, 1.0)

        # Even dimensions, matching the reference post-process.
        eh = fh - (fh % 2)
        ew = fw - (fw % 2)
        out = frame01[:, :eh, :ew].permute(1, 2, 0).cpu().numpy()
        _report(progress, 1.0, "done")
        return out


# ------------------------------------------------------------------- catalogue
def vae_files():
    """Candidate SeedVR2 VAE files under the VAE folder."""
    return [p for p in paths._list_files(paths.vae_dir(), (".safetensors",))
            if "seedvr2" in os.path.basename(p).lower()
            and "vae" in os.path.basename(p).lower()]


def default_vae_file():
    """The SeedVR2 VAE to use: curated pool first, else the folder scan."""
    pool = [
        item["path"] for item in paths.component_pools().get("vaes", [])
        if "seedvr2" in os.path.basename(item["path"]).lower()
    ]
    for path in pool:
        if os.path.isfile(path):
            return path
    files = vae_files()
    return files[0] if files else None


def is_loaded(path):
    with _lock:
        return path in _models


def load(dit_path, vae_path, device=None, dtype=_DIT_DTYPE):
    """Return a cached :class:`SeedVR2Upscaler` for the given files."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = os.path.abspath(dit_path)
    with _lock:
        cached = _models.get(key)
    if cached is not None:
        return cached
    dit, config, info = load_dit(dit_path, device=device, dtype=dtype)
    vae = load_vae(vae_path, device=device, dtype=_VAE_DTYPE)
    upscaler = SeedVR2Upscaler(dit, vae, device, config, info)
    upscaler.vae_path = os.path.abspath(vae_path)
    with _lock:
        _models[key] = upscaler
    return upscaler


def unload(key=None):
    """Drop cached SeedVR2 models; returns how many went away.

    ``key`` is a ``seedvr2:<dit path>`` cache key from :func:`loaded_models`.
    """
    with _lock:
        if key is None:
            count = len(_models)
            _models.clear()
            return count
        _, _, path = key.partition(":")
        if path in _models:
            del _models[path]
            return 1
        return 0


def loaded_models():
    """Describe cached SeedVR2 models, for the UI (see :mod:`engine.loaded_models`)."""
    from ..loaded_models import record

    with _lock:
        items = list(_models.items())
    records = []
    for dit_path, upscaler in items:
        records.append(record(
            f"Upscale ({upscaler.label})", dit_path,
            [("DiT", upscaler.dit), ("VAE", upscaler.vae)],
            key=f"seedvr2:{dit_path}",
        ))
    return [item for item in records if item is not None]
