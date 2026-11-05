"""Image upscaling with local Real-ESRGAN models.

Three kinds of model are supported, chosen by file type:

* ``.onnx`` -- an exported ONNX graph run through ``onnxruntime``.
* ``.safetensors`` / ``.pth`` -- the original ``RRDBNet`` weights (the standard
  Real-ESRGAN / Remacri upscalers) or an ``RCAN`` (optionally the
  PixelUnshuffle variant), loaded into a small torch module defined here.
* SeedVR2 -- a one-step diffusion restorer (see :mod:`engine.seedvr2`); it is a
  diffusion model, so its file lives in the diffusion-model folder rather than
  the upscale folder, and it is paired with the SeedVR2 VAE.

No ComfyUI code is imported and nothing is downloaded. The model location is
resolved through :mod:`engine.paths` (the model settings / ``PAI_MODELS`` tree)
so it is configured like generation. Models are discovered by scanning the
upscale folder, plus any curated entries from the model settings.

Inputs are float32 RGB in ``0..1``; large images are processed in overlapping
tiles that are feathered together so memory stays bounded.
"""

import math
import os
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import paths
from . import seedvr2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tile size (in input pixels) and the overlap blended between neighbours.
TILE = 512
OVERLAP = 16

# SeedVR2 restores at whatever resolution it is given, so the resize factor is
# ours to choose; 4x matches the reference workflow.
SEEDVR2_SCALE = 4.0

_lock = threading.Lock()
_sessions = {}
_models = {}


# --------------------------------------------------------------- RRDBNet (torch)
def _conv(in_ch, out_ch, kernel=3):
    return nn.Conv2d(in_ch, out_ch, kernel, 1, kernel // 2)


class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = _conv(num_feat, num_grow_ch)
        self.conv2 = _conv(num_feat + num_grow_ch, num_grow_ch)
        self.conv3 = _conv(num_feat + 2 * num_grow_ch, num_grow_ch)
        self.conv4 = _conv(num_feat + 3 * num_grow_ch, num_grow_ch)
        self.conv5 = _conv(num_feat + 4 * num_grow_ch, num_feat)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class _RRDBNet(nn.Module):
    """The Real-ESRGAN generator (``RRDBNet``), scale 1/2/4."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                 num_grow_ch=32, num_up=2):
        super().__init__()
        self.num_up = num_up
        self.conv_first = _conv(num_in_ch, num_feat)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = _conv(num_feat, num_feat)
        self.conv_up1 = _conv(num_feat, num_feat)
        if num_up >= 2:
            self.conv_up2 = _conv(num_feat, num_feat)
        self.conv_hr = _conv(num_feat, num_feat)
        self.conv_last = _conv(num_feat, num_out_ch)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.num_up >= 2:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ------------------------------------------------------------------- RCAN (torch)
def _pad_to_multiple(x, mult, mode="reflect"):
    """Reflect-pad the last two dims up to a multiple of ``mult``."""
    if mult <= 1:
        return x
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x


class _CALayer(nn.Module):
    """Channel attention: global pool -> 1x1 down/up -> sigmoid gate."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            _conv(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            _conv(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))


class _RCAB(nn.Module):
    """Residual channel attention block: conv-relu-conv-CA with a skip."""

    def __init__(self, num_feat, kernel_size=3, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            _conv(num_feat, num_feat, kernel_size),
            nn.ReLU(inplace=True),
            _conv(num_feat, num_feat, kernel_size),
            _CALayer(num_feat, reduction),
        )

    def forward(self, x):
        return self.body(x) + x


class _ResidualGroup(nn.Module):
    def __init__(self, num_feat, num_block, kernel_size=3, reduction=16):
        super().__init__()
        blocks = [_RCAB(num_feat, kernel_size, reduction) for _ in range(num_block)]
        blocks.append(_conv(num_feat, num_feat, kernel_size))
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return self.body(x) + x


class _RCAN(nn.Module):
    """Residual Channel Attention Network (optionally PixelUnshuffle input).

    With ``unshuffle_mod`` the input is pixel-unshuffled by ``4 // scale`` and
    the network runs at ``4x`` internally, so a ``2x`` model still consumes RGB
    (an RGBA-free 0..1 float tensor) while being up to ~4x cheaper.
    """

    def __init__(self, *, scale, num_group, num_block, num_feat, num_colors,
                 kernel_size=3, reduction=16, unshuffle_mod=False):
        super().__init__()
        self.scale = scale
        self.downscale_factor = 1
        up_scale = scale
        unshuffle_mod = unshuffle_mod and scale <= 2
        if unshuffle_mod:
            self.downscale_factor = 4 // scale
            up_scale = 4
            head = [
                nn.PixelUnshuffle(self.downscale_factor),
                _conv(num_colors * self.downscale_factor ** 2, num_feat, kernel_size),
            ]
        else:
            head = [_conv(num_colors, num_feat, kernel_size)]
        groups = [_ResidualGroup(num_feat, num_block, kernel_size, reduction)
                  for _ in range(num_group)]
        groups.append(_conv(num_feat, num_feat, kernel_size))
        tail = [_Upsampler(up_scale, num_feat), _conv(num_feat, num_colors, kernel_size)]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*groups)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = _pad_to_multiple(x, self.downscale_factor, "reflect")
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return x[..., :h * self.scale, :w * self.scale]


class _Upsampler(nn.Sequential):
    """``2**n`` nearest-neighbour upsample via conv + ``PixelShuffle``."""

    def __init__(self, scale, num_feat):
        modules = []
        for _ in range(int(round(math.log2(scale)))):
            modules.append(_conv(num_feat, 4 * num_feat, 3))
            modules.append(nn.PixelShuffle(2))
        super().__init__(*modules)


def _is_rcan(sd):
    return (
        "tail.1.weight" in sd
        and "body.0.body.0.body.0.weight" in sd
        and "body.0.body.0.body.3.conv_du.0.weight" in sd
        and ("head.0.weight" in sd or "head.1.weight" in sd)
    )


def _build_rcan(sd):
    """Build an ``_RCAN`` from its state dict; return ``(model, factor)``."""
    head_index = max(int(k.split(".")[1]) for k in sd if k.startswith("head."))
    unshuffle = head_index > 0
    num_colors = int(sd["tail.1.weight"].shape[0])
    num_feat = int(sd[f"head.{head_index}.weight"].shape[0])
    kernel_size = int(sd[f"head.{head_index}.weight"].shape[-1])
    in_ch = int(sd[f"head.{head_index}.weight"].shape[1])
    reduction = num_feat // int(sd["body.0.body.0.body.3.conv_du.0.weight"].shape[0])
    groups, blocks = set(), set()
    for k in sd:
        parts = k.split(".")
        if len(parts) == 7 and parts[0] == "body" and parts[2] == "body" \
                and parts[4] == "body" and parts[6] == "weight":
            groups.add(int(parts[1]))
            if parts[1] == "0":
                blocks.add(int(parts[3]))
    if unshuffle:
        downscale_factor = int(round((in_ch / num_colors) ** 0.5))
        factor = 4 // downscale_factor
    else:
        upsamples = sum(1 for k in sd if k.startswith("tail.0.") and k.endswith(".weight"))
        factor = 2 ** upsamples
    model = _RCAN(
        scale=factor, num_group=len(groups), num_block=len(blocks), num_feat=num_feat,
        num_colors=num_colors, kernel_size=kernel_size, reduction=reduction,
        unshuffle_mod=unshuffle,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(DEVICE), factor


def _remap_esrgan(sd):
    """Translate original-ESRGAN ``model.*`` keys to our RRDBNet naming.

    The paper-era ESRGAN serialises the whole net as a flattened
    ``model`` Sequential, so its keys look like ``model.0`` (conv_first),
    ``model.1.sub.N.RDBk.convC.0`` (dense blocks), ``model.1.sub.NB``
    (conv_body), then ``model.3/6/8/10`` (conv_up1/up2/hr/last).
    """
    out = {}
    for key, value in sd.items():
        if key.startswith("model.0."):
            out["conv_first." + key[len("model.0."):]] = value
        elif key.startswith("model.1.sub."):
            parts = key[len("model.1.sub."):].split(".")
            if len(parts) == 2:  # sub.NB.{weight,bias} -> conv_body
                out["conv_body." + parts[1]] = value
            else:  # sub.N.RDBk.convC.0.{weight,bias}
                block, rdb, conv, _zero, param = parts
                out[f"body.{block}.{rdb.lower()}.{conv}.{param}"] = value
        elif key.startswith("model.3."):
            out["conv_up1." + key[len("model.3."):]] = value
        elif key.startswith("model.6."):
            out["conv_up2." + key[len("model.6."):]] = value
        elif key.startswith("model.8."):
            out["conv_hr." + key[len("model.8."):]] = value
        elif key.startswith("model.10."):
            out["conv_last." + key[len("model.10."):]] = value
    return out


def _load_torch(path):
    """Load a ``.safetensors``/``.pth`` upscaler; return ``(model, scale)``."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and isinstance(sd.get("params"), dict):
            sd = sd["params"]
    if not isinstance(sd, dict):
        raise ValueError(f"Unrecognised upscale model file: {path}")
    sd = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    if "conv_first.weight" not in sd:
        if any(k.startswith("model.0.") for k in sd):
            sd = _remap_esrgan(sd)
    if "conv_first.weight" not in sd:
        if _is_rcan(sd):
            return _build_rcan(sd)
        raise ValueError(
            "Unsupported upscale architecture (expected Real-ESRGAN RRDBNet "
            "or RCAN)."
        )
    num_feat = int(sd["conv_first.weight"].shape[0])
    num_grow_ch = int(sd["body.0.rdb1.conv1.weight"].shape[0])
    num_block = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("body."))
    num_up = 2 if "conv_up2.weight" in sd else 1
    model = _RRDBNet(
        num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch, num_up=num_up
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(DEVICE), 2 ** num_up


# ------------------------------------------------------------------- catalogue
def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _curated():
    """Curated upscale models from the model settings, keyed by label."""
    pool = paths.component_pools().get("upscale", [])
    return {item["label"]: item["path"] for item in pool}


def _catalog():
    """``{label: path}`` for every discovered upscale model (curated wins)."""
    catalog = {_stem(p): p for p in paths.list_upscale_files()}
    catalog.update(_curated())
    # SeedVR2 diffusion models live in the diffusion folder (not the upscale
    # folder) because they are diffusion models; they still run through the
    # upscale panel. Their VAE is resolved separately (see engine.seedvr2).
    for path in seedvr2.list_dit_files():
        catalog.setdefault(_seedvr2_label(path), path)
    return catalog


def _seedvr2_label(path):
    return f"SeedVR2: {_stem(path)}"


def list_models():
    """Labels of the upscale models to offer, curated and scanned."""
    return list(_catalog())


def _model_path(model_name):
    path = _catalog().get(model_name)
    if path is None:
        raise ValueError(f"Unknown upscale model: {model_name}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Upscale model not found: {path}")
    return path


def _session_for(path):
    with _lock:
        session = _sessions.get(path)
    if session is not None:
        return session

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Upscaling from an ONNX model needs the 'onnxruntime' package, "
            "which is not installed in this Python environment."
        ) from exc

    options = ort.SessionOptions()
    options.log_severity_level = 3
    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    try:
        session = ort.InferenceSession(path, sess_options=options, providers=providers)
    except Exception:
        session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])

    with _lock:
        _sessions[path] = session
    return session


def _report(progress, frac, label):
    if progress is not None:
        try:
            progress(min(max(frac, 0.0), 1.0), label)
        except Exception:
            pass


def _check(cancel):
    """Raise ``Cancelled`` when the job's token has been tripped."""
    if cancel is not None:
        cancel.throw_if_cancelled()


def _has_alpha(image):
    if image.mode != "RGBA":
        return False
    return image.getchannel("A").getextrema() != (255, 255)


def _probe_scale(session, input_name):
    """Run a small probe to learn an ONNX model's integer scale factor."""
    probe = np.zeros((1, 3, 8, 8), dtype="float32")
    out = session.run(None, {input_name: probe})[0]
    return max(1, int(round(out.shape[-1] / 8)))


def _runner_for(path):
    """Return ``(run, factor)`` where ``run`` maps an NCHW tile to NCHW output."""
    if os.path.splitext(path)[1].lower() == ".onnx":
        session = _session_for(path)
        input_name = session.get_inputs()[0].name

        def run(tile):
            return session.run(None, {input_name: tile})[0]

        try:
            factor = _probe_scale(session, input_name)
        except Exception:
            factor = 4
        return run, factor

    with _lock:
        cached = _models.get(path)
    if cached is None:
        cached = _load_torch(path)
        with _lock:
            _models[path] = cached
    model, factor = cached

    def run(tile):
        with torch.no_grad():
            out = model(torch.from_numpy(tile).to(DEVICE))
        return out.clamp(0.0, 1.0).float().cpu().numpy()

    return run, factor


def _ramp(n, pad_start, pad_end):
    """A 1.0 plateau with linear ``0->1`` ramps over the padded edges."""
    ramp = np.ones(n, dtype="float32")
    if pad_start > 0:
        ramp[:pad_start] = np.linspace(0.0, 1.0, pad_start + 2, dtype="float32")[1:-1]
    if pad_end > 0:
        ramp[n - pad_end:] = np.linspace(1.0, 0.0, pad_end + 2, dtype="float32")[1:-1]
    return ramp


def _run_tiled(run, arr, factor, progress=None, cancel=None):
    """Upscale ``arr`` (H,W,3 float 0..1) by ``factor`` in blended tiles."""
    h, w = arr.shape[:2]
    out = np.zeros((h * factor, w * factor, 3), dtype="float32")
    weight = np.zeros((h * factor, w * factor, 1), dtype="float32")
    step = max(1, TILE - 2 * OVERLAP)
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    total = len(ys) * len(xs)
    done = 0
    for y0 in ys:
        for x0 in xs:
            _check(cancel)
            y1 = min(y0 + TILE, h)
            x1 = min(x0 + TILE, w)
            py0, px0 = max(0, y0 - OVERLAP), max(0, x0 - OVERLAP)
            py1, px1 = min(h, y1 + OVERLAP), min(w, x1 + OVERLAP)
            tile = arr[py0:py1, px0:px1].transpose(2, 0, 1)[None].astype("float32")
            res = run(tile)[0].transpose(1, 2, 0)
            oy, ox = py0 * factor, px0 * factor
            wh = _ramp(res.shape[0], (y0 - py0) * factor, (py1 - y1) * factor)
            ww = _ramp(res.shape[1], (x0 - px0) * factor, (px1 - x1) * factor)
            mask = (wh[:, None] * ww[None, :])[:, :, None]
            out[oy:oy + res.shape[0], ox:ox + res.shape[1]] += res * mask
            weight[oy:oy + res.shape[0], ox:ox + res.shape[1]] += mask
            done += 1
            _report(progress, 0.1 + 0.85 * done / total, "upscaling")
    return out / np.maximum(weight, 1e-6)


def upscale(image, model_name=None, progress=None, cancel=None):
    """Return ``image`` upscaled by the chosen model.

    Any alpha already present is preserved (scaled with the RGB output). ``cancel``
    is an optional cancellation token; when tripped the job raises
    :class:`engine.cancel.Cancelled`.
    """
    _check(cancel)
    if model_name is None:
        models = list_models()
        if not models:
            raise RuntimeError(
                "No upscale model found under " + paths.upscale_dir()
            )
        model_name = models[0]

    path = _model_path(model_name)
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba.convert("RGB")).astype("float32") / 255.0
    _check(cancel)

    if seedvr2.is_seedvr2_file(path):
        out = _run_seedvr2(path, arr, progress, cancel)
    else:
        _report(progress, 0.05, "loading model")
        run, factor = _runner_for(path)
        out = _run_tiled(run, arr, factor, progress, cancel)
        if float(out.max()) > 2.0:
            out = out / 255.0
    out = np.clip(out, 0.0, 1.0)

    result = Image.fromarray((out * 255.0).round().astype("uint8"), "RGB")
    # Always hand back RGBA: results are placed on canvas layers, which are RGBA.
    # A fully opaque source reports no alpha (see ``_has_alpha``) but still needs
    # an alpha channel, or compositing the layer later fails with "images do not
    # match".
    result = result.convert("RGBA")
    if _has_alpha(rgba):
        alpha = rgba.getchannel("A").resize(result.size, Image.LANCZOS)
        result.putalpha(alpha)
    _report(progress, 1.0, "done")
    return result


def _run_seedvr2(path, arr, progress, cancel):
    """Restore ``arr`` with SeedVR2, pairing ``path`` with the SeedVR2 VAE."""
    vae_path = seedvr2.default_vae_file()
    if not vae_path or not os.path.isfile(vae_path):
        raise RuntimeError(
            "SeedVR2 needs its VAE (seedvr2_ema_vae). Put the VAE file in "
            + paths.vae_dir()
        )
    _report(progress, 0.05, "loading model")
    upscaler = seedvr2.load(path, vae_path)
    return upscaler.upscale(
        arr, scale=SEEDVR2_SCALE, seed=0, color="lab",
        progress=progress, cancel=cancel,
    )


def unload(key=None):
    """Drop cached upscale sessions/models; returns how many went away.

    With no ``key`` every cache is cleared. With a ``key`` from
    :func:`loaded_models` (``"session:<path>"`` or ``"model:<path>"``) only that
    one entry is dropped, leaving the rest loaded.

    The caller must ensure no upscale job is running (it may be using one of the
    cached runners).
    """
    with _lock:
        if key is None:
            count = len(_sessions) + len(_models)
            _sessions.clear()
            _models.clear()
            count += seedvr2.unload()
            return count
        kind, _, path = key.partition(":")
        if kind == "seedvr2":
            return seedvr2.unload(f"seedvr2:{path}")
        store = _sessions if kind == "session" else _models if kind == "model" else None
        if store is None or path not in store:
            return 0
        del store[path]
        return 1


def loaded_models():
    """Describe the upscale models cached in memory, for the UI.

    Like :mod:`engine.background`, an ONNX session reports its file size while a
    torch model reports its parameter footprint. See :mod:`engine.loaded_models` for
    the record shape.
    """
    from .loaded_models import file_size, record

    with _lock:
        sessions = list(_sessions.items())
        models = list(_models.items())
    records = []
    for path, session in sessions:
        try:
            providers = session.get_providers()
        except Exception:
            providers = []
        device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        records.append({
            "family": "Upscale (ONNX)",
            "source": os.path.basename(path),
            "key": f"session:{path}",
            "components": [{
                "name": "Session",
                "params": 0,
                "bytes": file_size(path),
                "dtype": "onnx",
                "device": device,
            }],
        })
    for path, cached in models:
        model = cached[0] if isinstance(cached, tuple) else cached
        records.append(record(
            "Upscale (Real-ESRGAN)", path, [("Model", model)], key=f"model:{path}",
        ))
    records.extend(seedvr2.loaded_models())
    return [item for item in records if item is not None]
