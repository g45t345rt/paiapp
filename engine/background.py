"""Background removal using local BiRefNet / U2Net / MODNet / RMBG models.

Two on-disk formats are supported, chosen by file extension:

* ``.onnx`` -- an exported graph run through ``onnxruntime``. The graph family
  is detected from the session's own inputs and outputs, so a BiRefNet, U2Net or
  MODNet export each gets its own input size, normalisation and output:
  BiRefNet feeds a 1024px ImageNet-normalised square and returns logits, U2Net a
  320px ImageNet-normalised square whose (sigmoid-activated) side outputs are
  read from the largest, and MODNet a dynamic, aspect-preserving ImageNet-style
  ``0.5/0.5`` input whose output is a sigmoid matte.
* ``.pth`` / ``.pt`` / ``.safetensors`` / ``.bin`` -- a PyTorch checkpoint. Its
  architecture is read from the tensor names: the BRIA RMBG-1.4 ISNet
  checkpoint loads into :mod:`engine.rmbg14`, and a BiRefNet checkpoint
  (RMBG-2.0 and the community ``BiRefNet-*`` files) into
  :mod:`engine.birefnet`.

No ComfyUI code is imported and no model is downloaded. Only ``onnxruntime`` (or
``torch``, for a checkpoint), ``numpy`` and ``Pillow`` are used on top of the
model file, and the model location is resolved through :mod:`engine.paths` (the
model settings / ``PAI_MODELS`` tree) so it is configured like generation.

The exported BiRefNet graph takes ``pixel_values`` ``[1, 3, H, W]`` and returns
``alphas`` ``[1, 1, H, W]`` logits; the same ImageNet normalisation and
sigmoid/min-max post-processing the reference export uses is applied here. U2Net
already sigmoid-activates each side output, as does MODNet's matte, so for those
only the min-max step applies.
"""

import os
import threading

import numpy as np
from PIL import Image, ImageChops

from . import paths

# ImageNet statistics used by the BiRefNet and U2Net (ONNX) exports.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Symmetric statistics the MODNet matting export expects.
_MATTING_MEAN = (0.5, 0.5, 0.5)
_MATTING_STD = (0.5, 0.5, 0.5)

# Fixed inference resolution each family was trained/exported at. MODNet resizes
# its long edge to a reference size and rounds to a multiple of 32 (dynamic axes).
_BIREFNET_SIZE = 1024
_U2NET_SIZE = 320
_MODNET_REF = 512

# Suffixes each backend can load: an ONNX graph goes to onnxruntime, an ISNet
# checkpoint (RMBG-1.4) into the engine's own torch module.
_ONNX_SUFFIXES = (".onnx",)
_TORCH_SUFFIXES = (".pth", ".pt", ".safetensors", ".bin")

# Friendly label -> path parts relative to the configured background folder.
# Only entries whose file actually exists are offered by :func:`list_models`.
# A label can be pointed at an explicit file in the model settings.
MODELS = {
    "RMBG-2.0 (BiRefNet)": ("RMBG", "RMBG-2.0", "onnx", "model.onnx"),
    "RMBG-2.0 (BiRefNet, quantized)": ("RMBG", "RMBG-2.0", "onnx", "model_quantized.onnx"),
    "BiRefNet (general)": ("birefnet-general.onnx",),
    "U2Net": ("u2net.onnx",),
    "U2Net (portrait)": ("u2net_human_seg.onnx",),
    "U2Net (lite)": ("u2netp.onnx",),
    "MODNet (portrait)": ("modnet_photographic_portrait_matting.onnx",),
}

_lock = threading.Lock()
_sessions = {}
_models = {}


class _Spec:
    """How to feed and read one ONNX background graph.

    ``size`` is the square inference resolution, or the long-edge reference size
    when ``dynamic`` is set. ``activated`` says the graph output already went
    through the final sigmoid, so :func:`mask` does not apply it twice.
    ``max_norm`` divides by the input's own maximum (the BiRefNet export's
    convention) instead of 255.
    """

    __slots__ = ("size", "mean", "std", "activated", "output", "max_norm", "dynamic")

    def __init__(self, size, mean, std, activated, output=0, max_norm=False, dynamic=False):
        self.size = size
        self.mean = mean
        self.std = std
        self.activated = activated
        self.output = output
        self.max_norm = max_norm
        self.dynamic = dynamic


def _is_onnx(path):
    """Whether ``path`` is an exported ONNX graph onnxruntime can load."""
    return os.fspath(path).lower().endswith(_ONNX_SUFFIXES)


def _is_supported(path):
    """Whether ``path`` is a file one of the two backends can load."""
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    return suffix in _ONNX_SUFFIXES + _TORCH_SUFFIXES


def _curated():
    """Curated background models from the model settings, keyed by label.

    Only files a backend can load are kept. The curated pool commonly also
    lists weights in no supported format (a BEN ``.pth``, say); offering them
    would only fail at load time.
    """
    pool = paths.component_pools().get("background", [])
    return {
        item["label"]: item["path"]
        for item in pool
        if _is_supported(item["path"])
    }


def list_models():
    """Labels of the background models to offer, curated ones first.

    The curated ``background`` pool is listed as-is (its loadable entries only);
    the built-in entries are appended when their file exists on disk.
    """
    names = list(_curated())
    for name, parts in MODELS.items():
        if name not in names and os.path.isfile(paths.background_file(name, parts)):
            names.append(name)
    return names


def _model_path(model_name):
    curated = _curated()
    if model_name in curated:
        path = curated[model_name]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Background model not found: {path}")
        return path
    parts = MODELS.get(model_name)
    if parts is None:
        raise ValueError(f"Unknown background model: {model_name}")
    path = paths.background_file(model_name, parts)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Background model not found: {path}")
    return path


def _session_for(path):
    """The cached ``onnxruntime`` session for an exported ``.onnx`` model."""
    with _lock:
        session = _sessions.get(path)
    if session is not None:
        return session

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Background removal needs the 'onnxruntime' package, which is not "
            "installed in this Python environment."
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


def _torch_module():
    """Import and return the two torch-backed background families.

    :mod:`engine.rmbg14` (RMBG-1.4 ISNet) and :mod:`engine.birefnet` (BiRefNet);
    the concrete module for a file is picked by :func:`_is_birefnet` inside
    :func:`_load_torch`. Importing either pulls in torch, which an environment
    that only runs ONNX models does not need, so it is deferred (like
    ``onnxruntime`` above).
    """
    try:
        from . import birefnet, rmbg14  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Loading a .pth/.safetensors background model needs PyTorch, which "
            "is not installed in this Python environment."
        ) from exc
    return rmbg14, birefnet


def _is_birefnet(path):
    """Whether ``path`` looks like a BiRefNet checkpoint, from its tensor names.

    A ``.safetensors`` file exposes its keys without reading the tensors, so the
    family is cheap to check. For ``torch.load`` formats the keys are only known
    after a full read, so ``None`` is returned and :func:`_load_torch` falls back
    to trying ISNet first.
    """
    if not os.fspath(path).lower().endswith(".safetensors"):
        return None
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as handle:
            keys = list(handle.keys())
    except Exception:
        return None

    def stripped(key):
        for prefix in ("module.", "_orig_mod."):
            while key.startswith(prefix):
                key = key[len(prefix):]
        return key

    return any(
        stripped(key).startswith("decoder.")
        or (stripped(key).startswith("bb.") and ".layers." in key)
        for key in keys
    )


def _load_torch(path):
    """Load a checkpoint file, returning ``(module, model)`` for its family."""
    rmbg14, birefnet = _torch_module()
    birefnet_file = _is_birefnet(path)
    order = (birefnet, rmbg14) if birefnet_file else (rmbg14, birefnet)
    last = None
    for module in order:
        try:
            return module, module.load(path)
        except ValueError as exc:
            last = exc
    raise ValueError(
        "Unsupported background model architecture: expected a BRIA RMBG-1.4 "
        "ISNet or a BiRefNet checkpoint."
    ) from last


def _model_for(path):
    """The cached ``(module, model)`` ISNet/BiRefNet pair for a checkpoint file."""
    with _lock:
        entry = _models.get(path)
    if entry is None:
        entry = _load_torch(path)
        with _lock:
            _models[path] = entry
    return entry


def _spatial(shape):
    """The positive integer ``(H, W)`` of a tensor shape, or ``None`` if dynamic."""
    if not shape or len(shape) < 4:
        return None
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return h, w
    return None


def _last_dim(output):
    """The last (spatial) dimension of an output as an int, else 0."""
    shape = getattr(output, "shape", None) or []
    dim = shape[-1] if shape else None
    return dim if isinstance(dim, int) else 0


def _spec_for(session):
    """Classify an ONNX background graph from its declared inputs and outputs.

    The order matters: a multi-output graph is U2Net (its ``d0``..``d6`` side
    outputs), an explicit ``pixel_values`` input is a BiRefNet export, and a
    fixed square input resolves by size (low-res U2Net vs high-res BiRefNet).
    Anything dynamic is treated as MODNet portrait matting.
    """
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    name = (inputs[0].name or "").lower()
    spatial = _spatial(inputs[0].shape)
    fixed = spatial[0] if spatial else None

    if len(outputs) > 1:
        index = max(range(len(outputs)), key=lambda i: _last_dim(outputs[i]))
        return _Spec(_U2NET_SIZE, _IMAGENET_MEAN, _IMAGENET_STD, True, index)

    if "pixel_values" in name:
        return _Spec(
            fixed or _BIREFNET_SIZE, _IMAGENET_MEAN, _IMAGENET_STD, False, max_norm=True
        )

    if fixed is not None and fixed <= 384:
        return _Spec(fixed, _IMAGENET_MEAN, _IMAGENET_STD, True)
    if fixed is not None and fixed >= 768:
        return _Spec(fixed, _IMAGENET_MEAN, _IMAGENET_STD, False, max_norm=True)

    return _Spec(_MODNET_REF, _MATTING_MEAN, _MATTING_STD, True, dynamic=True)


def _spec_run(session):
    """Return ``run`` mapping a PIL image to the selected ``(H, W)`` output."""
    input_name = session.get_inputs()[0].name
    spec = _spec_for(session)
    index = spec.output

    def run(image):
        outputs = session.run(None, {input_name: _preprocess(image, spec)})
        return outputs[index][0, 0]

    return run


def _runner_for(path):
    """Return ``(run, activated)`` for a background model file.

    ``run`` maps a PIL image to an ``(H, W)`` float array of model output, and
    ``activated`` says whether that output already went through the final
    sigmoid, so :func:`mask` does not apply it twice.
    """
    if _is_onnx(path):
        session = _session_for(path)
        return _spec_run(session), _spec_for(session).activated

    if os.path.splitext(path)[1].lower() in _TORCH_SUFFIXES:
        module, model = _model_for(path)

        def run(image):
            return module.predict(model, image)

        # ISNet sigmoid-activates inside forward; BiRefNet returns raw logits.
        return run, module.ACTIVATED

    raise RuntimeError(
        "Unsupported background model file: "
        f"'{path}'. Use an exported '.onnx' graph (BiRefNet, U2Net or MODNet) "
        "or an RMBG-1.4 ISNet / BiRefNet checkpoint (.pth/.safetensors)."
    )


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


def _target_size(image, spec):
    """The ``(width, height)`` to feed a spec's graph for ``image``.

    A fixed spec resizes to a square; a dynamic one (MODNet) scales the long
    edge to the reference size and rounds both edges to a multiple of 32, which
    its exported graph requires.
    """
    if not spec.dynamic:
        return spec.size, spec.size
    w, h = image.size
    scale = spec.size / max(w, h, 1)
    w = max(32, int(round(w * scale)))
    h = max(32, int(round(h * scale)))
    return max(32, w - w % 32), max(32, h - h % 32)


def _preprocess(image, spec):
    """Resize, normalise and layout a PIL image for an ONNX graph."""
    resized = image.convert("RGB").resize(_target_size(image, spec), Image.LANCZOS)
    array = np.asarray(resized).astype("float32")
    if spec.max_norm:
        array = array / max(float(array.max()), 1e-6)
    else:
        array = array / 255.0
    array = (array - np.asarray(spec.mean, "float32")) / np.asarray(spec.std, "float32")
    return array.transpose(2, 0, 1)[None].astype("float32")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def mask(image, model_name=None, progress=None, cancel=None):
    """Return a grayscale ``L`` foreground mask for ``image`` (255 = keep)."""
    _check(cancel)
    if model_name is None:
        models = list_models()
        if not models:
            raise RuntimeError(
                "No background removal model found under " + paths.background_dir()
            )
        model_name = models[0]

    path = _model_path(model_name)
    _report(progress, 0.05, "loading model")
    run, activated = _runner_for(path)

    _check(cancel)
    _report(progress, 0.25, "removing background")
    output = run(image)

    _check(cancel)
    predicted = output if activated else _sigmoid(output)
    low = float(predicted.min())
    high = float(predicted.max())
    predicted = (predicted - low) / max(high - low, 1e-6)
    predicted = np.squeeze(predicted)

    result = Image.fromarray((predicted * 255).astype("uint8"), mode="L")
    _report(progress, 0.95, "finishing")
    return result.resize(image.size, Image.LANCZOS)


def remove_background(image, model_name=None, progress=None, cancel=None):
    """Return ``image`` as RGBA with its background made transparent.

    Any alpha already present is preserved by multiplying it with the predicted
    foreground mask. ``cancel`` is an optional cancellation token; when tripped
    the job raises :class:`engine.cancel.Cancelled`.
    """
    source = image.convert("RGBA")
    foreground = mask(source, model_name=model_name, progress=progress, cancel=cancel)
    _check(cancel)
    result = source.copy()
    result.putalpha(ImageChops.multiply(source.getchannel("A"), foreground))
    _report(progress, 1.0, "done")
    return result


def restrict_to_mask(cutout, original, mask):
    """Keep ``cutout`` inside ``mask`` and restore ``original`` outside it.

    ``cutout`` is a :func:`remove_background` result and ``original`` the image
    it was computed from, both RGBA at the same size. The optional ``mask`` (an
    ``L`` image, resized if needed) selects where the cutout's alpha applies;
    outside it the original alpha is kept, so a removal can be limited to a
    selection. Returns the combined RGBA image.
    """
    source = original.convert("RGBA")
    result = cutout.convert("RGBA")
    if mask is None:
        return result
    mask = mask.convert("L")
    if mask.size != source.size:
        mask = mask.resize(source.size, Image.NEAREST)
    alpha = Image.composite(
        result.getchannel("A"), source.getchannel("A"), mask,
    )
    combined = source.copy()
    combined.putalpha(alpha)
    return combined


def unload(key=None):
    """Drop cached background sessions/models; returns how many went away.

    With no ``key`` every cache is cleared. With a ``key`` from
    :func:`loaded_models` (``"session:<path>"`` or ``"model:<path>"``) only that
    one entry is dropped, leaving the rest loaded.

    The caller must ensure no background job is running (it may be using one of
    the cached runners).
    """
    with _lock:
        if key is None:
            count = len(_sessions) + len(_models)
            _sessions.clear()
            _models.clear()
            return count
        kind, _, path = key.partition(":")
        store = _sessions if kind == "session" else _models if kind == "model" else None
        if store is None or path not in store:
            return 0
        del store[path]
        return 1


def loaded_models():
    """Describe the background models cached in memory, for the UI.

    An ONNX session does not expose its weight sizes, so it reports the model
    file and its size on disk; a torch checkpoint (ISNet or BiRefNet) reports
    its parameter footprint instead. See :mod:`engine.loaded_models` for the
    record shape.
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
            "family": "Background (ONNX)",
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
    for path, entry in models:
        module, model = entry
        family = "BiRefNet" if module.__name__.endswith("birefnet") else "ISNet"
        records.append(record(
            f"Background ({family})", path, [("Model", model)], key=f"model:{path}",
        ))
    return [item for item in records if item is not None]
