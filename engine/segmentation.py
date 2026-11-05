"""Point-prompted object segmentation using local SAM ONNX models.

The user clicks an object and the network returns the mask of the thing under
the click. This is the Segment Anything (SAM) "click-to-segment" workflow: an
image encoder embeds the picture once, and a lightweight prompt decoder turns a
point (or points) into a mask, so repeat clicks on the same image are cheap.

Model layout: a SAM export is split into two ONNX graphs that must sit together
in the segmentation folder (see :func:`engine.paths.segmentation_dir`), named
``<label>_encoder.onnx`` and ``<label>_decoder.onnx``. This is the interface the
official ``segment-anything`` ONNX export (``scripts/export_onnx_model.py`` and
``OnnxImagePredictor``) produces:

* encoder: input ``input`` ``[1, 3, 1024, 1024]`` -> ``image_embeddings``
  ``[1, 256, 64, 64]``;
* decoder: inputs ``image_embeddings``, ``point_coords`` ``[1, N, 2]`` (in the
  encoder's 1024 frame), ``point_labels`` ``[1, N]``, ``mask_input``
  ``[1, 1, 256, 256]``, ``has_mask_input`` ``[1]`` and ``orig_im_size`` ``[2]``
  (``(height, width)``); outputs ``masks`` ``[1, 3, H, W]``, ``iou_predictions``
  ``[1, 3]`` and ``low_res_masks``.

The image is preprocessed the SAM way: RGB, resized to fit the encoder frame
(preserving aspect), placed at the top-left and padded to the frame with black.
The frame is taken from an AnyLabeling ``config.yaml`` beside the model
(``max_width``/``max_height``) when present, else from the graph's declared
input shape, else a square 1024. Raw 0..255 pixels are fed -- SAM's
normalisation is baked into the exported encoder. A point is scaled by the same
factor, so it lands in the encoder frame, and the returned mask is cropped back
out of the frame and scaled to the document.

No weights are downloaded and no ComfyUI code is imported. Only
``onnxruntime``, ``numpy`` and ``Pillow`` are used, and the model location is
resolved through :mod:`engine.paths` (the model settings / ``PAI_MODELS`` tree)
so it is configured like generation. This module holds and unloads its own
sessions, so it plugs into the Loaded Models panel the way
:mod:`engine.background` does.
"""

import hashlib
import os
import re
import threading

import numpy as np
from PIL import Image

from . import paths

# Square fallback encoder resolution, used when a graph declares symbolic dims
# and has no AnyLabeling ``config.yaml`` beside it.
ENCODER_SIZE = 1024

# The decoder's prompt mask is always 256x256 at the encoder's 64x64 grid; an
# empty one is fed when (as here) there is no earlier mask to refine.
MASK_INPUT_SIZE = 256

# A mask logit above this is foreground. Masks come back already thresholded by
# the export, so this only decides a value that is not plainly binary.
_MASK_THRESHOLD = 0.0

# Each graph is one file with one input/output, so the resolved names are cached
# on the session object (cheap to recompute, but the lookup is the hot path).
_ENCODER_SUFFIX = "_encoder.onnx"
_DECODER_SUFFIX = "_decoder.onnx"
_ONNX_SUFFIX = ".onnx"

_lock = threading.Lock()
_sessions = {}
# One cached image embedding per encoder path: ``(fingerprint, embeddings,
# scale, (width, height))``. Kept to a single entry so a click sequence on one
# image never recomputes it, while switching images cannot grow the cache.
_embeddings = {}


def _is_onnx(path):
    return os.fspath(path).lower().endswith(_ONNX_SUFFIX)


def list_models():
    """Labels of the SAM model pairs found in the segmentation folder.

    Two layouts are offered: the loose pair ``<label>_encoder.onnx`` +
    ``<label>_decoder.onnx``, and the extracted-archive folder
    ``<label>/encoder.onnx`` + ``<label>/decoder.onnx`` (what the Hub browser
    produces when it unpacks a SAM ONNX ``.zip``). Either way both graphs must be
    present, since they are used together.
    """
    directory = paths.segmentation_dir()
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    lower = {name.lower(): name for name in names}
    labels = []
    for name in sorted(names):
        if not name.lower().endswith(_ENCODER_SUFFIX):
            continue
        label = name[: -len(_ENCODER_SUFFIX)]
        if (label + _DECODER_SUFFIX).lower() in lower:
            labels.append(label)
    for name in sorted(names):
        if name in labels or not os.path.isdir(os.path.join(directory, name)):
            continue
        if _pair_in_dir(os.path.join(directory, name)) is not None:
            labels.append(name)
    return labels


def _role_file(names, role):
    """The ``.onnx`` file in ``names`` for ``role`` ("encoder"/"decoder")."""
    candidates = [
        name for name in names
        if name.lower().endswith(_ONNX_SUFFIX) and role in name.lower()
    ]
    if not candidates:
        return None
    for name in candidates:
        if name.lower() == role + _ONNX_SUFFIX:
            return name
    return sorted(candidates)[0]


def _pair_in_dir(folder):
    """``(encoder, decoder)`` for a folder holding both graphs, else ``None``."""
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    encoder = _role_file(names, "encoder")
    decoder = _role_file(names, "decoder")
    if encoder is None or decoder is None:
        return None
    return os.path.join(folder, encoder), os.path.join(folder, decoder)


def _decoder_path(encoder_path):
    """The decoder graph that pairs with ``encoder_path``.

    The ``_encoder.onnx`` / ``_decoder.onnx`` suffix convention of the export
    is reversed; a file without it (a curated single-name override) has no
    derivable partner, so ``None`` is returned.
    """
    if os.fspath(encoder_path).lower().endswith(_ENCODER_SUFFIX):
        return encoder_path[: -len(_ENCODER_SUFFIX)] + _DECODER_SUFFIX
    return None


def _resolve(label):
    """Encoder/decoder paths for ``label`` from the model settings or folder.

    The ``models.segmentation`` setting may point a label at an explicit encoder
    file; the decoder is then its ``_decoder.onnx`` sibling. Otherwise the pair
    lives under :func:`engine.paths.segmentation_dir`.
    """
    from . import settings as _settings

    section = _settings.section("models")
    overrides = section.get("segmentation")
    if isinstance(overrides, dict) and overrides.get(label):
        encoder = os.fspath(overrides[label])
        decoder = _decoder_path(encoder)
        if decoder is None:
            raise ValueError(
                f"Segmentation model '{label}' must be an '..._encoder.onnx' "
                "file so its decoder can be found."
            )
    else:
        directory = paths.segmentation_dir()
        encoder = os.path.join(directory, label + _ENCODER_SUFFIX)
        decoder = os.path.join(directory, label + _DECODER_SUFFIX)
        if not (os.path.isfile(encoder) and os.path.isfile(decoder)):
            # Fall back to the extracted-archive folder layout.
            pair = _pair_in_dir(os.path.join(directory, label))
            if pair is not None:
                encoder, decoder = pair
    if not os.path.isfile(encoder):
        raise FileNotFoundError(f"Segmentation encoder not found: {encoder}")
    if not os.path.isfile(decoder):
        raise FileNotFoundError(f"Segmentation decoder not found: {decoder}")
    return encoder, decoder


def _session_for(path):
    """The cached ``onnxruntime`` session for an ``.onnx`` graph."""
    with _lock:
        session = _sessions.get(path)
    if session is not None:
        return session

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Object segmentation needs the 'onnxruntime' package, which is not "
            "installed in this Python environment."
        ) from exc

    options = ort.SessionOptions()
    options.log_severity_level = 3
    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in available]
    try:
        session = ort.InferenceSession(path, sess_options=options, providers=providers)
    except Exception:
        session = ort.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"]
        )

    with _lock:
        _sessions[path] = session
    return session


def _normalized(image, frame_w=ENCODER_SIZE, frame_h=None):
    """Return ``(hwc, scale, content)`` for the encoder input.

    ``hwc`` is the ``[frame_h, frame_w, 3]`` image in 0..255 float: the picture
    scaled by ``min(frame_w / W, frame_h / H)`` and placed at the top-left, the
    rest filled black (the same warp the reference SAM ONNX implementations
    use). SAM's normalisation is baked into the exported encoder, so raw pixels
    are fed -- feeding mean/std-normalised data double-normalises and gives
    worse masks. ``scale`` is that factor (a point is multiplied by it to land
    in the frame) and ``content`` is the ``(new_w, new_h)`` size the image
    occupies inside the frame.
    """
    if frame_h is None:
        frame_h = frame_w
    width, height = image.size
    scale = min(frame_w / max(width, 1), frame_h / max(height, 1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.convert("RGB").resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (frame_w, frame_h), (0, 0, 0))
    canvas.paste(resized, (0, 0))
    return np.asarray(canvas).astype("float32"), scale, (new_w, new_h)


def preprocess(image, size=ENCODER_SIZE):
    """Return ``(array, scale)`` for the encoder as NCHW ``[1, 3, size, size]``."""
    hwc, scale, _content = _normalized(image, size, size)
    return hwc.transpose(2, 0, 1)[None].astype("float32"), scale


def _read_simple_yaml(path):
    """Parse the scalar ``key: value`` lines of a small YAML file.

    Only the flat scalars an AnyLabeling model ``config.yaml`` uses are needed;
    this avoids a PyYAML dependency. Values are ints/floats when they parse as
    such, else strings.
    """
    values = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip().strip("\"'")
                if not key or not value:
                    continue
                try:
                    values[key] = int(value)
                except ValueError:
                    try:
                        values[key] = float(value)
                    except ValueError:
                        values[key] = value
    except OSError:
        pass
    return values


def _config_frame(encoder_path):
    """``(frame_w, frame_h)`` from an AnyLabeling ``config.yaml``, or ``None``.

    Such an export declares its real input frame, which can be non-square (a
    ``1024x684`` graph would otherwise be fed a square and stretch the mask). The
    file sits beside the encoder. The keys used, in order, are
    ``max_width``/``max_height`` or ``input_width``/``input_height`` (AnyLabeling
    writes the former); a two-element ``input_size`` (``[w, h]``) is also
    accepted when the explicit keys are absent.
    """
    folder = os.path.dirname(encoder_path)
    for name in ("config.yaml", "config.yml"):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        values = _read_simple_yaml(path)
        width = values.get("max_width") or values.get("input_width")
        height = values.get("max_height") or values.get("input_height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)) \
                and width > 0 and height > 0:
            return int(width), int(height)
        pair = values.get("input_size")
        if isinstance(pair, str):
            numbers = [int(part) for part in re.findall(r"\d+", pair)][:2]
            if len(numbers) == 2 and numbers[0] > 3 and numbers[1] > 3:
                return numbers[0], numbers[1]
    return None


def _encoder_frame(shape, encoder_path=None, fallback=ENCODER_SIZE):
    """``(frame_w, frame_h)`` the encoder input wants.

    Preference order: an AnyLabeling ``config.yaml`` beside the model, then the
    graph's declared input shape (square ``[1, 3, 1024, 1024]`` or a fixed
    non-square frame), then a square ``fallback``.
    """
    if encoder_path is not None:
        frame = _config_frame(encoder_path)
        if frame is not None:
            return frame
    dims = list(shape or [])

    def concrete(value):
        return isinstance(value, int) and value > 3

    if len(dims) == 4:
        if concrete(dims[2]) and concrete(dims[3]):
            return int(dims[3]), int(dims[2])  # NCHW: W, H
        if dims[3] == 3 and concrete(dims[1]) and concrete(dims[2]):
            return int(dims[2]), int(dims[1])  # NHWC: W, H
    if len(dims) == 3:
        if concrete(dims[0]) and concrete(dims[1]):
            return int(dims[1]), int(dims[0])  # HWC: W, H
        if dims[0] == 3 and concrete(dims[1]) and concrete(dims[2]):
            return int(dims[2]), int(dims[1])  # CHW: W, H
    return fallback, fallback


def _encoder_inputs(hwc, shape):
    """Candidate layouts for the HWC image, most likely first.

    SAM exports disagree on layout: most use NCHW ``[1, 3, H, W]``, but some
    (the AnyLabeling-style ``input_image`` graphs) use NHWC ``[H, W, 3]`` or
    ``[1, H, W, 3]``. The channel axis is inferred from where the declared shape
    has a ``3``; when the dims are symbolic the common NCHW layout is tried
    first and the caller falls back to the other. Returning both lets
    :func:`_embed` retry the alternate layout if the graph rejects the first.
    """
    dims = list(shape or [])
    nchw = hwc.transpose(2, 0, 1)

    def chan_at(index):
        return index < len(dims) and dims[index] == 3

    if len(dims) == 4:
        if chan_at(3):
            return [hwc[None], nchw[None]]
        return [nchw[None], hwc[None]]
    if len(dims) == 3:
        if chan_at(2):
            return [hwc, nchw]
        return [nchw, hwc]
    return [nchw[None]]


def _fingerprint(image):
    """An identity for ``image`` so its embedding can be reused across clicks."""
    return (
        image.size,
        hashlib.sha1(image.convert("RGB").tobytes()).hexdigest(),
    )


def _encoder_io(session):
    """``(input_name, output_name, shape)`` of a SAM encoder graph.

    Exports differ in naming and layout: the input may be ``input``,
    ``input_image`` or ``image``; the image may be NCHW ``[1, 3, 1024, 1024]``,
    batchless CHW ``[3, 1024, 1024]`` or NHWC ``[H, W, 3]``. ``shape`` is the
    declared input shape, used both to lay the image out and to pick the frame.
    """
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    names = [item.name for item in inputs]
    in_name = next(
        (name for name in ("input", "input_image", "image", "pixel_values")
         if name in names),
        inputs[0].name,
    )
    shape = list(getattr(inputs[names.index(in_name)], "shape", []) or [])
    out_name = (
        "image_embeddings"
        if any(item.name == "image_embeddings" for item in outputs)
        else outputs[0].name
    )
    return in_name, out_name, shape


def _embed(image, encoder_path, progress=None, cancel=None):
    """The cached image embedding for ``image``, computing it if needed.

    Returns ``(embeddings, geometry)`` where ``geometry`` is
    ``(frame_w, frame_h, new_w, new_h, width, height)``: the encoder frame, the
    real image's size inside it, and the document size. The mask is mapped back
    through this in :func:`_mask_image`. The embeddings depend only on the image
    and its encoder, so a click sequence on one image embeds once.
    """
    fingerprint = _fingerprint(image)
    with _lock:
        cached = _embeddings.get(encoder_path)
    if cached is not None and cached[0] == fingerprint:
        return cached[1], cached[2]

    _check(cancel)
    _report(progress, 0.1, "loading encoder")
    session = _session_for(encoder_path)
    in_name, out_name, shape = _encoder_io(session)
    frame_w, frame_h = _encoder_frame(shape, encoder_path)
    hwc, _scale, (new_w, new_h) = _normalized(image, frame_w, frame_h)

    _check(cancel)
    _report(progress, 0.3, "embedding image")
    last = None
    for candidate in _encoder_inputs(hwc, shape):
        try:
            outputs = session.run([out_name], {in_name: candidate})
            break
        except Exception as exc:  # noqa: BLE001 - retried with the other layout
            last = exc
    else:
        raise RuntimeError(
            f"Segmentation encoder rejected the image (input '{in_name}' "
            f"declared shape {shape}): {last}"
        )
    embeddings = np.asarray(outputs[0])
    geometry = (frame_w, frame_h, new_w, new_h, image.size[0], image.size[1])

    entry = (fingerprint, embeddings, geometry)
    with _lock:
        _embeddings[encoder_path] = entry
    return embeddings, geometry


def build_prompt(points, labels, scale):
    """Return ``(coords, labels)`` arrays for the decoder from point prompts.

    ``points`` are document-pixel ``(x, y)`` pairs; ``labels`` is ``1`` for a
    foreground point and ``0`` for a background one (defaulting to all
    foreground). Coordinates are scaled into the encoder frame.
    """
    points = list(points)
    if labels is None:
        labels = [1] * len(points)
    coords = np.asarray(
        [[float(p[0]) * scale, float(p[1]) * scale] for p in points],
        dtype="float32",
    ).reshape(1, len(points), 2)
    label_array = np.asarray(list(labels), dtype="float32").reshape(1, len(points))
    return coords, label_array


def _find_input(names, *keys, exclude=()):
    """The first declared input name containing every ``key`` (not ``exclude``).

    Exports name the decoder's prompt inputs slightly differently (``point_coords``
    vs ``coords``, ``point_labels`` vs ``labels``), so inputs are matched by
    substring rather than by an exact name.
    """
    for name in names:
        low = name.lower()
        if all(key in low for key in keys) and not any(skip in low for skip in exclude):
            return name
    return None


def _decoder_feed(session, embeddings, coords, labels, frame):
    """The input feed for a SAM decoder graph, filling only what it declares.

    Inputs are matched by role (embeddings, point coordinates/labels, the empty
    prompt mask and the original image size) so minor naming differences between
    exports do not matter. ``frame`` is the encoder frame ``(height, width)`` and
    is what ``orig_im_size`` is set to -- the decoder then returns masks in that
    frame, which :func:`_mask_image` maps back to the document.
    """
    names = [item.name for item in session.get_inputs()]
    feed = {}
    embedding = _find_input(names, "embed")
    if embedding:
        feed[embedding] = embeddings
    point_coords = _find_input(names, "point", "coord") or _find_input(names, "coord")
    if point_coords:
        feed[point_coords] = coords
    point_labels = _find_input(names, "label")
    if point_labels:
        feed[point_labels] = labels
    has_mask = _find_input(names, "has_mask")
    if has_mask:
        feed[has_mask] = np.zeros((1,), "float32")
    mask_input = _find_input(names, "mask_input", "input", exclude=("has_",))
    if not mask_input:
        mask_input = _find_input(names, "mask", exclude=("has_", "label"))
    if mask_input:
        feed[mask_input] = np.zeros((1, 1, MASK_INPUT_SIZE, MASK_INPUT_SIZE), "float32")
    orig_size = _find_input(names, "orig")
    if orig_size:
        feed[orig_size] = np.asarray([frame[0], frame[1]], dtype="float32")
    return feed


def _missing_inputs(session, feed):
    """The decoder's declared inputs the feed does not cover."""
    return [item.name for item in session.get_inputs() if item.name not in feed]


def _output(outputs, names, wanted, index):
    """The output called ``wanted``, else the one at ``index``."""
    if wanted in names:
        return outputs[names.index(wanted)]
    if index < len(outputs):
        return outputs[index]
    return None


def _select_mask(raw, scores):
    """The chosen candidate as an ``L`` mask at the decoder's own resolution.

    A single prompt yields several candidate masks; the one with the highest
    predicted IoU is kept, thresholded to 0/255.
    """
    array = np.asarray(raw)
    if array.ndim == 4:
        array = array[0]
    elif array.ndim == 2:
        array = array[None]
    if array.ndim == 3 and array.shape[0] > 1:
        flat = None if scores is None else np.asarray(scores).reshape(-1)
        if flat is not None and flat.size >= array.shape[0]:
            best = int(np.argmax(flat[: array.shape[0]]))
        else:
            best = 0
        array = array[best]
    elif array.ndim == 3:
        array = array[0]

    if array.dtype == bool:
        binary = array
    else:
        values = array.astype("float32")
        threshold = 0.5 if (values.min() >= 0.0 and values.max() <= 1.0) else _MASK_THRESHOLD
        binary = values > threshold
    return Image.fromarray((binary.astype("uint8") * 255), mode="L")


def _map_mask(mask, geometry, mode="topleft"):
    """Map a frame-space ``mask`` to the document under one placement ``mode``.

    ``mode`` is ``"topleft"`` (the image sits at the frame origin, the rest
    black), ``"centered"`` (equal padding) or ``"stretch"`` (the image was
    stretched to fill the frame, so no crop).
    """
    frame_w, frame_h, new_w, new_h, width, height = geometry
    if mask.size == (width, height):
        return mask
    if mode == "stretch":
        return mask.resize((width, height), Image.NEAREST)
    if mode == "centered":
        offset_x = max(0, (frame_w - new_w) // 2)
        offset_y = max(0, (frame_h - new_h) // 2)
    else:
        offset_x = offset_y = 0
    left = max(0, int(round(offset_x * mask.width / frame_w)))
    top = max(0, int(round(offset_y * mask.height / frame_h)))
    crop_w = max(1, int(round(new_w * mask.width / frame_w)))
    crop_h = max(1, int(round(new_h * mask.height / frame_h)))
    box = (left, top, min(left + crop_w, mask.width), min(top + crop_h, mask.height))
    return mask.crop(box).resize((width, height), Image.NEAREST)


def _mask_image(raw, scores, geometry):
    """Turn the decoder's ``masks`` output into an ``L`` mask at document size.

    ``geometry`` is ``(frame_w, frame_h, new_w, new_h, width, height)``; the
    decoder returns masks in the encoder frame, so the content region is cropped
    out and scaled to the document (see :func:`_map_mask`).
    """
    return _map_mask(_select_mask(raw, scores), geometry)


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


def segment(image, points, labels=None, model_name=None, progress=None, cancel=None):
    """Return an ``L`` mask of the object under ``points`` in ``image``.

    ``points`` is a list of document-pixel ``(x, y)`` foreground prompts (a
    single click is the common case); an optional ``labels`` list marks points
    as foreground (``1``) or background (``0``). ``progress``/``cancel`` follow
    the other engine jobs (see :mod:`engine.cancel`).
    """
    points = list(points)
    if not points:
        raise ValueError("Segmentation needs at least one point prompt.")

    if model_name is None:
        models = list_models()
        if not models:
            raise RuntimeError(
                "No segmentation model found under "
                + paths.segmentation_dir()
                + ". Add a SAM ONNX encoder/decoder pair "
                "('<name>_encoder.onnx' and '<name>_decoder.onnx')."
            )
        model_name = models[0]

    encoder_path, decoder_path = _resolve(model_name)
    embeddings, geometry = _embed(image, encoder_path, progress, cancel)
    frame_w, frame_h, _new_w, _new_h, width, height = geometry
    scale = min(frame_w / max(width, 1), frame_h / max(height, 1))

    _check(cancel)
    _report(progress, 0.6, "segmenting")
    session = _session_for(decoder_path)
    coords, label_array = build_prompt(points, labels, scale)
    feed = _decoder_feed(session, embeddings, coords, label_array, (frame_h, frame_w))
    missing = _missing_inputs(session, feed)
    if missing:
        raise RuntimeError(
            "This segmentation decoder needs inputs this app does not provide "
            f"({', '.join(missing)}). It looks like a different SAM export than "
            "the point-prompt encoder/decoder pair; try a 'sam_vit_*' model from "
            "vietanhdev/segment-anything-onnx-models."
        )

    out_names = [item.name for item in session.get_outputs()]
    outputs = session.run(None, feed)

    _check(cancel)
    _report(progress, 0.9, "finishing")
    masks = _output(outputs, out_names, "masks", 0)
    scores_raw = _output(outputs, out_names, "iou_predictions", 1)
    if masks is None:
        raise RuntimeError("Segmentation model returned no masks.")
    scores = None if scores_raw is None else np.asarray(scores_raw).reshape(-1)
    frame_mask = _select_mask(masks, scores)
    result = _mask_image(masks, scores, geometry)
    click_xy = (min(max(int(points[0][0]), 0), width - 1),
                min(max(int(points[0][1]), 0), height - 1))
    # The prompt is on the object, so a correct mapping puts it in the mask. If
    # the primary placement misses it, try the other ways an export may have
    # placed the image in the frame.
    if result.getpixel(click_xy) != 255:
        for mode in ("stretch", "centered"):
            candidate = _map_mask(frame_mask, geometry, mode)
            if candidate.getpixel(click_xy) == 255:
                result = candidate
                break

    _report(progress, 1.0, "done")
    return result


def unload(key=None):
    """Drop cached sessions and embeddings; returns how many went away.

    With no ``key`` every cache is cleared. With a ``key`` from
    :func:`loaded_models` (``"session:<path>"``) only that session is dropped
    (and any embedding it produced).

    The caller must ensure no segmentation job is running.
    """
    with _lock:
        if key is None:
            count = len(_sessions) + len(_embeddings)
            _sessions.clear()
            _embeddings.clear()
            return count
        kind, _, path = key.partition(":")
        if kind != "session" or path not in _sessions:
            return 0
        del _sessions[path]
        if path.endswith(_ENCODER_SUFFIX):
            _embeddings.pop(path, None)
        return 1


def loaded_models():
    """Describe the segmentation graphs cached in memory, for the UI.

    The two graphs of a model are reported as separate sessions (an ONNX session
    does not expose its weight sizes, so each reports its file and size on
    disk). Keys are ``session:<path>``, matching the other ONNX sources.
    """
    from .loaded_models import file_size

    with _lock:
        sessions = list(_sessions.items())
    records = []
    for path, session in sessions:
        try:
            providers = session.get_providers()
        except Exception:
            providers = []
        device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        role = "Decoder" if path.lower().endswith(_DECODER_SUFFIX) else "Encoder"
        records.append({
            "family": "Segmentation (SAM, ONNX)",
            "source": os.path.basename(path),
            "key": f"session:{path}",
            "components": [{
                "name": role,
                "params": 0,
                "bytes": file_size(path),
                "dtype": "onnx",
                "device": device,
            }],
        })
    return records
