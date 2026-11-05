"""Detect the implementation ("kind") of a model or component file.

The Model Settings dialog uses this to label pool files and model files with
what they are (CLIP-L, T5-XXL, Qwen3-4B, Flux VAE, ...) and to narrow each model
row's component slot to the files that fit it, so a user who does not know which
file to assign gets concrete options instead of one flat list.

Detection is best-effort and cheap: it reads only a file's header -- the
``.safetensors`` JSON header or a ``.gguf``'s metadata and tensor names (both
memory-mapped, never the tensor data) -- falling back to filename tokens when a
header cannot be read. Results are cached by path and file stamp so a dialog
that redraws its rows does not re-read a header per redraw, and that cache is
persisted under the per-user app folder so a later session opens with the
already-known kinds warm instead of re-reading every file (see
:func:`save_cache`).
"""

import json
import mmap
import os
import re
import struct
import threading

TEXT_ENCODER = "text_encoder"
VAE = "vae"
MODEL = "model"
CONTROL = "control"
ADAPTER = "adapter"

# kind id -> (label, category). This order is the display order within a pool.
KINDS = {
    "clip_l": ("CLIP-L", TEXT_ENCODER),
    "clip_g": ("CLIP-G", TEXT_ENCODER),
    "t5xxl": ("T5-XXL", TEXT_ENCODER),
    "qwen3_06b": ("Qwen3-0.6B", TEXT_ENCODER),
    "qwen3_4b": ("Qwen3-4B", TEXT_ENCODER),
    "qwen3vl_4b": ("Qwen3-VL-4B", TEXT_ENCODER),
    "qwen3vl_8b": ("Qwen3-VL-8B", TEXT_ENCODER),
    "qwen3_5_9b": ("Qwen3.5-9B", TEXT_ENCODER),
    "qwen3vl_32b": ("Qwen3-VL-32B", TEXT_ENCODER),
    "qwen_image_vae": ("Qwen-Image VAE", VAE),
    "qwen_image_21_vae": ("Qwen-Image 2.1 VAE", VAE),
    "flux_vae": ("Flux VAE", VAE),
    "flux2_vae": ("FLUX.2 VAE", VAE),
    "h3_video_vae": ("MiniMax H3 Video VAE", VAE),
    "controlnet_sd15": ("ControlNet \u00b7 SD 1.5", CONTROL),
    "controlnet_sdxl": ("ControlNet \u00b7 SDXL", CONTROL),
    "controlnet_zimage": ("ControlNet \u00b7 Z-Image", CONTROL),
    "lllite": ("Anima LLLite", CONTROL),
    "anima_ref": ("Anima Ref Adapter", ADAPTER),
    "anypaint": ("Krea 2 AnyPaint", ADAPTER),
    "anima": ("Anima", MODEL),
    "qwen_image": ("Qwen-Image 2.1", MODEL),
    "flux": ("Flux", MODEL),
    "z_image": ("Z-Image", MODEL),
    "krea2": ("Krea 2", MODEL),
    "moss_image": ("MOSS Image", MODEL),
    "h3": ("MiniMax H3", MODEL),
    "fibo": ("Fibo 2.0", MODEL),
    "sd15": ("SD 1.5", MODEL),
    "sdxl": ("SDXL", MODEL),
    "stable_diffusion": ("Stable Diffusion", MODEL),
}

# The kinds offered for each category, in display order.
CATEGORY_KINDS = {
    TEXT_ENCODER: (
        "clip_l", "clip_g", "t5xxl", "qwen3_06b", "qwen3_4b", "qwen3vl_4b",
        "qwen3vl_8b", "qwen3_5_9b", "qwen3vl_32b",
    ),
    VAE: ("qwen_image_vae", "qwen_image_21_vae", "flux_vae", "flux2_vae",
          "h3_video_vae"),
    CONTROL: ("controlnet_sd15", "controlnet_sdxl", "controlnet_zimage", "lllite"),
    ADAPTER: ("anima_ref", "anypaint"),
    MODEL: ("anima", "qwen_image", "flux", "z_image", "krea2", "moss_image",
            "h3", "fibo", "sd15", "sdxl", "stable_diffusion"),
}

# Settings pool name -> detection category. Anima LLLite adapters are a
# ``CONTROL`` kind (``lllite``) but share the ``controlnet`` pool with the
# classic ControlNets, so no separate pool is listed here.
POOL_CATEGORY = {
    "text_encoders": TEXT_ENCODER,
    "vaes": VAE,
    "controlnet": CONTROL,
    "adapters": ADAPTER,
}

# The intended implementation ("kind") of each model family's component slot.
# The Model Settings dialog seeds each slot's Kind picker from this, and the
# engine uses it to pick a slot's default file from the curated pools when a
# model row leaves the slot on the family default (see ``engine.paths``).
SLOT_KIND = {
    ("anima", "clip"): "qwen3_06b",
    ("anima", "vae"): "qwen_image_vae",
    ("qwen_image", "text"): "qwen3vl_8b",
    ("qwen_image", "vae"): "qwen_image_21_vae",
    ("flux", "clip"): "clip_l",
    ("flux", "t5"): "t5xxl",
    ("flux", "vae"): "flux_vae",
    ("z_image", "text"): "qwen3_4b",
    ("z_image", "vae"): "flux_vae",
    ("krea2", "text"): "qwen3vl_4b",
    ("krea2", "vae"): "qwen_image_vae",
    ("moss_image", "text"): "qwen3_5_9b",
    ("moss_image", "vae"): "flux2_vae",
    ("h3", "text"): "qwen3vl_32b",
    ("h3", "vae"): "h3_video_vae",
}

# Component slot -> the curated pool its default file is drawn from.
SLOT_POOL = {
    "clip": "text_encoders",
    "clip_g": "text_encoders",
    "t5": "text_encoders",
    "text": "text_encoders",
    "vae": "vaes",
}

_FORMATS = (
    (".safetensors", "safetensors"),
    (".gguf", "gguf"),
    (".ckpt", "ckpt"),
    (".onnx", "onnx"),
    (".pth", "pth"),
    (".pt", "pt"),
    (".bin", "bin"),
)

_QWEN_VOCAB = 151936
_MAX_HEADER = 100_000_000

# GGUF metadata value types: fixed sizes, plus the two variable ones.
_GGUF_FIXED = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9

_CACHE = {}
_CACHE_LOCK = threading.Lock()
_CACHE_LOADED = False

# The warm kinds are persisted beside the settings (see
# :func:`~engine.settings.app_dir`) so a later session can skip the header reads
# a fresh process would otherwise repeat. Each record keeps the file's stamp
# (size, mtime), so an entry that no longer matches its file is not reused, and
# save prunes the records whose files have since changed or gone.
_CACHE_FILE_NAME = "model_kinds_cache.json"

# Cap on persisted records, so an unusually large model tree cannot grow the
# file without bound; detections beyond the cap are dropped on save.
_CACHE_MAX = 5000


def cache_path():
    """The on-disk kind cache path (under the per-user app folder)."""
    from . import settings as engine_settings

    return os.path.join(engine_settings.app_dir(), _CACHE_FILE_NAME)


def _ensure_loaded():
    """Fill ``_CACHE`` from disk once, on first use.

    Loading is deferred to the first :func:`detect`/:func:`cached` call rather
    than done at import, so app start pays nothing and no header read happens
    until a dialog or panel actually asks for a kind.
    """
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    with _CACHE_LOCK:
        if _CACHE_LOADED:
            return
        _CACHE_LOADED = True
        _read_cache_file()


def _read_cache_file():
    """Merge the persisted records into ``_CACHE`` (corrupt file -> no-op)."""
    try:
        with open(cache_path(), encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, ValueError):
        return
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, list) or len(record) != 5:
            continue
        path, category, size, mtime, kind = record
        if not isinstance(path, str) or not isinstance(size, int) or not isinstance(mtime, int):
            continue
        if category is not None and not isinstance(category, str):
            continue
        if kind is not None and not isinstance(kind, str):
            continue
        _CACHE[(path, category, (size, mtime))] = kind


def save_cache():
    """Persist the warm kind cache so a later session skips the header reads.

    Records whose file no longer matches its stamp are dropped, so an edited or
    removed file is re-read next time and the file does not grow with dead
    entries. Called after a detection pass (and on quit); best-effort, so an
    unwritable location is ignored.
    """
    _ensure_loaded()
    with _CACHE_LOCK:
        items = list(_CACHE.items())
    records = []
    for (path, category, stamp), kind in items:
        size, mtime = stamp
        if _stamp(path) != (size, mtime):
            continue
        records.append([path, category, size, mtime, kind])
    if len(records) > _CACHE_MAX:
        records = records[:_CACHE_MAX]
    try:
        path = cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(records, handle)
    except OSError:
        pass


def kind_label(kind):
    """A kind's display label (``""`` for an unknown kind)."""
    entry = KINDS.get(kind)
    return entry[0] if entry else ""


def category_of(kind):
    """The category a kind belongs to (``""`` for an unknown kind)."""
    entry = KINDS.get(kind)
    return entry[1] if entry else ""


def kinds(category):
    """The known kind ids for a category, in display order."""
    return CATEGORY_KINDS.get(category, ())


def format_of(path):
    """A file's container format from its suffix (``""`` when unknown)."""
    lower = (path or "").lower()
    for suffix, name in _FORMATS:
        if lower.endswith(suffix):
            return name
    return ""


def _stamp(path):
    try:
        status = os.stat(path)
    except OSError:
        return None
    return (status.st_size, int(status.st_mtime))


def detect(path, category=None):
    """The kind id for ``path`` (``None`` when it cannot be determined).

    ``category`` narrows the search (a text-encoder file is not mistaken for a
    VAE); with no category every category is tried in turn. Header-read results
    are cached by path, category and file stamp, and that cache is loaded from
    disk on first use, so repeated redraws and later sessions are cheap.
    """
    if not path:
        return None
    _ensure_loaded()
    stamp = _stamp(path)
    key = (os.fspath(path), category, stamp)
    if stamp is not None and key in _CACHE:
        return _CACHE[key]
    try:
        kind = _detect_uncached(path, category)
    except Exception:
        kind = None
    if stamp is not None:
        _CACHE[key] = kind
    return kind


def describe(path, category=None):
    """A short ``"CLIP-L \u00b7 safetensors"`` description of ``path``.

    Falls back to the format alone, then to ``""``. Used to annotate a file in
    the settings without claiming a kind it may not have.
    """
    return describe_kind(path, detect(path, category))


def describe_kind(path, kind):
    """The :func:`describe` text for an already-detected ``kind``.

    Lets a caller render the same annotation from a deferred or cached kind (see
    :func:`cached`) without triggering a header read.
    """
    parts = []
    label = kind_label(kind)
    if label:
        parts.append(label)
    fmt = format_of(path)
    if fmt:
        parts.append(fmt.upper() if fmt == "gguf" else fmt)
    return " \u00b7 ".join(parts)


def cached(path, category=None):
    """The kind :func:`detect` has already memoized for ``path``.

    Returns ``None`` when the file has not been detected for its current stamp
    (or the path is empty), so a caller can render from warm results only and
    leave the header read to a background pass. Disk-cached results count as
    warm, so a later session renders the rows immediately.
    """
    if not path:
        return None
    _ensure_loaded()
    stamp = _stamp(path)
    if stamp is None:
        return None
    return _CACHE.get((os.fspath(path), category, stamp))



def _detect_uncached(path, category):
    name = os.path.basename(path).lower()
    fmt = format_of(path)
    shapes = {}
    arch = None
    if fmt == "safetensors":
        shapes = _safetensors_shapes(path)
    elif fmt == "gguf":
        arch, shapes = _gguf_signature(path)
    if category == TEXT_ENCODER:
        return _text_encoder_kind(name, shapes, arch)
    if category == VAE:
        return _vae_kind(name, shapes)
    if category == CONTROL:
        return _control_kind(name, shapes)
    if category == ADAPTER:
        return _adapter_kind(name, shapes)
    if category == MODEL:
        return _model_kind(path, shapes, arch)
    return (
        _model_kind(path, shapes, arch)
        or _text_encoder_kind(name, shapes, arch)
        or _vae_kind(name, shapes)
        or _control_kind(name, shapes)
        or _adapter_kind(name, shapes)
    )


def _safetensors_shapes(path):
    """``{tensor name: shape tuple}`` from a ``.safetensors`` header (no data)."""
    try:
        with open(path, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size <= 0 or size > _MAX_HEADER:
                return {}
            header = json.loads(handle.read(size))
    except (OSError, ValueError, struct.error):
        return {}
    header.pop("__metadata__", None)
    shapes = {}
    for key, value in header.items():
        if isinstance(value, dict) and isinstance(value.get("shape"), list):
            shapes[key] = tuple(value["shape"])
    return shapes


def _gguf_signature(path):
    """``(general.architecture, {tensor name: shape})`` from a GGUF header.

    ``gguf.GGUFReader`` decodes every metadata value eagerly, and a text
    encoder's tokenizer metadata can hold hundreds of thousands of array
    elements — seconds of work the settings dialog must not pay just to label a
    file. This walks the header with a struct cursor instead and jumps over
    value payloads by size, touching only the bytes it needs.
    """
    try:
        with open(path, "rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                return _parse_gguf_header(data)
    except (OSError, ValueError, struct.error, OverflowError):
        return None, {}


def _parse_gguf_header(data):
    if len(data) < 24 or data[:4] != b"GGUF":
        return None, {}
    if struct.unpack_from("<I", data, 4)[0] < 2:
        return None, {}
    tensor_count, kv_count = struct.unpack_from("<QQ", data, 8)
    pos = 24
    arch = None
    for _ in range(kv_count):
        pos, key, value = _gguf_kv(data, pos)
        if key == "general.architecture" and isinstance(value, str):
            arch = value
    shapes = {}
    for _ in range(tensor_count):
        name_len = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        name = data[pos:pos + name_len].decode("utf-8", "replace")
        pos += name_len
        n_dims = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dims = struct.unpack_from("<%dQ" % n_dims, data, pos) if n_dims else ()
        pos += 8 * n_dims + 4 + 8  # dims, ggml type, data offset
        shapes[name] = tuple(int(d) for d in dims)
    return arch, shapes


def _gguf_kv(data, pos):
    """Read one metadata key/value; return ``(pos, key, value_or_None)``."""
    key_len = struct.unpack_from("<Q", data, pos)[0]
    pos += 8
    key = data[pos:pos + key_len].decode("utf-8", "replace")
    pos += key_len
    vtype = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    if vtype == _GGUF_STRING:
        size = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        value = data[pos:pos + size].decode("utf-8", "replace")
        return pos + size, key, value
    return _gguf_skip(data, pos, vtype), key, None


def _gguf_skip(data, pos, vtype):
    fixed = _GGUF_FIXED.get(vtype)
    if fixed is not None:
        return pos + fixed
    if vtype == _GGUF_STRING:
        size = struct.unpack_from("<Q", data, pos)[0]
        return pos + 8 + size
    if vtype == _GGUF_ARRAY:
        elem_type, count = struct.unpack_from("<IQ", data, pos)
        pos += 12
        if elem_type == _GGUF_STRING:
            for _ in range(count):
                size = struct.unpack_from("<Q", data, pos)[0]
                pos += 8 + size
        elif elem_type == _GGUF_ARRAY:
            for _ in range(count):
                pos = _gguf_skip(data, pos, elem_type)
        else:
            step = _GGUF_FIXED.get(elem_type)
            if step is None:
                raise ValueError("unknown GGUF array element type")
            pos += count * step
        return pos
    raise ValueError("unknown GGUF value type")


def _hidden_dim(shape, vocab=None):
    """The non-vocabulary dimension of an embedding's shape."""
    dims = [int(v) for v in shape]
    if not dims:
        return None
    if vocab is not None and vocab in dims:
        others = [d for d in dims if d != vocab]
        return others[-1] if others else None
    return min(dims)


# --------------------------------------------------------------- text encoders
def _text_encoder_kind(name, shapes, arch):
    if any(token in name for token in ("umt5", "mt5", "umt-5")):
        # WAN's UMT5 shares the T5 architecture but is not the T5-XXL the Flux
        # slot wants, so it is left unknown rather than mislabelled.
        return None
    token = _text_encoder_token(name)
    if token:
        return token
    if arch:
        low = arch.lower()
        if "t5" in low:
            return "t5xxl"
        if "qwen" in low:
            return _qwen_kind_from_shapes(shapes, is_vl="vl" in low)
    if shapes:
        if any(key.startswith("text_model.") for key in shapes):
            return _clip_kind_from_shapes(shapes)
        if "shared.weight" in shapes or any(
            key.startswith("encoder.block.") for key in shapes
        ):
            return "t5xxl"
        is_vl = any(
            key.startswith("model.visual") or key.startswith("v.blk.")
            for key in shapes
        )
        return _qwen_kind_from_shapes(shapes, is_vl)
    return None


def _text_encoder_token(name):
    if "clip_l" in name or "clip-l" in name:
        return "clip_l"
    if "clip_g" in name or "clip-g" in name or "clipg" in name:
        return "clip_g"
    # A bare "t5" must sit on a token boundary ("t5xxl", "t5-v1_1"), so names
    # like "resnet50" (which contains "t5") are not caught.
    if "t5xxl" in name or re.search(r"(^|[^a-z0-9])t5([^a-z0-9]|$)", name):
        return "t5xxl"
    if "qwen3.5" in name or "qwen3_5" in name or "qwen-3.5" in name:
        return "qwen3_5_9b"
    if "qwen" in name:
        is_vl = "vl" in name
        if "32b" in name:
            return "qwen3vl_32b" if is_vl else None
        if "0.6b" in name or "06b" in name or "0_6b" in name or "0-6b" in name:
            return "qwen3_06b"
        if "4b" in name:
            return "qwen3vl_4b" if is_vl else "qwen3_4b"
        if "8b" in name:
            return "qwen3vl_8b" if is_vl else None
    return None


def _clip_kind_from_shapes(shapes):
    embedding = shapes.get("text_model.embeddings.token_embedding.weight")
    if embedding and len(embedding) >= 2:
        hidden = int(embedding[1])
        if hidden == 768:
            return "clip_l"
        if hidden == 1280:
            return "clip_g"
    q_proj = shapes.get("text_model.encoder.layers.0.self_attn.q_proj.weight")
    if q_proj and q_proj:
        hidden = int(q_proj[0])
        if hidden == 768:
            return "clip_l"
        if hidden == 1280:
            return "clip_g"
    if any(key.startswith("text_model.") for key in shapes):
        return "clip_l"
    return None


def _qwen_kind_from_shapes(shapes, is_vl=False):
    embedding = None
    for key in (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "token_embd.weight",
    ):
        if key in shapes:
            embedding = shapes[key]
            break
    if not embedding:
        return None
    hidden = _hidden_dim(embedding, vocab=_QWEN_VOCAB)
    # A hybrid linear-attention stack at 4096 hidden is Qwen3.5-9B (the MOSS
    # Image text encoder); a plain 4096-wide Qwen3 is not one of the known slots.
    if hidden == 4096 and any("linear_attn" in key for key in shapes):
        return "qwen3_5_9b"
    if hidden == 5120:
        # The MiniMax H3 text encoder is the full Qwen3-VL-32B (hidden 5120).
        return "qwen3vl_32b" if is_vl else None
    if hidden == 1024:
        return "qwen3_06b"
    if hidden == 2560:
        return "qwen3vl_4b" if is_vl else "qwen3_4b"
    if hidden == 4096:
        return "qwen3vl_8b" if is_vl else None
    return None


# ------------------------------------------------------------------------ VAEs
def _vae_kind(name, shapes):
    token = _vae_token(name)
    if token:
        return token
    if shapes:
        if "bn.running_mean" in shapes:
            # The FLUX.2 VAE carries a latent BatchNorm that no other VAE has.
            return "flux2_vae"
        conv2 = shapes.get("conv2.weight")
        if conv2:
            z_dim = int(conv2[0])
            return "qwen_image_21_vae" if z_dim == 64 else "qwen_image_vae"
        if "conv1.weight" in shapes or any(
            key.startswith("encoder.downsamples") for key in shapes
        ):
            return "qwen_image_vae"
        if any(
            key.startswith("encoder.") or key.startswith("decoder.")
            or key in ("quant_conv.weight", "post_quant_conv.weight")
            for key in shapes
        ):
            return "flux_vae"
    return None


def _vae_token(name):
    if "vae" not in name:
        return "flux_vae" if name == "ae.safetensors" else None
    # MiniMax H3's video VAE (``minimax_h3_video_vae_*``): the only H3 VAE the
    # engine runs (the optional single-frame image VAE is not used).
    if "minimax" in name or "h3" in name:
        return "h3_video_vae"
    # Require "qwen" for a Qwen VAE, so WAN's "wan_2.1_vae" is not read as the
    # Qwen-Image 2.1 VAE just because of the "2.1".
    if "qwen" in name:
        return "qwen_image_21_vae" if ("2.1" in name or "2_1" in name) else "qwen_image_vae"
    if "flux2" in name or "flux-2" in name or "flux.2" in name:
        return "flux2_vae"
    if "flux" in name:
        return "flux_vae"
    return None


# ------------------------------------------------------------- diffusion models
def _model_kind(path, shapes, arch):
    return _model_from_shapes(shapes, arch) or _model_from_name(path)


def _model_from_shapes(shapes, arch):
    if shapes:
        original = set(shapes)
        # A ComfyUI single-file checkpoint prefixes every DiT tensor; strip it so
        # the same signatures match whether or not the prefix is present.
        stripped = original
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if any(key.startswith(prefix) for key in original):
                stripped = {key[len(prefix):] for key in original if key.startswith(prefix)}
                break
        if "shared_block_modulation.weight" in stripped and "video_embedder.weight" in stripped:
            return "moss_image"
        if any(key.startswith("double_blocks.") for key in stripped):
            return "flux"
        if any(key.startswith("txtfusion.") for key in stripped) or (
            "first.weight" in stripped and "tproj.1.weight" in stripped
        ):
            return "krea2"
        if "cap_embedder.1.weight" in stripped:
            return "z_image"
        if any(
            key.startswith("transformer_blocks.") or key.startswith("txt_in.")
            for key in stripped
        ) and any(key.startswith("img_in.") for key in stripped):
            return "qwen_image"
        if any(key.startswith("x_embedder.") for key in stripped) and any(
            key.startswith("blocks.") for key in stripped
        ):
            return "anima"
        if any(key.startswith("model.diffusion_model.") for key in original):
            return _classic_kind(original)
    if arch:
        low = arch.lower()
        if "flux" in low or "chroma" in low:
            return "flux"
        if "krea" in low:
            return "krea2"
        if "qwen" in low:
            return "qwen_image"
        if "z_image" in low or "z-image" in low or "lumina" in low:
            return "z_image"
        if "moss" in low:
            return "moss_image"
        if "fibo" in low:
            return "fibo"
        if "cosmos" in low:
            return "anima"
    return None


def _classic_kind(keys):
    """Classify a file that carries the ``model.diffusion_model.`` prefix.

    A classic SD checkpoint also bundles ``conditioner.``/``first_stage_model.``;
    an Anima file instead carries the Cosmos DiT and its ``llm_adapter.``.
    """
    if any(key.startswith("conditioner.embedders") for key in keys):
        return "sdxl"
    if any(key.startswith("conditioner.") for key in keys) or any(
        key.startswith("first_stage_model.") for key in keys
    ):
        return "sd15"
    if any(key.startswith("model.diffusion_model.llm_adapter") for key in keys) or any(
        key.startswith("model.diffusion_model.x_embedder") for key in keys
    ):
        return "anima"
    return None


def _model_from_name(path):
    from . import paths

    _group, family = paths.classify_model(path)
    if family == "stable_diffusion":
        # A safetensors checkpoint names its architecture in the header; a
        # pickled ``.ckpt`` cannot be read cheaply, so it stays the generic kind.
        if format_of(path) == "safetensors":
            return "sdxl" if paths.checkpoint_arch(path) == "sdxl" else "sd15"
        return "stable_diffusion"
    return family


# ------------------------------------------------------------- control adapters
def _control_kind(name, shapes):
    """The control-adapter kind: an Anima LLLite or a Stable Diffusion ControlNet.

    LLLite and ControlNet both live beside a generation model but are not one, so
    they get their own category; this also keeps a ControlNet file from being
    read as the diffusion model it was trained for.
    """
    if any(
        key.startswith("lllite_conditioning1.") or key.startswith("lllite_modules.")
        for key in shapes
    ):
        return "lllite"
    if "lllite" in name:
        return "lllite"
    if any(
        key.startswith("control_layers.") or key.startswith("control_all_x_embedder.")
        for key in shapes
    ):
        return "controlnet_zimage"
    is_controlnet = any(
        key.startswith(prefix)
        for key in shapes
        for prefix in (
            "controlnet_cond_embedding.", "controlnet_down_blocks.",
            "control_model.", "input_hint_block.", "zero_convs.",
        )
    )
    if is_controlnet or _looks_like_control(name):
        if "z-image" in name or "z_image" in name or "zimage" in name:
            return "controlnet_zimage"
        return _controlnet_arch(shapes, name)
    return None


def _looks_like_control(name):
    """Whether a filename reads as a ControlNet (diffusers or ComfyUI/LDM)."""
    return "controlnet" in name or "control_" in name or "control-" in name


def _controlnet_arch(shapes, name):
    """``controlnet_sdxl`` or ``controlnet_sd15`` for a ControlNet file.

    The cross-attention input width decides it (768 for SD 1.x, 2048 for SDXL),
    which holds for both the LDM and the diffusers layouts; a file whose header
    cannot be read falls back to its filename.
    """
    for key, shape in shapes.items():
        if key.endswith("attn2.to_k.weight") and len(shape) == 2:
            return "controlnet_sdxl" if int(shape[1]) == 2048 else "controlnet_sd15"
    if "sdxl" in name or "_xl" in name or "-xl" in name:
        return "controlnet_sdxl"
    return "controlnet_sd15"


# ------------------------------------------------------------- run-time adapters
def _adapter_kind(name, shapes):
    """The run-time adapter kind: the Anima reference adapter or Krea 2 AnyPaint.

    A reference adapter is a LoRA targeting the Anima DiT's ``blocks`` (see
    :mod:`engine.anima_ref`); AnyPaint is a Krea 2 functional adapter that adds
    arbitrary-mask editing (see :mod:`engine.krea2_anypaint`). Both are applied
    only in their own mode, so they live in their own pool rather than the
    always-on LoRAs. They are recognised by name first, then by LoRA weights whose
    names carry the DiT ``blocks`` path (so a renamed file is still labelled once
    its header is read).
    """
    if any(
        token in name
        for token in ("anypaint", "any_paint", "any-paint")
    ):
        return "anypaint"
    if any(
        token in name
        for token in ("anima_ref", "anima-ref", "ref_adapter", "ref-adapter")
    ):
        return "anima_ref"
    lora = any(
        key.endswith((".lora_down.weight", ".lora_A.weight")) for key in shapes
    )
    if lora and any(".blocks." in key for key in shapes):
        return "anima_ref"
    return None