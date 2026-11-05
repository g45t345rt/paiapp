"""MOSS Image text conditioning: Qwen3.5-9B last hidden states.

MOSS Image conditions its DiT on the **last** hidden states of a stock
**Qwen3.5-9B** text model (``text_dim = 4096``), tokenised with the Qwen3.5
tokenizer and the chat template with the thinking block left open (ComfyUI
``comfy/text_encoders/moss_image.py``: the ``enable_thinking=True`` template
trims the user turn and leaves an open `` thinking`` block):

    <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n thinking\\n

The model is a dense hybrid linear-attention transformer (``Qwen3_5TextModel`` /
``Qwen3_5Model`` in ``transformers`` >= 5.3). Weights may be a plain bf16
``.safetensors`` or ComfyUI's ``int8_tensorwise`` quantisation with the optional
``convrot`` rotation (``qwen3.5_9b_int8_convrot.safetensors``), which
:func:`_dequant` reconstructs. The config and tokenizer come from a Hugging Face
repo id (or a local directory) -- point ``repo`` at it, or pass a directory as
``weights_path`` to load everything from there.
"""

import json
import math
import os

import torch
from safetensors import safe_open

from .precision import build_on_dtype, build_on_meta, skip_random_init

# Where the Qwen3.5-9B config and tokenizer are read from when the model is
# given as a bare weights file. A local directory passed as ``weights_path``
# overrides this entirely.
_DEFAULT_REPO = "Qwen/Qwen3.5-9B"

_PREFIX = "<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n thinking\n"

# Row-chunk budget for loading weights; bounds the float32 transient so a 9B
# model loads within RAM (see ``_copy_weight``).
_CHUNK_BYTES = 128 << 20

_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2, "I16": 2,
    "I8": 1, "U8": 1, "BOOL": 1,
}


def _itemsize(dtype_name):
    return _DTYPE_BYTES.get(dtype_name, 4)


def _build_model(config):
    """Build the Qwen3.5 text model from ``config`` (multimodal or text-only).

    Prefers the text-only ``Qwen3_5TextModel`` (the vision tower is unused here
    and would only cost memory); the multimodal ``Qwen3_5Model`` is the fallback
    when only a full config is available.
    """
    import transformers

    text_config = getattr(config, "text_config", None) or config
    errors = []
    for name, cfg in (("Qwen3_5TextModel", text_config), ("Qwen3_5Model", config)):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls(cfg)
        except Exception as exc:  # wrong config flavour for this class
            errors.append(f"{name}: {exc}")
    if not errors:
        raise RuntimeError(
            "MOSS Image needs transformers>=5.3 for the Qwen3.5 text encoder "
            "(no Qwen3_5Model/Qwen3_5TextModel found)."
        )
    raise RuntimeError("Could not build the Qwen3.5 text model: " + "; ".join(errors))


# ------------------------------------------------------------- int8 / convrot
# ComfyUI's ``int8_tensorwise`` files store ``X.weight`` as int8 with a
# per-output-row ``X.weight_scale``; ``convrot`` additionally rotates groups of
# 256 along the last dim by a normalized regular Hadamard matrix. The rotation
# is symmetric and self-inverse, so un-rotating the dequantized weight lets a
# plain ``F.linear`` reproduce ComfyUI's result exactly.
def _hadamard(size):
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32,
    )
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / math.sqrt(size)


def _unrotate(weight, group_size):
    """Un-rotate the ``convrot`` rotation: ``W @ H_block`` along the last dim."""
    if group_size < 4 or weight.shape[-1] % group_size:
        return weight
    h = _hadamard(group_size).to(weight.dtype)
    rows, cols = weight.shape
    return (weight.reshape(rows, cols // group_size, group_size) @ h).reshape(rows, cols)


def _dequant(handle, names, key):
    """Reconstruct one tensor, dequantizing ComfyUI int8 (+ convrot) weights."""
    tensor = handle.get_tensor(key)
    scale_key = key + "_scale"
    if tensor.dtype != torch.int8 or scale_key not in names:
        return tensor
    value = tensor.to(torch.float32) * handle.get_tensor(scale_key).to(torch.float32)
    group_size, convrot = _quant_conf(handle, names, key)
    return _unrotate(value, group_size) if convrot else value


def _copy_weight(handle, names, key, dst, dtype):
    """Dequantize ``key`` into the ``dst`` parameter in bounded row chunks.

    Full-tensor materialisation would need ~4 GB for the 248320x4096 embedding
    alone; chunking keeps the float32 transient at ``_CHUNK_BYTES`` regardless of
    the tensor's size.
    """
    scale_key = key + "_scale"
    sl = handle.get_slice(key)
    shape = sl.get_shape()
    rows = int(shape[0])
    cols = int(shape[1]) if len(shape) > 1 else 1
    if sl.get_dtype() != "I8" or scale_key not in names:
        step = max(1, _CHUNK_BYTES // max(1, cols * _itemsize(sl.get_dtype())))
        for start in range(0, rows, step):
            dst[start:start + step].copy_(sl[start:start + step].to(dtype))
        return
    group_size, convrot = _quant_conf(handle, names, key)
    scale = handle.get_tensor(scale_key).to(torch.float32)
    per_row = scale.dim() >= 2
    scale_sl = handle.get_slice(scale_key) if per_row else None
    step = max(1, _CHUNK_BYTES // max(1, cols * 4))
    for start in range(0, rows, step):
        stop = start + step
        chunk = sl[start:stop].to(torch.float32)
        factor = scale_sl[start:stop].to(torch.float32) if per_row else scale
        value = chunk * factor
        if convrot:
            value = _unrotate(value, group_size)
        dst[start:stop].copy_(value.to(dtype))


def _strip_component_prefix(name):
    """Drop a leading module prefix so a tensor name matches metadata layer keys."""
    for prefix in (
        "model.diffusion_model.", "diffusion_model.", "model.language_model.",
        "language_model.", "model.",
    ):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _metadata_quant_layers(handle):
    """The ``_quantization_metadata`` layer map from the file's ``__metadata__``.

    Some int8 checkpoints (the MOSS DiT ``*_int8_convrot`` files) do not carry a
    sibling ``X.comfy_quant`` tensor at all; instead the whole quant config lives
    in the safetensors ``__metadata__`` as a JSON string keyed by module path
    (``layers.0.proj.q``). Returns ``{}`` when the file has no such metadata.
    """
    try:
        meta = handle.metadata() or {}
    except Exception:
        return {}
    raw = meta.get("_quantization_metadata")
    if not raw:
        return {}
    try:
        return json.loads(raw).get("layers", {}) or {}
    except (ValueError, TypeError):
        return {}


def _quant_conf(handle, names, key):
    """``(group_size, convrot)`` for a quantized weight (defaults when absent).

    A sibling ``X.comfy_quant`` tensor wins when present; otherwise the
    file-level ``_quantization_metadata`` is consulted (the MOSS DiT layout,
    which keys the config by module path rather than by tensor name).
    """
    conf_key = key[: -len(".weight")] + ".comfy_quant"
    if conf_key in names:
        conf = json.loads(bytes(handle.get_tensor(conf_key).to(torch.uint8).numpy().tobytes()).decode())
        return int(conf.get("convrot_groupsize", 256)), bool(conf.get("convrot"))
    name = _strip_component_prefix(key)
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    conf = _metadata_quant_layers(handle).get(name)
    if conf:
        return int(conf.get("convrot_groupsize", 256)), bool(conf.get("convrot"))
    return 0, False


def _load_weights(model, handle, dtype, tick=None):
    """Read the checkpoint into a ``{param name: tensor}`` state dict.

    A 9B model will not fit twice in RAM, and a single weight (the 248320x4096
    embedding) is ~1 GB int8 / ~4 GB float32, so every tensor is dequantized and
    materialised in bounded row chunks (see ``_copy_weight``). The returned dict
    is assigned into the (``meta``-built) model with ``assign=True``. ``tick`` is
    an optional ``tick(fraction)`` callback called as each weight is copied.
    """
    params = dict(model.named_parameters())
    state_keys = model.state_dict().keys()
    prefix = "language_model." if any(
        key.startswith("language_model.") for key in state_keys
    ) else ("model." if any(key.startswith("model.") for key in state_keys) else "")
    del state_keys

    names = set(handle.keys())
    keys = [key for key in sorted(names) if not key.endswith((".comfy_quant", "_scale"))]
    total = max(1, len(keys))
    loaded = set()
    state = {}
    if tick is not None:
        tick(0.0)
    with torch.no_grad():
        for position, key in enumerate(keys):
            name = _map_key(key, prefix)
            dst = params.get(name)
            if dst is not None:
                buf = torch.empty(dst.shape, dtype=dtype)
                _copy_weight(handle, names, key, buf, dtype)
                state[name] = buf
                loaded.add(name)
            if tick is not None and position % 8 == 0:
                tick((position + 1) / total)
    if tick is not None:
        tick(1.0)
    return state, [name for name in params if name not in loaded and "rotary" not in name]


class MossTextEncoder:
    """Qwen3.5-9B last hidden states for MOSS Image."""

    def __init__(self, weights_path, device="cpu", dtype=torch.bfloat16, repo=None,
                 tick=None):
        from transformers import AutoConfig, AutoTokenizer

        self.device = device
        self.dtype = dtype
        source = weights_path if os.path.isdir(weights_path) else (repo or _DEFAULT_REPO)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        config = AutoConfig.from_pretrained(source)
        self.missing = []

        if os.path.isdir(weights_path):
            if tick is not None:
                tick(0.5)
            with skip_random_init():
                model = build_on_dtype(dtype, lambda: _build_model(config))
            model = model.to(dtype=dtype)
            if tick is not None:
                tick(1.0)
        else:
            # Build on the ``meta`` device so no ~18 GB bf16 parameter storage is
            # allocated and no random initialiser runs: the checkpoint overwrites
            # every weight anyway, and the Qwen3.5 text model's initialisers cost
            # minutes over the ~9B parameters (``skip_random_init`` does not
            # intercept all of them). The loaded tensors are then *assigned* in
            # place of the empty parameters (``assign=True``), so the peak stays
            # near one copy of the weights.
            if tick is not None:
                tick(0.02)
            with build_on_meta(dtype, lambda: _build_model(config)) as model:
                pass
            with safe_open(weights_path, framework="pt", device="cpu") as handle:
                state, self.missing = _load_weights(model, handle, dtype, tick=tick)
            model.load_state_dict(state, strict=False, assign=True)
            model.tie_weights()
            del state

        # Materialise the loaded parameters in the target dtype/device. Parameters
        # the file did not carry stay on the ``meta`` device and are skipped:
        # ``model.to`` over the whole module would raise trying to copy their
        # non-existent storage.
        for parameter in model.parameters():
            if parameter.device.type != "meta" and (
                parameter.dtype != dtype or str(parameter.device) != str(device)
            ):
                parameter.data = parameter.to(device=device, dtype=dtype)
        model.eval()
        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, text):
        """Return the Qwen3.5 last hidden states ``[1, L, 4096]`` for ``text``."""
        input_ids = torch.tensor([self._ids(text or "")], device=self.device, dtype=torch.long)
        out = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
        return out.last_hidden_state

    def _ids(self, text):
        ids = self.tokenizer(self._template(text), add_special_tokens=False)["input_ids"]
        if not ids:
            ids = [self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0]
        return list(ids)

    def _template(self, text):
        """The open-think chat template (falls back to the raw string)."""
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
        except Exception:
            return f"{_PREFIX}{text}{_SUFFIX}"


def _map_key(key, prefix):
    """Map a text-encoder file key onto the built model's naming.

    The released files prefix the language model with ``model.language_model.``
    (the ComfyUI repack strips it to ``model.``); ``transformers`` builds a
    text-only model with bare keys (``embed_tokens.weight``) and a multimodal one
    with ``language_model.``. ``prefix`` is the form the built model uses.
    """
    if key.startswith("visual.") or key.startswith("model.visual."):
        return None
    for src in ("model.language_model.", "language_model.", "model."):
        if key.startswith(src):
            return prefix + key[len(src):]
    return key
