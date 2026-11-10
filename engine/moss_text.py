"""MOSS Image text conditioning: Qwen3.5-9B last hidden states.

MOSS Image conditions its DiT on the **last** hidden states of a stock
**Qwen3.5-9B** text model (``text_dim = 4096``), tokenised with the Qwen3.5
tokenizer and the chat template with the thinking block left open (ComfyUI
``comfy/text_encoders/moss_image.py``: the ``enable_thinking=True`` template
trims the user turn and leaves an open `` thinking`` block):

    <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n thinking\\n

The model is a dense hybrid linear-attention transformer (``Qwen3_5TextModel`` /
``Qwen3_5Model`` in ``transformers`` >= 5.3). Weights may be a plain bf16
``.safetensors`` or any of ComfyUI's text-encoder quantisations; the shared
codec in :mod:`engine.quant_formats` handles ``int8_tensorwise`` (+convrot),
packed ``nvfp4`` and ``asym_w4a8_int8`` (the latter is the
``qwen3.5_9b_w4a8_gptq`` text encoder: packed int4 codes, a Lloyd-Max codebook,
fp8 group scales and ConvRot). The config and
tokenizer come from a Hugging Face
repo id (or a local directory) -- point ``repo`` at it, or pass a directory as
``weights_path`` to load everything from there.
"""

import os

import torch
from safetensors import safe_open

from .precision import build_on_dtype, build_on_meta, skip_random_init
from .quant_formats import _copy_weight

# Where the Qwen3.5-9B config and tokenizer are read from when the model is
# given as a bare weights file. A local directory passed as ``weights_path``
# overrides this entirely.
_DEFAULT_REPO = "Qwen/Qwen3.5-9B"

_PREFIX = "<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n thinking\n"


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
