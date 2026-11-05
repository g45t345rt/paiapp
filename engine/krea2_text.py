"""Krea 2 text conditioning: Qwen3-VL-4B, a 12-layer hidden-state stack.

Krea 2 conditions its DiT on a stack of hidden states tapped from 12 layers of a
Qwen3-VL-4B language model (reference taps ``hidden_states[2, 5, 8, ..., 35]``),
shaped ``(B, num_text_layers, seq, text_dim)`` and consumed by the DiT's internal
``txtfusion`` adapter. The stack is flattened to ``(B, seq, num_text_layers *
text_dim)`` here and unpacked inside the model (matching ComfyUI's ``krea2`` text
encoder).

Prompt handling follows the reference: a marked Qwen chat prefix
(``<|im_start|>system … <|im_end|>\n<|im_start|>user\n``, 34 tokens, dropped from
the output), the prompt padded to a fixed 512-token block, then the
``<|im_end|>\n<|im_start|>assistant\n`` suffix (5 tokens) appended after the
padding. Positions count only real tokens, so the padded block does not shift the
suffix's mRoPE phase (see the diffusers ``Krea2Pipeline.get_text_hidden_states``).

Weights load one tensor at a time from a Qwen3-VL-4B GGUF (llama.cpp tensor
names, via :mod:`engine.gguf`) or a Hugging Face-layout safetensors file. The
model is built on the ``meta`` device (no parameter storage, no random init) and
each loaded tensor is *assigned* into it directly. Uses ``transformers``'
``Qwen3VLTextModel``; nothing from ComfyUI.
"""

import torch

from . import gguf as gguf_reader
from .qwen_text import _TOKENIZER_DATA, _map_name

# Qwen chat-template wrapper around the Krea 2 system/user prefix. The
# ``<|im_start|>``/``<|im_end|>`` markers are part of the conditioning: the
# marked prefix tokenises to exactly 34 tokens (dropped from the output) and the
# suffix to 5, matching the reference workflow.
PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# Taps into the encoder's ``hidden_states`` tuple (0 is the embedding output).
TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

_MAX_SEQUENCE_LENGTH = 512

# Container prefixes a HF-layout safetensors file may carry before the text
# tower's own module path.
_PREFIXES = (
    "model.language_model.", "language_model.", "text_model.", "transformer.",
    "model.",
)


def _strip_one_prefix(key):
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _weights_source(path, dtype):
    """A ``(names, shape_of, get)`` triple that reads one tensor at a time."""
    if path.lower().endswith(".gguf"):
        reader = gguf_reader.gguf.GGUFReader(path)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        return (
            set(tensors),
            lambda key: tuple(int(v) for v in reversed(tensors[key].shape)),
            lambda key: gguf_reader.dequantize(tensors[key], dtype=dtype),
        )
    from safetensors import safe_open

    from .krea2 import _fp8_reader

    handle = safe_open(path, framework="pt", device="cpu")
    names = set(handle.keys())
    return (
        names,
        lambda key: tuple(handle.get_slice(key).get_shape()),
        _fp8_reader(handle, names),
    )


def _param_index(names):
    """Map ``Qwen3VLTextModel`` parameter names to this file's keys.

    A GGUF file uses llama.cpp ``blk.N.*`` names (:func:`engine.qwen_text._map_name`);
    a Hugging Face safetensors file uses the module path, optionally behind a
    container prefix (``model.language_model.`` and friends).
    """
    index = {}
    for key in names:
        mapped = _map_name(key)
        if mapped is None:
            mapped = _strip_one_prefix(key)
        index.setdefault(mapped, key)
    return index


def _detect_config(shape):
    """Fill a ``Qwen3VLTextConfig`` from the checkpoint's tensor shapes.

    ``shape`` maps a ``Qwen3VLTextModel`` parameter name to its shape (or
    ``None`` when the file has no such tensor).
    """
    from transformers import Qwen3VLTextConfig

    norm = shape("layers.0.input_layernorm.weight")
    hidden = int(norm[0]) if norm else 2560
    gate = shape("layers.0.mlp.gate_proj.weight")
    intermediate = int(gate[0]) if gate else 9728
    q_norm = shape("layers.0.self_attn.q_norm.weight")
    head_dim = int(q_norm[0]) if q_norm else 128
    q_proj = shape("layers.0.self_attn.q_proj.weight")
    heads = int(q_proj[0]) // head_dim if q_proj else 32
    k_proj = shape("layers.0.self_attn.k_proj.weight")
    kv_heads = int(k_proj[0]) // head_dim if k_proj else 8
    embedding = shape("embed_tokens.weight")
    vocab = int(embedding[0]) if embedding else 151936
    layer_ids = []
    for layer in range(1, 512):
        key = f"layers.{layer}.input_layernorm.weight"
        if shape(key) is not None:
            layer_ids.append(layer)
    num_layers = 1 + max(layer_ids) if layer_ids else 36

    return Qwen3VLTextConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]},
        attention_bias=False,
        tie_word_embeddings=False,
    )


class Krea2TextEncoder:
    """Qwen3-VL-4B 12-layer hidden-state stack for Krea 2."""

    def __init__(self, weights_path, device="cpu", dtype=torch.bfloat16, tick=None):
        from transformers import Qwen2Tokenizer, Qwen3VLTextModel

        from .precision import build_on_meta

        self.device = device
        self.dtype = dtype
        self.tokenizer = Qwen2Tokenizer.from_pretrained(_TOKENIZER_DATA)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 151643

        names, shape_of, get = _weights_source(weights_path, dtype)
        index = _param_index(names)

        def shape(name):
            key = index.get(name)
            return shape_of(key) if key is not None else None

        # Build on the ``meta`` device so no parameter storage is allocated and no
        # random initialiser runs: the checkpoint overwrites every weight anyway,
        # and the bf16 ``normal_``/``uniform_`` CPU kernels cost minutes over the
        # ~4B parameters. The loaded tensors are then *assigned* in place of the
        # empty parameters (``assign=True``), so no second copy is materialised.
        if tick is not None:
            tick(0.1)
        with build_on_meta(
            dtype, lambda: Qwen3VLTextModel(_detect_config(shape))
        ) as model:
            pass
        params = dict(model.named_parameters())
        total = max(1, len(params))
        state = {}
        for position, name in enumerate(params):
            if tick is not None and position % 16 == 0:
                tick(0.1 + 0.9 * (position / total))
            key = index.get(name)
            if key is None:
                continue
            src = get(key)
            state[name] = src.reshape(params[name].shape).to(dtype)
            del src
        if tick is not None:
            tick(1.0)

        model.load_state_dict(state, strict=False, assign=True)
        model.tie_weights()
        loaded = set(state)
        self.missing = [n for n in params if n not in loaded and "rotary" not in n]
        self.unexpected = []

        # Materialise the loaded parameters in the target dtype/device. Parameters
        # the file did not carry stay on the ``meta`` device and are skipped:
        # ``model.to`` over the whole module would raise trying to copy their
        # non-existent storage. The text encoder runs on CPU in this app, so this
        # is a no-op that keeps a GPU request working too.
        for p in model.parameters():
            if p.device.type != "meta" and (p.dtype != dtype or str(p.device) != str(device)):
                p.data = p.to(device=device, dtype=dtype)
        model.eval()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._prefix_len = len(self._encode_ids(PREFIX))
        self._suffix_width = len(self._encode_ids(SUFFIX))

    def _encode_ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def encode(self, prompt):
        """Return ``(hidden, mask)``: the 12-layer stack ``[1, T, 12 * 2560]`` and
        its boolean token mask ``[1, T]``, both sliced past the system prefix.

        The prompt is tokenized verbatim: like ComfyUI's Krea 2 text encoder
        (``disable_weights=True``), ``(text:weight)`` emphasis is not interpreted.
        """
        hidden, attention_mask, _ids = self._forward(prompt or "")
        start = self._prefix_len
        return hidden[:, start:], attention_mask[:, start:]

    def _forward(self, prompt):
        start = self._prefix_len
        max_length = _MAX_SEQUENCE_LENGTH + start - self._suffix_width
        text = self.tokenizer(
            [PREFIX + (prompt or "")], truncation=True, padding="max_length",
            max_length=max_length, add_special_tokens=False, return_tensors="pt",
        )
        suffix = self.tokenizer([SUFFIX], add_special_tokens=False, return_tensors="pt")

        input_ids = torch.cat([text["input_ids"], suffix["input_ids"]], dim=1).to(self.device)
        attention_mask = torch.cat(
            [text["attention_mask"], suffix["attention_mask"]], dim=1
        ).bool().to(self.device)

        # Positions count only real tokens (padding consumes no position), matching
        # how the model was sampled; broadcast across the 3 mRoPE axes.
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, output_hidden_states=True,
        )
        if len(out.hidden_states) <= max(TAP_LAYERS):
            raise RuntimeError(
                f"Krea 2 text encoder exposes {len(out.hidden_states) - 1} layers but the "
                f"12-layer stack needs at least {max(TAP_LAYERS) + 1} (a Qwen3-VL-4B text tower)."
            )
        hidden = torch.stack([out.hidden_states[i] for i in TAP_LAYERS], dim=2)
        batch, seq, num_layers, dim = hidden.shape
        return hidden.reshape(batch, seq, num_layers * dim), attention_mask, input_ids[0].tolist()


def load_text_encoder(path, device="cpu", dtype=torch.bfloat16, tick=None):
    return Krea2TextEncoder(path, device=device, dtype=dtype, tick=tick)
