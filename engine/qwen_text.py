"""Qwen-Image 2.1 conditioning: Qwen3-VL-8B, text-only or with reference images.

Qwen-Image 2.1 conditions its DiT on the *last-layer hidden states* (before the
final norm) of the Qwen3-VL-8B model, run over a fixed template. The system turn
is tokenized and run, then dropped from the output; the DiT receives only the
user/assistant tokens. Matches ComfyUI's ``comfy/text_encoders/qwen_image21.py``
+ ``qwen3vl.py``.

Without reference images the prompt is a plain text template
(:meth:`Qwen3VLTextEncoder.encode`). With them, each condition image is placed in
the user turn as ``<image{i}><|vision_start|><|image_pad|><|vision_end|>`` and the
Qwen3-VL *vision* tower supplies those slots: :meth:`encode_with_images` returns
the hidden states plus the ``image_pad_mask`` marking the slots the DiT fills with
the condition latents (see :class:`engine.qwen_image.QwenRef`).

Weights load from a ``Qwen3VL-8B`` GGUF via :mod:`engine.gguf` or a Hugging
Face-layout safetensors file (Comfy-Org bf16 or ``*_int8_convrot``); the text
tower maps onto the model's ``language_model`` and the vision tower
(``model.visual``, including its deepstack mergers) loads when the file carries
it. A text-only GGUF (no ``model.visual.*`` tensors) still loads for text-to-image
but cannot condition on reference images; use a vision-carrying file for those.
The tokenizer reuses the Qwen2 BPE data already shipped for Anima
(:mod:`engine.tokenizer_data`). Uses ``transformers``'
``Qwen3VLForConditionalGeneration``, nothing from ComfyUI.
"""

import contextlib
import os

import torch

from . import gguf as gguf_reader
from .precision import build_on_meta

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOKENIZER_DATA = os.path.join(_HERE, "tokenizer_data")

IM_START = 151644
IM_END = 151645
PAD = 151643
VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655
VIDEO_PAD = 151656

SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"

# The template split around the prompt, so a prompt containing ``{``/``}`` is not
# reinterpreted by ``str.format``.
_USER_PREFIX = SYSTEM_PROMPT + "<|im_start|>user\n"
_USER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# One condition-image slot: ``<image{i}>`` names it, the vision markers bracket
# the placeholder the processor expands into the image's vision tokens.
_IMAGE_TEMPLATE = "<image{i}><|vision_start|><|image_pad|><|vision_end|>"

# The image-slot expansion: patch 16, 2x2 merge, so one merged token per 32
# pixels. The DiT then holds 4 latent tokens per slot (engine.qwen_image).
_PATCH = 16
_MERGE = 2
_SLOT_LATENT_TOKENS = 4

# llama.cpp -> transformers Qwen3VLTextModel suffixes for a ``blk.N.`` tensor.
_BLK_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

# llama.cpp -> transformers Qwen3VLVisionModel suffixes for a ``v.blk.N.`` tensor.
_VISION_BLK_MAP = {
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
    "attn_qkv.weight": "attn.qkv.weight",
    "attn_qkv.bias": "attn.qkv.bias",
    "attn_out.weight": "attn.proj.weight",
    "attn_out.bias": "attn.proj.bias",
    "ffn_gate.weight": "mlp.linear_fc1.weight",
    "ffn_gate.bias": "mlp.linear_fc1.bias",
    "ffn_down.weight": "mlp.linear_fc2.weight",
    "ffn_down.bias": "mlp.linear_fc2.bias",
}

# Plain top-level vision tensors, keyed by their llama.cpp name.
_VISION_TOP_MAP = {
    "v.patch_embd.weight": "patch_embed.proj.weight",
    "v.patch_embd.bias": "patch_embed.proj.bias",
    "v.position_embd.weight": "pos_embed.weight",
}


def _map_name(k):
    """Map a llama.cpp GGUF tensor name to a ``Qwen3VLTextModel`` name."""
    if k in ("output.weight",):  # lm_head, unused
        return None
    if k == "token_embd.weight":
        return "embed_tokens.weight"
    if k == "output_norm.weight":
        return "norm.weight"
    if k.startswith("blk."):
        idx, _, tail = k[len("blk."):].partition(".")
        mapped = _BLK_MAP.get(tail)
        return f"layers.{idx}.{mapped}" if mapped else None
    return None


def _map_vision_name(k):
    """Map a source tensor name to a ``Qwen3VLVisionModel`` name (``model.visual.``).

    Accepts the transformers/safetensors naming (``model.visual.*``) as-is and the
    llama.cpp GGUF naming (``v.*`` for the tower, ``mm.*`` for the merger). The
    merger mapping is best-effort: an unmapped key is reported as missing rather
    than silently dropped.
    """
    if k.startswith("model.visual."):
        return k
    if k.startswith("visual."):
        return "model." + k
    if k.startswith("v.blk."):
        idx, _, tail = k[len("v.blk."):].partition(".")
        mapped = _VISION_BLK_MAP.get(tail)
        return f"model.visual.blocks.{idx}.{mapped}" if mapped else None
    if k in _VISION_TOP_MAP:
        return "model.visual." + _VISION_TOP_MAP[k]
    if k == "v.post_ln.weight":  # pre-merger norm in some conversions
        return "model.visual.merger.norm.weight"
    if k == "v.post_ln.bias":
        return "model.visual.merger.norm.bias"
    if k.startswith("mm."):
        rest = k[len("mm."):]
        idx, _, tail = rest.partition(".")
        tail = tail or "weight"
        if idx == "0":
            return f"model.visual.merger.norm.{tail}"
        # The patch merger's two linears; the conversion may use mm.1/mm.2.
        linear = {"1": "linear_fc1", "2": "linear_fc2"}.get(idx)
        return f"model.visual.merger.{linear}.{tail}" if linear else None
    return None


def _text_config():
    from transformers import Qwen3VLTextConfig

    return Qwen3VLTextConfig(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]},
        attention_bias=False,
        # The full ``Qwen3VLForConditionalGeneration`` builds an ``lm_head``; tie it
        # to the embedding so the (unused) output projection costs no memory.
        tie_word_embeddings=True,
    )


def _vision_config():
    from transformers import Qwen3VLVisionConfig

    return Qwen3VLVisionConfig(
        depth=27,
        hidden_size=1152,
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=_PATCH,
        spatial_merge_size=_MERGE,
        temporal_patch_size=2,
        out_hidden_size=4096,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        hidden_act="gelu_pytorch_tanh",
    )


def _build_config():
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config=_text_config(),
        vision_config=_vision_config(),
        tie_word_embeddings=True,
        image_token_id=IMAGE_PAD,
        video_token_id=VIDEO_PAD,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
    )


def _safetensors_keys(name):
    """Candidate HF-layout keys for a ``Qwen3VLForConditionalGeneration`` param.

    The vision tower (``model.visual.*``) is stored under its own module path, so
    it maps directly. The language model's params are built under
    ``model.language_model.*`` but stored under ``model.*`` (occasionally
    ``language_model.*``) in the checkpoint. ``lm_head`` is tied to the embedding.
    """
    if name.startswith("model.visual."):
        return (name,)
    if name.startswith("model.language_model."):
        tail = name[len("model.language_model."):]
        return ("model." + tail, tail, "language_model." + tail)
    if name.startswith("lm_head."):
        return (name,)
    return ()


def _safetensors_weight(handle, names, key, shape, dtype):
    """Dequantize one safetensors tensor into a fresh tensor of ``shape``.

    Reuses :func:`engine.quant_formats._copy_weight` for the plain and ComfyUI int8
    (+ convrot) weights; fp8 weights are scaled through
    :func:`engine.krea2._fp8_reader`.
    """
    buf = torch.empty(shape, dtype=dtype)
    if handle.get_slice(key).get_dtype().startswith("F8"):
        from .krea2 import _fp8_reader

        buf.copy_(_fp8_reader(handle, names)(key).reshape(shape).to(dtype))
    else:
        from .quant_formats import _copy_weight

        _copy_weight(handle, names, key, buf, dtype)
    return buf


class Qwen3VLTextEncoder:
    """Qwen3-VL-8B last-hidden-state encoder for Qwen-Image 2.1.

    The text tower alone (:meth:`encode`) drives text-to-image and img2img. The
    vision tower is loaded alongside it when the weights file carries one; then
    :meth:`encode_with_images` conditions on reference images.
    """

    def __init__(self, weights_path, device="cpu", dtype=torch.bfloat16, tick=None):
        from transformers import Qwen2Tokenizer, Qwen3VLForConditionalGeneration

        self.device = device
        self.dtype = dtype
        self.tokenizer = Qwen2Tokenizer.from_pretrained(_TOKENIZER_DATA)

        # Build on the ``meta`` device: allocating and randomly initialising the
        # ~8.5B bf16 parameters only to overwrite every one from the checkpoint
        # costs minutes of CPU time (and ~17 GB of RAM) for nothing. The weights
        # are then *assigned* in place of the empty parameters (``assign=True``),
        # so no second copy is ever materialised.
        with build_on_meta(
            dtype, lambda: Qwen3VLForConditionalGeneration(_build_config())
        ) as model:
            pass
        params = dict(model.named_parameters())
        if os.path.splitext(weights_path)[1].lower() == ".safetensors":
            state = self._load_safetensors(params, weights_path, dtype, tick)
        else:
            state = self._load_gguf(params, weights_path, dtype, tick)

        # ``assign=True`` installs the loaded tensors as the parameters directly
        # instead of copying into the (non-existent) meta storage, and the unused
        # ``lm_head`` is re-tied to the embedding so nothing is left on meta.
        model.load_state_dict(state, strict=False, assign=True)
        model.tie_weights()
        loaded = set(state)

        # Materialise the loaded parameters in the target dtype/device. Parameters
        # the file did not carry (e.g. a text-only GGUF's vision tower) stay on the
        # meta device and are skipped: ``model.to()`` over the whole module would
        # raise trying to copy their non-existent storage. The text encoder runs on
        # CPU in this app, so this is a no-op that keeps a GPU request working too.
        for p in model.parameters():
            if p.device.type != "meta" and (p.dtype != dtype or str(p.device) != str(device)):
                p.data = p.to(device=device, dtype=dtype)
        model.eval()
        self.full = model
        for p in self.full.parameters():
            p.requires_grad_(False)
        # ``model`` is the text tower (the same contract the text-only path used);
        # ``full`` adds the vision tower the reference-image path needs.
        self.model = self.full.model.language_model
        # A param is "missing" only if the file has no tensor for it and it is not
        # the unused rotary cache or the tied ``lm_head`` (re-tied to the embedding
        # above); vision params are reported separately so a text-only file still
        # loads for t2i.
        vision_prefix = "model.visual."
        self.missing = [
            n for n in params
            if n not in loaded and "rotary_emb" not in n
            and not n.startswith("lm_head.") and not n.startswith(vision_prefix)
        ]
        self.vision_missing = [
            n for n in params
            if n not in loaded and n.startswith(vision_prefix) and "rotary" not in n
        ]
        self.has_vision = not self.vision_missing
        self._image_processor = None

    # -- weight loading -----------------------------------------------------
    @staticmethod
    def _load_gguf(params, weights_path, dtype, tick):
        """Dequantize a llama.cpp ``Qwen3VL`` GGUF into ``{param name: tensor}``."""
        reader = gguf_reader.gguf.GGUFReader(weights_path)
        tensors = list(reader.tensors)
        total = max(1, len(tensors))
        state = {}
        if tick is not None:
            tick(0.0)
        for position, tensor in enumerate(tensors):
            name = _map_name(tensor.name)
            if name is not None:
                name = "model.language_model." + name
            else:
                name = _map_vision_name(tensor.name)
            if name is None or name not in params:
                continue
            state[name] = gguf_reader.dequantize(tensor, dtype=dtype).reshape(params[name].shape)
            if tick is not None and position % 8 == 0:
                tick((position + 1) / total)
        if tick is not None:
            tick(1.0)
        return state

    @staticmethod
    def _load_safetensors(params, weights_path, dtype, tick):
        """Read a HF-layout safetensors (bf16/int8-convrot/fp8) into a state dict.

        Iterates the model's parameters (so only the tensors this build needs are
        read) and maps each to its checkpoint key (see :func:`_safetensors_keys`).
        """
        from safetensors import safe_open

        items = [
            (name, _safetensors_keys(name)) for name in params
        ]
        items = [(name, keys) for name, keys in items if keys]
        total = max(1, len(items))
        state = {}
        if tick is not None:
            tick(0.0)
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            names = set(handle.keys())
            for position, (name, keys) in enumerate(items):
                key = next((k for k in keys if k in names), None)
                if key is not None:
                    state[name] = _safetensors_weight(
                        handle, names, key, params[name].shape, dtype
                    )
                if tick is not None and position % 8 == 0:
                    tick((position + 1) / total)
        if tick is not None:
            tick(1.0)
        return state

    # -- text ---------------------------------------------------------------
    def _wrap(self, prompt):
        return f"{_USER_PREFIX}{prompt or ' '}{_USER_SUFFIX}"

    def _tokenize(self, prompt):
        return self.tokenizer(self._wrap(prompt), add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def encode(self, prompt):
        """Return the DiT text context ``[1, T, 4096]`` for ``prompt``.

        The prompt is tokenized verbatim: like ComfyUI's Qwen-Image text encoder
        (``disable_weights=True``), ``(text:weight)`` emphasis is not interpreted.
        """
        ids = self._tokenize(prompt or "")
        if not ids:
            ids = [PAD]
        hidden = self._forward(ids)
        return hidden[:, self._drop_from(ids):]

    @contextlib.contextmanager
    def _pre_norm(self):
        """Yield with the text tower's final RMSNorm neutralized for this call.

        Qwen-Image 2.1 conditions on the last decoder layer's output *before* the
        final RMSNorm. From transformers 5.0 the encoder's captured
        ``hidden_states[-1]`` is tied to ``last_hidden_state`` and so comes back
        *post*-norm (a third of the signal the DiT was trained on, which shows up
        as the prompt being largely ignored). A forward hook returning the norm's
        input replaces its output just for this forward, restoring the pre-norm
        states -- the same workaround the diffusers reference uses
        (``pipeline_qwenimage21.py``).
        """
        handle = self.model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            yield
        finally:
            handle.remove()

    def _forward(self, ids):
        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        with self._pre_norm():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
        return out.hidden_states[-1]  # last layer, before final norm

    @staticmethod
    def _drop_from(ids):
        """Index of the first user token: everything before the second ``<|im_start|>``."""
        starts = [i for i, t in enumerate(ids) if t == IM_START]
        return starts[1] if len(starts) > 1 else 0

    # -- reference images ---------------------------------------------------
    def _make_processor(self):
        from transformers import Qwen2VLImageProcessor

        self._image_processor = Qwen2VLImageProcessor(
            patch_size=_PATCH, temporal_patch_size=2, merge_size=_MERGE,
        )
        return self._image_processor

    def _resize_condition(self, image, target_area):
        """Resize one condition image to ``target_area`` keeping aspect, on a 32 px grid."""
        from PIL import Image

        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        factor = (target_area / max(1, width * height)) ** 0.5
        grid = _PATCH * _MERGE
        new_w = max(grid, round(width * factor / grid) * grid)
        new_h = max(grid, round(height * factor / grid) * grid)
        if (new_w, new_h) != (width, height):
            image = image.resize((new_w, new_h), Image.LANCZOS)
        return image

    @torch.no_grad()
    def encode_with_images(self, prompt, images, target_area):
        """Encode ``prompt`` together with ``images`` (a list of PIL images).

        Returns ``(hidden, image_pad_mask, sizes)``: the last-layer hidden states
        ``[1, T, 4096]`` (system turn dropped), the ``[1, T]`` bool mask of its
        image slots, and each image's pixel size ``(w, h)`` as the vision tower
        saw it. The caller must VAE encode each image resized to that size, so the
        vision slots and the condition latents agree regardless of the
        processor's own resize.
        """
        if not self.has_vision:
            raise RuntimeError(
                "Qwen-Image reference conditioning needs the Qwen3-VL vision tower, "
                "which this text-encoder file does not carry (missing "
                f"{len(self.vision_missing)} vision tensors). Use a vision-capable "
                "Qwen3VL file (a full HF safetensors, or a GGUF that carries the "
                "vision tower) to use reference images."
            )
        processor = self._image_processor or self._make_processor()
        images = [self._resize_condition(im, target_area) for im in images]

        pix = processor(images=images, return_tensors="pt")
        pixel_values = pix["pixel_values"].to(self.device, self.dtype)
        grid = pix["image_grid_thw"].to(self.device)

        pieces = []
        for index, (t, h, w) in enumerate(grid.tolist()):
            slots = (t * h * w) // (_MERGE * _MERGE)
            piece = _IMAGE_TEMPLATE.format(i=index + 1)
            pieces.append(piece.replace("<|image_pad|>", "<|image_pad|>" * slots))
        text = f"{_USER_PREFIX}{''.join(pieces)}{prompt or ' '}{_USER_SUFFIX}"
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        # Multimodal token-type ids for M-RoPE: 1 on the image-pad slots, 0 on text
        # (the values the processor would emit alongside the expanded input_ids).
        mm_token_type_ids = (input_ids == IMAGE_PAD).long()

        with self._pre_norm():
            out = self.full(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=grid,
                mm_token_type_ids=mm_token_type_ids,
                output_hidden_states=True,
            )
        # ``hidden_states[-1]`` is the last decoder layer's output, before the
        # final RMSNorm (kept pre-norm by :meth:`_pre_norm`).
        hidden = out.hidden_states[-1]
        drop = self._drop_from(ids)
        image_pad_mask = (input_ids[0] == IMAGE_PAD)[drop:]
        sizes = [(int(w) * _PATCH, int(h) * _PATCH) for _t, h, w in grid.tolist()]
        return hidden[:, drop:], image_pad_mask.unsqueeze(0), sizes


def load_text_encoder(path, device="cpu", dtype=torch.bfloat16, tick=None):
    return Qwen3VLTextEncoder(path, device=device, dtype=dtype, tick=tick)
