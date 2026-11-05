"""Krea 2 vision-language reference conditioning ("reference in prompt").

The Krea 2 functional adapters (AnyPaint) were trained with the reference image
embedded in the *text* conditioning through the Qwen3-VL vision tower, in addition
to the high-resolution reference latents. This module builds that encoder: the
full Qwen3-VL-4B model (vision tower + language model) from the same single-file
checkpoint :mod:`engine.krea2_text` reads, plus a Qwen3-VL image processor, and
returns the same 12-layer hidden-state stack the DiT's ``txtfusion`` consumes.

The reference image rides in the user message as a named vision placeholder
(``Picture 1: <|vision_start|><|image_pad|><|vision_end|>``); the processor expands
the pad to the image's token grid and emits ``mm_token_type_ids`` so the language
model applies M-RoPE over the image tokens.
"""

from __future__ import annotations

import torch

from .krea2_text import PREFIX, SUFFIX, TAP_LAYERS
from .qwen_text import _TOKENIZER_DATA

# Qwen3-VL-4B vision tower, as measured from the checkpoint (see the config at
# Qwen/Qwen3-VL-4B-Instruct). Falls back to these when a shape is unavailable.
_VISION = dict(
    depth=24, hidden_size=1024, intermediate_size=4096, num_heads=16,
    in_channels=3, patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
    out_hidden_size=2560, num_position_embeddings=2304,
    hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[5, 11, 17],
)

# The reference image is pre-scaled to this area for the vision pass.
_VL_MAX_PIXELS = 384 * 384

_cache = {}


def _vl_param_index(names):
    """Map Qwen3-VL model parameter names to this file's keys.

    A ComfyUI-repacked file stores the language tower at ``model.*`` and the
    vision tower at ``model.visual.*``; the ``Qwen3VLModel`` names them
    ``language_model.*`` and ``visual.*``.
    """
    index = {}
    for key in names:
        if key.endswith((".weight_scale", ".comfy_quant", ".input_scale")):
            continue
        if key.startswith("model.visual."):
            param = key[len("model."):]
        elif key.startswith("model."):
            param = "language_model." + key[len("model."):]
        elif key.startswith("visual."):
            param = key
        else:
            param = "language_model." + key
        index.setdefault(param, key)
    return index


def _build_config(shape):
    from transformers import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig

    hidden = (shape("language_model.layers.0.input_layernorm.weight") or (2560,))[0]
    gate = shape("language_model.layers.0.mlp.gate_proj.weight")
    intermediate = int(gate[0]) if gate else 9728
    q_norm = shape("language_model.layers.0.self_attn.q_norm.weight")
    head_dim = int(q_norm[0]) if q_norm else 128
    q_proj = shape("language_model.layers.0.self_attn.q_proj.weight")
    heads = int(q_proj[0]) // head_dim if q_proj else 32
    k_proj = shape("language_model.layers.0.self_attn.k_proj.weight")
    kv_heads = int(k_proj[0]) // head_dim if k_proj else 8
    embedding = shape("language_model.embed_tokens.weight")
    vocab = int(embedding[0]) if embedding else 151936
    layer_ids = [
        n for n in range(1, 512)
        if shape(f"language_model.layers.{n}.input_layernorm.weight") is not None
    ]
    num_layers = 1 + max(layer_ids) if layer_ids else 36

    text = Qwen3VLTextConfig(
        vocab_size=vocab, hidden_size=int(hidden), intermediate_size=intermediate,
        num_hidden_layers=num_layers, num_attention_heads=heads,
        num_key_value_heads=kv_heads, head_dim=head_dim, hidden_act="silu",
        max_position_embeddings=262144, rms_norm_eps=1e-6, rope_theta=5000000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20],
                      "mrope_interleaved": True},
        attention_bias=False, tie_word_embeddings=True,
    )
    vision = Qwen3VLVisionConfig(**_VISION)
    return Qwen3VLConfig(
        vision_config=vision, text_config=text, image_token_id=151655,
        video_token_id=151656, vision_start_token_id=151652,
        vision_end_token_id=151653,
    )


class Krea2VLEncoder:
    """Qwen3-VL-4B that encodes ``(prompt, reference image)`` into the DiT stack."""

    def __init__(self, weights_path, device="cpu", dtype=torch.bfloat16, tick=None):
        from safetensors import safe_open
        from transformers import (
            Qwen2Tokenizer, Qwen2VLImageProcessor, Qwen3VLModel, Qwen3VLProcessor,
            Qwen3VLVideoProcessor,
        )

        from .krea2 import _fp8_reader
        from .precision import build_on_meta

        self.device = device
        self.dtype = dtype

        handle = safe_open(weights_path, framework="pt", device="cpu")
        names = set(handle.keys())
        index = _vl_param_index(names)
        get = _fp8_reader(handle, names)

        def shape(name):
            key = index.get(name)
            return tuple(handle.get_slice(key).get_shape()) if key is not None else None

        if tick is not None:
            tick(0.1)
        config = _build_config(shape)
        with build_on_meta(dtype, lambda: Qwen3VLModel(config)) as model:
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
            state[name] = get(key).reshape(params[name].shape).to(dtype)
        if tick is not None:
            tick(1.0)
        model.load_state_dict(state, strict=False, assign=True)
        model.tie_weights()
        self.missing = [
            n for n in params if n not in state and "rotary" not in n
        ]
        for p in model.parameters():
            if p.device.type != "meta" and (
                p.dtype != dtype or str(p.device) != str(device)
            ):
                p.data = p.to(device=device, dtype=dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        tokenizer = Qwen2Tokenizer.from_pretrained(_TOKENIZER_DATA)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 151643
        self.tokenizer = tokenizer
        image_processor = Qwen2VLImageProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2,
            image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
            min_pixels=4 * 32 * 32, max_pixels=_VL_MAX_PIXELS,
        )
        video_processor = Qwen3VLVideoProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2,
            image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
        )
        self.processor = Qwen3VLProcessor(
            image_processor=image_processor, tokenizer=tokenizer,
            video_processor=video_processor,
        )
        self._prefix_len = len(
            tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
        )

    @torch.no_grad()
    def encode(self, prompt, image):
        """Return ``(hidden, mask)`` like :meth:`engine.krea2_text.Krea2TextEncoder.encode`,
        with the reference ``image`` embedded in the prompt via the vision tower."""
        image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        text = PREFIX + image_prompt + (prompt or "")
        inputs = self.processor(
            text=[text], images=[image.convert("RGB")], return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].bool().to(self.device)
        mm_types = inputs["mm_token_type_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        grid = inputs["image_grid_thw"].to(self.device)

        suffix = self.tokenizer(
            [SUFFIX], add_special_tokens=False, return_tensors="pt"
        )
        suffix_ids = suffix["input_ids"].to(self.device)
        suffix_mask = suffix["attention_mask"].bool().to(self.device)
        input_ids = torch.cat([input_ids, suffix_ids], dim=1)
        attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)
        mm_types = torch.cat(
            [mm_types, torch.zeros_like(suffix_ids, dtype=mm_types.dtype)], dim=1
        )

        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=grid,
            mm_token_type_ids=mm_types, output_hidden_states=True,
        )
        if len(out.hidden_states) <= max(TAP_LAYERS):
            raise RuntimeError("Qwen3-VL encoder exposed too few layers for the stack.")
        hidden = torch.stack([out.hidden_states[i] for i in TAP_LAYERS], dim=2)
        batch, seq, layers, dim = hidden.shape
        hidden = hidden.reshape(batch, seq, layers * dim)
        start = self._prefix_len
        return hidden[:, start:], attention_mask[:, start:]


def load_vl_encoder(weights_path, device="cpu", dtype=torch.bfloat16, tick=None):
    """Build (and cache) the Krea 2 Qwen3-VL reference encoder for ``weights_path``."""
    key = (weights_path, str(device), str(dtype))
    encoder = _cache.get(key)
    if encoder is None:
        encoder = Krea2VLEncoder(
            weights_path, device=device, dtype=dtype, tick=tick
        )
        _cache[key] = encoder
    return encoder
