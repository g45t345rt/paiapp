"""Classic ``Load Checkpoint`` support.

A classic checkpoint is a single ``.safetensors`` file that bundles the UNet,
the CLIP text encoder(s) and the VAE (SD 1.x / SD 2.x / SDXL, including the
many "Pony"/"Illustrious"/"Pixma" style finetunes). This module loads that raw
LDM/ComfyUI layout into ``diffusers`` modules using the *bundled* model configs
below, so no config download is needed for the architecture itself.

Only ``diffusers``, ``transformers``, ``torch`` and ``safetensors`` are used;
no ComfyUI code is imported. The diffusers conversion helpers translate the
``model.diffusion_model.*`` / ``conditioner.*`` / ``cond_stage_model.*`` /
``first_stage_model.*`` tensors into diffusers state dicts.
"""

import importlib.machinery
import importlib.util
import sys
import types

import torch

from . import prompt_weights
from .paths import arch_from_keys, checkpoint_arch, read_safetensors_keys


def _stub_broken_xformers():
    """diffusers imports ``xformers.ops`` if the package is present.

    Some installs ship an ``xformers`` whose native extension is incompatible
    with the active torch build, which makes that otherwise-optional import
    raise. We never select an xformers attention processor, so a spec-correct
    stub module is enough to keep diffusers importable.
    """
    if importlib.util.find_spec("xformers") is None:
        return
    try:
        import xformers.ops  # noqa: F401
        return
    except Exception:
        pass
    for name in ("xformers", "xformers.ops"):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        module.__path__ = []
        sys.modules[name] = module


_stub_broken_xformers()

from diffusers import AutoencoderKL, UNet2DConditionModel  # noqa: E402
from diffusers.loaders.single_file_utils import (  # noqa: E402
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_open_clip_checkpoint,
)
from transformers import (  # noqa: E402
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

_CLIP_REPO = "openai/clip-vit-large-patch14"

# ---------------------------------------------------------------------------
# Model configs (verbatim from the canonical SD 1.5 / SDXL releases).
# ---------------------------------------------------------------------------

_UNET_SD15 = {
    "in_channels": 4,
    "out_channels": 4,
    "sample_size": 64,
    "down_block_types": (
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D", "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D", "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
    ),
    "block_out_channels": (320, 640, 1280, 1280),
    "layers_per_block": 2,
    "cross_attention_dim": 768,
    "attention_head_dim": 8,
    "norm_num_groups": 32,
    "downsample_padding": 1,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "norm_eps": 1e-5,
    "use_linear_projection": False,
    "upcast_attention": False,
}

_UNET_SDXL = {
    "in_channels": 4,
    "out_channels": 4,
    "sample_size": 128,
    "down_block_types": (
        "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
    ),
    "up_block_types": (
        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D",
    ),
    "block_out_channels": (320, 640, 1280),
    "layers_per_block": 2,
    "cross_attention_dim": 2048,
    "attention_head_dim": (5, 10, 20),
    "transformer_layers_per_block": (1, 2, 10),
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 2816,
    "norm_num_groups": 32,
    "downsample_padding": 1,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "norm_eps": 1e-5,
    "use_linear_projection": True,
    "upcast_attention": False,
}

_VAE = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "latent_channels": 4,
    "norm_num_groups": 32,
    "sample_size": 512,
}

_TEXT_L = {
    "vocab_size": 49408,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": 77,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "projection_dim": 768,
}

_TEXT_G = {
    "vocab_size": 49408,
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_hidden_layers": 32,
    "num_attention_heads": 20,
    "max_position_embeddings": 77,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "projection_dim": 1280,
}


def _clip_text_state_dict(checkpoint, prefix):
    """LDM CLIP tensors keyed for a plain ``CLIPTextModel``.

    ``diffusers``' ``convert_ldm_clip_checkpoint`` strips only the LDM prefix, so
    its keys keep the ``text_model.`` submodule wrapper (``text_model.embeddings.
    ...``). transformers 5 flattens ``CLIPTextModel``: its ``embeddings`` and
    ``encoder`` sit at the top level with no ``text_model.`` wrapper, so leaving
    that prefix makes every converted key unexpected and the whole encoder
    silently stays randomly initialized (the image is then driven by the noise
    schedule alone). The nested ``CLIPTextModelWithProjection`` used for SDXL's
    second encoder still keeps the wrapper, so only the plain encoder is unwrapped
    here.
    """
    state = convert_ldm_clip_checkpoint(checkpoint, remove_prefix=prefix)
    return {
        (key[len("text_model."):] if key.startswith("text_model.") else key): value
        for key, value in state.items()
    }


def read_header_keys(path):
    """Return the tensor key names from a ``.safetensors`` file (no tensors)."""
    return sorted(read_safetensors_keys(path))


def detect_family(path_or_keys):
    """Return ``"sdxl"`` or ``"sd15"`` for a classic checkpoint."""
    if isinstance(path_or_keys, str):
        return checkpoint_arch(path_or_keys)
    keyset = path_or_keys if isinstance(path_or_keys, set) else set(path_or_keys)
    return arch_from_keys(keyset)


class ClassicCheckpoint:
    """A loaded classic checkpoint: UNet + text encoder(s) + VAE + tokenizer."""

    def __init__(self, path, device="cpu", unet_dtype=torch.float32, vae_dtype=torch.float32):
        from safetensors.torch import load_file

        self.path = path
        self.device = device
        self.family = detect_family(path)
        self.scale_factor = 0.13025 if self.family == "sdxl" else 0.18215

        state = load_file(path, device="cpu")

        # --- UNet ---------------------------------------------------------
        self.unet_config = _UNET_SDXL if self.family == "sdxl" else _UNET_SD15
        unet = UNet2DConditionModel.from_config(self.unet_config)
        unet_sd = convert_ldm_unet_checkpoint(state, self.unet_config)
        unet.load_state_dict(unet_sd, strict=False)
        self.unet = unet.to(device=device, dtype=unet_dtype).eval()

        # --- VAE ----------------------------------------------------------
        vae_config = dict(_VAE)
        vae_config["scaling_factor"] = self.scale_factor
        vae = AutoencoderKL.from_config(vae_config)
        vae.load_state_dict(convert_ldm_vae_checkpoint(state, _VAE), strict=False)
        self.vae = vae.to(device=device, dtype=vae_dtype).eval()

        # --- Text encoders ------------------------------------------------
        self.tokenizer = CLIPTokenizer.from_pretrained(_CLIP_REPO)
        if self.family == "sdxl":
            te1 = CLIPTextModel(CLIPTextConfig(**_TEXT_L))
            te1.load_state_dict(
                _clip_text_state_dict(state, "conditioner.embedders.0.transformer."),
                strict=False,
            )
            te2 = CLIPTextModelWithProjection(CLIPTextConfig(**_TEXT_G))
            te2.load_state_dict(
                convert_open_clip_checkpoint(te2, state, prefix="conditioner.embedders.1.model."),
                strict=False,
            )
            self.text_encoders = [te1.to(device=device, dtype=unet_dtype).eval(),
                                  te2.to(device=device, dtype=unet_dtype).eval()]
        else:
            te = CLIPTextModel(CLIPTextConfig(**_TEXT_L))
            te.load_state_dict(
                _clip_text_state_dict(state, "cond_stage_model.transformer."),
                strict=False,
            )
            self.text_encoders = [te.to(device=device, dtype=unet_dtype).eval()]

        del state
        for enc in self.text_encoders:
            for p in enc.parameters():
                p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @property
    def is_sdxl(self):
        return self.family == "sdxl"

    @torch.no_grad()
    def encode(self, text):
        """Tokenize ``text`` and return the UNet conditioning for it.

        ``(text:weight)`` emphasis is applied to the token hidden states (see
        :mod:`engine.prompt_weights`); a weight of 1.0 is a no-op and the pooled
        vector is left unweighted, matching ComfyUI.

        Returns ``{"prompt_embeds": Tensor[B,77,C], "pooled": Tensor[B,D]|None}``.
        """
        pairs = prompt_weights.parse(text or "")
        clean = prompt_weights.plain(pairs)
        embeds, pooled, ids = self._encode(clean)
        if prompt_weights.has_weights(pairs):
            empty, _, _ = self._encode("")
            weights = prompt_weights.token_weights(
                self.tokenizer, clean, pairs, target_ids=ids,
            )
            embeds = prompt_weights.interpolate(embeds, empty, weights)
        return {"prompt_embeds": embeds, "pooled": pooled}

    def _encode(self, text):
        tokens = self.tokenizer(
            text or "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        batch = tokens.input_ids.to(self.device)

        if self.is_sdxl:
            out1 = self.text_encoders[0](batch, output_hidden_states=True)
            out2 = self.text_encoders[1](batch, output_hidden_states=True)
            embeds = torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
            pooled = out2[0]
        else:
            out = self.text_encoders[0](batch)
            embeds = out.last_hidden_state
            pooled = None
        return embeds, pooled, batch[0].tolist()

    def time_ids(self, height, width):
        """SDXL ``add_time_ids`` for a full (uncropped) generation."""
        if not self.is_sdxl:
            return None
        ids = [height, width, 0, 0, height, width]
        return torch.tensor([ids], device=self.device, dtype=self.unet.dtype)


def install_region_processors(unet, mask_provider):
    """Install cross-attention processors that apply a per-region mask.

    ``mask_provider`` maps a cross-attention layer's query-token count to a
    boolean ``[1, 1, tokens, context]`` mask (or ``None`` to leave that layer
    unmasked). Only ``attn2`` (cross-attention) layers are replaced; returns the
    previous processors so :func:`clear_region_processors` can restore them.
    """
    from diffusers.models.attention_processor import AttnProcessor2_0

    class _RegionProcessor(AttnProcessor2_0):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, temb=None, *args, **kwargs):
            if encoder_hidden_states is not None:
                tokens = hidden_states.shape[-2]
                if hidden_states.ndim == 4:
                    tokens = hidden_states.shape[2] * hidden_states.shape[3]
                mask = mask_provider(tokens)
                if mask is not None:
                    # The regional mask is an additive bias, so it must share
                    # the hidden states' dtype (SDPA rejects a mismatch).
                    if mask.dtype != torch.bool and mask.dtype != hidden_states.dtype:
                        mask = mask.to(hidden_states.dtype)
                    attention_mask = mask
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                temb, *args, **kwargs,
            )

    previous = dict(unet.attn_processors)
    processors = {}
    for name, processor in previous.items():
        if name.endswith("attn2.processor"):
            processors[name] = _RegionProcessor()
        else:
            processors[name] = processor
    unet.set_attn_processor(processors)
    return previous


def clear_region_processors(unet, previous):
    """Restore the attention processors saved by :func:`install_region_processors`."""
    if previous:
        unet.set_attn_processor(previous)
