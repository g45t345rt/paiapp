"""ControlNet loading for the classic SD 1.x / SDXL family.

ControlNet adds a second, smaller network beside the UNet: it reads a "control
hint" (a canny edge map, a pose skeleton, a depth map, ...) and injects residuals
into the UNet's down and middle blocks, so a generation can follow a structure
the prompt alone cannot pin down. This module loads the ControlNet model; the
hint is prepared by :mod:`engine.preprocess` and the residuals are added by
:class:`engine.sampler.UNetApplyModel`.

Files come in two layouts, both handled here:

* the original LDM / ComfyUI single file -- ``input_blocks.*``,
  ``middle_block.*``, ``input_hint_block.*``, ``zero_convs.*``, ``time_embed.*``,
  optionally behind a ``control_model.`` prefix -- converted with diffusers' own
  ``convert_controlnet_checkpoint``;
* an already-diffusers single file -- ``down_blocks.*``, ``controlnet_*``,
  ``time_embedding.*`` -- loaded as it is.

The architecture (SD 1.x vs SDXL) is read from the file's cross-attention width,
so the right config is built without a download, and a small cache keeps a loaded
ControlNet alive between generations so switching checkpoints back and forth does
not reload it. Nothing here imports ComfyUI.
"""

import json
import os
import struct

import torch

# The diffusers ControlNet configs for the two classic architectures. They are
# the base UNet config plus ``conditioning_channels`` (the hint image's channel
# count), matching diffusers' own ``create_controlnet_diffusers_config_from_ldm``.
_SD15_CONFIG = {
    "in_channels": 4,
    "down_block_types": (
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D", "DownBlock2D",
    ),
    "block_out_channels": (320, 640, 1280, 1280),
    "layers_per_block": 2,
    "cross_attention_dim": 768,
    "attention_head_dim": 8,
    "conditioning_channels": 3,
    "use_linear_projection": False,
    "norm_num_groups": 32,
    "upcast_attention": False,
}

_SDXL_CONFIG = {
    "in_channels": 4,
    "down_block_types": (
        "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
    ),
    "block_out_channels": (320, 640, 1280),
    "layers_per_block": 2,
    "cross_attention_dim": 2048,
    "attention_head_dim": (5, 10, 20),
    "conditioning_channels": 3,
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 2816,
    "norm_num_groups": 32,
    "transformer_layers_per_block": (1, 2, 10),
    "upcast_attention": False,
}

_MAX_HEADER = 100_000_000

# (absolute path, arch) -> loaded ControlNetModel, so a checkpoint reload reuses
# the model it was used with.
_CACHE = {}


def _shapes(path):
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


def detect_arch(path):
    """``"sdxl"`` or ``"sd15"`` for a ControlNet file.

    The cross-attention input width decides it (768 for SD 1.x, 2048 for SDXL),
    which holds for both the LDM and the diffusers layouts. A file whose header
    cannot be read falls back to its filename.
    """
    for key, shape in _shapes(path).items():
        if key.endswith("attn2.to_k.weight") and len(shape) == 2:
            return "sdxl" if shape[1] == 2048 else "sd15"
    name = os.path.basename(path or "").lower()
    return "sdxl" if ("sdxl" in name or "_xl" in name or "-xl" in name) else "sd15"


def config_for(arch):
    """The diffusers ControlNet config dict for ``arch``."""
    return dict(_SDXL_CONFIG if arch == "sdxl" else _SD15_CONFIG)


def load(path, arch=None, device="cpu", dtype=torch.float32):
    """Load (and cache) the ControlNet at ``path``. Returns ``(model, arch)``.

    ``arch`` overrides detection when the caller already knows it (the classic
    checkpoint's family). Loading a diffusers-format file needs only the config
    built here, so no repository or network access is required.
    """
    arch = arch or detect_arch(path)
    key = (os.path.abspath(os.fspath(path)), arch)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached, arch

    from safetensors.torch import load_file
    from diffusers import ControlNetModel
    from diffusers.loaders.single_file_utils import convert_controlnet_checkpoint

    config = config_for(arch)
    state = load_file(path, device="cpu")
    # A ComfyUI wrapper sometimes prefixes every tensor; strip it so the
    # converter's own prefix checks see the layout they expect.
    if (
        "time_embed.0.weight" not in state
        and "time_embedding.linear_1.weight" not in state
    ):
        prefix = "model.diffusion_model."
        if any(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    converted = {
        name: tensor
        for name, tensor in convert_controlnet_checkpoint(state, config).items()
        if tensor is not None
    }
    model = ControlNetModel.from_config(config)
    model.load_state_dict(converted, strict=False)
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    _CACHE[key] = model
    return model, arch


def clear():
    """Drop every cached ControlNet (used when models are unloaded)."""
    _CACHE.clear()
