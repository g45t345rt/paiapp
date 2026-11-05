"""Flux 16-channel VAE (``AutoencoderKL``) from a ComfyUI safetensors file.

The 16-latent-channel ``AutoencoderKL`` topology; the latent shift/scale
(0.1159 / 0.3611) lives in :class:`engine.latent.FluxLatent`.
The weights are in LDM/ComfyUI layout, converted with diffusers' own helper
(imported through :mod:`engine.checkpoint`, which stubs a broken ``xformers``).
"""

import torch

_FLUX_VAE = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "latent_channels": 16,
    "norm_num_groups": 32,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
}


def load_vae(path, device="cpu", dtype=torch.float32):
    """Return ``(vae, missing, unexpected)`` for a Flux VAE safetensors file."""
    from safetensors.torch import load_file

    from .checkpoint import AutoencoderKL, convert_ldm_vae_checkpoint

    config = dict(_FLUX_VAE)
    vae = AutoencoderKL.from_config(config)
    state = load_file(path, device="cpu")
    converted = convert_ldm_vae_checkpoint(state, config)
    info = vae.load_state_dict(converted, strict=False)
    vae = vae.to(device=device, dtype=dtype).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, list(info.missing_keys), list(info.unexpected_keys)
