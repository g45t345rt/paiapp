"""FLUX.2 VAE (``AutoencoderKLFlux2``) for MOSS Image.

MOSS Image's VAE is the FLUX.2 VAE: a 32-channel ``AutoencoderKL`` at 1/8
resolution whose latent is patch-packed 2x2 and normalised per channel by a
``BatchNorm2d`` (``vae.bn``) to become the DiT's 128-channel, 1/16-resolution
latent. The reference Diffusers pipeline does that packing by hand around
``vae.encode``/``vae.decode`` (see ``pipeline_moss_image``); :func:`pack_latents`
and :func:`unpack_latents` mirror it here.

``AutoencoderKLFlux2`` is only in ``diffusers`` >= 0.38; the loader raises a
clear error on older versions.
"""

import torch

# The released FLUX.2 VAE config, used when a single ``.safetensors`` file is
# loaded instead of a Diffusers directory.
_FLUX2_VAE = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (128, 256, 512, 512),
    "decoder_block_out_channels": (96, 192, 384, 384),
    "layers_per_block": 2,
    "latent_channels": 32,
    "norm_num_groups": 32,
    "sample_size": 1024,
    "patch_size": (2, 2),
    "mid_block_add_attention": True,
    "use_quant_conv": True,
    "use_post_quant_conv": True,
    "batch_norm_eps": 1e-4,
    "batch_norm_momentum": 0.1,
}


def _vae_class():
    import diffusers

    cls = getattr(diffusers, "AutoencoderKLFlux2", None)
    if cls is None:
        raise RuntimeError(
            "MOSS Image needs diffusers>=0.38 for the FLUX.2 VAE "
            "(no AutoencoderKLFlux2 found)."
        )
    return cls


def load_vae(path, device="cpu", dtype=torch.float32):
    """Return ``(vae, missing, unexpected)`` for a FLUX.2 VAE.

    ``path`` may be a Diffusers directory (loaded with ``from_pretrained``) or a
    single ``.safetensors`` file in Diffusers or LDM/ComfyUI layout. For an LDM
    file the channel layout is read from the tensors themselves, so both the full
    FLUX.2 VAE and MOSS Image's smaller-decoder VAE load.
    """
    import os

    cls = _vae_class()
    if os.path.isdir(path):
        vae = cls.from_pretrained(path, torch_dtype=dtype)
        info = None
    else:
        from safetensors.torch import load_file

        state = load_file(path, device="cpu")
        if any(key.startswith("encoder.down.") for key in state):
            config = _vae_config_from_state(state)
            state = _convert_ldm_vae(state, config)
        else:
            config = dict(_FLUX2_VAE)
        vae = cls.from_config(config)
        info = vae.load_state_dict(state, strict=False)
    vae = vae.to(device=device, dtype=dtype).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    missing = list(info.missing_keys) if info is not None else []
    unexpected = list(info.unexpected_keys) if info is not None else []
    return vae, missing, unexpected


def _vae_config_from_state(state):
    """A FLUX.2 VAE config whose channel layout matches an LDM checkpoint.

    The full FLUX.2 VAE and MOSS Image's VAE differ only in the decoder widths,
    so they are read back from ``encoder.down.*``/``decoder.up.*`` instead of
    being hard-coded.
    """
    config = dict(_FLUX2_VAE)
    config["decoder_block_out_channels"] = tuple(config["block_out_channels"])
    for key, target in (("encoder.down", "block_out_channels"), ("decoder.up", "decoder_block_out_channels")):
        levels = sorted({
            int(name.split(".")[2]) for name in state
            if name.startswith(key + ".") and name.endswith("block.0.conv1.weight")
        })
        if levels:
            config[target] = tuple(int(state[f"{key}.{i}.block.0.conv1.weight"].shape[0]) for i in levels)
    config["layers_per_block"] = 1 + max(
        int(name.split(".")[4]) for name in state
        if name.startswith("encoder.down.0.block.") and name.endswith("conv1.weight")
    )
    conv_out = state.get("encoder.conv_out.weight")
    if conv_out is not None:
        config["latent_channels"] = int(conv_out.shape[0]) // 2
    return config


def _convert_ldm_vae(state, config):
    """LDM/ComfyUI FLUX.2 VAE keys to Diffusers, including ``bn`` and the quant convs."""
    from .checkpoint import convert_ldm_vae_checkpoint

    converted = convert_ldm_vae_checkpoint(state, config)
    for key, value in state.items():
        if key.startswith("bn."):
            converted[key] = value
        elif key.startswith("encoder.quant_conv."):
            converted["quant_conv." + key.rsplit(".", 1)[1]] = value
        elif key.startswith("decoder.post_quant_conv."):
            converted["post_quant_conv." + key.rsplit(".", 1)[1]] = value
    return converted


def _bn_stats(vae):
    """``(mean, std)`` of the VAE's latent BatchNorm as ``[1, C, 1, 1]`` views."""
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    mean = bn.running_mean.view(1, -1, 1, 1)
    std = (bn.running_var + eps).sqrt().view(1, -1, 1, 1)
    return mean, std


def pack_latents(vae, latent):
    """32-channel VAE latent -> 128-channel DiT latent (2x2 patch + BN norm).

    Inverse of :func:`unpack_latents`: patch-pack the spatial 2x2 neighbourhoods
    into channels, then apply the VAE's latent BatchNorm (what
    ``pipeline_moss_image`` does before ``vae.decode``).
    """
    mean, std = _bn_stats(vae)
    b, c, h, w = latent.shape
    packed = latent.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    return (packed - mean.to(packed)) / std.to(packed)


def unpack_latents(vae, latent):
    """128-channel DiT latent -> 32-channel VAE latent (BN denorm + 2x2 unpack)."""
    mean, std = _bn_stats(vae)
    z = latent * std.to(latent) + mean.to(latent)
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
