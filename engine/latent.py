"""Wan-21 latent format (used by the Qwen-Image VAE / Anima DiT)."""

import torch

_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


class Wan21:
    latent_channels = 16
    latent_dimensions = 3
    scale_factor = 1.0

    def __init__(self):
        self.latents_mean = torch.tensor(_MEAN).view(1, self.latent_channels, 1, 1, 1)
        self.latents_std = torch.tensor(_STD).view(1, self.latent_channels, 1, 1, 1)

    def process_in(self, latent):
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - mean) * self.scale_factor / std

    def process_out(self, latent):
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return latent * std / self.scale_factor + mean


class QwenImage21Latent:
    """64-channel Qwen-Image 2.1 latent (per-channel mean/std normalization)."""

    latent_channels = 64
    latent_dimensions = 2
    scale_factor = 1.0

    _MEAN = [
        0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
        1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
        2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
        -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
        -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
        -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
        -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
        0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
    ]
    _STD = [
        3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
        3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
        3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
        3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
        3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
        3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
        3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
        3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
    ]

    def __init__(self):
        self.latents_mean = torch.tensor(self._MEAN).view(1, self.latent_channels, 1, 1)
        self.latents_std = torch.tensor(self._STD).view(1, self.latent_channels, 1, 1)

    def process_in(self, latent):
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - mean) / std

    def process_out(self, latent):
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return latent * std + mean


class FluxLatent:
    """16-channel Flux latent (shift 0.1159, scale 0.3611)."""

    latent_channels = 16
    latent_dimensions = 2

    def __init__(self, scale_factor=0.3611, shift_factor=0.1159):
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class MossImageLatent:
    """128-channel MOSS Image latent.

    The per-channel normalization and the 2x2 patch packing live in the FLUX.2
    VAE wrapper (:func:`engine.moss_vae.pack_latents` / ``unpack_latents``), so
    the model-space latent the sampler works in is already normalized and these
    hooks are the identity (matching ComfyUI's ``latent_formats.Flux2``).
    """

    latent_channels = 128
    latent_dimensions = 2
    scale_factor = 1.0

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


class ClassicLatent:
    """Standard 4-channel SD 1.x / SDXL latent (no mean/std normalization)."""

    latent_channels = 4
    latent_dimensions = 2

    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent

