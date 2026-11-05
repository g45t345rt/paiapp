"""Vendored TAESD decoders for the live denoising preview.

A minimal, self-contained port of madebyollin's tiny distilled autoencoders
(the same architecture ComfyUI uses for its TAESD preview stream). Only the
*decoder* half is needed to turn a sampling-step latent into an image, so this
module omits the encoders.

Weights are fetched from Hugging Face on first use and cached by
``huggingface_hub``. There is no ``taesd`` pip dependency: any failure here --
the hub unreachable, weights missing, a shape mismatch -- returns ``None`` so
the caller falls back to the built-in :mod:`engine.preview` Latent2RGB decoder.
"""

import torch
import torch.nn as nn


def _conv(n_in, n_out, kernel=3, **kwargs):
    return nn.Conv2d(n_in, n_out, kernel, padding=kernel // 2, **kwargs)


class _Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class _Block(nn.Module):
    def __init__(self, n_in, n_out, use_gn=False):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in, n_out), nn.ReLU(),
            _conv(n_out, n_out), nn.ReLU(),
            _conv(n_out, n_out),
        )
        self.skip = (
            _conv(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.fuse = nn.ReLU()
        if use_gn:
            n_gn = n_in * 4
            self.pool = nn.Sequential(
                _conv(n_in, n_gn, 1, bias=False),
                nn.GroupNorm(4, n_gn),
                nn.ReLU(inplace=True),
                _conv(n_gn, n_in, 1, bias=False),
            )
        else:
            self.pool = None

    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


class _Decoder(nn.Sequential):
    def __init__(self, latent_channels=4, use_gn=False):
        super().__init__(
            _Clamp(), _conv(latent_channels, 64), nn.ReLU(),
            _Block(64, 64, use_gn), _Block(64, 64, use_gn), _Block(64, 64, use_gn),
            nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
            _Block(64, 64), _Block(64, 64), _Block(64, 64),
            nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
            _Block(64, 64), _Block(64, 64), _Block(64, 64),
            nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
            _Block(64, 64), _conv(64, 3),
        )


class _DecoderF16(nn.Sequential):
    def __init__(self, latent_channels=64, image_channels=4):
        super().__init__(
            _Clamp(), _conv(latent_channels, 256), nn.ReLU(),
            _Block(256, 256), _Block(256, 256), _Block(256, 256),
            nn.Upsample(scale_factor=2), _conv(256, 128, bias=False),
            _Block(128, 128), _Block(128, 128), _Block(128, 128),
            nn.Upsample(scale_factor=2), _conv(128, 64, bias=False),
            _Block(64, 64), _Block(64, 64), _Block(64, 64),
            nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
            _Block(64, 64), _conv(64, image_channels * 4), nn.PixelShuffle(2),
        )


class _DecoderFlux2(_Decoder):
    def __init__(self, latent_channels=128, use_gn=True):
        super().__init__(latent_channels=32, use_gn=True)

    def forward(self, x):
        b, _c, h, w = x.shape
        x = (
            x.reshape(b, 32, 2, 2, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(b, 32, h * 2, w * 2)
        )
        return super().forward(x)


# decoder name -> (Hugging Face repo, file, latent channels)
#
# Only the two checkpoints published in the native (non-diffusers) decoder
# layout are listed here. The flow families (Flux/Qwen/Wan) have no directly
# loadable TAESD decoder, so their preview falls back to Latent2RGB.
_SOURCES = {
    "taesd": ("madebyollin/taesd", "taesd_decoder.safetensors", 4),
    "taesdxl": ("madebyollin/taesdxl", "taesdxl_decoder.safetensors", 4),
}

_CACHE = {}


def _build(channels):
    if channels == 128:
        return _DecoderFlux2(channels)
    if channels == 64:
        return _DecoderF16(channels)
    return _Decoder(channels)


def load(name):
    """Return a cached decoder module for ``name``, or ``None`` when unavailable."""
    if name in _CACHE:
        return _CACHE[name]
    decoder = None
    spec = _SOURCES.get(name)
    if spec is not None:
        repo, filename, channels = spec
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            path = hf_hub_download(repo, filename)
            model = _build(channels)
            model.load_state_dict(load_file(path))
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            decoder = model
        except Exception:
            decoder = None
    _CACHE[name] = decoder
    return decoder
