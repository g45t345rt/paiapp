"""Live denoising-step previews.

While the sampler denoises, every step's ``x0`` prediction is a low-resolution
latent that already resembles the final image. This module turns that latent
into a small RGB PIL image so the UI can show the image forming, the way
ComfyUI's ``--preview-method`` and A1111 do.

Two decoders:

* ``latent2rgb`` -- a fixed per-family linear map from latent channels to RGB
  (the exact coefficient tables ComfyUI ships in its ``latent_formats``). No
  VAE, essentially free, rough colors. Always available.
* ``taesd`` -- a tiny distilled autoencoder (the ``taesd`` package). Much
  cheaper than the real VAE and near-photographic, but only exists for some
  families and needs its (downloaded) weights; every other case falls back to
  ``latent2rgb``.

Families whose VAE is evicted during low-VRAM / disk streaming still preview,
because ``latent2rgb`` never touches the VAE.
"""

import torch
from PIL import Image

# -- latent2rgb coefficient tables (ported from ComfyUI's latent_formats) ------
# Each entry is ``(factors, bias)`` where ``factors`` is ``[channels, 3]``.

_SD15 = (
    [
        [0.3512, 0.2297, 0.3227],
        [0.3250, 0.4974, 0.2350],
        [-0.2829, 0.1762, 0.2721],
        [-0.2120, -0.2616, -0.7177],
    ],
    None,
)
_SDXL = (
    [
        [0.3651, 0.4232, 0.4341],
        [-0.2533, -0.0042, 0.1068],
        [0.1076, 0.1111, -0.0362],
        [-0.3165, -0.2492, -0.2188],
    ],
    [0.1084, -0.0175, -0.0011],
)
_FLUX = (
    [
        [-0.0346, 0.0244, 0.0681],
        [0.0034, 0.0210, 0.0687],
        [0.0275, -0.0668, -0.0433],
        [-0.0174, 0.0160, 0.0617],
        [0.0859, 0.0721, 0.0329],
        [0.0004, 0.0383, 0.0115],
        [0.0405, 0.0861, 0.0915],
        [-0.0236, -0.0185, -0.0259],
        [-0.0245, 0.0250, 0.1180],
        [0.1008, 0.0755, -0.0421],
        [-0.0515, 0.0201, 0.0011],
        [0.0428, -0.0012, -0.0036],
        [0.0817, 0.0765, 0.0749],
        [-0.1264, -0.0522, -0.1103],
        [-0.0280, -0.0881, -0.0499],
        [-0.1262, -0.0982, -0.0778],
    ],
    [-0.0329, -0.0718, -0.0851],
)
_WAN21 = (
    [
        [-0.1299, -0.1692, 0.2932],
        [0.0671, 0.0406, 0.0442],
        [0.3568, 0.2548, 0.1747],
        [0.0372, 0.2344, 0.1420],
        [0.0313, 0.0189, -0.0328],
        [0.0296, -0.0956, -0.0665],
        [-0.3477, -0.4059, -0.2925],
        [0.0166, 0.1902, 0.1975],
        [-0.0412, 0.0267, -0.1364],
        [-0.1293, 0.0740, 0.1636],
        [0.0680, 0.3019, 0.1128],
        [0.0032, 0.0581, 0.0639],
        [-0.1251, 0.0927, 0.1699],
        [0.0060, -0.0633, 0.0005],
        [0.3477, 0.2275, 0.2950],
        [0.1984, 0.0913, 0.1861],
    ],
    [-0.1835, -0.0868, -0.3360],
)
_QWEN21 = (
    [
        [-0.0158, -0.0115, -0.0174], [0.0030, 0.0120, 0.0027],
        [0.0637, 0.0470, -0.0127], [0.0360, 0.0661, -0.0030],
        [0.0159, 0.0181, 0.0082], [0.0132, 0.0326, 0.0169],
        [0.0191, 0.0261, 0.0136], [-0.0146, -0.0276, -0.0361],
        [0.0187, -0.0024, -0.0072], [-0.1059, -0.0090, 0.0350],
        [-0.0195, -0.0226, -0.0138], [-0.0295, 0.0024, -0.0215],
        [0.0191, -0.0393, -0.0001], [-0.0144, -0.0166, -0.0272],
        [0.0389, 0.0430, 0.0445], [-0.0153, -0.0336, 0.0031],
        [0.0339, 0.0122, 0.0220], [-0.0136, -0.0078, -0.0120],
        [-0.0340, -0.0282, -0.0245], [-0.0133, -0.0176, -0.0133],
        [0.0109, -0.0087, 0.0096], [-0.0010, 0.0044, 0.0016],
        [0.0301, 0.0053, 0.0361], [-0.0281, -0.0205, -0.0032],
        [-0.0725, 0.0002, 0.0160], [-0.0036, 0.0158, 0.0807],
        [0.0087, 0.0040, -0.0053], [-0.0260, 0.0183, -0.0077],
        [-0.0039, -0.0035, -0.0107], [-0.0026, 0.0172, 0.0237],
        [0.0088, 0.0078, 0.0078], [-0.0087, -0.0310, -0.0122],
        [-0.0027, 0.0018, 0.0094], [-0.0064, 0.0292, -0.0256],
        [0.0594, 0.1049, 0.1180], [0.0103, -0.0103, -0.0026],
        [-0.0091, 0.0025, -0.0015], [0.0178, 0.0243, 0.0292],
        [-0.0063, -0.0012, 0.0202], [0.0452, 0.0246, 0.0143],
        [0.0149, 0.0270, 0.0052], [0.1484, 0.0801, 0.0804],
        [-0.0120, 0.0040, 0.0010], [0.0181, 0.0051, -0.0021],
        [0.0132, 0.0050, 0.0019], [0.0291, 0.0020, 0.0092],
        [0.0066, -0.0410, -0.1314], [-0.1153, -0.0629, -0.0802],
        [0.0258, 0.0378, 0.0298], [0.0375, 0.1139, 0.0468],
        [-0.0142, -0.0126, -0.0276], [0.0339, 0.0153, 0.0138],
        [0.0346, 0.0211, 0.0267], [0.0369, -0.0431, -0.0993],
        [-0.0052, -0.0092, 0.0056], [-0.0279, 0.0410, -0.0357],
        [0.0036, 0.0017, -0.0083], [-0.0441, -0.0367, -0.0454],
        [-0.0001, -0.0092, -0.0001], [-0.0222, -0.0183, -0.0051],
        [0.0039, 0.0053, -0.0184], [-0.0094, -0.0075, -0.0143],
        [-0.0066, -0.0088, -0.0063], [0.0220, 0.0074, 0.0100],
    ],
    [-0.1228, -0.1869, -0.3083],
)
_H3 = (
    [
        [-0.018555, 0.024344, -0.017536],
        [0.150164, 0.137244, 0.129221],
        [0.027367, -0.050369, -0.208606],
        [-0.000793, -0.164622, -0.323161],
        [-0.048556, 0.013970, -0.074286],
        [0.011740, 0.014172, -0.006906],
        [0.061517, 0.061212, 0.110025],
        [0.035321, 0.086879, 0.110059],
        [-0.017426, 0.002997, 0.035356],
        [0.531539, 0.548819, 0.624404],
        [-0.024968, -0.040234, -0.034302],
        [-0.032549, -0.029096, -0.017221],
        [0.022609, 0.020286, 0.050661],
        [-0.084001, -0.038131, -0.020805],
        [-0.018830, 0.010412, 0.061120],
        [0.020777, 0.011196, -0.030994],
        [-0.008390, -0.012201, -0.025687],
        [-0.013281, -0.002924, 0.006331],
        [0.000260, 0.001833, -0.011038],
        [0.105471, 0.100482, 0.132106],
        [0.016529, 0.015213, 0.009999],
        [-0.014015, -0.017438, -0.019134],
        [-0.033787, -0.009984, -0.019725],
        [0.004224, 0.017284, 0.027196],
    ],
    [0.057426, -0.022078, -0.071449],
)
_FLUX2 = (
    [
        [0.0058, 0.0113, 0.0073], [0.0495, 0.0443, 0.0836],
        [-0.0099, 0.0096, 0.0644], [0.2144, 0.3009, 0.3652],
        [0.0166, -0.0039, -0.0054], [0.0157, 0.0103, -0.0160],
        [-0.0398, 0.0902, -0.0235], [-0.0052, 0.0095, 0.0109],
        [-0.3527, -0.2712, -0.1666], [-0.0301, -0.0356, -0.0180],
        [-0.0107, 0.0078, 0.0013], [0.0746, 0.0090, -0.0941],
        [0.0156, 0.0169, 0.0070], [-0.0034, -0.0040, -0.0114],
        [0.0032, 0.0181, 0.0080], [-0.0939, -0.0008, 0.0186],
        [0.0018, 0.0043, 0.0104], [0.0284, 0.0056, -0.0127],
        [-0.0024, -0.0022, -0.0030], [0.1207, -0.0026, 0.0065],
        [0.0128, 0.0101, 0.0142], [0.0137, -0.0072, -0.0007],
        [0.0095, 0.0092, -0.0059], [0.0000, -0.0077, -0.0049],
        [-0.0465, -0.0204, -0.0312], [0.0095, 0.0012, -0.0066],
        [0.0290, -0.0034, 0.0025], [0.0220, 0.0169, -0.0048],
        [-0.0332, -0.0457, -0.0468], [-0.0085, 0.0389, 0.0609],
        [-0.0076, 0.0003, -0.0043], [-0.0111, -0.0460, -0.0614],
    ],
    [-0.0329, -0.0718, -0.0851],
)


def _flux2_reshape(t):
    return t.reshape(
        t.shape[0], 32, 2, 2, t.shape[-2], t.shape[-1]
    ).permute(0, 1, 4, 2, 5, 3).reshape(
        t.shape[0], 32, t.shape[-2] * 2, t.shape[-1] * 2
    )


def _select(latent_format, key=None):
    """Pick ``(factors, bias, reshape, taesd)`` for a latent format or family.

    Selection is by the latent format's class name (the engine's own
    ``engine.latent`` classes) with a channel-count fallback, so a new family
    that reuses an existing format still previews. ``key`` forces a family's
    table when there is no latent format object (the H3 manual sampler).
    """
    name = type(latent_format).__name__ if latent_format is not None else None
    if key == "h3" or name == "MiniMaxH3":
        return _H3[0], _H3[1], None, None
    if name == "ClassicLatent":
        scale = float(getattr(latent_format, "scale_factor", 0.18215) or 0.18215)
        table = _SDXL if abs(scale - 0.13025) < abs(scale - 0.18215) else _SD15
        taesd = "taesdxl" if table is _SDXL else "taesd"
        return table[0], table[1], None, taesd
    if name == "Wan21":
        return _WAN21[0], _WAN21[1], None, None
    if name == "QwenImage21Latent":
        return _QWEN21[0], _QWEN21[1], None, None
    if name == "MossImageLatent":
        return _FLUX2[0], _FLUX2[1], _flux2_reshape, None
    if name == "FluxLatent":
        return _FLUX[0], _FLUX[1], None, None
    channels = int(getattr(latent_format, "latent_channels", 0) or 0)
    if channels == 128:
        return _FLUX2[0], _FLUX2[1], _flux2_reshape, None
    if channels == 64:
        return _QWEN21[0], _QWEN21[1], None, None
    if channels == 16:
        return _WAN21[0], _WAN21[1], None, None
    if channels == 4:
        return _SD15[0], _SD15[1], None, None
    return None


def latent_to_rgb(latent, factors, bias=None):
    """Linear latent->RGB then ``0.5 + 0.5 * x``, matching ComfyUI.

    ``latent`` is ``[B, C, H, W]`` (or ``[B, C, 1, H, W]``, whose single frame is
    used) and the result is an ``[B, H, W, 3]`` float image in ``[0, 1]``.
    """
    if latent.ndim == 5:
        latent = latent[:, :, 0]
    chw = latent.permute(0, 2, 3, 1)
    weight = torch.tensor(factors, device=chw.device, dtype=chw.dtype)
    rgb = chw @ weight
    if bias is not None:
        rgb = rgb + torch.tensor(bias, device=chw.device, dtype=chw.dtype)
    return torch.clamp(rgb * 0.5 + 0.5, 0.0, 1.0)


def load_taesd(name):
    """Load (and cache) a taesd decoder, or ``None`` when unavailable.

    Delegates to :mod:`engine.taesd`, which vendors the tiny decoder and fetches
    its weights from Hugging Face. Any failure (unknown name, hub unreachable,
    weights missing) returns ``None`` so the caller falls back to
    ``latent2rgb``.
    """
    if not name:
        return None
    from . import taesd

    return taesd.load(name)


class LivePreview:
    """Decode a sampling-step ``x0`` latent into a small RGB PIL image."""

    def __init__(self, latent_format=None, key=None, mode="latent2rgb",
                 max_size=768):
        self.latent_format = latent_format
        self.key = key
        self.mode = mode
        self.max_size = max_size
        self._taesd_name = None
        self._taesd = None
        if mode == "taesd":
            table = _select(latent_format, key)
            self._taesd_name = table[3] if table is not None else None
            self._taesd = load_taesd(self._taesd_name)

    @property
    def available(self):
        if self.mode == "latent2rgb":
            return _select(self.latent_format, self.key) is not None
        if self.mode == "taesd":
            return self._taesd is not None
        return False

    def decode(self, latent, out_size=None):
        """Return a PIL RGB preview of ``latent``, or ``None`` on failure.

        ``out_size`` is the target ``(width, height)`` in pixels; the low-res
        decode is resized up to it so the preview lines up with the frame.
        """
        if latent is None:
            return None
        try:
            with torch.inference_mode():
                rgb = self._decode_tensor(latent)
        except Exception:
            return None
        if rgb is None:
            return None
        image = Image.fromarray(
            (rgb[0].cpu().numpy() * 255.0).round().astype("uint8"), "RGB"
        )
        if out_size is not None and image.size != tuple(out_size):
            image = image.resize(tuple(out_size), Image.BICUBIC)
        elif self.max_size and max(image.size) > self.max_size:
            scale = self.max_size / max(image.size)
            image = image.resize(
                (max(1, round(image.width * scale)),
                 max(1, round(image.height * scale))),
                Image.BICUBIC,
            )
        return image

    def _decode_tensor(self, latent):
        if self.mode == "taesd" and self._taesd is not None:
            x = latent
            if x.ndim == 5:
                x = x[:, :, 0]
            x = x.float()
            param = next(self._taesd.parameters())
            if param.device != x.device:
                self._taesd.to(x.device)
            # The decoder emits an image already in [0, 1] (ComfyUI clamps its
            # preview the same way).
            return self._taesd(x).clamp(0.0, 1.0).permute(0, 2, 3, 1)
        table = _select(self.latent_format, self.key)
        if table is None:
            return None
        factors, bias, reshape, _taesd = table
        x = latent.float()
        if self.latent_format is not None:
            x = self.latent_format.process_out(x)
        if x.ndim == 5:
            x = x[:, :, 0]
        if reshape is not None:
            x = reshape(x)
        return latent_to_rgb(x, factors, bias)


def build(latent_format=None, key=None, mode="off", max_size=768):
    """A :class:`LivePreview` for ``mode``, or ``None`` when preview is off.

    ``mode`` is ``off``, ``latent2rgb`` or ``taesd``; an unsupported mode (or a
    ``taesd`` request with no decoder) falls back to ``latent2rgb``, and a
    latent format with no table at all yields ``None``.
    """
    if not mode or mode == "off":
        return None
    previewer = LivePreview(latent_format, key=key, mode=mode, max_size=max_size)
    if previewer.available:
        return previewer
    if mode != "latent2rgb":
        fallback = LivePreview(latent_format, key=key, mode="latent2rgb",
                               max_size=max_size)
        if fallback.available:
            return fallback
    return None
