"""Krea 2 AnyPaint: arbitrary-mask inpainting, outpainting and image editing.

Port of the community functional adapter ``yijunwang2/krea2-anypaint`` on top of
the in-repo Krea 2 sampler. The adapter is a LoRA for Krea 2 Turbo that turns the
model into an edit model: a source image is placed on the output canvas, a binary
mask marks the pixels to (re)generate, and the model produces the complete image
with everything outside the mask (and outside the placed source) preserved.

The runtime contract, matching the upstream ``anypaint.py`` helper:

* one RGB source image and one binary mask (white = generated, black = kept);
* an output canvas whose dimensions are multiples of 16;
* an optional source bounding box in output pixels (outpainting places a smaller
  source on a larger canvas; the whole default is a same-canvas inpaint).

:func:`prepare_anypaint` builds the inputs the denoiser needs (a low-res
``condition`` reference, the full-canvas ``known_image`` and its ``keep_mask``).
The remaining helpers convert images to the packed latent grid the DiT reads so
the pipeline can build the reference tokens and the per-step known-latent
constraint.

Everything here is pure PIL / numpy / torch plus SciPy's separable grey dilation
(the upstream helper uses OpenCV only for the mask dilation), so it needs no
extra dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

try:
    from scipy.ndimage import grey_dilation as _grey_dilation
except Exception:  # pragma: no cover - SciPy is a core dependency; keep a fallback
    _grey_dilation = None

# The reference/condition image is encoded at this maximum edge (the card's fixed
# setting), snapped to a multiple of 16.
REFERENCE_MAX_EDGE = 384
# The boundary band left free for the model to redraw and blend, in pixels.
SEAM_PX = 32
# Krea 2's latent patch size.
PATCH = 2
# The Qwen-image VAE downsample factor (pixels per latent cell).
VAE_SCALE = 8


@dataclass(frozen=True)
class AnyPaintInput:
    """The images one AnyPaint request needs, ready for the denoiser."""

    condition: Image.Image
    known_image: Image.Image
    keep_mask: Image.Image
    generated_mask: Image.Image
    canvas_size: tuple
    source_bbox: tuple

    @property
    def reference_placement(self):
        """The reference's rotary placement: the full destination canvas."""
        return {"bbox_normalized": [0.0, 0.0, 1.0, 1.0]}


def _validate_canvas(canvas_size):
    width, height = canvas_size
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("Canvas dimensions must be positive multiples of 16")


def _resize_max_edge(image, max_edge):
    scale = min(1.0, max_edge / max(image.size))
    size = (
        max(16, int(round(image.width * scale)) // 16 * 16),
        max(16, int(round(image.height * scale)) // 16 * 16),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _median_color(values):
    if values.size == 0:
        return np.array([127, 127, 127], dtype=np.uint8)
    return np.median(values.reshape(-1, 3), axis=0).round().astype(np.uint8)


def _edge_aware_keep_mask(generated_mask, seam_px):
    """The keep mask: everything the model is *not* allowed to redraw.

    The generated region is dilated by ``seam_px`` so the model gets a boundary
    band to redraw and blend, then inverted (white = preserved).
    """
    generated = np.where(np.asarray(generated_mask.convert("L")) > 0, 255, 0).astype(
        np.uint8
    )
    if seam_px > 0:
        size = seam_px * 2 + 1
        if _grey_dilation is not None:
            generated = (
                _grey_dilation(generated, size=size).astype(np.uint8)
            )
        else:  # pragma: no cover - fallback when SciPy is unavailable
            generated = np.asarray(
                Image.fromarray(generated, mode="L").filter(
                    ImageFilter.MaxFilter(size)
                )
            )
    return ImageOps.invert(Image.fromarray(generated, mode="L"))


def prepare_anypaint(
    source,
    generated_mask,
    canvas_size,
    source_bbox=None,
    *,
    reference_max_edge=REFERENCE_MAX_EDGE,
    seam_px=SEAM_PX,
):
    """Prepare one arbitrary-mask inpainting/outpainting request.

    White mask pixels are generated. Black mask pixels are preserved. Pixels
    outside ``source_bbox`` are always generated, which makes the same helper
    work for both inpainting and outpainting.
    """
    _validate_canvas(canvas_size)
    width, height = canvas_size
    source = source.convert("RGB")
    if source_bbox is None:
        source_bbox = (0, 0, width, height)
    x0, y0, x1, y1 = source_bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Source bbox is outside the canvas: {source_bbox}")

    box_width, box_height = x1 - x0, y1 - y0
    source_ratio = source.width / source.height
    box_ratio = box_width / box_height
    tolerance = max(0.025, 2.0 / min(box_width, box_height))
    if abs(box_ratio / source_ratio - 1.0) > tolerance:
        raise ValueError("Source bbox must preserve the source image aspect ratio")

    placed = source.resize((box_width, box_height), Image.Resampling.LANCZOS)
    placed_values = np.asarray(placed, dtype=np.uint8)
    fill = _median_color(placed_values)
    known_values = np.empty((height, width, 3), dtype=np.uint8)
    known_values[:] = fill
    known_values[y0:y1, x0:x1] = placed_values
    known_image = Image.fromarray(known_values, mode="RGB")

    mask = generated_mask.convert("L")
    if mask.size == source.size:
        mask = mask.resize((box_width, box_height), Image.Resampling.NEAREST)
        canvas_mask = Image.new("L", canvas_size, 255)
        canvas_mask.paste(mask, (x0, y0))
        mask = canvas_mask
    elif mask.size != canvas_size:
        raise ValueError("Mask must match either the source image or the output canvas")
    mask = mask.point(lambda value: 255 if value > 127 else 0)

    outside = Image.new("L", canvas_size, 255)
    outside.paste(0, source_bbox)
    generated = Image.fromarray(
        np.maximum(np.asarray(mask, dtype=np.uint8), np.asarray(outside, dtype=np.uint8)),
        mode="L",
    )
    generated_values = np.asarray(generated, dtype=np.uint8) > 0
    if not generated_values.any():
        raise ValueError("The generated mask has no white pixels")

    condition_values = known_values.copy()
    known_pixels = condition_values[~generated_values]
    condition_values[generated_values] = _median_color(known_pixels)
    condition = _resize_max_edge(
        Image.fromarray(condition_values, mode="RGB"), reference_max_edge
    )

    return AnyPaintInput(
        condition=condition,
        known_image=known_image,
        keep_mask=_edge_aware_keep_mask(generated, seam_px),
        generated_mask=generated,
        canvas_size=(width, height),
        source_bbox=source_bbox,
    )


# ---------------------------------------------------------------------------
# Latent-grid helpers (the pipeline couples these to the VAE).
# ---------------------------------------------------------------------------

def pack_latents(latent, patch=PATCH):
    """``(B, C, 1, H, W)`` (or ``(B, C, H, W)``) -> ``(B, H/p * W/p, C*p*p)`` tokens."""
    if latent.ndim == 5:
        latent = latent[:, :, 0]
    b, c, h, w = latent.shape
    tokens = latent.reshape(b, c, h // patch, patch, w // patch, patch)
    tokens = tokens.permute(0, 2, 4, 1, 3, 5)
    return tokens.reshape(b, (h // patch) * (w // patch), c * patch * patch)


def reference_positions(ref_height, ref_width, target_grid, bbox=(0.0, 0.0, 1.0, 1.0),
                        frame=1, patch=PATCH, device=None):
    """The ``(seq, 3)`` rotary coordinates of one clean reference image.

    ``ref_height``/``ref_width`` are the reference's *latent* dimensions (its
    token grid is ``ref_height // patch`` by ``ref_width // patch``). The reference
    sits on frame axis ``frame`` and its ``(y, x)`` coordinates are spread across
    ``target_grid`` (``(height, width)`` in tokens) through the normalized
    ``bbox`` -- the "destination-relative" placement the adapter was trained with.
    ``bbox`` defaults to the full canvas.
    """
    ref_h, ref_w = ref_height // patch, ref_width // patch
    target_h, target_w = target_grid
    x0, y0, x1, y1 = bbox
    ids = torch.zeros(ref_h, ref_w, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ys = y0 * target_h + (torch.arange(ref_h, device=device, dtype=torch.float32) + 0.5) * (
        (y1 - y0) * target_h / ref_h
    ) - 0.5
    xs = x0 * target_w + (torch.arange(ref_w, device=device, dtype=torch.float32) + 0.5) * (
        (x1 - x0) * target_w / ref_w
    ) - 0.5
    ids[..., 1] = ys[:, None]
    ids[..., 2] = xs[None, :]
    return ids.reshape(-1, 3)


def known_token_mask(keep_mask, latent_height, latent_width, patch=PATCH):
    """A ``(1, tokens, 1)`` bool mask: a token is kept only if all its pixels are.

    ``keep_mask`` is the full-canvas keep image (white = preserved); it is
    downsampled to the latent grid and reduced over each ``patch x patch`` cell.
    """
    small = keep_mask.convert("L").resize(
        (latent_width, latent_height), Image.Resampling.NEAREST
    )
    arr = np.asarray(small) > 127
    tokens = torch.from_numpy(arr).reshape(
        latent_height // patch, patch, latent_width // patch, patch
    )
    return tokens.amin(dim=(1, 3)).reshape(1, -1, 1)


def expand_token_mask(token_mask, latent_height, latent_width, patch=PATCH):
    """``(1, tokens, 1)`` token mask -> ``(1, 1, 1, H, W)`` spatial keep for latents."""
    ht, wt = latent_height // patch, latent_width // patch
    grid = token_mask.reshape(1, 1, ht, wt)
    return grid.repeat_interleave(patch, dim=2).repeat_interleave(patch, dim=3)


def restore_known(latents, known_latents, keep, initial_noise, sigma_next):
    """Pin the known region back to its clean latent at the next noise level.

    Mirrors the adapter's per-step constraint: the preserved tokens are replaced
    by the known latent noised to ``sigma_next`` with the same initial noise, so
    the unmasked content stays stable through the whole trajectory.
    """
    sigma = torch.as_tensor(sigma_next, device=latents.device, dtype=latents.dtype)
    noised = known_latents + sigma * (initial_noise - known_latents)
    return torch.where(keep, noised, latents)
