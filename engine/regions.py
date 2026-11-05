"""Per-region prompt conditioning (regional prompting).

A run may split its prompt by image region: each region carries its own prompt
and a spatial mask, and cross-attention is masked so an image token only attends
to its region's prompt block (plus a shared base block). This is the
"attention-couple" / regional-prompter approach used by ComfyUI's ComfyCouple and
A1111's Regional Prompter.

Each region also carries a ``strength``: the mask is an *additive* attention
bias rather than a boolean allow-mask, so a region's columns can be scaled
relative to the base block (see :func:`region_biases`). A strength of 1.0 is a
no-op, so an unweighted run behaves exactly as the older boolean mask.

The mask building here is family-agnostic; each family wires the mask into its
own attention (the Anima DiT takes a cross-attention mask directly, and the SD
UNet gets a custom attention processor).
"""

import math

import numpy as np
import torch
from PIL import Image

# Downsampling factors a UNet cross-attention layer may sit at, relative to the
# latent grid (each down/up level halves the resolution).
_SCALES = (1, 2, 4, 8, 16, 32)


def _grid(mask_image, width, height):
    """``mask_image`` resampled to a boolean ``(height, width)`` grid."""
    small = mask_image.convert("L").resize((width, height), Image.BILINEAR)
    return np.asarray(small).astype("float32") > 127.0


def token_regions(region_entries, grid_w, grid_h):
    """Per-region boolean masks at a token grid, as an array ``(tokens, regions)``.

    Rows are image tokens in row-major order (the order both the Anima DiT and
    the diffusers UNet flatten a latent grid into). Each region's mask image is
    resampled to the ``grid_w``/``grid_h`` token grid.
    """
    tokens = grid_w * grid_h
    table = np.zeros((tokens, len(region_entries)), dtype=bool)
    for index, region in enumerate(region_entries):
        mask = region.get("mask")
        if mask is None:
            continue
        table[:, index] = _grid(mask, grid_w, grid_h).reshape(-1)
    return table


def _bias_for(strength):
    """The additive logit bias that scales a region block by ``strength``.

    A softmax weight is ``exp(logit)``, so a bias of ``log(strength)`` multiplies
    a region's columns by ``strength`` relative to the base block: ``1.0`` (the
    default) is a no-op, ``0`` blocks the block entirely for its tokens, and
    larger values amplify it.
    """
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        return 0.0
    if strength != strength:  # NaN
        return 0.0
    if strength <= 0.0:
        return -math.inf
    return math.log(strength)


def region_biases(region_entries):
    """The additive attention bias for each region's ``strength`` (default 1.0)."""
    return [_bias_for(region.get("strength", 1.0)) for region in region_entries]


def cross_attn_mask(region_table, block_lengths, device, biases=None):
    """An additive cross-attention mask routing tokens to prompt blocks.

    ``block_lengths`` is the token length of each context block, the shared base
    block first and then one per region. Every token sees the base block; a token
    inside region ``i`` additionally sees region ``i``'s block, scaled by that
    region's bias (``biases``, one per region, from :func:`region_biases`).
    Blocked entries are ``-inf`` and the mask is added to the attention logits,
    so it must share the query dtype. Shape ``[1, 1, tokens, sum(block_lengths)]``
    so it broadcasts over heads.
    """
    tokens, count = region_table.shape
    offsets = np.cumsum([0] + list(block_lengths)).astype(int)
    length = int(offsets[-1])
    bias = np.full((tokens, length), -np.inf, dtype="float32")
    bias[:, offsets[0]:offsets[1]] = 0.0
    for index in range(count):
        value = 0.0 if biases is None else float(biases[index])
        if value == -np.inf:
            continue
        bias[region_table[:, index], offsets[index + 1]:offsets[index + 2]] = value
    return torch.from_numpy(bias)[None, None].to(device=device)


def pad_length(total, multiple):
    """``total`` rounded up to a multiple of ``multiple`` (unchanged when falsy)."""
    if not multiple:
        return int(total)
    return -(-int(total) // int(multiple)) * int(multiple)


def joint_attn_mask(region_table, block_lengths, image_len, device,
                    text_len=None, text_keep=None, biases=None):
    """An additive ``[1, 1, L, L]`` joint self-attention mask for regional prompting.

    Unlike :func:`cross_attn_mask` (which masks a separate cross-attention with
    text keys only), this fits the joint-attention families (Flux, Z-Image,
    Qwen-Image, Krea 2) where the text and image tokens share one attention. The
    sequence is the concatenated text blocks -- the shared base block first, then
    one per region -- followed by the image tokens.

    Every query keeps full access to the image tokens, so image self-attention is
    unchanged; an image token additionally attends to the base block and to each
    region block whose mask covers it (scaled by that region's bias, ``biases``),
    and is blocked from every other region block. Text (prompt) queries keep full
    access, so the prompt tokens themselves are unchanged.

    The mask is an additive bias (``-inf`` blocks, ``0`` allows), so it must
    share the query dtype; a strength of 1.0 contributes a plain ``0`` and
    reproduces the older boolean allow-mask.

    ``region_table`` is the ``(image_tokens, regions)`` table from
    :func:`token_regions`; when ``image_len`` exceeds its row count the extra
    (padded) image tokens attend to the base block only. ``text_len`` overrides
    the text length when the caller pads the text sequence (Z-Image); ``text_keep``
    is an optional ``(text_len,)`` boolean of which text columns are real, so a
    family's own text padding mask (Krea 2) is respected.
    """
    tokens, count = region_table.shape
    offsets = np.cumsum([0] + list(block_lengths)).astype(int)
    text_total = int(offsets[-1]) if text_len is None else int(text_len)
    length = text_total + int(image_len)
    bias = np.zeros((length, length), dtype="float32")

    rows = np.arange(text_total, length)
    bias[text_total:, :text_total] = -np.inf
    bias[text_total:, offsets[0]:offsets[1]] = 0.0
    real_rows = rows[:min(tokens, int(image_len))]
    for index in range(count):
        cols = np.arange(offsets[index + 1], offsets[index + 2])
        covered = real_rows[region_table[:, index]]
        value = 0.0 if biases is None else float(biases[index])
        if covered.size and cols.size and value != -np.inf:
            bias[covered[:, None], cols[None, :]] = value
    if text_keep is not None:
        keep = np.asarray(text_keep, dtype=bool)[:text_total]
        blocked = np.flatnonzero(~keep)
        if blocked.size:
            bias[:, blocked] = -np.inf
    return torch.from_numpy(bias)[None, None].to(device=device)


def token_dims(tokens, base_w, base_h):
    """The ``(grid_w, grid_h)`` a UNet cross-attention layer's ``tokens`` imply.

    Each downsampling level halves the latent grid, so the layer's token count
    identifies its resolution. Returns ``None`` when no level matches (an
    unexpected shape), so the caller can leave that layer unmasked.
    """
    for scale in _SCALES:
        width = base_w // scale
        height = base_h // scale
        if width > 0 and height > 0 and width * height == tokens:
            return width, height
    return None


def unet_mask_provider(region_entries, base_w, base_h, block_lengths, device):
    """A ``tokens -> mask`` function for a UNet's cross-attention layers.

    Builds (and caches) the ``[1, 1, tokens, sum(block_lengths)]`` mask for each
    layer resolution the run visits. Returns ``None`` for a resolution it cannot
    map, so the caller falls back to unmasked attention there.
    """
    cache = {}
    biases = region_biases(region_entries)

    def provide(tokens):
        if tokens in cache:
            return cache[tokens]
        dims = token_dims(tokens, base_w, base_h)
        mask = None
        if dims is not None:
            table = token_regions(region_entries, dims[0], dims[1])
            mask = cross_attn_mask(table, block_lengths, device, biases=biases)
        cache[tokens] = mask
        return mask

    return provide
