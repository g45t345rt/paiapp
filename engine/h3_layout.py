"""MiniMax H3 packed-sequence layout (plain torch/numpy).

H3's DiT attends over **one packed 1-D sequence**
``[text | keyframe conditions | target audio | target video]``. This builds that
layout for a **text still** (the ``t2va`` case: no keyframes/references), i.e. the
position grid, per-row modality tags and the three index tensors the DiT consumes.

Ported from Diffusers' ``MiniMaxH3PrepareLayoutStep.build_packed_sequence``
(Apache-2.0, MiniMax/HF teams). The rotary grid is float64 and the spatial axes
use ``numpy.linspace(endpoint=False)`` (reproduced exactly, since it differs from
``torch.linspace``). Modality tags: ``0`` video, ``1`` text, ``2`` audio.
"""

import numpy as np
import torch

# One latent frame spans `5/3 * frames_per_latent` rotary units; the (1,4,4,4,4)
# pattern mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2


def _spatial_position_grid(dim, patch, sqrt_area):
    """Aspect-normalized spatial rotary axis, centred and scaled by 32 (float64)."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _frame_position_grid(latent_height, latent_width, patch_h, patch_w):
    """The ``(h, w)`` rotary coordinates of one latent frame plus the width axis."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1), width_grid


def _temporal_position_grid(num_latent_frames, origin):
    """Rotary time of every latent frame (non-uniform: ``5/3 * (1,4,4,4,4)``)."""
    spans = torch.tensor(
        [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
         for i in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def build_still_packed_sequence(num_text_tokens, latent_height, latent_width,
                                num_audio_latents, audio_channels, audio_tag=AUDIO_TAG,
                                video_tag=VIDEO_TAG, patch_size=(1, 2, 2)):
    """Build the packed layout for a text still (``t2va``, one latent frame).

    Returns ``(position_ids, token_tags, video_indices, audio_indices, text_indices)``:

    - ``position_ids`` ``[seq, 3]`` float64 ``(t, h, w)`` rotary coordinates;
    - ``token_tags`` ``[seq]`` long (0 video / 1 text / 2 audio);
    - the three index tensors give each modality's rows in the packed sequence.
    """
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_latent_frames = 1
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_audio_rows + num_video_rows

    audio_start = num_text_tokens
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    # Text rows sit on the time axis at their row index; media rows continue it.
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # Audio rows are channel-major and share the video clock; no height coordinate,
    # pinned to the two extremes of the width grid.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat([
        torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
        torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
    ])

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    video_indices = torch.arange(video_start, sequence_length)
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = TEXT_TAG
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag

    return position_ids, token_tags, video_indices, audio_indices, text_indices
