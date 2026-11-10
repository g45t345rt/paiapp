"""MiniMax H3 video+audio DiT in plain torch (ComfyUI layout).

H3 runs one stack of blocks over a single packed 1-D sequence
``[text | cond | audio | video]``; attention is full self-attention. This mirrors
ComfyUI's ``comfy/ldm/minimax/model.py`` so the released checkpoints load with a
near-identity key map (fused ``qkv_proj``, ``mlp.fc1/2``, ``adaln_proj.linear``).

The released ``*_fl2va_pruned_int8_convrot`` checkpoint is the **curve form**: its
AdaLN consumes a small shared basis interpolated from ``adaln_t_table`` (1025 x
``time_embed_dim``) instead of a time-embedding MLP, with ``apply_silu=False``.

Timesteps: the model takes the *video* sigma and derives ``t = 1 - sigma`` per
stream; audio runs on its own shifted schedule (video shift 12, audio shift 3).
The forward returns the **raw data-ward velocity** (the scheduler uses
``x0 = x_t + sigma * v``).

Only ``torch`` is required; int8/convrot dequant is handled by the loader.
"""

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import attention

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0

# modality tags
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2

# MiniMax-H3-Image's PDD head bank is baked into the checkpoint metadata. The
# values below are the released 6-step grid and are only a fallback for a
# converted checkpoint that lost its ``__metadata__`` block.
H3_IMAGE_PDD_STEPS = 6
H3_IMAGE_PDD_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)
# Reference-image rows are conditioned at the clean end of the schedule (the
# image model has no audio stream to shift).
VISUAL_COND_TIMESTEP = 0.999


def image_config(path):
    """``(pdd_steps, pdd_sigmas)`` from an H3-Image checkpoint's metadata.

    The Comfy-Org repackage stores ``{"transformer": {"pdd_steps": 6,
    "pdd_sigmas": [...]}}`` in the safetensors ``__metadata__``. A missing block
    falls back to the released 6-step grid.
    """
    steps, sigmas = H3_IMAGE_PDD_STEPS, list(H3_IMAGE_PDD_SIGMAS)
    try:
        from .safetensors_io import open_safetensors

        handle = open_safetensors(path)
        try:
            meta = handle.metadata() or {}
        finally:
            handle.close()
        raw = meta.get("config")
        if raw:
            config = json.loads(raw)
            transformer = config.get("transformer") or {}
            steps = int(transformer.get("pdd_steps") or steps)
            if transformer.get("pdd_sigmas"):
                sigmas = [float(value) for value in transformer["pdd_sigmas"]]
    except Exception:
        pass
    if len(sigmas) != steps + 1:
        steps, sigmas = H3_IMAGE_PDD_STEPS, list(H3_IMAGE_PDD_SIGMAS)
    return steps, sigmas


def time_shift_sigma(sigma, from_shift, to_shift):
    """Re-shift a flow-matching sigma from one resolution shift to another."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def _linear_dtype(module):
    """The dtype a linear's input must be cast to.

    ``FP8Linear`` stores ``weight`` in fp8 but computes in ``compute_dtype``.
    Casting activations to ``weight.dtype`` would round them to e4m3 -- and any
    magnitude above ~448 becomes NaN, which the Qwen3-VL conditioning has.
    """
    return getattr(module, "compute_dtype", module.weight.dtype)


def patchify_video(latent, patch_size=(1, 2, 2)):
    """``[B, C, T, H, W]`` -> ``[B*T/pt*H/ph*W/pw, C*pt*ph*pw]`` (frame-major)."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    """``[B, C=32, ch=2, T]`` -> ``[ch*T, 32]`` channel-major."""
    b, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _apply_split_half_rope(x, cos, sin):
    """Rotate the leading ``rotary_dim`` channels of every head (split-half).

    ``x`` is ``[S, heads, head_dim]``; ``cos``/``sin`` are ``[S, rotary_dim]``.
    """
    rotary_dim = cos.shape[-1]
    xr, xp = x[..., :rotary_dim], x[..., rotary_dim:]
    c = cos.to(x.dtype)[:, None, :]
    s = sin.to(x.dtype)[:, None, :]
    x1, x2 = xr.chunk(2, dim=-1)
    xr = xr * c + torch.cat((-x2, x1), dim=-1) * s
    return torch.cat((xr, xp), dim=-1)


class _AdalnProj(nn.Module):
    """Per-block AdaLN projection: ``t_dim -> expand * hidden * modalities``,
    chunked into ``expand`` tensors of ``[M*modalities, hidden]``."""

    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=False):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)

    def forward(self, t_emb):
        x = F.silu(t_emb) if self.apply_silu else t_emb
        x = self.linear(x.to(_linear_dtype(self.linear)))
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


class _Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x, rope=None):
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = q.view(s, self.heads, self.head_dim)
        k = k.view(s, self.heads, self.head_dim)
        v = v.view(s, self.heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            q = _apply_split_half_rope(q, cos, sin)
            k = _apply_split_half_rope(k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = attention(q, k, v).squeeze(0)
        return self.out_proj(out.transpose(0, 1).reshape(s, -1))


class _MLP(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class _RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = _Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = _MLP(hidden, ffn)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        return self.mlp(self.norm2(x)) + x


class _TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps):
        super().__init__()
        self.blocks = nn.ModuleList([
            _RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden, eps=final_eps)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class _DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, apply_silu):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = _Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = _MLP(hidden, ffn)
        self.adaln_proj = _AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu)

    def forward(self, x, t_emb, mod_segments, rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)

        def scale_shift(h, shift, scale):
            for a, b, row in mod_segments:
                h[a:b] = h[a:b] * (1.0 + scale[row].to(h.dtype)) + shift[row].to(h.dtype)
            return h

        def gate(x, gate_vec, other):
            for a, b, row in mod_segments:
                x[a:b] = x[a:b] + gate_vec[row].to(x.dtype) * other[a:b]
            return x

        h = scale_shift(self.norm1(x), shift_msa, scale_msa)
        x = gate(x, gate_msa, self.attn(h, rope))
        h = scale_shift(self.norm2(x), shift_mlp, scale_mlp)
        return gate(x, gate_mlp, self.mlp(h))


class _FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu,
                 pdd_steps=1):
        super().__init__()
        self.video_dim = video_dim
        self._pdd_steps = max(1, int(pdd_steps))
        self.norm = nn.RMSNorm(hidden, eps=eps)
        self.adaln_proj = _AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu)
        self.video_out = nn.Linear(hidden, video_dim * self._pdd_steps, bias=True)
        self.audio_out = (
            nn.Linear(hidden, audio_dim * self._pdd_steps, bias=True)
            if audio_dim is not None else None
        )

    @property
    def pdd_steps(self):
        return self.video_out.weight.shape[0] // self.video_dim

    def forward(self, x, t_emb, video_seg, audio_seg):
        shift, scale = self.adaln_proj(t_emb)

        def mod(seg):
            a, b, row = seg
            return (self.norm(x[a:b]) * (1.0 + scale[row].to(scale.dtype))
                    + shift[row].to(shift.dtype)).to(torch.float32)

        video = mod(video_seg)
        audio = mod(audio_seg) if self.audio_out is not None else None
        if self.pdd_steps > 1:
            raise RuntimeError("PDD heads need the image forward (see forward_image).")
        return self.video_out(video), (self.audio_out(audio) if audio is not None else None)

    def forward_image(self, x, t_emb, video_seg, sigma, sample_sigmas, pdd_sigmas):
        """One image-model head pass, using the PDD bank when the checkpoint has one."""
        shift, scale = self.adaln_proj(t_emb)
        a, b, row = video_seg
        h = (self.norm(x[a:b]) * (1.0 + scale[row].to(scale.dtype))
             + shift[row].to(shift.dtype)).to(torch.float32)
        if self.pdd_steps == 1:
            return self.video_out(h)
        return _pdd_head(self.video_out, h, pdd_sigmas, sigma, sample_sigmas,
                         self.pdd_steps)


def _pdd_head(head, h, grid, sigma, sample_sigmas, steps):
    """Evaluate a PDD head bank for the interval ``[sigma, sigma_next]``.

    The first row block is a full head; the later blocks are offsets distilled on
    the checkpoint's decreasing sigma grid. A step consumes the heads of every
    grid interval it spans, weighted by ``dt`` (ComfyUI's ``_pdd_head``).
    """
    grid = [float(value) for value in grid]
    if sample_sigmas is None:
        raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule.")
    schedule = torch.as_tensor(sample_sigmas, dtype=torch.float32).flatten()
    sigma = float(sigma)
    i = int((schedule - sigma).abs().argmin())
    sigma_next = float(schedule[min(i + 1, schedule.shape[0] - 1)])
    start = min(range(steps + 1), key=lambda k: abs(grid[k] - sigma))
    stop = min(range(steps + 1), key=lambda k: abs(grid[k] - sigma_next))
    start = min(start, steps - 1)
    stop = max(stop, start + 1)

    dt = [grid[k] - grid[k + 1] for k in range(start, stop)]
    total = sum(dt)
    weights = (torch.tensor(dt, dtype=h.dtype, device=h.device) / total
               if total else torch.full((len(dt),), 1.0 / len(dt), dtype=h.dtype,
                                        device=h.device))
    weight = head.weight.reshape(steps, -1, head.weight.shape[1])
    bias = head.bias.reshape(steps, -1)
    first = max(start, 1)
    merged_weight = weight[0] + torch.einsum(
        "n,noi->oi", weights[first - start:], weight[first:stop])
    merged_bias = bias[0] + torch.einsum(
        "n,no->o", weights[first - start:], bias[first:stop])
    return F.linear(h, merged_weight, merged_bias)


class MiniMaxH3DiT(nn.Module):
    """MiniMax-H3 joint video+audio diffusion transformer (curve-form AdaLN)."""

    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 time_embed_dim=8, adaln_curve_size=1025, rope_inv_freq_len=16,
                 norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
                 sigma_shift_video=12.0, sigma_shift_audio=3.0, pdd_steps=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.time_embed_dim = time_embed_dim
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        video_patch_dim = latents_dim * math.prod(self.patch_size)

        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_patch_proj = (
            nn.Linear(audio_latents_dim, hidden_size, bias=True)
            if audio_latents_dim is not None else None
        )
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)

        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32))
        self.register_buffer("adaln_t_table", torch.empty(adaln_curve_size, time_embed_dim, dtype=torch.float32))

        self.token_refiner = _TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                           attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,
                                           final_norm_eps)
        self.blocks = nn.ModuleList([
            _DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                      time_embed_dim, norm_eps, qk_norm_eps, apply_silu=False)
            for _ in range(num_layers)
        ])
        self.final_layer = _FinalLayer(hidden_size, time_embed_dim, video_patch_dim,
                                       audio_latents_dim, final_norm_eps, apply_silu=False,
                                       pdd_steps=pdd_steps)

        self._block_storage = None
        self._block_compute = None
        self._resident_count = 0

    # ------------------------------------------------------------- placement
    def enable_block_offload(self, storage_device, compute_device):
        """Keep ``blocks`` on ``storage_device``; stream them through ``compute_device``."""
        self._block_storage = torch.device(storage_device)
        self._block_compute = torch.device(compute_device)

    def set_resident_blocks(self, count, device):
        """Hold the first ``count`` blocks on ``device``; stream the rest."""
        self._resident_count = max(0, min(int(count), len(self.blocks)))
        for block in self.blocks[:self._resident_count]:
            block.to(device=device)

    def place_head(self, device):
        """Move everything except the streamed ``blocks`` to ``device``."""
        device = torch.device(device)
        for name, child in self.named_children():
            if name != "blocks":
                child.to(device=device)
        self.adaln_t_table = self.adaln_t_table.to(device)
        self.blocks.to(self._block_storage if self._block_storage is not None else device)

    def _curve_embed(self, t_vals):
        """Interpolate the AdaLN time-embedding curve at ``t`` in [0, 1]."""
        table = self.adaln_t_table
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    def _rope_freqs(self, position_ids):
        pos = position_ids.to(torch.float32).to(self.rope.inv_freq.device)
        per_axis = pos.unsqueeze(-1) * self.rope.inv_freq.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)

    def forward_still(self, video_x, audio_x, context, layout, sigma_v):
        """One text-to-image still forward.

        ``video_x`` ``[1,24,1,H,W]``, ``audio_x`` ``[1,32,2,T]``, ``context``
        ``[T_text,5120]`` (Qwen3-VL layer-50 states), ``layout`` from
        :func:`engine.h3_layout.build_still_packed_sequence`, ``sigma_v`` the
        current video sigma. Returns the raw video and audio velocities for the
        sampler's data-ward convention.
        """
        device = video_x.device
        dtype = context.dtype
        position_ids, token_tags, video_indices, audio_indices, text_indices = layout

        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        video_embed = self.video_patch_proj(video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(audio_rows).to(dtype)
        text_embed = self.token_refiner(
            self.condition_proj(context.to(_linear_dtype(self.condition_proj))))

        seq_len = position_ids.shape[0]
        x = torch.empty(seq_len, self.hidden_size, dtype=dtype, device=device)
        x[text_indices] = text_embed
        x[audio_indices] = audio_embed
        x[video_indices] = video_embed

        # t = 1 - sigma per stream
        t_v = 1.0 - sigma_v
        t_a = 1.0 - time_shift_sigma(sigma_v, self.sigma_shift_video, self.sigma_shift_audio)

        # two distinct timesteps for a still: text/video share t_v, audio t_a
        t_vals = torch.tensor([t_v, t_a], dtype=torch.float32, device=device)
        t_emb = self._curve_embed(t_vals)
        tv_row = 0
        ta_row = 1
        text_tag = int(token_tags[text_indices[0]].item()) if text_indices.numel() else TEXT_TAG
        mod_segments = [
            (int(text_indices[0]), int(text_indices[-1]) + 1, tv_row * 3 + text_tag),
            (int(audio_indices[0]), int(audio_indices[-1]) + 1, ta_row * 3 + AUDIO_TAG),
            (int(video_indices[0]), int(video_indices[-1]) + 1, tv_row * 3 + VIDEO_TAG),
        ]

        rope = self._rope_freqs(position_ids)
        rope_pair = (rope.cos(), rope.sin())

        for index, block in enumerate(self.blocks):
            streamed = self._block_compute is not None and index >= self._resident_count
            if streamed:
                block.to(device=self._block_compute)
            x = block(x, t_emb, mod_segments, rope_pair)
            if streamed:
                block.to(device=self._block_storage)

        video_seg = (int(video_indices[0]), int(video_indices[-1]) + 1, tv_row)
        audio_seg = (int(audio_indices[0]), int(audio_indices[-1]) + 1, ta_row)
        v, a = self.final_layer(x, t_emb, video_seg, audio_seg)

        lat_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3] // 2, video_x.shape[4] // 2
        video_out = unpatchify_video(v, lat_t, lat_h, lat_w, self.latents_dim, self.patch_size)
        audio_out = unpack_audio(a)
        return video_out, audio_out

    def forward_image(self, video_x, context, layout, sigma, sample_sigmas,
                      pdd_sigmas, ref_latents=(), seed=0):
        """One MiniMax-H3-Image forward (no audio stream).

        ``video_x`` is the target latent ``[1,24,1,H,W]``; ``layout`` comes from
        :func:`engine.h3_layout.build_image_packed_sequence` and ``ref_latents``
        are the encoded reference latents in the same order as its
        ``ref_indices``. Reference rows are conditioned at the clean end of the
        schedule (with the checkpoint's 0.999 noise augmentation), while text and
        target rows share the current sigma. The image checkpoint's velocity has
        the opposite sign from the video model, so the result is negated.
        """
        device = video_x.device
        dtype = context.dtype
        position_ids, token_tags, video_indices, text_indices, ref_indices = layout

        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        video_embed = self.video_patch_proj(video_rows).to(dtype)
        text_embed = self.token_refiner(
            self.condition_proj(context.to(_linear_dtype(self.condition_proj))))

        seq_len = position_ids.shape[0]
        x = torch.empty(seq_len, self.hidden_size, dtype=dtype, device=device)
        x[text_indices] = text_embed
        for index, latent in zip(ref_indices, ref_latents):
            rows = patchify_video(latent.to(torch.float32), self.patch_size)
            if VISUAL_COND_TIMESTEP < 1.0:
                # ComfyUI restarts the same CPU RNG stream for every reference.
                generator = torch.Generator("cpu").manual_seed(int(seed))
                noise = torch.randn(rows.shape, generator=generator, dtype=torch.float32)
                rows = (VISUAL_COND_TIMESTEP * rows
                        + (1.0 - VISUAL_COND_TIMESTEP) * noise.to(rows.device))
            x[index] = self.video_patch_proj(rows).to(dtype)
        x[video_indices] = video_embed

        t_v = 1.0 - float(sigma)
        t_ref = max(t_v, VISUAL_COND_TIMESTEP)
        unique_t = [t_v] if t_ref == t_v else [t_v, t_ref]
        t_row = {value: index for index, value in enumerate(unique_t)}
        t_emb = self._curve_embed(torch.tensor(unique_t, dtype=torch.float32, device=device))

        text_tag = (int(token_tags[text_indices[0]].item())
                    if text_indices.numel() else TEXT_TAG)
        mod_segments = [
            (int(text_indices[0]), int(text_indices[-1]) + 1,
             t_row[t_v] * 3 + text_tag),
        ]
        for index in ref_indices:
            if index.numel():
                mod_segments.append((int(index[0]), int(index[-1]) + 1,
                                     t_row[t_ref] * 3 + VIDEO_TAG))
        mod_segments.append((int(video_indices[0]), int(video_indices[-1]) + 1,
                             t_row[t_v] * 3 + VIDEO_TAG))

        rope = self._rope_freqs(position_ids)
        rope_pair = (rope.cos(), rope.sin())

        for index, block in enumerate(self.blocks):
            streamed = self._block_compute is not None and index >= self._resident_count
            if streamed:
                block.to(device=self._block_compute)
            x = block(x, t_emb, mod_segments, rope_pair)
            if streamed:
                block.to(device=self._block_storage)

        video_seg = (int(video_indices[0]), int(video_indices[-1]) + 1, t_row[t_v])
        v = self.final_layer.forward_image(
            x, t_emb, video_seg, sigma, sample_sigmas, pdd_sigmas)
        lat_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3] // 2, video_x.shape[4] // 2
        return -unpatchify_video(v, lat_t, lat_h, lat_w, self.latents_dim, self.patch_size)


class MiniMaxH3ImageDiT(MiniMaxH3DiT):
    """MiniMax-H3-Image: the H3 DiT without the audio stream, one latent frame."""

    def __init__(self, pdd_steps=1, **kwargs):
        kwargs.pop("audio_latents_dim", None)
        super().__init__(audio_latents_dim=None, pdd_steps=pdd_steps, **kwargs)


# Submodules kept in float32 in the released mixed-precision checkpoint.
_FP32_MODULES = ("video_patch_proj", "audio_patch_proj", "final_layer.video_out",
                 "final_layer.audio_out")


def _is_fp32_island(name):
    return any(name == m + ".weight" or name == m + ".bias" or name.startswith(m + ".")
               for m in _FP32_MODULES) or name.endswith(".adaln_proj.linear.weight") \
        or name.endswith(".adaln_proj.linear.bias") or name in ("adaln_t_table", "rope.inv_freq")


def _materialize_meta(module):
    """Give every ``meta`` parameter/buffer in ``module`` real (empty) storage.

    ``to_fp8`` only swaps the block linears, so the norms and the fp32 AdaLN
    linears of a ``meta``-built block are still storage-less and cannot be filled
    by ``copy_``. This replaces those in place with empty CPU tensors.
    """
    for name, param in list(module.named_parameters(recurse=True)):
        if not param.is_meta:
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        parent._parameters[leaf] = nn.Parameter(
            torch.empty(param.shape, dtype=param.dtype), requires_grad=False)
    for name, buf in list(module.named_buffers(recurse=True)):
        if not buf.is_meta:
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        parent._buffers[leaf] = torch.empty(buf.shape, dtype=buf.dtype)


def load_dit(path, device="cpu", dtype=torch.bfloat16, model=None, tick=None,
             fp8="auto", model_cls=MiniMaxH3DiT, model_kwargs=None):
    """Load a MiniMax H3 DiT checkpoint (ComfyUI layout, int8/convrot aware).

    Builds a full :class:`MiniMaxH3DiT` unless a ``model_cls``/prebuilt ``model``
    is given (`MiniMaxH3ImageDiT` for the image checkpoint). The key map is
    near-identity (ComfyUI names match this module's), so loading is a per-tensor
    :func:`engine.quant_formats._dequant` copy. fp32 islands are preserved.

    Memory
    ------
    The DiT is ~20B parameters, so a dense bf16 build is ~40 GB of private
    commit. On a machine whose pagefile cannot cover that, the *loader itself*
    dies with ``0xC0000005`` / Windows error 1455 (commit-limit exhaustion) while
    reading the mmap'd checkpoint -- before :func:`engine.precision.place_dit`
    ever gets a chance to shrink the model. Since ``place_dit`` converts an H3
    this size to fp8 on any consumer GPU anyway, we build on the ``meta`` device
    and load the block linears straight into :class:`engine.precision.FP8Linear`,
    halving the resident footprint (~21 GB) so the load fits. Pass ``fp8="off"``
    (or ``"full"``) to keep the old dense dtype path, and pass a prebuilt
    ``model`` to patch it in place.
    """
    from .quant_formats import _copy_weight, _dequant
    from .precision import (build_on_dtype, build_on_meta, skip_random_init,
                            to_fp8, _want_fp8)
    from .safetensors_io import open_safetensors

    kwargs = dict(model_kwargs or {})

    def factory():
        return model_cls(**kwargs)

    # Quantize during load only when the block linears can actually hold fp8 and
    # the caller has not asked for full precision. A CPU-only build (dtype fp32)
    # keeps `to_fp8` from converting the fp32 weights, so it must stay dense.
    quantize = (model is None and dtype != torch.float32
                and _want_fp8(fp8) is not False)

    if not quantize:
        if model is None:
            with skip_random_init():
                model = build_on_dtype(dtype, factory)
        # Pin the fp32 islands so the dequantized weights land at checkpoint
        # precision.
        for name, module in model.named_modules():
            if any(name == m or name.startswith(m + ".") for m in _FP32_MODULES) \
                    or name.endswith(".adaln_proj"):
                module.float()

        handle = open_safetensors(path)
        names = set(handle.keys())
        try:
            params = dict(model.named_parameters())
            buffers = dict(model.named_buffers())
            missing = []
            with torch.no_grad():
                for name, dst in list(params.items()) + list(buffers.items()):
                    if name not in names:
                        missing.append(name)
                        continue
                    tensor = _dequant(handle, names, name)
                    dst.copy_(tensor.reshape(dst.shape).to(dst.dtype))
        finally:
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
        if tick is not None:
            tick(1.0)
        return model.to(device=device), missing

    # Low-memory path: meta build, dense fp32 islands, blocks kept in fp8.
    with build_on_meta(dtype, factory) as model:
        pass
    for name, module in model.named_modules():
        if any(name == m or name.startswith(m + ".") for m in _FP32_MODULES) \
                or name.endswith(".adaln_proj"):
            module.float()

    handle = open_safetensors(path)
    names = set(handle.keys())
    missing = []
    try:
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        with torch.no_grad():
            # Head modules (patch/final/condition/refiner) are small: assign them
            # directly. ``clone`` detaches them from the mmap, whose lifetime
            # ends with ``handle`` (a bare view would fault on first use).
            head = {}
            for name, dst in params.items():
                if name.startswith("blocks."):
                    continue
                if name not in names:
                    missing.append(name)
                    continue
                head[name] = _dequant(handle, names, name).reshape(dst.shape).to(
                    dst.dtype).clone()
            model.load_state_dict(head, strict=False, assign=True)
            del head, params

            # Blocks: swap the linears to fp8 first, then fill them in place with
            # the bounded-transient chunked copier, so no bf16 block is ever
            # resident.
            for index, block in enumerate(model.blocks):
                prefix = f"blocks.{index}."
                to_fp8(block)
                _materialize_meta(block)
                for name, dst in block.named_parameters():
                    full = prefix + name
                    if full not in names:
                        missing.append(full)
                        continue
                    _copy_weight(handle, names, full, dst, dst.dtype)
                if tick is not None:
                    tick((index + 1) / len(model.blocks))

            for name, buf in buffers.items():
                if name.startswith("blocks."):
                    continue
                if name not in names:
                    missing.append(name)
                    continue
                _copy_weight(handle, names, name, buf, buf.dtype)
    finally:
        try:
            handle.__exit__(None, None, None)
        except Exception:
            pass
    if tick is not None:
        tick(1.0)
    return model.to(device=device), missing

