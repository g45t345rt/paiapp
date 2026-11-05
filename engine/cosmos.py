"""Cosmos-Predict2 DiT (``MiniTrainDIT``) used by Anima.

This is a clean-room re-implementation of the Cosmos-Predict2 image DiT that
``anima_aestheticV11.safetensors`` contains: adaLN-LoRA modulated transformer
blocks (self-attention, cross-attention, MLP) with 3D RoPE, 2x2 patchify and a
padding-mask channel. Only standard ``torch`` is used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import attention


def _rms_norm(x, weight, eps):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return _rms_norm(x, self.weight, self.eps)


def apply_rotary_pos_emb(t, freqs):
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


def _frame_positions(t_index, height, width, h0, w0, device):
    """Per-token ``(t, h, w)`` positions for one frame, offset by ``h0``/``w0``.

    Used to build explicit RoPE coordinates for a reference frame shifted past
    the target grid (see :meth:`MiniTrainDIT.forward_with_ref`).
    """
    hh, ww = torch.meshgrid(
        torch.arange(height, device=device) + h0,
        torch.arange(width, device=device) + w0,
        indexing="ij",
    )
    return torch.full_like(hh, t_index).flatten(), hh.flatten(), ww.flatten()


def _rope_from_positions(pe, t, h, w):
    """3D RoPE rotations for explicit per-token ``(t, h, w)`` positions.

    ``VideoRopePosition3DEmb.forward`` only lays tokens out on a dense grid;
    the reference adapter needs the reference frame's rows shifted by a whole
    image height, so the rotations are built from explicit coordinates here.
    """
    device = t.device
    h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(device))
    t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(device))

    def rotations(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack(
            [angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1
        )

    emb = torch.cat(
        [rotations(t, t_freqs), rotations(h, h_freqs), rotations(w, w_freqs)], dim=-2
    )
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _pad_to_patch(x, patch_temporal, patch_spatial):
    """Zero-pad ``x`` (``[B, C, T, H, W]``) so every axis divides by its patch."""
    t, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    pt = (patch_temporal - t % patch_temporal) % patch_temporal
    ph = (patch_spatial - h % patch_spatial) % patch_spatial
    pw = (patch_spatial - w % patch_spatial) % patch_spatial
    if pt or ph or pw:
        x = F.pad(x, (0, pw, 0, ph, 0, pt))
    return x


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, n_heads=8, head_dim=64):
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def _sdpa(self, q, k, v, attn_mask=None):
        b, s, h, d = q.shape
        q = q.reshape(b, s, h * d)
        k = k.reshape(k.shape[0], k.shape[1], h * d)
        v = v.reshape(v.shape[0], v.shape[1], h * d)
        q = q.view(b, s, h, d).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], h, d).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], h, d).transpose(1, 2)
        # A regional mask is an additive bias and must share the query dtype.
        if attn_mask is not None and attn_mask.dtype != torch.bool and attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(q.dtype)
        out = attention(q, k, v, attn_mask=attn_mask)
        return out.transpose(1, 2).reshape(b, s, h * d)

    def compute_qkv(self, x, context=None, rope_emb=None):
        """Projected, normed and (self-attention) roped ``q, k, v`` as ``[B, S, H, D]``.

        Split out of :meth:`forward` so the asymmetric reference adapter can
        cache the reference stream's keys and values and concatenate them onto
        the target's own.
        """
        q = self.q_proj(x)
        ctx = x if context is None else context
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)
        b, s = q.shape[0], q.shape[1]
        q = self.q_norm(q.view(b, s, self.n_heads, self.head_dim))
        k = self.k_norm(k.view(k.shape[0], k.shape[1], self.n_heads, self.head_dim))
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.head_dim)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)
        return q, k, v

    def compute_attention(self, q, k, v, attn_mask=None):
        """Attention over already-projected ``q, k, v`` (see :meth:`compute_qkv`)."""
        return self.output_proj(self._sdpa(q, k, v, attn_mask))

    def forward(self, x, context=None, rope_emb=None, attn_mask=None):
        q, k, v = self.compute_qkv(x, context, rope_emb)
        return self.compute_attention(q, k, v, attn_mask)


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T):
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / half_dim
        emb = torch.exp(exponent)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb.reshape(
            timesteps_B_T.shape[0], timesteps_B_T.shape[1], -1
        ).to(device=timesteps.device)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features, use_adaln_lora=True):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

    def forward(self, sample):
        emb = self.activation(self.linear_1(sample))
        emb = self.linear_2(emb)
        if self.use_adaln_lora:
            return sample, emb
        return emb, None


class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size, temporal_patch_size, in_channels, out_channels):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            nn.Identity(),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels,
                bias=False,
            ),
        )

    def forward(self, x):
        _, _, _, H, W = x.shape
        r = self.temporal_patch_size
        m = self.spatial_patch_size
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // r, r, H // m, m, W // m, m)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels, adaln_lora_dim=256):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels,
            bias=False,
        )
        self.hidden_size = hidden_size
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, 2 * hidden_size, bias=False),
        )

    def forward(self, x, emb, adaln_lora=None):
        mod = self.adaln_modulation(emb) + adaln_lora[:, :, : 2 * self.hidden_size]
        shift, scale = mod.chunk(2, dim=-1)
        shift = shift[:, :, None, None, :]
        scale = scale[:, :, None, None, :]
        x = self.layer_norm(x) * (1 + scale) + shift
        return self.linear(x)


class Block(nn.Module):
    def __init__(self, x_dim, context_dim, num_heads, mlp_ratio=4.0, adaln_lora_dim=256):
        super().__init__()
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads)
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, x_dim // num_heads)
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))
        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_dim, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
        )
        self.adaln_modulation_cross_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_dim, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
        )
        self.adaln_modulation_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_dim, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
        )

    def forward(self, x, emb, crossattn_emb, rope_emb, adaln_lora, cross_attn_mask=None):
        sh, sc, ga = (self.adaln_modulation_self_attn(emb) + adaln_lora).chunk(3, dim=-1)
        ch, cc, cg = (self.adaln_modulation_cross_attn(emb) + adaln_lora).chunk(3, dim=-1)
        mh, mc, mg = (self.adaln_modulation_mlp(emb) + adaln_lora).chunk(3, dim=-1)

        sh, sc, ga = sh[:, :, None, None, :], sc[:, :, None, None, :], ga[:, :, None, None, :]
        ch, cc, cg = ch[:, :, None, None, :], cc[:, :, None, None, :], cg[:, :, None, None, :]
        mh, mc, mg = mh[:, :, None, None, :], mc[:, :, None, None, :], mg[:, :, None, None, :]

        B, T, H, W, D = x.shape

        nx = self.layer_norm_self_attn(x) * (1 + sc) + sh
        r = self.self_attn(nx.reshape(B, T * H * W, D), None, rope_emb=rope_emb)
        x = x + ga * r.reshape(B, T, H, W, D)

        nx = self.layer_norm_cross_attn(x) * (1 + cc) + ch
        r = self.cross_attn(
            nx.reshape(B, T * H * W, D), crossattn_emb, rope_emb=rope_emb,
            attn_mask=cross_attn_mask,
        )
        x = x + cg * r.reshape(B, T, H, W, D)

        nx = self.layer_norm_mlp(x) * (1 + mc) + mh
        x = x + mg * self.mlp(nx)
        return x


class VideoRopePosition3DEmb(nn.Module):
    def __init__(self, head_dim, len_h, len_w, len_t, base_fps=24,
                 h_extrapolation_ratio=1.0, w_extrapolation_ratio=1.0,
                 t_extrapolation_ratio=1.0, enable_fps_modulation=True):
        super().__init__()
        self.base_fps = base_fps
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )
        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def forward(self, x, fps=None):
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor
        device = x.device
        h_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
        w_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
        t_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))

        B, T, H, W, _ = x.shape
        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)
        h = torch.outer(seq[:H], h_freqs)
        w = torch.outer(seq[:W], w_freqs)
        t = torch.outer(seq[:T], t_freqs)

        def rot(e):
            return torch.stack([torch.cos(e), -torch.sin(e), torch.sin(e), torch.cos(e)], dim=-1)

        h = rot(h)
        w = rot(w)
        t = rot(t)
        h = h[None, :, None].expand(T, H, W, -1, -1)
        w = w[None, None, :].expand(T, H, W, -1, -1)
        t = t[:, None, None].expand(T, H, W, -1, -1)
        em = torch.cat([t, h, w], dim=-2)
        em = em.reshape(T * H * W, -1, 2, 2)
        return em.float()


def _block_mods(block, emb, adaln):
    """``(shift, scale, gate)`` ``[B, T, 1, 1, D]`` triples for self-attn, cross-attn and MLP."""
    out = []
    for modulation in (
        block.adaln_modulation_self_attn,
        block.adaln_modulation_cross_attn,
        block.adaln_modulation_mlp,
    ):
        m = modulation(emb) + adaln
        out.append(tuple(t[:, :, None, None, :] for t in m.chunk(3, dim=-1)))
    return out


def _mod_norm(x, layer_norm, scale, shift):
    return layer_norm(x) * (1 + scale) + shift


def _asym_block(block, xt, xr, emb_t, emb_r, adaln_t, adaln_r, context, rope_t, rope_r,
                ref_kv=None):
    """``Block.forward`` with asymmetric attention (the v2 adapter).

    The reference stream attends only to itself and skips text cross-attention,
    so its keys and values do not depend on the step, the prompt or the target
    and can be computed once and reused. The target attends to itself plus the
    reference (its cached keys/values concatenated after its own) and then to the
    text. Returns ``(xt, xr, (k_r, v_r))``.
    """
    attn = block.self_attn
    k_r = v_r = None
    if xr is not None:
        sa, _ca, mlp = _block_mods(block, emb_r, adaln_r)
        b, t, h, w, _ = xr.shape
        n = _mod_norm(xr, block.layer_norm_self_attn, sa[1], sa[0]).flatten(1, 3)
        q_r, k_r, v_r = attn.compute_qkv(n, None, rope_emb=rope_r)
        out = attn.compute_attention(q_r, k_r, v_r).reshape(b, t, h, w, -1)
        xr = xr + sa[2] * out
        n = _mod_norm(xr, block.layer_norm_mlp, mlp[1], mlp[0])
        xr = xr + mlp[2] * block.mlp(n)
    elif ref_kv is not None:
        k_r, v_r = ref_kv
    if xt is not None:
        sa, ca, mlp = _block_mods(block, emb_t, adaln_t)
        b, t, h, w, _ = xt.shape
        n = _mod_norm(xt, block.layer_norm_self_attn, sa[1], sa[0]).flatten(1, 3)
        q_t, k_t, v_t = attn.compute_qkv(n, None, rope_emb=rope_t)
        n_q = q_t.shape[1]
        if k_r is not None:
            k_t = torch.cat([k_t, k_r.to(k_t)], dim=1)
            v_t = torch.cat([v_t, v_r.to(v_t)], dim=1)
        out = attn.compute_attention(q_t, k_t, v_t)[:, :n_q].reshape(b, t, h, w, -1)
        xt = xt + sa[2] * out
        n = _mod_norm(xt, block.layer_norm_cross_attn, ca[1], ca[0]).flatten(1, 3)
        out = block.cross_attn(n, context).reshape(b, t, h, w, -1)
        xt = xt + ca[2] * out
        n = _mod_norm(xt, block.layer_norm_mlp, mlp[1], mlp[0])
        xt = xt + mlp[2] * block.mlp(n)
    return xt, xr, (k_r, v_r)


def _frame_embed(dit, latent, t_index, h0, w0, sigma):
    """Patch embed, RoPE and timestep embed of one frame ``[B, C, 1, H, W]``."""
    x = _pad_to_patch(latent, dit.patch_temporal, dit.patch_spatial)
    if dit.concat_padding_mask:
        mask = torch.zeros(
            x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4],
            dtype=x.dtype, device=x.device,
        )
        x = torch.cat([x, mask], dim=1)
    x_emb = dit.x_embedder(x)
    _, _, hp, wp, _ = x_emb.shape
    rope = _rope_from_positions(
        dit.pos_embedder, *_frame_positions(t_index, hp, wp, h0, w0, latent.device)
    )
    emb, adaln = dit.t_embedder[1](
        dit.t_embedder[0](sigma.reshape(-1, 1).float()).to(x_emb.dtype)
    )
    if x_emb.dtype == torch.float16:
        x_emb = x_emb.float()
    return x_emb, rope.unsqueeze(1).unsqueeze(0), dit.t_embedding_norm(emb), adaln


class MiniTrainDIT(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, model_channels=2048,
                 num_blocks=28, num_heads=16, crossattn_emb_channels=1024,
                 patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
                 max_img_h=240, max_img_w=240, max_frames=128,
                 adaln_lora_dim=256, rope_h_extrapolation_ratio=4.0,
                 rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0):
        super().__init__()
        self.model_channels = model_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.concat_padding_mask = concat_padding_mask

        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=model_channels // num_heads,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            h_extrapolation_ratio=rope_h_extrapolation_ratio,
            w_extrapolation_ratio=rope_w_extrapolation_ratio,
            t_extrapolation_ratio=rope_t_extrapolation_ratio,
            enable_fps_modulation=False,
        )

        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=True),
        )

        cin = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(patch_spatial, patch_temporal, cin, model_channels)
        self.blocks = nn.ModuleList([
            Block(model_channels, crossattn_emb_channels, num_heads, 4.0, adaln_lora_dim)
            for _ in range(num_blocks)
        ])
        self.final_layer = FinalLayer(
            model_channels, patch_spatial, patch_temporal, out_channels, adaln_lora_dim
        )
        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

    def unpatchify(self, x):
        B, T, H, W, _ = x.shape
        p = self.patch_spatial
        t = self.patch_temporal
        C = x.shape[-1] // (p * p * t)
        x = x.reshape(B, T, H, W, p, p, t, C)
        x = x.permute(0, 7, 1, 6, 2, 4, 3, 5)
        return x.reshape(B, C, T * t, H * p, W * p)

    def forward(self, x, timesteps, context, cross_attn_mask=None):
        if self.concat_padding_mask:
            mask = torch.zeros(
                x.shape[0], 1, x.shape[3], x.shape[4], dtype=x.dtype, device=x.device
            )
            x = torch.cat([x, mask.unsqueeze(1).repeat(1, 1, x.shape[2], 1, 1)], dim=1)

        x = self.x_embedder(x)
        rope = self.pos_embedder(x)[None, :, None]

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_emb, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(x.dtype))
        t_emb = self.t_embedding_norm(t_emb)

        for block in self.blocks:
            x = block(x, t_emb, context, rope, adaln_lora, cross_attn_mask)

        x = self.final_layer(x, t_emb, adaln_lora=adaln_lora)
        return self.unpatchify(x)

    def forward_with_refs(self, x, timesteps, context, refs, cross_attn_mask=None):
        """Forward with zero or more reference frames (one per adapter slot).

        ``refs`` is a list of ``(latent, t_index, h0, w0)``: each clean reference
        latent (``[B, C, 1, h, w]`` or ``[B, C, h, w]``, resized to the target
        grid when its size differs) is concatenated after the target frame at
        timestep 0 and the slot's RoPE ``t`` index, with its ``h``/``w`` token
        coordinates shifted by ``h0``/``w0`` so the model cannot copy it
        pixel-aligned (no pose copying). Only the target frame (frame 0) is
        returned, cropped back to ``x``'s original size. This mirrors the
        ComfyUI "Anima Ref Adapter" ``_ref_forward``.
        """
        orig = x.shape
        x = _pad_to_patch(x, self.patch_temporal, self.patch_spatial)
        frames = [x[:, :, :1]]
        specs = [(0, 0, 0)]
        for latent, t_index, h0, w0 in refs:
            ref = latent.unsqueeze(2) if latent.ndim == 4 else latent
            if tuple(ref.shape[-2:]) != tuple(x.shape[-2:]):
                shrink = ref.shape[-2] > x.shape[-2]
                ref = F.interpolate(
                    ref[:, :, 0].float(), size=tuple(x.shape[-2:]),
                    mode="area" if shrink else "bilinear",
                ).unsqueeze(2)
            frames.append(ref[:, :, :1].to(device=x.device, dtype=x.dtype))
            specs.append((int(t_index), int(h0), int(w0)))
        frames = torch.cat(frames, dim=2)
        if self.concat_padding_mask:
            mask = torch.zeros(
                frames.shape[0], 1, frames.shape[2], frames.shape[-2], frames.shape[-1],
                dtype=frames.dtype, device=frames.device,
            )
            frames = torch.cat([frames, mask], dim=1)

        x_emb = self.x_embedder(frames)
        _, _, hp, wp, _ = x_emb.shape
        positions = [
            _frame_positions(t, hp, wp, h, w, x.device) for (t, h, w) in specs
        ]
        rope = _rope_from_positions(
            self.pos_embedder,
            torch.cat([p[0] for p in positions]),
            torch.cat([p[1] for p in positions]),
            torch.cat([p[2] for p in positions]),
        )
        rope = rope.unsqueeze(1).unsqueeze(0)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        target_t = timesteps.reshape(timesteps.shape[0], -1)[:, :1]
        timesteps = torch.cat(
            [target_t] + [torch.zeros_like(target_t)] * len(refs), dim=1
        )
        t_emb, adaln_lora = self.t_embedder[1](
            self.t_embedder[0](timesteps).to(x_emb.dtype)
        )
        t_emb = self.t_embedding_norm(t_emb)

        for block in self.blocks:
            x_emb = block(x_emb, t_emb, context, rope, adaln_lora, cross_attn_mask)

        out = self.final_layer(x_emb, t_emb, adaln_lora=adaln_lora)
        out = self.unpatchify(out)[:, :, :1]
        return out[:, :, : orig[-3], : orig[-2], : orig[-1]]

    def forward_with_ref(self, x, timesteps, context, ref, offset_ratio=1.0,
                         cross_attn_mask=None):
        """Forward with one reference as a second, clean latent frame.

        A thin wrapper over :meth:`forward_with_refs` for the pilot single-frame
        adapter: the frame takes RoPE ``t`` index 1 and an ``h`` shift of
        ``round(offset_ratio * hp)`` target token heights.
        """
        hp = -(-x.shape[-2] // self.patch_spatial)
        return self.forward_with_refs(
            x, timesteps, context,
            [(ref, 1, round(offset_ratio * hp), 0)], cross_attn_mask,
        )

    def encode_ref_asym(self, ref, offset):
        """Per-block reference keys/values for the v2 asymmetric adapter.

        ``ref`` is a ``[B, C, 1, h, w]`` (or ``[B, C, h, w]``) clean latent at
        ``offset = (h0, w0)`` shifted RoPE coordinates (``h0`` in token rows).
        The reference stream is run once -- self-attention and MLP only, no text
        cross-attention -- and its ``(k, v)`` from every block are returned for
        :meth:`forward_asym` to reuse at every step and CFG branch.
        """
        ref = ref.unsqueeze(2) if ref.ndim == 4 else ref
        xr, rope_r, emb_r, adaln_r = _frame_embed(
            self, ref, 1, offset[0], offset[1],
            torch.zeros(ref.shape[0], device=ref.device),
        )
        cache = []
        for block in self.blocks:
            _, xr, kv = _asym_block(
                block, None, xr, None, emb_r, None, adaln_r, None, None, rope_r
            )
            cache.append(kv)
        return cache

    def forward_asym(self, x, timesteps, context, ref_kv):
        """Forward the target frame alone, attending to cached reference keys/values.

        ``ref_kv`` is :meth:`encode_ref_asym`'s per-block ``(k, v)``. Only the
        target frame (frame 0) is returned, cropped back to ``x``'s original size.
        """
        orig = x.shape
        x = _pad_to_patch(x, self.patch_temporal, self.patch_spatial)
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        sigma = timesteps.reshape(timesteps.shape[0], -1)[:, 0]
        xt, rope_t, emb_t, adaln_t = _frame_embed(self, x[:, :, :1], 0, 0, 0, sigma)
        batch = xt.shape[0]
        for index, block in enumerate(self.blocks):
            k, v = ref_kv[index]
            if k.shape[0] != batch:
                k = k.expand(batch, *k.shape[1:])
                v = v.expand(batch, *v.shape[1:])
            xt, _, _ = _asym_block(
                block, xt, None, emb_t, None, adaln_t, None, context, rope_t, None,
                (k, v),
            )
        out = self.final_layer(xt, emb_t, adaln_lora=adaln_t)
        out = self.unpatchify(out)[:, :, :1]
        return out[:, :, : orig[-3], : orig[-2], : orig[-1]]


def load_cosmos_dit(path, dtype=torch.bfloat16):
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    state = {
        k[len("model.diffusion_model."):]: v
        for k, v in state.items()
        if k.startswith("model.diffusion_model.")
    }
    model = MiniTrainDIT()
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [
        k for k in unexpected if "llm_adapter" not in k
    ]
    return model, missing, unexpected
