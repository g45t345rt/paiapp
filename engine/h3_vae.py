"""MiniMax H3 video VAE (``AutoencoderKLMiniMaxH3``) in plain torch.

H3's VAE is a causal 3D CNN **encoder** plus a non-causal **ViT decoder**: every
latent voxel becomes a token, the decoder attends over them (with 4 learned
register tokens and a zero token) and expands each token into a
``patch_size_t x patch_size x patch_size`` pixel block. 24 latent channels,
16x spatial / 4x temporal compression (``f16t4d24``).

Ported from Diffusers' ``AutoencoderKLMiniMaxH3`` (Apache-2.0, MiniMax/HF teams),
rewritten to depend only on ``torch`` -- the wrapped ``dispatch_attention_fn``,
``FeedForward`` and the split-half rotary processor are inlined.

Pixel convention: ``decode`` returns ImageNet-normalized RGB (apply
``*pixel_std + pixel_mean`` and clamp for ``[0, 1]``); see :func:`to_pixels`.

A single-frame (still) latent is special. The reference decodes one latent frame
as the first frame of a clip: this module repeats it to a full
``tokens_chunk_size`` group and keeps frame ``frame_pre_padding`` (what Fizgig's
H3-still nodes and the newer ComfyUI VAE do), because a lone latent frame decodes
banded on its own.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet statistics, the VAE's pixel space (not the usual [-1, 1]).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _CausalConv3d(nn.Conv3d):
    """3D conv with reflect spatial padding and causal (zero, front) temporal padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 spatial_padding=0, temporal_padding=0, spatial_padding_mode="reflect"):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, x):
        if self.spatial_padding > 0:
            p = self.spatial_padding
            x = F.pad(x, (p, p, p, p, 0, 0), mode=self.spatial_padding_mode)
        if self.temporal_padding > 0:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class _TGroupNorm(nn.GroupNorm):
    """GroupNorm with statistics computed per frame (time folded into batch)."""

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
        x = super().forward(x)
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class _ResnetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups=32, norm_eps=1e-6,
                 spatial_padding_mode="reflect"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = _TGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = _CausalConv3d(in_channels, out_channels, 3, spatial_padding=1,
                                   temporal_padding=2, spatial_padding_mode=spatial_padding_mode)
        self.norm2 = _TGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = _CausalConv3d(out_channels, out_channels, 3, spatial_padding=1,
                                   temporal_padding=2, spatial_padding_mode=spatial_padding_mode)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = _CausalConv3d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = x
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + h


class _Downsample3d(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_stride=1, spatial_stride=2,
                 spatial_padding_mode="reflect"):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = _CausalConv3d(in_channels, out_channels, 3,
                                  stride=(temporal_stride, spatial_stride, spatial_stride),
                                  spatial_padding=0, temporal_padding=2,
                                  spatial_padding_mode=spatial_padding_mode)

    def forward(self, x):
        if self.spatial_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(x)


class _DownBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, temporal_downsample_factor,
                 spatial_downsample_factor, norm_num_groups=32, norm_eps=1e-6,
                 spatial_padding_mode="reflect"):
        super().__init__()
        self.resnets = nn.ModuleList([
            _ResnetBlock3d(in_channels if i == 0 else out_channels, out_channels,
                           norm_num_groups=norm_num_groups, norm_eps=norm_eps,
                           spatial_padding_mode=spatial_padding_mode)
            for i in range(num_layers)
        ])
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList([
                _Downsample3d(out_channels, out_channels,
                              temporal_stride=temporal_downsample_factor,
                              spatial_stride=spatial_downsample_factor,
                              spatial_padding_mode=spatial_padding_mode)
            ])

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
        return x


class _Encoder3d(nn.Module):
    def __init__(self, in_channels=3, out_channels=48,
                 block_out_channels=(128, 256, 256, 512, 512, 1024), layers_per_block=2,
                 spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
                 temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
                 norm_num_groups=32, norm_eps=1e-6, spatial_padding_mode="reflect"):
        super().__init__()
        self.conv_in = _CausalConv3d(in_channels, block_out_channels[0], 3, spatial_padding=1,
                                     temporal_padding=2, spatial_padding_mode=spatial_padding_mode)
        block_in = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList([
            _DownBlock3d(in_channels=block_in[i], out_channels=block_out_channels[i],
                         num_layers=layers_per_block,
                         temporal_downsample_factor=temporal_downsample_factors[i],
                         spatial_downsample_factor=spatial_downsample_factors[i],
                         norm_num_groups=norm_num_groups, norm_eps=norm_eps,
                         spatial_padding_mode=spatial_padding_mode)
            for i in range(len(block_out_channels))
        ])
        self.norm_out = _TGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = _CausalConv3d(block_out_channels[-1], out_channels, 3, spatial_padding=1,
                                      temporal_padding=2, spatial_padding_mode=spatial_padding_mode)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class _RotaryPosEmbed(nn.Module):
    """3-axis rotary: length-normalized coordinates to [-1, 1), scaled by 2*pi."""

    def __init__(self, dim, theta=100.0, num_axes=3):
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"dim {dim} must be divisible by 2*num_axes {2 * num_axes}.")
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class _Attn(nn.Module):
    """Self-attention with per-head RMSNorm and split-half rotary on the first
    ``rotary_dim`` channels (inlined from Diffusers' processor)."""

    def __init__(self, dim, heads, dim_head, eps=1e-5, bias=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner, bias=bias)
        self.to_k = nn.Linear(dim, inner, bias=bias)
        self.to_v = nn.Linear(dim, inner, bias=bias)
        self.to_out = nn.Linear(inner, dim, bias=bias)

    def forward(self, x, rotary_emb=None):
        q = self.to_q(x).unflatten(2, (self.heads, -1))
        k = self.to_k(x).unflatten(2, (self.heads, -1))
        v = self.to_v(x).unflatten(2, (self.heads, -1))
        q = self.norm_q(q.float()).to(v.dtype)
        k = self.norm_k(k.float()).to(v.dtype)
        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            rotary_dim = cos.shape[-1]
            qr, qp = q[..., :rotary_dim], q[..., rotary_dim:]
            kr, kp = k[..., :rotary_dim], k[..., rotary_dim:]
            q1, q2 = qr.chunk(2, dim=-1)
            k1, k2 = kr.chunk(2, dim=-1)
            q = torch.cat([qr * cos + torch.cat([-q2, q1], -1) * sin, qp], dim=-1)
            k = torch.cat([kr * cos + torch.cat([-k2, k1], -1) * sin, kp], dim=-1)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.to_out(out.transpose(1, 2).flatten(2))


class _SwiGLU(nn.Module):
    def __init__(self, dim, mult=4, bias=True):
        super().__init__()
        inner = dim * mult
        self.w1 = nn.Linear(dim, 2 * inner, bias=bias)
        self.w2 = nn.Linear(inner, dim, bias=bias)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class _TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ffn_mult=4, eps=1e-5, bias=True):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = _Attn(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = _SwiGLU(dim, mult=ffn_mult, bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x, rotary_emb=None):
        dtype = self.attn.to_q.weight.dtype
        h = self.norm1(x.float()).to(dtype)
        x = x + self.attn(h, rotary_emb) * self.scale1
        h = self.norm2(x.float()).to(dtype)
        return x + self.ff(h) * self.scale2


class _ViTDecoder3d(nn.Module):
    def __init__(self, in_channels=24, out_channels=3, patch_size=16, patch_size_t=4,
                 num_layers=36, num_attention_heads=32, attention_head_dim=64,
                 num_register_tokens=4, ffn_mult=4, rope_theta=100.0, rope_dim_ratio=0.75,
                 norm_eps=1e-5):
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rope = _RotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        # ComfyUI's checkpoint appends a *learned* mask token where diffusers used
        # a zero token; load it (fall back to zeros if a checkpoint omits it).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.transformer_blocks = nn.ModuleList([
            _TransformerBlock(dim=dim, heads=num_attention_heads, dim_head=attention_head_dim,
                              ffn_mult=ffn_mult, eps=norm_eps)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        x = self.proj_in(x)
        num_patches = x.shape[1]
        reg = self.register_tokens.expand(b, -1, -1).to(x.dtype)
        cls = self.mask_token.expand(b, -1, -1).to(x.dtype)
        x = torch.cat([x, reg, cls], dim=1)

        grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=x.device) / size) - 1.0
                 for size in (t, h, w)]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(b, -1, -1)
        suffix = position_ids.new_zeros((b, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            x = block(x, rotary_emb)

        x = self.proj_out(self.norm_out(x).to(self.proj_out.weight.dtype))[:, :num_patches, :]
        p, pt = self.patch_size, self.patch_size_t
        x = x.view(b, t, h, w, self.out_channels, pt, p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(b, self.out_channels, t * pt, h * p, w * p)


class MiniMaxH3VideoVAE(nn.Module):
    """H3's causal-CNN-encoder / ViT-decoder video VAE (f16t4d24)."""

    def __init__(self, in_channels=3, out_channels=3, latent_channels=24,
                 block_out_channels=(128, 256, 256, 512, 512, 1024), layers_per_block=2,
                 spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
                 temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
                 norm_num_groups=32, norm_eps=1e-6, spatial_padding_mode="reflect",
                 decoder_num_layers=36, decoder_num_attention_heads=32,
                 decoder_attention_head_dim=64, decoder_num_register_tokens=4,
                 decoder_ffn_mult=4, decoder_rope_theta=100.0, decoder_rope_dim_ratio=0.75,
                 decoder_norm_eps=1e-5, clip_length=17, token_drop=3,
                 latents_mean=None, latents_std=None, tile_size=256, tile_overlap_min=64):
        super().__init__()
        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(temporal_downsample_factors)
        self.clip_length = clip_length
        self.token_drop = token_drop

        self.encoder = _Encoder3d(
            in_channels=in_channels, out_channels=2 * latent_channels,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            spatial_downsample_factors=spatial_downsample_factors,
            temporal_downsample_factors=temporal_downsample_factors,
            norm_num_groups=norm_num_groups, norm_eps=norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = _ViTDecoder3d(
            in_channels=latent_channels, out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
            patch_size_t=self.temporal_compression_ratio,
            num_layers=decoder_num_layers, num_attention_heads=decoder_num_attention_heads,
            attention_head_dim=decoder_attention_head_dim,
            num_register_tokens=decoder_num_register_tokens, ffn_mult=decoder_ffn_mult,
            rope_theta=decoder_rope_theta, rope_dim_ratio=decoder_rope_dim_ratio,
            norm_eps=decoder_norm_eps,
        )

        self.frame_pre_padding = (-clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(
            self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0)

        self.use_tiling = True
        self.tile_sample_min_height = tile_size
        self.tile_sample_min_width = tile_size
        self.tile_sample_min_overlap_height = tile_overlap_min
        self.tile_sample_min_overlap_width = tile_overlap_min

        mean = latents_mean if latents_mean is not None else (0.0,) * latent_channels
        std = latents_std if latents_std is not None else (1.0,) * latent_channels
        self.register_buffer("latents_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("latents_std", torch.tensor(std, dtype=torch.float32))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1),
                             persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1),
                             persistent=False)

    # ------------------------------------------------------------------ tiling
    def _split_tiles(self, length, tile_size, min_overlap):
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio
        starts = [0]
        for i in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[i])
        return starts, [tile_size] * num_tiles, overlaps

    def _blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(self, tiles, height_overlaps, width_overlaps):
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode_clip(self, x):
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x))
        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_ov = self._split_tiles(height, self.tile_sample_min_height,
                                               self.tile_sample_min_overlap_height)
        x_idx, x_len, x_ov = self._split_tiles(width, self.tile_sample_min_width,
                                               self.tile_sample_min_overlap_width)
        rows = [[self.quant_conv(self.encoder(x[..., i:i + il, j:j + jl]))
                 for j, jl in zip(x_idx, x_len)] for i, il in zip(y_idx, y_len)]
        ratio = self.spatial_compression_ratio
        return self._stitch_tiles(rows, [o // ratio for o in y_ov], [o // ratio for o in x_ov])

    def _decode_clip(self, z):
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))
        ratio = self.spatial_compression_ratio
        height, width = z.shape[-2] * ratio, z.shape[-1] * ratio
        y_idx, y_len, y_ov = self._split_tiles(height, self.tile_sample_min_height,
                                               self.tile_sample_min_overlap_height)
        x_idx, x_len, x_ov = self._split_tiles(width, self.tile_sample_min_width,
                                               self.tile_sample_min_overlap_width)
        rows = [[self.decoder(self.post_quant_conv(
                    z[..., i // ratio:i // ratio + il // ratio, j // ratio:j // ratio + jl // ratio]))
                 for j, jl in zip(x_idx, x_len)] for i, il in zip(y_idx, y_len)]
        return self._stitch_tiles(rows, y_ov, x_ov)

    # ------------------------------------------------------------------ pixels
    def _to_pixels(self, x):
        x = x.float() * self.pixel_std.to(x.device) + self.pixel_mean.to(x.device)
        return x.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------ decode
    def decode(self, z):
        """Decode normalized latents ``[B, 24, T_lat, H, W]`` to pixels ``[B, 3, T, H, W]``.

        One latent frame decodes as a still (repeated into a full chunk, keeping
        frame ``frame_pre_padding``); longer latents decode as a clip.
        """
        z = z * self.latents_std.view(1, -1, 1, 1, 1).to(z) + self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        if z.shape[2] == 1:
            group = z.repeat(1, 1, self.tokens_chunk_size, 1, 1)
            clip = self._decode_clip(group)
            frame = clip[:, :, self.frame_pre_padding:self.frame_pre_padding + 1]
            return self._to_pixels(frame)
        return self._to_pixels(self._decode_clip_chunked(z))

    def _decode_clip_chunked(self, z):
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.token_drop
        ratio = self.temporal_compression_ratio
        chunk_frames = tokens_chunk_size * ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        decoded, overlap = [], None
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            clip = self._decode_clip(z[:, :, start:start + tokens_chunk_size + self.token_overlap])
            for j in range(int(token_drop > 0) + 1):
                chunk = clip[:, :, j * chunk_frames:j * chunk_frames + chunk_frames]
                chunk = chunk[:, :, self.frame_pre_padding:]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    decoded.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded.append(overlap)
        dec = torch.cat(decoded, dim=2)
        if pad_tokens > 0:
            intra_tail = self.clip_length % ratio
            before = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (before + k) % tokens_chunk_size == 0 else ratio
                for k in range(pad_tokens))
            dec = dec[:, :, :-pad_frames]
        return dec

    # ------------------------------------------------------------------ encode
    def encode(self, x):
        """Encode pixels ``[B, 3, T, H, W]`` (ImageNet space) to normalized latents."""
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.shape[2] == 1:
            moments = self._encode_clip(x)
        else:
            if x.shape[2] % self.clip_length != 0:
                pad = x[:, :, -1:].repeat(1, 1, (-x.shape[2]) % self.clip_length, 1, 1)
                x = torch.cat([x, pad], dim=2)
            moments = torch.cat([
                self._encode_clip(x[:, :, i * self.clip_length:(i + 1) * self.clip_length])
                for i in range(x.shape[2] // self.clip_length)
            ], dim=2)
            if self.token_drop > 0:
                moments = moments[:, :, :-self.token_drop]
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        return (mean - self.latents_mean.view(1, -1, 1, 1, 1).to(mean)) \
            / self.latents_std.view(1, -1, 1, 1, 1).to(mean)


def to_pixels(vae, latents):
    """Convenience: decode normalized latents straight to ``[0, 1]`` pixels."""
    return vae.decode(latents)


def _dest_names(key):
    """Map a ComfyUI VAE checkpoint key to this module's parameter name(s).

    Returns a list of ``(dst_name, part)`` where ``part`` is ``None`` for a plain
    copy or ``(index, 3)`` to take one third of a fused ``to_qkv`` tensor.
    """
    if not key.startswith(("decoder.", "encoder.", "quant_conv", "post_quant_conv",
                           "latents_mean", "latents_std")):
        return [(key, None)]
    if key.startswith("decoder."):
        if key.startswith("decoder.x_embedder."):
            return [("decoder.proj_in." + key.rsplit(".", 1)[1], None)]
        if key.startswith("decoder.transformer_blocks."):
            parts = key.split(".")
            if "attn" in parts and parts[parts.index("attn") + 1] == "to_qkv":
                base = ".".join(parts[:parts.index("attn") + 1])
                leaf = parts[-1]
                return [(f"{base}.to_{n}.{leaf}", (i, 3)) for i, n in enumerate(("q", "k", "v"))]
        return [(key, None)]
    if key.startswith("encoder.down."):
        parts = key.split(".")
        di = parts[parts.index("down") + 1]
        rest = parts[parts.index("down") + 2:]
        # encoder.down.N.downsample.conv.* -> encoder.down_blocks.N.downsamplers.0.conv.*
        if rest[0] == "downsample":
            return [(f"encoder.down_blocks.{di}.downsamplers.0.{'.'.join(rest[1:])}", None)]
        # encoder.down.N.block.M.<rest> -> encoder.down_blocks.N.resnets.M.<rest>
        mi = rest[1]
        r = rest[2:]
        if r[0] == "nin_shortcut":
            r = ["conv_shortcut"] + r[1:]
        return [(f"encoder.down_blocks.{di}.resnets.{mi}.{'.'.join(r)}", None)]
    return [(key, None)]


def load_vae(path, device="cpu", dtype=torch.bfloat16, model=None, tick=None):
    """Load a MiniMax H3 video VAE checkpoint (ComfyUI layout, int8/convrot aware)."""
    from .moss_text import _dequant
    from .safetensors_io import open_safetensors

    if model is None:
        model = MiniMaxH3VideoVAE()
    model = model.to(dtype=dtype)
    handle = open_safetensors(path)
    names = set(handle.keys())
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    modules = dict(model.named_modules())
    loaded = set()
    try:
        with torch.no_grad():
            for key in list(names):
                if key.endswith("_scale") or key.endswith(".comfy_quant"):
                    continue
                tensor = _dequant(handle, names, key)
                for dst_name, part in _dest_names(key):
                    dst = params.get(dst_name)
                    if dst is None:
                        dst = buffers.get(dst_name)
                    if dst is None:
                        continue
                    value = tensor
                    if part is not None:
                        idx, total = part
                        # The fused to_qkv rows are interleaved per head
                        # ([h0 q,k,v | h1 q,k,v | ...]); see the reference's
                        # view(b, s, heads, 3*dim_head) + chunk(3, -1).
                        module = modules.get(dst_name.rsplit(".", 1)[0])
                        if not (hasattr(module, "heads") and hasattr(module, "dim_head")):
                            module = modules.get(dst_name.rsplit(".", 2)[0])
                        heads = getattr(module, "heads", None)
                        dim_head = getattr(module, "dim_head", None)
                        if heads and dim_head:
                            tail = tensor.shape[1:]
                            value = tensor.reshape(heads, total, dim_head, *tail)[:, idx]
                            value = value.reshape(heads * dim_head, *tail)
                        else:
                            value = torch.chunk(tensor, total, dim=0)[idx]
                    dst.copy_(value.reshape(dst.shape).to(dst.dtype))
                    loaded.add(dst_name)
    finally:
        try:
            handle.__exit__(None, None, None)
        except Exception:
            pass
    missing = [n for n in list(params) + list(buffers)
               if n not in loaded and not n.endswith("rope.inv_freq")
               and n not in ("pixel_mean", "pixel_std")]
    if tick is not None:
        tick(1.0)
    return model.to(device=device), missing
