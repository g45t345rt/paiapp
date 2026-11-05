"""The BiRefNet network (Swin-L backbone + decoder) in plain torch.

BiRefNet (Zheng et al., "Bilateral Reference for High-Resolution Dichotomous
Image Segmentation") is published as PyTorch checkpoints (``.safetensors`` /
``.pth``) -- RMBG-2.0 and the community ``BiRefNet`` / ``BiRefNet-*`` files --
with no ONNX graph in the file, so a checkpoint needs its architecture to load
into. This module ports the reference ``BiRefNet`` (Swin-L vision backbone,
multi-scale input pyramid, squeeze module and the deformable-ASPP decoder) so
:mod:`engine.background` can run such a checkpoint the way it runs an RMBG-1.4
ISNet one in :mod:`engine.rmbg14`.

Only ``torch``, ``torchvision`` (for the deformable convolution), ``numpy`` and
``Pillow`` are used; nothing is downloaded and no ComfyUI code is imported. The
tensor names match the checkpoints these models ship, so :func:`load` loads with
``strict=True`` and reports a mismatched file instead of silently mis-loading a
different architecture.

The default configuration here is the one the released weights were trained
with: the Swin-L backbone (``embed_dim`` 192, depths ``[2, 2, 18, 2]``, heads
``[6, 12, 24, 48]``, window 12), multi-scale input concatenation, a three-level
context concat into the squeeze module, and the ``ASPPDeformable`` decoder
blocks. Inference follows the reference: RGB resized to 1024x1024, ImageNet
normalisation, and the final map is read back by :mod:`engine.background`. The
network's output is raw logits (no sigmoid), which the caller applies.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Device the checkpoint is moved to (matches :mod:`engine.rmbg14`).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Whether the reference's eval-time behaviour uses batch norm (it was trained
# with a batch size above 1, so the checkpoints carry BatchNorm statistics) and
# the fused scaled-dot-product attention path.
USE_BN = True
SDPA_ENABLED = True

# Inference resolution and ImageNet statistics the reference uses.
INPUT_SIZE = (1024, 1024)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# The network's final map is raw logits, so the caller must sigmoid it.
ACTIVATED = False

# Context channels the released weights concatenate onto the deepest feature
# before the squeeze module (the three shallow stages' doubled halves, deep to
# shallow). All of these are re-derived from a checkpoint by :func:`_infer_config`.
_CXT = (384, 768, 1536)


def _to_2tuple(value):
    """A ``(value, value)`` pair unless ``value`` already is one."""
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value)


class DropPath(nn.Module):
    """Per-sample stochastic depth (identity unless training)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


class Mlp(nn.Module):
    """The transformer block's position-wise feed-forward network."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _window_partition(x, window_size):
    """Split ``(B, H, W, C)`` into ``(num_windows*B, ws, ws, C)`` windows."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def _window_reverse(windows, window_size, h, w):
    """Inverse of :func:`_window_partition`."""
    c = int(windows.shape[-1])
    x = windows.view(-1, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, h, w, c)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop_prob = attn_drop

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(n, n, -1).permute(2, 0, 1).unsqueeze(0).to(dtype=q.dtype, device=q.device)

        if SDPA_ENABLED:
            attn_mask = relative_position_bias
            if mask is not None:
                attn_mask = attn_mask + mask.to(dtype=q.dtype).unsqueeze(1)
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop_prob if self.training else 0.0,
                is_causal=False,
            )
            x = x.transpose(1, 2).reshape(b_, n, c)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) + relative_position_bias
            if mask is not None:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
            attn = self.attn_drop(self.softmax(attn))
            x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """A Swin transformer block (W-MSA or SW-MSA followed by an MLP)."""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=_to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        b, length, c = x.shape
        h, w = self.H, self.W
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if SDPA_ENABLED:
                nw, n = mask_matrix.size(0), mask_matrix.size(1)
                attn_mask = mask_matrix.unsqueeze(0).expand(b, nw, n, n).reshape(b * nw, n, n)
            else:
                attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = _window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = _window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.view(b, h * w, c)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Downsample a stage by merging 2x2 patches."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, h, w):
        b, length, c = x.shape
        x = x.view(b, h, w, c)
        if (h % 2 == 1) or (w % 2 == 1):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(b, -1, 4 * c)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    """One Swin stage: a stack of blocks and an optional patch merging."""

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, h, w):
        hp = int(np.ceil(h / self.window_size)) * self.window_size
        wp = int(np.ceil(w / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, hp, wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0).to(x.dtype)

        for blk in self.blocks:
            blk.H, blk.W = h, w
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, h, w)
            return x, h, w, x_down, (h + 1) // 2, (w + 1) // 2
        return x, h, w, x, h, w


class PatchEmbed(nn.Module):
    """Image to patch embedding (a strided conv, then an optional norm)."""

    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size,
                              stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, h, w = x.size()
        if w % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
        if h % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            wh, ww = x.size(2), x.size(3)
            x = self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x


class SwinTransformer(nn.Module):
    """The Swin transformer backbone, returning the four stage feature maps."""

    def __init__(self, patch_size=4, in_channels=3, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels,
                                      embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(BasicLayer(
                dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer],
                num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            ))
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        for i_layer in out_indices:
            self.add_module(f"norm{i_layer}", norm_layer(self.num_features[i_layer]))

    def forward(self, x):
        x = self.patch_embed(x)
        wh, ww = x.size(2), x.size(3)
        outs = []
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        for i in range(self.num_layers):
            x_out, h, w, x, wh, ww = self.layers[i](x, wh, ww)
            if i in self.out_indices:
                x_out = getattr(self, f"norm{i}")(x_out)
                out = x_out.view(-1, h, w, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)


class DeformableConv2d(nn.Module):
    """A modulated deformable convolution (torchvision's ``deform_conv2d``)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=False):
        super().__init__()
        from torchvision.ops import deform_conv2d

        self._deform_conv2d = deform_conv2d
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        self.modulator_conv = nn.Conv2d(in_channels, kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size, stride=stride,
                                        padding=self.padding, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=self.padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        return self._deform_conv2d(
            input=x, offset=offset, weight=self.regular_conv.weight,
            bias=self.regular_conv.bias, padding=self.padding, mask=modulator,
            stride=self.stride,
        )


class _ASPPModuleDeformable(nn.Module):
    """One deformable atrous-conv branch of :class:`ASPPDeformable`."""

    def __init__(self, in_channels, planes, kernel_size, padding):
        super().__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes) if USE_BN else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    """Atrous spatial pyramid pooling with deformable convolutions."""

    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=(1, 3, 7)):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256
        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(in_channels, self.in_channelster, conv_size,
                                  padding=int(conv_size // 2))
            for conv_size in parallel_block_sizes
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_channelster) if USE_BN else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(self.in_channelster * (2 + len(self.aspp_deforms)),
                               out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if USE_BN else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x_deforms = [branch(x) for branch in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, *x_deforms, x5), dim=1)
        return self.dropout(self.relu(self.bn1(self.conv1(x))))


class BasicDecBlk(nn.Module):
    """The reference decoder block: conv-BN-ReLU, deformable ASPP, conv-BN."""

    def __init__(self, in_channels=64, out_channels=64, inter_channels=64,
                 attention=True, parallel_block_sizes=(1, 3, 7)):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        if attention:
            self.dec_att = ASPPDeformable(in_channels=inter_channels,
                                          parallel_block_sizes=parallel_block_sizes)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels) if USE_BN else nn.Identity()
        self.bn_out = nn.BatchNorm2d(out_channels) if USE_BN else nn.Identity()

    def forward(self, x):
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        if hasattr(self, "dec_att"):
            x = self.dec_att(x)
        return self.bn_out(self.conv_out(x))


class BasicLatBlk(nn.Module):
    """The reference lateral block (a 1x1 conv)."""

    def __init__(self, in_channels=64, out_channels=64, ks=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ks, s, p)

    def forward(self, x):
        return self.conv(x)


class SimpleConvs(nn.Module):
    """A two-conv input-projection block for the multi-scale skip inputs."""

    def __init__(self, in_channels, out_channels, inter_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


def _image2patches(image, patch_ref):
    """Split ``image`` into ``patch_ref``-sized patches, channels-major.

    The reference's ``rearrange`` ``b c (hg h) (wg w) -> b (c hg wg) h w``.
    """
    b, c, h_img, w_img = image.shape
    h, w = patch_ref.shape[2], patch_ref.shape[3]
    gh, gw = h_img // h, w_img // w
    x = image.view(b, c, gh, h, gw, w)
    return x.permute(0, 1, 2, 4, 3, 5).reshape(b, c * gh * gw, h, w)


class Decoder(nn.Module):
    """The BiRefNet decoder (multi-scale skip inputs + deformable ASPP blocks).

    ``channels`` are the lateral widths, deep to shallow (already doubled for
    multi-scale input concat). ``dec_ipt`` / ``split`` add the multi-scale
    input-projection blocks, ``out_ref`` the gradient-attention gates and
    ``ms_supervision`` the training-only side heads; each switch is inferred
    from the checkpoint so the module matches the file exactly.
    """

    def __init__(self, channels, dec_ipt=True, split=True, out_ref=True,
                 ms_supervision=True, attention=True, parallel_block_sizes=(1, 3, 7)):
        super().__init__()
        self.dec_ipt = dec_ipt
        self.out_ref = out_ref
        dec_blk_out_channels = channels[1:] + [channels[-1] // 2]
        ipt_out = [c // 8 for c in channels]
        if dec_ipt:
            ipt_in = [2 ** i * 3 for i in (10, 8, 6, 4, 0)] if split else [3] * 5
            dec_blk_in_channels = [
                channels[i] + ipt_out[max(0, i - 1)] for i in range(len(channels))
            ]
        else:
            dec_blk_in_channels = list(channels)

        if dec_ipt:
            self.ipt_blk5 = SimpleConvs(ipt_in[0], ipt_out[0])
            self.ipt_blk4 = SimpleConvs(ipt_in[1], ipt_out[0])
            self.ipt_blk3 = SimpleConvs(ipt_in[2], ipt_out[1])
            self.ipt_blk2 = SimpleConvs(ipt_in[3], ipt_out[2])
            self.ipt_blk1 = SimpleConvs(ipt_in[4], ipt_out[3])

        blk = lambda i: BasicDecBlk(  # noqa: E731
            dec_blk_in_channels[i], dec_blk_out_channels[i],
            attention=attention, parallel_block_sizes=parallel_block_sizes,
        )
        self.decoder_block4 = blk(0)
        self.decoder_block3 = blk(1)
        self.decoder_block2 = blk(2)
        self.decoder_block1 = blk(3)
        out_in = dec_blk_out_channels[3] + (ipt_out[3] if dec_ipt else 0)
        self.conv_out1 = nn.Sequential(nn.Conv2d(out_in, 1, 1, 1, 0))

        self.lateral_block4 = BasicLatBlk(channels[1], dec_blk_out_channels[0])
        self.lateral_block3 = BasicLatBlk(channels[2], dec_blk_out_channels[1])
        self.lateral_block2 = BasicLatBlk(channels[3], dec_blk_out_channels[2])

        if ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(dec_blk_out_channels[0], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(dec_blk_out_channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(dec_blk_out_channels[2], 1, 1, 1, 0)
        if out_ref:
            n_gdt = 16
            self.gdt_convs_4 = self._gdt(dec_blk_out_channels[0], n_gdt)
            self.gdt_convs_3 = self._gdt(dec_blk_out_channels[1], n_gdt)
            self.gdt_convs_2 = self._gdt(dec_blk_out_channels[2], n_gdt)
            self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(n_gdt, 1, 1, 1, 0))
            self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(n_gdt, 1, 1, 1, 0))
            self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(n_gdt, 1, 1, 1, 0))
            self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(n_gdt, 1, 1, 1, 0))
            self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(n_gdt, 1, 1, 1, 0))
            self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(n_gdt, 1, 1, 1, 0))

    @staticmethod
    def _gdt(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels) if USE_BN else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        x, x1, x2, x3, x4 = features

        if self.dec_ipt:
            patches = _image2patches(x, patch_ref=x4)
            x4 = torch.cat((x4, self.ipt_blk5(F.interpolate(
                patches, size=x4.shape[2:], mode="bilinear", align_corners=True))), 1)
        p4 = self.decoder_block4(x4)
        if self.out_ref:
            p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()

        _p3 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=True)
        _p3 = _p3 + self.lateral_block4(x3)
        if self.dec_ipt:
            patches = _image2patches(x, patch_ref=_p3)
            _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(
                patches, size=x3.shape[2:], mode="bilinear", align_corners=True))), 1)
        p3 = self.decoder_block3(_p3)
        if self.out_ref:
            p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()

        _p2 = F.interpolate(p3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        _p2 = _p2 + self.lateral_block3(x2)
        if self.dec_ipt:
            patches = _image2patches(x, patch_ref=_p2)
            _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(
                patches, size=x2.shape[2:], mode="bilinear", align_corners=True))), 1)
        p2 = self.decoder_block2(_p2)
        if self.out_ref:
            p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()

        _p1 = F.interpolate(p2, size=x1.shape[2:], mode="bilinear", align_corners=True)
        _p1 = _p1 + self.lateral_block2(x1)
        if self.dec_ipt:
            patches = _image2patches(x, patch_ref=_p1)
            _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(
                patches, size=x1.shape[2:], mode="bilinear", align_corners=True))), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode="bilinear", align_corners=True)

        if self.dec_ipt:
            patches = _image2patches(x, patch_ref=_p1)
            _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(
                patches, size=x.shape[2:], mode="bilinear", align_corners=True))), 1)
        return [self.conv_out1(_p1)]


class BiRefNet(nn.Module):
    """The BiRefNet network (``forward`` returns the final logit map in a list).

    The constructor defaults describe the released RMBG-2.0 / ``BiRefNet``
    weights; :func:`_infer_config` overrides them from a checkpoint so other
    variants (a different backbone width, or a decoder without the
    multi-scale input concat) load into the matching architecture.
    """

    def __init__(self, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
                 window_size=12, lateral_channels=None, cxt=_CXT,
                 squeeze=True, dec_ipt=True, split=True, out_ref=True,
                 ms_supervision=True, attention=True, parallel_block_sizes=(1, 3, 7)):
        super().__init__()
        self.bb = SwinTransformer(
            patch_size=4, in_channels=3, embed_dim=embed_dim, depths=list(depths),
            num_heads=list(num_heads), window_size=window_size,
        )
        if lateral_channels is None:
            lateral_channels = [2 * embed_dim * 2 ** (3 - i) for i in range(4)]
        self.cxt = list(cxt)
        if squeeze:
            self.squeeze_module = nn.Sequential(BasicDecBlk(
                lateral_channels[0] + sum(self.cxt), lateral_channels[0],
                attention=attention, parallel_block_sizes=parallel_block_sizes,
            ))
        self.decoder = Decoder(
            lateral_channels, dec_ipt=dec_ipt, split=split, out_ref=out_ref,
            ms_supervision=ms_supervision, attention=attention,
            parallel_block_sizes=parallel_block_sizes,
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.bb(x)
        _, _, h, w = x.shape
        pyramid = F.interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
        x1_, x2_, x3_, x4_ = self.bb(pyramid)
        x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode="bilinear",
                                          align_corners=True)], 1)
        x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode="bilinear",
                                          align_corners=True)], 1)
        x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode="bilinear",
                                          align_corners=True)], 1)
        x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode="bilinear",
                                          align_corners=True)], 1)
        if self.cxt:
            x4 = torch.cat((
                *[F.interpolate(feature, size=x4.shape[2:], mode="bilinear",
                                align_corners=True)
                  for feature in (x1, x2, x3)[-len(self.cxt):]],
                x4,
            ), 1)
        if hasattr(self, "squeeze_module"):
            x4 = self.squeeze_module(x4)
        return self.decoder([x, x1, x2, x3, x4])


def preprocess(image, size=INPUT_SIZE):
    """Resize and ImageNet-normalise a PIL image into the ``[1, 3, H, W]`` input."""
    resized = image.convert("RGB").resize(size, Image.BILINEAR)
    array = np.asarray(resized).astype("float32") / 255.0
    array = (array - np.asarray(NORM_MEAN, "float32")) / np.asarray(NORM_STD, "float32")
    return torch.from_numpy(array.transpose(2, 0, 1)[None]).float()


def _strip_prefixes(state):
    """Drop the ``module.`` / ``_orig_mod.`` wrappers a checkpoint may carry.

    DataParallel adds ``module.``; a ``torch.compile``d training run adds
    ``_orig_mod.``. Either can nest, so the prefixes are stripped repeatedly.
    """
    prefixes = ("module.", "_orig_mod.")
    cleaned = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def _infer_config(state):
    """Read a BiRefNet checkpoint's architecture switches from its tensor names.

    The released weights all share this network's layout, but individual files
    differ (a wider/narrower Swin backbone, a decoder without the multi-scale
    input concat, gradient-attention gates present or not). Every switch below
    is observable in the state dict, so the module is built to match the file
    exactly and the strict load then verifies it.
    """
    import re

    keys = list(state.keys())
    embed = state["bb.patch_embed.proj.weight"].shape[0]
    rpbt = state["bb.layers.0.blocks.0.attn.relative_position_bias_table"]
    window = int(round((rpbt.shape[0] ** 0.5 + 1) / 2))

    pattern = re.compile(r"bb\.layers\.(\d+)\.blocks\.(\d+)\.")
    stage_blocks = {}
    for key in keys:
        match = pattern.match(key)
        if match:
            stage_blocks.setdefault(int(match.group(1)), set()).add(int(match.group(2)))
    depths = [max(stage_blocks.get(s, {0})) + 1 for s in range(4)]
    heads = [
        state[f"bb.layers.{s}.blocks.0.attn.relative_position_bias_table"].shape[1]
        for s in range(4)
    ]

    lateral = [2 * embed * 2 ** (3 - i) for i in range(4)]
    squeeze = any(key.startswith("squeeze_module.") for key in keys)
    dec_ipt = any(key.startswith("decoder.ipt_blk") for key in keys)
    split = state["decoder.ipt_blk5.conv1.weight"].shape[1] > 3 if dec_ipt else True
    out_ref = any(".gdt_convs_" in key for key in keys)
    attention = any(".dec_att." in key for key in keys)
    ms_supervision = any(".conv_ms_spvn_" in key for key in keys)

    parallel = (1, 3, 7)
    if attention:
        sizes = {}
        for key, value in state.items():
            match = re.search(
                r"dec_att\.aspp_deforms\.(\d+)\.atrous_conv\.offset_conv\.weight", key
            )
            if match:
                sizes[int(match.group(1))] = int(round((value.shape[0] / 2) ** 0.5))
        if sizes:
            parallel = tuple(sizes[i] for i in sorted(sizes))

    cxt = ()
    if squeeze:
        diff = state["squeeze_module.0.conv_in.weight"].shape[1] - lateral[0]
        shallow = [2 * embed * 2 ** i for i in (0, 1, 2)]
        for n in range(3, -1, -1):
            if sum(shallow[3 - n:]) == diff:
                cxt = tuple(shallow[3 - n:])
                break

    return dict(
        embed_dim=embed, depths=tuple(depths), num_heads=tuple(heads), window_size=window,
        lateral_channels=lateral, cxt=cxt, squeeze=squeeze, dec_ipt=dec_ipt, split=split,
        out_ref=out_ref, ms_supervision=ms_supervision, attention=attention,
        parallel_block_sizes=parallel,
    )


def load(path):
    """Load a BiRefNet checkpoint into a :class:`BiRefNet` on :data:`DEVICE`.

    A ``.safetensors`` file is read through ``safetensors``, anything else with
    ``torch.load`` (tolerating the ``state_dict``/``params`` wrappers and a
    ``module.`` / ``_orig_mod.`` prefix). The architecture is built from the
    checkpoint's own tensor names (:func:`_infer_config`) and loaded strictly,
    so a file of another family raises :class:`ValueError` instead of producing
    garbage masks.
    """
    if os.path.splitext(path)[1].lower() == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        for wrapper in ("state_dict", "params"):
            if isinstance(state, dict) and isinstance(state.get(wrapper), dict):
                state = state[wrapper]
    if not isinstance(state, dict):
        raise ValueError(f"Unrecognised background model file: {path}")

    state = _strip_prefixes(state)
    try:
        model = BiRefNet(**_infer_config(state))
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "Unsupported background model architecture (expected a BiRefNet "
            "checkpoint)."
        ) from exc
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Unsupported background model architecture (expected a BiRefNet "
            "checkpoint)."
        ) from exc
    model.eval()
    return model.to(DEVICE)


def predict(model, image):
    """Run ``model`` on a PIL image and return the ``(H, W)`` logit map.

    The map is raw logits (the network applies no sigmoid), so the caller
    rescales it after applying the sigmoid itself.
    """
    with torch.no_grad():
        scaled = model(preprocess(image).to(DEVICE))
    return scaled[-1][0, 0].cpu().numpy()
