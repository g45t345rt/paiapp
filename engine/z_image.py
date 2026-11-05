"""Z-Image (``lumina2``) DiT in plain torch.

Z-Image / Z-Image-Turbo (Tongyi) is a Lumina-2-style single-stream DiT
(``NextDiT``). Ported from ComfyUI's ``comfy/ldm/lumina/model.py`` (Apache-2.0),
rewritten to depend only on ``torch``. Text2image / image2image only (no Omni
reference latents, no Ming-Image layers, no pixel-space head).

Effective config (detected from the checkpoint): ``dim=3840``, 30 layers,
30 heads, ``cap_feat_dim=2560`` (Qwen3-4B), ``patch_size=2``, ``in_channels=16``,
RoPE ``axes_dims=[32,48,48]`` ``theta=256``, ``ffn_dim_multiplier=8/3``,
``z_image_modulation=True``, ``time_scale=1000``, context/image tokens padded to
a multiple of 32 with learned pad tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flux import EmbedND, apply_rope
from . import gguf as gguf_reader
from .attention import attention
from .precision import build_on_dtype, build_on_meta


def timestep_embedding(t, dim, max_period=10000):
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq=256, output_size=None):
        super().__init__()
        self.frequency_embedding_size = freq
        self.mlp = nn.Sequential(
            nn.Linear(freq, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size or hidden_size, bias=True),
        )

    def forward(self, t, dtype):
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size).to(dtype))


class _JointAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, qk_norm):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, (n_heads + 2 * n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        if qk_norm:
            self.q_norm = _RMSNorm(self.head_dim)
            self.k_norm = _RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, freqs_cis, mask=None):
        b, length, _ = x.shape
        xq, xk, xv = torch.split(
            self.qkv(x),
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim,
             self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        xq = xq.view(b, length, self.n_heads, self.head_dim)
        xk = xk.view(b, length, self.n_kv_heads, self.head_dim)
        xv = xv.view(b, length, self.n_kv_heads, self.head_dim)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq, xk = apply_rope(xq, xk, freqs_cis)
        if self.n_heads != self.n_kv_heads:
            rep = self.n_heads // self.n_kv_heads
            xk = xk.unsqueeze(3).repeat(1, 1, 1, rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, rep, 1).flatten(2, 3)
        xq = xq.movedim(1, 2)
        xk = xk.movedim(1, 2)
        xv = xv.movedim(1, 2)
        # A regional mask is an additive bias and must share the query dtype.
        if mask is not None and mask.dtype != torch.bool and mask.dtype != xq.dtype:
            mask = mask.to(xq.dtype)
        out = attention(xq, xk, xv, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, length, self.n_heads * self.head_dim)
        return self.out(out)


class _FeedForward(nn.Module):
    def __init__(self, dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        hidden = int(ffn_dim_multiplier * dim) if ffn_dim_multiplier is not None else dim
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


class _JointTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier,
                 norm_eps, qk_norm, modulation, z_image_modulation):
        super().__init__()
        self.modulation = modulation
        self.attention = _JointAttention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = _FeedForward(dim, multiple_of, ffn_dim_multiplier)
        self.attention_norm1 = _RMSNorm(dim, norm_eps)
        self.attention_norm2 = _RMSNorm(dim, norm_eps)
        self.ffn_norm1 = _RMSNorm(dim, norm_eps)
        self.ffn_norm2 = _RMSNorm(dim, norm_eps)
        if modulation:
            if z_image_modulation:
                self.adaLN_modulation = nn.Sequential(
                    nn.Linear(min(dim, 256), 4 * dim, bias=True)
                )
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(), nn.Linear(min(dim, 1024), 4 * dim, bias=True)
                )

    def forward(self, x, freqs_cis, adaln_input=None, mask=None):
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(_modulate(self.attention_norm1(x), scale_msa), freqs_cis, mask)
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(_modulate(self.ffn_norm1(x), scale_mlp))
            )
        else:
            x = x + self.attention_norm2(self.attention(self.attention_norm1(x), freqs_cis, mask))
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class _ControlBlock(_JointTransformerBlock):
    """A Z-Image control block: a transformer block plus before/after projections.

    The control branch is a small copy of the DiT's main path. Its first block
    fuses the main hidden state into the control stream (``before_proj(c) + x``);
    every block emits a skip (``after_proj``) that the matching main layer adds
    back, scaled by the conditioning strength. Both projections are zero-initialized
    so an untrained block is an exact no-op.
    """

    def __init__(self, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier,
                 norm_eps, qk_norm, modulation, z_image_modulation, first=False):
        super().__init__(dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier,
                         norm_eps, qk_norm, modulation, z_image_modulation)
        self.before_proj = nn.Linear(dim, dim) if first else None
        self.after_proj = nn.Linear(dim, dim)
        for proj in (self.before_proj, self.after_proj):
            if proj is not None:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

    def forward(self, c, x, freqs_cis, adaln_input=None, mask=None):
        if self.before_proj is not None:
            c = self.before_proj(c) + x
        c = super().forward(c, freqs_cis, adaln_input, mask)
        return c, self.after_proj(c)


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, z_image_modulation):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        min_mod = 256 if z_image_modulation else 1024
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(min(hidden_size, min_mod), hidden_size, bias=True)
        )

    def forward(self, x, c):
        return self.linear(_modulate(self.norm_final(x), self.adaLN_modulation(c)))


def _pad_to_multiple(feats, pad_token, multiple):
    if multiple is None:
        return feats
    extra = (-feats.shape[1]) % multiple
    if extra == 0:
        return feats
    pad = pad_token.to(device=feats.device, dtype=feats.dtype).unsqueeze(0).repeat(feats.shape[0], extra, 1)
    return torch.cat([feats, pad], dim=1)


class NextDiT(nn.Module):
    def __init__(self, patch_size=2, in_channels=16, dim=3840, n_layers=30,
                 n_refiner_layers=2, n_heads=30, n_kv_heads=30, multiple_of=256,
                 ffn_dim_multiplier=8.0 / 3.0, norm_eps=1e-5, qk_norm=True,
                 cap_feat_dim=2560, axes_dims=(32, 48, 48), axes_lens=(1536, 512, 512),
                 rope_theta=256.0, z_image_modulation=True, time_scale=1000.0,
                 pad_tokens_multiple=32):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = in_channels
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.z_image_modulation = z_image_modulation
        self.time_scale = time_scale
        self.pad_tokens_multiple = pad_tokens_multiple
        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, dim, bias=True)
        self.noise_refiner = nn.ModuleList([
            _JointTransformerBlock(dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier,
                                   norm_eps, qk_norm, modulation=True,
                                   z_image_modulation=z_image_modulation)
            for _ in range(n_refiner_layers)
        ])
        self.context_refiner = nn.ModuleList([
            _JointTransformerBlock(dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier,
                                   norm_eps, qk_norm, modulation=False,
                                   z_image_modulation=z_image_modulation)
            for _ in range(n_refiner_layers)
        ])
        self.t_embedder = _TimestepEmbedder(min(dim, 1024), output_size=256 if z_image_modulation else None)
        self.cap_embedder = nn.Sequential(
            _RMSNorm(cap_feat_dim, norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )
        self.layers = nn.ModuleList([
            _JointTransformerBlock(dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier,
                                   norm_eps, qk_norm, modulation=True,
                                   z_image_modulation=z_image_modulation)
            for _ in range(n_layers)
        ])
        self.final_layer = _FinalLayer(dim, patch_size, self.out_channels, z_image_modulation)
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=list(axes_dims))
        if pad_tokens_multiple is not None:
            self.x_pad_token = nn.Parameter(torch.zeros(1, dim))
            self.cap_pad_token = nn.Parameter(torch.zeros(1, dim))

        self._block_storage = None
        self._block_compute = None

        # ControlNet branch (built lazily by ``build_control`` when a control model
        # is attached); ``None`` means the DiT runs as a plain text-to-image model.
        self.control_places = None
        self.control_map = None
        self.control_in_dim = None

    def enable_block_offload(self, storage_device, compute_device):
        self._block_storage = torch.device(storage_device)
        self._block_compute = torch.device(compute_device)

    def place_head(self, device):
        for mod in (self.x_embedder, self.t_embedder, self.cap_embedder, self.final_layer, self.rope_embedder):
            mod.to(device=device)
        for name in ("x_pad_token", "cap_pad_token"):
            if hasattr(self, name):
                getattr(self, name).data = getattr(self, name).data.to(device=device)
        storage = self._block_storage if self._block_storage is not None else device
        for group in (self.noise_refiner, self.context_refiner, self.layers):
            group.to(storage)
        if self.control_places is not None:
            self.control_x_embedder.to(device=device)
            for group in (self.control_noise_refiner, self.control_layers):
                group.to(storage)

    def build_control(self, places, control_in_dim, n_refiner_layers=2, dtype=torch.bfloat16):
        """Create the ControlNet submodules (the weights are loaded by the caller).

        The control branch mirrors the main path -- a patch embedder, ``n_refiner_layers``
        noise refiner blocks and one block per entry in ``places`` -- so it can be
        streamed by the same block offload as the main layers. Rebuilding replaces
        any previous control branch.
        """
        self.control_places = list(places)
        self.control_map = {place: index for index, place in enumerate(self.control_places)}
        self.control_in_dim = int(control_in_dim)
        p = self.patch_size

        def make_block(first):
            return _ControlBlock(
                self.dim, self.n_heads, self.n_kv_heads, self.multiple_of,
                self.ffn_dim_multiplier, self.norm_eps, self.qk_norm,
                modulation=True, z_image_modulation=self.z_image_modulation, first=first,
            )

        self.control_x_embedder = build_on_dtype(
            dtype, lambda: nn.Linear(p * p * self.control_in_dim, self.dim, bias=True)
        )
        self.control_noise_refiner = build_on_dtype(dtype, lambda: nn.ModuleList([
            _JointTransformerBlock(
                self.dim, self.n_heads, self.n_kv_heads, self.multiple_of,
                self.ffn_dim_multiplier, self.norm_eps, self.qk_norm,
                modulation=True, z_image_modulation=self.z_image_modulation,
            )
            for _ in range(n_refiner_layers)
        ]))
        self.control_layers = build_on_dtype(dtype, lambda: nn.ModuleList([
            make_block(first=(index == 0)) for index in range(len(self.control_places))
        ]))

    def place_control(self):
        """Move a freshly built control branch to the DiT's head/storage devices."""
        if self.control_places is None:
            return
        if self._block_storage is None:
            # Resident: the whole model already sits on one device.
            device = next(self.layers.parameters()).device
            self.control_x_embedder.to(device=device)
            self.control_noise_refiner.to(device=device)
            self.control_layers.to(device=device)
            return
        head = self._block_compute if self._block_compute is not None else self._block_storage
        self.control_x_embedder.to(device=head)
        self.control_noise_refiner.to(self._block_storage)
        self.control_layers.to(self._block_storage)

    def control_modules(self):
        """The control branch as one module, or ``None`` when none is attached."""
        if self.control_places is None:
            return None
        return nn.ModuleList([
            self.control_x_embedder, *self.control_noise_refiner, *self.control_layers,
        ])

    def _stream(self, module, *args, **kwargs):
        if self._block_compute is not None:
            module.to(device=self._block_compute)
        out = module(*args, **kwargs)
        if self._block_compute is not None:
            module.to(device=self._block_storage)
        return out

    def _pos_ids(self, start_t, h_tokens, w_tokens, batch, device):
        ids = torch.zeros((batch, h_tokens * w_tokens, 3), dtype=torch.float32, device=device)
        ids[:, :, 0] = start_t
        ids[:, :, 1] = torch.arange(h_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, w_tokens).flatten()
        ids[:, :, 2] = torch.arange(w_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(h_tokens, 1).flatten()
        return ids

    def _unpatchify(self, x, h_tokens, w_tokens):
        p = self.patch_size
        x = x.view(x.shape[0], h_tokens, w_tokens, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
        return x

    def forward(self, x, timesteps, context, control_latent=None, control_scale=1.0,
                cross_attn_mask=None):
        batch, c, height, width = x.shape
        p = self.patch_size
        h_tokens, w_tokens = height // p, width // p

        t = 1.0 - timesteps
        adaln_input = self.t_embedder(t * self.time_scale, dtype=x.dtype)

        cap = self.cap_embedder(context.to(x.dtype))
        cap = _pad_to_multiple(cap, self.cap_pad_token, self.pad_tokens_multiple) \
            if self.pad_tokens_multiple is not None else cap
        cap_len = cap.shape[1]

        img = self.x_embedder(
            x.view(batch, c, h_tokens, p, w_tokens, p).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        )
        img = _pad_to_multiple(img, self.x_pad_token, self.pad_tokens_multiple) \
            if self.pad_tokens_multiple is not None else img

        cap_pos = torch.zeros((batch, cap_len, 3), dtype=torch.float32, device=x.device)
        cap_pos[:, :, 0] = torch.arange(cap_len, dtype=torch.float32, device=x.device) + 1.0
        cap_freqs = self.rope_embedder(cap_pos).movedim(1, 2)
        for layer in self.context_refiner:
            cap = self._stream(layer, cap, cap_freqs)

        img_len = h_tokens * w_tokens
        x_pos = self._pos_ids(cap_len + 1, h_tokens, w_tokens, batch, x.device)
        if img.shape[1] > img_len:
            x_pos = F.pad(x_pos, (0, 0, 0, img.shape[1] - img_len))
        x_freqs = self.rope_embedder(x_pos).movedim(1, 2)
        for layer in self.noise_refiner:
            img = self._stream(layer, img, x_freqs, adaln_input)

        full = torch.cat([cap, img], dim=1)
        freqs = torch.cat([cap_freqs, x_freqs], dim=1)

        # ControlNet-Union: the control branch runs once and emits one hint per
        # placed layer; each of those main layers then adds its hint back, scaled
        # by the conditioning strength.
        hints = None
        if (control_latent is not None and control_scale > 0
                and self.control_places is not None):
            hints = self._control_hints(
                control_latent, full, freqs, cap_len, x_freqs, adaln_input,
                cross_attn_mask,
            )

        for index, layer in enumerate(self.layers):
            full = self._stream(layer, full, freqs, adaln_input, mask=cross_attn_mask)
            if hints is not None and index in self.control_map:
                full = full + hints[self.control_map[index]] * control_scale

        full = self.final_layer(full, adaln_input)
        full = full[:, cap_len:cap_len + img_len]
        return -self._unpatchify(full, h_tokens, w_tokens)

    def _control_hints(self, control_latent, full, freqs, cap_len, x_freqs, adaln_input,
                       mask=None):
        """Run the control branch and return one hint per placed main layer.

        The control latent (the VAE-encoded control image, already normalized to
        the model's latent space) is patch-embedded, refined, then laid out as
        ``[caption, control]`` so it aligns element-wise with the main sequence.
        """
        batch = control_latent.shape[0]
        p = self.patch_size
        h_tokens = control_latent.shape[-2] // p
        w_tokens = control_latent.shape[-1] // p
        control = self.control_x_embedder(
            control_latent.view(batch, -1, h_tokens, p, w_tokens, p)
            .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        )
        control = _pad_to_multiple(control, self.x_pad_token, self.pad_tokens_multiple) \
            if self.pad_tokens_multiple is not None else control
        for layer in self.control_noise_refiner:
            control = self._stream(layer, control, x_freqs, adaln_input)
        control = torch.cat([full[:, :cap_len], control], dim=1)
        hints = []
        for layer in self.control_layers:
            control, skip = self._stream(layer, control, full, freqs, adaln_input, mask=mask)
            hints.append(skip)
        return hints


def _detect_config(sd):
    cap = sd["cap_embedder.1.weight"]
    dim = int(cap.shape[0])
    config = dict(
        patch_size=int(round((int(sd["x_embedder.weight"].shape[1]) / 16) ** 0.5)),
        in_channels=16,
        dim=dim,
        cap_feat_dim=int(cap.shape[1]),
        n_layers=1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers.")),
        qk_norm=True,
    )
    if dim == 3840:  # Z-Image
        config.update(
            n_heads=30, n_kv_heads=30, axes_dims=(32, 48, 48), axes_lens=(1536, 512, 512),
            rope_theta=256.0, ffn_dim_multiplier=8.0 / 3.0, z_image_modulation=True,
            time_scale=1000.0,
        )
    else:  # original Lumina 2
        config.update(
            n_heads=24, n_kv_heads=8, axes_dims=(32, 32, 32), axes_lens=(300, 512, 512),
            rope_theta=10000.0, ffn_dim_multiplier=4.0, z_image_modulation=False,
            time_scale=1.0,
        )
    config["pad_tokens_multiple"] = 32 if "cap_pad_token" in sd else None
    return config


def _strip_prefix(sd):
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if any(k.startswith(prefix) for k in sd):
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return sd


def load_z_image(path, device="cpu", dtype=torch.bfloat16, stream=False,
                 stream_device=None, ram_budget=0, prefetch_depth=0, pin_blocks=True):
    """Build the Z-Image NextDiT and load weights from safetensors or GGUF.

    With ``stream`` (GGUF only) the model is built on the meta device and its
    blocks are read from disk on demand (see :mod:`engine.streaming`), trading
    speed for a much smaller RAM footprint.
    """
    if path.lower().endswith(".gguf"):
        if stream:
            from . import streaming as streaming_module

            source = streaming_module.GgufSource(path)

            class _ShapeView:
                __slots__ = ("shape",)

                def __init__(self, shape):
                    self.shape = shape

            shapes = {
                t.name: _ShapeView(tuple(int(v) for v in reversed(t.shape)))
                for t in source.reader.tensors
            }
            config = _detect_config(shapes)
            model, _streamer = streaming_module.build_streaming_model(
                lambda: NextDiT(**config), source, device=stream_device or device,
                dtype=dtype, ram_budget=ram_budget, prefetch_depth=prefetch_depth,
                pin_blocks=pin_blocks,
            )
            missing = [n for n in dict(model.named_parameters())
                       if source.source_for(n) is None]
            return model, missing, []

        reader = gguf_reader.gguf.GGUFReader(path)
        sd = {t.name: gguf_reader.dequantize(t, dtype=dtype) for t in reader.tensors}
    else:
        from safetensors.torch import load_file
        sd = _strip_prefix(load_file(path, device="cpu"))

    # Build on the ``meta`` device: no parameter allocation and no random
    # initialiser over the whole DiT. ``sd`` already holds every tensor, so the
    # reshaped/cast tensors replace it in place and are *assigned* (``assign=True``),
    # avoiding a second full copy of the weights.
    with build_on_meta(dtype, lambda: NextDiT(**_detect_config(sd))) as model:
        pass
    params = dict(model.named_parameters())
    loaded = set()
    for name, dst in params.items():
        src = sd.get(name)
        if src is None:
            continue
        sd[name] = src.reshape(dst.shape).to(dtype)
        loaded.add(name)
    model.load_state_dict(sd, strict=False, assign=True)
    missing = [n for n in params if n not in loaded]
    # Materialise the loaded parameters in the target dtype/device; params the
    # file did not carry stay on ``meta`` and are skipped.
    for p in model.parameters():
        if p.device.type != "meta" and (p.dtype != dtype or str(p.device) != str(device)):
            p.data = p.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, missing, []
