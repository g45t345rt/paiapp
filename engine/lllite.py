"""ControlNet-LLLite for the Anima DiT.

LLLite is kohya's light control adapter: a small shared convolutional trunk turns
the control image into a token-resolution condition, and a tiny per-linear patch
(``down`` -> ``mid`` -> ``up``, gated by a FiLM term) perturbs the input of
selected DiT linears before they run. It is a few MB rather than a full
ControlNet, and one adapter covers a control type (canny, depth, pose, ...).

This mirrors kohya's ``networks/control_net_lllite_anima.py`` (Anima v2 weight
format) so community adapters -- e.g. the ``Comfy-Org/Anima-LLLite`` set -- load
unchanged. The metric layout is read from the file's ``lllite.*`` safetensors
metadata, with a shape-based fallback, and the target linears are matched by the
saved per-module key names so no ``target_layers`` parsing is needed. Only
standard ``torch`` is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The atrous dilations the conditioning trunk's optional ASPP block uses by
# default (kohya's ``ASPP_DEFAULT_DILATIONS``).
ASPP_DEFAULT_DILATIONS = (1, 2, 4, 8)

# Saved-key prefixes/fields (kohya's named weight format).
_COND_SAVED = "lllite_conditioning1."
_COND_INTERNAL = "conditioning1."
_DEPTH_SUFFIX = ".depth_embed"


def _group_norm(channels):
    """GroupNorm with the largest divisor of ``channels`` up to 8 groups."""
    groups = 8
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class _ResBlock(nn.Module):
    """Pre-activation ResBlock: GN -> SiLU -> Conv3x3 -> GN -> SiLU -> Conv3x3."""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: one branch per dilation plus a global pool.

    A ``dilation=1`` branch is a 1x1 conv; larger dilations are dilated 3x3
    convs; a global average pool is added so the block is resolution-agnostic.
    """

    def __init__(self, channels, dilations):
        super().__init__()
        branches = []
        for dilation in dilations:
            if dilation == 1:
                conv = nn.Conv2d(channels, channels, kernel_size=1)
            else:
                conv = nn.Conv2d(
                    channels, channels, kernel_size=3, padding=dilation, dilation=dilation
                )
            branches.append(nn.Sequential(conv, _group_norm(channels), nn.SiLU()))
        self.branches = nn.ModuleList(branches)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1), _group_norm(channels), nn.SiLU()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * (len(dilations) + 1), channels, kernel_size=1),
            _group_norm(channels),
            nn.SiLU(),
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        outs = [branch(x) for branch in self.branches]
        pooled = self.global_conv(self.global_pool(x))
        pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        outs.append(pooled)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
    """The shared conditioning trunk: control image -> ``[B, S, cond_emb_dim]``.

    The image is at pixel resolution (latent x8); two stride-4 convolutions bring
    it to the DiT token grid (``/16`` = latent``/2``), then ResBlocks refine it and
    a 1x1 projection + LayerNorm produce one condition vector per token.
    """

    def __init__(self, cond_dim, cond_emb_dim, n_resblocks, use_aspp=False,
                 aspp_dilations=ASPP_DEFAULT_DILATIONS, cond_in_channels=3):
        super().__init__()
        half = cond_dim // 2
        self.cond_in_channels = cond_in_channels
        self.conv1 = nn.Conv2d(cond_in_channels, half, kernel_size=4, stride=4)
        self.norm1 = _group_norm(half)
        self.conv2 = nn.Conv2d(half, half, kernel_size=3, stride=1, padding=1)
        self.norm2 = _group_norm(half)
        self.conv3 = nn.Conv2d(half, cond_dim, kernel_size=4, stride=4)
        self.norm3 = _group_norm(cond_dim)
        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None
        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for block in self.resblocks:
            h = block(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        batch, channels, height, width = h.shape
        h = h.view(batch, channels, height * width).permute(0, 2, 1).contiguous()
        return self.out_norm(h)


class LLLiteLinear(nn.Module):
    """One patched DiT linear: perturbs its input, then runs the original linear.

    ``up`` is zero-initialized, so an adapter starts as an exact identity and does
    nothing until weights are loaded. ``cond_to_film`` is also zero-initialized so
    the FiLM term starts neutral.
    """

    def __init__(self, name, org_module, cond_emb_dim, mlp_dim, multiplier=1.0):
        super().__init__()
        self.lllite_name = name
        # Held in a list so the wrapped linear is not registered as a submodule
        # (it belongs to the DiT, and must not appear in this state dict).
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.multiplier = multiplier

        in_features = org_module.in_features
        self.down = nn.Linear(in_features, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)
        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)
        self.up = nn.Linear(mlp_dim, in_features)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # Filled by LLLite.set_cond_image; a shared tensor (cond_emb) plus this
        # module's depth vector. None means "inactive" (identity).
        self.cond_emb = None
        self.depth_emb = None
        self.org_forward = None

    def apply_to(self):
        """Replace the wrapped linear's forward with this module's (once)."""
        if self.org_forward is None:
            self.org_forward = self.org_module[0].forward
            self.org_module[0].forward = self.forward

    def remove_from(self):
        """Restore the wrapped linear's original forward."""
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward
            self.org_forward = None

    def forward(self, x):
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        # Attention q/k/v arrive as (B, S, D); the MLP fc1 arrives unflattened as
        # (B, T, H, W, D). Flatten the latter, restore it on the way out.
        orig_shape = x.shape
        is_5d = x.dim() == 5
        if is_5d:
            batch, time, height, width, dim = orig_shape
            x = x.reshape(batch, time * height * width, dim)

        cond_local = self.cond_emb + self.depth_emb
        if x.shape[0] // 2 == cond_local.shape[0] and x.shape[0] != cond_local.shape[0]:
            # Classifier-free guidance batches [uncond, cond]; the adapter saw one.
            cond_local = cond_local.repeat(2, 1, 1)

        h = F.silu(self.down(x))
        gamma, beta = self.cond_to_film(cond_local).chunk(2, dim=-1)
        m = self.mid(torch.cat([cond_local, h], dim=-1))
        m = F.silu(m * (1 + gamma) + beta)
        out = self.up(m) * self.multiplier

        y = self.org_forward(x + out)
        if is_5d:
            y = y.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], -1)
        return y


# Which Linear child of which attention maps to which atomic target. Cross
# attention's k/v are text-side and shape-mismatched, so only its q is patched.
_ATTN_TARGETS = {
    (True, "q_proj"): "self_attn_q_pre",
    (True, "k_proj"): "self_attn_kv_pre",
    (True, "v_proj"): "self_attn_kv_pre",
    (False, "q_proj"): "cross_attn_q_pre",
}


def _iter_targets(dit):
    """Yield ``(saved name, linear)`` for every patchable DiT linear, in walk order.

    Names match kohya's ``lllite_dit_{module}_{child}`` (dots -> underscores), so
    they line up with the adapter's saved keys. Order is the DiT's registration
    order, which is the order the depth embeddings were indexed in at training.
    """
    for name, module in dit.named_modules():
        cls = module.__class__.__name__
        if cls == "Attention":
            is_self = bool(getattr(module, "is_selfattn", False))
            for child_name, child in module.named_children():
                if not isinstance(child, nn.Linear):
                    continue
                if (is_self, child_name) not in _ATTN_TARGETS:
                    continue
                yield f"lllite_dit_{name}_{child_name}".replace(".", "_"), child
        elif cls == "GPT2FeedForward":
            child = getattr(module, "layer1", None)
            if isinstance(child, nn.Linear):
                yield f"lllite_dit_{name}_layer1".replace(".", "_"), child


class LLLite(nn.Module):
    """The Anima ControlNet-LLLite adapter (conditioning trunk + per-linear patches).

    ``names`` restricts the patches to the linears the weights actually contain,
    so the adapter matches the file rather than a separate ``target_layers`` spec.
    """

    def __init__(self, dit, cond_emb_dim=32, mlp_dim=64, cond_dim=64, n_resblocks=1,
                 use_aspp=False, aspp_dilations=ASPP_DEFAULT_DILATIONS,
                 cond_in_channels=3, inpaint_masked_input=False, multiplier=1.0, names=None):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        self.cond_in_channels = cond_in_channels
        # The training-side RGB policy for a 4-channel (inpainting) adapter: when
        # set, the hint's RGB is zeroed where the mask is kept. It does not affect
        # the network's forward, only how the caller builds the cond image.
        self.inpaint_masked_input = inpaint_masked_input
        self.multiplier = multiplier
        # The shared conditioning (one tensor per token) once a control image is
        # set, and whether the adapter is currently contributing. Kept apart from
        # the per-module install so a step can pause/resume the adapter without
        # re-running the conditioning trunk (see :meth:`set_active`).
        self._shared = None
        self._active = True
        self.conditioning1 = _Conditioning1(
            cond_dim, cond_emb_dim, n_resblocks,
            use_aspp=use_aspp, aspp_dilations=aspp_dilations,
            cond_in_channels=cond_in_channels,
        )
        targets = [
            (name, linear) for name, linear in _iter_targets(dit)
            if names is None or name in names
        ]
        self._order = [name for name, _ in targets]
        self.lllite_modules = nn.ModuleDict({
            name: LLLiteLinear(name, linear, cond_emb_dim, mlp_dim, multiplier)
            for name, linear in targets
        })
        self.depth_embeds = nn.Parameter(torch.zeros(len(targets), cond_emb_dim))

    def set_multiplier(self, multiplier):
        self.multiplier = float(multiplier)
        for module in self.lllite_modules.values():
            module.multiplier = float(multiplier)

    def set_cond_image(self, cond_image):
        """Compute and install the condition for every patch (``None`` clears)."""
        if cond_image is None:
            self.clear_cond_image()
            return
        self._shared = self.conditioning1(cond_image)
        if self._active:
            self._install()

    def _install(self):
        for index, name in enumerate(self._order):
            module = self.lllite_modules[name]
            module.cond_emb = self._shared
            module.depth_emb = self.depth_embeds[index]

    def _clear(self):
        for module in self.lllite_modules.values():
            module.cond_emb = None
            module.depth_emb = None

    def set_active(self, active):
        """Enable or pause the adapter without recomputing its condition.

        The sampler schedules adapter start/end fractions with this: an inactive
        adapter's patched linears fall back to their original forward.
        """
        self._active = bool(active)
        if self._active and self._shared is not None:
            self._install()
        else:
            self._clear()

    def clear_cond_image(self):
        self._shared = None
        self._clear()

    def apply_to(self):
        for module in self.lllite_modules.values():
            module.apply_to()

    def remove_from(self):
        for module in self.lllite_modules.values():
            module.remove_from()


def _read_metadata(path):
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as handle:
            return dict(handle.metadata() or {})
    except Exception:
        return {}


def _count_resblocks(state):
    indices = set()
    for key in state:
        if key.startswith(_COND_SAVED + "resblocks."):
            rest = key[len(_COND_SAVED + "resblocks."):]
            head = rest.split(".", 1)[0]
            if head.isdigit():
                indices.add(int(head))
    return len(indices)


def _infer_out_channels(state, key):
    tensor = state.get(key)
    return None if tensor is None else int(tensor.shape[0])


def config_for(state, metadata):
    """Resolve the adapter config from ``lllite.*`` metadata, else from shapes."""

    def value(name, fallback):
        return metadata.get(name, fallback)

    conv1 = state.get(_COND_SAVED + "conv1.weight")
    cond_in_channels = int(value(
        "lllite.cond_in_channels", int(conv1.shape[1]) if conv1 is not None else 3,
    ))
    cond_dim = int(value("lllite.cond_dim", _infer_out_channels(state, _COND_SAVED + "conv3.weight") or 64))
    cond_emb_dim = int(value("lllite.cond_emb_dim", _infer_out_channels(state, _COND_SAVED + "proj.weight") or 32))
    n_resblocks = int(value("lllite.cond_resblocks", _count_resblocks(state)))
    use_aspp = str(value("lllite.use_aspp", "true" if any("aspp" in k for k in state) else "false")).lower() == "true"
    dilations = value("lllite.aspp_dilations", ",".join(str(d) for d in ASPP_DEFAULT_DILATIONS))
    aspp_dilations = tuple(int(d) for d in str(dilations).split(",") if d.strip())
    mlp_dim = int(value("lllite.mlp_dim", _infer_mlp_dim(state) or 64))
    inpaint_masked_input = str(
        value("lllite.inpaint_masked_input", "false")
    ).lower() == "true"
    return {
        "cond_emb_dim": cond_emb_dim,
        "mlp_dim": mlp_dim,
        "cond_dim": cond_dim,
        "n_resblocks": n_resblocks,
        "use_aspp": use_aspp,
        "aspp_dilations": aspp_dilations,
        "cond_in_channels": cond_in_channels,
        "inpaint_masked_input": inpaint_masked_input,
    }


def _infer_mlp_dim(state):
    for key, tensor in state.items():
        if key.endswith(".down.weight"):
            return int(tensor.shape[0])
    return None


def _module_names(state):
    """The adapter's per-linear module names, from the saved keys."""
    names = set()
    for key in state:
        if key.startswith(_COND_SAVED):
            continue
        if key.endswith(_DEPTH_SUFFIX):
            names.add(key[: -len(_DEPTH_SUFFIX)])
        else:
            names.add(key.split(".", 1)[0])
    return names


def _to_internal_state(state, order):
    """Rename the file's keys to this model's layout (kohya's named -> internal)."""
    index = {name: i for i, name in enumerate(order)}
    renamed = {}
    depth = {}
    for key, tensor in state.items():
        if key.startswith(_COND_SAVED):
            renamed[_COND_INTERNAL + key[len(_COND_SAVED):]] = tensor
        elif key.endswith(_DEPTH_SUFFIX):
            name = key[: -len(_DEPTH_SUFFIX)]
            if name in index:
                depth[index[name]] = tensor
        else:
            head = key.split(".", 1)[0]
            if head in index:
                renamed[f"lllite_modules.{key}"] = tensor
            else:
                renamed[key] = tensor
    if depth:
        renamed["depth_embeds"] = torch.stack([depth[i] for i in range(len(order))], dim=0)
    return renamed


def load(dit, path, device="cpu", dtype=torch.bfloat16, multiplier=1.0):
    """Build an :class:`LLLite` for ``dit`` from ``path`` (weights not applied).

    The caller must call :meth:`LLLite.apply_to` to install it and
    :meth:`LLLite.remove_from` to detach it. Returns ``(model, info)`` where info
    is the (missing, unexpected) key report.
    """
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    config = config_for(state, _read_metadata(path))
    names = _module_names(state)
    model = LLLite(dit, names=names, multiplier=multiplier, **config)
    renamed = _to_internal_state(state, model._order)
    info = model.load_state_dict(renamed, strict=False)
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, info
