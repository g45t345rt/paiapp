"""LoRA application.

LoRA files come in several naming layouts. Each stores a low-rank pair per
targeted layer -- ``lora_A``/``lora_B`` (diffusers/PEFT) or
``lora_down``/``lora_up`` (kohya) -- with an optional ``alpha`` scalar. The
factors are kept as they are and the update ``x @ A^T @ B^T * alpha / rank *
strength`` is added to the target layer's output through a forward wrapper, so
the cached base weights are never mutated and the delta is never materialised:
the forward costs two rank-sized matmuls and holds ``rank * (in + out)`` values
instead of a second ``in * out`` matrix.

The module a key targets is looked up on the model directly (after dropping a
known container prefix such as ``diffusion_model.``), or by a normalised match
against the model's module names, which lets underscore-style keys
(``lora_unet_blocks_0_self_attn``) reach dotted paths
(``blocks.0.self_attn``). Any number of LoRAs can be stacked on the same layer;
each update is kept separately and summed at call time.
"""

import torch
import torch.nn.functional as F

# Key suffixes holding a low-rank factor or its scalar, most specific first.
# The leading "." is added by ``_split`` so the trailing base path keeps no dot.
_DOWN_SUFFIXES = (
    "lora_A.default.weight", "lora_A.weight",
    "lora_down.default.weight", "lora_down.weight",
)
_UP_SUFFIXES = (
    "lora_B.default.weight", "lora_B.weight",
    "lora_up.default.weight", "lora_up.weight",
)
_ALPHA_SUFFIXES = ("lora_alpha", "alpha")

# Container prefixes a LoRA key may carry before the model's own module path,
# so one file works against the bare module tree the engine builds.
_PREFIXES = (
    "diffusion_model.", "transformer.", "unet.", "model.",
    "diffusion_model_", "lora_unet_", "lora_transformer_", "lora_dit_",
)


def _split(key, suffixes):
    """``key`` without a trailing ``.<suffix>``, or ``None`` when it lacks one."""
    for suffix in suffixes:
        if key.endswith("." + suffix):
            return key[: -(len(suffix) + 1)]
    return None


def _collect(lora_sd):
    """Group a LoRA state dict's tensors by their target module path."""
    pairs = {}
    for key, tensor in lora_sd.items():
        base = _split(key, _DOWN_SUFFIXES)
        if base is not None:
            pairs.setdefault(base, {})["down"] = tensor
            continue
        base = _split(key, _UP_SUFFIXES)
        if base is not None:
            pairs.setdefault(base, {})["up"] = tensor
            continue
        base = _split(key, _ALPHA_SUFFIXES)
        if base is not None:
            pairs.setdefault(base, {})["alpha"] = tensor
    return pairs


def _norm(name):
    """A module name reduced to its alphanumerics, for punctuation-free match."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _index(model):
    """A normalised-name -> module-path map, cached on the model (uniq only)."""
    cached = getattr(model, "_pai_lora_index", None)
    if cached is not None:
        return cached
    index = {}
    for name, _module in model.named_modules():
        if name:
            index.setdefault(_norm(name), []).append(name)
    try:
        model._pai_lora_index = index
    except Exception:
        pass
    return index


def _resolve(model, base, index):
    """The module a LoRA key addresses, or ``None`` when it cannot be found."""
    names = [base]
    for prefix in _PREFIXES:
        if base.startswith(prefix):
            names.append(base[len(prefix):])
    for name in names:
        try:
            return model.get_submodule(name)
        except AttributeError:
            pass
    for name in names:
        matches = index.get(_norm(name))
        if matches and len(matches) == 1:
            return model.get_submodule(matches[0])
    return None


def apply_lora(model, lora_sd, strength=1.0, source=None):
    """Patch ``model`` in place with ``lora_sd``. Returns the number applied.

    ``source`` labels the LoRA (usually its file path) and is kept beside each
    update so :func:`usage` can report the memory one LoRA holds on the model.
    """
    pairs = _collect(lora_sd)
    index = _index(model)
    applied = 0
    for base, parts in pairs.items():
        if "down" not in parts or "up" not in parts:
            continue
        down = parts["down"].float()
        up = parts["up"].float()
        if down.ndim != 2 or up.ndim != 2:
            continue
        module = _resolve(model, base, index)
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if up.shape[0] != weight.shape[0] or down.shape[-1] != weight.shape[-1]:
            continue

        rank = down.shape[0]
        scale = strength
        if "alpha" in parts and rank > 0:
            alpha = float(parts["alpha"].reshape(-1)[0])
            scale *= alpha / rank
        # Keep the low-rank factors beside the layer's weight: the forward then
        # costs two rank-sized matmuls and holds ``rank * (in + out)`` values,
        # instead of a full ``out * in`` delta and a second full matmul.
        store_dtype = weight.dtype
        if store_dtype not in (
            torch.float16, torch.bfloat16, torch.float32, torch.float64,
        ):
            store_dtype = torch.bfloat16
        down = down.to(device=weight.device, dtype=store_dtype).contiguous()
        up = up.to(device=weight.device, dtype=store_dtype).contiguous()

        _wrap(module, down, up, scale, source, strength)
        applied += 1
    return applied


def _wrap(module, down, up, scale, source, strength):
    """Add one LoRA's low-rank update to ``module``'s output via a wrapper."""
    entries = getattr(module, "_pai_lora", None)
    if entries is None:
        entries = module._pai_lora = []
    entries.append((down, up, scale, source, strength))
    if getattr(module, "_pai_lora_wrapped", False):
        return
    module._pai_lora_wrapped = True
    original = module.forward
    is_embedding = isinstance(module, torch.nn.Embedding)

    def forward(x, *args, **kwargs):
        out = original(x, *args, **kwargs)
        for down, up, scale, _source, _strength in module._pai_lora:
            d = down.to(device=out.device, dtype=out.dtype)
            u = up.to(device=out.device, dtype=out.dtype)
            if is_embedding:
                # weight is ``[num_embeddings, dim]``, up ``[num_embeddings, rank]``
                # and down ``[rank, dim]``, so the update is ``up[x] @ down``.
                update = torch.matmul(F.embedding(x, u), d)
            else:
                # weight is ``[out, in]``, down ``[rank, in]``, up ``[out, rank]``,
                # so ``F.linear(F.linear(x, down), up)`` gives ``x @ A^T @ B^T``.
                update = F.linear(F.linear(x, d), u)
            out = out + update * scale
        return out

    module.forward = forward


def remove_lora(model):
    """Drop every LoRA applied by :func:`apply_lora` from ``model``."""
    for module in model.modules():
        if getattr(module, "_pai_lora_wrapped", False):
            del module.forward
            module._pai_lora_wrapped = False
            module._pai_lora = []


def usage(model):
    """Summarise the LoRAs :func:`apply_lora` left on ``model``, for the UI.

    Returns one ``{"source", "strength", "bytes", "layers"}`` per LoRA, where
    ``bytes`` is the footprint of the low-rank factors it holds in memory and
    ``layers`` how many modules it targets. Empty when the model carries none.
    """
    found = {}
    for module in model.modules():
        for down, up, _scale, source, strength in getattr(module, "_pai_lora", None) or ():
            item = found.get(source)
            if item is None:
                item = found[source] = {
                    "source": source, "strength": strength, "bytes": 0, "layers": 0,
                }
            item["bytes"] += down.numel() * down.element_size()
            item["bytes"] += up.numel() * up.element_size()
            item["layers"] += 1
    return list(found.values())