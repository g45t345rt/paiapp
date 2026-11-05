"""Anima reference adapter (offset-RoPE reference frame + LoRA).

A clean-room port of the ComfyUI "Anima Ref Adapter"
(``ElectronGirl/ComfyUI-Anima-Ref-Adapter``) into the engine. A reference image
is VAE-encoded into a clean latent and a LoRA -- active **only** while a
reference is attached -- teaches the DiT to read it. Two inference paths are
supported, selected by the adapter file's metadata:

* the **pilot** (v1, no metadata, rank-32): the reference rides as a second,
  clean latent frame at timestep 0 whose RoPE ``h`` coordinate is shifted past
  the target grid; both frames share the same blocks and text cross-attention
  (:meth:`engine.cosmos.MiniTrainDIT.forward_with_ref`).
* **v2** (metadata ``{"arch": "asym"}``, rank-64): asymmetric attention. The
  reference stream attends only to itself and skips text cross-attention, so its
  per-block keys and values do not depend on the step, the prompt or the target
  and are computed once per reference (:meth:`RefAdapter.ref_kv`) and reused by
  every step and CFG branch; the target then attends to itself plus the cached
  reference (:meth:`engine.cosmos.MiniTrainDIT.forward_asym`).

This module holds the adapter LoRA, the reference key/value cache and the
conditioning the guider reads.
"""

import json
import os

import torch
import torch.nn.functional as F

from . import lora as _lora

# The adapter LoRA's forward hooks only fire while this key matches, so a plain
# single-frame run (or the guider's reference-free base branch) is unaffected.
_ACTIVE = {"key": None}

_STORE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

# Adapters trained before the "slots" metadata carry one character reference
# frame with the pilot's RoPE t index 1 and a one-image-height shift. Used when
# no adapter is loaded (the frame is attached without a LoRA).
DEFAULT_SLOTS = {"character": {"t": 1, "offset_h": 1.0, "offset_w": 0}}


def _read_metadata(path):
    """The ``anima_ref_adapter`` JSON metadata of an adapter file, or ``{}``."""
    try:
        from safetensors import safe_open

        with safe_open(path, "pt") as handle:
            raw = (handle.metadata() or {}).get("anima_ref_adapter")
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


class AnimaRefConditioning:
    """DiT cross-attention context with zero or more reference latents attached.

    ``refs`` maps an adapter slot name (``"character"``, ``"style"``) to its
    clean reference latent. The guider keeps two variants of the same prompt: one
    with the references (the ``positive``/``negative`` branches) and one without
    (the ``negative_base`` branch). Empty ``refs`` means "no reference", which
    also disables the adapter LoRA for that forward. A bare latent is accepted
    for convenience and treated as the character reference.
    """

    __slots__ = ("text", "refs")

    def __init__(self, text, refs=None):
        self.text = text
        if refs is None:
            self.refs = {}
        elif isinstance(refs, dict):
            self.refs = {slot: ref for slot, ref in refs.items() if ref is not None}
        else:
            self.refs = {"character": refs}

    @property
    def ref(self):
        """The character reference (or the first present), for single-ref paths."""
        if "character" in self.refs:
            return self.refs["character"]
        return next(iter(self.refs.values()), None)

    def without_ref(self):
        """The same text context with no reference latents attached."""
        return AnimaRefConditioning(self.text, None)


class RefAdapter:
    """The Anima reference adapter LoRA, applied through gated forward hooks.

    The weights are matched to the DiT's modules with the same flexible naming
    :mod:`engine.lora` uses, but kept in this class so the update can be toggled
    per forward pass: the LoRA only applies while :func:`activate` has made this
    adapter the active one (i.e. while a reference is attached).

    The safetensors metadata selects the inference path: ``{"arch": "asym"}`` is
    the v2 asymmetric adapter (see :meth:`ref_kv`), anything else is the pilot
    two-frame adapter. ``slots`` gives each reference frame's RoPE ``t`` index
    and ``(h, w)`` shift; ``ref_lora`` says whether the LoRA also applies to the
    reference stream itself (the v2 default).
    """

    def __init__(self, path, strength=1.0):
        from safetensors.torch import load_file

        self.path = path
        self.strength = float(strength)
        self.state = load_file(path, device="cpu")
        self.key = (os.path.abspath(path), float(strength))
        self._entries = None
        self.slots = DEFAULT_SLOTS
        self.asym = False
        self.ref_lora = True
        self.ref_cache = {}
        metadata = _read_metadata(path)
        if metadata:
            self.slots = metadata.get("slots") or self.slots
            self.asym = metadata.get("arch") == "asym"
            self.ref_lora = metadata.get("ref_lora", True)

    def ref_kv(self, dit, ref, x, offset_ratio=1.0):
        """Cached per-block reference keys/values for the asymmetric (v2) forward.

        ``ref`` is the clean reference latent (any size), ``x`` the noisy target
        latent whose token height fixes the RoPE shift. The reference stream is
        run once -- with the LoRA active unless the adapter says otherwise -- and
        kept as the single cache entry, so a new reference (or a different target
        size) replaces it rather than accumulating VRAM.
        """
        ref = ref.unsqueeze(2) if ref.ndim == 4 else ref
        hp = -(-x.shape[-2] // dit.patch_spatial)
        spec = self.slots.get("character") or next(iter(self.slots.values()))
        offset = (
            round(float(spec["offset_h"]) * offset_ratio * hp),
            int(spec.get("offset_w", 0)),
        )
        key = (
            id(dit), tuple(ref.shape), offset, ref.dtype, x.device,
            float(ref.float().sum()),
        )
        cached = self.ref_cache.get(key)
        if cached is None:
            self.ref_cache.clear()
            previous = activate(self if self.ref_lora else None)
            try:
                with torch.no_grad():
                    cached = dit.encode_ref_asym(
                        ref.to(device=x.device, dtype=ref.dtype), offset
                    )
            finally:
                restore(previous)
            self.ref_cache[key] = cached
        return cached

    def _resolve(self, dit):
        """``(module, down, up, scale)`` for every layer the LoRA targets."""
        pairs = _lora._collect(self.state)
        index = _lora._index(dit)
        entries = []
        for base, parts in pairs.items():
            if "down" not in parts or "up" not in parts:
                continue
            down = parts["down"].float()
            up = parts["up"].float()
            if down.ndim != 2 or up.ndim != 2:
                continue
            module = _lora._resolve(dit, base, index)
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            if up.shape[0] != weight.shape[0] or down.shape[-1] != weight.shape[-1]:
                continue
            rank = down.shape[0]
            scale = self.strength
            if "alpha" in parts and rank > 0:
                scale *= float(parts["alpha"].reshape(-1)[0]) / rank
            dtype = weight.dtype if weight.dtype in _STORE_DTYPES else torch.bfloat16
            entries.append((
                module,
                down.to(device=weight.device, dtype=dtype).contiguous(),
                up.to(device=weight.device, dtype=dtype).contiguous(),
                scale,
            ))
        return entries

    def install(self, dit):
        """Attach the LoRA to ``dit``'s modules (idempotent). Returns layers."""
        if self._entries is None:
            self._entries = self._resolve(dit)
        for module, down, up, scale in self._entries:
            installed = getattr(module, "_pai_ref_hooks", None)
            if installed is None:
                installed = module._pai_ref_hooks = {}
            if id(self) in installed:
                continue

            def hook(module, inputs, output, down=down, up=up, scale=scale,
                     key=self.key):
                if _ACTIVE["key"] != key:
                    return output
                x = inputs[0]
                if down.device != x.device or down.dtype != x.dtype:
                    down = down.to(device=x.device, dtype=x.dtype)
                    up = up.to(device=x.device, dtype=x.dtype)
                return output + F.linear(F.linear(x, down), up) * scale

            installed[id(self)] = module.register_forward_hook(hook)
        return len(self._entries)

    def layers(self):
        """How many DiT layers the LoRA patched (0 before :meth:`install`)."""
        return 0 if self._entries is None else len(self._entries)


def activate(adapter):
    """Make ``adapter``'s LoRA apply to the next forwards; returns the old key."""
    previous = _ACTIVE["key"]
    _ACTIVE["key"] = adapter.key if adapter is not None else None
    return previous


def restore(previous):
    """Restore the active key returned by :func:`activate`."""
    _ACTIVE["key"] = previous
