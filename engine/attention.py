"""Attention dispatch for the DiTs: SageAttention when enabled, else PyTorch SDPA.

``sageattention``'s ``sageattn`` is a drop-in replacement for
``torch.nn.functional.scaled_dot_product_attention`` that quantizes QK to int8
(SageAttention2) and is noticeably faster on Ada/Hopper GPUs. It is optional and
has hard constraints -- the package is CUDA-only, takes only bf16/fp16 with a 64
or 128 head dimension, and accepts no attention mask or GQA -- so every call it
cannot serve falls back to SDPA unchanged.

The mode defaults to *off* and is owned here (like :mod:`engine.precision` owns
the placement policy); :func:`engine.tuning.apply` writes it from the Settings
Generation tab. An untouched install therefore keeps the exact previous
behaviour, and a machine without the package (CPU, ROCm, older CUDA) silently
stays on SDPA.
"""

import os
import warnings

import torch
import torch.nn.functional as F

#: Head dimensions the SageAttention kernels accept (SageAttention2: 64 or 128).
_HEAD_DIMS = (64, 128)

#: Minimum release: SageAttention2 with the stable ``sageattn`` /
#: ``tensor_layout`` API.
_MIN_VERSION = "2.0.0"

#: Live policy, overwritten by :func:`configure` (from :mod:`engine.tuning`).
_CONFIG = {"sage_attention": False}

# Import result cache: ``_FN`` is the ``sageattn`` callable or ``None``, resolved
# once per process. ``_FAILED`` latches after a kernel rejection so a shape the
# runtime refuses is not retried every step.
_FN = None
_CHECKED = False
_FAILED = False


def configure(**values):
    """Update the dispatch policy from :mod:`engine.tuning`.

    Only recognised keys are written, so a caller cannot introduce a typo key.
    Enabling the mode clears a previous failure latch so a re-enable retries the
    kernel rather than staying silently on SDPA.
    """
    global _FAILED
    for key, value in values.items():
        if key in _CONFIG:
            _CONFIG[key] = value
    if values.get("sage_attention"):
        _FAILED = False
    return dict(_CONFIG)


def config():
    """A copy of the live policy (for status text and the Settings dialog)."""
    return dict(_CONFIG)


def enabled():
    """Whether SageAttention is requested *and* actually usable here."""
    return bool(_CONFIG.get("sage_attention")) and _load() is not None


def _load():
    """Import ``sageattn`` once; return it, or ``None`` when unavailable.

    Returns ``None`` -- and never raises -- when the package is missing, too old,
    or there is no CUDA device, so callers can treat "no kernel" and "not
    installed" the same way. ``PAI_SAGE_ATTENTION=0`` forces the SDPA path even
    when the package is installed.
    """
    global _FN, _CHECKED
    if _CHECKED:
        return _FN
    _CHECKED = True
    if os.environ.get("PAI_SAGE_ATTENTION", "1") == "0":
        return _FN
    try:
        if not torch.cuda.is_available():
            return _FN
        import sageattention
        try:
            from packaging.version import parse

            if parse(sageattention.__version__) < parse(_MIN_VERSION):
                return _FN
        except (AttributeError, ImportError, ValueError):
            # No version metadata: accept it rather than refusing a working build.
            pass
        from sageattention import sageattn

        _FN = sageattn
    except Exception:
        _FN = None
    return _FN


def _supported(q, k, v):
    """Whether this call's device/dtype/shape fits the SageAttention kernels."""
    return (
        q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.dtype == k.dtype == v.dtype
        and q.shape[-1] in _HEAD_DIMS
        and k.shape[-2] == v.shape[-2]
    )


def attention(q, k, v, attn_mask=None, *, is_causal=False, enable_gqa=False,
              scale=None):
    """Multi-head attention: SageAttention when usable, else PyTorch SDPA.

    Mirrors ``torch.nn.functional.scaled_dot_product_attention`` for the
    arguments this engine uses. ``q``/``k``/``v`` are ``(B, heads, L, D)`` (the
    ``HND`` layout). SageAttention is used only when it is enabled, installed and
    usable, and the call has no mask, no causality and no GQA and a supported
    dtype/head dimension; every other call -- and any runtime kernel rejection --
    uses SDPA unchanged, so switching the mode can never break a run.
    """
    global _FAILED
    no_causal = is_causal is False or is_causal is None
    if (
        not _FAILED
        and attn_mask is None
        and not enable_gqa
        and no_causal
        and bool(_CONFIG.get("sage_attention"))
    ):
        fn = _load()
        if fn is not None and _supported(q, k, v):
            try:
                return fn(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    tensor_layout="HND", sm_scale=scale,
                )
            except Exception as error:  # noqa: BLE001 - fall back, never fail a run
                _FAILED = True
                warnings.warn(
                    f"SageAttention rejected a call ({error}); falling back to "
                    "PyTorch SDPA for the rest of this session.",
                    RuntimeWarning,
                )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=enable_gqa,
        scale=scale,
    )
