"""Introspection over the engine's in-memory model caches.

The sidebar's Loaded Models panel calls :func:`loaded` to show what the engine is still
holding after previous jobs, and :func:`unload` to drop either everything or one record's
model. Each engine module owns its own cache and exposes a
``loaded_models()`` returning records shaped as::

    {
        "family": str,      # human label, e.g. "Anima"
        "source": str,      # file name the model was loaded from
        "key": str,         # optional; accepted by unload() to drop just this model
        "components": [     # one entry per held module
            {"name", "params", "bytes", "dtype", "device"},
        ],
        "loras": [          # optional; LoRAs patched into the family's model
            {"source", "strength", "bytes", "layers"},
        ],
    }

This module holds the shared helpers and aggregates the sources. It never
imports a source itself: a cache can only be populated once its module has been
imported, so only the modules already in :data:`sys.modules` are inspected. That
keeps the panel from pulling torch or onnxruntime in just to draw an empty list.
"""

import gc
import os
import sys


# Modules that own a model cache and expose ``loaded_models()``/``unload()``.
_SOURCES = ("pipeline", "background", "upscale", "segmentation")


def summarize(component):
    """Return ``{params, bytes, dtype, device}`` for a loaded component.

    ``component`` is a ``torch.nn.Module`` or a wrapper exposing one as
    ``.model`` (the text encoders). Non-torch objects (and an install without
    torch) return ``None`` so the caller can skip them. ``dtype`` and ``device``
    are the ones holding the most bytes, so a model partly offloaded to CPU
    still reports where its weight mass sits.
    """
    try:
        import torch
    except ImportError:
        return None
    if isinstance(component, torch.nn.Module):
        module = component
    else:
        module = getattr(component, "model", None)
    if not isinstance(module, torch.nn.Module):
        return None

    params = 0
    nbytes = 0
    dtypes = {}
    devices = {}
    for tensor in module.parameters():
        params += tensor.numel()
        size = tensor.numel() * tensor.element_size()
        nbytes += size
        _bump(dtypes, str(tensor.dtype).replace("torch.", ""), size)
        _bump(devices, str(tensor.device), size)
    for tensor in module.buffers():
        size = tensor.numel() * tensor.element_size()
        nbytes += size
        _bump(dtypes, str(tensor.dtype).replace("torch.", ""), size)
        _bump(devices, str(tensor.device), size)
    return {
        "params": params,
        "bytes": nbytes,
        "dtype": _dominant(dtypes),
        "device": _dominant(devices),
    }


def record(family, source, components, key=None, loras=None):
    """Build one ``loaded_models`` record from ``(name, component)`` pairs.

    Entries whose component cannot be summarised are dropped; ``None`` is
    returned when nothing remains so the caller can skip the family. ``key``
    names the source module's cache slot, so the panel can unload just this
    model through :func:`unload`. ``loras`` is an optional list of
    ``{"source", "strength", "bytes", "layers"}`` rows (see
    :func:`engine.lora.usage`) for the LoRAs patched into the family's model.
    """
    rows = []
    for name, component in components:
        info = summarize(component)
        if info is None:
            continue
        info["name"] = name
        rows.append(info)
    if not rows:
        return None
    row = {
        "family": family,
        "source": os.path.basename(source) if source else "",
        "components": rows,
    }
    if key is not None:
        row["key"] = key
    if loras:
        row["loras"] = list(loras)
    return row


def file_size(path):
    """Byte size of ``path`` on disk, or 0 when it cannot be read."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _bump(totals, key, size):
    totals[key] = totals.get(key, 0) + size


def _dominant(totals):
    if not totals:
        return ""
    return max(totals, key=totals.get)


def loaded():
    """Records for every engine component currently cached in memory.

    Each record carries a ``key`` that :func:`unload` accepts to drop just that
    model. Keys are namespaced with the source module (e.g. ``pipeline:base``)
    so they stay unique across modules.
    """
    records = []
    for name in _SOURCES:
        module = sys.modules.get(f"{__package__}.{name}")
        if module is None:
            continue
        getter = getattr(module, "loaded_models", None)
        if getter is None:
            continue
        try:
            items = getter()
        except Exception:
            continue
        for item in items:
            local = item.get("key")
            if local is not None:
                item["key"] = f"{name}:{local}"
            records.append(item)
    return records


def unload(key=None):
    """Drop engine model caches and release cached GPU memory.

    With no ``key`` every cache is cleared, matching :func:`loaded`. With a
    ``key`` from a ``loaded()`` record, only the owning module is asked to drop
    that one model, leaving the rest loaded. Returns the number of cached
    models/sessions dropped.

    The caller must ensure no generation, background or upscale job is running:
    those jobs use the cached objects directly.
    """
    if key:
        name, _, local = key.partition(":")
        names = (name,) if name in _SOURCES else ()
    else:
        local = None
        names = _SOURCES
    dropped = 0
    for name in names:
        module = sys.modules.get(f"{__package__}.{name}")
        if module is None:
            continue
        clear = getattr(module, "unload", None)
        if clear is None:
            continue
        try:
            dropped += clear(local) or 0
        except Exception:
            continue
    # Break reference cycles left by the cache dicts and free the allocator's
    # cached VRAM so the next model can use it.
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return dropped
