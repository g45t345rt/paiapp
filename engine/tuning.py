"""Generation-engine tuning (the ``generation_engine`` section of settings.json).

This is the single place the engine and the UI agree on how a generation should
be *placed and run* on the hardware, as opposed to *what* to render (the
per-model sampler recipes in :mod:`engine.paths`) or *which* files to use
(``engine.settings.section("models")``).

Everything here is optional: :data:`DEFAULTS` reproduces the behaviour the
engine had before these controls existed (fully automatic precision and
placement, the original VRAM margins, compiled VAE decode, encoders on CPU,
models kept loaded, tiled decode off, weights loaded into RAM). A user who never
opens the Generation tab of the Settings dialog gets exactly the old behaviour.

The module stays torch-free so it can be imported by UI code (and by
:mod:`engine.loaded_models`) without pulling in the ML stack. The values are
pushed into :mod:`engine.precision` by :func:`apply`, which is called when the
dialog saves and once per generation (see :func:`engine.pipeline.generate`).
"""

from . import settings

#: The settings.json section this module owns.
SECTION = "generation_engine"

#: Tri-state precision / placement policies.
PRECISION_CHOICES = (
    ("Auto (fit, else fp8, else stream)", "auto"),
    ("Always fp8 when it helps", "fp8"),
    ("Always full precision", "full"),
)

#: Where the text encoders run.
TEXT_ENCODER_CHOICES = (
    ("CPU (lowest VRAM use)", "cpu"),
    ("GPU when it fits", "auto"),
)

#: Every knob with its default. ``ram_budget_gb == 0`` and ``prefetch_depth == 0``
#: mean "decide automatically".
DEFAULTS = {
    "precision": "auto",
    "vram_reserve_gb": 2.5,
    "stream_reserve_gb": 4.0,
    "keep_resident": True,
    "compile_vae": True,
    "sage_attention": False,
    "text_encoder_device": "cpu",
    "auto_unload": True,
    "auto_unload_minutes": 10,
    "tiled_decode": False,
    "tile_size": 512,
    "tile_overlap": 64,
    "disk_stream": False,
    "ram_budget_gb": 0.0,
    "prefetch_depth": 0,
}

# Bounds applied by :func:`normalize` (min, max). A value outside is clamped.
_BOUNDS = {
    "vram_reserve_gb": (0.0, 32.0),
    "stream_reserve_gb": (0.0, 48.0),
    "auto_unload_minutes": (1, 240),
    "tile_size": (128, 2048),
    "tile_overlap": (0, 512),
    "ram_budget_gb": (0.0, 512.0),
    "prefetch_depth": (0, 6),
}

_FLOAT_FIELDS = ("vram_reserve_gb", "stream_reserve_gb", "ram_budget_gb")
_INT_FIELDS = ("auto_unload_minutes", "tile_size", "tile_overlap", "prefetch_depth")
_BOOL_FIELDS = ("keep_resident", "compile_vae", "auto_unload", "tiled_decode",
                "disk_stream", "sage_attention")


def _coerce(field, value):
    """Coerce one stored value to its default's type, or the default on junk."""
    default = DEFAULTS[field]
    try:
        if field in _BOOL_FIELDS:
            if isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if field in _INT_FIELDS:
            return int(value)
        if field in _FLOAT_FIELDS:
            return float(value)
    except (TypeError, ValueError):
        return default
    return value


def normalize(values):
    """Return ``values`` merged over :data:`DEFAULTS`, coerced and clamped.

    Unknown keys are dropped and malformed values fall back to their default, so
    a corrupt or hand-edited settings file never reaches the engine.
    """
    result = dict(DEFAULTS)
    if not isinstance(values, dict):
        return result
    for field, default in DEFAULTS.items():
        if field not in values:
            continue
        got = _coerce(field, values[field])
        if field == "precision" or field == "text_encoder_device":
            if got not in tuple(choice[1] for choice in _CHOICES_BY_FIELD[field]):
                got = default
        low, high = _BOUNDS.get(field, (None, None))
        if low is not None and isinstance(got, (int, float)) and not isinstance(got, bool):
            got = max(low, min(high, got))
        result[field] = got
    return result


_CHOICES_BY_FIELD = {
    "precision": PRECISION_CHOICES,
    "text_encoder_device": TEXT_ENCODER_CHOICES,
}


def read():
    """The normalized engine-tuning values (defaults when nothing is stored)."""
    return normalize(settings.section(SECTION))


def write(values):
    """Persist ``values`` (normalized) to the ``generation_engine`` section."""
    normalized = normalize(values)
    if normalized == DEFAULTS:
        settings.set_section(SECTION, None)
    else:
        settings.set_section(SECTION, normalized)
    return normalized


def reset():
    """Drop the section, restoring every default."""
    settings.set_section(SECTION, None)
    return dict(DEFAULTS)


def apply(values=None):
    """Push the tuning into :mod:`engine.precision` and return what was applied.

    Imported lazily so this module (and the Settings dialog) never needs torch.
    """
    normalized = read() if values is None else normalize(values)
    from . import attention, precision

    precision.configure(
        fp8_mode=normalized["precision"],
        vram_reserve=int(normalized["vram_reserve_gb"] * 1e9),
        stream_reserve=int(normalized["stream_reserve_gb"] * 1e9),
        keep_resident=normalized["keep_resident"],
        compile_vae=normalized["compile_vae"],
        tiled_decode=normalized["tiled_decode"],
        tile_size=normalized["tile_size"],
        tile_overlap=normalized["tile_overlap"],
        text_encoder_device=normalized["text_encoder_device"],
        disk_stream=normalized["disk_stream"],
        ram_budget=int(normalized["ram_budget_gb"] * 1e9),
        prefetch_depth=normalized["prefetch_depth"],
    )
    attention.configure(sage_attention=normalized["sage_attention"])
    return normalized


def auto_unload_seconds():
    """Idle seconds before models are dropped, or ``0`` when disabled."""
    values = read()
    if not values["auto_unload"]:
        return 0
    return int(values["auto_unload_minutes"]) * 60


def human_size(count):
    """``"1.4 GB"``-style label for a byte count (used by the dialog previews)."""
    try:
        count = float(count)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if count < 1024 or unit == "TB":
            return f"{count:.0f} {unit}" if unit in ("B", "KB") else f"{count:.2f} {unit}"
        count /= 1024
    return f"{count:.2f} TB"