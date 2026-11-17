"""Small formatters for canvas dimensions shared by the UI dialogs."""

import math


def _compact_ratio(value):
    """Format a ratio side without noisy trailing zeros."""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def aspect_ratio_label(width, height):
    """A compact width:height label, e.g. ``16:9`` or ``1:1.41``.

    Integer sizes use an exact reduced ratio when it stays readable (both
    sides at most 32); other sizes fall back to a compact decimal ratio, so
    millimetre sizes such as ``215.9 x 279.4`` still read as ``1:1.29``.
    """
    try:
        width_value = float(width)
        height_value = float(height)
    except (TypeError, ValueError):
        return "—"
    if width_value < 1 or height_value < 1:
        return "—"
    if width_value.is_integer() and height_value.is_integer():
        width_int = int(width_value)
        height_int = int(height_value)
        divisor = math.gcd(width_int, height_int)
        left, right = width_int // divisor, height_int // divisor
        if left <= 32 and right <= 32:
            return f"{left}:{right}"
    if width_value >= height_value:
        return f"{_compact_ratio(width_value / height_value)}:1"
    return f"1:{_compact_ratio(height_value / width_value)}"


def megapixels_label(width, height):
    """A compact megapixel label for a pixel size."""
    try:
        value = int(width) * int(height) / 1_000_000
    except (TypeError, ValueError):
        return "—"
    return f"{value:.1f} MP"
