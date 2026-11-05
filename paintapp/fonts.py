"""System font discovery and caching for the text tool.

Pillow's ``ImageFont.truetype`` needs a font *file*, not a family name, so this
module scans the platform's font directories once, asks FreeType for each file's
real family and style, and groups the files into a small family registry:

    families()               -> sorted family names for a chooser
    resolve(family, ...)     -> the file backing a family/style, or ``None``
    load(family, size, ...)  -> a cached ``ImageFont`` instance

When no system font is found, a single ``"Default"`` family backed by Pillow's
built-in font is offered so the text tool still works.
"""

import os
import sys
import threading

from PIL import ImageFont

# TrueType/OpenType faces, including TrueType collections (``.ttc``), which hold
# several faces in one file.
_EXTENSIONS = (".ttf", ".otf", ".ttc")

# Buckets within a family. The order is the preference used when a requested
# style is missing (bold -> regular, etc.).
_BUCKETS = ("regular", "bold", "italic", "bolditalic")

# Styles that count as the "plain" face of a family.
_PLAIN_STYLES = ("regular", "normal", "book", "roman", "medium")

_registry = None
_registry_lock = threading.Lock()
_prewarm_started = False
_font_cache = {}


def _candidate_dirs():
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA", "")
        return [
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
            os.path.join(local, "Microsoft", "Windows", "Fonts"),
        ]
    if sys.platform == "darwin":
        return [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.join(home, "Library", "Fonts"),
        ]
    return [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.join(home, ".fonts"),
        os.path.join(home, ".local", "share", "fonts"),
    ]


def _registry_font_files():
    """Absolute font paths recorded in the Windows registry.

    The directory walk finds most faces; the registry also lists fonts installed
    outside those directories. Non-Windows platforms (and any registry error)
    simply yield nothing.
    """
    if not sys.platform.startswith("win"):
        return []
    try:
        import winreg
    except ImportError:
        return []
    paths = []
    key_names = (
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows NT\CurrentVersion\Fonts",
    )
    for key_name in key_names:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_name) as key:
                count = winreg.QueryInfoKey(key)[1]
                for index in range(count):
                    try:
                        _name, value, _kind = winreg.EnumValue(key, index)
                    except OSError:
                        continue
                    if isinstance(value, str) and os.path.isabs(value) \
                            and value.lower().endswith(_EXTENSIONS):
                        paths.append(value)
        except OSError:
            continue
    return paths


def _iter_font_files():
    seen = set()
    for path in _registry_font_files():
        key = os.path.normcase(path)
        if key not in seen and os.path.isfile(path):
            seen.add(key)
            yield path
    for directory in _candidate_dirs():
        if not directory or not os.path.isdir(directory):
            continue
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if not name.lower().endswith(_EXTENSIONS):
                    continue
                path = os.path.join(root, name)
                key = os.path.normcase(path)
                if key in seen:
                    continue
                seen.add(key)
                yield path


def _faces(path, indexed):
    """Yield ``(family, style, index)`` for each face in ``path``.

    Only TrueType collections are probed past face 0, since a plain ``.ttf``
    holds a single face and a second open would just fail.
    """
    limit = 64 if indexed else 1
    for index in range(limit):
        try:
            font = ImageFont.truetype(path, 12, index=index)
        except Exception:
            break
        try:
            family, style = font.getname()
        except Exception:
            family, style = None, ""
        if family:
            yield family, style or "", index


def _bucket(style):
    style = style.lower()
    bold = any(word in style for word in (
        "bold", "black", "heavy", "semibold", "demibold",
    ))
    italic = "italic" in style or "oblique" in style
    if bold and italic:
        return "bolditalic"
    if bold:
        return "bold"
    if italic:
        return "italic"
    return "regular"


def _build_registry():
    families = {}
    for path in _iter_font_files():
        indexed = path.lower().endswith(".ttc")
        for family, style, index in _faces(path, indexed):
            entry = families.setdefault(family, {})
            bucket = _bucket(style)
            style_l = style.lower()
            # Prefer the most "canonical" file for a bucket so e.g. "Arial Black"
            # never displaces "Arial Bold".
            rank = 0 if bucket != "regular" or style_l in _PLAIN_STYLES else 1
            existing = entry.get(bucket)
            if existing is None or rank < existing[0]:
                entry[bucket] = (rank, path, index)
    return families


def _ensure_registry():
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = _build_registry()
    return _registry


def prewarm():
    """Build the font registry on a background thread.

    The first :func:`families` call scans every platform font directory, which
    can visibly stall the UI when the picker first opens. Warming the registry at
    startup moves that cost off the first interaction. Repeated calls are cheap:
    only the first starts a thread.
    """
    global _prewarm_started
    with _registry_lock:
        if _prewarm_started or _registry is not None:
            return None
        _prewarm_started = True
    thread = threading.Thread(target=_ensure_registry, name="pai-fonts", daemon=True)
    thread.start()
    return thread


def families():
    """Sorted list of available font family names."""
    registry = _ensure_registry()
    names = sorted(registry, key=str.lower)
    return names or ["Default"]


def default_family():
    """A sensible initial family, falling back to the first available."""
    available = families()
    for preferred in ("Arial", "DejaVu Sans", "Helvetica", "Verdana", "Calibri"):
        if preferred in available:
            return preferred
    return available[0]


# Name hints for a coarse serif/sans/mono grouping (there is no portable way to
# read a font's classification from FreeType, so this is a best-effort match).
_MONO_HINTS = (
    "mono", "consolas", "courier", "menlo", "inconsolata", "terminal",
    "fixed", "cascadia", "hack", "lucida console", "source code", "code",
)
_SERIF_HINTS = (
    "serif", "times", "georgia", "garamond", "cambria", "antiqua", "palatino",
    "baskerville", "didot", "bodoni", "rockwell", "constantia", "minion",
    "roman", "slab", "book",
)


def category(name):
    """A coarse grouping for a family name: ``"Mono"``, ``"Serif"`` or ``"Sans"``."""
    lowered = str(name).lower()
    if any(hint in lowered for hint in _MONO_HINTS):
        return "Mono"
    if any(hint in lowered for hint in _SERIF_HINTS):
        return "Serif"
    return "Sans"


def resolve(family, bold=False, italic=False):
    """Return ``(path, index)`` for ``family`` and style, or ``None``.

    ``index`` selects a face inside a TrueType collection. Missing styles
    degrade gracefully: bold falls back to regular, italic to regular, and so on,
    so any family always yields *some* usable face.
    """
    registry = _ensure_registry()
    entry = registry.get(family)
    if not entry:
        for name, candidate in registry.items():
            if name.lower() == str(family).lower():
                entry = candidate
                break
    if not entry:
        return None
    wanted = []
    if bold and italic:
        wanted = ["bolditalic", "bold", "italic", "regular"]
    elif bold:
        wanted = ["bold", "regular", "bolditalic", "italic"]
    elif italic:
        wanted = ["italic", "regular", "bolditalic", "bold"]
    else:
        wanted = ["regular", "bold", "italic", "bolditalic"]
    for bucket in wanted:
        if bucket in entry:
            return entry[bucket][1], entry[bucket][2]
    for bucket in _BUCKETS:
        if bucket in entry:
            return entry[bucket][1], entry[bucket][2]
    return None


def load(family, size, bold=False, italic=False):
    """Return a cached ``ImageFont`` for the family/style/size."""
    size = max(1, int(round(size)))
    resolved = resolve(family, bold=bold, italic=italic)
    key = (resolved, size)
    cached = _font_cache.get(key)
    if cached is not None:
        return cached
    font = None
    if resolved is not None:
        path, index = resolved
        try:
            font = ImageFont.truetype(path, size, index=index)
        except Exception:
            font = None
    if font is None:
        font = _load_default(size)
    _font_cache[key] = font
    return font


def _load_default(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
