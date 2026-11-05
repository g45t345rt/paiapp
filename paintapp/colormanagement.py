"""ICC colour management.

The document pipeline is 8-bit RGB and space-agnostic: layers, brushes,
filters and the compositor all just move RGB numbers around. Colour management
is therefore applied at the *boundaries* rather than inside the pipeline:

* **Import** - an opened image's embedded profile is converted into the canvas
  working space (see :func:`convert_from_embedded`).
* **Display** - the composited preview is converted from the working space to
  the monitor space (assumed sRGB), optionally soft-proofed against a print
  profile (see :func:`to_display`).
* **Export** - the flattened image is converted to the chosen output profile
  and that profile is embedded in the file (see :func:`convert` and
  :func:`icc_bytes`).

Profiles are identified by a string: the sentinel ``"sRGB"`` for the built-in
working space, an ``"icc:<sha1>"`` id for a profile embedded in an opened image
(registered in memory by :func:`source_profile_id`), or an absolute path to a
``.icc``/``.icm`` file (the built-in sRGB when the id is empty or unknown).

``Pillow`` bundles the LittleCMS bindings as :mod:`PIL.ImageCms`, so no extra
dependency is needed. Transforms are expensive to build, so they are cached;
the arrays they produce are the only per-call cost. When either endpoint is
unknown the conversion degrades to a no-op, so a missing or corrupt profile
never blocks opening or exporting.
"""

import hashlib
import io
import os

from PIL import ImageCms

from engine import settings as engine_settings

# Sentinel id for the built-in sRGB working/monitor space.
SRGB = "sRGB"

# Extensions scanned for user profiles.
_PROFILE_EXT = (".icc", ".icm")

# Rendering intents, in display order: ``(name, label, value)``. ``value`` is
# the LittleCMS intent number (0-3) :func:`ImageCms.buildTransform` expects.
INTENTS = (
    ("perceptual", "Perceptual", 0),
    ("relative", "Relative Colorimetric", 1),
    ("saturation", "Saturation", 2),
    ("absolute", "Absolute Colorimetric", 3),
)
_INTENT_VALUES = {name: value for name, _label, value in INTENTS}
_INTENT_LABELS = {name: label for name, label, _value in INTENTS}

DEFAULT_INTENT = "perceptual"
# Soft proofing simulates a print device, where relative colorimetric is the
# usual choice (absolute is what LittleCMS themes; relative reads better).
DEFAULT_PROOF_INTENT = "relative"

# Caches: profiles by id, embedded profiles by hash, transforms by their key,
# ids that failed to load (so a bad file is not retried every call), and the
# built-in sRGB bytes.
_profiles = {}
_embedded = {}
_transforms = {}
_failed = set()
_srgb_bytes = None


# ------------------------------------------------------------------- helpers
def intent_value(name):
    """The LittleCMS intent number for an intent ``name`` (default perceptual)."""
    return _INTENT_VALUES.get(name, _INTENT_VALUES[DEFAULT_INTENT])


def intent_label(name):
    """The display label for an intent ``name``."""
    return _INTENT_LABELS.get(name, _INTENT_LABELS[DEFAULT_INTENT])


def intent_names():
    """Intent names in display order."""
    return [name for name, _label, _value in INTENTS]


def is_srgb(identifier):
    """True when ``identifier`` is empty or the built-in sRGB sentinel."""
    return not identifier or identifier == SRGB


def profiles_dir():
    """The per-user folder scanned for ``.icc``/``.icm`` profiles."""
    return os.path.join(engine_settings.app_dir(), "profiles")


def profile_label(identifier):
    """A short human label for a profile id (file name, or ``"sRGB"``)."""
    if is_srgb(identifier):
        return "sRGB"
    if identifier.startswith("icc:"):
        return "Embedded profile"
    return os.path.splitext(os.path.basename(identifier))[0]


def list_profiles():
    """All selectable profiles as ``(identifier, label)``, sRGB first."""
    items = [(SRGB, "sRGB")]
    directory = profiles_dir()
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        names = []
    for name in names:
        if os.path.splitext(name)[1].lower() in _PROFILE_EXT:
            path = os.path.join(directory, name)
            items.append((path, profile_label(path)))
    return items


def profile_color_space(identifier):
    """The ICC header colour space (``"rgb"``/``"cmyk"``/``"gray"``/``"lab"``)."""
    data = icc_bytes(identifier)
    if not data or len(data) < 20:
        return None
    try:
        return bytes(data[16:20]).decode("ascii", "replace").strip().lower()
    except Exception:
        return None


def list_cmyk_profiles():
    """The selectable profiles whose colour space is CMYK."""
    return [
        (ident, label)
        for ident, label in list_profiles()
        if profile_color_space(ident) == "cmyk"
    ]


def import_profile(source_path):
    """Copy an ICC file into the profiles folder and return its identifier.

    A name collision keeps both files (a numeric suffix is added) so importing
    never overwrites an existing profile. Returns ``None`` when the file cannot
    be read or is not a usable profile.
    """
    if not source_path or not os.path.isfile(source_path):
        return None
    if _load_path(source_path) is None:
        return None
    directory = profiles_dir()
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return None
    name = os.path.basename(source_path)
    dest = os.path.join(directory, name)
    base, ext = os.path.splitext(dest)
    index = 2
    while os.path.exists(dest) and os.path.abspath(dest) != os.path.abspath(source_path):
        dest = f"{base}_{index}{ext}"
        index += 1
    if os.path.abspath(dest) != os.path.abspath(source_path):
        try:
            with open(source_path, "rb") as src, open(dest, "wb") as out:
                out.write(src.read())
        except OSError:
            return None
    refresh()
    return dest


def refresh():
    """Drop cached transforms and profiles (call after the folder changes)."""
    _profiles.clear()
    _transforms.clear()
    _failed.clear()


# -------------------------------------------------------------- profile load
def _load_path(path):
    """Load (and cache) an ``ImageCmsProfile`` from a file path, or ``None``."""
    if path in _profiles:
        return _profiles[path]
    if path in _failed:
        return None
    try:
        profile = ImageCms.getOpenProfile(path)
    except Exception:
        _failed.add(path)
        return None
    _profiles[path] = profile
    return profile


def _load(identifier):
    """The ``ImageCmsProfile`` for a profile id, or ``None`` when unusable."""
    if is_srgb(identifier):
        if SRGB not in _profiles:
            try:
                _profiles[SRGB] = ImageCms.ImageCmsProfile(
                    ImageCms.createProfile("sRGB")
                )
            except Exception:
                _failed.add(SRGB)
                return None
        return _profiles.get(SRGB)
    if identifier.startswith("icc:"):
        return _embedded.get(identifier)
    return _load_path(identifier)


def icc_bytes(identifier):
    """The ICC profile bytes for an id, ready for ``Image.save(icc_profile=...)``."""
    global _srgb_bytes
    if is_srgb(identifier):
        if _srgb_bytes is None:
            profile = _load(SRGB)
            _srgb_bytes = profile.tobytes() if profile is not None else None
        return _srgb_bytes
    profile = _load(identifier)
    if profile is None:
        return None
    try:
        return profile.tobytes()
    except Exception:
        return None


def source_profile_id(image):
    """The id of ``image``'s embedded ICC profile, or ``None`` when it has none.

    A profile whose bytes match the built-in sRGB is reported as :data:`SRGB`
    so importing an sRGB-tagged image stays a no-op. Other profiles are
    registered in memory under an ``"icc:<sha1>"`` id.
    """
    data = (getattr(image, "info", {}) or {}).get("icc_profile")
    if not data:
        return None
    srgb = icc_bytes(SRGB)
    if srgb and bytes(data) == srgb:
        return SRGB
    key = "icc:" + hashlib.sha1(bytes(data)).hexdigest()
    if key not in _embedded and key not in _failed:
        try:
            _embedded[key] = ImageCms.ImageCmsProfile(io.BytesIO(bytes(data)))
        except Exception:
            _failed.add(key)
            return None
    return key if key in _embedded else None


# ---------------------------------------------------------------- transforms
def _transform(src_id, dst_id, mode, intent, proof_id=None, proof_intent=None):
    """A cached transform from ``src_id`` to ``dst_id`` for ``mode``, or ``None``.

    With ``proof_id`` set, builds a soft-proof transform that simulates the
    proof device while still outputting to ``dst_id`` (the monitor).
    """
    key = (
        src_id or SRGB, dst_id or SRGB, mode, intent_value(intent),
        proof_id if (proof_id and not is_srgb(proof_id)) else None,
        intent_value(proof_intent or DEFAULT_PROOF_INTENT) if proof_id else None,
    )
    cached = _transforms.get(key)
    if cached is not None:
        return cached
    src = _load(src_id) or _load(SRGB)
    dst = _load(dst_id) or _load(SRGB)
    if src is None or dst is None:
        return None
    intent_num = intent_value(intent)
    try:
        if key[4]:
            proof = _load(proof_id)
            if proof is not None:
                transform = ImageCms.buildProofTransform(
                    src, dst, proof, mode, mode,
                    renderingIntent=intent_num,
                    proofRenderingIntent=intent_value(proof_intent or DEFAULT_PROOF_INTENT),
                    flags=ImageCms.Flags.SOFTPROOFING,
                )
            else:
                transform = ImageCms.buildTransform(src, dst, mode, mode, intent_num)
        else:
            transform = ImageCms.buildTransform(src, dst, mode, mode, intent_num)
    except Exception:
        return None
    _transforms[key] = transform
    return transform


def _apply(image, transform):
    """Apply ``transform`` to ``image``, preserving a separate alpha channel."""
    try:
        return ImageCms.applyTransform(image, transform)
    except Exception:
        pass
    alpha = None
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
    rgb = image.convert("RGB")
    try:
        out = ImageCms.applyTransform(rgb, transform)
    except Exception:
        return image
    if alpha is not None:
        if out.mode != "RGBA":
            out = out.convert("RGBA")
        out.putalpha(alpha)
    return out


def _convert_mode(image):
    return "RGBA" if image.mode == "RGBA" else "RGB"


def convert(image, src_id, dst_id, intent=DEFAULT_INTENT):
    """Convert ``image`` from ``src_id`` to ``dst_id``. Returns it unchanged
    when the two match or a needed profile is unavailable."""
    if is_srgb(src_id) and is_srgb(dst_id):
        return image
    transform = _transform(src_id, dst_id, _convert_mode(image), intent)
    if transform is None:
        return image
    return _apply(image, transform)


def convert_from_embedded(image, dst_id, intent=DEFAULT_INTENT):
    """Convert ``image`` from its embedded profile into ``dst_id``.

    Untagged images are returned as-is (their pixels are assumed to already be
    in the working space).
    """
    src_id = source_profile_id(image)
    if src_id is None:
        return image
    return convert(image, src_id, dst_id, intent)


def to_display(image, working_id, display_id=SRGB, proof_id=None,
               intent=DEFAULT_INTENT, proof_intent=DEFAULT_PROOF_INTENT):
    """The working-space composite converted for the monitor, soft-proofed.

    Identity (and cheap) when the working and display spaces are both sRGB and
    no proof profile is set, which is the default configuration.
    """
    proof_on = bool(proof_id) and not is_srgb(proof_id)
    if is_srgb(working_id) and is_srgb(display_id) and not proof_on:
        return image
    transform = _transform(
        working_id, display_id, _convert_mode(image), intent,
        proof_id=proof_id if proof_on else None, proof_intent=proof_intent,
    )
    if transform is None:
        return convert(image, working_id, display_id, intent)
    return _apply(image, transform)


def to_cmyk(image, working_id, cmyk_id, intent=DEFAULT_INTENT):
    """Convert ``image`` to CMYK via an ICC transform, or ``None`` on failure.

    ``None`` signals the caller to fall back to a naive ``convert("CMYK")`` so
    a missing or non-CMYK profile still produces output.
    """
    if not cmyk_id:
        return None
    src = _load(working_id) or _load(SRGB)
    dst = _load(cmyk_id)
    if src is None or dst is None:
        return None
    key = ("cmyk", working_id or SRGB, cmyk_id, intent_value(intent))
    transform = _transforms.get(key)
    if transform is None:
        try:
            transform = ImageCms.buildTransform(
                src, dst, "RGB", "CMYK", intent_value(intent)
            )
        except Exception:
            return None
        _transforms[key] = transform
    try:
        return ImageCms.applyTransform(image.convert("RGB"), transform)
    except Exception:
        return None
