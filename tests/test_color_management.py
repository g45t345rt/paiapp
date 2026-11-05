"""Unit tests for ICC colour management.

Run: .venv\\Scripts\\python.exe tests\\test_color_management.py
"""

import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image, ImageCms  # noqa: E402

from paintapp import colormanagement as cm  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def near(name, got, want, tol=1):
    if got is None or abs(got - want) > tol:
        FAILED.append(f"{name}: got {got!r} want {want!r} (+/-{tol})")
    else:
        print(f"ok  {name}")


def register_srgb_embedded(ident="icc:test"):
    """Register the built-in sRGB under a non-sentinel embedded id."""
    profile = ImageCms.ImageCmsProfile(io.BytesIO(cm.icc_bytes(cm.SRGB)))
    cm._embedded[ident] = profile
    return ident


# --------------------------------------------------------------------- intents
def test_intents():
    check("perceptual value", cm.intent_value("perceptual"), 0)
    check("absolute value", cm.intent_value("absolute"), 3)
    check("unknown intent falls back", cm.intent_value("nope"), 0)
    check("intent label", cm.intent_label("relative"), "Relative Colorimetric")
    check("intent names order", cm.intent_names(),
          ["perceptual", "relative", "saturation", "absolute"])
    check("intent names are valid", all(
        cm.intent_value(n) is not None for n in cm.intent_names()), True)


def test_is_srgb():
    check("empty is srgb", cm.is_srgb(""), True)
    check("sentinel is srgb", cm.is_srgb("sRGB"), True)
    check("path is not srgb", cm.is_srgb("/x/y.icc"), False)


def test_srgb_profile():
    data = cm.icc_bytes(cm.SRGB)
    check("srgb bytes present", bool(data), True)
    check("srgb colour space", cm.profile_color_space(cm.SRGB), "rgb")
    check("srgb label", cm.profile_label(cm.SRGB), "sRGB")


# ------------------------------------------------------------------- conversion
def test_convert_identity():
    image = Image.new("RGB", (3, 3), (10, 20, 30))
    check("identity returns same object",
          cm.convert(image, "sRGB", "sRGB") is image, True)


def test_source_profile_untagged():
    check("untagged has no profile",
          cm.source_profile_id(Image.new("RGB", (2, 2))), None)
    check("empty image info has no profile",
          cm.source_profile_id(Image.new("RGBA", (2, 2), (0, 0, 0, 0))), None)


def test_source_profile_srgb_embedded():
    image = Image.new("RGB", (2, 2), (1, 2, 3))
    image.info["icc_profile"] = cm.icc_bytes(cm.SRGB)
    check("embedded srgb detected", cm.source_profile_id(image), cm.SRGB)


def test_convert_from_embedded_untagged():
    image = Image.new("RGB", (2, 2))
    check("untagged import unchanged",
          cm.convert_from_embedded(image, "sRGB") is image, True)


def test_convert_from_embedded_srgb():
    image = Image.new("RGB", (2, 2), (10, 20, 30))
    image.info["icc_profile"] = cm.icc_bytes(cm.SRGB)
    out = cm.convert_from_embedded(image, "sRGB")
    check("srgb-tagged import is a no-op", out is image, True)


def test_registered_profile_transform():
    ident = register_srgb_embedded("icc:test-transform")
    image = Image.new("RGB", (4, 4), (120, 130, 140))
    out = cm.convert(image, cm.SRGB, ident)
    check("transform returns an image", isinstance(out, Image.Image), True)
    check("transform keeps size", out.size, (4, 4))
    near("transform pixels stay close", out.getpixel((0, 0))[0], 120, tol=2)
    check("transform is cached",
          any(key[1] == ident for key in cm._transforms), True)


def test_transform_preserves_alpha():
    ident = register_srgb_embedded("icc:test-alpha")
    image = Image.new("RGBA", (4, 4), (10, 200, 30, 123))
    out = cm.convert(image, cm.SRGB, ident)
    check("alpha output stays RGBA", out.mode, "RGBA")
    near("alpha preserved", out.getchannel("A").getpixel((0, 0)), 123)


# --------------------------------------------------------------- display/proof
def test_to_display_noop():
    image = Image.new("RGBA", (2, 2), (1, 2, 3, 4))
    check("display noop identity", cm.to_display(image, "sRGB") is image, True)


def test_to_display_unknown_proof_falls_back():
    image = Image.new("RGB", (2, 2), (1, 2, 3))
    out = cm.to_display(image, "sRGB", cm.SRGB, "/nonexistent.icc")
    check("display proof fallback returns an image",
          isinstance(out, Image.Image), True)


def test_to_cmyk_requires_profile():
    check("no cmyk profile -> None",
          cm.to_cmyk(Image.new("RGB", (2, 2)), "sRGB", ""), None)
    check("bad cmyk profile -> None",
          cm.to_cmyk(Image.new("RGB", (2, 2)), "sRGB", "/nonexistent.icc"), None)


# ----------------------------------------------------------------- profiles dir
def test_import_and_list_profiles():
    profiles_tmp = tempfile.mkdtemp()
    source_tmp = tempfile.mkdtemp()
    original = cm.profiles_dir
    cm.profiles_dir = lambda: profiles_tmp
    try:
        source = os.path.join(source_tmp, "My Profile.icc")
        with open(source, "wb") as handle:
            handle.write(cm.icc_bytes(cm.SRGB))
        ident = cm.import_profile(source)
        check("imported returns a path", bool(ident) and os.path.isfile(ident), True)
        check("imported copied into profiles dir",
              os.path.dirname(ident), profiles_tmp)

        labels = [label for _ident, label in cm.list_profiles()]
        check("sRGB listed first", labels[0], "sRGB")
        check("imported profile listed", "My Profile" in labels, True)
        check("no cmyk profiles yet", cm.list_cmyk_profiles(), [])
        check("imported colour space", cm.profile_color_space(ident), "rgb")
    finally:
        cm.profiles_dir = original
        cm.refresh()


def test_import_rejects_non_profile():
    tmp = tempfile.mkdtemp()
    original = cm.profiles_dir
    cm.profiles_dir = lambda: tmp
    try:
        bad = os.path.join(tmp, "not_a_profile.icc")
        with open(bad, "wb") as handle:
            handle.write(b"this is not an ICC profile")
        check("rejects junk file", cm.import_profile(bad), None)
    finally:
        cm.profiles_dir = original
        cm.refresh()


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
