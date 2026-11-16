"""Unit tests for the expanded filter options.

Run: .venv\\Scripts\\python.exe tests\\test_filter_options.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from paintapp import filters  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def solid(color, size=(48, 36), alpha=200):
    return Image.new("RGBA", size, (*color, alpha))


def apply(spec, image):
    return np.asarray(filters.apply_chain(image, [spec]), dtype=np.int16)


def param_max(filter_id, key):
    for param in filters.FILTERS[filter_id]["params"]:
        if param["key"] == key:
            return param["max"]
    return None


def test_channel_choice():
    source = solid((100, 50, 200))
    spec = filters.new_spec("levels")
    spec["params"].update({"channel": "red", "in_black": 128})
    out = apply(spec, source)
    check("levels targets red only", tuple(out[0, 0, :3]), (0, 50, 200))

    spec = filters.new_spec("curves")
    spec["params"].update({"channel": "blue",
                           "points": [[0, 255], [255, 0]]})
    out = apply(spec, source)
    check("curves targets blue only", tuple(out[0, 0, :3]), (100, 50, 55))


def test_old_specs_get_defaults():
    loaded = filters.normalize_spec({
        "type": "vignette", "enabled": True, "params": {"strength": 40},
    })
    params = loaded["params"]
    check("vignette radius default", params["radius"], 0)
    check("vignette feather default", params["feather"], 100)
    check("vignette centre default", params["center_x"], 50)
    check("vignette colour default", params["color"], "#000000")

    loaded = filters.normalize_spec({
        "type": "gradient_map", "enabled": True,
        "params": {"start": "#000000", "end": "#ffffff"},
    })
    check("gradient mid default", loaded["params"]["mid"], 50)
    check("gradient reverse default", loaded["params"]["reverse"], 0)

    loaded = filters.normalize_spec({
        "type": "quantize", "enabled": True, "params": {"colors": 8},
    })
    check("quantize dither default", bool(loaded["params"]["dither"]), False)

    loaded = filters.normalize_spec({
        "type": "glow", "enabled": True,
        "params": {"radius": 5.0, "strength": 60},
    })
    check("glow threshold default", loaded["params"]["threshold"], 170)
    check("glow colour default", loaded["params"]["color"], "#ffffff")

    loaded = filters.normalize_spec({
        "type": "radial_blur", "enabled": True, "params": {"strength": 30},
    })
    check("radial mode default", loaded["params"]["mode"], "zoom")
    check("radial centre default", loaded["params"]["center_x"], 50)

    for filter_id in ("levels", "curves"):
        loaded = filters.normalize_spec(filters.new_spec(filter_id))
        check(f"{filter_id} channel default", loaded["params"]["channel"], "rgb")


def test_vignette():
    source = solid((255, 255, 255), size=(64, 64))
    spec = filters.new_spec("vignette")
    spec["params"]["strength"] = 0
    check("vignette zero strength is a no-op", np.array_equal(
        apply(spec, source), np.asarray(source, dtype=np.int16)), True)

    spec = filters.new_spec("vignette")
    spec["params"].update({"center_x": 0, "center_y": 0, "radius": 10,
                           "feather": 0, "strength": 100})
    out = apply(spec, source)
    check("vignette follows the centre",
          int(out[63, 63, 0]) < int(out[0, 0, 0]), True)

    spec = filters.new_spec("vignette")
    spec["params"].update({"radius": 0, "feather": 0, "strength": 100,
                           "color": "#ff0000"})
    out = apply(spec, source)
    check("vignette uses the tint", tuple(out[63, 63, :3]), (255, 0, 0))
    check("vignette keeps alpha", int(out[0, 0, 3]), 200)

    spec = filters.new_spec("vignette")
    spec["params"]["radius"] = 100
    check("vignette radius clears the corners",
          tuple(apply(spec, source)[63, 63, :3]), (255, 255, 255))


def test_radial_blur():
    source = Image.new("RGBA", (64, 64), (0, 0, 0, 200))
    for x in range(64):
        for y in range(64):
            source.putpixel((x, y), (x * 4, y * 4, 128, 200))

    spec = filters.new_spec("radial_blur")
    zoom = apply(spec, source)
    spec["params"]["mode"] = "spin"
    spin = apply(spec, source)
    check("radial modes differ", np.array_equal(zoom, spin), False)
    check("radial keeps alpha", int(zoom[0, 0, 3]), 200)

    spec = filters.new_spec("radial_blur")
    spec["params"].update({"center_x": 0, "center_y": 0})
    corner = apply(spec, source)
    spec["params"].update({"center_x": 100, "center_y": 100})
    check("radial centre matters", np.array_equal(corner, apply(spec, source)),
          False)
    check("radial deterministic", np.array_equal(
        apply(filters.new_spec("radial_blur"), source),
        apply(filters.new_spec("radial_blur"), source)), True)


def test_glow():
    dark = solid((100, 100, 100))
    spec = filters.new_spec("glow")
    spec["params"]["threshold"] = 170
    check("glow ignores dark pixels", np.array_equal(
        apply(spec, dark), np.asarray(dark, dtype=np.int16)), True)

    spec["params"]["threshold"] = 50
    check("glow threshold admits midtones", np.array_equal(
        apply(spec, dark), np.asarray(dark, dtype=np.int16)), False)

    source = solid((200, 200, 200))
    spec = filters.new_spec("glow")
    spec["params"].update({"threshold": 170, "color": "#ff0000"})
    out = apply(spec, source)
    check("glow tints red", int(out[0, 0, 0]) > 200, True)
    check("glow leaves other channels", tuple(out[0, 0, 1:3]), (200, 200))
    check("glow keeps alpha", int(out[0, 0, 3]), 200)


def test_quantize_dither():
    arr = np.zeros((48, 48, 4), dtype=np.uint8)
    ramp = np.linspace(0, 255, 48, dtype=np.uint8)
    arr[..., :3] = ramp[None, :, None]
    arr[..., 3] = 255
    source = Image.fromarray(arr)
    spec = filters.new_spec("quantize")
    spec["params"]["colors"] = 2
    flat = apply(spec, source)
    spec["params"]["dither"] = 1
    dithered = apply(spec, source)
    check("quantize dither changes output", np.array_equal(dithered, flat), False)
    check("quantize keeps alpha", int(dithered[0, 0, 3]), 255)


def test_gradient_map():
    source = solid((128, 128, 128))
    spec = filters.new_spec("gradient_map")
    spec["params"].update({"start": "#000000", "end": "#ffffff"})
    forward = apply(spec, source)
    spec["params"]["reverse"] = 1
    reverse = apply(spec, source)
    check("gradient reverse swaps ends", np.array_equal(forward, reverse), False)
    check("gradient reverse is the inverse",
          bool(abs(int(reverse[0, 0, 0]) + int(forward[0, 0, 0]) - 255) <= 2),
          True)

    spec = filters.new_spec("gradient_map")
    base = apply(spec, source)
    spec["params"]["mid"] = 25
    check("gradient midpoint changes output", np.array_equal(
        base, apply(spec, source)), False)


def test_ranges():
    check("gaussian radius range", param_max("gaussian_blur", "radius"), 200.0)
    check("box radius range", param_max("box_blur", "radius"), 200.0)
    check("lens radius range", param_max("lens_blur", "radius"), 200.0)
    check("unsharp radius range", param_max("unsharp", "radius"), 200.0)
    check("median size range", param_max("median", "size"), 31)
    check("motion distance range", param_max("motion_blur", "distance"), 250)


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
