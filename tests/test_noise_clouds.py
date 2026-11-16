"""Unit tests for the Noise and Clouds filters.

Run: .venv\\Scripts\\python.exe tests\\test_noise_clouds.py
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


def source(size=(96, 72)):
    return Image.new("RGBA", size, (120, 120, 120, 200))


def apply(spec, image=None):
    return np.asarray(
        filters.apply_chain(image if image is not None else source(), [spec]),
        dtype=np.int16,
    )


def neighbour_diff(arr):
    gray = arr[..., :3].astype(np.float32).mean(axis=2)
    return float(np.abs(np.diff(gray, axis=0)).mean()
                 + np.abs(np.diff(gray, axis=1)).mean())


def test_registered():
    check("noise present", "noise" in filters.FILTERS, True)
    check("clouds present", "clouds" in filters.FILTERS, True)
    check("clouds category", filters.FILTERS["clouds"]["category"], "pixel")
    check("clouds after noise", filters.filters_in_category("pixel")[:2],
          ["pixelate", "quantize"])


def test_defaults():
    noise = filters.new_spec("noise")["params"]
    check("noise amount default", noise["amount"], 24.0)
    check("noise scale default", noise["scale"], 1.0)
    check("noise mono default", noise["monochrome"], 1)
    clouds = filters.new_spec("clouds")["params"]
    check("clouds scale default", clouds["scale"], 128)
    check("clouds octaves default", clouds["octaves"], 5)
    check("clouds blend default", clouds["blend"], "replace")
    check("clouds opacity default", clouds["opacity"], 100)
    check("clouds tone default", clouds["tone"], [[0, 0], [255, 255]])
    check("clouds describes", filters.describe(filters.new_spec("clouds"))
          .startswith("Clouds"), True)


def test_round_trip():
    spec = filters.new_spec("clouds")
    spec["params"].update({
        "scale": 64, "octaves": 3, "roughness": 70, "seed": 42,
        "contrast": 30, "blend": "overlay", "opacity": 80, "monochrome": 0,
        "tone": [[0, 10], [255, 240]],
    })
    loaded = filters.normalize_spec(spec)
    for key, value in spec["params"].items():
        check(f"clouds round trip {key}", loaded["params"][key], value)

    spec = filters.new_spec("noise")
    spec["params"].update({"amount": 60.0, "scale": 8.0, "seed": 7})
    loaded = filters.normalize_spec(spec)
    check("noise round trip amount", loaded["params"]["amount"], 60.0)
    check("noise round trip scale", loaded["params"]["scale"], 8.0)
    check("noise round trip seed", loaded["params"]["seed"], 7)


def test_noise_behaviour():
    spec = filters.new_spec("noise")
    spec["params"]["seed"] = 3
    first = apply(spec)
    second = apply(spec)
    check("noise deterministic", np.array_equal(first, second), True)

    spec["params"]["seed"] = 4
    check("noise seed changes result", np.array_equal(first, apply(spec)), False)

    spec = filters.new_spec("noise")
    spec["params"].update({"amount": 0.0})
    check("noise zero amount is a no-op", np.array_equal(
        apply(spec), np.asarray(source(), dtype=np.int16)), True)

    spec = filters.new_spec("noise")
    mono = apply(spec)
    check("noise monochrome", np.array_equal(mono[..., 0], mono[..., 1])
          and np.array_equal(mono[..., 1], mono[..., 2]), True)
    spec["params"]["monochrome"] = 0
    colour = apply(spec)
    check("noise colour differs per channel",
          np.array_equal(colour[..., 0], colour[..., 1]), False)

    grain = apply(filters.new_spec("noise"))
    spec = filters.new_spec("noise")
    spec["params"]["scale"] = 16.0
    clumpy = apply(spec)
    check("noise scale is smoother", neighbour_diff(clumpy) < neighbour_diff(grain),
          True)
    check("noise keeps alpha", int(grain[..., 3].min()), 200)


def test_clouds_behaviour():
    spec = filters.new_spec("clouds")
    spec["params"]["seed"] = 5
    first = apply(spec)
    check("clouds deterministic", np.array_equal(first, apply(spec)), True)

    spec["params"]["seed"] = 6
    check("clouds seed changes result", np.array_equal(first, apply(spec)), False)

    spec = filters.new_spec("clouds")
    mono = apply(spec)
    check("clouds monochrome", np.array_equal(mono[..., 0], mono[..., 1])
          and np.array_equal(mono[..., 1], mono[..., 2]), True)
    check("clouds keeps alpha", int(mono[..., 3].min()), 200)
    check("clouds is not flat", int(mono[..., 0].std()) > 5, True)

    spec["params"]["monochrome"] = 0
    colour = apply(spec)
    check("clouds colour differs per channel",
          np.array_equal(colour[..., 0], colour[..., 1]), False)

    spec = filters.new_spec("clouds")
    spec["params"]["octaves"] = 1
    spec["params"]["roughness"] = 100
    coarse = neighbour_diff(apply(spec))
    spec["params"]["octaves"] = 8
    detailed = neighbour_diff(apply(spec))
    check("clouds detail adds structure", detailed > coarse, True)

    spec = filters.new_spec("clouds")
    plain = float(apply(spec)[..., 0].std())
    spec["params"]["contrast"] = 80
    punchy = float(apply(spec)[..., 0].std())
    check("clouds contrast widens range", punchy > plain, True)

    spec = filters.new_spec("clouds")
    spec["params"]["opacity"] = 0
    check("clouds zero opacity is a no-op", np.array_equal(
        apply(spec), np.asarray(source(), dtype=np.int16)), True)

    spec = filters.new_spec("clouds")
    spec["params"]["tone"] = [[0, 0], [255, 0]]
    check("clouds tone curve applies", int(apply(spec)[..., :3].max()), 0)

    base = Image.new("RGBA", (64, 64), (30, 60, 90, 255))
    spec = filters.new_spec("clouds")
    replaced = apply(spec, base)
    check("clouds replace ignores base", bool((replaced[..., :3] !=
          np.array([30, 60, 90])).any()), True)

    spec["params"]["blend"] = "multiply"
    spec["params"]["opacity"] = 100
    multiplied = apply(spec, base)
    check("clouds multiply keeps base tint",
          int(multiplied[..., 0].max()) <= 30
          and int(multiplied[..., 2].max()) <= 90, True)


def test_tiny_image():
    tiny = Image.new("RGBA", (1, 1), (100, 100, 100, 255))
    for fid in ("noise", "clouds"):
        out = filters.apply_chain(tiny, [filters.new_spec(fid)])
        check(f"{fid} handles 1x1", out.size, (1, 1))


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
