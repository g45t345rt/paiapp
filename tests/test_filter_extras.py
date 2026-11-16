"""Unit tests for the second round of filter options.

Run: .venv\\Scripts\\python.exe tests\\test_filter_extras.py
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


def solid(color, size=(32, 24), alpha=200):
    return Image.new("RGBA", size, (*color, alpha))


def apply(spec, image):
    return np.asarray(filters.apply_chain(image, [spec]), dtype=np.int16)


def ramp(size=(64, 24)):
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    values = np.linspace(0, 255, size[0], dtype=np.uint8)
    arr[..., :3] = values[None, :, None]
    arr[..., 3] = 255
    return Image.fromarray(arr)


def defaults(filter_id):
    return filters.normalize_spec(filters.new_spec(filter_id))["params"]


def test_defaults():
    params = defaults("grayscale")
    check("gray red default", params["red"], 30)
    check("gray green default", params["green"], 59)
    check("gray blue default", params["blue"], 11)
    check("gray tint default", params["tint"], "#ffffff")

    params = defaults("threshold")
    check("threshold channel default", params["channel"], "luma")
    check("threshold softness default", params["softness"], 0)

    params = defaults("color_balance")
    check("balance range default", params["range"], "all")
    check("balance preserve default", params["preserve_luminosity"], 0)

    params = defaults("halftone")
    check("halftone colour default", params["color"], "#000000")
    check("halftone opacity default", params["opacity"], 100)

    params = defaults("motion_blur")
    check("motion taper default", params["taper"], 0)
    params = defaults("lens_blur")
    check("lens threshold default", params["threshold"], 170)
    check("lens bloom default", params["bloom"], 60)
    params = defaults("sharpen")
    check("sharpen radius default", params["radius"], 1.0)
    check("sharpen threshold default", params["threshold"], 0)
    params = defaults("emboss")
    check("emboss angle default", params["angle"], 135)
    check("emboss strength default", params["strength"], 100)
    check("emboss mix default", params["mix"], 100)
    params = defaults("oil_paint")
    check("oil cleanliness default", params["cleanliness"], 5)
    check("oil shine default", params["shine"], 0)
    params = defaults("watercolor")
    check("watercolor diffusion default", params["diffusion"], 0.0)
    check("watercolor edges default", params["edges"], 100)
    params = defaults("photo_filter")
    check("photo preserve default", params["preserve_luminosity"], 0)
    params = defaults("selective")
    check("selective mode default", params["mode"], "relative")
    check("selective range default", params["target"], "0")
    params = defaults("pixelate")
    check("pixelate shape default", params["shape"], "square")


def test_grayscale():
    source = solid((255, 0, 0))
    spec = filters.new_spec("grayscale")
    spec["params"].update({"red": 100, "green": 0, "blue": 0})
    check("gray uses red weight", int(apply(spec, source)[0, 0, 0]), 255)

    spec["params"].update({"red": 0, "green": 100})
    out = apply(spec, source)
    check("gray ignores red weight when zero", int(out[0, 0, 0]), 0)

    source = solid((255, 255, 255))
    spec = filters.new_spec("grayscale")
    spec["params"]["tint"] = "#ff0000"
    out = apply(spec, source)
    check("gray tint", tuple(out[0, 0, :3]), (255, 0, 0))

    spec["params"]["amount"] = 0
    check("gray zero amount is a no-op", np.array_equal(
        apply(spec, source), np.asarray(source, dtype=np.int16)), True)


def test_threshold():
    source = solid((200, 20, 20))
    spec = filters.new_spec("threshold")
    spec["params"].update({"channel": "luma", "level": 100})
    out = apply(spec, source)
    check("threshold luma dark", int(out[0, 0, 0]), 0)

    spec["params"]["channel"] = "red"
    out = apply(spec, source)
    check("threshold red channel bright", int(out[0, 0, 0]), 255)

    source = ramp()
    spec = filters.new_spec("threshold")
    spec["params"].update({"level": 128, "softness": 0})
    hard = apply(spec, source)
    check("threshold hard is two-valued",
          set(np.unique(hard[..., 0])) <= {0, 255}, True)

    spec["params"]["softness"] = 50
    soft = apply(spec, source)
    values = np.unique(soft[..., 0])
    check("threshold softness creates ramps",
          bool(((values > 0) & (values < 255)).any()), True)
    check("threshold keeps alpha", int(soft[0, 0, 3]), 255)


def test_posterize_dither():
    source = ramp()
    spec = filters.new_spec("posterize")
    spec["params"].update({"bits": 2, "dither": 0})
    flat = apply(spec, source)
    spec["params"]["dither"] = 1
    dithered = apply(spec, source)
    check("posterize dither changes output", np.array_equal(flat, dithered), False)
    check("posterize keeps alpha", int(dithered[0, 0, 3]), 255)


def test_color_balance_ranges():
    source = ramp()
    spec = filters.new_spec("color_balance")
    spec["params"].update({"red": 100, "range": "shadows"})
    out = apply(spec, source)
    original = np.asarray(source, dtype=np.int16)
    check("balance shadows lift dark tones", int(out[0, 2, 0]) > 2, True)
    check("balance shadows spare highlights", int(out[0, -1, 0]),
          int(original[0, -1, 0]))

    spec["params"]["range"] = "highlights"
    spec["params"]["red"] = -100
    out = apply(spec, source)
    check("balance highlights spare shadows", int(out[0, 2, 0]),
          int(original[0, 2, 0]))
    check("balance highlights pull bright tones", int(out[0, -1, 0]) < 255, True)

    source = solid((100, 100, 100))
    spec = filters.new_spec("color_balance")
    spec["params"].update({"red": 50, "preserve_luminosity": 1})
    out = apply(spec, source)
    luma = 0.299 * out[0, 0, 0] + 0.587 * out[0, 0, 1] + 0.114 * out[0, 0, 2]
    check("balance preserves luminosity", abs(float(luma) - 100.0) < 3.0, True)


def test_sharpen_and_edges():
    source = Image.new("RGBA", (48, 32), (100, 100, 100, 255))
    for x in range(24, 48):
        for y in range(32):
            source.putpixel((x, y), (200, 200, 200, 255))

    spec = filters.new_spec("sharpen")
    spec["params"]["factor"] = 0.0
    check("sharpen zero amount is a no-op", np.array_equal(
        apply(spec, source), np.asarray(source, dtype=np.int16)), True)

    base = apply(filters.new_spec("sharpen"), source)
    spec = filters.new_spec("sharpen")
    spec["params"]["radius"] = 8.0
    check("sharpen radius matters", np.array_equal(base, apply(spec, source)),
          False)

    spec = filters.new_spec("find_edges")
    normal = apply(spec, source)
    spec["params"]["invert"] = 1
    inverted = apply(spec, source)
    check("find edges invert",
          np.array_equal(inverted[..., :3], 255 - normal[..., :3]), True)


def test_lens_and_motion_blur():
    source = solid((230, 230, 230), size=(40, 40))
    spec = filters.new_spec("lens_blur")
    spec["params"]["bloom"] = 0
    no_bloom = apply(spec, source)
    spec["params"]["bloom"] = 80
    check("lens bloom matters", np.array_equal(no_bloom, apply(spec, source)),
          False)

    source = solid((0, 0, 0), size=(40, 40))
    for x in range(40):
        for y in range(40):
            source.putpixel((x, y), (x * 6, y * 6, 128, 200))
    spec = filters.new_spec("motion_blur")
    spec["params"].update({"distance": 12, "angle": 0})
    uniform = apply(spec, source)
    spec["params"]["taper"] = 100
    check("motion taper matters", np.array_equal(uniform, apply(spec, source)),
          False)
    spec = filters.new_spec("motion_blur")
    spec["params"]["angle"] = 90
    check("motion angle matters", np.array_equal(uniform, apply(spec, source)),
          False)
    check("motion keeps alpha", int(uniform[0, 0, 3]), 200)


def test_stylize_amounts():
    source = solid((120, 90, 60))
    for x in range(32):
        for y in range(24):
            if (x + y) % 7 == 0:
                source.putpixel((x, y), (220, 210, 200, 200))

    for filter_id in ("contour", "detail"):
        spec = filters.new_spec(filter_id)
        spec["params"]["amount"] = 0
        check(f"{filter_id} zero amount is a no-op", np.array_equal(
            apply(spec, source), np.asarray(source, dtype=np.int16)), True)
        spec["params"]["amount"] = 100
        check(f"{filter_id} changes output", np.array_equal(
            apply(spec, source), np.asarray(source, dtype=np.int16)), False)

    for filter_id in ("edge_enhance", "smooth"):
        spec = filters.new_spec(filter_id)
        spec["params"]["amount"] = 0
        check(f"{filter_id} zero amount is a no-op", np.array_equal(
            apply(spec, source), np.asarray(source, dtype=np.int16)), True)

    spec = filters.new_spec("emboss")
    spec["params"]["strength"] = 0
    flat = apply(spec, source)
    check("emboss zero strength is flat grey",
          tuple(np.unique(flat[..., 0])), (128,))

    spec = filters.new_spec("emboss")
    spec["params"]["mix"] = 0
    check("emboss zero mix is a no-op", np.array_equal(
        apply(spec, source), np.asarray(source, dtype=np.int16)), True)

    spec = filters.new_spec("emboss")
    a = apply(spec, source)
    spec["params"]["angle"] = 45
    check("emboss angle matters", np.array_equal(a, apply(spec, source)), False)


def test_oil_and_watercolor():
    source = solid((90, 140, 200))
    for x in range(32):
        for y in range(24):
            if (x * y) % 11 == 0:
                source.putpixel((x, y), (210, 60, 30, 200))

    spec = filters.new_spec("oil_paint")
    coarse = apply(spec, source)
    spec["params"]["cleanliness"] = 2
    check("oil cleanliness matters", np.array_equal(coarse, apply(spec, source)),
          False)
    spec = filters.new_spec("oil_paint")
    spec["params"]["shine"] = 80
    check("oil shine matters", np.array_equal(coarse, apply(spec, source)), False)
    check("oil keeps alpha", int(coarse[0, 0, 3]), 200)

    spec = filters.new_spec("watercolor")
    base = apply(spec, source)
    spec["params"]["diffusion"] = 8.0
    check("watercolor diffusion matters", np.array_equal(base, apply(spec, source)),
          False)
    spec = filters.new_spec("watercolor")
    spec["params"]["edges"] = 0
    soft = apply(spec, source)
    check("watercolor edges matter", np.array_equal(soft, base), False)


def test_photo_filter_and_selective():
    source = solid((80, 80, 80))
    spec = filters.new_spec("photo_filter")
    spec["params"].update({"color": "#ff8800", "density": 80})
    plain = apply(spec, source)
    spec["params"]["preserve_luminosity"] = 1
    kept = apply(spec, source)
    luma = 0.299 * kept[0, 0, 0] + 0.587 * kept[0, 0, 1] + 0.114 * kept[0, 0, 2]
    check("photo filter preserves luminosity", abs(float(luma) - 80.0) < 3.0,
          True)
    check("photo filter preserve changes output",
          np.array_equal(plain, kept), False)

    source = solid((120, 120, 120))
    spec = filters.new_spec("selective")
    spec["params"].update({"target": "0", "cyan_red": 50})
    relative = apply(spec, source)
    spec["params"]["mode"] = "absolute"
    absolute = apply(spec, source)
    check("selective mode changes output",
          np.array_equal(relative, absolute), False)
    check("selective relative lifts red", int(relative[0, 0, 0]) > 120, True)
    check("selective absolute lifts more", int(absolute[0, 0, 0]) >
          int(relative[0, 0, 0]), True)

    loaded = filters.normalize_spec({
        "type": "selective", "enabled": True,
        "params": {"target": 3, "cyan_red": 10},
    })
    check("selective loads old int range", loaded["params"]["target"], "3")
    check("selective loads old shifts", loaded["params"]["cyan_red"], 10)


def test_pixelate_shape():
    source = ramp()
    spec = filters.new_spec("pixelate")
    spec["params"].update({"size": 8, "shape": "square"})
    square = apply(spec, source)
    spec["params"]["shape"] = "circle"
    circle = apply(spec, source)
    check("pixelate shape changes output", np.array_equal(square, circle), False)
    original = np.asarray(source, dtype=np.int16)
    check("pixelate rounds leave block corners",
          int(circle[0, 0, 0]), int(original[0, 0, 0]))
    check("pixelate square fills block corners",
          int(square[0, 0, 0]) != int(original[0, 0, 0]), True)


def test_halftone_color_opacity():
    source = solid((0, 0, 0), size=(40, 40))
    spec = filters.new_spec("halftone")
    spec["params"]["color"] = "#ff0000"
    out = apply(spec, source)
    check("halftone inks in colour", tuple(out[10, 10, :3]), (255, 0, 0))

    spec["params"]["opacity"] = 0
    check("halftone zero opacity is a no-op", np.array_equal(
        apply(spec, source), np.asarray(source, dtype=np.int16)), True)

    source = solid((120, 120, 120), size=(40, 40))
    spec = filters.new_spec("halftone")
    spec["params"].update({"color": "#3366cc", "opacity": 60})
    check("halftone keeps alpha", int(apply(spec, source)[0, 0, 3]), 200)


def test_halftone_gpu_matches_cpu():
    from paintapp import gpuimage
    if not gpuimage.available():
        print("ok  halftone colour gpu unavailable (skipped)")
        return
    rows = np.linspace(0, 255, 30, dtype=np.uint8)[:, None]
    cols = np.linspace(0, 255, 36, dtype=np.uint8)[None, :]
    arr = np.empty((30, 36, 4), dtype=np.uint8)
    arr[..., 0] = rows
    arr[..., 1] = cols
    arr[..., 2] = 128
    arr[..., 3] = 200
    source = Image.fromarray(arr)

    spec = filters.new_spec("halftone")
    spec["params"].update({"color": "#cc3344", "opacity": 70})
    gpu_out = apply(spec, source)
    original = gpuimage.available
    gpuimage.available = lambda: False
    try:
        cpu_out = apply(spec, source)
    finally:
        gpuimage.available = original
    difference = np.abs(gpu_out - cpu_out)
    check("halftone colour gpu matches cpu",
          bool(difference.max() <= 1 and difference.mean() < 0.5), True)


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
