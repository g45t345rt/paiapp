"""Unit tests for the pure-image ControlNet preprocessors.

Run: .venv\\Scripts\\python.exe tests\\test_preprocess.py
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import preprocess  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def check_true(name, condition):
    check(name, bool(condition), True)


def _shape():
    image = Image.new("RGB", (128, 96), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 16, 96, 78), fill=(210, 120, 70))
    draw.line((4, 90, 124, 6), fill=(255, 255, 255), width=2)
    return image


def _gray(image, name="none", size=None):
    rendered = preprocess.prepare(image, name, size or image.size)
    return np.asarray(rendered.convert("L"), dtype=np.uint8)


def test_public_names():
    names = preprocess.PREPROCESSORS
    check_true("names are unique", len(set(names)) == len(names))
    check("first is none", names[0], "none")
    for name in (
        "canny", "lineart", "softedge", "sketch", "scribble", "mlsd",
        "normal", "threshold", "invert", "tile",
    ):
        check_true(f"{name} is listed", name in names)


def test_prepare_contract():
    check("none image", preprocess.prepare(None, "canny", (32, 24)), None)

    flat = Image.new("RGB", (40, 30), (90, 90, 90))
    for name in preprocess.PREPROCESSORS:
        out = preprocess.prepare(flat, name, (64, 48))
        check(f"{name} size", out.size, (64, 48))
        check(f"{name} mode", out.mode, "RGB")

    fallback = preprocess.prepare(flat, "not-a-preprocessor", (20, 16))
    check("unknown size", fallback.size, (20, 16))
    check("unknown mode", fallback.mode, "RGB")


def test_invert_and_tile():
    black = Image.new("RGB", (8, 8), (0, 0, 0))
    inverted = preprocess.prepare(black, "invert", (8, 8))
    check("invert black", inverted.getpixel((0, 0)), (255, 255, 255))

    source = _shape()
    none = preprocess.prepare(source, "none", (48, 36))
    tile = preprocess.prepare(source, "tile", (48, 36))
    check_true(
        "tile is the resized source",
        np.array_equal(np.asarray(none), np.asarray(tile)),
    )


def test_threshold():
    image = Image.new("RGB", (32, 16), (20, 20, 20))
    ImageDraw.Draw(image).rectangle((16, 0, 31, 15), fill=(230, 230, 230))
    out = np.asarray(preprocess.prepare(image, "threshold", (32, 16)).convert("L"))
    check("threshold values are binary", sorted(set(out.ravel().tolist())), [0, 255])
    check("dark half is black", int(out[8, 8]), 0)
    check("light half is white", int(out[8, 24]), 255)


def test_edges():
    flat = Image.new("RGB", (48, 48), (64, 64, 64))
    canny = _gray(flat, "canny", (48, 48))
    check("canny flat is black", int(canny.max()), 0)

    shape = _shape()
    canny = _gray(shape, "canny")
    check_true("canny finds edges", int(canny.max()) == 255)

    lineart = _gray(shape, "lineart")
    check_true("lineart keeps a light paper", float(lineart.mean()) > 127.0)

    soft = np.asarray(preprocess.prepare(shape, "softedge", shape.size).convert("L"))
    check_true("softedge has soft values", len(np.unique(soft)) > 2)
    check_true("softedge finds edges", int(soft.max()) > 64)


def test_sketch_and_scribble():
    shape = _shape()
    sketch = _gray(shape, "sketch")
    check_true("sketch has paper and ink", int(sketch.min()) < 200 < int(sketch.max()))

    scribble = _gray(shape, "scribble")
    values = set(np.unique(scribble).tolist())
    check_true("scribble is binary", values.issubset({0, 255}))
    check_true("scribble has ink", int((scribble == 0).sum()) > 0)
    ink_fraction = float((scribble == 0).mean())
    check_true("scribble stays sparse", ink_fraction < 0.35)


def test_mlsd():
    image = Image.new("RGB", (128, 128), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 24, 112, 104), outline=(255, 255, 255), width=2)
    out = np.asarray(preprocess.prepare(image, "mlsd", (256, 192)).convert("L"))
    check("mlsd size", out.shape, (192, 256))
    check_true("mlsd draws lines", int((out > 0).sum()) > 0)

    flat = Image.new("RGB", (64, 64), (30, 30, 30))
    blank = _gray(flat, "mlsd", (64, 64))
    check("mlsd flat is black", int(blank.max()), 0)


def test_normal():
    flat = Image.new("RGB", (32, 32), (128, 128, 128))
    out = preprocess.prepare(flat, "normal", (32, 32))
    pixel = out.getpixel((16, 16))
    check_true(
        "flat normal points forward",
        all(abs(a - b) <= 1 for a, b in zip(pixel, (127, 127, 255))),
    )

    shape = _shape()
    edge = np.asarray(preprocess.prepare(shape, "normal", shape.size))
    check_true("edges tilt the normal", int(edge[..., 2].min()) < 250)
    check_true("normals stay in range", int(edge.min()) >= 0 and int(edge.max()) <= 255)


if __name__ == "__main__":
    test_public_names()
    test_prepare_contract()
    test_invert_and_tile()
    test_threshold()
    test_edges()
    test_sketch_and_scribble()
    test_mlsd()
    test_normal()
    if FAILED:
        print("\n%d failure(s):" % len(FAILED))
        for failure in FAILED:
            print("FAIL", failure)
        sys.exit(1)
    print("\nall ok")

