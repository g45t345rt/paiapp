"""Unit tests for the editable path model (run: .venv\\Scripts\\python.exe tests\\test_path.py)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import path  # noqa: E402
from paintapp.document import Canvas, Layer  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _spec(mode="fill", **overrides):
    anchor = path.Path()
    anchor.add(10, 10)
    anchor.add(110, 10)
    anchor.add(110, 110)
    return path.from_path(
        anchor, mode=mode, fill_color="#ff0000", stroke_color="#0000ff",
        width=6, outline=0, feather=0, opacity=100, antialias=True,
        cap="round", join="round", **overrides,
    )


def test_normalize_rejects_short_paths():
    check("none is dropped", path.normalize(None), None)
    check("empty anchors dropped", path.normalize({"anchors": []}), None)
    check("single anchor dropped", path.normalize({"anchors": [[0, 0]]}), None)
    check("bad anchors dropped", path.normalize({"anchors": ["x", "y"]}), None)


def test_normalize_fills_defaults():
    spec = path.normalize({"anchors": [[0, 0], [10, 0]]})
    check("mode default", spec["mode"], "fill")
    check("cap default", spec["cap"], "round")
    check("join default", spec["join"], "round")
    check("closed default", spec["closed"], False)
    check("opacity default", spec["opacity"], 100)
    check("handles padded", len(spec["handles_in"]), 2)
    check("handle matches anchor", spec["handles_in"][0], [0.0, 0.0])


def test_normalize_clamps_fields():
    spec = path.normalize({
        "anchors": [[0, 0], [1, 1]], "width": 999999, "outline": 99999,
        "feather": -5, "opacity": 500, "mode": "bogus", "cap": "bogus",
    })
    check("width clamped", spec["width"], path.MAX_WIDTH)
    check("outline clamped", spec["outline"], 200)
    check("feather floored", spec["feather"], 0)
    check("opacity clamped", spec["opacity"], 100)
    check("mode falls back", spec["mode"], "fill")
    check("cap falls back", spec["cap"], "round")


def test_clone_is_independent():
    spec = _spec()
    copy = path.clone(spec)
    copy["anchors"][0][0] = 999
    check("original untouched", spec["anchors"][0][0], 10.0)


def test_from_and_to_path_round_trip():
    spec = _spec(mode="stroke")
    rebuilt = path.to_path(spec)
    check("anchor count", len(rebuilt.anchors), 3)
    check("closed", rebuilt.closed, False)
    again = path.from_path(rebuilt, mode="stroke", fill_color="#ff0000",
                           stroke_color="#0000ff", width=6, outline=0,
                           feather=0, opacity=100, antialias=True, cap="round",
                           join="round")
    check("geometry round-trips", again["anchors"], spec["anchors"])


def test_bounds_pads_for_stroke():
    spec = _spec(mode="stroke")
    left, top, right, bottom = path.bounds(spec)
    check("left padded", left, 10 - 5)
    check("right padded", right, 110 + 5)


def test_hit_fill_and_stroke():
    fill = _spec(mode="fill")
    check("inside fill", path.hit(fill, 60, 40), True)
    check("outside fill", path.hit(fill, 5, 5), False)
    stroke = _spec(mode="stroke")
    check("on stroke", path.hit(stroke, 110, 60), True)
    check("off stroke", path.hit(stroke, 60, 60), False)


def test_anchor_and_handle_hit():
    spec = _spec()
    check("anchor found", path.anchor_hit(spec, 11, 11, 5), 0)
    check("anchor missed", path.anchor_hit(spec, 60, 60, 5), None)
    moved = _spec()
    path.set_out_handle(moved, 0, 40, 10)
    check("handle found", path.handle_hit(moved, 40, 10, 5), ("out", 0))
    check("corner handle ignored", path.handle_hit(spec, 10, 10, 5), None)


def test_geometry_editing():
    spec = _spec()
    path.move_anchor(spec, 1, 120, 20)
    check("anchor moved", spec["anchors"][1], [120.0, 20.0])
    path.set_out_handle(spec, 1, 150, 20)
    check("out handle set", spec["handles_out"][1], [150.0, 20.0])
    check("in handle mirrored", spec["handles_in"][1], [90.0, 20.0])
    path.remove_anchor(spec, 2)
    check("anchor removed", len(spec["anchors"]), 2)
    at = path.insert_anchor(spec, 0, 50, 50)
    check("insert index", at, 1)
    check("inserted point", spec["anchors"][1], [50.0, 50.0])


def test_translate_moves_all_points():
    spec = _spec()
    path.translate(spec, 5, 7)
    check("anchor shifted", spec["anchors"][0], [15.0, 17.0])
    check("handle shifted", spec["handles_out"][0], [15.0, 17.0])


def test_paste_paints_pixels():
    layer = Layer("p", (140, 140), (0, 0, 0, 0))
    path.paste(layer.image, _spec(), (0, 0))
    check("fill paints ink", layer.image.getchannel("A").getbbox() is not None, True)


def test_paste_respects_offset():
    plain = Layer("p", (140, 140), (0, 0, 0, 0))
    path.paste(plain.image, _spec(), (0, 0))
    shifted = Layer("p", (140, 140), (0, 0, 0, 0))
    path.paste(shifted.image, _spec(), (6, 6))
    base = plain.image.getchannel("A").getbbox()
    moved = shifted.image.getchannel("A").getbbox()
    check("offset shifts ink left", moved[0], base[0] - 6)
    check("offset shifts ink up", moved[1], base[1] - 6)


def test_paste_clips_outside_layer():
    layer = Layer("p", (16, 16), (0, 0, 0, 0))
    path.paste(layer.image, _spec(), (0, 0))
    check("off-layer path draws in bounds",
          layer.image.getchannel("A").getbbox() is not None, True)


def test_layer_composits_paths():
    layer = Layer("p", (140, 140), (0, 0, 0, 0))
    layer.paths = [_spec()]
    layer.invalidate_paths()
    composed = layer.composited_image()
    check("composite has ink", composed.getchannel("A").getbbox() is not None, True)


def test_rasterize_paths_bakes_and_clears():
    layer = Layer("p", (140, 140), (0, 0, 0, 0))
    layer.paths = [_spec()]
    check("rasterize reports change", layer.rasterize_paths(), True)
    check("paths cleared", layer.paths, [])
    check("pixels baked", layer.image.getchannel("A").getbbox() is not None, True)
    check("second rasterize is a no-op", layer.rasterize_paths(), False)


def test_copy_keeps_independent_paths():
    layer = Layer("p", (140, 140), (0, 0, 0, 0))
    layer.paths = [_spec()]
    duplicate = layer.copy()
    check("copy has path", len(duplicate.paths), 1)
    duplicate.paths[0]["anchors"][0][0] = 999
    check("original untouched", layer.paths[0]["anchors"][0][0], 10.0)


def test_canvas_snapshot_restores_paths():
    canvas = Canvas("c", (140, 140))
    layer = canvas.add_layer(name="P")
    layer.paths = [_spec(mode="both")]
    canvas.restore(canvas.snapshot())
    restored = canvas.layers[0]
    check("paths restored", len(restored.paths), 1)
    check("mode restored", restored.paths[0]["mode"], "both")
    check("anchors restored", restored.paths[0]["anchors"][0], [10.0, 10.0])


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
