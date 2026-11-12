"""Unit tests for the vector stroke model (run: .venv\\Scripts\\python.exe tests\\test_vector.py)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import math  # noqa: E402

from paintapp import shape, vector  # noqa: E402
from paintapp.document import Canvas, Layer  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_normalize_rejects_empty():
    check("none is dropped", vector.normalize(None), None)
    check("empty points dropped", vector.normalize({"points": []}), None)
    check("bad points dropped", vector.normalize({"points": ["x"]}), None)


def test_normalize_fills_defaults():
    spec = vector.normalize({"points": [[0, 0], [10, 10]]})
    check("width default", spec["width"], vector.DEFAULT_WIDTH)
    check("color default", spec["color"], "#000000")
    check("opacity default", spec["opacity"], 1.0)
    check("cap default", spec["cap"], "round")
    check("closed default", spec["closed"], False)


def test_normalize_clamps_fields():
    spec = vector.normalize({
        "points": [[0, 0]], "width": 999999, "opacity": 5, "cap": "bogus",
        "join": "bogus",
    })
    check("width clamped", spec["width"], vector.MAX_WIDTH)
    check("opacity clamped", spec["opacity"], 1.0)
    check("cap falls back", spec["cap"], "round")
    check("join falls back", spec["join"], "round")


def test_clone_is_independent():
    spec = vector.normalize({"points": [[0, 0], [1, 1]], "width": 4})
    copy = vector.clone(spec)
    copy["points"][0][0] = 99
    check("original untouched", spec["points"][0][0], 0)


def test_bounds_pads_for_width():
    spec = vector.normalize({"points": [[10, 10], [20, 20]], "width": 6})
    left, top, right, bottom = vector.bounds(spec)
    check("left padded", left, 10 - 4)
    check("right padded", right, 20 + 4)


def test_hit_stroke_uses_width():
    spec = vector.normalize({"points": [[0, 0], [100, 0]], "width": 10})
    check("on the line", vector.hit_stroke(spec, 50, 0), True)
    check("just off the line", vector.hit_stroke(spec, 50, 4), True)
    check("far off the line", vector.hit_stroke(spec, 50, 40), False)


def test_nearest_segment_and_insert():
    spec = vector.normalize({"points": [[0, 0], [100, 0], [100, 100]]})
    index, t = vector.nearest_segment(spec, 50, 5)
    check("nearest is first segment", index, 0)
    check("t is mid-segment", round(t, 1), 0.5)
    vector.insert_point(spec, index, 50, 5)
    check("point inserted", spec["points"][1], [50.0, 5.0])


def test_move_and_remove_point():
    spec = vector.normalize({"points": [[0, 0], [10, 0], [20, 0]]})
    vector.move_point(spec, 1, 11, 7)
    check("point moved", spec["points"][1], [11.0, 7.0])
    check("remove returns true", vector.remove_point(spec, 0), True)
    check("point count", len(spec["points"]), 2)
    check("remove out of range", vector.remove_point(spec, 9), False)


def test_erase_untouched_stroke():
    spec = vector.normalize({"points": [[0, 0], [100, 0]], "width": 4})
    pieces = vector.erase(spec, 50, 500, 10)
    check("untouched stays one stroke", len(pieces), 1)
    check("untouched keeps points", len(pieces[0]["points"]), 2)


def test_erase_middle_splits_into_two():
    spec = vector.normalize({
        "points": [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], "width": 4,
    })
    pieces = vector.erase(spec, 20, 0, 5)
    check("middle cut splits", len(pieces), 2)
    check("first piece points", [p[0] for p in pieces[0]["points"]], [0.0, 10.0])
    check("second piece points", [p[0] for p in pieces[1]["points"]], [30.0, 40.0])


def test_erase_end_shrinks():
    spec = vector.normalize({"points": [[0, 0], [10, 0], [20, 0]], "width": 4})
    pieces = vector.erase(spec, 0, 0, 5)
    check("end cut keeps one piece", len(pieces), 1)
    check("end cut drops node", [p[0] for p in pieces[0]["points"]], [10.0, 20.0])


def test_erase_swallows_stroke():
    spec = vector.normalize({"points": [[0, 0], [5, 0]], "width": 4})
    check("whole stroke erased", vector.erase(spec, 2, 0, 50), [])


def test_correct_width_uniform():
    spec = vector.normalize({"points": [[0, 0], [10, 0]], "width": 3})
    vector.correct_width(spec, 12)
    check("width corrected", spec["width"], 12)


def test_paste_paints_pixels():
    layer = Layer("v", (64, 64), (0, 0, 0, 0))
    spec = vector.new_stroke([[8, 32], [56, 32]], 6, "#000000")
    vector.paste(layer.image, spec, (0, 0))
    check("stroke paints ink", layer.image.getchannel("A").getbbox() is not None, True)


def test_paste_respects_offset():
    spec = vector.new_stroke([[8, 32], [56, 32]], 6, "#000000")
    plain = Layer("v", (64, 64), (0, 0, 0, 0))
    vector.paste(plain.image, spec, (0, 0))
    shifted = Layer("v", (64, 64), (0, 0, 0, 0))
    vector.paste(shifted.image, spec, (4, 4))
    base = plain.image.getchannel("A").getbbox()
    moved = shifted.image.getchannel("A").getbbox()
    check("offset shifts ink left", moved[0], base[0] - 4)
    check("offset shifts ink up", moved[1], base[1] - 4)


def test_paste_clips_outside_layer():
    layer = Layer("v", (16, 16), (0, 0, 0, 0))
    spec = vector.new_stroke([[-100, 8], [200, 8]], 6, "#000000")
    vector.paste(layer.image, spec, (0, 0))
    check("off-layer stroke still draws in-bounds", layer.image.getchannel("A").getbbox() is not None, True)


def test_layer_composits_strokes():
    layer = Layer("v", (32, 32), (0, 0, 0, 0))
    layer.vector = True
    layer.strokes = [vector.new_stroke([[4, 16], [28, 16]], 6, "#ff0000")]
    layer.invalidate_vectors()
    composed = layer.composited_image()
    check("composite has ink", composed.getchannel("A").getbbox() is not None, True)


def test_rasterize_vectors_bakes_and_clears():
    layer = Layer("v", (32, 32), (0, 0, 0, 0))
    layer.vector = True
    layer.strokes = [vector.new_stroke([[4, 16], [28, 16]], 6, "#000000")]
    check("rasterize reports change", layer.rasterize_vectors(), True)
    check("strokes cleared", layer.strokes, [])
    check("no longer vector", layer.vector, False)
    check("pixels baked", layer.image.getchannel("A").getbbox() is not None, True)
    check("second rasterize is a no-op", layer.rasterize_vectors(), False)


def test_simplify_keeps_ends():
    points = [[0, 0], [0.1, 0], [0.2, 0], [50, 0]]
    kept = vector.simplify(points, distance=1.5)
    check("first kept", kept[0], [0, 0])
    check("last kept", kept[-1], [50, 0])


def test_canvas_snapshot_restores_vector_layer():
    canvas = Canvas("c", (32, 32))
    layer = canvas.add_layer(name="V", vector=True)
    layer.strokes = [vector.new_stroke([[1, 1], [9, 9]], 3, "#000000")]
    canvas.restore(canvas.snapshot())
    restored = canvas.layers[0]
    check("vector flag restored", restored.vector, True)
    check("strokes restored", len(restored.strokes), 1)
    check("layer stays vector", restored.strokes[0]["width"], 3)


def _deviation(points, polyline):
    """Worst distance from each of ``points`` to the ``polyline`` segments."""
    worst = 0.0
    for px, py in points:
        best = min(
            vector._distance_to_segment(px, py, a[0], a[1], b[0], b[1])[0]
            for a, b in zip(polyline, polyline[1:])
        )
        worst = max(worst, best)
    return worst


def test_handles_default_to_corners():
    spec = vector.normalize({"points": [[0, 0], [4, 4]]})
    check("in handles padded", spec["handles_in"], [[0.0, 0.0], [4.0, 4.0]])
    check("out handles padded", spec["handles_out"], [[0.0, 0.0], [4.0, 4.0]])


def test_flatten_straight_is_polyline():
    spec = vector.normalize({"points": [[0, 0], [10, 0], [10, 10]]})
    check("a corner stroke flattens to its anchors",
          vector.flatten(spec), [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])


def test_flatten_closed_loops_back():
    spec = vector.normalize(
        {"points": [[0, 0], [10, 0], [10, 10]], "closed": True})
    points = vector.flatten(spec)
    check("closed loop ends on its first point", points[-1], points[0])
    check("closed loop has a closing segment", len(points), 4)


def test_curve_bows_between_anchors():
    spec = vector.normalize({
        "points": [[0, 0], [100, 0]],
        "handles_out": [[0, 100], [100, 0]],
        "handles_in": [[0, 0], [100, 100]],
    })
    midpoint = vector.flatten(spec)[8]
    check("curve midpoint x", round(midpoint[0], 1), 50.0)
    check("curve midpoint bows off the chord", round(midpoint[1], 1), 75.0)


def test_bounds_follow_the_curve():
    spec = vector.normalize({
        "points": [[0, 0], [100, 0]],
        "handles_out": [[0, 100], [100, 0]],
        "handles_in": [[0, 0], [100, 100]],
        "width": 2,
    })
    left, top, right, bottom = vector.bounds(spec)
    check("bounds reach the bow", bottom > 75.0, True)


def test_move_point_moves_handles():
    spec = vector.normalize({
        "points": [[0, 0], [10, 0]],
        "handles_out": [[0, 5], [10, 0]], "handles_in": [[0, 3], [10, 0]],
    })
    vector.move_point(spec, 0, 1, 1)
    check("anchor moved", spec["points"][0], [1.0, 1.0])
    check("out handle followed", spec["handles_out"][0], [1.0, 6.0])
    check("in handle followed", spec["handles_in"][0], [1.0, 4.0])


def test_split_segment_keeps_shape():
    spec = vector.normalize({
        "points": [[0, 0], [100, 0]],
        "handles_out": [[0, 100], [100, 0]],
        "handles_in": [[0, 0], [100, 100]],
    })
    before = vector.flatten(spec)
    new_index = vector.split_segment(spec, 0, 0.5)
    check("anchor inserted after the first", new_index, 1)
    check("new anchor lands on the curve",
          [round(v, 1) for v in spec["points"][1]], [50.0, 75.0])
    check("split preserves the curve", _deviation(before, vector.flatten(spec)) < 0.5, True)


def test_nearest_segment_on_curve():
    spec = vector.normalize({
        "points": [[0, 0], [100, 0], [100, 100]],
        "handles_out": [[0, 80], [100, 0], [100, 100]],
        "handles_in": [[0, 0], [100, 80], [100, 100]],
    })
    index, t = vector.nearest_segment(spec, 50, 60)
    check("nearest is the first anchor segment", index, 0)
    check("t is mid-curve", round(t, 1), 0.5)


def test_erase_keeps_handles_on_pieces():
    spec = vector.normalize({
        "points": [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]],
        "handles_out": [[2, 0], [12, 5], [22, 5], [32, 5], [40, 0]],
        "handles_in": [[0, 0], [8, 5], [18, 5], [28, 5], [38, 0]],
    })
    pieces = vector.erase(spec, 20, 0, 5)
    check("middle cut still splits", len(pieces), 2)
    check("first piece keeps handle count",
          len(pieces[0]["handles_in"]), len(pieces[0]["points"]))
    check("second piece points", [p[0] for p in pieces[1]["points"]], [30.0, 40.0])
    check("second piece handles follow",
          len(pieces[1]["handles_out"]), len(pieces[1]["points"]))


def test_transforms_move_handles():
    spec = vector.normalize({
        "points": [[0, 0], [10, 0]],
        "handles_out": [[0, 5], [10, 5]], "handles_in": [[0, -5], [10, -5]],
    })
    vector.translate(spec, 1, 2)
    check("translate moves handles", spec["handles_out"][0], [1.0, 7.0])
    vector.scale(spec, 2, 2, 0, 0)
    check("scale moves handles", spec["handles_out"][0], [2.0, 14.0])
    vector.rotate(spec, math.pi / 2, 0, 0)
    check("rotate moves handle x", round(spec["handles_out"][0][0], 3), -14.0)
    check("rotate moves handle y", round(spec["handles_out"][0][1], 3), 2.0)


def test_set_handle_mirrors():
    spec = vector.normalize({"points": [[0, 0], [10, 0]]})
    vector.set_handle(spec, 0, "out", 4, 4)
    check("out handle set", spec["handles_out"][0], [4.0, 4.0])
    check("in handle mirrored", spec["handles_in"][0], [-4.0, -4.0])
    vector.set_handle(spec, 0, "out", 4, 4, mirror=False)
    check("corner leaves the other side", spec["handles_in"][0], [-4.0, -4.0])


def test_paste_curve_paints_off_chord():
    layer = Layer("v", (64, 64), (0, 0, 0, 0))
    spec = vector.new_stroke(
        [[8, 56], [56, 56]], 4, "#000000",
        handles_out=[[8, 8], [56, 56]], handles_in=[[8, 56], [56, 8]])
    vector.paste(layer.image, spec, (0, 0))
    check("ink at the bow", layer.image.getpixel((32, 20))[3] > 0, True)
    check("chord stays empty", layer.image.getpixel((32, 56))[3], 0)


def test_handle_hit():
    spec = vector.normalize({
        "points": [[0, 0], [10, 0]],
        "handles_out": [[0, 8], [10, 0]], "handles_in": [[0, 0], [10, 0]],
    })
    check("out handle found", vector.handle_hit(spec, 0, 8, 2), ("out", 0))
    check("corner handle ignored", vector.handle_hit(spec, 0, 0, 2), None)
    check("far miss", vector.handle_hit(spec, 50, 50, 2), None)


def test_set_corner_and_smooth():
    spec = vector.normalize({
        "points": [[0, 0], [10, 0], [20, 10]],
        "handles_out": [[0, 0], [15, 15], [20, 10]],
        "handles_in": [[0, 0], [5, -5], [20, 10]],
    })
    vector.set_corner(spec, 1)
    check("corner collapses in", spec["handles_in"][1], [10.0, 0.0])
    check("corner collapses out", spec["handles_out"][1], [10.0, 0.0])
    vector.set_smooth(spec, 1)
    out = spec["handles_out"][1]
    mid = spec["handles_in"][1]
    check("smooth pulls a handle out", out[0] > 10.0, True)
    check("smooth handles are collinear",
          round((out[0] - 10.0) * (mid[1] - 0.0) - (out[1] - 0.0) * (mid[0] - 10.0), 3),
          0.0)
    check("smooth handles straddle the anchor",
          (out[0] - 10.0) * (mid[0] - 10.0) < 0, True)


def test_toggle_closed():
    spec = vector.normalize({"points": [[0, 0], [10, 0], [10, 10]]})
    check("closes", vector.toggle_closed(spec), True)
    check("opens", vector.toggle_closed(spec), False)


def test_split_at_open_stroke():
    spec = vector.normalize({"points": [[0, 0], [10, 0], [20, 0], [30, 0]]})
    pieces = vector.split_at(spec, 1)
    check("open split makes two", len(pieces), 2)
    check("first piece", [p[0] for p in pieces[0]["points"]], [0.0, 10.0])
    check("second piece", [p[0] for p in pieces[1]["points"]], [10.0, 20.0, 30.0])
    check("ends cannot split", len(vector.split_at(spec, 0)), 1)


def test_split_at_closed_stroke_opens_it():
    spec = vector.normalize(
        {"points": [[0, 0], [10, 0], [10, 10]], "closed": True})
    pieces = vector.split_at(spec, 1)
    check("closed split stays one stroke", len(pieces), 1)
    check("now open", pieces[0]["closed"], False)
    check("reordered from the split", pieces[0]["points"][0], [10.0, 0.0])
    check("meets itself again", pieces[0]["points"][-1], [10.0, 0.0])


def test_erase_whole_mode():
    spec = vector.normalize({"points": [[0, 0], [10, 0], [20, 0]], "width": 4})
    check("whole mode removes the touched line",
          vector.erase(spec, 10, 0, 5, mode="whole"), [])
    check("whole mode keeps an untouched line",
          len(vector.erase(spec, 100, 100, 5, mode="whole")), 1)


def test_scale_width():
    spec = vector.normalize({"points": [[0, 0], [10, 0]], "width": 4})
    vector.scale_width(spec, 2.0)
    check("thickened", spec["width"], 8)
    vector.scale_width(spec, 0.05)
    check("narrowed to the minimum", spec["width"], 1)


def test_simplify_spec_collapses_collinear_run():
    spec = vector.normalize({"points": [[i, 0] for i in range(21)]})
    vector.simplify_spec(spec, tolerance=0.5)
    check("collinear run collapses to its ends", len(spec["points"]), 2)
    check("ends kept", spec["points"], [[0.0, 0.0], [20.0, 0.0]])


def test_simplify_spec_keeps_a_corner():
    spec = vector.normalize({"points": [[i, 0] for i in range(0, 11)]
                            + [[10, 5], [10, 10]]})
    vector.simplify_spec(spec, tolerance=0.5)
    check("corner is kept", [10.0, 0.0] in spec["points"], True)


def test_connect_joins_stubs():
    a = vector.normalize({"points": [[0, 0], [10, 0]]})
    b = vector.normalize({"points": [[11, 0], [20, 0]]})
    far = vector.normalize({"points": [[100, 100], [110, 100]]})
    joined = vector.connect([a, b, far], tolerance=2.0)
    check("two near stubs merge", len(joined), 2)
    merged = [spec for spec in joined if len(spec["points"]) == 3][0]
    check("merged anchors", [p[0] for p in merged["points"]], [0.0, 10.0, 20.0])
    check("distant stub untouched",
          any(len(spec["points"]) == 2 for spec in joined), True)


def test_connect_reverses_to_join():
    a = vector.normalize({"points": [[0, 0], [10, 0]]})
    c = vector.normalize({"points": [[30, 0], [10.5, 0]]})
    joined = vector.connect([a, c], tolerance=2.0)
    check("reversed stub joins", len(joined), 1)
    check("joined in order", [p[0] for p in joined[0]["points"]],
          [0.0, 10.0, 30.0])


def test_fill_paints_inside_closed_stroke():
    layer = Layer("v", (64, 64), (0, 0, 0, 0))
    spec = vector.new_stroke([[10, 10], [50, 10], [50, 50], [10, 50]], 2,
                             "#000000", closed=True, fill="#ff0000")
    vector.paste(layer.image, spec, (0, 0))
    red, green, blue, alpha = layer.image.getpixel((30, 30))
    check("fill centre opaque", alpha, 255)
    check("fill centre is red", red > 200 and green < 80 and blue < 80, True)
    check("outside stays clear", layer.image.getpixel((2, 2))[3], 0)


def test_fill_skipped_when_open():
    layer = Layer("v", (64, 64), (0, 0, 0, 0))
    spec = vector.new_stroke([[10, 10], [50, 50]], 2, "#000000",
                             fill="#ff0000")
    vector.paste(layer.image, spec, (0, 0))
    check("open stroke is not filled", layer.image.getpixel((45, 20))[3], 0)


def test_fill_round_trips():
    spec = vector.new_stroke([[0, 0], [10, 0], [10, 10]], 3, "#000000",
                             closed=True, fill="#00ff00")
    check("fill kept by clone", vector.clone(spec)["fill"], "#00ff00")
    check("no fill by default",
          vector.normalize({"points": [[0, 0], [1, 0]]})["fill"], None)


def _contour(points):
    return {"points": [list(point) for point in points]}


def test_nonzero_fill_keeps_opposite_wound_hole():
    # SVG's default fill rule is nonzero: a nested contour wound the other way
    # (an icon's knocked-out hole) must stay clear, not fill solid.
    fill = vector.new_fill(
        [_contour([(0, 0), (12, 0), (12, 12), (0, 12)]),
         _contour([(3, 3), (3, 9), (9, 9), (9, 3)])],
        "#000000", even_odd=False)
    layer = Layer("v", (12, 12), (0, 0, 0, 0))
    vector.paste_fill(layer.image, fill, (0, 0))
    check("nonzero ring is painted", layer.image.getpixel((1, 6))[3], 255)
    check("nonzero hole stays clear", layer.image.getpixel((6, 6))[3], 0)


def test_nonzero_fill_unions_same_wound_contours():
    fill = vector.new_fill(
        [_contour([(0, 0), (12, 0), (12, 12), (0, 12)]),
         _contour([(3, 3), (9, 3), (9, 9), (3, 9)])],
        "#000000", even_odd=False)
    layer = Layer("v", (12, 12), (0, 0, 0, 0))
    vector.paste_fill(layer.image, fill, (0, 0))
    check("same winding does not cut a hole",
          layer.image.getpixel((6, 6))[3], 255)




def _square_fill():
    contour = {
        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "handles_in": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "handles_out": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }
    return vector.new_fill([contour], "#000000")


def test_item_anchor_over_a_fill():
    fill = _square_fill()
    check("anchors enumerated", vector.item_anchor_points(fill),
          [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    check("anchor hit", vector.item_anchor_hit(fill, 10, 0, 2), 1)
    check("anchor count", vector.item_anchor_count(fill), 4)


def test_item_move_point_on_fill():
    fill = _square_fill()
    vector.item_move_point(fill, 1, 12, -1)
    check("anchor moved", fill["contours"][0]["points"][1], [12.0, -1.0])


def test_item_remove_point_drops_degenerate_contour():
    fill = _square_fill()
    check("a triangle still bounds an area", vector.item_remove_point(fill, 2), True)
    check("three anchors left", vector.item_anchor_count(fill), 3)
    check("removing again empties the fill", vector.item_remove_point(fill, 0), False)


def test_item_hit_fill_interior_and_edge():
    fill = _square_fill()
    check("inside hits", vector.item_hit(fill, 5, 5), True)
    check("outside misses", vector.item_hit(fill, 20, 20), False)
    check("near edge hits with tolerance", vector.item_hit(fill, 10.5, 5, 2.0), True)


def test_item_transform_fill():
    fill = _square_fill()
    vector.item_translate(fill, 5, 0)
    check("translated", fill["contours"][0]["points"][0], [5.0, 0.0])
    vector.item_scale(fill, 2, 2, 0, 0)
    check("scaled", fill["contours"][0]["points"][1], [30.0, 0.0])
    vector.item_rotate(fill, math.pi / 2, 0, 0)
    check("rotated x", round(fill["contours"][0]["points"][1][0], 3), 0.0)
    check("rotated y", round(fill["contours"][0]["points"][1][1], 3), 30.0)


def test_item_snapshot_restore_fill():
    fill = _square_fill()
    snapshot = vector.item_snapshot(fill)
    vector.item_move_point(fill, 0, 99, 99)
    vector.item_restore(fill, snapshot)
    check("geometry restored", fill["contours"][0]["points"][0], [0.0, 0.0])


def test_item_split_fill_contour():
    fill = _square_fill()
    near = vector.item_nearest_segment(fill, 5, 0)
    check("nearest is on the first contour", near[0], 0)
    vector.item_split(fill, near[0], near[1], near[2])
    check("anchor added", vector.item_anchor_count(fill), 5)


def test_item_handle_pairs_from_a_curve():
    spec = vector.new_stroke([[0, 0], [10, 0]], 3, "#000000",
                             handles_out=[[0, 8], [10, 0]])
    stubs = vector.item_handle_pairs(spec)
    check("one handle stub", len(stubs), 1)
    check("stub reaches the handle", stubs[0][2:], [0.0, 8.0])


def test_variable_width_profile():
    spec = vector.new_stroke([[8, 32], [56, 32]], 6, "#000000",
                             widths=[2, 20])
    check("profile stored", spec["widths"], [2.0, 20.0])
    check("profile detected", vector.has_width_profile(spec), True)
    left, top, right, bottom = vector.bounds(spec)
    check("bounds use the widest anchor", bottom - top, 22.0)
    layer = Layer("v", (64, 64), (0, 0, 0, 0))
    vector.paste(layer.image, spec, (0, 0))
    check("thin end is only on the line",
          layer.image.getpixel((10, 26))[3], 0)
    check("thick end reaches well off the line",
          layer.image.getpixel((54, 26))[3], 255)


def test_width_profile_survives_split():
    spec = vector.new_stroke([[0, 0], [100, 0]], 6, "#000000",
                             widths=[2, 20])
    vector.split_segment(spec, 0, 0.5)
    check("split interpolates width", spec["widths"][1], 11.0)


def test_dash_runs_and_gaps():
    spec = vector.new_stroke([[0, 10], [100, 10]], 4, "#000000",
                             dash=[10, 10], cap="butt")
    runs = vector.outline_polygons(spec)
    check("dash yields separate runs", len(runs), 5)
    layer = Layer("v", (120, 20), (0, 0, 0, 0))
    vector.paste(layer.image, spec, (0, 0))
    check("ink on a dash", layer.image.getpixel((5, 10))[3] > 0, True)
    check("gap stays clear", layer.image.getpixel((15, 10))[3], 0)


def test_arrow_marker_adds_ink():
    plain = vector.new_stroke([[8, 32], [56, 32]], 4, "#000000")
    marker = vector.new_stroke([[8, 32], [56, 32]], 4, "#000000",
                               arrow_end="triangle")
    a = Layer("v", (64, 64), (0, 0, 0, 0))
    b = Layer("v", (64, 64), (0, 0, 0, 0))
    vector.paste(a.image, plain, (0, 0))
    vector.paste(b.image, marker, (0, 0))
    check("marker widens the ink",
          b.image.getchannel("A").getbbox()[3]
          > a.image.getchannel("A").getbbox()[3], True)


def test_gradient_fill_paints_stops():
    gradient = {
        "type": "linear", "stops": [[0, "#ff0000"], [1, "#0000ff"]],
        "start": [10, 10], "end": [50, 10],
    }
    spec = vector.new_stroke([[10, 10], [50, 10], [50, 50], [10, 50]], 2,
                             "#000000", closed=True, fill_gradient=gradient)
    layer = Layer("v", (64, 64), (0, 0, 0, 0))
    vector.paste(layer.image, spec, (0, 0))
    left = layer.image.getpixel((14, 30))
    right = layer.image.getpixel((46, 30))
    check("gradient start is red", left[0] > left[2], True)
    check("gradient end is blue", right[2] > right[0], True)


def _square_fill_spec(left, top, right, bottom):
    contour = {"points": [[left, top], [right, top], [right, bottom],
                         [left, bottom]]}
    return vector.new_fill([contour], "#000000")


def test_boolean_operations():
    first = _square_fill_spec(5, 5, 35, 35)
    second = _square_fill_spec(20, 20, 50, 50)
    union = vector.boolean([first, second], "union")
    intersect = vector.boolean([first, second], "intersect")
    subtract = vector.boolean([first, second], "subtract")
    exclude = vector.boolean([first, second], "exclude")
    check("union exists", union is not None, True)
    check("intersect exists", intersect is not None, True)
    check("subtract exists", subtract is not None, True)
    check("exclude exists", exclude is not None, True)

    def ink(spec, points):
        layer = Layer("v", (64, 64), (0, 0, 0, 0))
        vector.paste_fill(layer.image, spec, (0, 0))
        return [layer.image.getpixel(point)[3] > 0 for point in points]

    check("union covers both",
          ink(union, [(10, 10), (30, 30), (45, 45)]), [True, True, True])
    check("intersect is the overlap",
          ink(intersect, [(10, 10), (30, 30), (45, 45)]),
          [False, True, False])
    check("subtract removes the overlap",
          ink(subtract, [(10, 10), (30, 30), (45, 45)]),
          [True, False, False])
    check("exclude keeps only the non-overlap",
          ink(exclude, [(10, 10), (30, 30), (45, 45)]),
          [True, False, True])


def test_duplicate_align_and_distribute():
    items = [
        vector.new_stroke([[0, 0], [10, 0]], 2, "#000000"),
        vector.new_stroke([[40, 40], [50, 40]], 2, "#000000"),
        vector.new_stroke([[80, 100], [90, 100]], 2, "#000000"),
    ]
    duplicate = vector.duplicate_item(items[0], 5, 7)
    check("duplicate is independent", duplicate["points"][0], [5.0, 7.0])
    vector.align_items(items, "left")
    lefts = [vector.item_bounds(item)[0] for item in items]
    check("aligned to one left edge", len(set(lefts)), 1)
    vector.distribute_items(items, "v")
    centres = [round((vector.item_bounds(item)[1]
                      + vector.item_bounds(item)[3]) / 2.0, 2)
               for item in items]
    check("evenly distributed", centres, [0.0, 50.0, 100.0])


def test_item_blend_mode_multiplies():
    layer = Layer("v", (32, 32), (200, 200, 200, 255))
    spec = vector.new_stroke([[2, 16], [30, 16]], 8, "#808080",
                             blend="multiply")
    vector.paste(layer.image, spec, (0, 0))
    pixel = layer.image.getpixel((16, 16))
    check("multiply darkens the backdrop", pixel[0] < 150, True)
    check("multiply stays opaque", pixel[3], 255)


def test_boolean_with_strokes():
    first = vector.new_stroke([[0, 0], [30, 0]], 10, "#000000")
    second = vector.new_stroke([[15, -10], [15, 20]], 10, "#000000")
    result = vector.boolean([first, second], "union")
    check("strokes can be unioned", result is not None, True)
    check("result has anchors", vector.item_anchor_count(result) > 0, True)



def test_shape_geometry_is_editable():
    rect = shape.vector_geometry("rectangle", (5, 6, 35, 26))
    check("rectangle has four anchors", len(rect["points"]), 4)
    check("rectangle is closed", rect["closed"], True)
    check("rectangle corners", rect["points"][0:2],
          [[5.0, 6.0], [35.0, 6.0]])
    rounded = shape.vector_geometry("rectangle", (0, 0, 40, 40), radius=8)
    check("rounded rectangle has curve controls", len(rounded["points"]) > 4,
          True)
    ellipse = shape.vector_geometry("ellipse", (10, 10, 50, 30))
    check("ellipse is four cubic anchors", len(ellipse["points"]), 4)
    check("ellipse is closed", ellipse["closed"], True)
    check("ellipse has off-anchor handles",
          ellipse["handles_out"][0] != ellipse["points"][0], True)
    line = shape.vector_geometry("line", (0, 0, 20, 7), shift=True)
    check("shift constrains the line to an axis", line["points"][1][1], 0.0)


def test_vector_item_selection_switches_tools_mode():
    from types import SimpleNamespace
    from paintapp.drawing import DrawingController

    layer = Layer("v", (40, 40), (0, 0, 0, 0))
    layer.vector = True
    layer.strokes.append(vector.new_stroke([[0, 0], [20, 20]], 3, "#000000"))

    class Tools:
        vector_tool = False
        vector_mode = "draw"

        def set_vector_mode(self, mode):
            self.vector_mode = mode

    class App:
        tools = Tools()
        current = SimpleNamespace(layers=[layer])
        layer_panel = None

    controller = object.__new__(DrawingController)
    controller.app = App()
    controller._sync_vector_selection = lambda: None
    controller._update_vector_overlay = lambda: None
    controller._vector_cur = None
    controller._vector_sel = None
    controller._vector_selection = []
    controller._vector_anchor = None
    check("selection succeeds", controller.select_vector_item(layer, 0), True)
    check("selection opens edit mode",
          controller.app.tools.vector_mode, "edit")


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
