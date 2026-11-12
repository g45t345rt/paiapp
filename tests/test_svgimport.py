"""Unit tests for the SVG vector importer (run: .venv\\Scripts\\python.exe tests\\test_svgimport.py)."""

import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import svgimport, vector  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def first(text):
    return svgimport.parse(text)["strokes"][0]


def test_path_mlz_imports_points():
    spec = first('<svg><path d="M 10 10 L 90 10 L 90 90 Z" '
                 'fill="none" stroke="#ff0000" stroke-width="4"/></svg>')
    check("anchors imported", spec["points"], [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0]])
    check("path closed", spec["closed"], True)
    check("stroke colour", spec["color"], "#ff0000")
    check("stroke width", spec["width"], 4)


def test_cubic_handles_imported():
    spec = first('<svg><path d="M 0 0 C 0 100 100 100 100 0" '
                 'fill="none" stroke="black"/></svg>')
    check("curve anchors", spec["points"], [[0.0, 0.0], [100.0, 0.0]])
    check("out handle", spec["handles_out"][0], [0.0, 100.0])
    check("in handle", spec["handles_in"][1], [100.0, 100.0])


def test_relative_and_shorthand_commands():
    spec = first('<svg><path d="m 10 10 l 10 0 h 10 v 10 s 10 10 20 0" '
                 'fill="none" stroke="black"/></svg>')
    check("relative anchors", spec["points"][:4],
          [[10.0, 10.0], [20.0, 10.0], [30.0, 10.0], [30.0, 20.0]])
    check("reflected control becomes a corner without a prior curve",
          spec["handles_out"][-2], [30.0, 20.0])
    check("smooth second control", spec["handles_in"][-1], [40.0, 30.0])
    check("smooth end point", spec["points"][-1], [50.0, 20.0])


def test_arc_becomes_curve():
    spec = first('<svg><path d="M 0 0 A 10 10 0 0 1 20 0" '
                 'fill="none" stroke="black"/></svg>')
    check("arc ends on target", [round(v, 3) for v in spec["points"][-1]], [20.0, 0.0])
    check("arc bows above the chord", vector.bounds(spec)[1] < -1.0, True)


def test_rect_shape():
    spec = first('<svg><rect x="5" y="6" width="30" height="20" '
                 'fill="none" stroke="black"/></svg>')
    check("rect corners", spec["points"],
          [[5.0, 6.0], [35.0, 6.0], [35.0, 26.0], [5.0, 26.0]])
    check("rect closed", spec["closed"], True)


def test_rounded_rect_has_curves():
    spec = first('<svg><rect x="0" y="0" width="20" height="20" rx="5" '
                 'fill="none" stroke="black"/></svg>')
    check("rounded rect has extra anchors", len(spec["points"]) > 4, True)


def test_ellipse_shape():
    spec = first('<svg><ellipse cx="50" cy="50" rx="20" ry="10" '
                 'fill="none" stroke="black"/></svg>')
    check("ellipse closed", spec["closed"], True)
    left, top, right, bottom = vector.bounds(spec)
    check("ellipse spans its rx", round(right - left, 1), 43.0)
    check("ellipse spans its ry", round(bottom - top, 1), 23.0)


def test_polygon_and_polyline():
    polygon = first('<svg><polygon points="0,0 10,0 10,10" '
                    'fill="none" stroke="black"/></svg>')
    polyline = first('<svg><polyline points="0,0 10,0 10,10" '
                     'fill="none" stroke="black"/></svg>')
    check("polygon closes", polygon["closed"], True)
    check("polyline stays open", polyline["closed"], False)


def test_translate_and_scale_transform():
    spec = first('<svg><g transform="translate(5,5) scale(2)">'
                 '<line x1="0" y1="0" x2="1" y2="1" stroke="black"/></g></svg>')
    check("transform applied", spec["points"], [[5.0, 5.0], [7.0, 7.0]])


def test_rotate_transform():
    spec = first('<svg><g transform="rotate(90 0 0)">'
                 '<line x1="10" y1="0" x2="20" y2="0" stroke="black"/></g></svg>')
    check("rotate x", [round(p[0], 3) for p in spec["points"]], [0.0, 0.0])
    check("rotate y", [round(p[1], 3) for p in spec["points"]], [10.0, 20.0])


def test_style_inheritance_and_override():
    inherited = first('<svg><g stroke="red" stroke-width="3">'
                      '<path d="M0 0 L 10 0" fill="none"/></g></svg>')
    check("group stroke inherited", inherited["color"], "#ff0000")
    check("group width inherited", inherited["width"], 3)
    overridden = first('<svg><path d="M0 0 L 10 0" fill="none" '
                       'style="stroke:#00ff00;stroke-width:5"/></svg>')
    check("style attribute wins", overridden["color"], "#00ff00")
    check("style width wins", overridden["width"], 5)


def test_fill_only_becomes_compound_fill():
    doc = svgimport.parse('<svg><rect x="0" y="0" width="10" height="10" '
                          'fill="#123456"/></svg>')
    check("fill-only shape has no strokes", len(doc["strokes"]), 0)
    check("fill-only shape is one compound fill", len(doc["fills"]), 1)
    fill = doc["fills"][0]
    check("fill colour used", fill["color"], "#123456")
    check("fill has one contour",
          len(vector.fill_contour_paths(fill)), 1)


def test_hidden_elements_skipped():
    doc = svgimport.parse('<svg><rect display="none" width="10" height="10"/>'
                          '<rect visibility="hidden" width="10" height="10"/>'
                          '<rect width="10" height="10" fill="black"/></svg>')
    total = len(doc["strokes"]) + len(doc["fills"])
    check("hidden shapes dropped", total, 1)


def test_viewbox_scales_to_viewport():
    spec = first('<svg viewBox="0 0 10 10" width="100" height="100">'
                 '<line x1="1" y1="1" x2="2" y2="2" stroke="black"/></svg>')
    check("viewBox scale applied", spec["points"], [[10.0, 10.0], [20.0, 20.0]])


def test_multiple_subpaths_and_mirror_colors():
    doc = svgimport.parse('<svg><path d="M0 0 L10 0 M0 5 L10 5" '
                          'fill="none" stroke="black"/></svg>')
    check("each subpath is a stroke", len(doc["strokes"]), 2)


def test_opacity_maps_to_spec():
    spec = first('<svg><path d="M0 0 L10 0" fill="none" stroke="black" '
                 'stroke-opacity="0.5" opacity="0.5"/></svg>')
    check("opacities multiply", round(spec["opacity"], 3), 0.25)


def test_fit_to_centres_within_canvas():
    strokes = [vector.new_stroke([[0, 0], [100, 0]], 2, "#000000")]
    placed, _ = svgimport.fit_to(strokes, (50, 50))
    points = placed[0]["points"]
    check("scaled to fit width", round(points[1][0] - points[0][0], 1), 50.0)
    check("centred horizontally", round(points[0][0], 1), 0.0)


def test_fit_to_never_enlarges():
    strokes = [vector.new_stroke([[0, 0], [10, 0]], 2, "#000000")]
    placed, _ = svgimport.fit_to(strokes, (500, 500))
    check("small art is only centred", round(placed[0]["points"][1][0] - placed[0]["points"][0][0], 1), 10.0)


def test_load_reads_a_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "art.svg")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('<svg><line x1="0" y1="0" x2="4" y2="4" '
                         'stroke="black"/></svg>')
        doc = svgimport.load(path)
    check("file parsed", len(doc["strokes"]), 1)


def test_invalid_svg_raises():
    try:
        svgimport.parse("not svg at all")
        check("invalid svg raises", False, True)
    except ValueError:
        check("invalid svg raises", True, True)
    try:
        svgimport.parse("<html><body/></html>")
        check("non-svg root raises", False, True)
    except ValueError:
        check("non-svg root raises", True, True)


def test_export_round_trips_geometry():
    spec = vector.new_stroke(
        [[10, 10], [90, 10]], 4, "#ff0000",
        handles_out=[[10, 60], [90, 10]], handles_in=[[10, 10], [90, 60]],
        closed=True)
    text = svgimport.export([spec], size=(100, 100))
    doc = svgimport.parse(text)
    check("one stroke exported", len(doc["strokes"]), 1)
    back = doc["strokes"][0]
    check("anchors round-trip",
          [[round(c, 1) for c in p] for p in back["points"]],
          [[10.0, 10.0], [90.0, 10.0]])
    check("closed round-trips", back["closed"], True)
    check("colour round-trips", back["color"], "#ff0000")
    check("width round-trips", back["width"], 4)
    check("out handle round-trips",
          [round(c, 1) for c in back["handles_out"][0]], [10.0, 60.0])


def test_export_straight_uses_lines():
    spec = vector.new_stroke([[0, 0], [10, 0], [10, 10]], 2, "#000000")
    data = svgimport.path_data(spec)
    check("straight segments are lines", "C" not in data, True)
    check("starts at the first anchor", data.startswith("M 0 0"), True)


def test_export_fill_and_opacity():
    spec = vector.new_stroke([[0, 0], [10, 0], [10, 10]], 3, "#000000",
                             closed=True, fill="#00ff00", opacity=0.5)
    text = svgimport.export([spec], size=(20, 20))
    check("fill written", 'fill="#00ff00"' in text, True)
    check("opacity written", 'opacity="0.5"' in text, True)


def test_export_import_preserves_fill():
    spec = vector.new_stroke([[0, 0], [10, 0], [10, 10]], 3, "#000000",
                             closed=True, fill="#00ff00")
    text = svgimport.export([spec], size=(20, 20))
    back = svgimport.parse(text)["strokes"][0]
    check("fill round-trips", back["fill"], "#00ff00")


def test_export_viewport_hugs_geometry():
    strokes = [vector.new_stroke([[10, 20], [40, 50]], 2, "#000000")]
    text = svgimport.export(strokes)
    check("viewBox generated", 'viewBox="' in text, True)
    check("round-trips through the parser", len(svgimport.parse(text)["strokes"]), 1)


def test_imported_strokes_composite():
    from paintapp.document import Canvas  # noqa: E402

    doc = svgimport.parse('<svg><line x1="0" y1="0" x2="30" y2="30" '
                          'stroke="black"/></svg>')
    canvas = Canvas("c", (64, 64))
    layer = canvas.add_layer(name="svg", vector=True)
    layer.strokes, layer.fills = svgimport.fit_to(doc["strokes"], (64, 64),
                                                  fills=doc["fills"])
    layer.invalidate_vectors()
    composed = layer.composited_image()
    check("imported art composites", composed.getchannel("A").getbbox() is not None, True)


def test_even_odd_fill_keeps_holes():
    from paintapp.document import Canvas  # noqa: E402

    doc = svgimport.parse(
        '<svg viewBox="0 0 10 10" fill-rule="evenodd">'
        '<path fill="#000000" d="M0 0 H10 V10 H0 Z M3 3 H7 V7 H3 Z"/></svg>')
    check("one compound fill", len(doc["fills"]), 1)
    check("even-odd rule kept", doc["fills"][0]["even_odd"], True)
    canvas = Canvas("c", (10, 10))
    layer = canvas.add_layer(name="f", vector=True)
    layer.fills = doc["fills"]
    layer.invalidate_vectors()
    image = layer.composited_image()
    check("outer contour is filled", image.getpixel((1, 1))[3] > 0, True)
    check("inner contour is a hole", image.getpixel((5, 5))[3], 0)


def test_nonzero_fill_keeps_holes_on_composite():
    # Many icon packs (Bootstrap, Phosphor) draw a filled ring as one nonzero
    # path whose inner contour is wound the other way; the hole must survive.
    from paintapp.document import Canvas  # noqa: E402

    doc = svgimport.parse(
        '<svg viewBox="0 0 10 10"><path fill="#000000" '
        'd="M0 0 H10 V10 H0 Z M3 3 V7 H7 V3 Z"/></svg>')
    check("one compound fill", len(doc["fills"]), 1)
    check("nonzero rule kept", doc["fills"][0]["even_odd"], False)
    canvas = Canvas("c", (10, 10))
    layer = canvas.add_layer(name="f", vector=True)
    layer.fills = doc["fills"]
    layer.invalidate_vectors()
    image = layer.composited_image()
    check("outer contour is filled", image.getpixel((1, 1))[3] > 0, True)
    check("inner contour is a hole", image.getpixel((5, 5))[3], 0)


def test_export_round_trips_compound_fill():
    svg = ('<svg viewBox="0 0 10 10" fill-rule="evenodd"><path fill="#112233" '
           'd="M0 0 H10 V10 H0 Z M3 3 H7 V7 H3 Z"/></svg>')
    doc = svgimport.parse(svg)
    text = svgimport.export([], fills=doc["fills"], size=(10, 10))
    check("fill-rule written", 'fill-rule="evenodd"' in text, True)
    check("fill colour written", 'fill="#112233"' in text, True)
    back = svgimport.parse(text)
    check("fill round-trips", len(back["fills"]), 1)
    check("both contours round-trip", len(back["fills"][0]["contours"]), 2)


def test_current_color_stroke_resolves_to_black():
    # Lucide/Iconoir-style icons paint with``stroke="currentColor"``; the keyword
    # must resolve (to black, SVG's initial ``color``) or the icon imports empty.
    spec = first('<svg fill="none" stroke="currentColor" stroke-width="2">'
                 '<path d="M2 2 L22 22"/></svg>')
    check("currentColor stroke imports", spec["color"], "#000000")
    check("currentColor stroke width", spec["width"], 2)


def test_current_color_fill_resolves_to_black():
    doc = svgimport.parse('<svg><path fill="currentColor" '
                          'd="M0 0 H10 V10 H0 Z"/></svg>')
    check("currentColor fill is black", doc["fills"][0]["color"], "#000000")


def test_current_color_follows_color_property():
    doc = svgimport.parse('<svg color="#ff0000"><path fill="currentColor" '
                          'd="M0 0 H10 V10 H0 Z"/></svg>')
    check("currentColor follows color", doc["fills"][0]["color"], "#ff0000")


def test_stroke_only_icon_composites_ink():
    from paintapp.document import Canvas  # noqa: E402

    doc = svgimport.parse('<svg fill="none" stroke="currentColor" '
                          'stroke-width="2"><path d="M2 2 L22 22"/></svg>')
    canvas = Canvas("c", (24, 24))
    layer = canvas.add_layer(name="svg", vector=True)
    layer.strokes, layer.fills = svgimport.fit_to(
        doc["strokes"], (24, 24), fills=doc["fills"])
    layer.invalidate_vectors()
    check("stroke icon has ink",
          layer.composited_image().getchannel("A").getbbox() is not None, True)


def test_non_finite_coordinates_are_sanitized():
    doc = svgimport.parse('<svg viewBox="0 0 24 24"><path d="M0 0 L24e400 24" '
                          'fill="none" stroke="black"/></svg>')
    spec = doc["strokes"][0]
    check("no infinite coordinate",
          all(math.isfinite(c) for point in spec["points"] for c in point), True)


def test_huge_coordinate_is_clamped():
    doc = svgimport.parse('<svg viewBox="0 0 24 24">'
                          '<path d="M0 0 L99999999999 24" fill="none" '
                          'stroke="black"/></svg>')
    spec = doc["strokes"][0]
    check("coordinate clamped",
          spec["points"][-1][0], svgimport.MAX_COORDINATE)


def test_extreme_stroke_width_is_clamped():
    spec = first('<svg><path d="M0 0 L10 0" fill="none" stroke="black" '
                 'stroke-width="1e400"/></svg>')
    check("width clamped", spec["width"], vector.MAX_WIDTH)


def test_oversized_paste_is_scaled_to_fit():
    doc = svgimport.parse(
        '<svg viewBox="0 0 1000 1000"><path d="M0 0 L1000 1000" '
        'fill="none" stroke="black" stroke-width="20"/></svg>')
    strokes, fills = svgimport.fit_within(doc["strokes"], doc["fills"], (100, 100))
    reach = max(point[0] for point in strokes[0]["points"])
    check("oversized art is scaled into the canvas", reach <= 100, True)


def test_art_already_within_paste_stays_put():
    doc = svgimport.parse('<svg viewBox="0 0 24 24"><path d="M2 2 L22 22" '
                          'fill="none" stroke="black"/></svg>')
    strokes, _fills = svgimport.fit_within(
        doc["strokes"], doc["fills"], (800, 640))
    check("in-bounds art is untouched", strokes[0]["points"][0],
          doc["strokes"][0]["points"][0])


def test_dash_array_round_trip():
    spec = first('<svg><path d="M 0 0 L 50 0" fill="none" stroke="black" '
                 'stroke-width="2" stroke-dasharray="4 2" '
                 'stroke-dashoffset="1"/></svg>')
    check("dash imported", spec["dash"], [4.0, 2.0])
    check("dash offset imported", spec["dash_offset"], 1.0)
    text = svgimport.export([spec], size=(60, 20))
    check("dash exported", 'stroke-dasharray="4 2"' in text, True)
    check("dash offset exported", 'stroke-dashoffset="1"' in text, True)
    again = svgimport.parse(text)["strokes"][0]
    check("dash round-trips", again["dash"], [4.0, 2.0])


def test_gradient_and_marker_export():
    gradient = {
        "type": "linear", "stops": [[0, "#ff0000"], [1, "#0000ff"]],
        "start": [0, 0], "end": [40, 0],
    }
    spec = vector.new_stroke(
        [[0, 0], [40, 0], [40, 40], [0, 40]], 3, "#000000", closed=True,
        fill_gradient=gradient, arrow_end="triangle")
    text = svgimport.export([spec], size=(60, 60))
    check("linear gradient defined", "<linearGradient" in text, True)
    check("gradient referenced", "url(#gradient0)" in text, True)
    check("marker defined", "<marker" in text, True)
    check("marker referenced", "marker-end=" in text, True)


def test_width_profile_exports_as_outline():
    spec = vector.new_stroke([[0, 10], [40, 10]], 4, "#000000",
                             widths=[2, 16])
    text = svgimport.export([spec], size=(60, 60))
    check("profiled stroke exported without stroke paint",
          'stroke="none"' in text, True)
    check("profiled stroke has an outline path", text.count("<path") >= 1, True)


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
