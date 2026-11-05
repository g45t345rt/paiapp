"""Unit tests for ``project.save`` progress (run: .venv\\Scripts\\python.exe tests\\test_project_save.py).

``save`` runs on the Tk thread and reports progress through a callback so the
status bar can advance while the window blocks. These tests drive it with a
stand-in app and check the fractions it reports and that the archive it writes
round-trips through :func:`project.read`.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

import project  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def fake_layer(name, color, mask=False):
    return SimpleNamespace(
        name=name,
        image=Image.new("RGBA", (4, 4), color),
        mask=Image.new("L", (4, 4), 128) if mask else None,
        visible=True,
        locked=False,
        opacity=1.0,
        offset=(0, 0),
        group=None,
        clip=False,
        blend="normal",
        filters=[],
        texts=[],
    )


def fake_asset(name, color):
    return SimpleNamespace(
        name=name,
        image=Image.new("RGBA", (3, 3), color),
        id=1,
        source="generated",
        prompt="a test",
    )


def fake_vector_layer(name):
    layer = fake_layer(name, (0, 0, 0, 0))
    layer.vector = True
    layer.strokes = [{
        "points": [[1.0, 1.0], [3.0, 3.0]], "width": 4, "color": "#ff0000",
        "opacity": 1.0, "closed": False, "cap": "round", "join": "round",
    }]
    return layer


def fake_path_layer(name):
    layer = fake_layer(name, (0, 0, 0, 0))
    layer.paths = [{
        "anchors": [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0]],
        "handles_in": [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0]],
        "handles_out": [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0]],
        "closed": False, "mode": "fill", "fill_color": "#ff0000",
        "stroke_color": "#0000ff", "width": 4, "outline": 0, "feather": 0,
        "opacity": 100, "antialias": True, "cap": "round", "join": "round",
    }]
    return layer


def fake_app(layers, assets=()):
    canvas = SimpleNamespace(
        name="Canvas 1",
        width=4,
        height=4,
        active=0,
        layer_counter=len(layers),
        group_counter=0,
        guides=[],
        groups=[],
        assets=list(assets),
        asset_counter=len(assets),
        layers=layers,
    )
    return SimpleNamespace(
        active_canvas=0,
        canvas_counter=1,
        color="#000000",
        brush_size=8,
        tools=SimpleNamespace(export_settings=lambda: {}),
        palette=["#000000"],
        selected_swatch=0,
        canvases=[canvas],
    )


def test_save_reports_progress_per_image():
    app = fake_app([
        fake_layer("A", (255, 0, 0, 255)),
        fake_layer("B", (0, 255, 0, 255), mask=True),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        seen = []
        project.save(app, path, progress=seen.append)
    check("progress starts at zero", seen[0], 0.0)
    check("progress ends complete", seen[-1], 1.0)
    check("one step per image", len(seen), 4)
    check(
        "progress never goes backwards",
        all(later >= earlier for earlier, later in zip(seen, seen[1:])),
        True,
    )


def test_save_round_trips_through_read():
    app = fake_app([fake_layer("A", (1, 2, 3, 255))])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        manifest = project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
    check("manifest format", manifest["format"], project.FORMAT)
    check("image decoded", (0, 0) in payload["images"], True)


def test_save_persists_canvas_assets():
    app = fake_app([fake_layer("A", (1, 2, 3, 255))], assets=[fake_asset("gen", (9, 9, 9, 255))])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        manifest = project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
    check("asset recorded in manifest", len(manifest["canvases"][0]["assets"]), 1)
    check("asset decoded", (0, 0) in payload["assets"], True)
    check(
        "asset keeps its name",
        manifest["canvases"][0]["assets"][0]["name"],
        "gen",
    )


def test_save_persists_vector_strokes():
    app = fake_app([fake_vector_layer("V")])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "vector.pai")
        manifest = project.save(app, path)
    entry = manifest["canvases"][0]["layers"][0]
    check("vector layer flagged", entry["vector"], True)
    check("stroke persisted", len(entry["strokes"]), 1)
    check("stroke points persisted", entry["strokes"][0]["points"][1], [3.0, 3.0])


def test_save_persists_curve_geometry_and_fill():
    layer = fake_vector_layer("V")
    layer.strokes = [{
        "points": [[1.0, 1.0], [5.0, 5.0]],
        "handles_in": [[1.0, 1.0], [5.0, 1.0]],
        "handles_out": [[1.0, 5.0], [5.0, 5.0]],
        "width": 4, "color": "#123456", "opacity": 1.0, "closed": True,
        "cap": "round", "join": "round", "fill": "#00ff00",
    }]
    app = fake_app([layer])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "curve.pai")
        manifest = project.save(app, path)
    stroke = manifest["canvases"][0]["layers"][0]["strokes"][0]
    check("curve out handle persisted", stroke["handles_out"][0], [1.0, 5.0])
    check("curve in handle persisted", stroke["handles_in"][1], [5.0, 1.0])
    check("fill persisted", stroke["fill"], "#00ff00")
    check("closed persisted", stroke["closed"], True)


def test_save_persists_fills():
    layer = fake_vector_layer("V")
    layer.strokes = []
    layer.fills = [{
        "contours": [{
            "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]],
            "handles_in": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]],
            "handles_out": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]],
        }],
        "color": "#abcdef", "opacity": 1.0, "antialias": True,
        "even_odd": True,
    }]
    app = fake_app([layer])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fills.pai")
        manifest = project.save(app, path)
    entry = manifest["canvases"][0]["layers"][0]
    check("fill persisted", len(entry["fills"]), 1)
    check("fill colour persisted", entry["fills"][0]["color"], "#abcdef")


def test_save_persists_paths():
    app = fake_app([fake_path_layer("P")])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "path.pai")
        manifest = project.save(app, path)
    entry = manifest["canvases"][0]["layers"][0]
    check("path persisted", len(entry["paths"]), 1)
    check("path anchors persisted", entry["paths"][0]["anchors"][2], [3.0, 3.0])
    check("path mode persisted", entry["paths"][0]["mode"], "fill")


def test_save_empty_project_still_reports_done():
    app = fake_app([])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.pai")
        seen = []
        project.save(app, path, progress=seen.append)
    check("empty project reports done", seen, [0.0])


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
