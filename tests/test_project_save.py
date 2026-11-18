"""Unit tests for ``project.save`` progress (run: .venv\\Scripts\\python.exe tests\\test_project_save.py).

``save`` runs on the Tk thread and reports progress through a callback so the
status bar can advance while the window blocks. These tests drive it with a
stand-in app and check the fractions it reports and that the archive it writes
round-trips through :func:`project.read`.
"""

import json
import os
import sys
import tempfile
import zipfile
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
        layers=layers,
    )
    library = SimpleNamespace(assets=list(assets), asset_counter=len(assets))
    return SimpleNamespace(
        asset_library=library,
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


def test_save_persists_project_assets():
    app = fake_app([fake_layer("A", (1, 2, 3, 255))], assets=[fake_asset("gen", (9, 9, 9, 255))])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        manifest = project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
    check("asset recorded at project level", len(manifest["assets"]), 1)
    check("no per-canvas assets", "assets" in manifest["canvases"][0], False)
    check("asset decoded", 0 in payload["assets"], True)
    check("asset keeps its name", payload["library"].assets[0].name, "gen")
    check("canvas shares library", payload["canvases"][0].library is payload["library"], True)


def test_legacy_canvas_assets_load_globally():
    """A pre-19 file with per-canvas assets loads them into one project list."""
    import json
    import zipfile
    from PIL import Image
    import io

    def png(color):
        buf = io.BytesIO()
        Image.new("RGBA", (2, 2), color).save(buf, "PNG")
        return buf.getvalue()

    manifest = {
        "format": project.FORMAT, "version": 18, "canvases": [
            {"name": "A", "width": 4, "height": 4, "layers": [], "asset_counter": 1,
             "assets": [{"id": 1, "name": "a1", "file": "canvases/0000/assets/0000_a1.png"}]},
            {"name": "B", "width": 4, "height": 4, "layers": [], "asset_counter": 1,
             "assets": [{"id": 1, "name": "b1", "file": "canvases/0001/assets/0000_b1.png"}]},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "old.pai")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(project.MANIFEST, json.dumps(manifest))
            zf.writestr("canvases/0000/assets/0000_a1.png", png((255, 0, 0, 255)))
            zf.writestr("canvases/0001/assets/0000_b1.png", png((0, 255, 0, 255)))
        payload = project.read(SimpleNamespace(canvases=[]), path)
    library = payload["library"]
    check("both legacy assets loaded", [a.name for a in library.assets], ["a1", "b1"])
    check("legacy ids made unique", len({a.id for a in library.assets}), 2)
    check("counter past ids", library.asset_counter >= max(a.id for a in library.assets), True)
    check(
        "every canvas sees all assets",
        all(doc.assets is library.assets for doc in payload["canvases"]),
        True,
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


def test_save_persists_asset_organization():
    from paintapp.document import AssetLibrary

    app = fake_app([fake_layer("A", (1, 2, 3, 255))])
    library = AssetLibrary()
    refs = library.new_group("Refs")
    library.add_asset(
        name="grouped", image=Image.new("RGBA", (3, 3)), source="generated",
        group=refs.id, tags=["portrait", "red"], favorite=True,
    )
    library.add_asset(name="loose", image=Image.new("RGBA", (3, 3)))
    library.move_group(refs)
    app.asset_library = library
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        manifest = project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
    restored = payload["library"]
    check("group counter saved", manifest["asset_group_counter"], 1)
    check("group saved", [
        (entry["id"], entry["name"]) for entry in manifest["asset_groups"]
    ], [(1, "Refs")])
    check("group restored", [
        (group.id, group.name) for group in restored.groups
    ], [(1, "Refs")])
    grouped = restored.find_by_id(1)
    loose = restored.find_by_id(2)
    check("group membership restored", grouped.group, 1)
    check("tags restored", grouped.tags, ["portrait", "red"])
    check("favorite restored", grouped.favorite, True)
    check("ungrouped asset stays ungrouped", loose.group, None)
    check("ungrouped has no tags", loose.tags, [])


def test_legacy_assets_default_to_ungrouped():
    """Assets saved before organization existed load ungrouped and untagged."""
    import json
    import zipfile
    import io

    def png(color):
        buf = io.BytesIO()
        Image.new("RGBA", (2, 2), color).save(buf, "PNG")
        return buf.getvalue()

    manifest = {
        "format": project.FORMAT, "version": 19,
        "asset_counter": 1,
        "assets": [{"id": 1, "name": "old", "file": "assets/0000_old.png"}],
        "canvases": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "old.pai")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(project.MANIFEST, json.dumps(manifest))
            zf.writestr("assets/0000_old.png", png((1, 2, 3, 255)))
        payload = project.read(SimpleNamespace(canvases=[]), path)
    asset = payload["library"].assets[0]
    check("legacy asset ungrouped", asset.group, None)
    check("legacy asset untagged", asset.tags, [])
    check("legacy asset not favorite", asset.favorite, False)
    check("legacy asset keeps id", asset.id, 1)


def test_canvas_view_round_trips():
    """A canvas camera is saved and restored so reopening keeps the viewport."""
    app = fake_app([fake_layer("A", (1, 2, 3, 255))])
    app.canvases[0].view_state = {
        "zoom": 2.5,
        "anchor": [30.0, 40.0],
        "angle": 0.5,
        "flip": True,
        "flip_v": False,
        "pivot": [100.0, 120.0],
    }
    app.canvases[0].exploded_view_state = {
        "zoom": 0.5,
        "anchor": [5.0, 6.0],
        "angle": 0.0,
        "flip": False,
        "flip_v": False,
        "pivot": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        manifest = project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
    loaded = payload["canvases"][0]
    check("stacked zoom saved", manifest["canvases"][0]["view"]["zoom"], 2.5)
    check("stacked zoom restored", loaded.view_state["zoom"], 2.5)
    check("stacked anchor restored", loaded.view_state["anchor"], [30.0, 40.0])
    check("stacked rotation restored", loaded.view_state["angle"], 0.5)
    check("stacked flip restored", loaded.view_state["flip"], True)
    check("stacked pivot restored", loaded.view_state["pivot"], [100.0, 120.0])
    check("grid zoom restored", loaded.exploded_view_state["zoom"], 0.5)
    check("grid anchor restored", loaded.exploded_view_state["anchor"], [5.0, 6.0])


def test_malformed_view_state_is_ignored():
    """A damaged camera entry falls back to the default view, not a failed load."""
    check("missing view", project._load_view_state(None), None)
    check("non-numeric zoom", project._load_view_state(
        {"zoom": "wide", "anchor": [0, 0]}
    ), None)
    check("missing anchor", project._load_view_state({"zoom": 2.0}), None)
    check("nan anchor", project._load_view_state(
        {"zoom": 2.0, "anchor": [float("nan"), 0]}
    ), None)


def test_manifest_is_written_first():
    app = fake_app([fake_layer("A", (1, 2, 3, 255))])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        project.save(app, path)
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    check("manifest is the first member", names[0], project.MANIFEST)


def test_save_keeps_previous_versions_as_backup():
    def saved_layer_name(path):
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read(project.MANIFEST))
        return manifest["canvases"][0]["layers"][0]["name"]

    first = fake_app([fake_layer("First", (255, 0, 0, 255))])
    second = fake_app([fake_layer("Second", (0, 255, 0, 255))])
    third = fake_app([fake_layer("Third", (0, 0, 255, 255))])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        project.save(first, path)
        check("no backup before the second save", os.path.exists(path + ".bak"), False)
        project.save(second, path)
        project.save(third, path)
        check("newest backup holds the previous save",
              saved_layer_name(path + ".bak"), "Second")
        check("older backup holds the save before that",
              saved_layer_name(path + ".bak.1"), "First")
        check("current file holds the latest save",
              saved_layer_name(path), "Third")


def test_failed_verification_keeps_original():
    app = fake_app([fake_layer("A", (1, 2, 3, 255))])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        project.save(app, path)
        with open(path, "rb") as handle:
            original = handle.read()

        real_verify = project._verify_archive

        def fail_verify(*_args, **_kwargs):
            raise OSError("simulated corruption")

        project._verify_archive = fail_verify
        try:
            raised = False
            try:
                project.save(app, path)
            except OSError:
                raised = True
        finally:
            project._verify_archive = real_verify

        with open(path, "rb") as handle:
            after = handle.read()
        check("save fails when verification fails", raised, True)
        check("original survives a failed save", after, original)
        check("temp file is cleaned up", os.path.exists(path + ".tmp"), False)


def test_verify_rejects_missing_member():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.pai")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(project.MANIFEST, "{}")
        raised = False
        try:
            project._verify_archive(path, ["missing.png"])
        except OSError:
            raised = True
    check("verification catches a missing member", raised, True)


def test_verify_rejects_bad_checksum():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.pai")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr(project.MANIFEST, "{}")
            zf.writestr("layer.png", b"payload")
        data = bytearray(open(path, "rb").read())
        data[data.find(b"payload")] = ord("P")
        with open(path, "wb") as handle:
            handle.write(data)
        raised = False
        try:
            project._verify_archive(path, ["layer.png"])
        except OSError:
            raised = True
    check("verification catches a bad checksum", raised, True)


def test_unchanged_layers_reuse_encoded_png():
    """A second save skips PNG encoding for layers whose content token is unchanged."""
    layer = fake_layer("A", (1, 2, 3, 255))
    layer.pixels_token = ("cache-test", object())
    app = fake_app([layer])
    real_encode = project._encode_png
    calls = []

    def counting_encode(image):
        calls.append(image)
        return real_encode(image)

    project._encode_png = counting_encode
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample.pai")
            project.save(app, path)
            check("first save encodes the layer", len(calls), 1)
            project.save(app, path)
            check("unchanged save reuses the encoded PNG", len(calls), 1)
            layer.pixels_token = ("cache-test", "changed")
            project.save(app, path)
            check("edited layer is encoded again", len(calls), 2)
    finally:
        project._encode_png = real_encode


def test_loaded_project_primes_encode_cache():
    """Opening a project seeds the save cache with the archive's PNGs."""
    app = fake_app([
        fake_layer("A", (1, 2, 3, 255)),
        fake_layer("B", (4, 5, 6, 255), mask=True),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        project.save(app, path)
        with project._encode_cache_lock:
            project._encode_cache.clear()
            project._encode_cache_size = 0
        payload = project.read(SimpleNamespace(canvases=[]), path)
        canvases = payload["canvases"]
        keys = [
            project._cache_key(canvases[0].layers[0], "layer"),
            project._cache_key(canvases[0].layers[1], "layer"),
            project._cache_key(canvases[0].layers[1], "mask"),
        ]
        with zipfile.ZipFile(path) as archive:
            expected = [
                archive.read("canvases/0000/layers/0000_A.png"),
                archive.read("canvases/0000/layers/0001_B.png"),
                archive.read("canvases/0000/layers/0001_B_mask.png"),
            ]
    for index, (key, want) in enumerate(zip(keys, expected)):
        check(f"loaded image {index} primed", project._cache_get(key), want)


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
