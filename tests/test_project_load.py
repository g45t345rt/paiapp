"""Unit tests for the project load split (run: .venv\\Scripts\\python.exe tests\\test_project_load.py).

``project.read`` decodes a ``.pai`` archive on the job queue's worker thread, so
these tests drive it with a stand-in app (it only reads ``app.canvases`` for a
fallback size when loading a version 1 file) and check the decoded payload, the
progress it reports, and that a cancelled job stops instead of decoding on.
"""

import io
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
from engine.cancel import CancelToken, Cancelled  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def fake_app():
    """A stand-in app: ``read`` only needs a ``canvases`` list to fall back on."""
    return SimpleNamespace(canvases=[])


def build_project(path, layer_specs):
    """Write a minimal one-canvas ``.pai`` archive holding ``layer_specs``.

    Each spec is ``{"name": str, "mask": bool}``; every layer gets a 4x4 RGBA
    PNG and, when asked, a 4x4 ``L`` mask, matching what ``project.save`` emits.
    """
    manifest = {
        "format": project.FORMAT,
        "version": project.VERSION,
        "canvas_counter": 1,
        "active_canvas": 0,
        "canvases": [{
            "name": "Canvas 1",
            "width": 4,
            "height": 4,
            "active": 0,
            "layer_counter": len(layer_specs),
            "layers": [],
        }],
    }
    entry = manifest["canvases"][0]
    with zipfile.ZipFile(path, "w") as zf:
        for index, spec in enumerate(layer_specs):
            stem = f"canvases/0000/layers/{index:04d}_{spec['name']}"
            image = Image.new("RGBA", (4, 4), (index, 0, 0, 255))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            zf.writestr(f"{stem}.png", buf.getvalue())
            layer_entry = {"name": spec["name"], "file": f"{stem}.png"}
            if spec.get("mask"):
                mask_buf = io.BytesIO()
                Image.new("L", (4, 4), 128).save(mask_buf, format="PNG")
                zf.writestr(f"{stem}_mask.png", mask_buf.getvalue())
                layer_entry["mask"] = f"{stem}_mask.png"
            entry["layers"].append(layer_entry)
        zf.writestr(project.MANIFEST, json.dumps(manifest))
    return manifest


def test_read_decodes_layers_and_masks():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        build_project(path, [{"name": "A"}, {"name": "B", "mask": True}])
        payload = project.read(fake_app(), path)
    check("one canvas entry", len(payload["canvas_entries"]), 1)
    check("images decoded", sorted(payload["images"]), [(0, 0), (0, 1)])
    check("masks decoded", sorted(payload["masks"]), [(0, 1)])
    check("image mode", payload["images"][(0, 0)].mode, "RGBA")
    check("mask mode", payload["masks"][(0, 1)].mode, "L")
    check("manifest carried", payload["manifest"]["format"], project.FORMAT)


def test_read_reports_progress_per_image():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        build_project(path, [{"name": "A"}, {"name": "B", "mask": True}])
        seen = []
        project.read(fake_app(), path, progress=lambda f, label: seen.append((f, label)))
    check("progress starts at zero", seen[0], (0.0, "reading project"))
    check("progress ends complete", seen[-1][0], 1.0)
    check(
        "one step per image",
        [label for _frac, label in seen],
        ["reading project", "layer 1/3", "layer 2/3", "layer 3/3"],
    )
    check(
        "progress never goes backwards",
        all(later >= earlier for (earlier, _), (later, _) in zip(seen, seen[1:])),
        True,
    )


def test_read_completes_a_project_with_no_layers():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.pai")
        build_project(path, [])
        seen = []
        project.read(fake_app(), path, progress=lambda f, label: seen.append((f, label)))
    check("empty project reports done", seen, [(0.0, "reading project"), (1.0, "")])


def test_read_stops_when_cancelled():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.pai")
        build_project(path, [{"name": "A"}, {"name": "B"}])
        token = CancelToken()
        token.cancel()
        try:
            project.read(fake_app(), path, cancel=token)
        except Cancelled:
            check("cancelled read raises", True, True)
        else:
            check("cancelled read raises", False, True)


def test_read_rejects_a_foreign_archive():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "other.pai")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(project.MANIFEST, json.dumps({"format": "nope"}))
        try:
            project.read(fake_app(), path)
        except ValueError:
            check("foreign archive rejected", True, True)
        else:
            check("foreign archive rejected", False, True)


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
