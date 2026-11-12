"""Regression test for the Model Settings component pool folder fallback.

Run: .venv\\Scripts\\python.exe tests\\test_modelsettings_pools.py

A file downloaded into a pool's engine folder must be selectable in a model
row's component slot even when it was never curated into the pool (e.g. a Hub
download whose pool additions were not saved). Covers the VAE slot of the
MiniMax H3 Image family.
"""

import os
import sys
import tempfile
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import paths  # noqa: E402
from paintapp import modelsettings as ms  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _dialog(pools):
    dialog = object.__new__(ms.ModelSettingsDialog)
    dialog.pools = {name: [] for _title, name in ms._POOL_CATEGORIES}
    dialog.pools.update(pools)
    dialog.entries = []
    dialog._kinds_ready = True
    return dialog


def test_folder_vae_is_listed_in_slot():
    with tempfile.TemporaryDirectory() as folder:
        vae = os.path.join(folder, "minimax_h3_image_vae_fp16.safetensors")
        open(vae, "wb").close()
        with mock.patch.object(
            paths, "component_path", return_value=folder,
        ):
            dialog = _dialog({})
            items = dialog._pool_items("vaes")
            check("folder file listed", [item["path"] for item in items], [vae])

            kind = ms.engine_kinds.SLOT_KIND[("h3_image", "vae")]
            choices, mapping = dialog._slot_file_choices("vaes", kind, None)
            check("slot lists file",
                  mapping.get("minimax_h3_image_vae_fp16  (MiniMax H3 Image VAE)"),
                  vae)
            check("default is first", choices[0], "(default)")


def test_curated_entry_wins_over_folder_file():
    with tempfile.TemporaryDirectory() as folder:
        vae = os.path.join(folder, "minimax_h3_image_vae_fp16.safetensors")
        open(vae, "wb").close()
        curated = {"label": "Curated", "path": vae}
        with mock.patch.object(
            paths, "component_path", return_value=folder,
        ):
            dialog = _dialog({"vaes": [curated]})
            items = dialog._pool_items("vaes")
            check("one entry per path", len(items), 1)
            check("curated label kept", items[0]["label"], "Curated")


def test_detection_pass_includes_folder_file():
    with tempfile.TemporaryDirectory() as folder:
        vae = os.path.join(folder, "minimax_h3_image_vae_fp16.safetensors")
        open(vae, "wb").close()
        with mock.patch.object(
            paths, "component_path", return_value=folder,
        ):
            dialog = _dialog({})
            pending = dialog._pending_detections()
            check("folder file detected",
                  (vae, ms.engine_kinds.VAE) in pending, True)


def test_wizard_result_creates_row_with_components():
    dialog = _dialog({})
    for name in (
        "_finish", "_new_pool_row", "_refresh_pool_count", "_expand_pool",
        "_new_entry_row", "_refresh_entry_count", "_expand_group",
        "_remake_entry_row",
    ):
        setattr(dialog, name, mock.Mock(return_value=None))
    dialog._pools_canvas = None
    dialog._entries_canvas = None

    with tempfile.TemporaryDirectory() as folder:
        model = os.path.join(folder, "flux1-dev-Q4_K_S.gguf")
        clip = os.path.join(folder, "clip_l.safetensors")
        vae = os.path.join(folder, "flux_vae.safetensors")
        for path in (model, clip, vae):
            open(path, "wb").close()
        payload = {
            "family": "flux", "group": "Flux", "label": "FLUX.1 dev",
            "path": model,
            "components": {"clip": clip, "vae": vae},
            "warnings": [],
        }
        dialog._on_wizard_ready(payload)
        check("one entry", len(dialog.entries), 1)
        entry = dialog.entries[0]
        check("entry path", entry["path"], model)
        check("entry family", entry["family"], "flux")
        check("entry components", entry["components"],
              {"clip": clip, "vae": vae})
        check("clip curated",
              [item["path"] for item in dialog.pools["text_encoders"]], [clip])
        check("vae curated",
              [item["path"] for item in dialog.pools["vaes"]], [vae])

        dialog._on_wizard_ready(payload)
        check("no duplicate entry", len(dialog.entries), 1)
        check("no duplicate pool item",
              len(dialog.pools["text_encoders"]), 1)
        check("existing row rebuilt",
              dialog._remake_entry_row.called, True)


def test_wizard_remote_fibo_adds_repo_entry():
    dialog = _dialog({})
    for name in (
        "_finish", "_new_pool_row", "_refresh_pool_count", "_expand_pool",
        "_new_entry_row", "_refresh_entry_count", "_expand_group",
        "_remake_entry_row",
    ):
        setattr(dialog, name, mock.Mock(return_value=None))
    dialog._pools_canvas = None
    dialog._entries_canvas = None

    payload = {
        "family": "fibo", "group": "Fibo 2.0", "label": "Fibo 2.0 Dev",
        "path": "briaai/Fibo-2.0-dev", "components": {}, "warnings": [],
    }
    dialog._on_wizard_ready(payload)
    check("fibo entry", len(dialog.entries), 1)
    check("fibo repo id", dialog.entries[0]["path"], "briaai/Fibo-2.0-dev")
    check("fibo no components", dialog.entries[0]["components"], {})


def test_tool_wizard_result_curates_pools():
    dialog = _dialog({})
    for name in (
        "_finish", "_new_pool_row", "_refresh_pool_count", "_expand_pool",
    ):
        setattr(dialog, name, mock.Mock(return_value=None))
    dialog._pools_canvas = None

    with tempfile.TemporaryDirectory() as folder:
        background = os.path.join(folder, "birefnet-general.safetensors")
        upscale = os.path.join(folder, "RealESRGAN_x4.pth")
        for path in (background, upscale):
            open(path, "wb").close()
        payload = {
            "tool": "background", "label": "BiRefNet (general)",
            "path": background,
            "files": [
                {"path": background, "pool": "background",
                 "label": "BiRefNet (general)"},
                {"path": upscale, "pool": "upscale",
                 "label": "Real-ESRGAN x4"},
            ],
            "warnings": [],
        }
        dialog._on_tool_wizard_ready(payload)
        check("background tool curated",
              [item["path"] for item in dialog.pools["background"]],
              [background])
        check("upscale tool curated",
              [item["path"] for item in dialog.pools["upscale"]],
              [upscale])
        check("tool pool row added", dialog._new_pool_row.call_count, 2)

        dialog._on_tool_wizard_ready(payload)
        check("no duplicate background item",
              len(dialog.pools["background"]), 1)
        check("no duplicate upscale item", len(dialog.pools["upscale"]), 1)


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