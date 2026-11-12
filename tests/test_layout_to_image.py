"""Unit tests for the MOSS layout-to-image builder (paintapp.layout_to_image).

Run: .venv\\Scripts\\python.exe tests\\test_layout_to_image.py

The pure conversion half and :class:`LayoutModel` always run. The editor half
builds the real widget tree on a withdrawn Tk root; it is skipped (with a note)
when no display is available.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import layout_to_image as layout  # noqa: E402

FAILED = []
SKIPPED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_empty_builds_none():
    check("no rows", layout.build_layout([], "cap", (1024, 1024), (1024, 1024)), None)
    check(
        "rows without boxes",
        layout.build_layout(
            [{"label": "object", "caption": "x"}], "cap", (1024, 1024), (1024, 1024)
        ),
        None,
    )
    check(
        "tiny box skipped",
        layout.build_layout(
            [{"label": "object", "bbox": [10, 10, 12, 12]}], "cap",
            (1024, 1024), (1024, 1024),
        ),
        None,
    )


def test_bbox_is_abs_centralized_yxyx():
    rows = [{"label": "text", "caption": "title", "text": "EXPLORE",
             "bbox": [320, 64, 704, 192]}]
    out = layout.build_layout(rows, "cap", (1024, 1024), (1024, 1024))
    item = out["items"][0]
    check("bbox yxyx", item["bbox"], [-448, -192, -320, 192])
    check("text kept", item["text"], "EXPLORE")
    check("coord space", out["bbox_coord_space"], "abs_centralized")
    check("bbox format", out["bbox_format"], "yxyx")


def test_text_only_for_text_labels():
    rows = [{"label": "object", "caption": "c", "text": "EXPLORE",
             "bbox": [0, 0, 100, 100]}]
    out = layout.build_layout(rows, "cap", (1024, 1024), (1024, 1024))
    check("object drops text", "text" in out["items"][0], False)


def test_scales_to_output_size():
    rows = [{"label": "object", "bbox": [512, 0, 1024, 512]}]
    out = layout.build_layout(rows, "cap", (1024, 1024), (512, 512))
    check("scaled bbox", out["items"][0]["bbox"], [-256, 0, 0, 256])


def test_layout_prompt_has_meta():
    rows = [{"label": "object", "bbox": [0, 0, 100, 100]}]
    text = layout.layout_prompt(rows, "A scene", ["poster", ""],
                                (1024, 1024), (1024, 1024))
    doc = json.loads(text)
    check("caption", doc["summary"]["caption"], "A scene")
    check("meta drops blanks", doc["summary"]["meta"], ["poster"])
    check("one item", len(doc["items"]), 1)


def test_layout_prompt_none_without_items():
    check("no items", layout.layout_prompt([], "cap", [], (1024, 1024), (1024, 1024)), None)


def test_bbox_tags_order_and_character_ids():
    rows = [
        {"label": "person", "name": "barista", "caption": "A barista.",
         "bbox": [100, 100, 300, 900]},
        {"label": "text", "text": "HELLO", "caption": "black text",
         "bbox": [50, 20, 250, 80]},
        {"label": "panel", "caption": "Top-left panel.", "bbox": [0, 0, 500, 500]},
        {"label": "object", "caption": "A cup.", "bbox": [600, 600, 900, 900]},
        {"label": "person", "name": "barista", "caption": "The same barista.",
         "bbox": [350, 100, 600, 900]},
    ]
    text = layout.build_bbox(rows, "A comic", [], (1000, 1000))
    check("bbox lines", text.split("\n"), [
        "A comic",
        "p[0,0,500,500] Top-left panel.",
        "pe:1[100,100,300,900] A barista.",
        "pe:1[350,100,600,900] The same barista.",
        "o[600,600,900,900] A cup.",
        't[50,20,250,80]"HELLO" black text',
    ])


def test_bbox_scales_and_escapes():
    rows = [{"label": "panel", "bbox": [0, 0, 250, 500]},
            {"label": "text", "text": 'He said "hi"\nbye', "bbox": [0, 0, 250, 500]}]
    text = layout.build_bbox(rows, "", [], (500, 1000))
    check("scaled box", text.split("\n")[0], "p[0,0,500,500]")
    check("escaped text", text.split("\n")[1], 't[0,0,500,500]"He said \\"hi\\"\\nbye"')


def test_bbox_headers_and_fallbacks():
    rows = [{"label": "mystery", "caption": "thing", "bbox": [0, 0, 100, 100]}]
    text = layout.build_bbox(
        rows, "", ["@comic; Digital illustration", "~A bare room", "loose"], (1000, 1000))
    check("headers", text.split("\n"), [
        "@comic; Digital illustration",
        "~A bare room",
        "o[0,0,100,100] thing",
    ])


def test_bbox_none_without_items():
    check("bbox no items", layout.build_bbox([], "cap", [], (1024, 1024)), None)


def test_layout_prompt_dispatches_by_family():
    rows = [{"label": "panel", "caption": "P", "bbox": [0, 0, 500, 500]}]
    moss = layout.layout_prompt(rows, "cap", [], (1000, 1000), (1000, 1000),
                                family="moss_image")
    check("moss is json", json.loads(moss)["bbox_format"], "yxyx")
    krea2 = layout.layout_prompt(rows, "cap", [], (1000, 1000), (1000, 1000),
                                 family="krea2")
    check("krea2 is dsl", krea2, "cap\np[0,0,500,500] P")


def test_family_labels_present():
    check("krea2 label", "panel" in layout.FAMILY_LABELS["krea2"], True)
    check("krea2 supported", "krea2" in layout.SUPPORTED_FAMILIES, True)


def test_model_round_trip_and_prompt():
    model = layout.LayoutModel()
    model.enabled = True
    row = model.add_box([320, 64, 704, 192], label="text", caption="title",
                        text="EXPLORE")
    check("row added", row is not None, True)
    stored = model.settings()
    check("one stored row", len(stored), 1)
    check("stored bbox", stored[0]["bbox"], [320, 64, 704, 192])
    doc = json.loads(model.layout_text("moss_image", "A poster", (1024, 1024),
                                       (1024, 1024)))
    check("prompt caption", doc["summary"]["caption"], "A poster")
    check("prompt label", doc["items"][0]["label"], "text")
    check("prompt text", doc["items"][0]["text"], "EXPLORE")

    overlay = model.overlay_items()
    check("overlay has one item", len(overlay), 1)
    check("overlay bbox", overlay[0]["bbox"], (320, 64, 704, 192))
    check("overlay color", overlay[0]["color"], layout.PALETTE[0])
    check("overlay title is the name", overlay[0]["title"], "Item 1")

    model.rows[0]["color"] = "#ff00aa"
    check("overlay uses explicit color",
          model.overlay_items()[0]["color"], "#ff00aa")
    check("settings keep color", model.settings()[0]["color"], "#ff00aa")

    model.set_settings([{"label": "object", "caption": "m", "text": "",
                         "bbox": [368, 416, 656, 896], "name": "Item 9",
                         "color": "#00ff00"}],
                       meta=["poster"])
    rows = model.settings()
    check("reload count", len(rows), 1)
    check("reload bbox", rows[0]["bbox"], [368, 416, 656, 896])
    check("reload color", rows[0]["color"], "#00ff00")
    check("reload meta", model.meta_tags(), ["poster"])


def _build_editor(model):
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError:
        SKIPPED.append("layout editor (no display)")
        return None, None, None
    root.withdraw()
    calls = []
    editor = layout.LayoutEditor(
        root, model, supported=True,
        on_capture=lambda: (10, 10, 60, 60),
        on_change=lambda: calls.append(1),
    )
    return root, editor, calls


def test_editor_writes_back():
    model = layout.LayoutModel()
    built = _build_editor(model)
    if built[0] is None:
        return
    root, editor, calls = built
    model.enabled = True
    editor.enabled_var.set(True)
    editor._on_enabled_toggle()
    check("enable writes model", model.enabled, True)

    editor._on_add_from_selection()
    check("add from selection", len(model.rows), 1)
    check("captured bbox", model.rows[0]["bbox"], [10, 10, 60, 60])
    check("change reported", bool(calls), True)

    widgets = editor._row_widgets[0]
    widgets["name_var"].set("robot box")
    widgets["label_var"].set("text")
    widgets["text_text"].insert("1.0", "HELLO")
    widgets["text_text"].edit_modified(True)
    editor._on_text(widgets)
    widgets["caption_text"].insert("1.0", "a friendly robot")
    widgets["caption_text"].edit_modified(True)
    editor._on_caption(widgets)
    check("name written", model.rows[0]["name"], "robot box")
    check("text written", model.rows[0]["text"], "HELLO")
    check("caption written", model.rows[0]["caption"], "a friendly robot")
    root.destroy()


def test_editor_updates_in_place():
    model = layout.LayoutModel()
    model.enabled = True
    first = model.add_box((10, 10, 50, 50))
    second = model.add_box((60, 60, 100, 100))
    built = _build_editor(model)
    if built[0] is None:
        return
    root, editor, calls = built
    frame = editor._row_widgets[0]["frame"]
    label = editor._row_widgets[0]["box_label"]

    model.rows[0]["bbox"] = [20, 20, 40, 40]
    editor.refresh_boxes()
    root.update_idletasks()
    check("refresh keeps the row widgets",
          editor._row_widgets[0]["frame"] is frame, True)
    check("refresh updates the box label",
          label.cget("text"), "x 20  y 20  w 20  h 20")
    check("refresh does not report a change", len(calls), 0)

    editor._remove_row(second)
    root.update_idletasks()
    check("remove keeps the other row",
          editor._row_widgets[0]["frame"] is frame, True)
    check("remove drops the model row", len(model.rows), 1)

    added = model.add_box((200, 200, 300, 300))
    editor.add_row(added)
    root.update_idletasks()
    check("add keeps the existing row",
          editor._row_widgets[0]["frame"] is frame, True)
    check("add appends a row", len(editor._row_widgets), 2)
    check("add syncs the enable switch", editor.enabled_var.get(), True)
    root.destroy()


def test_controller_move_and_resize():
    from types import SimpleNamespace

    from paintapp.drawing import DrawingController

    model = layout.LayoutModel()
    model.enabled = True
    model.add_box((100, 100, 300, 300))
    tools = SimpleNamespace(layout_supported=lambda: True, layout_model=model)
    app = SimpleNamespace(tools=tools, current=SimpleNamespace(size=(800, 640)))
    ctrl = DrawingController(app)

    check("hit interior is move", ctrl._layout_hit(150, 150, 1.0)[1], "move")
    check("hit nw handle", ctrl._layout_hit(100, 100, 1.0)[1], "nw")
    check("hit misses", ctrl._layout_hit(500, 500, 1.0), (None, None))

    moved = ctrl._layout_adjust_box(
        (100, 100, 300, 300), "move", (150, 150), (200, 180))
    check("move shifts", moved, (150.0, 130.0, 350.0, 330.0))
    resized = ctrl._layout_adjust_box(
        (100, 100, 300, 300), "se", (300, 300), (400, 420))
    check("resize se", resized, (100.0, 100.0, 400.0, 420.0))

    model.enabled = False
    check("disabled box not hit", ctrl._layout_hit(150, 150, 1.0), (None, None))


def test_layout_overlapping_cycle():
    from types import SimpleNamespace

    from paintapp.drawing import DrawingController

    model = layout.LayoutModel()
    model.enabled = True
    bottom = model.add_box((100, 100, 300, 300))
    top = model.add_box((120, 120, 280, 280))
    tools = SimpleNamespace(layout_supported=lambda: True, layout_model=model)
    app = SimpleNamespace(tools=tools, current=SimpleNamespace(size=(800, 640)))
    ctrl = DrawingController(app)

    # A plain hit always finds the topmost box at the overlap.
    check("topmost first", ctrl._layout_hit(200, 200, 1.0), (top, "move"))
    # The first press picks the topmost; pressing the same spot again steps
    # down to the box underneath, then wraps back to the top.
    check("first press topmost",
          ctrl._layout_cycle_row(200, 200, top, "move", 1.0), top)
    check("cycle to underneath",
          ctrl._layout_cycle_row(200, 200, top, "move", 1.0), bottom)
    check("cycle wraps",
          ctrl._layout_cycle_row(200, 200, bottom, "move", 1.0), top)
    # A handle press never cycles; it resets the cycle.
    check("handle no cycle",
          ctrl._layout_cycle_row(200, 200, top, "nw", 1.0), top)
    check("cycle reset", ctrl._layout_last_pick, None)
    # A press at a new spot starts again from the topmost.
    ctrl._layout_cycle_row(200, 200, top, "move", 1.0)
    check("new spot topmost",
          ctrl._layout_cycle_row(150, 150, top, "move", 1.0), top)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    for note in SKIPPED:
        print(f"skip {note}")
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
