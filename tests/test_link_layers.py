"""Unit tests for Link Layers (run: .venv\\Scripts\\python.exe tests\\test_link_layers.py).

A Link Layer renders a *master* layer's content while keeping its own position
and appearance, is read-only, survives save/load, and can be unlinked into an
independent copy. These tests drive the document model and the ``PaintApp``
actions directly (no Tk root) plus a project save/load round trip.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

import project  # noqa: E402
from paintapp.app import PaintApp  # noqa: E402
from paintapp.document import Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def make_doc(size=(4, 4)):
    doc = Canvas("Canvas", size)
    doc.add_layer(name="Layer 1")
    return doc


def promote(doc, index=0, master_id=1):
    master = doc.layers[index]
    master.is_master = True
    master.master_id = master_id
    return master


def link_to(doc, master, index=None, offset=(0, 0)):
    link = doc.add_layer(name="link", index=index)
    link.link_id = master.master_id
    link.master = master
    link.offset = offset
    return link


def fake_app(doc):
    return SimpleNamespace(
        current=doc,
        canvases=[doc],
        active=doc.active,
        root=None,
        active_layer=doc.active_layer,
        _push_undo=lambda *_a, **_k: None,
        _refresh_layer_list=lambda *_a, **_k: None,
        _group_sel=None,
        set_result=lambda *_a, **_k: None,
        render=lambda *_a, **_k: None,
        _activate_canvas=lambda *_a, **_k: None,
        _on_layer_select=lambda *_a, **_k: None,
        _transform_mode_active=lambda *_a, **_k: False,
        drawing=SimpleNamespace(sync_transform=lambda *_a, **_k: None),
    )


def make_project_app():
    """A stand-in coordinator good enough for :func:`project.snapshot`/``apply``."""
    app = SimpleNamespace(
        canvases=[],
        active_canvas=0,
        canvas_counter=0,
        master_counter=0,
        color="#000000",
        brush_size=8,
        tools=SimpleNamespace(
            export_settings=lambda: {}, apply_settings=lambda _s: None
        ),
        palette=["#000000"],
        selected_swatch=0,
        _text_style=None,
        _loading_project=False,
        guides_shared=False,
        shared_guides=[],
        set_color=lambda *_a, **_k: None,
    )

    def new_canvas(name=None, size=(4, 4), fill=(0, 0, 0, 0), layers=1):
        doc = Canvas(name or f"Canvas {len(app.canvases) + 1}", size)
        for index in range(layers):
            doc.add_layer(fill=fill if index == 0 else (0, 0, 0, 0))
        app.canvases.append(doc)
        return doc

    app._new_canvas = new_canvas
    app._relink_masters = lambda: PaintApp._relink_masters(app)
    return app


def test_link_renders_master_content():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False  # hidden master still supplies the link's content
    link = link_to(doc, master)

    flat = doc.composite_transparent()
    check("link shows master pixel alpha", flat.getpixel((1, 1))[3], 255)
    check("link shows master pixel color", flat.getpixel((1, 1))[:3], (255, 0, 0))


def test_link_offset_is_per_instance():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((0, 0), fill=(0, 255, 0, 255))
    master.visible = False
    link_to(doc, master, offset=(2, 2))

    flat = doc.composite_transparent()
    check("content moved by link offset", flat.getpixel((2, 2))[3], 255)
    check("origin is empty", flat.getpixel((0, 0))[3], 0)


def test_master_edit_updates_link():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    link_to(doc, master)

    master.draw.point((1, 1), fill=(0, 0, 255, 255))
    master.invalidate_filters()
    flat = doc.composite_transparent()
    check("link follows master edit", flat.getpixel((1, 1))[:3], (0, 0, 255))


def test_link_is_locked_master_is_not():
    doc = make_doc()
    master = promote(doc)
    link = link_to(doc, master)
    check("link is locked", link.is_locked(), True)
    check("master is editable", master.is_locked(), False)


def test_remove_master_unlinks_and_demotes():
    import paintapp.app as app_module

    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    link = link_to(doc, master)
    doc.active = doc.layers.index(master)

    calls = {}

    class FakeDialogs:
        def ask_remove_master(self, root, name, links):
            calls["links"] = links
            return True

        def ask_yes_no(self, *args, **kwargs):
            return True

    fake = fake_app(doc)
    fake._links_to_master = lambda mid: PaintApp._links_to_master(fake, mid)
    fake._unlink_links_to = lambda mid, m: PaintApp._unlink_links_to(fake, mid, m)
    original = app_module.dialogs
    app_module.dialogs = FakeDialogs()
    try:
        PaintApp.remove_master(fake)
    finally:
        app_module.dialogs = original

    check("master demoted", master.is_master, False)
    check("master id cleared", master.master_id, None)
    check("one link listed", len(calls["links"]), 1)
    check("link unlinked", link.link_id, None)
    check("link keeps pixels", link.image.getpixel((1, 1))[:3], (255, 0, 0))


def test_remove_master_without_links():
    import paintapp.app as app_module

    doc = make_doc()
    master = promote(doc)
    doc.active = doc.layers.index(master)

    asked = {"yes_no": 0}

    class FakeDialogs:
        def ask_remove_master(self, root, name, links):
            asked["links"] = links
            return True

        def ask_yes_no(self, *args, **kwargs):
            asked["yes_no"] += 1
            return True

    fake = fake_app(doc)
    fake._links_to_master = lambda mid: PaintApp._links_to_master(fake, mid)
    fake._unlink_links_to = lambda mid, m: PaintApp._unlink_links_to(fake, mid, m)
    original = app_module.dialogs
    app_module.dialogs = FakeDialogs()
    try:
        PaintApp.remove_master(fake)
    finally:
        app_module.dialogs = original

    check("no link dialog", asked.get("links"), None)
    check("plain confirm shown", asked["yes_no"], 1)
    check("master demoted without links", master.is_master, False)


def test_unlink_makes_independent_copy():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    link = link_to(doc, master)
    doc.active = doc.layers.index(link)

    PaintApp.unlink_layer(fake_app(doc))

    check("link id cleared", link.link_id, None)
    check("no longer a link", link.is_link, False)
    check("owns the pixels", link.image.getpixel((1, 1))[:3], (255, 0, 0))
    check("unlocked after unlink", link.is_locked(), False)

    # Editing the old master must no longer change the unlinked layer.
    master.draw.point((1, 1), fill=(0, 0, 0, 0))
    check("independent after unlink", link.image.getpixel((1, 1))[:3], (255, 0, 0))


def test_relink_after_restore():
    doc = make_doc()
    master = promote(doc)
    link_to(doc, master)
    snapshot = doc.snapshot()

    doc.restore(snapshot)
    restored_link = doc.layers[1]
    check("master unresolved after restore", restored_link.master, None)

    PaintApp._relink_masters(SimpleNamespace(canvases=[doc]))
    check("master re-resolved", restored_link.master is doc.layers[0], True)


def test_link_per_instance_transform():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    link = link_to(doc, master)
    # Translate the instance by (1, 1): content (1, 1) -> (2, 2).
    link.set_transform((1.0, 0.0, 0.0, 1.0, 1.0, 1.0))

    flat = doc.composite_transparent()
    check("transformed link pixel", flat.getpixel((2, 2))[3], 255)
    check("original position empty", flat.getpixel((1, 1))[3], 0)
    check("master pixels untouched", master.image.getpixel((1, 1))[:3], (255, 0, 0))


def test_transform_is_per_instance():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    plain = link_to(doc, master)
    moved = link_to(doc, master)
    moved.set_transform((1.0, 0.0, 0.0, 1.0, 1.0, 0.0))

    flat = doc.composite_transparent()
    check("plain link stays put", flat.getpixel((1, 1))[3], 255)
    check("moved link is independent", flat.getpixel((2, 1))[3], 255)


def test_clear_transform_drops_matrix():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    link = link_to(doc, master)
    link.set_transform((1.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    link.clear_transform()

    # Clearing drops the matrix; the buffer keeps the offset the transform set,
    # so the content still renders from there.
    flat = doc.composite_transparent()
    check("transform cleared", link.transform, None)
    check("offset kept", link.offset, (1, 1))
    check("content at offset", flat.getpixel((2, 2))[3], 255)


def test_save_load_round_trip():
    app = make_project_app()
    source = app._new_canvas("Source", (4, 4))
    source.layers = []
    master = source.add_layer(name="Master")
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.is_master = True
    master.master_id = 1
    target = app._new_canvas("Target", (4, 4))
    target.layers = []
    link = target.add_layer(name="Link")
    link.link_id = 1
    link.master = master
    link.set_transform((2.0, 0.0, 0.0, 2.0, 0.0, 0.0))
    app.master_counter = 1

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "links.pai")
        manifest = project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
        reloaded = make_project_app()
        project.apply(reloaded, payload)

    entry = manifest["canvases"][1]["layers"][0]
    check("link id persisted", entry["link_id"], 1)
    check("master flag persisted", manifest["canvases"][0]["layers"][0]["is_master"], True)

    new_link = reloaded.canvases[1].layers[0]
    new_master = reloaded.canvases[0].layers[0]
    check("link restored", new_link.link_id, 1)
    check("master re-resolved across canvases", new_link.master is new_master, True)
    check("transform persisted", new_link.transform, (2.0, 0.0, 0.0, 2.0, 0.0, 0.0))
    check("master counter restored", reloaded.master_counter, 1)


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
