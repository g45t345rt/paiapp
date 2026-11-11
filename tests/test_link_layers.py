"""Unit tests for Link Layers (run: .venv\\Scripts\\python.exe tests\\test_link_layers.py).

A Link Layer renders a *master* layer's content while keeping its own position,
appearance and per-instance mask, is read-only, survives save/load, and can be
unlinked into an independent copy. These tests drive the document model and the
``PaintApp`` actions directly (no Tk root) plus a project save/load round trip.
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
from paintapp import gizmo  # noqa: E402
from paintapp.document import Canvas  # noqa: E402
from paintapp.drawing import DrawingController  # noqa: E402
from paintapp.gizmo import TransformGizmo  # noqa: E402

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
    app = SimpleNamespace(
        current=doc,
        canvases=[doc],
        active=doc.active,
        root=None,
        show_exploded=False,
        selection=None,
        guides_shared=False,
        area=SimpleNamespace(current_view=lambda: None),
        active_layer=doc.active_layer,
        _push_undo=lambda *_a, **_k: None,
        _refresh_layer_list=lambda *_a, **_k: None,
        _group_sel=None,
        set_result=lambda *_a, **_k: None,
        render=lambda *_a, **_k: None,
        set_mask_edit=lambda *_a, **_k: None,
        _sync_gen_size=lambda *_a, **_k: None,
        _update_title=lambda *_a, **_k: None,
        _shift_pose_scene=lambda *_a, **_k: None,
        _activate_canvas=lambda *_a, **_k: None,
        _on_layer_select=lambda *_a, **_k: None,
        _transform_mode_active=lambda *_a, **_k: False,
        drawing=SimpleNamespace(sync_transform=lambda *_a, **_k: None),
    )
    # The mask actions under test call these back through ``self``.
    app._active_layer_locked = lambda mask=False: PaintApp._active_layer_locked(
        app, mask=mask
    )
    app._active_mask_layer = lambda: PaintApp._active_mask_layer(app)
    app._mask_from_selection = lambda layer, invert=False: (
        PaintApp._mask_from_selection(app, layer, invert)
    )
    return app


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


def test_save_load_keeps_link_offset():
    """A link's move (its offset) survives save/load, not just its transform.

    A link re-renders its master's content at its own offset, so the loader must
    not re-place the fallback pixels: placing would enlarge the buffer and
    recentre a positive offset to ``(0, 0)``, silently undoing the move.
    """
    app = make_project_app()
    source = app._new_canvas("Source", (8, 8))
    source.layers = []
    master = source.add_layer(name="Master")
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.is_master = True
    master.master_id = 1
    target = app._new_canvas("Target", (8, 8))
    target.layers = []
    link = target.add_layer(name="Link")
    link.link_id = 1
    link.master = master
    link.offset = (3, 2)
    app.master_counter = 1

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "links.pai")
        project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
        reloaded = make_project_app()
        project.apply(reloaded, payload)

    new_link = reloaded.canvases[1].layers[0]
    check("link offset persisted", new_link.offset, (3, 2))


def test_link_instance_mask_hides_only_that_instance():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    hidden = link_to(doc, master)
    hidden.mask = Image.new("L", hidden.size, 255)
    hidden.mask.putpixel((1, 1), 0)
    shown = link_to(doc, master, offset=(2, 0))

    flat = doc.composite_transparent()
    check("masked instance hidden", flat.getpixel((1, 1))[3], 0)
    check("unmasked instance visible", flat.getpixel((3, 1))[3], 255)
    check("master pixels untouched", master.image.getpixel((1, 1))[:3], (255, 0, 0))
    check("mask lives on the link", hidden.mask.getpixel((1, 1)), 0)


def test_link_mask_does_not_touch_the_master():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    link = link_to(doc, master)
    link.mask = Image.new("L", link.size, 0)

    check(
        "link hidden", link.composited_image().getchannel("A").getpixel((1, 1)), 0
    )
    check(
        "master still shows",
        master.composited_image().getchannel("A").getpixel((1, 1)), 255,
    )


def test_link_mask_multiplies_the_master_mask():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    master.mask = Image.new("L", master.size, 255)
    master.mask.putpixel((1, 1), 128)
    link = link_to(doc, master)
    link.mask = Image.new("L", link.size, 255)
    link.mask.putpixel((1, 1), 128)

    alpha = doc.composite_transparent().getpixel((1, 1))[3]
    check("both masks folded", 0 < alpha < 128, True)


def test_link_mask_change_invalidates_caches():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    link = link_to(doc, master)
    check("visible before", link.composited_image().getchannel("A").getpixel((1, 1)), 255)
    before = Canvas._layer_signature(link)

    link.mask = Image.new("L", link.size, 0)

    check("hidden after", link.composited_image().getchannel("A").getpixel((1, 1)), 0)
    check("signature changed", Canvas._layer_signature(link) != before, True)
    check("canvas follows", doc.composite_transparent().getpixel((1, 1))[3], 0)


def test_link_mask_follows_a_transform():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    link = link_to(doc, master)
    # Scale the instance by 2: content (1, 1) lands at (2, 2).
    link.set_transform((2.0, 0.0, 0.0, 2.0, 0.0, 0.0))

    check("unmasked scaled link shows", doc.composite_transparent().getpixel((2, 2))[3] > 0, True)

    link.mask = Image.new("L", link.size, 0)
    flat = doc.composite_transparent()
    check("scaled link masked out", flat.getpixel((2, 2))[3], 0)
    check("scaled link masked out elsewhere", flat.getpixel((0, 0))[3], 0)


def test_unlink_folds_the_instance_mask():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    link = link_to(doc, master)
    link.mask = Image.new("L", link.size, 255)
    link.mask.putpixel((1, 1), 0)
    doc.active = doc.layers.index(link)

    PaintApp.unlink_layer(fake_app(doc))

    check("mask kept", link.mask.getpixel((1, 1)), 0)
    check("pixels kept", link.image.getpixel((1, 1))[:3], (255, 0, 0))
    check("still hidden after unlink", doc.composite_transparent().getpixel((1, 1))[3], 0)


def test_link_mask_survives_a_restore():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    link = link_to(doc, master)
    link.mask = Image.new("L", link.size, 255)
    link.mask.putpixel((1, 1), 0)

    doc.restore(doc.snapshot())
    restored = doc.layers[1]
    PaintApp._relink_masters(SimpleNamespace(canvases=[doc]))

    check("mask restored", restored.mask.getpixel((1, 1)), 0)
    check("hidden after restore", doc.composite_transparent().getpixel((1, 1))[3], 0)


def test_link_mask_save_load_round_trip():
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
    link.mask = Image.new("L", link.size, 255)
    link.mask.putpixel((1, 1), 0)
    app.master_counter = 1

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "links.pai")
        manifest = project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
        reloaded = make_project_app()
        project.apply(reloaded, payload)

    entry = manifest["canvases"][1]["layers"][0]
    check("mask file recorded", bool(entry["mask"]), True)

    new_link = reloaded.canvases[1].layers[0]
    check("mask restored", new_link.mask is not None, True)
    check("mask pixel restored", new_link.mask.getpixel((1, 1)), 0)
    check("link restored", new_link.link_id, 1)
    check(
        "hidden after reload",
        reloaded.canvases[1].composite_transparent().getpixel((1, 1))[3], 0,
    )


def test_link_mask_actions_are_allowed_but_apply_is_not():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.visible = False
    link = link_to(doc, master)
    doc.active = doc.layers.index(link)

    fake = fake_app(doc)
    fake.selection = None
    check("mask edits allowed on a link", PaintApp._active_layer_locked(fake, mask=True), False)
    check("pixel edits still refused", PaintApp._active_layer_locked(fake), True)

    PaintApp.add_layer_mask(fake)
    check("link gained a mask", link.mask is not None, True)
    check("mask reveals all", link.mask.getpixel((1, 1)), 255)

    PaintApp.invert_layer_mask(fake)
    check("inverted to hide", link.mask.getpixel((1, 1)), 0)

    PaintApp.delete_layer_mask(fake)
    check("mask deleted", link.mask, None)

    link.locked = True
    check("a locked link refuses mask edits", PaintApp._active_layer_locked(fake, mask=True), True)


def test_moved_link_mask_stays_aligned():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((1, 1), fill=(255, 0, 0, 255))
    master.draw.point((0, 0), fill=(0, 255, 0, 255))
    master.visible = False
    link = link_to(doc, master, offset=(1, 1))
    link.mask = Image.new("L", link.size, 255)
    link.mask.putpixel((1, 1), 0)

    # The mask is in the link's own buffer frame, which its offset places in the
    # document: content (1, 1) shows at (2, 2), hidden by mask pixel (1, 1).
    flat = doc.composite_transparent()
    check("moved content hidden", flat.getpixel((2, 2))[3], 0)
    check("unmasked content shows", flat.getpixel((1, 1))[3], 255)


def test_link_mask_follows_a_canvas_crop():
    doc = make_doc()
    master = promote(doc)
    master.draw.point((2, 2), fill=(255, 0, 0, 255))
    master.visible = False
    link = link_to(doc, master)
    link.mask = Image.new("L", link.size, 255)
    link.mask.putpixel((2, 2), 0)

    check("hidden before crop", doc.composite_transparent().getpixel((2, 2))[3], 0)

    PaintApp.crop_canvas(fake_app(doc), (1, 1, 4, 4))

    check("canvas cropped", doc.size, (3, 3))
    check("mask cropped with it", link.mask.size, (3, 3))
    check("mask pixel kept", link.mask.getpixel((1, 1)), 0)
    check("still hidden after crop", doc.composite_transparent().getpixel((1, 1))[3], 0)


def test_quick_transform_works_on_a_link():
    """The Transform panel's flip/rotate buttons edit a link per instance."""
    doc = make_doc()
    master = promote(doc)
    master.set_image(Image.new("RGBA", doc.size, (255, 0, 0, 255)))
    master.visible = False
    link = link_to(doc, master)
    doc.active = doc.layers.index(link)

    app = fake_app(doc)
    app.tools = SimpleNamespace(
        transform_tool=True, transform_snap_anchors=True,
        transform_interpolation="bicubic", transform_rotation_snap=15,
    )
    instance = TransformGizmo()
    check("gizmo begins on a link", instance.begin(app, None), True)

    controller = SimpleNamespace(app=app, gizmo=instance)
    controller._transform_ready = (
        lambda: instance.active and instance.layer is app.active_layer())
    controller.sync_transform = lambda: None
    controller._apply_transform_options = lambda: None
    controller._transform_link_quick = (
        lambda action: DrawingController._transform_link_quick(controller, action))
    DrawingController._transform_quick(controller, "rotate_180")

    check("link gained a per-instance transform", link.transform is not None, True)
    # The 4x4 opaque content is turned half a circle about its centre, so the
    # content pixel at the top left lands at the bottom right.
    x, y = gizmo.apply(link.transform, 0.5, 0.5)
    check("link content rotated about its centre",
          (round(x + link.offset[0], 3), round(y + link.offset[1], 3)),
          (3.5, 3.5))


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
