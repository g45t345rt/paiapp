"""Project load/save for the ``.pai`` format.

A ``.pai`` file is a zip archive holding a small JSON manifest and one PNG per
layer, so projects stay portable and human-inspectable::

    project.json                          manifest (canvases, settings)
    canvases/0000/layers/0000_Name.png    one RGBA PNG per layer, bottom-to-top
    canvases/0000/layers/0000_Name_mask.png   optional 8-bit layer mask
    assets/0000_Name.png                  project-wide assets, shared by canvases

A project can hold several canvases, each with its own size and layer stack.
Each canvas may also carry layer groups, referenced from its layers by id.
Version 1 files (a single implicit canvas at ``layers/``) still load; files
written before groups existed simply have no ``groups`` entry. Files written
before version 19 kept assets per canvas (``canvases/NNNN/assets/``); those
load into the one project-wide asset list. Version 20 adds asset groups, tags,
favorites and a project-wide asset group counter; older files load with every
asset ungrouped and untagged.

The functions take the running :class:`paint.PaintApp` so the caller can keep
all of its state (canvases, palette, generation panel) in one place.

Loading and saving are each split in two so they can run on a job queue's
worker thread. :func:`read` decodes the archive into plain data (reporting
progress and honouring cancellation) and :func:`apply` installs it on the Tk
thread; :func:`snapshot` gathers the plain data on the Tk thread (it reads the
tkinter variables) and :func:`write` encodes it off-thread. :func:`load` and
:func:`save` are the synchronous one-call combinations.
"""

import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from paintapp import filters, gradient, path, poses, textrender, vector
from paintapp.document import AssetLibrary, Canvas, Group, normalize_blend

FORMAT = "pai"
VERSION = 20
MANIFEST = "project.json"

# Maps manifest keys to the tkinter variables that hold the generate panel.
# These are the fallback values the panel shows for a canvas/layer with no saved
# settings. Prompts and settings are not listed here: the canvas-level settings
# live on each canvas, and the positive prompt and the layer-level settings on
# each layer, so they travel with their canvas/layer entry.
_GEN_VARS = [
    ("model", "model_var"),
    ("negative", "negative_var"),
    ("steps", "steps_var"),
    ("width", "gen_width_var"),
    ("height", "gen_height_var"),
    ("seed", "seed_var"),
    ("cfg", "cfg_var"),
    ("denoise", "denoise_var"),
    ("source", "source_var"),
    ("region", "region_var"),
    ("output", "output_var"),
    ("preview", "preview_var"),
    ("bg_model", "bg_model_var"),
    ("upscale_model", "upscale_model_var"),
]


def _safe_name(name):
    cleaned = "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip()
    return cleaned or "layer"


# Pillow releases the GIL while it runs zlib on PNG data, so encoding/decoding
# the layer images on a small thread pool scales with the CPU's cores.
_WORKERS = max(1, min(8, os.cpu_count() or 1))

# Fraction of a load's progress bar spent decoding images; the rest covers
# preparing the layers (see :func:`_warm_caches`), which usually costs more.
_DECODE_SHARE = 0.3


def _encode_png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _decode_png(data, mode):
    image = Image.open(io.BytesIO(data))
    image.load()
    # ``convert`` copies even when the mode already matches; skip that copy.
    return image if image.mode == mode else image.convert(mode)


def _run_parallel(func, items, on_result, cancel=None):
    """Call ``func(*item)`` for each item on a thread pool, in order.

    ``on_result(index, result)`` runs on the calling thread as results arrive
    (in submission order), and ``cancel`` is checked between them; pending work
    is dropped when the job is cancelled or an item fails.
    """
    if not items:
        return
    if _WORKERS == 1 or len(items) == 1:
        for index, item in enumerate(items):
            if cancel is not None:
                cancel.throw_if_cancelled()
            on_result(index, func(*item))
        return
    pool = ThreadPoolExecutor(max_workers=_WORKERS)
    try:
        futures = [pool.submit(func, *item) for item in items]
        for index, future in enumerate(futures):
            if cancel is not None:
                cancel.throw_if_cancelled()
            on_result(index, future.result())
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _load_guides(entries):
    """Coerce saved guide pairs to ``(orient, pos)``, dropping malformed ones."""
    guides = []
    for entry in entries or []:
        try:
            orient, pos = entry
        except (TypeError, ValueError):
            continue
        if orient in ("h", "v"):
            guides.append((orient, int(round(float(pos)))))
    return guides


def _place_layer(image, size, offset):
    """Position a saved layer without rescaling or clipping its content.

    Returns ``(image, offset)`` for a buffer large enough to hold both the
    canvas (``size``) and ``image`` placed at ``offset``. Off-canvas pixels are
    kept, so they survive a save/load round trip and can be painted on later.
    """
    ox, oy = offset
    left = min(0, ox)
    top = min(0, oy)
    right = max(size[0], ox + image.width)
    bottom = max(size[1], oy + image.height)
    if (left, top) == (ox, oy) and (right - left, bottom - top) == image.size:
        # The image already covers the canvas at its offset (the usual case):
        # the placed buffer would be an identical copy, so skip the composite.
        return image, (left, top)
    out = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
    # The buffer is fully transparent, so a straight copy equals compositing.
    out.paste(image, (ox - left, oy - top))
    return out, (left, top)


def _place_layer_mask(mask, size, offset):
    """Align an 8-bit mask to a placed layer's buffer (see :func:`_place_layer`)."""
    ox, oy = offset
    dest = (ox - min(0, ox), oy - min(0, oy))
    if dest == (0, 0) and mask.size == tuple(size):
        return mask
    out = Image.new("L", size, 0)
    out.paste(mask, dest)
    return out


def _generation_state(app):
    state = {}
    for key, attr in _GEN_VARS:
        var = getattr(app, attr, None)
        if var is not None:
            try:
                state[key] = var.get()
            except Exception:
                pass
    return state


def _migrate_generation_state(state):
    """Fold a pre-v11 generation state into the source/region pair.

    Format <= 10 stored the img2img source as an ``init_image`` flag plus an
    ``init_source`` name, and the selected-region inpaint as ``init_clip``.
    Older files also spelled the layer source "Layer"; it is now "Selected
    Layer". A state that already has ``source`` (or never had an init image) is
    returned unchanged.
    """
    if "source" in state or "init_image" not in state:
        return state
    legacy_source = state.get("init_source")
    if legacy_source == "Layer":
        legacy_source = "Selected Layer"
    state = dict(state)
    state["source"] = (
        legacy_source
        if state["init_image"] and legacy_source in ("Canvas", "Selected Layer")
        else "Off"
    )
    state["region"] = "Selection" if state.get("init_clip") else "Whole image"
    return state


def snapshot(app):
    """Collect a manifest and detached copies of every layer/mask image.

    Runs on the Tk thread, because it reads the app's tkinter variables (the
    generation state) and copies each layer image. The returned payload holds
    plain data and standalone :class:`~PIL.Image.Image` objects, so
    :func:`write` can encode it on a worker thread without touching widgets.
    """
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "active_canvas": app.active_canvas,
        "canvas_counter": app.canvas_counter,
        "master_counter": int(getattr(app, "master_counter", 0)),
        "color": app.color,
        "brush_size": app.brush_size,
        "tool_settings": app.tools.export_settings(),
        "text_style": getattr(app, "_text_style", None),
        "palette": list(app.palette),
        "selected_swatch": app.selected_swatch,
        "generation": _generation_state(app),
        # When guides are shared, every canvas draws the one app-level list, so
        # it travels at the top level instead of inside a canvas entry.
        "guides_shared": bool(getattr(app, "guides_shared", False)),
        "shared_guides": [
            [orient, int(pos)] for orient, pos in getattr(app, "shared_guides", [])
        ],
        "canvases": [],
        "asset_counter": 0,
        "asset_group_counter": 0,
        "asset_groups": [],
        "assets": [],
    }

    images = {}
    library = _app_library(app)
    manifest["asset_counter"] = int(library.asset_counter)
    manifest["asset_group_counter"] = int(getattr(library, "group_counter", 0))
    for group in getattr(library, "groups", []) or []:
        manifest["asset_groups"].append({
            "id": int(group.id),
            "name": group.name,
            "collapsed": bool(getattr(group, "collapsed", False)),
        })
    for asset_index, asset in enumerate(library.assets):
        # Assets are project-wide reference images; they keep their own
        # dimensions, so no offset is stored.
        filename = f"assets/{asset_index:04d}_{_safe_name(asset.name)}.png"
        images[filename] = asset.image.copy()
        manifest["assets"].append({
            "id": int(getattr(asset, "id", asset_index + 1)),
            "name": asset.name,
            "source": getattr(asset, "source", ""),
            "prompt": getattr(asset, "prompt", ""),
            # The Generate panel fields that produced a generated image, so
            # selecting the asset can restore them after a reload.
            "settings": dict(getattr(asset, "settings", {}) or {}),
            "group": getattr(asset, "group", None),
            "tags": list(getattr(asset, "tags", []) or []),
            "favorite": bool(getattr(asset, "favorite", False)),
            "file": filename,
        })
    for canvas_index, canvas in enumerate(app.canvases):
        entry = {
            "name": canvas.name,
            "width": canvas.width,
            "height": canvas.height,
            "active": canvas.active,
            "layer_counter": canvas.layer_counter,
            "group_counter": canvas.group_counter,
            "gen_settings": dict(getattr(canvas, "gen_settings", {}) or {}),
            "working_profile": getattr(canvas, "working_profile", "sRGB"),
            "proof_profile": getattr(canvas, "proof_profile", None),
            "proof_intent": getattr(canvas, "proof_intent", "relative"),
            "guides": [[orient, int(pos)] for orient, pos in canvas.guides],
            "show_exploded": bool(getattr(canvas, "show_exploded", False)),
            "groups": [
                {
                    "id": group.id,
                    "name": group.name,
                    "visible": bool(group.visible),
                    "opacity": float(group.opacity),
                    "collapsed": bool(group.collapsed),
                    "keep_empty": bool(group.keep_empty),
                }
                for group in canvas.groups
            ],
            "layers": [],
        }
        for layer_index, layer in enumerate(canvas.layers):
            # Copy the image so edits made after the snapshot cannot bleed into
            # the file that :func:`write` encodes on the worker thread.
            filename = (
                f"canvases/{canvas_index:04d}/layers/"
                f"{layer_index:04d}_{_safe_name(layer.name)}.png"
            )
            # A link layer has no pixels of its own; store what it currently
            # shows so the file still renders somewhere that does not know about
            # links, and let the master's own PNG be the source of truth. Its
            # own per-instance mask is stored beside it (below), so it is left
            # out here rather than baked into the pixels twice.
            if getattr(layer, "is_link", False):
                images[filename] = layer.composited_image(
                    include_link_mask=False
                ).copy()
            else:
                images[filename] = layer.image.copy()
            mask_file = None
            if layer.mask is not None:
                mask_file = (
                    f"canvases/{canvas_index:04d}/layers/"
                    f"{layer_index:04d}_{_safe_name(layer.name)}_mask.png"
                )
                images[mask_file] = layer.mask.copy()
            entry["layers"].append({
                "name": layer.name,
                "visible": bool(layer.visible),
                "collapsed": bool(getattr(layer, "collapsed", False)),
                "locked": bool(layer.locked),
                "opacity": float(layer.opacity),
                "offset": [int(layer.offset[0]), int(layer.offset[1])],
                "prompt": getattr(layer, "positive_prompt", ""),
                "gen_settings": dict(getattr(layer, "gen_settings", {}) or {}),
                "group": layer.group.id if layer.group is not None else None,
                "clip": bool(layer.clip),
                "blend": layer.blend,
                # Link Layer identity: a master carries its stable id, a link
                # carries the id of the master it points at (possibly on another
                # canvas). ``apply`` re-resolves the links once every canvas is
                # built.
                "is_master": bool(getattr(layer, "is_master", False)),
                "master_id": getattr(layer, "master_id", None),
                "link_id": getattr(layer, "link_id", None),
                # Per-instance transform of a link layer (content pixels to
                # document), or ``None`` when the link shows the master as-is.
                "transform": (
                    [float(value) for value in layer.transform]
                    if getattr(layer, "transform", None) is not None
                    else None
                ),
                "file": filename,
                "mask": mask_file,
                "filters": [filters.clone(spec) for spec in layer.filters],
                "texts": [
                    clone for clone in (
                        textrender.clone(spec) for spec in layer.texts
                    )
                    if clone is not None
                ],
                "gradients": [
                    spec
                    for spec in (
                        gradient.to_json(entry_spec)
                        for entry_spec in getattr(layer, "gradients", [])
                    )
                    if spec is not None
                ],
                "vector": bool(getattr(layer, "vector", False)),
                "strokes": [
                    clone for clone in (
                        vector.clone(spec) for spec in getattr(layer, "strokes", [])
                    )
                    if clone is not None
                ],
                "fills": [
                    clone for clone in (
                        vector.clone_fill(spec)
                        for spec in getattr(layer, "fills", [])
                    )
                    if clone is not None
                ],
                "paths": [
                    clone for clone in (
                        path.clone(spec) for spec in getattr(layer, "paths", [])
                    )
                    if clone is not None
                ],
                "poses": [
                    clone for clone in (
                        poses.clone(spec)
                        for spec in getattr(layer, "poses", [])
                    )
                    if clone is not None
                ],
                "pose_scene": poses.clone_scene(
                    getattr(layer, "pose_scene", None)
                ),
            })
        manifest["canvases"].append(entry)
    return {"manifest": manifest, "images": images}


def _app_library(app):
    """The app's project-wide :class:`AssetLibrary` (falls back to a canvas's)."""
    library = getattr(app, "asset_library", None)
    if library is None:
        canvases = getattr(app, "canvases", None) or []
        library = getattr(canvases[0], "library", None) if canvases else None
    return library if library is not None else AssetLibrary()


def write(payload, path, progress=None, cancel=None):
    """Encode a :func:`snapshot` payload into a ``.pai`` archive at ``path``.

    Runs on a worker thread, so it touches no widgets: it writes one PNG per
    copied image plus the manifest, reporting ``progress(frac)`` once per
    written image (a layer's PNG, plus one more when it carries a mask) and
    checking ``cancel`` before each. The archive is built at ``path + ".tmp"``
    and moved into place, so a crash or a cancelled write never damages an
    existing project.
    """
    manifest = payload["manifest"]
    images = payload["images"]
    total = len(images)
    done = 0
    if progress is not None:
        progress(0.0)

    tmp = path + ".tmp"
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            filenames = [
                asset_entry.get("file")
                for asset_entry in manifest.get("assets", []) or []
            ]
            for entry in manifest["canvases"]:
                filenames += [
                    layer_entry.get(key)
                    for layer_entry in entry["layers"]
                    for key in ("file", "mask")
                ]
            jobs = [
                (filename, images[filename])
                for filename in filenames
                if filename and images.get(filename) is not None
            ]

            def store(index, data):
                nonlocal done
                # PNG data is already deflated; compressing it again in the
                # zip costs time and saves nothing.
                zf.writestr(jobs[index][0], data, compress_type=zipfile.ZIP_STORED)
                done += 1
                if progress is not None:
                    progress(done / total if total else 1.0)

            _run_parallel(
                lambda _name, image: _encode_png(image), jobs, store, cancel
            )
            zf.writestr(MANIFEST, json.dumps(manifest, indent=2))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return manifest


def save(app, path, progress=None):
    """Write ``app`` to a ``.pai`` project at ``path``. Returns the manifest.

    The synchronous one-call path (:func:`snapshot` then :func:`write`), for
    callers with nothing to run between the two phases (see, in contrast, the
    queued save in :meth:`paintapp.app.PaintApp.save_project`).
    """
    return write(snapshot(app), path, progress=progress)


def read(app, path, progress=None, cancel=None):
    """Decode a ``.pai`` project into plain data, ready for :func:`apply`.

    Runs on the job queue's worker thread, so it touches no widgets: it reads
    the archive's manifest and decodes every layer image and mask, returning
    them as a payload the Tk thread installs with :func:`apply`. ``progress``
    is called as ``progress(frac, label)`` once per decoded image so the load
    can report on the status bar, and ``cancel`` is checked between images,
    raising :class:`~engine.cancel.Cancelled` when the job is cancelled.
    """
    if cancel is not None:
        cancel.throw_if_cancelled()
    if progress is not None:
        progress(0.0, "reading project")

    prior = list(getattr(app, "canvases", []))
    fallback = prior[0].size if prior else None

    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read(MANIFEST).decode("utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError("Not a .pai project file.")

        if "canvases" in manifest:
            canvas_entries = manifest["canvases"]
        else:
            # Version 1 files held a single implicit canvas.
            old_canvas = tuple(manifest.get("canvas") or ()) or fallback or (0, 0)
            canvas_entries = [{
                "name": "Canvas 1",
                "width": old_canvas[0],
                "height": old_canvas[1],
                "active": manifest.get("active", 0),
                "layer_counter": manifest.get("layer_counter", 0),
                "layers": manifest.get("layers", []),
            }]

        # Flatten the canvas layer entries so progress is one step per decoded
        # image: a layer's PNG, plus one more when it carries a mask.
        layer_entries = [
            (canvas_index, layer_index, layer_entry)
            for canvas_index, entry in enumerate(canvas_entries)
            for layer_index, layer_entry in enumerate(entry.get("layers", []) or [])
        ]
        # One project-wide asset list. Files before version 19 kept assets on
        # each canvas; those are appended (in canvas order) after any
        # project-level ones, so an old project shows them all globally.
        asset_entries = [
            (False, asset_entry)
            for asset_entry in manifest.get("assets", []) or []
        ] + [
            (True, asset_entry)
            for entry in canvas_entries
            for asset_entry in entry.get("assets", []) or []
        ]
        total = sum(
            1 + (1 if layer_entry.get("mask") else 0)
            for _canvas, _layer, layer_entry in layer_entries
        ) + len(asset_entries)
        images = {}
        masks = {}
        assets = {}
        # (target dict, key, archive member, mode, progress label) per image,
        # in the same order the progress steps were always reported.
        jobs = []
        for canvas_index, layer_index, layer_entry in layer_entries:
            key = (canvas_index, layer_index)
            jobs.append((images, key, layer_entry["file"], "RGBA", "layer"))
            mask_file = layer_entry.get("mask")
            if mask_file:
                jobs.append((masks, key, mask_file, "L", "layer"))
        for asset_index, (_legacy, asset_entry) in enumerate(asset_entries):
            jobs.append((
                assets, asset_index, asset_entry["file"], "RGBA", "asset",
            ))

        def read_job(index):
            # ZipFile reads are not thread-safe; only the decode is pooled.
            if cancel is not None:
                cancel.throw_if_cancelled()
            return zf.read(jobs[index][2]), jobs[index][3]

        done = 0

        def install(index, image):
            nonlocal done
            target, key, _member, _mode, label = jobs[index]
            target[key] = image
            done += 1
            if progress is not None:
                share = _DECODE_SHARE if layer_entries else 1.0
                progress(share * done / total, f"{label} {done}/{total}")

        _run_parallel(
            _decode_png, [read_job(i) for i in range(len(jobs))], install, cancel
        )

    if progress is not None and not total:
        # A project with no layer images still finished loading.
        progress(1.0, "")
    payload = {
        "manifest": manifest,
        "canvas_entries": canvas_entries,
        "images": images,
        "masks": masks,
        "assets": assets,
        "asset_entries": asset_entries,
        "fallback": fallback,
    }
    # Build the canvases here too, and pre-render what the first paint needs,
    # so the Tk thread only has to install them (see :func:`apply`).
    payload["canvases"] = _build_canvases(payload, cancel=cancel)
    _warm_caches(payload["canvases"], cancel=cancel, progress=progress)
    return payload


def _relink(canvases):
    """Point every link layer at its master (mirrors ``PaintApp._relink_masters``)."""
    masters = {
        layer.master_id: layer
        for canvas in canvases for layer in canvas.layers
        if getattr(layer, "is_master", False) and layer.master_id is not None
    }
    for canvas in canvases:
        for layer in canvas.layers:
            if getattr(layer, "link_id", None) is not None:
                layer.master = masters.get(layer.link_id)


def _warm_caches(canvases, cancel=None, progress=None):
    """Render each layer's composite once so its caches are filled.

    The layer panel's previews and the first viewport paint ask every layer for
    its composite (filters, texts, paths, link transforms). Those are cached on
    the layer, so computing them here, on the worker and before the layers are
    visible to the Tk thread, keeps that work from freezing the window when
    the project is installed. Failures are ignored: the Tk thread simply
    renders the layer itself as before. ``progress`` advances from
    :data:`_DECODE_SHARE` to 1.0, one step per layer.
    """
    from paintapp import layers as layer_panel

    layers = [layer for canvas in canvases for layer in canvas.layers]
    for index, layer in enumerate(layers, 1):
        if cancel is not None:
            cancel.throw_if_cancelled()
        try:
            layer.composited_image(include_mask=True)
            # Also fills the unmasked composite the layer panel shows.
            layer_panel.prepare_thumbnail(layer)
        except Exception:
            pass
        if progress is not None:
            frac = _DECODE_SHARE + (1.0 - _DECODE_SHARE) * index / len(layers)
            progress(frac, f"preparing layer {index}/{len(layers)}")


def _build_canvases(payload, cancel=None):
    """Build detached :class:`Canvas` objects from a decoded payload.

    Touches no app or widget state, so it can run on the worker thread.
    """
    canvas_entries = payload["canvas_entries"]
    images = payload["images"]
    masks = payload["masks"]
    fallback = payload["fallback"]
    library = _build_library(payload)
    payload["library"] = library

    canvases = []
    for canvas_index, entry in enumerate(canvas_entries):
        if cancel is not None:
            cancel.throw_if_cancelled()
        width = int(entry.get("width") or 0)
        height = int(entry.get("height") or 0)
        size = (width, height) if width and height else (fallback or (800, 640))
        doc = Canvas(entry.get("name", f"Canvas {canvas_index + 1}"), size)
        doc.library = library
        canvases.append(doc)
        doc.layers = []
        doc.gen_settings = dict(entry.get("gen_settings") or {})
        doc.working_profile = str(entry.get("working_profile") or "sRGB")
        doc.proof_profile = entry.get("proof_profile") or None
        doc.proof_intent = str(entry.get("proof_intent") or "relative")
        doc.guides = _load_guides(entry.get("guides"))
        doc.show_exploded = bool(entry.get("show_exploded", False))
        groups = {}
        for group_entry in entry.get("groups", []) or []:
            group = Group(
                group_entry.get("name", "Group"),
                int(group_entry.get("id", len(groups) + 1)),
            )
            group.visible = bool(group_entry.get("visible", True))
            group.opacity = float(group_entry.get("opacity", 1.0))
            group.collapsed = bool(group_entry.get("collapsed", False))
            group.keep_empty = bool(group_entry.get("keep_empty", False))
            groups[group.id] = group
            doc.groups.append(group)
        for layer_index, layer_entry in enumerate(entry.get("layers", [])):
            image = images.get((canvas_index, layer_index))
            layer = doc.add_layer(name=layer_entry.get("name", "Layer"))
            offset = layer_entry.get("offset") or (0, 0)
            offset = (int(offset[0]), int(offset[1]))
            # A link re-renders its master's content at its own offset instead
            # of keeping the stored pixels, so keep its saved buffer frame and
            # offset as-is: placing it would enlarge the buffer and recentre a
            # positive offset to (0, 0), dropping the instance's move on reload.
            if image is not None and layer_entry.get("link_id") is None:
                placed, placed_offset = _place_layer(image, doc.size, offset)
                layer.set_image(placed)
            else:
                if image is not None:
                    layer.set_image(image)
                placed_offset = offset
            layer.offset = placed_offset
            mask = masks.get((canvas_index, layer_index))
            if mask is not None:
                if layer_entry.get("link_id") is not None:
                    # A link's mask is stored in its own buffer frame, which is
                    # what it is composited in; placing it would shift it.
                    layer.mask = mask
                else:
                    layer.mask = _place_layer_mask(mask, layer.image.size, offset)
            layer.visible = bool(layer_entry.get("visible", True))
            layer.collapsed = bool(layer_entry.get("collapsed", False))
            layer.locked = bool(layer_entry.get("locked", False))
            layer.opacity = float(layer_entry.get("opacity", 1.0))
            layer.positive_prompt = str(layer_entry.get("prompt", "") or "")
            layer.gen_settings = dict(layer_entry.get("gen_settings") or {})
            layer.group = groups.get(layer_entry.get("group"))
            layer.clip = bool(layer_entry.get("clip", False))
            layer.blend = normalize_blend(layer_entry.get("blend"))
            layer.is_master = bool(layer_entry.get("is_master", False))
            layer.master_id = layer_entry.get("master_id")
            layer.link_id = layer_entry.get("link_id")
            transform = layer_entry.get("transform")
            layer.transform = (
                tuple(float(value) for value in transform)
                if transform is not None else None
            )
            layer.filters = [
                spec
                for spec in (
                    filters.normalize_spec(entry_spec)
                    for entry_spec in layer_entry.get("filters", [])
                )
                if spec is not None
            ]
            layer.texts = [
                spec
                for spec in (
                    textrender.normalize(entry_spec)
                    for entry_spec in layer_entry.get("texts", [])
                )
                if spec is not None
            ]
            layer.gradients = [
                spec
                for spec in (
                    gradient.from_json(entry_spec)
                    for entry_spec in layer_entry.get("gradients", [])
                )
                if spec is not None
            ]
            layer.vector = bool(layer_entry.get("vector", False))
            layer.strokes = [
                spec
                for spec in (
                    vector.normalize(entry_spec)
                    for entry_spec in layer_entry.get("strokes", [])
                )
                if spec is not None
            ]
            layer.fills = [
                spec
                for spec in (
                    vector.normalize_fill(entry_spec)
                    for entry_spec in layer_entry.get("fills", [])
                )
                if spec is not None
            ]
            layer.paths = [
                spec
                for spec in (
                    path.normalize(entry_spec)
                    for entry_spec in layer_entry.get("paths", [])
                )
                if spec is not None
            ]
            layer.poses, layer.pose_scene = poses.load_layer_poses(
                layer_entry.get("poses", layer_entry.get("mannequins", [])),
                layer_entry.get("pose_scene"),
                layer.size,
            )
        doc.layer_counter = int(entry.get("layer_counter", len(doc.layers)))
        doc.group_counter = int(entry.get("group_counter", len(doc.groups)))
        doc.active = (
            max(0, min(int(entry.get("active", 0)), len(doc.layers) - 1))
            if doc.layers else 0
        )
    _relink(canvases)
    return canvases


def _build_library(payload):
    """Build the project-wide :class:`AssetLibrary` from a decoded payload.

    Project-level assets keep their saved ids. Assets from a pre-19 file were
    numbered per canvas, so their ids can collide; they are renumbered after
    the highest project-level id instead.
    """
    manifest = payload.get("manifest") or {}
    assets = payload.get("assets", {})
    entries = payload.get("asset_entries") or []
    library = AssetLibrary()
    for group_entry in manifest.get("asset_groups", []) or []:
        try:
            group_id = int(group_entry.get("id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if group_id <= 0:
            continue
        if library.find_group(group_id) is not None:
            continue
        library.add_group(
            group_entry.get("name", "Group"),
            group_id,
            collapsed=group_entry.get("collapsed", False),
        )
    try:
        counter = int(manifest.get("asset_group_counter", 0) or 0)
    except (TypeError, ValueError):
        counter = 0
    library.group_counter = max(
        [counter] + [group.id for group in library.groups]
    )
    kept = []
    for asset_index, (legacy, asset_entry) in enumerate(entries):
        image = assets.get(asset_index)
        if image is not None:
            kept.append((legacy, asset_entry, image))
    next_id = max(
        [int(manifest.get("asset_counter", 0) or 0)]
        + [int(entry.get("id", 0) or 0) for legacy, entry, _ in kept if not legacy]
    )
    for legacy, asset_entry, image in kept:
        group_id = asset_entry.get("group")
        if group_id is not None:
            try:
                group_id = int(group_id)
            except (TypeError, ValueError):
                group_id = None
            if library.find_group(group_id) is None:
                group_id = None
        asset = library.add_asset(
            name=asset_entry.get("name", "Asset"),
            image=image,
            source=asset_entry.get("source", ""),
            prompt=asset_entry.get("prompt", ""),
            settings=asset_entry.get("settings", {}),
            # Rebuild the saved order (newest first) rather than
            # front-inserting each restore, which would reverse it.
            index=len(library.assets),
            group=group_id,
            tags=asset_entry.get("tags", []),
            favorite=asset_entry.get("favorite", False),
        )
        if legacy:
            next_id += 1
            asset.id = next_id
        else:
            asset.id = int(asset_entry.get("id", asset.id))
    library.asset_counter = max(
        [next_id] + [asset.id for asset in library.assets]
    )
    return library


def apply(app, payload):
    """Install a :func:`read` payload into ``app``. Returns the manifest.

    Runs on the Tk thread. :func:`read` has already built the canvases, layers
    and masks off-thread, so this installs them and pushes the saved palette,
    tool and generation-panel state back into their widgets.
    """
    manifest = payload["manifest"]
    canvases = payload.get("canvases")
    if canvases is None:
        canvases = _build_canvases(payload)
    # A payload installs once; later loads build fresh canvases.
    payload["canvases"] = None

    library = payload.get("library") or _build_library(payload)
    app.asset_library = library
    for doc in canvases:
        doc.library = library
    app.canvases = list(canvases)
    app.active_canvas = 0

    app.canvas_counter = int(manifest.get("canvas_counter", len(app.canvases)))
    app.master_counter = max(
        int(manifest.get("master_counter", 0)),
        max(
            (layer.master_id or 0
             for doc in app.canvases for layer in doc.layers
             if getattr(layer, "is_master", False)),
            default=0,
        ),
    )
    # Every canvas (and so every master layer) now exists; resolve link layers
    # to their masters. A link whose master is missing stays unresolved and
    # simply shows its own (stored) pixels.
    relink = getattr(app, "_relink_masters", None)
    if relink is not None:
        relink()
    app.active_canvas = max(
        0, min(int(manifest.get("active_canvas", 0)), len(app.canvases) - 1)
    )
    app.color = manifest.get("color", app.color)
    app.brush_size = int(manifest.get("brush_size", app.brush_size))
    tools = getattr(app, "tools", None)
    if tools is not None:
        tools.apply_settings(manifest.get("tool_settings"))
    text_style = manifest.get("text_style")
    if isinstance(text_style, dict):
        app._text_style = text_style
    palette = manifest.get("palette")
    if isinstance(palette, list) and palette:
        app.palette = list(palette)
        if hasattr(app, "swatches"):
            for box, color in zip(app.swatches, app.palette):
                box.config(bg=color)
    app.selected_swatch = min(
        int(manifest.get("selected_swatch", app.selected_swatch)),
        len(app.palette) - 1,
    )

    app.set_color(app.color)
    if hasattr(app, "size_var"):
        app.size_var.set(app.brush_size)

    state = _migrate_generation_state(manifest.get("generation", {}))
    # The saved settings live on the canvases and layers; these globals are only
    # the fallback the panel shows where they have none, so suppress the panel's
    # write-back while they are pushed in (or they would pollute the just-loaded
    # canvas).
    app._loading_project = True
    try:
        for key, attr in _GEN_VARS:
            var = getattr(app, attr, None)
            if var is not None and key in state:
                try:
                    var.set(state[key])
                except Exception:
                    pass
    finally:
        app._loading_project = False

    # Shared guides live on the app rather than a canvas. A file saved in shared
    # mode carries the list at the top level; an older one (or an empty list)
    # seeds it from the first canvas so the mode still has something to draw.
    app.guides_shared = bool(manifest.get("guides_shared", False))
    app.shared_guides = _load_guides(manifest.get("shared_guides"))
    if app.guides_shared and not app.shared_guides and app.canvases:
        app.shared_guides = [tuple(guide) for guide in app.canvases[0].guides]

    return manifest


def load(app, path):
    """Load a ``.pai`` project at ``path`` into ``app``. Returns the manifest.

    The synchronous one-call path, for callers with nothing to report between
    the two phases. Loading a project from the user's File menu runs
    :func:`read` on the job queue and :func:`apply` on the Tk thread instead, so
    a large project decodes without freezing the window (see
    :meth:`paintapp.app.PaintApp.open_project`).
    """
    return apply(app, read(app, path))
