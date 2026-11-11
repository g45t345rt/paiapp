"""The :class:`PaintApp` coordinator that wires the components together."""

import datetime
import io
import os
import queue
import time
import tkinter as tk
from tkinter import filedialog
from xml.sax.saxutils import escape as _xml_escape

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageGrab, ImageTk
from PIL import PngImagePlugin

import project

from . import dialogs
from . import colormanagement
from . import combobox as _combobox
from . import filters
from . import fonts
from . import modelsettings
from . import path
from . import settingsdialog
from . import svgimport
from . import textrender
from . import theme
from . import vector
from .canvasarea import (
    CanvasArea, DEFAULT_GRID_SIZE, DEFAULT_OFFCANVAS_BG, prompt_canvas_size,
    prompt_new_canvas,
)
from .assets import AssetsPanel
from .brushes import BrushLibraryPanel
from .colorpicker import ColorPicker
from .constants import (    APP_NAME, APP_VERSION, DONATE_URL, CONTRIBUTE_URL, DISCORD_URL, CANVAS_W,
    CANVAS_H, MAX_CANVAS_SIZE, SIDEBAR_MIN_W, PANELS, PANEL_SIDES,
    DEFAULT_PANEL_SIDES, DEFAULT_COLLAPSED_PANELS, DEFAULT_PANEL_GROUPS, rgba,
    TOOL_RAIL_W, TOOL_RAIL_MIN_W, TOOL_RAIL_MAX_W,
)
from .document import Canvas
from .drawing import DrawingController, ToolsPanel
from .generation import (
    GenerationPanel,
    OUTPUT_ASSETS,
    OUTPUT_LAYER,
    OUTPUT_NEW_LAYER,
    REGION_MASK,
    REGION_SELECTION,
    SOURCE_ASSET,
    SOURCE_CANVAS,
    SOURCE_LAYER,
)
from .controlnet_panel import (
    HINT_ASSET, HINT_CANVAS, HINT_LAYER, HINT_SOURCE, ControlNetPanel,
)
from .references import ReferencesPanel
from .historypanel import HistoryPanel
from .jobqueue import (
    CANCELLED, Job, JobQueue, JobQueuePanel, QUEUED, RUNNING,
)
from .layers import LayerPanel
from .logpanel import LogPanel
from .loaded_models_panel import LoadedModelsPanel
from .monitor import SystemMonitor
from .removal import BackgroundPanel
from .regions_panel import MASK_PAINTED, RegionsPanel, decode_mask
from .selection import Selection
from .shortcuts import ShortcutManager
from .sidebar import Sidebar, ToolsRail
from .ui import PopupMenu, StatusBar, build_menubar
from .upscale_panel import UpscalePanel


# Every picker should show its full value in the drop-down, so swap in the
# auto-widening combobox before any panel or dialog creates one.
_combobox.install()


# Pillow resize filters keyed by the export dialog's resample ids.
_RESAMPLE_FILTERS = {
    "nearest": Image.NEAREST,
    "box": Image.BOX,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}

# DPI assumed for a print page when the export dialog leaves the DPI field blank.
_PRINT_FALLBACK_DPI = 300

# Milliseconds between live-stroke repaints. The stroke dabs are stamped on every
# motion event; only the viewport repaint is deferred to this timer, so the event
# loop keeps collecting pointer samples instead of stalling on a render per
# event. That stall is worst when zoomed out, where one sample already spans many
# document pixels, so each coalesced-away sample chords straight across the curve
# the hand made and the stroke looks like it skips (see PaintApp.schedule_live_render).
_LIVE_STROKE_RENDER_MS = 5

# Reference settings (Anima + Qwen) are a canvas-level concern: they must not
# travel with the active layer, or selecting another layer would swap the
# reference images. They are stripped from the per-layer generation settings.
_REFERENCE_SETTING_KEYS = (
    "ref_refs", "ref_lora", "ref_strength", "ref_guidance", "ref_correction",
    "ref_offset", "ref_megapixels", "qwen_refs",
)

# Layout Control is layer-scoped: each layer draws its own set of boxes for its
# generation prompt, so it travels with the active layer. The canvas keeps a copy
# as the fallback a layer with no settings of its own inherits, mirroring how the
# rest of the generation settings resolve.


def _hex_to_rgba(color):
    """Parse a ``#rrggbb``/``#rgb`` string into an opaque RGBA tuple."""
    text = (color or "#ffffff").lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), 255)
    except (ValueError, IndexError):
        return (255, 255, 255, 255)


def _capitalize(text):
    """Upper-case the first character (used for status labels)."""
    return text[:1].upper() + text[1:] if text else text


def _stitch_side_by_side(images, gap=8, background=(255, 255, 255)):
    """Lay ``images`` out left-to-right at a common height on a white canvas.

    Used by the Anima reference adapter, whose convention is that two characters
    are two references side by side: several assigned references become one
    stitched reference image. Non-tallest images are scaled to the tallest one's
    height (aspect preserved) so different source sizes line up.
    """
    images = [im.convert("RGB") for im in images]
    if len(images) == 1:
        return images[0]
    height = max(im.height for im in images)
    scaled = []
    for im in images:
        if im.height != height:
            width = max(1, round(im.width * height / im.height))
            im = im.resize((width, height), Image.LANCZOS)
        scaled.append(im)
    total = sum(im.width for im in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (total, height), background)
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def _export_icc(image, options, working_profile):
    """The ICC profile bytes to embed in an exported file, or ``None``.

    Follows the image's actual space: a CMYK image embeds the CMYK profile it
    was made with; otherwise the export target profile (or the working space
    when no target is set). Disabled when ``options["profile_embed"]`` is false.
    Module-level and ``self``-free so it can run on the export worker (and be
    exercised by tests) without an app instance.
    """
    if not options.get("profile_embed", True):
        return None
    if image.mode == "CMYK":
        identifier = options.get("cmyk_profile") or ""
    else:
        identifier = options.get("profile_target") or (
            working_profile or colormanagement.SRGB
        )
    return colormanagement.icc_bytes(identifier)


# Descriptive metadata fields shared by the export dialog, PNG text chunks, EXIF
# and XMP. ``None`` fields are omitted, so a blank dialog entry writes nothing.
_EXPORT_METADATA_FIELDS = (
    ("title", "Title"),
    ("author", "Author"),
    ("copyright", "Copyright"),
    ("description", "Description"),
    ("keywords", "Keywords"),
    ("software", "Software"),
    ("created", "Creation Time"),
)
# EXIF/TIFF tags for the fields with a standard home. Keywords and the exact
# creation date are written through XMP instead.
_EXIF_METADATA_TAGS = {
    "title": 0x010D,       # DocumentName
    "description": 0x010E,  # ImageDescription
    "software": 0x0131,    # Software
    "author": 0x013B,      # Artist
    "copyright": 0x8298,   # Copyright
}


def _export_metadata(options):
    """The non-empty descriptive metadata from ``options["metadata"]``."""
    raw = options.get("metadata")
    if not isinstance(raw, dict):
        return {}
    cleaned = {}
    for key, _label in _EXPORT_METADATA_FIELDS:
        value = raw.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                cleaned[key] = value
    return cleaned


def _exif_datetime(text):
    """Format a user-entered date as the EXIF ``YYYY:MM:DD HH:MM:SS`` form.

    Accepts ISO 8601 and the common date-only spellings; anything unparseable is
    passed through unchanged so a free-form value is not silently discarded.
    """
    text = (text or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        for fmt in (
            "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d", "%Y/%m/%d", "%Y:%m:%d",
        ):
            try:
                parsed = datetime.datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    return parsed.strftime("%Y:%m:%d %H:%M:%S") if parsed else text


def _export_exif(metadata):
    """An :class:`Image.Exif` carrying the standard EXIF/TIFF text tags."""
    exif = Image.Exif()
    for key, tag in _EXIF_METADATA_TAGS.items():
        value = metadata.get(key)
        if value:
            exif[tag] = value
    created = _exif_datetime(metadata.get("created"))
    if created:
        exif[0x0132] = created  # DateTime
    return exif


def _export_xmp(metadata):
    """A minimal XMP packet for the descriptive fields, or ``b""``."""
    if not metadata:
        return b""
    parts = []
    title = metadata.get("title")
    if title:
        parts.append(
            '<dc:title><rdf:Alt><rdf:li xml:lang="x-default">'
            f"{_xml_escape(title)}</rdf:li></rdf:Alt></dc:title>"
        )
    author = metadata.get("author")
    if author:
        parts.append(
            "<dc:creator><rdf:Seq><rdf:li>"
            f"{_xml_escape(author)}</rdf:li></rdf:Seq></dc:creator>"
        )
    copyright_text = metadata.get("copyright")
    if copyright_text:
        parts.append(
            '<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">'
            f"{_xml_escape(copyright_text)}</rdf:li></rdf:Alt></dc:rights>"
        )
    description = metadata.get("description")
    if description:
        parts.append(
            '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">'
            f"{_xml_escape(description)}</rdf:li></rdf:Alt></dc:description>"
        )
    keywords = metadata.get("keywords")
    if keywords:
        items = [
            item.strip()
            for item in keywords.replace(";", ",").split(",")
            if item.strip()
        ]
        if items:
            tags = "".join(f"<rdf:li>{_xml_escape(item)}</rdf:li>" for item in items)
            parts.append(f"<dc:subject><rdf:Bag>{tags}</rdf:Bag></dc:subject>")
    software = metadata.get("software")
    if software:
        parts.append(f"<xmp:CreatorTool>{_xml_escape(software)}</xmp:CreatorTool>")
    created = metadata.get("created")
    if created:
        stamp = _xml_escape(created)
        parts.append(
            f"<xmp:CreateDate>{stamp}</xmp:CreateDate>"
            f"<xmp:ModifyDate>{stamp}</xmp:ModifyDate>"
        )
    packet = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '<rdf:Description rdf:about=""'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
        ' xmlns:xmp="http://ns.adobe.com/xap/1.0/">\n'
        + "\n".join(parts)
        + "\n</rdf:Description>\n</rdf:RDF>\n</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )
    return packet.encode("utf-8")


def _export_pnginfo(metadata):
    """A :class:`PngImagePlugin.PngInfo` with the descriptive text chunks."""
    info = PngImagePlugin.PngInfo()
    for key, label in _EXPORT_METADATA_FIELDS:
        value = metadata.get(key)
        if value:
            info.add_text(label, value)
    return info


# Wording that marks a status message as a failure. Hit before the warning
# markers, so a message that both "failed" and mentions "no layer" reads red.
_ERROR_MARKERS = ("error:", "failed", "could not", "unable to")

# Wording that marks a refused or impossible action: guard clauses and
# validations. These read amber in the status bar and the Logs panel.
_WARN_MARKERS = (
    " first", "already", "nothing to", "there is no", "there are no",
    "no canvas", "no layer", "no selection", "no visible", "no guides",
    "no filters", "no text", "no mask", "no size", "has no", "have no",
    "is empty", "are empty", "empty", "locked", "hidden", "exceeds",
    "not ", "select a", "select at least", "select one",
)

# Leading words of a confirmation of a completed action, which reads green.
_SUCCESS_PREFIXES = (
    "added", "applied", "cleared", "clone", "copied", "cropped", "cut",
    "default canvas size set", "deleted", "deselected", "duplicated", "edited",
    "exported", "feathered", "filled", "fitted", "flattened", "flipped",
    "generated", "grouped", "inverted", "merged", "model settings saved",
    "moved", "opened", "pasted", "preloaded", "rasterized", "removed",
    "renamed", "reset", "resized", "rotated", "saved", "selected",
    "transformed", "ungrouped", "upscaled", "updated",
)


def _classify_result(text):
    """Return the severity of status ``text``: ``info``, ``success``, ``warn``
    or ``error``.

    Most messages are ordinary info; the wording is inspected so the many
    existing :meth:`PaintApp.set_result` call sites color themselves without
    every one being tagged by hand. A caller can still force a level.
    """
    message = (text or "").strip()
    if not message:
        return "info"
    low = message.lower()
    if low.startswith("error") or any(m in low for m in _ERROR_MARKERS):
        return "error"
    if any(m in low for m in _WARN_MARKERS):
        return "warn"
    if low.startswith(_SUCCESS_PREFIXES):
        return "success"
    return "info"


# Win32 ``MonitorFromWindow`` flag: use the monitor nearest the window even
# when the window's rectangle does not intersect any monitor.
_MONITOR_DEFAULTTONEAREST = 2


def _frame_handle(window):
    """The native window handle of ``window``'s decorative frame, or ``None``.

    Tk wraps every toplevel in a decorative frame window; the platform calls
    behind fullscreen act on that frame rather than on the client window. Only
    Windows builds expose them, so every other platform yields ``None`` and the
    fullscreen helpers below become no-ops there.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        window.update_idletasks()
        return (
            ctypes.windll.user32.GetParent(window.winfo_id())
            or window.winfo_id()
        )
    except Exception:
        return None


def _monitor_rect(window):
    """The ``(left, top, right, bottom)`` bounds of ``window``'s monitor.

    Tk's ``-fullscreen`` fills the primary display on Windows regardless of
    where the window sits (its geometry code pins the window to ``0,0`` at the
    primary screen size), so the bounds of the display the window currently
    occupies are needed to move it onto the right monitor afterwards. Returns
    ``None`` off Windows, when the window has no frame yet, or when the monitor
    cannot be queried.
    """
    hwnd = _frame_handle(window)
    if hwnd is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.MonitorFromWindow.restype = ctypes.c_void_p
        user32.MonitorFromWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        monitor = user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONEAREST)
        if not monitor:
            return None

        class _MonitorInfo(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        user32.GetMonitorInfoW.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_MonitorInfo)
        ]
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        rect = info.rcMonitor
        return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        return None


def _move_window_to(window, rect):
    """Move ``window``'s native frame onto the screen ``rect``.

    Asked of the OS directly because Tk ignores geometry requests while a
    window is fullscreen (its ``UpdateGeometryInfo`` returns early in that
    state, so ``wm geometry`` has no effect until fullscreen is left). Does
    nothing off Windows.
    """
    hwnd = _frame_handle(window)
    if hwnd is None:
        return
    try:
        import ctypes

        left, top, right, bottom = rect
        swp_nozorder = 0x0004
        swp_noactivate = 0x0010
        ctypes.windll.user32.SetWindowPos(
            hwnd, 0, int(left), int(top),
            int(right - left), int(bottom - top),
            swp_nozorder | swp_noactivate,
        )
    except Exception:
        pass


class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        theme.init(self.root, theme.load_preference())
        self.busy = False
        self.queue = queue.Queue()
        self.job_queue = JobQueue(self.queue.put)
        # Project open/save run on their own worker, so they never wait behind a
        # long generation job. Its messages are tagged "project" on the way out
        # so _poll_queue can route them to the shared panel and handlers.
        self.project_queue = JobQueue(
            lambda message: self.queue.put(("project",) + tuple(message))
        )
        self.project_path = None
        self.canvas_counter = 0
        # Names master layers (Link Layer sources) with stable ids so links can
        # be re-resolved after a project load or an undo restore.
        self.master_counter = 0
        self.canvases = []
        self.active_canvas = 0
        self._clipboard = None
        # Fingerprint of the OS clipboard at the time ``_clipboard`` was filled,
        # so Paste can tell the in-app copy apart from a newer external image.
        self._clipboard_mark = None
        # Set while a dialog asks for a one-shot canvas sample; holds the
        # callback that receives the picked ``#rrggbb`` color.
        self._canvas_pick = None
        self._load_settings()
        # Auto-unload: the idle clock starts now and is refreshed by every job
        # event; a periodic tick drops cached engine models once it expires (see
        # ``_auto_unload_tick``). The interval is the Generation setting.
        self._last_model_activity = time.monotonic()
        self.root.after(30_000, self._auto_unload_tick)
        self.show_offscreen = True
        # Whether the active layer's bounds are outlined on the canvas (see
        # ``set_layer_outline``); toggled from the Layers panel.
        self.show_layer_outline = True
        # Frozen content bounds ``((left, top), (w, h))`` while a Move drag is in
        # progress, else ``None``. With the off-canvas preview on, the content
        # bounds follow the dragged layer, so the view's scroll region would be
        # rebuilt - and the camera re-anchored - on every motion event; on a real
        # window that nudges the view by a pixel each frame, so the untouched
        # layers appear to jitter. Holding the bounds steady for the drag keeps
        # the region fixed and the other layers still (see
        # ``DrawingController._begin_move`` / ``_end_move``).
        self._move_bounds = None
        self.mask_overlay_color = (255, 0, 0)
        # Mask editing state (layer mask, inpaint mask) the canvas currently
        # shows a veil for, so leaving either mode by switching tools can
        # repaint the view and drop the veil without waiting for a click.
        self._mask_preview_state = (False, False)
        self._text_style = None
        # Warm the system font registry off the main thread so the first font
        # picker open does not scan every font directory synchronously.
        fonts.prewarm()
        self._filter_sel = None
        self._text_sel = None
        self._path_sel = None
        self._group_sel = None
        self._group_filter_sel = None
        # Visibility captured by :meth:`isolate_layer` so it can be put back by
        # :meth:`unisolate_layer`; ``None`` when no layer is isolated.
        self._isolate_state = None
        self.modified = False
        # Set while project.load pushes the saved generation state into the
        # panel, so those writes are not stored back onto the loaded canvas.
        self._loading_project = False
        # Set while :meth:`_sync_gen_settings` is showing a canvas/layer's stored
        # settings in the panels. A panel that reconfigures itself there (for
        # example a ControlNet family switch) must not write back, or it would
        # overwrite the stored settings with the not-yet-synced panel state.
        self._syncing_settings = False
        # The image currently shown in the canvas viewport for review - a
        # finished generation or an asset picked from the Assets panel - or
        # ``None`` (see :meth:`_show_generation_preview` and
        # :meth:`preview_asset`).
        self._preview = None
        # The canvas currently receiving live denoising-step preview frames, or
        # ``None`` (see :meth:`_on_gen_preview`). Cleared when the run settles.
        self._live_preview_doc = None
        # Pending live-stroke repaint timer id, or ``None`` when no stroke frame
        # is queued (see :meth:`schedule_live_render`).
        self._stroke_render_job = None

        # Open with a canvas ready to paint on; File > New / Add Canvas create
        # more, and an opened project/image replaces this default one.
        self._new_canvas("Canvas 1", size=(CANVAS_W, CANVAS_H))
        self.drawing = DrawingController(self)

        # Shortcuts are registered before the menus so their accelerator labels
        # can be read back from the manager, then bound once the menu exists.
        self.shortcuts = ShortcutManager(self)
        build_menubar(self)
        self.shortcuts.apply()

        self.statusbar = StatusBar(self.root, self._canvas_size_text())
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.monitor = SystemMonitor(
            lambda cpu, gpu, ram: self.queue.put(("sysstats", cpu, gpu, ram))
        ).start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_body()
        self._apply_interface_settings()

        # Pin the window to its natural size so that resizing the sidebar
        # redistributes space internally instead of resizing the whole window.
        self.root.update_idletasks()
        self.root.geometry(
            f"{self.root.winfo_reqwidth()}x{self.root.winfo_reqheight()}"
        )
        self.root.minsize(SIDEBAR_MIN_W * 2 + 320, 480)

        self._sync_gen_size()
        self._refresh_layer_list()
        self._refresh_history()
        self._update_title()
        self._report_tool()
        self._apply_cursor_state()
        self._set_modified(False)
        self.render()
        self.root.after(100, self._poll_queue)
        theme.apply()
        # Set the icon once the body is built: the Windows path maps the window
        # to reach its frame, and doing that earlier would flash an empty window.
        self._set_app_icon()

    def _set_app_icon(self):
        """Use ``pai_icon.png`` beside ``paint.py`` as the window/taskbar icon.

        The reference is kept on the instance because Tk does not copy the
        ``PhotoImage``; letting it be collected would blank the icon. A missing
        or unreadable file is ignored, leaving Tk's default icon in place.
        """
        icon = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "pai_icon.png",
        )
        try:
            image = Image.open(icon).convert("RGBA")
        except (OSError, ValueError):
            return
        try:
            self._icon_photo = ImageTk.PhotoImage(image)
            self.root.iconphoto(True, self._icon_photo)
        except (tk.TclError, RuntimeError):
            self._icon_photo = None
        if os.name == "nt":
            self._set_taskbar_icon(image)

    def _set_taskbar_icon(self, image):
        """Give the Windows taskbar button our icon instead of the Python one.

        The taskbar reads a window's *small* icon, which ``iconphoto`` does not
        set, so the button keeps the interpreter's icon. Tk's ``wm iconbitmap``
        would set both that and the large icon, but it is unreliable with the
        PNG-compressed frames Pillow writes and could leave the window with no
        icon at all, so both ``WM_SETICON`` sizes are set here directly from a
        temporary ``.ico`` re-encoded at the sizes the shell asks for. The path
        is kept on the instance for the process lifetime.
        """
        import ctypes
        import tempfile

        try:
            handle = tempfile.NamedTemporaryFile(
                prefix="pai_icon_", suffix=".ico", delete=False
            )
            handle.close()
            image.save(
                handle.name,
                format="ICO",
                sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                       (128, 128), (256, 256)],
            )
        except (OSError, ValueError):
            return
        self._icon_path = handle.name
        hwnd = _frame_handle(self.root)
        if hwnd is None:
            return
        try:
            user32 = ctypes.windll.user32
            user32.LoadImageW.restype = ctypes.c_void_p
            user32.LoadImageW.argtypes = [
                ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint,
            ]
            user32.SendMessageW.argtypes = [
                ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            image_icon = 1
            load_from_file = 0x0010
            wm_seticon = 0x0080
            for size, which in ((32, 1), (16, 0)):  # ICON_BIG, ICON_SMALL
                hicon = user32.LoadImageW(
                    None, self._icon_path, image_icon, size, size,
                    load_from_file,
                )
                if hicon:
                    user32.SendMessageW(hwnd, wm_seticon, which, hicon)
        except Exception:
            pass

    # ------------------------------------------------------------------ layout
    def _build_body(self):
        body = tk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)
        # Four columns: the dedicated tool rail, the left panel rail, the canvas
        # (which absorbs the slack), and the right panel rail. A sidebar with no
        # docked panels removes itself from the grid, so the canvas simply takes
        # the freed space.
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)

        self.area = CanvasArea(
            body,
            on_activate=self._activate_canvas,
            on_rename=self._prompt_rename,
            on_view_created=self.drawing.attach,
            on_reorder=self._reorder_canvases,
            on_gallery_close=self._gallery_closed,
            on_gallery_zoom=self._report_gallery_zoom,
            on_add_canvas=self.add_canvas,
            on_guides_changed=self._guides_changed,
            on_context=self._canvas_tab_menu,
        )

        # The tool selector gets its own resizable rail, separate from the panel
        # sidebars, so it can be narrowed to an icon-only strip on its own.
        self.tools_rail = ToolsRail(body, width=self.tool_rail_width)
        self.left_sidebar = Sidebar(body, self.root, side="left")
        self.right_sidebar = Sidebar(body, self.root, side="right")
        self.left_sidebar.other = self.right_sidebar
        self.right_sidebar.other = self.left_sidebar
        self.left_sidebar.on_move = self._on_panel_moved
        self.right_sidebar.on_move = self._on_panel_moved
        # ``self.sidebar`` stays as the default rail so existing code keeps
        # working; panels are routed by :meth:`_add_panel` instead.
        self.sidebar = self.right_sidebar

        self._panel_hosts = {}
        self._panel_sides = self._load_panel_sides()

        # Mount the rails before the canvas: each claims its width first,
        # then the area takes only the space that is left and shrinks as a
        # rail grows.
        self.tools_rail.mount(row=0, column=0, sticky="ns", pady=8)
        self.left_sidebar.mount(row=0, column=1, sticky="ns", pady=8)
        self.area.grid(row=0, column=2, sticky="nsew", padx=(8, 0), pady=8)
        self.right_sidebar.mount(row=0, column=3, sticky="ns", pady=8)
        self.area.rebuild(self.canvases)

        self._build_panels()

    # -------------------------------------------------------------- sidebars
    def _panel_sidebar(self, key):
        """The sidebar a panel should be built into, from the saved layout."""
        side = self._panel_sides.get(key, "right")
        return self.left_sidebar if side == "left" else self.right_sidebar

    def _add_panel(self, key, title, index=None):
        """Register ``key`` in its saved sidebar and return its content frame.

        ``index`` optionally places the panel at that slot in the rail so a
        panel can be built after a dependency but still sit above it.
        """
        sidebar = self._panel_sidebar(key)
        content = sidebar.add_panel(
            key, title, collapsed=key in DEFAULT_COLLAPSED_PANELS, index=index
        )
        self._panel_hosts[key] = sidebar.host(key)
        return content

    def panel_side(self, key):
        """Return ``"left"`` or ``"right"`` for the panel ``key``."""
        host = self._panel_hosts.get(key)
        if host is not None:
            return host.dock.side
        return self._panel_sides.get(key, "right")

    def move_panel(self, key, side):
        """Dock the panel ``key`` in the left or right sidebar."""
        host = self._panel_hosts.get(key)
        if host is None:
            return
        side = "left" if side == "left" else "right"
        dest = self.left_sidebar if side == "left" else self.right_sidebar
        if host.dock is dest:
            return
        host.dock.move_panel(host, dest)
        self.set_result(f"Moved {host.title} to the {side} sidebar.")

    def toggle_panel_side(self, key):
        """Send the panel ``key`` to the sidebar on the other side."""
        self.move_panel(key, "left" if self.panel_side(key) == "right" else "right")

    def _on_panel_moved(self, host):
        """Record a panel's new side after a move and remember it on disk."""
        self._panel_sides[host.key] = host.dock.side
        self._save_panel_sides()

    def _load_panel_sides(self):
        stored = theme.read_settings().get("panel_sides")
        sides = {}
        for key, _title in PANELS:
            value = stored.get(key) if isinstance(stored, dict) else None
            sides[key] = (
                value if value in PANEL_SIDES
                else DEFAULT_PANEL_SIDES.get(key, "right")
            )
        return sides

    def _save_panel_sides(self):
        settings = theme.read_settings()
        settings["panel_sides"] = dict(self._panel_sides)
        theme.write_settings(settings)

    def _load_tool_rail_width(self):
        """The saved tool-rail width, clamped to its allowed range."""
        stored = theme.read_settings().get("tool_rail_width")
        try:
            width = int(stored)
        except (TypeError, ValueError):
            width = TOOL_RAIL_W
        return max(TOOL_RAIL_MIN_W, min(width, TOOL_RAIL_MAX_W))

    def _save_tool_rail_width(self, width):
        """Persist a tool-rail resize so it is restored on the next launch."""
        try:
            width = max(TOOL_RAIL_MIN_W, min(int(width), TOOL_RAIL_MAX_W))
        except (TypeError, ValueError):
            return
        settings = theme.read_settings()
        settings["tool_rail_width"] = width
        theme.write_settings(settings)

    # ------------------------------------------------------------ settings
    def _load_settings(self):
        """Load the persisted interface/canvas preferences into attributes.

        Values that need widgets to exist (rulers, guides, stats, brush size)
        are stored here and pushed onto the built UI by
        :meth:`_apply_interface_settings`.
        """
        settings = theme.read_settings()
        interface = settings.get("interface")
        interface = interface if isinstance(interface, dict) else {}
        self.show_rulers = bool(interface.get("rulers", True))
        self.show_guides = bool(interface.get("guides", True))
        # Whether every canvas draws the same guides or keeps its own. Shared
        # guides live here on the app; per-canvas guides (the default) live on
        # each document's ``Canvas.guides``.
        self.guides_shared = bool(interface.get("shared_guides", False))
        self.shared_guides = []
        # Grid overlay and snapping. The grid is a canvas aid rather than a
        # document property, so it lives on the app like the rulers and guides.
        self.show_grid = bool(interface.get("grid", False))
        self.grid_snap = bool(interface.get("grid_snap", False))
        try:
            grid_size = int(interface.get("grid_size", DEFAULT_GRID_SIZE))
        except (TypeError, ValueError):
            grid_size = DEFAULT_GRID_SIZE
        self.grid_size = max(1, grid_size)
        try:
            opacity = float(interface.get("offscreen_opacity", 0.35))
        except (TypeError, ValueError):
            opacity = 0.35
        self.offscreen_opacity = max(0.0, min(1.0, opacity))
        bg = interface.get("offscreen_bg")
        if isinstance(bg, (list, tuple)) and len(bg) == 3:
            self.offscreen_bg = tuple(int(c) for c in bg)
        else:
            self.offscreen_bg = DEFAULT_OFFCANVAS_BG
        stats = interface.get("stats")
        stats = stats if isinstance(stats, dict) else {}
        self._stat_visibility = {
            kind: bool(stats.get(kind, True)) for kind in ("cpu", "ram", "gpu")
        }
        canvas = settings.get("canvas")
        canvas = canvas if isinstance(canvas, dict) else {}
        try:
            width = int(canvas.get("width", CANVAS_W))
            height = int(canvas.get("height", CANVAS_H))
        except (TypeError, ValueError):
            width, height = CANVAS_W, CANVAS_H
        self.default_canvas_size = (
            max(1, min(width, MAX_CANVAS_SIZE)),
            max(1, min(height, MAX_CANVAS_SIZE)),
        )
        self._default_brush_size = canvas.get("brush_size")
        color = settings.get("color")
        color = color if isinstance(color, dict) else {}
        stored_profile = color.get("working_profile")
        self.default_working_profile = (
            stored_profile if isinstance(stored_profile, str) and stored_profile
            else colormanagement.SRGB
        )
        self._export_defaults = self._load_export_defaults()
        # Width of the dedicated tool rail, restored so the user's resize sticks.
        self.tool_rail_width = self._load_tool_rail_width()
        # Push the saved generation-engine tuning into the engine now, so the
        # first preload/generation already honours it (generation re-applies it
        # each run, but a preload or an unload should not need a generation first).
        try:
            from engine import tuning
            tuning.apply()
        except Exception:
            pass

    def _apply_interface_settings(self):
        """Push the loaded preferences onto the widgets that are now built."""
        self.area.set_rulers_visible(self.show_rulers)
        self.area.set_guides_visible(self.show_guides)
        self.area.set_shared_guides(self.shared_guides if self.guides_shared else None)
        self.area.set_grid_visible(self.show_grid)
        self.area.set_grid_size(self.grid_size)
        self.area.set_grid_snap(self.grid_snap)
        self.area.set_layer_outline_visible(self.show_layer_outline)
        self.area.set_bg_color(self.offscreen_bg)
        if hasattr(self, "ruler_var"):
            self.ruler_var.set(self.show_rulers)
        if hasattr(self, "guide_var"):
            self.guide_var.set(self.show_guides)
        if hasattr(self, "shared_guides_var"):
            self.shared_guides_var.set(self.guides_shared)
        if hasattr(self, "grid_var"):
            self.grid_var.set(self.show_grid)
        if hasattr(self, "grid_snap_var"):
            self.grid_snap_var.set(self.grid_snap)
        for kind, visible in self._stat_visibility.items():
            var = getattr(self, f"{kind}_stat_var", None)
            if var is not None:
                var.set(visible)
            self.statusbar.set_stat_visible(kind, visible)
        if self._default_brush_size is not None:
            try:
                self.brush_size = int(self._default_brush_size)
            except (TypeError, ValueError):
                pass

    def _build_panels(self):
        content = self._add_panel("color", "Color")
        self.colorpicker = ColorPicker(content, on_change=self._on_color_changed)
        self.colorpicker.pack(fill=tk.X)

        # The tool selector is built into the dedicated tool rail, not a panel
        # sidebar; its options still get their own "Tool Settings" panel. The
        # rail reports its width so the buttons swap between icon-only and full.
        self.tools = ToolsPanel(
            self.tools_rail.content,
            settings_parent=self._add_panel("tool_settings", "Tool Settings"),
            on_mode_change=self._on_mode_change, on_brush_size=self._on_brush_size,
            on_mask_edit=self.toggle_mask_edit,
            get_text_style=self.default_text_style,
            on_text_style=self._on_text_style,
            on_mask_action=self._on_mask_action,
            on_mask_overlay=self.set_mask_overlay_color,
            on_path_action=self.drawing.finish_path,
            on_path_settings=self.drawing.refresh_path,
            on_crop_action=self.drawing.apply_crop,
            on_crop_reset=self.drawing.clear_crop,
            on_clone_clear=self.drawing.clear_clone_source,
            on_paste=self.paste,
            on_pick_color=self.begin_canvas_pick,
            on_select_radius=self.drawing.reapply_select_radius,
            get_color=lambda: self.color,
            on_warp_action=self.drawing.on_warp_action,
            on_puppet_action=self.drawing.on_puppet_action,
            on_poses_action=self.drawing.on_poses_action,
            on_vector_action=self.drawing.on_vector_action,
            on_transform_action=self.drawing.on_transform_action,
            on_vector_settings=self.drawing.refresh_vector,
            on_layout_change=self._on_layout_change,
            get_layout_family=lambda: self.gen.model_family()
            if hasattr(self, "gen") else None,
            on_layout_capture=self._layout_selection_box,
        )
        self.tools.pack(fill=tk.X)
        self.tools_rail.on_resize = self.tools.set_width
        self.tools_rail.on_width_commit = self._save_tool_rail_width

        content = self._add_panel("brushes", "Brushes", index=1)
        self.brush_library = BrushLibraryPanel(
            content, self.tools, on_apply=self._on_brush_preset,
            on_status=self.set_result,
        )
        self.brush_library.pack(fill=tk.BOTH, expand=True)

        content = self._add_panel("layers", "Layers")
        self.layer_panel = LayerPanel(
            content,
            on_add=self.add_layer, on_delete=self.delete_layer,
            on_duplicate=self.duplicate_layer,
            on_clear=self.clear,
            on_move=self.move_layer, on_toggle=self.toggle_visible,
            on_lock=self.toggle_lock,
            on_isolate=self.isolate_layer, on_show_all=self.show_all_layers,
            on_unisolate=self.unisolate_layer, is_isolated=self.has_isolation,
            on_merge=self.merge_layers, on_select=self._on_layer_select,
            on_opacity=self._on_opacity, on_rename=self.rename_layer,
            on_add_mask=self.add_layer_mask,
            on_add_to_assets=self.add_layer_to_assets,
            on_delete_mask=self.delete_layer_mask,
            on_apply_mask=self.apply_layer_mask,
            on_mask_edit=self.set_mask_edit,
            on_outline=self.set_layer_outline,
            on_filter_select=self._on_filter_select,
            on_filter_edit=self._on_filter_edit,
            on_filter_toggle=self.toggle_selected_filter,
            on_filter_remove=self.remove_selected_filter,
            on_filter_move=self.move_selected_filter,
            on_filter_clear=self.clear_filters,
            on_text_select=self._on_text_select,
            on_text_edit=self._on_text_edit,
            on_text_remove=self.remove_selected_text,
            on_text_move=self.move_selected_text,
            on_path_select=self._on_path_select,
            on_path_edit=self._on_path_edit,
            on_path_remove=self.remove_selected_path,
            on_path_move=self.move_selected_path,
            on_apply_filters=self.bake_filters,
            on_rasterize_texts=self.rasterize_selected_texts,
            on_rasterize_paths=self.rasterize_selected_paths,
            on_rasterize_path=self.rasterize_selected_path,
            on_apply_filter=self.apply_selected_filter,
            on_rasterize_text=self.rasterize_selected_text,
            on_group_select=self._on_group_select,
            on_group_rename=self.rename_selected_group,
            on_group_toggle=self.toggle_group_visible,
            on_group_collapse=self.toggle_group_collapse,
            on_group_ungroup=self.ungroup_selected_group,
            on_group_flatten=self.flatten_selected_group,
            on_group_delete=self.delete_selected_group,
            on_group_move=self.move_selected_group,
            on_group_create=self.group_selected_layers,
            on_group_add=self.add_group,
            on_group_assign=self.move_selected_layers_to_group,
            on_clip=self.toggle_clip,
            on_blend=self._on_blend,
            on_group_filter_select=self._on_group_filter_select,
            on_group_filter_edit=self._on_group_filter_edit,
            on_group_filter_toggle=self.toggle_selected_group_filter,
            on_group_filter_remove=self.remove_selected_group_filter,
            on_group_filter_move=self.move_selected_group_filter,
            on_group_filter_clear=self.clear_group_filters,
            on_group_filter_apply=self.apply_group_filters,
            on_reorder=self.reorder_layers,
            on_master_convert=self.convert_to_master,
            on_master_remove=self.remove_master,
            on_link_create=self.create_link_layer,
            on_unlink=self.unlink_layer,
            on_select_master=self.select_master,
            on_link_reset_transform=self.reset_link_transform,
            get_master_choices=self.master_choices,
            on_move_to_canvas=self.move_layer_to_canvas,
            get_canvas_choices=self.canvas_move_choices,
            on_group_duplicate=self.duplicate_selected_group,
            on_text_duplicate=self.duplicate_selected_text,
            on_text_toggle=self.toggle_selected_text_visible,
            on_path_duplicate=self.duplicate_selected_path,
            on_path_toggle=self.toggle_selected_path_visible,
        )
        self.layer_panel.pack(fill=tk.X)

        content = self._add_panel("assets", "Assets")
        self.assets_panel = AssetsPanel(
            content,
            on_add=self.add_asset_from_file,
            on_add_layer=self.add_layer_to_assets,
            on_insert=self.insert_asset,
            on_preview=self.preview_asset,
            on_save=self.save_asset,
            on_rename=self.rename_asset,
            on_delete=self.delete_asset,
            on_drag=self._drag_asset_over_layer,
            on_drop=self._drop_asset_on_layer,
            on_select=self._on_asset_select,
        )
        self.assets_panel.pack(fill=tk.X)

        content = self._add_panel("history", "History")
        self.history_panel = HistoryPanel(
            content, on_select=self._on_history_select,
        )
        self.history_panel.pack(fill=tk.X)

        content = self._add_panel("generate", "Generate")
        self.gen = GenerationPanel(
            content, on_generate=self.start_generate,
            get_canvas_size=lambda: (
                self.current.size if self.current is not None
                else (CANVAS_W, CANVAS_H)
            ),
            on_cancel=self.cancel_generate,
            on_keep=self.stop_generate_keep,
            on_positive_prompt=self._on_positive_prompt,
            on_preload=self.start_preload,
            on_edit_inpaint_mask=self.set_inpaint_mask_edit,
            on_settings=self._on_gen_settings,
        )
        self.gen.pack(fill=tk.X)

        content = self._add_panel("references", "References")
        self.references_panel = ReferencesPanel(
            content, on_settings=self._on_reference_settings,
            on_capture=self._capture_reference_selection,
        )
        self.references_panel.pack(fill=tk.X)

        content = self._add_panel("controlnet", "ControlNet")
        self.controlnet_panel = ControlNetPanel(
            content, on_settings=self._on_control_settings,
        )
        self.controlnet_panel.pack(fill=tk.X)

        content = self._add_panel("region_prompts", "Region Prompts")
        self.regions_panel = RegionsPanel(
            content, on_settings=self._on_region_settings,
            on_capture=self._capture_region_mask,
        )
        self.regions_panel.pack(fill=tk.X)

        # The Layout Control panel sits below ControlNet and Region Prompts.
        self.tools.set_layout_parent(self._add_panel("layout", "Layout Control"))

        # Point both panels at the default model's family once they all exist, so
        # a family change can save the whole stack without a missing sibling.
        self.controlnet_panel.set_family(self.gen.model_family())
        self.regions_panel.set_family(self.gen.model_family())
        self.references_panel.set_family(self.gen.model_family())
        self.tools.set_layout_family(self.gen.model_family())

        content = self._add_panel("loaded_models", "Loaded Models")
        self.loaded_models_panel = LoadedModelsPanel(
            content, on_unload=self.unload_models,
        )
        self.loaded_models_panel.pack(fill=tk.X)

        content = self._add_panel("background", "Remove Background")
        self.bg = BackgroundPanel(
            content, on_remove=self.start_remove_background,
            on_cancel=self.cancel_remove_background,
        )
        self.bg.pack(fill=tk.X)

        content = self._add_panel("upscale", "Upscale")
        self.upscale = UpscalePanel(
            content, on_upscale=self.start_upscale,
            on_cancel=self.cancel_upscale,
        )
        self.upscale.pack(fill=tk.X)

        content = self._add_panel("queue", "Queue")
        self.job_panel = JobQueuePanel(
            content,
            on_action=self._dismiss_job,
            on_cancel_all=self._cancel_all_jobs,
            on_clear_finished=self._clear_finished_jobs,
        )
        self.job_panel.pack(fill=tk.X)

        content = self._add_panel("logs", "Logs")
        self.log_panel = LogPanel(content)
        self.log_panel.pack(fill=tk.X)

        # Aliases expected by project.py, which treats PaintApp as a bag of state.
        self.model_var = self.gen.model_var
        self.negative_var = self.gen.negative_var
        self.steps_var = self.gen.steps_var
        self.gen_width_var = self.gen.gen_width_var
        self.gen_height_var = self.gen.gen_height_var
        self.seed_var = self.gen.seed_var
        self.cfg_var = self.gen.cfg_var
        self.denoise_var = self.gen.denoise_var
        self.source_var = self.gen.source_var
        self.region_var = self.gen.region_var
        self.output_var = self.gen.output_var
        self.preview_var = self.gen.preview_var
        self.bg_model_var = self.bg.bg_model_var
        self.upscale_model_var = self.upscale.upscale_model_var
        self.size_var = self.tools.size_var

        self._apply_default_groups()

    def _apply_default_groups(self):
        """Tab the panels listed in :data:`DEFAULT_PANEL_GROUPS` on startup.

        A group is only formed when every member ended up on the same sidebar
        (the saved layout can move a panel to the other side); otherwise the
        members are left as they are.
        """
        for active, members in DEFAULT_PANEL_GROUPS:
            hosts = [
                self._panel_hosts[key] for key in members
                if key in self._panel_hosts
            ]
            if len(hosts) < 2:
                continue
            sidebar = hosts[0].dock
            if any(host.dock is not sidebar for host in hosts):
                continue
            target = hosts[0]
            for host in hosts[1:]:
                item = sidebar.group_of(target.key) or target
                sidebar.merge_items(host, item)
            group = sidebar.group_of(target.key)
            if group is not None:
                sidebar.set_group_active(group, active)
                # Tabbing members together expands them (see
                # ``Sidebar._adopt``), so a group whose shown panel starts
                # collapsed is folded back to its header here.
                if active in DEFAULT_COLLAPSED_PANELS:
                    sidebar.set_group_collapsed(group, True)

    # ------------------------------------------------------------------ canvases
    @property
    def current(self):
        if not self.canvases:
            return None
        return self.canvases[self.active_canvas]

    @property
    def layers(self):
        return self.current.layers if self.current is not None else []

    @layers.setter
    def layers(self, value):
        if self.current is not None:
            self.current.layers = value

    @property
    def active(self):
        return self.current.active if self.current is not None else 0

    @active.setter
    def active(self, value):
        if self.current is not None:
            self.current.active = value

    @property
    def guides(self):
        """The guides currently in effect: the shared list, or the active canvas's."""
        if self.guides_shared:
            return self.shared_guides
        return self.current.guides if self.current is not None else []

    @property
    def show_exploded(self):
        """Whether the active canvas is in the exploded-grid view.

        The mode is stored per canvas, so one canvas can be stacked while
        another shows the grid; switching tabs keeps each one's choice.
        """
        return bool(self.current.show_exploded) if self.current is not None else False

    @show_exploded.setter
    def show_exploded(self, value):
        if self.current is not None:
            self.current.show_exploded = bool(value)

    def _new_canvas(self, name=None, size=(CANVAS_W, CANVAS_H), fill=(0, 0, 0, 0),
                    layers=1):
        self.canvas_counter += 1
        if name is None:
            name = f"Canvas {self.canvas_counter}"
        doc = Canvas(name, size)
        doc.working_profile = getattr(
            self, "default_working_profile", colormanagement.SRGB
        ) or colormanagement.SRGB
        try:
            count = max(1, int(layers))
        except (TypeError, ValueError):
            count = 1
        # Only the first (bottom) layer takes the chosen background fill; the
        # rest start transparent.
        for index in range(count):
            doc.add_layer(fill=fill if index == 0 else (0, 0, 0, 0))
        self.canvases.append(doc)
        return doc

    def _activate_canvas(self, index):
        if not self.canvases:
            self.active_canvas = 0
            self.area.refresh_titles(self.canvases)
            self._refresh_layer_list()
            self._update_title()
            self._apply_cursor_state()
            self.render()
            return
        index = max(0, min(index, len(self.canvases) - 1))
        self.active_canvas = index
        self.area.select(index)
        self.drawing.commit_text()
        self.drawing.cancel_path()
        self.drawing.discard_warp()
        self.drawing.discard_puppet()
        self._refresh_layer_list()
        self._update_title()
        self._apply_cursor_state()
        self.drawing.sync_warp()
        self.drawing.sync_puppet()
        self._sync_layout_overlay()
        self.render()
        view = self.area.current_view()
        if view is not None:
            self.report_zoom(view)
        if self._transform_mode_active():
            self.drawing.sync_transform()
        # The exploded view is a per-canvas mode: mirror the newly active
        # canvas's choice, hide the (document-space) rulers there, and frame the
        # grid so switching tabs lands on the whole board.
        if hasattr(self, "exploded_var"):
            self.exploded_var.set(self.show_exploded)
        if self.show_exploded:
            self.area.set_rulers_visible(False)
            layout = self.current._exploded_layout if self.current else None
            if view is not None and layout is not None and layout.size[0] > 0:
                view.fit_document_rect((0, 0, layout.size[0], layout.size[1]))

    def _refresh_canvas_tabs(self):
        self.area.refresh_titles(self.canvases)

    def add_canvas(self):
        result = prompt_new_canvas(
            self.root,
            initial=self.default_canvas_size,
            default_name=f"Canvas {self.canvas_counter + 1}",
        )
        if result is None:
            return
        self._new_canvas(
            name=result["name"], size=result["size"], fill=result["fill"],
            layers=result["layers"],
        )
        self.area.add(self.canvases[-1])
        self._set_modified(True)
        self._activate_canvas(len(self.canvases) - 1)
        self._refresh_gallery_if_visible()
        if result.get("make_default"):
            self._set_default_canvas_size(result["size"])

    def duplicate_canvas(self, index=None):
        """Copy a canvas, layers and settings included, into a new tab.

        The copy is fully independent, so editing either canvas leaves the
        other untouched. ``index`` defaults to the active canvas.
        """
        if not self.canvases:
            self.set_result("There is no canvas to duplicate.")
            return
        if index is None:
            index = self.active_canvas
        if not (0 <= index < len(self.canvases)):
            return
        source = self.canvases[index]
        self.canvas_counter += 1
        duplicate = source.copy(name=f"{source.name} copy")
        self.canvases.append(duplicate)
        self.area.add(duplicate)
        self._set_modified(True)
        self._activate_canvas(len(self.canvases) - 1)
        self._refresh_gallery_if_visible()
        self.set_result(f"Duplicated canvas '{source.name}'.")

    def _set_default_canvas_size(self, size):
        """Remember ``size`` as the default new-canvas size (Settings > Canvas)."""
        width, height = int(size[0]), int(size[1])
        self.default_canvas_size = (width, height)
        settings = theme.read_settings()
        canvas = settings.get("canvas")
        canvas = canvas if isinstance(canvas, dict) else {}
        canvas["width"] = width
        canvas["height"] = height
        settings["canvas"] = canvas
        theme.write_settings(settings)
        self.set_result(
            f"Default canvas size set to {width} x {height}."
        )

    def delete_canvas(self, index=None):
        if not self.canvases:
            return
        if index is None:
            index = self.active_canvas
        if not (0 <= index < len(self.canvases)):
            return
        if not dialogs.ask_yes_no(
            self.root,
            "Delete Canvas",
            f"Delete the canvas '{self.canvases[index].name}'?",
        ):
            return
        self._clear_generation_preview()
        # Break links to any master living on the canvas being deleted, so no
        # link is left pointing at a master that no longer exists.
        doomed_masters = [
            (layer.master_id, layer)
            for layer in self.canvases[index].layers
            if getattr(layer, "is_master", False) and layer.master_id is not None
        ]
        del self.canvases[index]
        for master_id, master in doomed_masters:
            self._unlink_links_to(master_id, master)
        self.area.remove(index)
        self._set_modified(True)
        if not self.canvases:
            self.active_canvas = 0
        elif index < self.active_canvas:
            self.active_canvas -= 1
        elif index == self.active_canvas:
            self.active_canvas = min(index, len(self.canvases) - 1)
        self._activate_canvas(self.active_canvas)
        self._relink_masters()
        self._refresh_gallery_if_visible()

    def rename_canvas(self):
        self._prompt_rename(self.active_canvas)

    def _shift_pose_scene(self, layer, dx, dy):
        """Move a layer's poses scene anchor with a canvas origin change.

        The Poses figures and their ground grid are anchored in document space,
        so a resize, crop or fit must shift the scene the same way the layer
        offset and the text specs are shifted, or the figures drift off the new
        origin.
        """
        scene = getattr(layer, "pose_scene", None)
        if not isinstance(scene, dict):
            return
        scene["x"] = float(scene.get("x", 0.0)) + dx
        scene["y"] = float(scene.get("y", 0.0)) + dy
        layer.invalidate_poses()

    def resize_canvas(self):
        """Resize the current canvas, keeping existing content centered."""
        if self.current is None:
            self.set_result("There is no canvas to resize.")
            return
        width, height = self.current.size
        size = prompt_canvas_size(
            self.root, initial=(width, height), title="Resize Canvas"
        )
        if size is None or size == (width, height):
            return
        self._push_undo("Resize canvas")
        doc = self.current
        # Content is anchored at the canvas centre: growing shifts every layer
        # out by half the delta so it stays put relative to the new middle.
        dx = (size[0] - width) // 2
        dy = (size[1] - height) // 2
        for layer in doc.layers:
            ox, oy = layer.offset
            layer.offset = (ox + dx, oy + dy)
            self._shift_pose_scene(layer, dx, dy)
        doc.width, doc.height = size
        self._sync_gen_size()
        self._update_title()
        self.render()
        self.set_result(f"Resized canvas to {size[0]} x {size[1]}.")

    def crop_canvas(self, box):
        """Crop the current canvas to ``box`` ``(left, top, right, bottom)``.

        Every layer is folded to canvas coordinates first, then its pixels and
        mask are cropped so content outside the rectangle is discarded.
        """
        width, height = self.current.size
        left, top, right, bottom = (int(round(v)) for v in box)
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        new_w = right - left
        new_h = bottom - top
        if new_w < 1 or new_h < 1:
            self.set_result("Crop area is empty.")
            return
        self._push_undo("Crop canvas")
        doc = self.current
        doc.bake_offsets()
        for layer in doc.layers:
            # A link layer's pixels belong to its master, so only its offset
            # moves with the new origin; its content is not cropped. Its own
            # mask still follows the master content it shapes.
            if layer.is_link:
                ox, oy = layer.offset
                layer.offset = (ox - left, oy - top)
                layer.crop_mask((left, top, right, bottom))
                continue
            layer.set_image(layer.image.crop((left, top, right, bottom)))
            if layer.mask is not None:
                layer.mask = layer.mask.crop((left, top, right, bottom))
            layer.offset = (0, 0)
            for spec in layer.texts:
                spec["x"] = spec.get("x", 0.0) - left
                spec["y"] = spec.get("y", 0.0) - top
            layer.invalidate_texts()
            for spec in layer.paths:
                path.translate(spec, -left, -top)
            layer.invalidate_paths()
            self._shift_pose_scene(layer, -left, -top)
        # Guides are document annotations, so they move with the new origin -
        # unless they are shared across canvases, where cropping one canvas
        # must not disturb the guide set the other tabs draw.
        if not self.guides_shared:
            shifted = []
            for orient, pos in doc.guides:
                pos = pos - (left if orient == "v" else top)
                limit = new_w if orient == "v" else new_h
                if 0 <= pos <= limit:
                    shifted.append((orient, pos))
            doc.guides = shifted
        doc.width, doc.height = new_w, new_h
        view = self.area.current_view()
        if view is not None:
            view.set_selection(None)
        self._sync_gen_size()
        self._update_title()
        self.render()
        self.set_result(f"Cropped canvas to {new_w} x {new_h}.")

    def crop_to_selection(self):
        """Crop the canvas to the active selection's bounding box."""
        if self.current is None:
            self.set_result("There is no canvas to crop.")
            return
        selection = self.selection
        if selection is None or selection.is_empty():
            self.set_result("Select a region first.")
            return
        box = selection.bbox
        width, height = self.current.size
        if box[0] <= 0 and box[1] <= 0 \
                and box[2] >= width and box[3] >= height:
            self.set_result("The selection already covers the canvas.")
            return
        self.crop_canvas(box)

    def crop_to_content(self):
        """Trim the canvas to the non-transparent pixels of the visible layers."""
        if self.current is None:
            self.set_result("There is no canvas to crop.")
            return
        doc = self.current
        box = doc.composite_transparent().getchannel("A").getbbox()
        if box is None:
            self.set_result("The canvas has no visible content to crop to.")
            return
        if box == (0, 0, doc.width, doc.height):
            self.set_result("The canvas is already cropped to its content.")
            return
        self.crop_canvas(box)

    def fit_canvas_to_layers(self):
        """Resize the canvas to the union of the visible layers' extents.

        Layer pixels keep their document position by shifting every layer's
        offset, so content outside the old canvas is preserved rather than
        cropped. Hidden layers are ignored when computing the extents.
        """
        if self.current is None:
            self.set_result("There is no canvas to fit.")
            return
        doc = self.current
        bounds = None
        for layer in doc.layers:
            if not layer.visible or layer.opacity <= 0.0:
                continue
            ox, oy = layer.offset
            right = ox + layer.image.width
            bottom = oy + layer.image.height
            if bounds is None:
                bounds = (ox, oy, right, bottom)
            else:
                bounds = (
                    min(bounds[0], ox), min(bounds[1], oy),
                    max(bounds[2], right), max(bounds[3], bottom),
                )
        if bounds is None:
            self.set_result("There are no visible layers to fit the canvas to.")
            return
        left, top, right, bottom = bounds
        new_w = right - left
        new_h = bottom - top
        if new_w < 1 or new_h < 1:
            self.set_result("The visible layers have no size.")
            return
        if new_w > MAX_CANVAS_SIZE or new_h > MAX_CANVAS_SIZE:
            self.set_result(
                f"The layers need {new_w} x {new_h}, which exceeds the "
                f"{MAX_CANVAS_SIZE} pixel canvas limit."
            )
            return
        if (left, top) == (0, 0) and (new_w, new_h) == doc.size:
            self.set_result("The canvas already fits the visible layers.")
            return
        self._push_undo("Fit canvas to layers")
        for layer in doc.layers:
            ox, oy = layer.offset
            layer.offset = (ox - left, oy - top)
            # Text objects are anchored in document space, so they need the
            # same origin shift as the layer's pixels.
            for spec in layer.texts:
                spec["x"] = spec.get("x", 0.0) - left
                spec["y"] = spec.get("y", 0.0) - top
            layer.invalidate_texts()
            for spec in layer.paths:
                path.translate(spec, -left, -top)
            layer.invalidate_paths()
            self._shift_pose_scene(layer, -left, -top)
        # Guides shift with the new origin, unless they are shared across
        # canvases (see :meth:`crop_canvas`).
        if not self.guides_shared:
            shifted = []
            for orient, pos in doc.guides:
                pos = pos - (left if orient == "v" else top)
                limit = new_w if orient == "v" else new_h
                if 0 <= pos <= limit:
                    shifted.append((orient, pos))
            doc.guides = shifted
        doc.width, doc.height = new_w, new_h
        view = self.area.current_view()
        if view is not None:
            view.set_selection(None)
        self._sync_gen_size()
        self._update_title()
        self.render()
        self.set_result(f"Fitted canvas to {new_w} x {new_h}.")

    def _guide_cuts(self, doc):
        """Interior cut positions the ruler guides define, as ``(xs, ys)``.

        ``xs`` and ``ys`` always include the canvas edges, so the canvas is
        partitioned into ``(len(xs) - 1) * (len(ys) - 1)`` rectangles. A guide
        sitting on an edge (or outside the canvas) adds no cut and is ignored.
        """
        xs = {0, doc.width}
        ys = {0, doc.height}
        for orient, pos in self.guides:
            pos = int(round(pos))
            if orient == "v" and 0 < pos < doc.width:
                xs.add(pos)
            elif orient == "h" and 0 < pos < doc.height:
                ys.add(pos)
        return sorted(xs), sorted(ys)

    def _slice_regions(self, doc):
        """The guide-partitioned rectangles of ``doc``, in reading order.

        Returns a list of ``(left, top, right, bottom)`` boxes; a canvas with
        no interior guides yields a single box covering the whole canvas.
        """
        xs, ys = self._guide_cuts(doc)
        return [
            (xs[xi], ys[yi], xs[xi + 1], ys[yi + 1])
            for yi in range(len(ys) - 1)
            for xi in range(len(xs) - 1)
        ]

    def can_slice(self):
        """True when the active canvas has guides that split it into 2+ regions."""
        return self.current is not None and len(self._slice_regions(self.current)) > 1

    def slice_guides_to_canvases(self):
        """Cut the canvas along its guides into one new canvas per region."""
        if self.current is None:
            self.set_result("There is no canvas to slice.")
            return
        doc = self.current
        regions = self._slice_regions(doc)
        if len(regions) <= 1:
            self.set_result("Add at least one guide inside the canvas to slice it.")
            return
        image = doc.composite_transparent()
        name = doc.name
        for index, (left, top, right, bottom) in enumerate(regions, start=1):
            piece = image.crop((left, top, right, bottom))
            canvas = self._new_canvas(
                name=f"{name} slice {index}",
                size=(right - left, bottom - top),
            )
            canvas.layers[0].set_image(piece)
            self.area.add(canvas)
        self._set_modified(True)
        self._activate_canvas(len(self.canvases) - 1)
        self._refresh_gallery_if_visible()
        self.set_result(
            f"Sliced '{name}' into {len(regions)} canvas(es)."
        )

    def slice_guides_to_layers(self):
        """Cut the canvas along its guides into one new layer per region.

        Each slice is placed at the region's document position (via its layer
        offset), so the slices tile the canvas exactly where the guides were.
        The source layers are left untouched.
        """
        if self.current is None:
            self.set_result("There is no canvas to slice.")
            return
        doc = self.current
        regions = self._slice_regions(doc)
        if len(regions) <= 1:
            self.set_result("Add at least one guide inside the canvas to slice it.")
            return
        image = doc.composite_transparent()
        self._push_undo("Slice to layers")
        for index, (left, top, right, bottom) in enumerate(regions, start=1):
            layer = self._add_layer(name=f"Slice {index}")
            layer.set_image(image.crop((left, top, right, bottom)))
            layer.offset = (left, top)
        self.active = len(self.layers) - 1
        self._group_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(
            f"Sliced '{doc.name}' into {len(regions)} layer(s)."
        )

    def _prompt_rename(self, index):
        if not (0 <= index < len(self.canvases)):
            return
        name = dialogs.ask_string(
            self.root,
            "Rename Canvas",
            "Canvas name:",
            initialvalue=self.canvases[index].name,
            delete_label="Delete",
        )
        if name is dialogs.DELETED:
            self.delete_canvas(index)
            return
        if name is None:
            return
        name = name.strip()
        if not name:
            return
        if name != self.canvases[index].name:
            canvas = self.canvases[index]
            canvas.history.push(canvas, self, "Rename canvas")
            self._set_modified(True)
        self.canvases[index].name = name
        self._refresh_history()
        self._refresh_canvas_tabs()
        if index == self.active_canvas:
            self._update_title()
        self._refresh_gallery_if_visible()

    def _canvas_tab_menu(self, index, x_root, y_root):
        """Right-click menu for a canvas tab: rename, resize, delete and more."""
        if not (0 <= index < len(self.canvases)):
            return
        menu = PopupMenu(self.area.tabs)
        menu.add_command(
            "Rename Canvas...", command=lambda: self._prompt_rename(index)
        )
        menu.add_command("Resize Canvas...", command=self.resize_canvas)
        menu.add_separator()
        menu.add_command(
            "Crop to Selection", command=self.crop_to_selection,
            enabled=lambda: self.selection is not None
            and not self.selection.is_empty(),
        )
        menu.add_command("Crop to Content", command=self.crop_to_content)
        menu.add_command("Fit Canvas to Layers", command=self.fit_canvas_to_layers)
        menu.add_separator()
        menu.add_command(
            "Slice Guides to Canvases", command=self.slice_guides_to_canvases,
            enabled=self.can_slice,
        )
        menu.add_command(
            "Slice Guides to Layers", command=self.slice_guides_to_layers,
            enabled=self.can_slice,
        )
        menu.add_separator()
        menu.add_command("New Canvas...", command=self.add_canvas)
        menu.add_command(
            "Duplicate Canvas", command=lambda: self.duplicate_canvas(index),
            accelerator=lambda: self.shortcuts.accelerator("canvas_duplicate"),
        )
        menu.add_command(
            "Delete Canvas", command=lambda: self.delete_canvas(index),
            enabled=lambda: len(self.canvases) >= 1,
        )
        menu.post(x_root, y_root)

    def _cycle_canvas(self, delta):
        if len(self.canvases) <= 1:
            return
        index = (self.active_canvas + delta) % len(self.canvases)
        self._activate_canvas(index)

    def _reorder_canvases(self, documents, active_doc=None):
        """Apply a new tab order reported after a drag, keeping ``active_doc`` current."""
        documents = list(documents)
        if active_doc is None:
            active_doc = self.current
        if documents != self.canvases:
            self.canvases = documents
            self._set_modified(True)
        self.active_canvas = self.canvases.index(active_doc)
        self._refresh_layer_list()
        self._update_title()
        self.render()

    def _refresh_gallery_if_visible(self):
        """Re-sync the gallery when the canvas list changes while it is open."""
        if self.area.gallery_visible():
            self.area.show_gallery()
        elif self.area.gallery_side_visible():
            self.area.refresh_gallery_side()

    # ------------------------------------------------------------------ gallery
    def show_gallery(self):
        """Show every canvas as a thumbnail board instead of the tab editor.

        A docked sidebar gallery is kept configured (and simply hidden behind
        the board) so it returns when the board is closed.
        """
        self.area.show_gallery()
        if hasattr(self, "gallery_var"):
            self.gallery_var.set(True)
        self.statusbar.set_status("gallery: click to select, double-click to open")
        self._report_gallery_zoom()

    def hide_gallery(self):
        """Leave the gallery and return to the current canvas."""
        self.area.hide_gallery()
        if hasattr(self, "gallery_var"):
            self.gallery_var.set(False)

    def toggle_gallery(self):
        if self.area.gallery_visible():
            self.hide_gallery()
        else:
            self.show_gallery()

    def show_gallery_side(self, side=None):
        """Dock the thumbnail board beside the canvas, keeping the editor live."""
        if side is None:
            side = getattr(self, "_gallery_side_pref", "left")
        self.hide_gallery()
        self.area.show_gallery_side(side)
        self._gallery_side_pref = side
        if hasattr(self, "gallery_side_var"):
            self.gallery_side_var.set(True)
        self.statusbar.set_status(
            "gallery sidebar: click to select, double-click to open"
        )
        self._report_gallery_zoom()

    def hide_gallery_side(self):
        """Remove the docked thumbnail board."""
        self.area.hide_gallery_side()
        if hasattr(self, "gallery_side_var"):
            self.gallery_side_var.set(False)

    def toggle_gallery_side(self):
        if self.area.gallery_side_visible():
            self.hide_gallery_side()
        else:
            self.show_gallery_side()

    def _gallery_closed(self):
        """Sync the menu state when the full-screen gallery closes from inside.

        The docked sidebar gallery keeps its own state, so only the board's
        flag is cleared here.
        """
        if hasattr(self, "gallery_var"):
            self.gallery_var.set(False)

    def _report_gallery_zoom(self):
        # Only the full-screen board has a zoom; the docked strip is a fixed
        # width-fitting list, so its scroll is not reported as a canvas zoom.
        if not self.area.gallery_visible():
            return
        gallery = self.area.gallery
        if gallery is not None:
            self.statusbar.set_zoom(gallery)

    # ------------------------------------------------------------------ layers
    def active_layer(self):
        if self.current is None:
            return None
        return self.current.active_layer()

    def _multi_layer_selection(self):
        """The selected layer indices when more than one layer is selected.

        Returns ``None`` for a single (or empty) selection, so callers keep
        their one-layer path, which acts on the active layer.
        """
        if self.current is None or not hasattr(self, "layer_panel"):
            return None
        indices = list(self.layer_panel.selected_indices(len(self.layers)))
        return indices if len(indices) > 1 else None

    def _layer_pixels_blocked(self, layer):
        """True when ``layer``'s pixels cannot be edited (link, lock, hidden).

        The silent per-layer counterpart of :meth:`_active_layer_locked`, used
        by the multi-layer actions to skip layers instead of aborting.
        """
        return bool(
            layer.is_link or layer.locked
            or (not layer.visible and not self.show_exploded)
        )

    def _active_layer_locked(self, mask=False):
        """True (and reports) when the active layer cannot be modified.

        ``mask=True`` asks about the layer's *mask* instead of its pixels: a
        link layer owns no pixels, but its per-instance mask is its own, so mask
        edits are allowed on it even though pixel edits are not.
        """
        if self.current is None:
            self.set_result("There is no canvas. Add a canvas first.")
            return True
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer. Add a layer first.")
            return True
        if layer.is_link and not mask:
            self.set_result(
                f"Layer '{layer.name}' is a link layer. Edit its master layer."
            )
            return True
        # The exploded view shows every layer, so a hidden layer is editable
        # there; only the explicit lock still blocks it.
        if layer.locked or (not layer.visible and not self.show_exploded):
            if not layer.visible and not self.show_exploded:
                self.set_result(f"Layer '{layer.name}' is hidden (locked).")
            else:
                self.set_result(f"Layer '{layer.name}' is locked.")
            return True
        return False

    def _add_layer(self, name=None, fill=(0, 0, 0, 0), index=None):
        if self.current is None:
            return None
        return self.current.add_layer(name=name, fill=fill, index=index)

    def composite(self):
        return self.current.composite() if self.current is not None else None

    def schedule_live_render(self):
        """Queue one live-stroke repaint, coalescing the events in between.

        A stroke handler that repaints the viewport on every motion event stalls
        the event loop, and the window manager then coalesces motion samples -
        most damaging when zoomed out, where one sample already spans many
        document pixels, so each dropped one chords straight across the curve the
        hand made (the stroke "skips"). The dabs are still stamped in the handler;
        deferring only the repaint keeps the loop draining samples, and the timer
        also lands a trailing frame after the last event, so a stroke held still
        mid-drag still catches up.
        """
        if self._stroke_render_job is not None:
            return
        self._stroke_render_job = self.root.after(
            _LIVE_STROKE_RENDER_MS, self._run_live_render)

    def _run_live_render(self):
        self._stroke_render_job = None
        self.render()

    def flush_live_render(self):
        """Repaint any deferred live frame at once (a stroke press or release)."""
        if self._stroke_render_job is not None:
            try:
                self.root.after_cancel(self._stroke_render_job)
            except tk.TclError:
                pass
            self._stroke_render_job = None
        self.render()

    def render(self):
        if getattr(self, "_loading_project", False):
            # ``project.apply`` is still installing the canvases; panel
            # callbacks fired by its variable writes would paint a half-built
            # project. ``_finish_project_load`` renders once it is complete.
            return
        if self.current is None:
            self._refresh_gallery_if_visible()
            return
        if self.area.gallery_visible():
            # The tab editor is hidden; re-sync the thumbnail board so edits made
            # while it is open (filters, undo, layer changes) show up.
            self._refresh_gallery_if_visible()
            return
        if self.area.gallery_side_visible():
            # A docked board shows beside the live canvas, so refresh its
            # thumbnails while the canvas keeps rendering below.
            self.area.refresh_gallery_side()
        view = self.area.current_view()
        if view is None:
            return
        # The layer panel mirrors each layer's pixels as a tiny preview; let it
        # notice edits here (a stroke, a filter, an undo) without the panel
        # paying for a downscale until the pixels have settled.
        if hasattr(self, "layer_panel"):
            self.layer_panel.sync_previews()
        layer = self.active_layer()
        mask_origin = (0, 0)
        if not self.tools.mask_overlay_show:
            # Overlay hidden: paint the mask blind, showing the raw pixels.
            preview = None
        elif self.tools.inpaint_mask_edit and self.current is not None:
            # The overlay tints the dark side of the mask; the inpaint mask is
            # inverted from a layer mask, so tint the painted (255) region.
            mask = self.current.inpaint_mask
            preview = ImageChops.invert(mask) if mask is not None else None
        elif layer is not None and self.tools.mask_edit and layer.mask is not None:
            preview = layer.mask
            # A layer mask lives in the layer's own buffer frame, which sits at
            # its offset. Layers are normally baked to the canvas, but a link
            # is not - and its mask must still land over the right pixels.
            mask_origin = tuple(layer.offset)
        else:
            preview = None
        view.set_mask_overlay_color(rgba(self.tools.mask_overlay)[:3])
        view.set_mask_preview(preview, mask_origin)
        self.drawing.sync_clone_source()
        # Hand the view providers that flatten only the region it needs, so a
        # repaint costs the viewport size rather than the whole canvas. In
        # exploded view the providers instead lay the layers out in a grid.
        layout = None
        if (self.show_exploded and self.current is not None
                and self.current.layers):
            layout = self._ensure_exploded_layout()
        if layout is not None:
            view.set_layout(layout)
            view.set_source_providers(
                self._exploded_content_bounds, self._exploded_region_composite
            )
            view.set_symmetry(None)
        else:
            view.clear_layout()
            view.set_source_providers(self._content_bounds, self._region_composite)
            view.set_symmetry(self.symmetry_guide())
        # A live brush/eraser/mask stroke paints between mouse moves; hand the
        # view the document rectangle it changed so only that is recomposited
        # instead of the whole viewport (the large-canvas paint lag). The
        # exploded view lays layers out in a grid that the patch cannot model, so
        # it is not patched; the off-canvas preview is, because its provider
        # applies the same off-canvas veil to the patched rectangle, so the
        # patched frame still matches a full overview render.
        live = False
        dirty = None
        drawing = getattr(self, "drawing", None)
        if (drawing is not None and getattr(drawing, "_stroke", None) is not None
                and layout is None):
            live = True
            dirty = getattr(drawing, "_paint_dirty", None)
            drawing._paint_dirty = None
        view.render(live=live, dirty=dirty)

    def _content_bounds(self):
        """Full content bounds ``(origin, size)`` for the view's scroll region."""
        if self.current is None:
            return (0, 0), (0, 0)
        if self._move_bounds is not None:
            # A Move drag holds the scroll region steady so the camera cannot be
            # re-anchored (and the untouched layers jitter) as the moved layer
            # changes the content bounds.
            return self._move_bounds
        if self.show_offscreen:
            left, top, right, bottom = self.current.content_bounds()
            return (left, top), (right - left, bottom - top)
        return (0, 0), self.current.size

    def _region_composite(self, region, scale=1.0, draft=False, anchor=None):
        """Flatten one document ``region`` for viewport-only rendering.

        Passes the active layer index so the canvas can reuse its cached
        backdrop and recomposite only the layer being edited (see
        :meth:`Canvas._flatten_cached`, shared by both composite paths). ``scale``
        is the requested pixels per document pixel; a zoomed-out view passes a
        value below 1 so the flatten is bounded by the viewport's pixels.
        ``draft``/``anchor`` drive the live stroke patch (see
        :meth:`CanvasView._live_composite`).
        """
        if self.current is None:
            return None
        if self.show_offscreen:
            return self.current.composite_overview(
                self.offscreen_opacity, region, scale, self.active,
                draft=draft, anchor=anchor,
            )
        return self.current.composite_region(
            region, self.active, scale, draft=draft, anchor=anchor
        )

    def toggle_offscreen(self):
        """Show or hide layer content that lies outside the canvas."""
        self.show_offscreen = self.offscreen_var.get()
        self.render()
        self.set_result(
            "Showing off-canvas content." if self.show_offscreen
            else "Hiding off-canvas content."
        )

    # ------------------------------------------------------------ exploded view
    def _exploded_key_for(self):
        """A fingerprint of the layer stack's structure (not its pixels).

        Visibility is not part of it: the exploded view shows every layer
        regardless, so hiding or showing one must not rebuild the grid.
        """
        if self.current is None:
            return None
        return tuple(
            (id(layer), layer.image.size)
            for layer in self.current.layers
        )

    def _ensure_exploded_layout(self):
        """The current canvas's exploded layout, rebuilt only when it changes."""
        from paintapp import exploded

        if self.current is None:
            return None
        canvas = self.current
        key = self._exploded_key_for()
        if canvas._exploded_layout is not None and canvas._exploded_key == key:
            return canvas._exploded_layout
        if not canvas.layers:
            canvas._exploded_layout = None
        else:
            canvas._exploded_layout = exploded.compute(canvas)
        canvas._exploded_key = key
        return canvas._exploded_layout

    def _exploded_content_bounds(self):
        """Full layout bounds ``(origin, size)`` for the scroll region."""
        layout = self._ensure_exploded_layout()
        if layout is None:
            return (0, 0), (0, 0)
        return layout.origin, layout.size

    def _exploded_region_composite(self, region, scale=1.0, draft=False,
                                   anchor=None):
        """Flatten one exploded-layout ``region`` for viewport-only rendering.

        ``draft``/``anchor`` are accepted so the provider matches the view's
        call shape; the exploded grid is not patched, so they are ignored.
        """
        layout = self._ensure_exploded_layout()
        if layout is None or self.current is None:
            return None
        return self.current.composite_layout(layout, region, scale)

    def toggle_exploded(self):
        """Switch the canvas between the stacked and exploded-grid views."""
        if hasattr(self, "exploded_var"):
            # A menu checkbutton flips its variable before invoking this
            # command, so when the variable disagrees with the canvas's actual
            # state the click chose the new value. A keyboard shortcut calls
            # this directly without flipping the variable, so there the state
            # must be inverted instead.
            requested = bool(self.exploded_var.get())
            if requested == self.show_exploded:
                requested = not self.show_exploded
            self.set_exploded(requested)
        else:
            self.set_exploded(not self.show_exploded)

    def set_exploded(self, enabled):
        """Show (``True``) or hide (``False``) the exploded-grid view."""
        enabled = bool(enabled)
        if enabled and self.current is None:
            return
        view = self.area.current_view()
        self.show_exploded = enabled
        if hasattr(self, "exploded_var"):
            self.exploded_var.set(enabled)
        if enabled:
            # A path, warp, puppet or transform session is anchored in document
            # space; leave it behind rather than draw it misaligned in a cell.
            self.drawing.cancel_path()
            self.drawing.discard_warp()
            self.drawing.discard_puppet()
            if self._transform_mode_active():
                self.drawing.cancel_transform()
            self.area.set_rulers_visible(False)
        else:
            if self.current is not None:
                self.current._exploded_layout = None
            self.area.set_rulers_visible(self.show_rulers)
        self.render()
        if enabled:
            layout = self.current._exploded_layout if self.current else None
            if view is not None and layout is not None and layout.size[0] > 0:
                # Frame the tile a pending preview now sits on instead of the
                # whole board, so it is the thing that fills the viewport.
                rect = (0, 0, layout.size[0], layout.size[1])
                if view.has_generation_preview():
                    rect = view.preview_fit_bounds() or rect
                view.fit_document_rect(rect)
        self.set_result(
            "Exploded layers on \u2014 click a tile to edit that layer."
            if enabled else "Exploded layers off."
        )

    def set_offscreen_opacity(self, value):
        self.offscreen_opacity = value
        if hasattr(self, "opacity_var"):
            self.opacity_var.set(value)
        if self.show_offscreen:
            self.render()

    def toggle_rulers(self):
        """Show or hide the top and left ruler strips around the canvas."""
        self.set_rulers(not self.show_rulers)
        self.set_result(
            "Rulers shown." if self.show_rulers else "Rulers hidden."
        )

    def set_rulers(self, visible):
        """Show (``True``) or hide (``False``) the ruler strips."""
        self.show_rulers = bool(visible)
        if hasattr(self, "ruler_var"):
            self.ruler_var.set(self.show_rulers)
        self.area.set_rulers_visible(self.show_rulers)

    def toggle_fullscreen(self):
        """Toggle the main window between windowed and fullscreen (F11)."""
        self.set_fullscreen(not self.is_fullscreen())

    def set_fullscreen(self, fullscreen):
        """Enter (``True``) or leave (``False``) fullscreen mode.

        On Windows Tk always fills the primary monitor, so the monitor the
        window currently occupies is captured first and the window is then
        moved onto it: fullscreen covers the display the user is working on
        rather than the primary one.
        """
        fullscreen = bool(fullscreen)
        target = _monitor_rect(self.root) if fullscreen else None
        self.root.attributes("-fullscreen", fullscreen)
        if target is not None:
            _move_window_to(self.root, target)
        if hasattr(self, "fullscreen_var"):
            self.fullscreen_var.set(fullscreen)
        self._refresh_native_frame()
        self.set_result("Fullscreen on." if fullscreen else "Fullscreen off.")

    def _refresh_native_frame(self):
        """Restore the themed native frame after a fullscreen toggle.

        Entering or leaving fullscreen makes Windows rebuild the toplevel's
        native frame, which drops the DWM dark-mode attributes applied at
        startup, so the title bar reverts to the light system color. Reapplying
        them once the new frame exists keeps dark mode intact. Scheduled idle so
        it runs after the rebuilt frame is in place.
        """
        try:
            self.root.after_idle(
                lambda: theme.refresh_native_frame(self.root)
            )
        except tk.TclError:
            pass

    def is_fullscreen(self):
        """Whether the main window is currently fullscreen."""
        try:
            return bool(int(self.root.attributes("-fullscreen")))
        except (tk.TclError, TypeError, ValueError):
            return False

    def _guides_changed(self):
        """Snapshot the document just before a guide is added, moved or removed."""
        self._push_undo("Guides")

    def toggle_guides(self):
        """Show or hide the ruler guides over every canvas."""
        self.set_guides(not self.show_guides)
        self.set_result(
            "Guides shown." if self.show_guides else "Guides hidden."
        )

    def set_guides(self, visible):
        """Show (``True``) or hide (``False``) the ruler guides."""
        self.show_guides = bool(visible)
        if hasattr(self, "guide_var"):
            self.guide_var.set(self.show_guides)
        self.area.set_guides_visible(self.show_guides)

    def toggle_grid(self):
        """Show or hide the grid overlay over every canvas."""
        self.set_grid(not self.show_grid)
        self.set_result(
            "Grid shown." if self.show_grid else "Grid hidden."
        )

    def set_grid(self, visible):
        """Show (``True``) or hide (``False``) the grid overlay."""
        self.show_grid = bool(visible)
        if hasattr(self, "grid_var"):
            self.grid_var.set(self.show_grid)
        self.area.set_grid_visible(self.show_grid)

    def toggle_grid_snap(self):
        """Enable or disable snapping drawing coordinates to the grid."""
        self.set_grid_snap(not self.grid_snap)
        self.set_result(
            "Grid snapping on." if self.grid_snap else "Grid snapping off."
        )

    def set_grid_snap(self, enabled):
        """Enable (``True``) or disable (``False``) grid snapping."""
        self.grid_snap = bool(enabled)
        if hasattr(self, "grid_snap_var"):
            self.grid_snap_var.set(self.grid_snap)
        self.area.set_grid_snap(self.grid_snap)

    def set_layer_outline(self, enabled):
        """Show or hide the active layer's bounds outline on the canvas.

        Toggled from the Layers panel; drawn in the same accent color the
        exploded view uses for the selected cell.
        """
        self.show_layer_outline = bool(enabled)
        if hasattr(self, "layer_panel"):
            self.layer_panel.set_layer_outline(self.show_layer_outline)
        self.area.set_layer_outline_visible(self.show_layer_outline)

    def set_grid_size(self, size):
        """Set the grid cell size in document pixels."""
        try:
            size = max(1, int(size))
        except (TypeError, ValueError):
            return
        self.grid_size = size
        if hasattr(self, "grid_size_var"):
            self.grid_size_var.set(self.grid_size)
        self.area.set_grid_size(self.grid_size)
        self.set_result(f"Grid size set to {self.grid_size} px.")

    def clear_guides(self):
        """Delete every guide currently in effect.

        In shared mode that is the one list every canvas draws; otherwise it is
        the active canvas's own guides.
        """
        if self.guides_shared:
            if not self.shared_guides:
                self.set_result("There are no guides to clear.")
                return
            self._push_undo("Clear guides")
            self.shared_guides = []
            self._sync_shared_guides()
            self.set_result("Cleared guides.")
            return
        doc = self.current
        if doc is None:
            self.set_result("There is no canvas.")
            return
        if not doc.guides:
            self.set_result("The active canvas has no guides.")
            return
        self._push_undo("Clear guides")
        doc.guides = []
        self.render()
        self.set_result("Cleared guides.")

    def set_guides_shared(self, shared):
        """Choose whether every canvas shows the same guides or its own.

        Shared guides live on the app, so every tab draws and edits one list;
        per-canvas guides (the default) stay on each :class:`document.Canvas`.
        Turning sharing on seeds the shared list from the active canvas the
        first time; turning it off leaves each canvas's own guides untouched.
        """
        shared = bool(shared)
        if shared and not self.shared_guides:
            doc = self.current
            guides = doc.guides if doc is not None else []
            self.shared_guides = [tuple(guide) for guide in guides]
        self.guides_shared = shared
        self._sync_shared_guides()
        if hasattr(self, "shared_guides_var"):
            self.shared_guides_var.set(shared)
        self.set_result(
            "Guides are shared by every canvas." if shared
            else "Each canvas keeps its own guides."
        )

    def toggle_shared_guides(self):
        """Switch between shared and per-canvas guides."""
        self.set_guides_shared(not self.guides_shared)

    def _sync_shared_guides(self):
        """Point every canvas view at the shared guide list, or at its own."""
        self.area.set_shared_guides(self.shared_guides if self.guides_shared else None)

    def stat_visible(self, kind):
        """Whether the CPU/RAM/GPU ``kind`` reading is shown in the status bar."""
        return bool(self._stat_visibility.get(kind, True))

    def set_stat_visible(self, kind, visible):
        """Show or hide a CPU/RAM/GPU reading in the status bar."""
        self._stat_visibility[kind] = bool(visible)
        var = getattr(self, f"{kind}_stat_var", None)
        if var is not None:
            var.set(bool(visible))
        self.statusbar.set_stat_visible(kind, visible)

    def set_offscreen_bg(self, rgb):
        """Apply a background color behind off-canvas content on every tab."""
        self.offscreen_bg = tuple(rgb)
        self.area.set_bg_color(self.offscreen_bg)

    def choose_offscreen_bg(self):
        """Prompt for the off-canvas background color with live preview."""
        chosen = dialogs.ask_offscreen_color(
            self.root,
            self.offscreen_bg,
            DEFAULT_OFFCANVAS_BG,
            self.set_offscreen_bg,
        )
        if chosen is not None:
            self.set_offscreen_bg(chosen)
            self.set_result("Off-canvas background color changed.")

    # ------------------------------------------------------------------ zoom
    def report_zoom(self, view):
        self.drawing.refresh_cursor()
        self.statusbar.set_zoom(view)

    def report_cursor(self, x, y):
        self.statusbar.set_cursor(x, y)

    def _report_layer(self):
        layer = self.active_layer()
        if layer is None:
            self.statusbar.set_layer(
                "(no canvas)" if self.current is None else "(no layer)"
            )
            return
        masked = (self.tools.mask_edit and layer.mask is not None) \
            or self.tools.inpaint_mask_edit
        self.statusbar.set_layer(layer.name, masked=masked)

    def _tool_name(self):
        tools = self.tools
        if tools.drag_tool:
            return "drag"
        if tools.move_tool:
            return "move"
        if tools.transform_tool:
            return "transform"
        if tools.arrow_tool:
            return "arrow"
        if tools.paste_tool:
            return "paste"
        if tools.select_tool:
            if getattr(tools, "select_transform", False):
                return "transform"
            return {"rect": "select", "lasso": "lasso", "wand": "wand"}.get(
                getattr(tools, "select_mode", "rect"), "select"
            )
        if tools.layout_tool:
            return "layout"
        if tools.crop_tool:
            return "crop"
        if tools.path_tool:
            return "path"
        if tools.vector_tool:
            return "vector"
        if tools.poses_tool:
            return "poses"
        if tools.liquify_tool:
            return "liquify"
        if tools.warp_tool:
            return "warp"
        if tools.puppet_tool:
            return "puppet"
        if tools.inpaint_mask_edit:
            return "inpaint mask"
        if tools.mask_edit:
            return "mask"
        if tools.text_tool:
            return "text"
        if tools.eyedropper:
            return "pick"
        if tools.fill_tool:
            return "fill"
        if tools.gradient_tool:
            return "gradient"
        if tools.shape_tool:
            return "shape"
        if tools.clone_tool:
            return "clone"
        if tools.smudge_tool:
            return "smudge"
        if tools.dodge_tool:
            return "dodge"
        if tools.burn_tool:
            return "burn"
        if tools.eraser:
            return "eraser"
        return "brush"

    def _report_tool(self):
        name = self._tool_name()
        self.statusbar.set_tool(name, self.tools.brush_size)
        if hasattr(self, "tool_var"):
            self.tool_var.set(name)

    def _set_modified(self, flag):
        self.modified = flag
        self.statusbar.set_modified(flag)

    def zoom_in(self):
        view = self.area.current_view()
        if view is not None:
            view.zoom_in()
            self.report_zoom(view)

    def zoom_out(self):
        view = self.area.current_view()
        if view is not None:
            view.zoom_out()
            self.report_zoom(view)

    def reset_zoom(self):
        view = self.area.current_view()
        if view is not None:
            view.reset_zoom()
            self.report_zoom(view)

    # ------------------------------------------------------------ view rotate
    def rotate_view_cw(self):
        self._rotate_view(15.0)

    def rotate_view_ccw(self):
        self._rotate_view(-15.0)

    def _rotate_view(self, degrees):
        view = self.area.current_view()
        if view is not None:
            view.rotate_view(degrees)
            self.report_zoom(view)

    def flip_view_horizontal(self):
        view = self.area.current_view()
        if view is not None:
            view.flip_view()
            self.report_zoom(view)

    def flip_view_vertical(self):
        view = self.area.current_view()
        if view is not None:
            view.flip_view_vertical()
            self.report_zoom(view)

    def reset_view(self):
        view = self.area.current_view()
        if view is not None:
            view.reset_view()
            self.report_zoom(view)

    def _refresh_layer_list(self, selected=None):
        if self.current is None:
            self.layer_panel.refresh([], 0)
            self._report_layer()
            self._sync_prompt()
            self._sync_gen_settings()
            self._refresh_asset_list()
            self._refresh_preview_bar()
            return
        self._validate_filter_sel()
        self._validate_text_sel()
        self._validate_path_sel()
        self._validate_group_sel()
        self._validate_group_filter_sel()
        self.layer_panel.refresh(
            self.layers, self.active, selected,
            filter_selection=self._filter_sel,
            text_selection=self._text_sel,
            path_selection=self._path_sel,
            group_selection=self._group_sel,
            groups=self.current.groups,
            group_filter_selection=self._group_filter_sel,
        )
        self._report_layer()
        self._sync_prompt()
        self._sync_gen_settings()
        self._refresh_asset_list()
        self._refresh_preview_bar()

    def _refresh_asset_list(self):
        """Show the active canvas's assets in the Assets panel."""
        if not hasattr(self, "assets_panel"):
            return
        assets = self.current.assets if self.current is not None else []
        self.assets_panel.refresh(assets)

    def add_asset_from_file(self):
        """Add an image file to the active canvas's assets."""
        if self.current is None:
            self.set_result("There is no canvas to add an asset to.")
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Add Asset",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            image = Image.open(path)
            image.load()
        except Exception as exc:
            dialogs.show_error(
                self.root, "Add Asset", f"Could not open image:\n{exc}"
            )
            return
        self._push_undo("Add asset")
        asset = self.current.add_asset(
            name=os.path.basename(path), image=image, source="imported"
        )
        self._refresh_asset_list()
        self._select_asset(asset)
        self.set_result(f"Added '{asset.name}' to the canvas assets.")

    def _select_asset(self, asset):
        """Highlight ``asset`` in the Assets panel (best effort)."""
        if not hasattr(self, "assets_panel") or self.current is None:
            return
        try:
            index = self.current.assets.index(asset)
        except ValueError:
            return
        self.assets_panel.refresh(self.current.assets, selected=index)

    def add_layer_to_assets(self, index=None):
        """Store a layer's composited pixels in the active canvas's assets.

        The image is captured as the layer composites - filters, texts, vectors,
        poses, paths and mask folded in - so the asset matches what the layer
        shows. ``index`` defaults to the first selected layer, then the active
        layer.
        """
        if self.current is None:
            self.set_result("There is no canvas to add an asset to.")
            return
        if index is None:
            multi = self._multi_layer_selection()
            if multi is not None:
                self._push_undo("Add layers to assets")
                asset = None
                for i in multi:
                    layer = self.layers[i]
                    image = layer.composited_image()
                    image = (
                        image.copy() if image.mode == "RGBA"
                        else image.convert("RGBA")
                    )
                    asset = self.current.add_asset(
                        name=layer.name, image=image, source="layer",
                    )
                self._refresh_asset_list()
                self._select_asset(asset)
                self.set_result(f"Added {len(multi)} layers to the assets.")
                return
            indices = (
                self.layer_panel.selected_indices(len(self.layers))
                if hasattr(self, "layer_panel") else []
            )
            index = indices[0] if indices else self.active
        if not (0 <= index < len(self.layers)):
            self.set_result("There is no layer to add to the assets.")
            return
        layer = self.layers[index]
        image = layer.composited_image()
        image = image.copy() if image.mode == "RGBA" else image.convert("RGBA")
        self._push_undo("Add layer to assets")
        asset = self.current.add_asset(
            name=layer.name, image=image, source="layer",
        )
        self._refresh_asset_list()
        self._select_asset(asset)
        self.set_result(f"Added layer '{layer.name}' to the assets.")

    def _on_asset_select(self, index):
        """Restore a generated asset's settings into the Generate panel.

        Only generated assets carry the fields their run used; an imported asset
        (or one saved before settings were recorded) leaves the panel alone. The
        load is guarded inside the panel, so it does not echo back into the
        active canvas/layer's own stored settings.
        """
        if self.current is None or not hasattr(self, "gen"):
            return
        asset = self._asset_at(index)
        if asset is None or asset.source != "generated":
            return
        settings = getattr(asset, "settings", None)
        if settings:
            self.gen.apply_asset_settings(settings)
            self._sync_control_panels()
            self.references_panel.set_settings(settings)
            self.controlnet_panel.set_settings(self._control_rows(settings))
            self.controlnet_panel.set_enabled(self._control_enabled(settings))
            self.regions_panel.set_settings(self._region_rows(settings))
            self.regions_panel.set_enabled(self._region_enabled(settings))
            self.tools.layout_model.set_settings(
                self._layout_rows(settings), meta=self._layout_meta(settings)
            )
            self.tools.layout_model.enabled = self._layout_enabled(settings)
            self.tools.layout_model.show_caption = self._layout_show_caption(
                settings)
            self.tools.refresh_layout_settings()
            self._sync_layout_overlay()

    def insert_asset(self, index):
        """Insert an asset onto a new layer above the active one."""
        self._insert_asset_above(index, self.active + 1)

    def insert_asset_at_layer(self, index, layer_index):
        """Insert an asset onto a new layer above the layer at ``layer_index``.

        Used when an asset is dragged out of the Assets panel and dropped on a
        layer row: the new layer lands above the drop target instead of above
        the active layer.
        """
        if not (0 <= layer_index < len(self.layers)):
            return
        self._insert_asset_above(
            index, layer_index + 1, above=self.layers[layer_index].name
        )

    def _insert_asset_above(self, index, position, above=None):
        """Add a layer at ``position`` holding the asset at ``index``."""
        if self.current is None:
            self.set_result("There is no canvas to insert into.")
            return
        asset = self._asset_at(index)
        if asset is None:
            return
        self._push_undo("Insert asset")
        layer = self._add_layer(name=asset.name, index=position)
        layer.set_image(self._centered_asset_image(asset))
        self.active = self.layers.index(layer)
        self._refresh_layer_list()
        self.render()
        if above is None:
            self.set_result(f"Inserted '{asset.name}' as a new layer.")
        else:
            self.set_result(f"Inserted '{asset.name}' above '{above}'.")

    def _centered_asset_image(self, asset):
        """A canvas-sized RGBA buffer holding ``asset`` centered on the canvas.

        New layers are canvas-sized, so the asset is pasted into a full-canvas
        buffer rather than used at its own size; centering keeps an image that
        is smaller than the canvas from landing in the top-left corner.
        """
        image = asset.image
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        buffer = Image.new("RGBA", self.current.size, (0, 0, 0, 0))
        x = (self.current.width - image.width) // 2
        y = (self.current.height - image.height) // 2
        buffer.paste(image, (x, y), image)
        return buffer

    def _drag_asset_over_layer(self, index, x_root, y_root):
        """While an asset is dragged, tint the layer row under the pointer."""
        if not hasattr(self, "layer_panel") or self.current is None:
            return
        target = self.layer_panel.layer_row_at(x_root, y_root)
        self.layer_panel.set_drop_target(target)

    def _drop_asset_on_layer(self, index, x_root, y_root):
        """Drop an asset dragged from the Assets panel onto a layer or canvas.

        A release on a layer row (or one of its filter/text rows) inserts the
        asset above that layer; a release over the canvas inserts it as a new
        layer above the active one. Anywhere else just clears the highlight.
        """
        if self.current is None:
            return
        target = None
        if hasattr(self, "layer_panel"):
            target = self.layer_panel.layer_row_at(x_root, y_root)
            self.layer_panel.set_drop_target(None)
        if target is not None:
            self.insert_asset_at_layer(index, target)
        elif self._drop_over_canvas(x_root, y_root):
            self.insert_asset(index)

    def _drop_over_canvas(self, x_root, y_root):
        """True when ``(x_root, y_root)`` is inside the active canvas view."""
        view = self.area.current_view()
        if view is None:
            return False
        try:
            left = view.winfo_rootx()
            top = view.winfo_rooty()
            right = left + view.winfo_width()
            bottom = top + view.winfo_height()
        except tk.TclError:
            return False
        return left <= x_root <= right and top <= y_root <= bottom

    def save_asset(self, index):
        """Save an asset's pixels to an image file."""
        asset = self._asset_at(index)
        if asset is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save Asset As", defaultextension=".png",
            initialfile=asset.name, filetypes=[("PNG", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        # The asset image is a standalone copy, so it can be written off-thread.
        image = asset.image
        name = os.path.basename(path)

        def runner(progress, cancel):
            cancel.throw_if_cancelled()
            progress(0.1, "saving", None)
            image.save(path)
            progress(1.0, "done", None)
            return name

        def finisher(_result, error, cancelled):
            if cancelled:
                self.statusbar.set_status("cancelled")
                self.set_result(f"Save {name} cancelled.", "warn")
            elif error is not None:
                self._report_export_failure("Save Asset", error)
            else:
                self.set_result(f"Saved '{name}'.")

        self.submit_project_job("export", f"Save {name}", runner, finisher)
        self.set_result(f"Saving {name}\u2026")

    def rename_asset(self, index):
        """Rename an asset (its label only; pixels are untouched)."""
        asset = self._asset_at(index)
        if asset is None:
            return
        name = dialogs.ask_string(
            self.root, "Rename Asset", "Asset name:", initialvalue=asset.name,
        )
        if name is None:
            return
        name = name.strip()
        if not name or name == asset.name:
            return
        self._push_undo("Rename asset")
        asset.name = name
        self._refresh_asset_list()
        self._select_asset(asset)

    def delete_asset(self, index):
        """Remove an asset from the active canvas."""
        asset = self._asset_at(index)
        if asset is None:
            return
        if not dialogs.ask_yes_no(
            self.root, "Delete Asset", f"Delete the asset '{asset.name}'?",
        ):
            return
        self._push_undo("Delete asset")
        self.current.remove_asset(asset)
        self._refresh_asset_list()

    def _asset_at(self, index):
        """The asset at ``index`` on the active canvas, or ``None``."""
        if self.current is None or index is None:
            return None
        assets = self.current.assets
        if 0 <= index < len(assets):
            return assets[index]
        return None

    def selected_asset(self):
        """The asset highlighted in the Assets panel, or ``None``."""
        if not hasattr(self, "assets_panel"):
            return None
        return self._asset_at(self.assets_panel.selected_index())

    def _validate_group_sel(self):
        """Drop the group selection when it no longer belongs to the canvas."""
        if self._group_sel is not None and self._group_sel not in self.current.groups:
            self._group_sel = None

    def _validate_group_filter_sel(self):
        """Drop the group filter selection when it no longer points at a filter."""
        sel = self._group_filter_sel
        if sel is None:
            return
        group, filter_index = sel
        if group not in self.current.groups:
            self._group_filter_sel = None
            return
        if not (0 <= filter_index < len(group.filters)):
            self._group_filter_sel = None

    def _validate_filter_sel(self):
        """Drop the filter selection when it no longer points at a real filter."""
        sel = self._filter_sel
        if sel is None:
            return
        layer_index, filter_index = sel
        if not (0 <= layer_index < len(self.layers)):
            self._filter_sel = None
            return
        if not (0 <= filter_index < len(self.layers[layer_index].filters)):
            self._filter_sel = None

    def _validate_text_sel(self):
        """Drop the text selection when it no longer points at a real object."""
        sel = self._text_sel
        if sel is None:
            return
        layer_index, text_index = sel
        if not (0 <= layer_index < len(self.layers)):
            self._text_sel = None
            return
        if not (0 <= text_index < len(self.layers[layer_index].texts)):
            self._text_sel = None

    def _validate_path_sel(self):
        """Drop the path selection when it no longer points at a real object."""
        sel = self._path_sel
        if sel is None:
            return
        layer_index, path_index = sel
        if not (0 <= layer_index < len(self.layers)):
            self._path_sel = None
            return
        if not (0 <= path_index < len(self.layers[layer_index].paths)):
            self._path_sel = None

    def _on_layer_select(self, index):
        self.active = index
        self._filter_sel = None
        self._text_sel = None
        self._path_sel = None
        self._group_sel = None
        self._group_filter_sel = None
        self.area.set_outline_group(None)
        self.drawing.commit_text()
        self.drawing.finish_path()
        self.layer_panel.set_opacity(self.active_layer().opacity)
        self.layer_panel.update_mask_state(self.active_layer())
        self._report_layer()
        self._sync_prompt()
        self._sync_gen_settings()
        if self.tools.mask_edit and self.active_layer().mask is None:
            self.set_mask_edit(False)
        if self._transform_mode_active():
            self.drawing.sync_transform()
        self.render()

    def select_layer(self, delta):
        """Move the active layer one row up (``-1``) or down (``1``) the list.

        ``delta`` follows the on-screen list, which is drawn top-most-first, so
        the Up and Down arrow keys walk the same column the user sees. Only
        layer rows are visited: group headers and filter, text and path rows are
        skipped, and a layer inside a collapsed group is out of reach. At the
        top or bottom of the list the selection stays put.
        """
        if self.current is None or not self.layers:
            return
        target = self.layer_panel.step_layer(delta)
        if target is None or target == self.active:
            return
        self._on_layer_select(target)
        self._refresh_layer_list()
        self.layer_panel.reveal_layer(target)

    # ----------------------------------------------------------- prompt box
    def _on_positive_prompt(self, text):
        """Store the positive prompt on the active layer so it survives a switch."""
        layer = self.active_layer()
        if layer is not None:
            layer.positive_prompt = text

    def _on_gen_settings(self, values):
        """Store the Generate panel's fields on the active layer and its canvas.

        The whole panel is remembered on both levels, so switching back to a
        layer (or canvas) restores it: the layer's copy is the specific one and
        the canvas's copy is the fallback a layer with none inherits. The
        ControlNet stack is folded in here so both panels share one stored dict
        (and the project format is unchanged).
        """
        if self._loading_project or self.current is None or not values:
            return
        self._store_gen_settings()
        self._sync_control_panels()

    def _on_control_settings(self, rows):
        """Store the ControlNet panel's stack alongside the Generate fields."""
        if self._loading_project or self.current is None:
            return
        self._store_gen_settings()

    def _on_region_settings(self, rows):
        """Store the Regions panel's rows alongside the Generate fields."""
        if self._loading_project or self.current is None:
            return
        self._store_gen_settings()

    def _on_reference_settings(self, values):
        """Store the References panel's fields alongside the Generate fields."""
        if self._loading_project or self.current is None:
            return
        self._store_gen_settings()

    def _on_layout_change(self):
        """Persist the Layout tool's rows and refresh the canvas overlay."""
        if self._loading_project or self.current is None:
            return
        self._store_gen_settings()
        self._sync_layout_overlay()

    def _sync_control_panels(self):
        """Point the References, ControlNet, Regions and Layout panels at the family."""
        family = self.gen.model_family()
        if hasattr(self, "references_panel"):
            self.references_panel.set_family(family)
        if hasattr(self, "controlnet_panel"):
            self.controlnet_panel.set_family(family)
        if hasattr(self, "regions_panel"):
            self.regions_panel.set_family(family)
        if hasattr(self, "tools"):
            self.tools.set_layout_family(family)
        self._sync_layout_overlay()

    def _store_gen_settings(self):
        """Write the merged Generate + References + ControlNet + Regions fields."""
        if self._syncing_settings:
            # A panel reconfigured itself while stored settings were being shown;
            # do not let that echo overwrite what the canvas already holds.
            return
        values = self.gen.settings()
        if values is None:
            return
        values = dict(values)
        if hasattr(self, "references_panel"):
            refs = self.references_panel.settings()
            if refs:
                values.update(refs)
        if hasattr(self, "controlnet_panel"):
            values["controlnets"] = self.controlnet_panel.settings()
            values["controlnet_enabled"] = self.controlnet_panel.enabled()
        if hasattr(self, "regions_panel"):
            values["region_prompts"] = self.regions_panel.settings()
            values["region_prompts_enabled"] = self.regions_panel.enabled()
        if hasattr(self, "tools"):
            values["layout"] = self.tools.layout_model.settings()
            values["layout_meta"] = self.tools.layout_model.meta_tags()
            values["layout_enabled"] = self.tools.layout_model.enabled
            values["layout_show_caption"] = self.tools.layout_model.show_caption
        self.current.gen_settings = dict(values)
        layer = self.active_layer()
        if layer is not None:
            # Keep the canvas-level references out of the per-layer copy, so
            # switching layers does not swap them. Layout Control is per-layer,
            # so it is kept on the layer.
            layer_values = dict(values)
            for key in _REFERENCE_SETTING_KEYS:
                layer_values.pop(key, None)
            layer.gen_settings = layer_values

    @staticmethod
    def _control_rows(stored):
        """The stored ControlNet rows, migrating the old single-adapter keys."""
        rows = (stored or {}).get("controlnets")
        if isinstance(rows, list):
            return rows
        legacy = (stored or {}).get("controlnet")
        if legacy and legacy != "(none)":
            return [{
                "kind": None,
                "model": None,
                "display": legacy,
                "hint": HINT_SOURCE,
                "preprocessor": (stored or {}).get("control_preprocessor", "none"),
                "strength": (stored or {}).get("control_strength", 1.0),
                "start": 0.0,
                "end": 1.0,
            }]
        return []

    @staticmethod
    def _region_rows(stored):
        """The stored Region Prompts panel rows (a plain list, or none)."""
        rows = (stored or {}).get("region_prompts")
        return rows if isinstance(rows, list) else []

    @staticmethod
    def _control_enabled(stored):
        """The stored ControlNet master switch (defaults off)."""
        return (stored or {}).get("controlnet_enabled", False) is True

    @staticmethod
    def _region_enabled(stored):
        """The stored Region Prompts master switch (defaults off)."""
        return (stored or {}).get("region_prompts_enabled", False) is True

    @staticmethod
    def _layout_rows(stored):
        """The stored Layout panel rows (a plain list, or none)."""
        rows = (stored or {}).get("layout")
        return rows if isinstance(rows, list) else []

    @staticmethod
    def _layout_meta(stored):
        """The stored Layout meta tags (a plain list, or none)."""
        meta = (stored or {}).get("layout_meta")
        return meta if isinstance(meta, list) else []

    @staticmethod
    def _layout_enabled(stored):
        """The stored Layout master switch (defaults off)."""
        return (stored or {}).get("layout_enabled", False) is True

    @staticmethod
    def _layout_show_caption(stored):
        """The stored Layout "show caption in box" switch (defaults off)."""
        return (stored or {}).get("layout_show_caption", False) is True

    def _sync_prompt(self):
        """Show the active layer's positive prompt."""
        if not hasattr(self, "gen"):
            return
        if self.current is None:
            self.gen.set_prompt("")
            return
        self.gen.set_prompt(
            getattr(self.active_layer(), "positive_prompt", "")
        )

    def _sync_gen_settings(self):
        """Show the active canvas/layer's stored generation settings, if any.

        A layer with its own settings wins over the canvas's; a target with none
        falls back to the selected model's defaults, with the size following the
        canvas so a fresh layer starts at the canvas dimensions.
        """
        self._syncing_settings = True
        try:
            self._sync_gen_settings_panels()
        finally:
            self._syncing_settings = False

    def _sync_gen_settings_panels(self):
        if not hasattr(self, "gen") or self.current is None:
            return
        layer = self.active_layer()
        stored = None
        if layer is not None and getattr(layer, "gen_settings", None):
            stored = layer.gen_settings
        elif getattr(self.current, "gen_settings", None):
            stored = self.current.gen_settings
        if stored:
            self.gen.set_settings(stored)
        else:
            self.gen.reset_to_model_defaults()
            self._sync_gen_size()
        self._sync_control_panels()
        # References are canvas-scoped: always read them from the canvas, never
        # from the layer, so switching layers keeps the same reference list.
        # Layout Control is layer-scoped and reads from ``stored`` (the layer,
        # falling back to the canvas), so each layer keeps its own boxes.
        canvas_settings = getattr(self.current, "gen_settings", None) or {}
        if hasattr(self, "references_panel"):
            # Only touch the panel when the stored references actually differ, so
            # a layer switch does not rebuild the rows.
            stored_refs = {
                key: canvas_settings[key]
                for key in _REFERENCE_SETTING_KEYS if key in canvas_settings
            }
            current = self.references_panel.settings() or {}
            current_refs = {
                key: current[key]
                for key in _REFERENCE_SETTING_KEYS if key in current
            }
            if stored_refs != current_refs:
                if stored_refs:
                    self.references_panel.set_settings(canvas_settings)
                else:
                    self.references_panel.reset()
        self.controlnet_panel.set_settings(
            self._control_rows(stored) if stored else []
        )
        self.controlnet_panel.set_enabled(
            self._control_enabled(stored)
        )
        self.regions_panel.set_settings(
            self._region_rows(stored) if stored else []
        )
        self.regions_panel.set_enabled(
            self._region_enabled(stored)
        )
        self.tools.layout_model.set_settings(
            self._layout_rows(stored),
            meta=self._layout_meta(stored),
        )
        self.tools.layout_model.enabled = self._layout_enabled(stored)
        self.tools.layout_model.show_caption = self._layout_show_caption(
            stored)
        self.tools.refresh_layout_settings()
        self._sync_layout_overlay()

    def _sync_gen_size(self):
        """Set the generation size to the active canvas size.

        When the active layer or its canvas already has stored settings, their
        width/height are updated too, so a canvas resize keeps the stored size
        matching what the panel now shows.
        """
        if not hasattr(self, "gen") or self.current is None:
            return
        width, height = self.current.size
        self.gen.set_size(width, height)
        layer = self.active_layer()
        stored = None
        if layer is not None and getattr(layer, "gen_settings", None):
            stored = layer.gen_settings
        elif getattr(self.current, "gen_settings", None):
            stored = self.current.gen_settings
        if stored:
            stored["width"] = int(width)
            stored["height"] = int(height)

    def add_layer(self):
        if self.current is None:
            self.set_result("There is no canvas. Add a canvas first.")
            return
        self._push_undo("New layer")
        layer = self._add_layer(index=self.active + 1)
        self.active = self.layers.index(layer)
        self._group_sel = None
        self._refresh_layer_list()
        self.render()

    def add_vector_layer(self):
        """Add a new vector layer above the active one and select it."""
        if self.current is None:
            self.set_result("There is no canvas. Add a canvas first.")
            return
        self._push_undo("New vector layer")
        layer = self._add_layer(name="Vector Layer", index=self.active + 1)
        layer.vector = True
        self.active = self.layers.index(layer)
        self._group_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Added vector layer '{layer.name}'.")

    def rasterize_vector(self):
        """Bake the active layer's vector lines into pixels."""
        layer = self.active_layer()
        if layer is None or not (
                getattr(layer, "strokes", None) or getattr(layer, "fills", None)):
            self.set_result("The active layer has no vector lines to rasterize.")
            return
        self._push_undo("Rasterize vector")
        layer.rasterize_vectors()
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Rasterized vector layer '{layer.name}'.")

    def rasterize_poses(self):
        """Bake the active layer's poses figures into pixels."""
        layer = self.active_layer()
        if layer is None or not getattr(layer, "poses", None):
            self.set_result("The active layer has no poses to rasterize.")
            return
        self._push_undo("Rasterize poses")
        layer.rasterize_poses()
        self.drawing.sync_poses()
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Rasterized poses on '{layer.name}'.")

    def duplicate_layer(self):
        multi = self._multi_layer_selection()
        if multi is not None:
            self._push_undo("Duplicate layers")
            # Insert from the top down so lower indices stay valid; each copy
            # lands just above its source.
            copies = []
            for index in sorted(multi, reverse=True):
                source = self.layers[index]
                duplicate = source.copy(name=f"{source.name} copy")
                self.layers.insert(index + 1, duplicate)
                copies.append(duplicate)
            selected = sorted(self.layers.index(c) for c in copies)
            self.active = selected[-1]
            self._refresh_layer_list(selected)
            self.render()
            self.set_result(f"Duplicated {len(copies)} layers.")
            return
        indices = self.layer_panel.selected_indices(len(self.layers))
        index = indices[0] if len(indices) == 1 else self.active
        if not (0 <= index < len(self.layers)):
            self.set_result("There is no layer to duplicate.")
            return
        source = self.layers[index]
        self._push_undo("Duplicate layer")
        duplicate = source.copy(name=f"{source.name} copy")
        self.layers.insert(index + 1, duplicate)
        self.active = index + 1
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Duplicated layer '{source.name}'.")

    # ------------------------------------------------------------- link layers
    def _master_layers(self):
        """Every master layer in the project, as ``(canvas, layer)`` pairs.

        Masters can live in any canvas, so the Link Layer picker lists them all;
        the canvas is carried so the menu can disambiguate two masters that
        share a name.
        """
        masters = []
        for canvas in self.canvases:
            for layer in canvas.layers:
                if getattr(layer, "is_master", False):
                    masters.append((canvas, layer))
        return masters

    def _next_master_id(self):
        """A project-unique id for a new master layer."""
        self.master_counter += 1
        return self.master_counter

    def _relink_masters(self):
        """Re-point every link layer at its master after a structural rebuild.

        Called after a project load, an undo/redo restore or a canvas delete:
        the layers are rebuilt, so each link's resolved ``master`` object is
        stale and must be found again by ``link_id``.
        """
        masters = {}
        for canvas in self.canvases:
            for layer in canvas.layers:
                if getattr(layer, "is_master", False) and layer.master_id is not None:
                    masters[layer.master_id] = layer
        for canvas in self.canvases:
            for layer in canvas.layers:
                if getattr(layer, "link_id", None) is not None:
                    layer.master = masters.get(layer.link_id)

    def _links_to_master(self, master_id):
        """Every ``(canvas, layer)`` whose link layer points at ``master_id``."""
        if master_id is None:
            return []
        links = []
        for canvas in self.canvases:
            for layer in canvas.layers:
                if layer.link_id == master_id:
                    links.append((canvas, layer))
        return links

    def convert_to_master(self):
        """Promote the active layer to a master a link layer can point at."""
        multi = self._multi_layer_selection()
        if multi is not None:
            targets = [
                self.layers[i] for i in multi
                if not self.layers[i].is_link and not self.layers[i].is_master
            ]
            if not targets:
                self.set_result("No selected layer can become a master.")
                return
            self._push_undo("Convert to master")
            for layer in targets:
                layer.is_master = True
                layer.master_id = self._next_master_id()
            self._refresh_layer_list(multi)
            self.set_result(f"{len(targets)} layer(s) are now masters.")
            return
        layer = self.active_layer()
        if self.current is None or layer is None:
            self.set_result("There is no layer to convert.")
            return
        if layer.is_link:
            self.set_result("A link layer cannot become a master.")
            return
        if layer.is_master:
            self.set_result(f"Layer '{layer.name}' is already a master.")
            return
        self._push_undo("Convert to master")
        layer.is_master = True
        layer.master_id = self._next_master_id()
        self._refresh_layer_list()
        self.set_result(
            f"Layer '{layer.name}' is now a master (id {layer.master_id})."
        )

    def remove_master(self):
        """Demote the active master, unlinking any layers that point at it.

        With links present a dialog lists them so the user can confirm breaking
        each into an independent copy; otherwise a plain confirmation is shown.
        """
        multi = self._multi_layer_selection()
        if multi is not None:
            self._remove_masters(
                [self.layers[i] for i in multi
                 if getattr(self.layers[i], "is_master", False)],
                multi,
            )
            return
        layer = self.active_layer()
        if self.current is None or layer is None:
            self.set_result("There is no layer to remove as a master.")
            return
        if not getattr(layer, "is_master", False):
            self.set_result(f"Layer '{layer.name}' is not a master.")
            return
        links = self._links_to_master(layer.master_id)
        if links:
            listing = [
                (canvas.name if canvas is not self.current else "This canvas",
                 link.name)
                for canvas, link in links
            ]
            if not dialogs.ask_remove_master(self.root, layer.name, listing):
                return
        elif not dialogs.ask_yes_no(
            self.root,
            "Remove Master Layer",
            f"Remove master status from '{layer.name}'?",
        ):
            return
        self._push_undo("Remove master layer")
        if links:
            self._unlink_links_to(layer.master_id, layer)
        layer.is_master = False
        layer.master_id = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Layer '{layer.name}' is no longer a master.")

    def _remove_masters(self, masters, selected):
        """Demote several masters behind one confirmation and undo step."""
        if not masters:
            self.set_result("No selected layer is a master.")
            return
        listing = []
        for master in masters:
            for canvas, link in self._links_to_master(master.master_id):
                where = canvas.name if canvas is not self.current else "This canvas"
                listing.append((where, f"{link.name}  (of {master.name})"))
        names = ", ".join(m.name for m in masters)
        if listing:
            if not dialogs.ask_remove_master(self.root, names, listing):
                return
        elif not dialogs.ask_yes_no(
            self.root,
            "Remove Master Layer",
            f"Remove master status from {len(masters)} layers?",
        ):
            return
        self._push_undo("Remove master layers")
        for master in masters:
            self._unlink_links_to(master.master_id, master)
            master.is_master = False
            master.master_id = None
        self._refresh_layer_list(selected)
        self.render()
        self.set_result(f"{len(masters)} layer(s) are no longer masters.")

    def master_choices(self):
        """Masters offered by the Link Layer picker: ``(label, layer)`` pairs."""
        choices = []
        for canvas, layer in self._master_layers():
            label = layer.name
            if canvas is not self.current:
                label = f"{layer.name}  ({canvas.name})"
            choices.append((label, layer))
        return choices

    def create_link_layer(self, master):
        """Add a link layer to the active canvas pointing at ``master``."""
        if self.current is None:
            self.set_result("There is no canvas to add a link layer to.")
            return
        if master is None or not getattr(master, "is_master", False):
            self.set_result("Select a master layer to link to.")
            return
        if master.master_id is None:
            master.master_id = self._next_master_id()
        self._push_undo("Create link layer")
        layer = self._add_layer(
            name=f"{master.name} (link)", index=self.active + 1
        )
        layer.link_id = master.master_id
        layer.master = master
        self.active = self.layers.index(layer)
        self._group_sel = None
        self._refresh_layer_list()
        self.render()
        if self._transform_mode_active():
            self.drawing.sync_transform()
        self.set_result(f"Linked to master '{master.name}'.")

    def unlink_layer(self):
        """Break the active link layer into an independent copy of its master."""
        multi = self._multi_layer_selection()
        if multi is not None:
            targets = [self.layers[i] for i in multi if self.layers[i].is_link]
            if not targets:
                self.set_result("No selected layer is a link layer.")
                return
            self._push_undo("Unlink layers")
            for layer in targets:
                layer.adopt_content(layer.content_source())
                layer.link_id = None
                layer.master = None
            self._refresh_layer_list(multi)
            self.render()
            if self._transform_mode_active():
                self.drawing.sync_transform()
            self.set_result(f"Unlinked {len(targets)} layer(s).")
            return
        layer = self.active_layer()
        if layer is None or not layer.is_link:
            self.set_result("The active layer is not a link layer.")
            return
        source = layer.content_source()
        self._push_undo("Unlink layer")
        layer.adopt_content(source)
        layer.link_id = None
        layer.master = None
        self._refresh_layer_list()
        self.render()
        if self._transform_mode_active():
            self.drawing.sync_transform()
        self.set_result(f"Unlinked '{layer.name}' into an independent copy.")

    def _unlink_links_to(self, master_id, master):
        """Break every link to a deleted ``master`` into a local copy."""
        for canvas in self.canvases:
            for layer in canvas.layers:
                if layer.link_id != master_id:
                    continue
                if master is not None:
                    layer.adopt_content(master)
                layer.link_id = None
                layer.master = None

    def reset_link_transform(self):
        """Drop a link layer's per-instance transform (back to the master)."""
        multi = self._multi_layer_selection()
        if multi is not None:
            targets = [
                self.layers[i] for i in multi
                if self.layers[i].is_link
                and getattr(self.layers[i], "transform", None) is not None
            ]
            if not targets:
                self.set_result("No selected link layer has a transform.")
                return
            self._push_undo("Reset link transform")
            for layer in targets:
                layer.clear_transform()
            self._refresh_layer_list(multi)
            self.render()
            if self._transform_mode_active():
                self.drawing.sync_transform()
            self.set_result(f"Reset the transform on {len(targets)} layer(s).")
            return
        layer = self.active_layer()
        if layer is None or not layer.is_link:
            self.set_result("The active layer is not a link layer.")
            return
        if layer.transform is None:
            self.set_result("This link layer has no transform to reset.")
            return
        self._push_undo("Reset link transform")
        layer.clear_transform()
        self._refresh_layer_list()
        self.render()
        if self._transform_mode_active():
            self.drawing.sync_transform()
        self.set_result(f"Reset the transform on '{layer.name}'.")

    def select_master(self):
        """Jump to the active link layer's master (its canvas and row)."""
        layer = self.active_layer()
        if layer is None or not layer.is_link:
            self.set_result("The active layer is not a link layer.")
            return
        master = layer.master
        if master is None:
            self.set_result("The master layer could not be found.")
            return
        for index, canvas in enumerate(self.canvases):
            if master in canvas.layers:
                self._activate_canvas(index)
                self.active = canvas.layers.index(master)
                self._on_layer_select(self.active)
                self.set_result(f"Selected master '{master.name}'.")
                return
        self.set_result("The master layer could not be found.")

    def layer_from_selection(self):
        selection = self.selection
        if selection is None or selection.is_empty():
            self.set_result("Select a region first.")
            return
        if self.active_layer() is None:
            self.set_result("There is no layer to copy from.")
            return
        self._push_undo("Layer from selection")
        image = self._masked_pixels(selection)
        layer = self._add_layer(index=self.active + 1)
        layer.set_image(image)
        self.active = self.layers.index(layer)
        self._group_sel = None
        self._refresh_layer_list()
        self.render()
        left, top, right, bottom = selection.bbox
        self.set_result(
            f"Created layer '{layer.name}' from "
            f"{right - left} x {bottom - top} selection."
        )

    def delete_layer(self, indices=None):
        if not self.layers:
            self.set_result("There is no layer to delete.")
            return
        if indices is None:
            indices = self.layer_panel.selected_indices(len(self.layers))
            if not indices:
                indices = [self.active]
        indices = list(indices)
        for index in indices:
            # A link layer is read-only but can still be deleted; only an
            # explicit lock or a hidden layer blocks deletion.
            if self.layers[index].is_locked() and not self.layers[index].is_link:
                self.set_result("Cannot delete a hidden or locked layer.")
                return
        if len(indices) == 1:
            message = f"Delete the layer '{self.layers[indices[0]].name}'?"
        else:
            message = f"Delete {len(indices)} layers?"
        if not dialogs.ask_yes_no(self.root, "Delete Layer", message):
            return
        self._push_undo("Delete layer")
        # Capture the masters being deleted so any links to them can be broken
        # into local copies instead of dangling.
        deleted_masters = {
            self.layers[index].master_id: self.layers[index]
            for index in indices
            if self.layers[index].is_master
            and self.layers[index].master_id is not None
        }
        for index in reversed(indices):
            del self.layers[index]
        self.current.normalize_groups()
        for master_id, master in deleted_masters.items():
            self._unlink_links_to(master_id, master)
        self.active = min(indices[0], len(self.layers) - 1) if self.layers else 0
        self._refresh_layer_list()
        self.render()
        if self._transform_mode_active():
            self.drawing.sync_transform()

    def rename_layer(self):
        indices = self.layer_panel.selected_indices(len(self.layers))
        index = indices[0] if len(indices) == 1 else self.active
        if not (0 <= index < len(self.layers)):
            self.set_result("There is no layer to rename.")
            return
        layer = self.layers[index]
        name = dialogs.ask_string(
            self.root,
            "Rename Layer",
            "Layer name:",
            initialvalue=layer.name,
            delete_label="Delete",
        )
        if name is dialogs.DELETED:
            self.delete_layer([index])
            return
        if name is None:
            return
        name = name.strip()
        if not name:
            return
        if name != layer.name:
            self._push_undo("Rename layer")
        layer.name = name
        self._refresh_layer_list()
        self.set_result(f"Renamed layer to {name}.")

    def move_layer(self, delta):
        """Move the active layer one block up (``1``) or down (``-1``).

        A layer in a group moves together with the rest of its group, so the
        group's members always stay adjacent. With several layers selected,
        every block holding a selected layer moves one step together; a block
        already at the edge (or stuck behind one that is) stays put.
        """
        multi = self._multi_layer_selection()
        if multi is not None:
            self._move_blocks(set(multi), delta, "Move layers")
            return
        self._move_block(self.active, delta, "Move layer")

    def _move_blocks(self, indices, delta, label):
        """Move every block that holds one of ``indices`` one step by ``delta``."""
        blocks = self._layer_blocks()
        chosen = [any(i in indices for i in b[1]) for b in blocks]
        order = list(range(len(blocks)))
        moved = False
        # Walk from the leading edge so a selected block only swaps with an
        # unselected neighbour; runs of selected blocks move as one.
        slots = range(len(order) - 2, -1, -1) if delta > 0 else range(1, len(order))
        for slot in slots:
            other = slot + delta
            if chosen[order[slot]] and not chosen[order[other]]:
                order[slot], order[other] = order[other], order[slot]
                moved = True
        if not moved:
            return
        selected_layers = [self.layers[i] for i in sorted(indices)]
        active_layer = self.layers[self.active]
        self._push_undo(label)
        ordered = []
        for slot in order:
            ordered.extend(self.layers[i] for i in blocks[slot][1])
        self.current.layers = ordered
        self.active = ordered.index(active_layer)
        self._refresh_layer_list(sorted(ordered.index(l) for l in selected_layers))
        self.render()

    def _layer_blocks(self):
        """The layer stack as ordered blocks: a bare layer, or a group run.

        Returns a list of ``(group, indices)`` in stack order (bottom-to-top),
        where ``group`` is ``None`` for a standalone layer.
        """
        blocks = []
        index = 0
        count = len(self.layers)
        while index < count:
            group = self.layers[index].group
            if group is None:
                blocks.append((None, [index]))
                index += 1
                continue
            member_indices = []
            while index < count and self.layers[index].group is group:
                member_indices.append(index)
                index += 1
            blocks.append((group, member_indices))
        return blocks

    def _move_block(self, index, delta, label):
        """Swap the block containing ``index`` with its neighbour."""
        blocks = self._layer_blocks()
        position = None
        for slot, (_group, indices) in enumerate(blocks):
            if index in indices:
                position = slot
                break
        if position is None:
            return
        neighbour = position + delta
        if neighbour < 0 or neighbour >= len(blocks):
            return
        active_layer = self.layers[self.active]
        self._push_undo(label)
        blocks[position], blocks[neighbour] = blocks[neighbour], blocks[position]
        ordered = []
        for _group, indices in blocks:
            ordered.extend(self.layers[i] for i in indices)
        self.current.layers = ordered
        self.active = self.current.layers.index(active_layer)
        self._refresh_layer_list()
        self.render()

    def reorder_layers(self, source, target, above):
        """Move the row ``source`` beside the ``target`` row in the stack.

        Used when a row is dragged in the Layers panel. ``source`` and
        ``target`` are row descriptors from the panel, either
        ``("layer", index)`` or ``("group", group)``, and ``above`` is whether
        the drop lands above the target row on screen (higher in the stack).

        Dropping a layer on a group's header, or between its members, adds it to
        that group; dropping it on a root layer takes it out of any group it was
        in, so a layer can be dragged into and out of a group directly. A
        dragged group moves with all of its members, so they stay adjacent.
        """
        doc = self.current
        if doc is None or not doc.layers:
            return
        if source[0] == "group":
            self._reorder_group_block(source[1], target, above)
            return
        if source[0] != "layer":
            return
        layers = doc.layers
        source_index = source[1]
        if not (0 <= source_index < len(layers)):
            return
        if target[0] == "layer" and target[1] == source_index:
            return
        layer = layers[source_index]
        original_group = getattr(layer, "group", None)
        rest = [item for item in layers if item is not layer]
        if target[0] == "group":
            group = target[1]
            layer.group = group
            members = group.members(rest)
            if members:
                first = rest.index(members[0])
                last = rest.index(members[-1])
                insert_at = last + 1 if above else first
            else:
                # An empty group has no members to sit next to, so the layer
                # lands at the top of the stack, where the empty header shows.
                insert_at = len(rest)
        elif target[0] == "layer":
            target_layer = layers[target[1]]
            if target_layer is layer:
                return
            target_pos = rest.index(target_layer)
            layer.group = getattr(target_layer, "group", None)
            insert_at = target_pos + 1 if above else target_pos
        else:
            return
        rest.insert(insert_at, layer)
        # A drop that lands where the layer already was, and in the same group,
        # changes nothing visible; do not record an undo step for it.
        if rest == layers and layer.group is original_group:
            return
        active_layer = layers[self.active]
        self._push_undo("Reorder layers")
        doc.layers = rest
        doc.normalize_groups()
        self._clear_row_selections()
        if active_layer in doc.layers:
            self.active = doc.layers.index(active_layer)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Moved '{layer.name}' in the layer stack.")

    def _reorder_group_block(self, group, target, above):
        """Move the block of layers in ``group`` beside the ``target`` block.

        ``target`` is a panel row descriptor: another group's header, or a layer
        whose block is the destination. The group's members move together so
        they stay adjacent.
        """
        doc = self.current
        layers = doc.layers
        if not group.members(layers):
            # An empty group has no place in the stack to move.
            return
        blocks = self._layer_blocks()

        def _slot_of(entries, target):
            if target[0] == "group":
                return next(
                    (slot for slot, (block_group, _i) in enumerate(entries)
                     if block_group is target[1]), None,
                )
            if target[0] == "layer":
                return next(
                    (slot for slot, (_g, indices) in enumerate(entries)
                     if target[1] in indices), None,
                )
            return None

        source_slot = _slot_of(blocks, ("group", group))
        if source_slot is None:
            return
        target_slot = _slot_of(blocks, target)
        if target_slot is None or target_slot == source_slot:
            return
        active_layer = layers[self.active]
        self._push_undo("Reorder group")
        block = blocks.pop(source_slot)
        target_slot = _slot_of(blocks, target)
        blocks.insert(target_slot + 1 if above else target_slot, block)
        ordered = []
        for _block_group, indices in blocks:
            ordered.extend(layers[index] for index in indices)
        doc.layers = ordered
        self._clear_row_selections()
        if active_layer in ordered:
            self.active = ordered.index(active_layer)
        self._group_sel = group
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Moved group '{group.name}' in the layer stack.")

    def _clear_row_selections(self):
        """Drop filter, text and path row selections whose indices just shifted.

        A reorder renumbers the layers, so a stored ``(layer_index, ...)`` no
        longer points at the object the user had selected. The active layer and
        the selected group are tracked by object and survive on their own.
        """
        self._filter_sel = None
        self._text_sel = None
        self._path_sel = None

    def canvas_move_choices(self):
        """Targets for the Layers panel's Move to Canvas command.

        Returns ``(canvas name, canvas index)`` pairs for every canvas except
        the active one, so the layer panel can build the cascade without knowing
        how canvases are stored.
        """
        return [
            (canvas.name, index)
            for index, canvas in enumerate(self.canvases)
            if index != self.active_canvas
        ]

    def move_layer_to_canvas(self, target_index):
        """Move the selected layer(s) from the active canvas to another canvas.

        Layers keep their pixels, masks, filters and editable objects; only the
        owning canvas changes. A moved layer leaves its group behind, since a
        group belongs to one canvas. The move is a single undo step recorded on
        both canvases' histories, so undoing from either tab returns the layers
        to where they came from.
        """
        source = self.current
        if source is None:
            self.set_result("There is no canvas to move a layer from.")
            return
        if not (0 <= target_index < len(self.canvases)):
            self.set_result("That canvas no longer exists.")
            return
        target = self.canvases[target_index]
        if target is source:
            self.set_result("The layer is already on this canvas.")
            return

        indices = self.layer_panel.selected_indices(len(source.layers))
        if not indices:
            indices = [source.active]
        indices = sorted({i for i in indices if 0 <= i < len(source.layers)})
        if not indices:
            self.set_result("There is no layer to move.")
            return
        layers = [source.layers[i] for i in indices]
        for layer in layers:
            if layer.is_locked() and not layer.is_link:
                self.set_result("Cannot move a hidden or locked layer.")
                return

        # Remember each layer's original stack position and group so undo can
        # put it back exactly where it was.
        original = []
        for layer in layers:
            try:
                index = source.layers.index(layer)
            except ValueError:
                continue
            original.append((layer, index, layer.group))
        if not original:
            self.set_result("There is no layer to move.")
            return

        # Insert above the target's active layer. Keeping the selection order
        # preserves how the moved layers stack relative to each other.
        base = target.active + 1 if target.layers else 0
        base = max(0, min(base, len(target.layers)))
        moved = [layer for layer, _index, _group in original]

        def _detach(canvas, layer):
            """Remove ``layer`` from ``canvas`` if present; True when removed."""
            if layer not in canvas.layers:
                return False
            canvas.remove_layer(layer)
            return True

        def _attach(canvas, layer, index, group=None):
            """Insert ``layer`` into ``canvas`` unless it is already there."""
            if layer in canvas.layers:
                return False
            # Re-admit a group that normalize_groups dropped when its last
            # member left, unless its id is already taken by a newer group.
            if group is not None and group not in canvas.groups:
                if any(existing.id == group.id for existing in canvas.groups):
                    group = None
                else:
                    canvas.groups.append(group)
            canvas.insert_layer(layer, index, group=group)
            return True

        def _undo_move():
            for layer in reversed(moved):
                _detach(target, layer)
            for layer, index, group in original:
                _attach(source, layer, index, group)

        def _redo_move():
            for layer, _index, _group in original:
                _detach(source, layer)
            for offset, layer in enumerate(moved):
                _attach(target, layer, base + offset)

        label = (
            f"Move layer to canvas '{target.name}'"
            if len(moved) == 1
            else f"Move {len(moved)} layers to canvas '{target.name}'"
        )
        # Record on both canvases so undo works from either tab. The callbacks
        # are idempotent, so replaying an entry whose change already happened is
        # a no-op rather than a double move.
        source.history.push_action(label, _undo_move, _redo_move)
        target.history.push_action(label, _undo_move, _redo_move)

        _redo_move()
        self._set_modified(True)
        self.active_canvas = target_index
        self._after_history_change()
        self.set_result(
            f"Moved layer '{moved[0].name}' to '{target.name}'."
            if len(moved) == 1
            else f"Moved {len(moved)} layers to '{target.name}'."
        )

    def toggle_visible(self):
        if not self.layers:
            self.set_result("There is no layer to show or hide.")
            return
        indices = self.layer_panel.selected_indices(len(self.layers))
        if not indices:
            indices = [self.active]
        self._push_undo("Toggle visibility")
        for index in indices:
            layer = self.layers[index]
            layer.visible = not layer.visible
        self._refresh_layer_list(indices)
        self.render()

    def _capture_isolation(self):
        """Remember each layer and group's visibility for :meth:`unisolate_layer`."""
        if self.current is None:
            return None
        return (
            self.current,
            {id(layer): layer.visible for layer in self.layers},
            {id(group): group.visible for group in self.current.groups},
        )

    def has_isolation(self):
        """True when the visibility captured by :meth:`isolate_layer` can be restored."""
        state = self._isolate_state
        if state is None:
            return False
        canvas, layer_vis, _ = state
        if canvas is not self.current:
            return False
        return any(id(layer) in layer_vis for layer in self.layers)

    def isolate_layer(self):
        """Hide every layer except the active (or single selected) one."""
        if not self.layers:
            self.set_result("There is no layer to isolate.")
            return
        indices = self.layer_panel.selected_indices(len(self.layers))
        if len(indices) > 1:
            keep = set(indices)
            if all(
                layer.visible == (i in keep)
                for i, layer in enumerate(self.layers)
            ):
                self.set_result("The selected layers are already isolated.")
                return
            self._push_undo("Isolate layers")
            if self._isolate_state is None:
                self._isolate_state = self._capture_isolation()
            for i, layer in enumerate(self.layers):
                layer.visible = i in keep
                group = getattr(layer, "group", None)
                if i in keep and group is not None:
                    group.visible = True
            self._refresh_layer_list(sorted(keep))
            self.render()
            self.set_result(f"Isolated {len(keep)} layers.")
            return
        index = indices[0] if indices else self.active
        if not (0 <= index < len(self.layers)):
            return
        if all(
            layer.visible == (i == index)
            for i, layer in enumerate(self.layers)
        ):
            self.set_result(f"'{self.layers[index].name}' is already isolated.")
            return
        self._push_undo("Isolate layer")
        # Keep the first snapshot taken, so chaining isolates still restores the
        # visibility from before isolation started rather than the last isolate.
        if self._isolate_state is None:
            self._isolate_state = self._capture_isolation()
        for i, layer in enumerate(self.layers):
            layer.visible = (i == index)
        # A hidden group would keep the isolated layer invisible too, so reveal
        # the target's group while leaving the other layers hidden.
        group = getattr(self.layers[index], "group", None)
        if group is not None:
            group.visible = True
        self._refresh_layer_list([index])
        self.render()
        self.set_result(f"Isolated layer '{self.layers[index].name}'.")

    def unisolate_layer(self):
        """Put back the layer and group visibility captured by the last isolate."""
        state = self._isolate_state
        if state is None:
            self.set_result("There is no isolation to restore.")
            return
        canvas, layer_vis, group_vis = state
        if canvas is not self.current:
            self._isolate_state = None
            self.set_result("The isolated layers are on another canvas.")
            return
        # A history change rebuilds the layer objects, so the captured ids no
        # longer match and the snapshot is stale.
        if not any(id(layer) in layer_vis for layer in self.layers):
            self._isolate_state = None
            self.set_result("The layer stack changed; nothing to restore.")
            return
        self._push_undo("Unisolate layer")
        for layer in self.layers:
            if id(layer) in layer_vis:
                layer.visible = layer_vis[id(layer)]
        for group in canvas.groups:
            if id(group) in group_vis:
                group.visible = group_vis[id(group)]
        self._isolate_state = None
        self._refresh_layer_list()
        self.render()
        self.set_result("Restored layer visibility.")

    def toggle_isolate(self):
        """Isolate the active layer, or restore visibility if already isolated."""
        if self.has_isolation():
            self.unisolate_layer()
        else:
            self.isolate_layer()

    def show_all_layers(self):
        """Make every layer on the active canvas visible again."""
        if not self.layers:
            self.set_result("There are no layers to show.")
            return
        if all(layer.visible for layer in self.layers):
            self.set_result("All layers are already visible.")
            return
        self._push_undo("Show all layers")
        for layer in self.layers:
            layer.visible = True
        # Showing everything supersedes any isolation, so drop its snapshot.
        self._isolate_state = None
        self._refresh_layer_list()
        self.render()
        self.set_result("Showing all layers.")

    def toggle_lock(self):
        if not self.layers:
            self.set_result("There is no layer to lock or unlock.")
            return
        indices = self.layer_panel.selected_indices(len(self.layers))
        if not indices:
            indices = [self.active]
        self._push_undo("Toggle lock")
        for index in indices:
            layer = self.layers[index]
            layer.locked = not layer.locked
        self._refresh_layer_list(indices)

    def toggle_clip(self):
        """Clip the selected layers to the layer below, or unclip them."""
        if not self.layers:
            self.set_result("There is no layer to clip.")
            return
        indices = self.layer_panel.selected_indices(len(self.layers))
        if not indices:
            indices = [self.active]
        # The bottom layer has no layer beneath it, so it can only be unclipped.
        targets = [
            index for index in indices
            if self.layers[index].clip or index > 0
        ]
        if not targets:
            self.set_result("The bottom layer has nothing to clip to.")
            return
        self._push_undo("Toggle clipping")
        for index in targets:
            layer = self.layers[index]
            layer.clip = not layer.clip
        self._refresh_layer_list(targets)
        self.render()
        clipped = sum(1 for i in targets if self.layers[i].clip)
        self.set_result(f"{clipped} layer(s) clipped.")

    def _on_opacity(self, _value=None):
        if not self.layers:
            return
        indices = self.layer_panel.selected_indices(len(self.layers))
        if not indices:
            indices = [self.active]
        value = self.layer_panel.opacity_var.get() / 100.0
        for index in indices:
            layer = self.layers[index]
            # Opacity is a per-instance property, so a link layer accepts it
            # even though its pixels are read-only.
            if not layer.is_locked() or layer.is_link:
                layer.opacity = value
        self.render()

    def _on_blend(self, mode):
        """Set the blend mode of the selected layers."""
        if not self.layers:
            return
        indices = self.layer_panel.selected_indices(len(self.layers))
        if not indices:
            indices = [self.active]
        self._push_undo("Change blend mode")
        for index in indices:
            self.layers[index].blend = mode
        self._refresh_layer_list(indices)
        self.render()
        self.set_result(f"Blend mode: {self.layers[self.active].blend}.")

    # ------------------------------------------------------------------- groups
    def _on_group_select(self, group):
        """Select a group header; the active layer is left untouched.

        A group is only a way to organize layers, so selecting its header is not
        an edit target: it does not change which layer tools act on.
        """
        self._group_sel = group
        self._filter_sel = None
        self._text_sel = None
        self._path_sel = None
        self.area.set_outline_group(group)
        self._sync_prompt()

    def _selected_group(self):
        return self.layer_panel.selected_group()

    def add_group(self):
        """Add a new empty group to the layer list."""
        if self.current is None:
            self.set_result("There is no canvas. Add a canvas first.")
            return
        self._push_undo("Add group")
        group = self.current.add_group()
        self._group_sel = group
        self._filter_sel = None
        self._refresh_layer_list()
        self.set_result(f"Added group '{group.name}'.")

    def move_selected_layers_to_group(self, group):
        """Move the selected layers into ``group``."""
        if group not in self.current.groups:
            return
        indices = self.layer_panel.selected_indices(len(self.layers))
        if not indices:
            self.set_result("Select at least one layer to move.")
            return
        if all(self.layers[i].group is group for i in indices):
            self.set_result(f"Already in group '{group.name}'.")
            return
        self._push_undo("Move layers to group")
        for index in indices:
            self.layers[index].group = group
        # A group created empty becomes a normal member-backed group once it is
        # populated, so it is culled again if the layers later leave it.
        group.keep_empty = False
        self.current.normalize_groups()
        self._group_sel = group
        self._filter_sel = None
        self.active = self.current.layers.index(
            group.members(self.current.layers)[-1]
        )
        self._refresh_layer_list()
        self.render()
        self.set_result(
            f"Moved {len(indices)} layer(s) into '{group.name}'."
        )

    def group_selected_layers(self):
        """Wrap the selected layers in a new group."""
        indices = self.layer_panel.selected_indices(len(self.layers))
        if len(indices) < 2:
            self.set_result("Select at least two layers to group.")
            return
        self._push_undo("Group layers")
        members = [self.layers[i] for i in indices]
        group = self.current.new_group(members=members)
        self.current.normalize_groups()
        self._group_sel = group
        self._filter_sel = None
        self.active = self.current.layers.index(group.members(self.current.layers)[-1])
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Grouped {len(members)} layers into '{group.name}'.")

    def ungroup_selected_group(self):
        group = self._selected_group()
        if group is None:
            self.set_result("Select a group first.")
            return
        self._push_undo("Ungroup")
        self.current.remove_group(group)
        self.current.normalize_groups()
        self._group_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Ungrouped '{group.name}'.")

    def flatten_selected_group(self):
        """Merge every member of the selected group into one layer."""
        group = self._selected_group()
        if group is None:
            self.set_result("Select a group first.")
            return
        members = [
            index for index, layer in enumerate(self.layers)
            if layer.group is group
        ]
        if not members:
            return
        if any(self.layers[i].is_locked() for i in members):
            self.set_result("Cannot flatten a hidden or locked layer.")
            return
        self._push_undo("Flatten group")
        target = self.layers[members[0]]
        for index in members:
            self.layers[index].rasterize_texts()
        merged = self.current.merged_image([self.layers[i] for i in members])
        if group.filters:
            # A group filter is non-destructive over the members' combined
            # pixels, so fold the chain into the flattened result before the
            # group opacity, matching how it is composited.
            merged = filters.apply_chain(merged, group.filters)
        if group.opacity < 1.0:
            # The merged pixels carry each member's own opacity but not the
            # group's, so fold the group opacity into the result's alpha.
            alpha = merged.getchannel("A").point(
                lambda value, o=group.opacity: int(value * o)
            )
            merged.putalpha(alpha)
        target.set_image(merged)
        target.offset = (0, 0)
        target.visible = True
        target.opacity = 1.0
        target.mask = None
        target.filters = []
        target.group = None
        for index in reversed(members[1:]):
            del self.layers[index]
        self.current.normalize_groups()
        self.active = self.layers.index(target)
        self._group_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Flattened group into '{target.name}'.")

    def delete_selected_group(self):
        group = self._selected_group()
        if group is None:
            self.set_result("Select a group first.")
            return
        members = [
            index for index, layer in enumerate(self.layers)
            if layer.group is group
        ]
        if members:
            self.delete_layer(members)
            return
        if not dialogs.ask_yes_no(
            self.root, "Delete Group", f"Delete the group '{group.name}'?"
        ):
            return
        self._push_undo("Delete group")
        self.current.remove_group(group)
        self._group_sel = None
        self._refresh_layer_list()
        self.set_result(f"Deleted group '{group.name}'.")

    def rename_selected_group(self):
        group = self._selected_group()
        if group is None:
            self.set_result("Select a group first.")
            return
        name = dialogs.ask_string(
            self.root,
            "Rename Group",
            "Group name:",
            initialvalue=group.name,
            delete_label="Delete",
        )
        if name is dialogs.DELETED:
            self.delete_selected_group()
            return
        if name is None:
            return
        name = name.strip()
        if not name:
            return
        if name != group.name:
            self._push_undo("Rename group")
        group.name = name
        self._refresh_layer_list()
        self.set_result(f"Renamed group to {name}.")

    def toggle_group_visible(self):
        group = self._selected_group()
        if group is None:
            self.set_result("Select a group first.")
            return
        self._push_undo("Toggle group visibility")
        group.visible = not group.visible
        self._refresh_layer_list()
        self.render()

    def toggle_group_collapse(self):
        group = self._selected_group()
        if group is None:
            return
        group.collapsed = not group.collapsed
        self._refresh_layer_list()

    def move_selected_group(self, delta):
        group = self._selected_group()
        if group is None:
            self.set_result("Select a group first.")
            return
        members = [
            index for index, layer in enumerate(self.layers)
            if layer.group is group
        ]
        if members:
            self._move_block(members[0], delta, "Move group")

    def duplicate_selected_group(self):
        """Duplicate the selected group and all its members, just above it."""
        group = self._selected_group()
        if group is None:
            self.set_result("Select a group first.")
            return
        members = [
            index for index, layer in enumerate(self.layers)
            if layer.group is group
        ]
        self._push_undo("Duplicate group")
        name = f"{group.name} copy"
        if members:
            copies = [
                self.layers[index].copy(name=f"{self.layers[index].name} copy")
                for index in members
            ]
            insert_at = members[-1] + 1
            self.layers[insert_at:insert_at] = copies
            duplicate = self.current.new_group(name, copies)
        else:
            duplicate = self.current.add_group(name)
        duplicate.visible = group.visible
        duplicate.opacity = group.opacity
        duplicate.collapsed = group.collapsed
        duplicate.filters = [filters.clone(spec) for spec in group.filters]
        self._group_sel = duplicate
        self._group_filter_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Duplicated group '{group.name}'.")

    # ------------------------------------------------------------------ filters
    def _on_filter_select(self, layer_index, filter_index):
        """Select a filter row in the layer panel and make its layer active."""
        self.active = layer_index
        self._filter_sel = (layer_index, filter_index)
        self._text_sel = None
        self._path_sel = None
        self._group_sel = None
        self._group_filter_sel = None
        layer = self.layers[layer_index]
        self.layer_panel.set_opacity(layer.opacity)
        self.layer_panel.update_mask_state(layer)
        self._report_layer()
        if self.tools.mask_edit and layer.mask is None:
            self.set_mask_edit(False)
        self.render()

    def add_filter(self, filter_id):
        """Append ``filter_id`` to the selected group, else the active layer.

        A group filter is non-destructive and applies to the group's members
        composited together (see ``document.Group``), so one filter edits every
        member at once. When the filter has options to adjust, its settings
        dialog opens straight away; cancelling the dialog drops the new filter.
        """
        if filter_id not in filters.FILTERS:
            return
        group = self._selected_group()
        if group is not None:
            self._push_undo("Add group filter")
            spec = filters.new_spec(filter_id)
            group.filters.append(spec)
            self._group_filter_sel = (group, len(group.filters) - 1)
            self._refresh_layer_list()
            self.render()
            if self._prompt_new_filter(spec, lambda: None) is None:
                self._drop_undo()
                group.filters.remove(spec)
                self._group_filter_sel = None
                self.set_result(f"Cancelled '{filters.label(filter_id)}'.")
            else:
                self.set_result(
                    f"Added '{filters.describe(spec)}' to group '{group.name}'."
                )
            self._refresh_layer_list()
            self.render()
            return
        if self._active_layer_locked():
            return
        self._push_undo("Add filter")
        layer = self.active_layer()
        spec = filters.new_spec(filter_id)
        layer.filters.append(spec)
        layer.invalidate_filters()
        self._filter_sel = (self.active, len(layer.filters) - 1)
        self._refresh_layer_list()
        self.render()
        if self._prompt_new_filter(spec, layer.invalidate_filters) is None:
            self._drop_undo()
            layer.filters.remove(spec)
            self._filter_sel = None
            self.set_result(f"Cancelled '{filters.label(filter_id)}'.")
        else:
            self.set_result(f"Added '{filters.describe(spec)}' to '{layer.name}'.")
        layer.invalidate_filters()
        self._refresh_layer_list()
        self.render()

    def _prompt_new_filter(self, spec, refresh):
        """Open a freshly added filter's settings dialog when it has options.

        ``refresh`` invalidates the edited target's pixels so the live preview
        updates as the controls change. Returns the chosen parameter dict, or
        ``None`` when the dialog is cancelled. Filters without options keep
        their defaults and are not prompted.
        """
        entry = filters.FILTERS.get(spec.get("type"))
        if entry is None or not entry["params"]:
            return spec.get("params", {})

        def preview(params):
            spec["params"] = params
            refresh()
            self.render()

        values = dialogs.ask_filter(self.root, spec, on_preview=preview)
        if values is None:
            return None
        spec["params"] = values
        return values

    def _group_filter_target(self):
        """The ``(group, index)`` of the selected group filter, or ``None``."""
        group = self._selected_group()
        sel = self._group_filter_sel
        if group is None or sel is None or sel[0] is not group:
            return None
        filter_index = sel[1]
        if not (0 <= filter_index < len(group.filters)):
            return None
        return group, filter_index

    def _filter_target(self):
        """The ``(layer, index)`` of the selected filter, or ``None``."""
        sel = self._filter_sel
        if sel is None:
            return None
        layer_index, filter_index = sel
        if not (0 <= layer_index < len(self.layers)):
            return None
        layer = self.layers[layer_index]
        if not (0 <= filter_index < len(layer.filters)):
            return None
        return layer, filter_index

    def _on_filter_edit(self, layer_index, filter_index):
        self._filter_sel = (layer_index, filter_index)
        self.edit_selected_filter()

    def edit_selected_filter(self):
        """Open the parameter dialog for the selected filter.

        Target the selected group filter when a group is selected, otherwise the
        selected layer filter.
        """
        if self._group_filter_target() is not None:
            self.edit_selected_group_filter()
            return
        target = self._filter_target()
        if target is None:
            self.set_result("Select a filter first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        spec = layer.filters[index]
        original = dict(spec.get("params", {}))
        self._push_undo("Edit filter")

        def preview(params):
            spec["params"] = params
            layer.invalidate_filters()
            self.render()

        values = dialogs.ask_filter(
            self.root, spec, delete_label="Delete", on_preview=preview
        )
        if values is dialogs.DELETED:
            spec["params"] = original
            self._drop_undo()
            self.remove_selected_filter()
            return
        if values is None:
            spec["params"] = original
            self._drop_undo()
            layer.invalidate_filters()
            self._refresh_layer_list()
            self.render()
            return
        spec["params"] = values
        layer.invalidate_filters()
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Updated '{filters.describe(spec)}'.")

    def toggle_selected_filter(self):
        """Enable or disable the selected filter."""
        target = self._filter_target()
        if target is None:
            self.set_result("Select a filter first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        self._push_undo("Toggle filter")
        spec = layer.filters[index]
        spec["enabled"] = not spec.get("enabled", True)
        layer.invalidate_filters()
        self._refresh_layer_list()
        self.render()
        state = "enabled" if spec["enabled"] else "disabled"
        self.set_result(f"{filters.label(spec['type'])} {state}.")

    def remove_selected_filter(self):
        """Delete the selected filter from its layer."""
        target = self._filter_target()
        if target is None:
            self.set_result("Select a filter first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        self._push_undo("Remove filter")
        spec = layer.filters.pop(index)
        layer.invalidate_filters()
        self._filter_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Removed '{filters.label(spec['type'])}'.")

    def move_selected_filter(self, delta):
        """Move the selected filter up (``1``) or down (``-1``) the chain."""
        target = self._filter_target()
        if target is None:
            self.set_result("Select a filter first.")
            return
        layer, index = target
        target_index = index + delta
        if target_index < 0 or target_index >= len(layer.filters):
            return
        if layer.is_locked():
            self._active_layer_locked()
            return
        self._push_undo("Reorder filter")
        layer.filters[index], layer.filters[target_index] = (
            layer.filters[target_index], layer.filters[index],
        )
        layer.invalidate_filters()
        self._filter_sel = (self.layers.index(layer), target_index)
        self._refresh_layer_list()
        self.render()

    def clear_filters(self):
        """Remove every filter from the selected group or the active layer."""
        group = self._selected_group()
        if group is not None:
            self.clear_group_filters()
            return
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer with filters.")
            return
        if not layer.filters:
            self.set_result("The active layer has no filters.")
            return
        if self._active_layer_locked():
            return
        self._push_undo("Clear filters")
        layer.filters = []
        layer.invalidate_filters()
        self._filter_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Cleared filters from '{layer.name}'.")

    # --------------------------------------------------------------------- text
    def _on_text_select(self, layer_index, text_index):
        """Select a text row in the layer panel and make its layer active."""
        self.active = layer_index
        self._text_sel = (layer_index, text_index)
        self._path_sel = None
        self._filter_sel = None
        self._group_sel = None
        self._group_filter_sel = None
        layer = self.layers[layer_index]
        self.layer_panel.set_opacity(layer.opacity)
        self.layer_panel.update_mask_state(layer)
        self._report_layer()
        if self.tools.mask_edit and layer.mask is None:
            self.set_mask_edit(False)
        self.render()

    def _text_target(self):
        """The ``(layer, index)`` of the selected text object, or ``None``."""
        sel = self._text_sel
        if sel is None:
            return None
        layer_index, text_index = sel
        if not (0 <= layer_index < len(self.layers)):
            return None
        layer = self.layers[layer_index]
        if not (0 <= text_index < len(layer.texts)):
            return None
        return layer, text_index

    def _on_text_edit(self, layer_index, text_index):
        self._text_sel = (layer_index, text_index)
        self.edit_selected_text()

    def edit_selected_text(self):
        """Re-open the selected text object for inline editing on the canvas."""
        target = self._text_target()
        if target is None:
            self.set_result("Select a text object first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        self.drawing.edit_text_object(layer, index)

    def remove_selected_text(self):
        """Delete the selected text object from its layer."""
        target = self._text_target()
        if target is None:
            self.set_result("Select a text object first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        self.drawing.commit_text()
        self._push_undo("Remove text")
        layer.texts.pop(index)
        layer.invalidate_texts()
        self._text_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Removed text from '{layer.name}'.")

    def move_selected_text(self, delta):
        """Move the selected text up (``1``) or down (``-1``) in draw order."""
        target = self._text_target()
        if target is None:
            self.set_result("Select a text object first.")
            return
        layer, index = target
        target_index = index + delta
        if target_index < 0 or target_index >= len(layer.texts):
            return
        if layer.is_locked():
            self._active_layer_locked()
            return
        self.drawing.commit_text()
        self._push_undo("Reorder text")
        layer.texts[index], layer.texts[target_index] = (
            layer.texts[target_index], layer.texts[index],
        )
        layer.invalidate_texts()
        self._text_sel = (self.layers.index(layer), target_index)
        self._refresh_layer_list()
        self.render()

    def duplicate_selected_text(self):
        """Copy the selected text object, placing the copy just above it."""
        target = self._text_target()
        if target is None:
            self.set_result("Select a text object first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        self.drawing.commit_text()
        copy = textrender.clone(layer.texts[index])
        if copy is None:
            return
        self._push_undo("Duplicate text")
        layer.texts.insert(index + 1, copy)
        layer.invalidate_texts()
        self._text_sel = (self.layers.index(layer), index + 1)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Duplicated text on '{layer.name}'.")

    def toggle_selected_text_visible(self):
        """Show or hide the selected text object without removing it."""
        target = self._text_target()
        if target is None:
            self.set_result("Select a text object first.")
            return
        layer, index = target
        self.drawing.commit_text()
        self._push_undo("Toggle text visibility")
        spec = layer.texts[index]
        if spec.get("hidden"):
            spec.pop("hidden", None)
        else:
            spec["hidden"] = True
        layer.invalidate_texts()
        self._text_sel = (self.layers.index(layer), index)
        self._refresh_layer_list()
        self.render()

    # --------------------------------------------------------------------- path
    def _on_path_select(self, layer_index, path_index):
        """Select a path row in the layer panel and make its layer active."""
        self.active = layer_index
        self._path_sel = (layer_index, path_index)
        self._text_sel = None
        self._filter_sel = None
        self._group_sel = None
        self._group_filter_sel = None
        layer = self.layers[layer_index]
        self.layer_panel.set_opacity(layer.opacity)
        self.layer_panel.update_mask_state(layer)
        self._report_layer()
        if self.tools.mask_edit and layer.mask is None:
            self.set_mask_edit(False)
        self.render()

    def _path_target(self):
        """The ``(layer, index)`` of the selected path object, or ``None``."""
        sel = self._path_sel
        if sel is None:
            return None
        layer_index, path_index = sel
        if not (0 <= layer_index < len(self.layers)):
            return None
        layer = self.layers[layer_index]
        if not (0 <= path_index < len(layer.paths)):
            return None
        return layer, path_index

    def _on_path_edit(self, layer_index, path_index):
        self._path_sel = (layer_index, path_index)
        self.edit_selected_path()

    def edit_selected_path(self):
        """Re-open the selected path object for editing on the canvas."""
        target = self._path_target()
        if target is None:
            self.set_result("Select a path object first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        self.drawing.edit_path_object(layer, index)

    def remove_selected_path(self):
        """Delete the selected path object from its layer."""
        target = self._path_target()
        if target is None:
            self.set_result("Select a path object first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        self.drawing.finish_path()
        self._push_undo("Remove path")
        layer.paths.pop(index)
        layer.invalidate_paths()
        self._path_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Removed path from '{layer.name}'.")

    def move_selected_path(self, delta):
        """Move the selected path up (``1``) or down (``-1``) in draw order."""
        target = self._path_target()
        if target is None:
            self.set_result("Select a path object first.")
            return
        layer, index = target
        target_index = index + delta
        if target_index < 0 or target_index >= len(layer.paths):
            return
        if layer.is_locked():
            self._active_layer_locked()
            return
        self.drawing.finish_path()
        self._push_undo("Reorder path")
        layer.paths[index], layer.paths[target_index] = (
            layer.paths[target_index], layer.paths[index],
        )
        layer.invalidate_paths()
        self._path_sel = (self.layers.index(layer), target_index)
        self._refresh_layer_list()
        self.render()

    def duplicate_selected_path(self):
        """Copy the selected path object, placing the copy just above it."""
        target = self._path_target()
        if target is None:
            self.set_result("Select a path object first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        self.drawing.finish_path()
        copy = path.clone(layer.paths[index])
        if copy is None:
            return
        self._push_undo("Duplicate path")
        layer.paths.insert(index + 1, copy)
        layer.invalidate_paths()
        self._path_sel = (self.layers.index(layer), index + 1)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Duplicated path on '{layer.name}'.")

    def toggle_selected_path_visible(self):
        """Show or hide the selected path object without removing it."""
        target = self._path_target()
        if target is None:
            self.set_result("Select a path object first.")
            return
        layer, index = target
        self.drawing.finish_path()
        self._push_undo("Toggle path visibility")
        spec = layer.paths[index]
        if spec.get("hidden"):
            spec.pop("hidden", None)
        else:
            spec["hidden"] = True
        layer.invalidate_paths()
        self._path_sel = (self.layers.index(layer), index)
        self._refresh_layer_list()
        self.render()

    def rasterize_selected_paths(self):
        """Fold the active layer's path objects into its pixels (destructive)."""
        multi = self._multi_layer_selection()
        if multi is not None:
            self.drawing.finish_path()
            self._bake_layers(
                multi, lambda l: bool(l.paths), lambda l: l.rasterize_paths(),
                "Rasterize paths", "paths", "_path_sel",
            )
            return
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer with paths.")
            return
        if self._active_layer_locked():
            return
        self.drawing.finish_path()
        if not layer.paths:
            self.set_result("The active layer has no paths.")
            return
        self._push_undo("Rasterize paths")
        layer.rasterize_paths()
        self._path_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Rasterized paths on '{layer.name}'.")

    def rasterize_selected_path(self):
        """Bake just the selected path object into the layer's pixels."""
        target = self._path_target()
        if target is None:
            self.set_result("Select a path object first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        if not (0 <= index < len(layer.paths)):
            return
        spec = layer.paths[index]
        self.drawing.finish_path()
        if spec not in layer.paths:
            return
        self._push_undo("Rasterize path")
        layer.paths.remove(spec)
        image = layer.begin_write()
        path.paste(image, spec, layer.offset)
        layer.set_image(image)
        layer.invalidate_paths()
        self._path_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Rasterized path on '{layer.name}'.")

    def _bake_layers(self, indices, has_content, bake, label, noun, sel_attr):
        """Run ``bake`` on every selected layer that ``has_content``.

        Locked, hidden and link layers are skipped rather than aborting the
        whole action; everything baked shares one undo step.
        """
        layers = [self.layers[i] for i in indices]
        candidates = [l for l in layers if has_content(l)]
        targets = [l for l in candidates if not self._layer_pixels_blocked(l)]
        if not targets:
            if candidates:
                self.set_result(f"The selected layers with {noun} are locked.")
            else:
                self.set_result(f"No selected layer has {noun}.")
            return
        self._push_undo(label)
        for layer in targets:
            bake(layer)
        setattr(self, sel_attr, None)
        self._refresh_layer_list(indices)
        self.render()
        skipped = len(candidates) - len(targets)
        message = f"{label} on {len(targets)} layer(s)."
        if skipped:
            message += f" Skipped {skipped} locked layer(s)."
        self.set_result(message)

    def bake_filters(self):
        """Flatten the active layer's filter chain into its pixels (destructive).

        Mirrors 'Apply Layer Mask': the base pixels become what the enabled
        filters produce and the chain is cleared, so later copy/cut/selection
        edits work on the filtered result. When a group is selected the group's
        members are flattened into one layer instead (see ``apply_group_filters``),
        since a group filter acts on their combined pixels.
        """
        group = self._selected_group()
        if group is not None:
            self.apply_group_filters()
            return
        multi = self._multi_layer_selection()
        if multi is not None:
            def bake(layer):
                layer.set_image(layer.filtered_image())
                layer.filters = []
                layer.invalidate_filters()
            self._bake_layers(
                multi, lambda l: bool(l.filters), bake,
                "Apply filters", "filters", "_filter_sel",
            )
            return
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer with filters.")
            return
        if not layer.filters:
            self.set_result("The active layer has no filters.")
            return
        if self._active_layer_locked():
            return
        self._push_undo("Apply filters")
        result = layer.filtered_image()
        layer.set_image(result)
        layer.filters = []
        layer.invalidate_filters()
        self._filter_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Applied filters to '{layer.name}'.")

    def apply_selected_filter(self):
        """Bake just the selected filter into the layer's pixels (destructive).

        Unlike 'Apply Filters' (which flattens the whole chain), this bakes the
        one selected filter and drops it from the chain, leaving the others
        editable.
        """
        target = self._filter_target()
        if target is None:
            self.set_result("Select a filter first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        self._push_undo("Apply filter")
        spec = layer.filters.pop(index)
        if spec.get("enabled", True):
            layer.set_image(filters.apply_chain(layer.image, [spec]))
        else:
            layer.invalidate_filters()
        self._filter_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(
            f"Applied '{filters.label(spec['type'])}' to '{layer.name}'."
        )

    # ------------------------------------------------------------- group filters
    def _on_group_filter_select(self, group, filter_index):
        """Select a group's filter row; the group becomes the filter target."""
        self._group_sel = group
        self._group_filter_sel = (group, filter_index)
        self._filter_sel = None
        self._text_sel = None
        self._path_sel = None
        self._report_layer()

    def _on_group_filter_edit(self, group, filter_index):
        self._group_sel = group
        self._group_filter_sel = (group, filter_index)
        self.edit_selected_group_filter()

    def edit_selected_group_filter(self):
        """Open the parameter dialog for the selected group filter."""
        target = self._group_filter_target()
        if target is None:
            self.set_result("Select a group filter first.")
            return
        group, index = target
        spec = group.filters[index]
        original = dict(spec.get("params", {}))
        self._push_undo("Edit group filter")

        def preview(params):
            spec["params"] = params
            self.render()

        values = dialogs.ask_filter(
            self.root, spec, delete_label="Delete", on_preview=preview
        )
        if values is dialogs.DELETED:
            spec["params"] = original
            self._drop_undo()
            self.remove_selected_group_filter()
            return
        if values is None:
            spec["params"] = original
            self._drop_undo()
            self._refresh_layer_list()
            self.render()
            return
        spec["params"] = values
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Updated '{filters.describe(spec)}'.")

    def toggle_selected_group_filter(self):
        """Enable or disable the selected group filter."""
        target = self._group_filter_target()
        if target is None:
            self.set_result("Select a group filter first.")
            return
        group, index = target
        self._push_undo("Toggle group filter")
        spec = group.filters[index]
        spec["enabled"] = not spec.get("enabled", True)
        self._refresh_layer_list()
        self.render()
        state = "enabled" if spec["enabled"] else "disabled"
        self.set_result(f"{filters.label(spec['type'])} {state}.")

    def remove_selected_group_filter(self):
        """Delete the selected filter from its group."""
        target = self._group_filter_target()
        if target is None:
            self.set_result("Select a group filter first.")
            return
        group, index = target
        self._push_undo("Remove group filter")
        spec = group.filters.pop(index)
        self._group_filter_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Removed '{filters.label(spec['type'])}'.")

    def move_selected_group_filter(self, delta):
        """Move the selected group filter up (``1``) or down (``-1``)."""
        target = self._group_filter_target()
        if target is None:
            self.set_result("Select a group filter first.")
            return
        group, index = target
        target_index = index + delta
        if target_index < 0 or target_index >= len(group.filters):
            return
        self._push_undo("Reorder group filter")
        group.filters[index], group.filters[target_index] = (
            group.filters[target_index], group.filters[index],
        )
        self._group_filter_sel = (group, target_index)
        self._refresh_layer_list()
        self.render()

    def clear_group_filters(self):
        """Remove every filter from the selected group."""
        group = self._selected_group()
        if group is None:
            self.set_result("Select a group first.")
            return
        if not group.filters:
            self.set_result(f"Group '{group.name}' has no filters.")
            return
        self._push_undo("Clear group filters")
        group.filters = []
        self._group_filter_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Cleared filters from group '{group.name}'.")

    def apply_group_filters(self):
        """Bake a group's filter chain by flattening the group (destructive).

        A group filter acts on the members' combined pixels, so baking it per
        member would change the result (a blur would leave seams). Flattening the
        group folds the chain into the merged pixels instead; see
        :meth:`flatten_selected_group`.
        """
        self.flatten_selected_group()

    def rasterize_selected_texts(self):
        """Fold the active layer's text objects into its pixels (destructive).

        Mirrors 'Apply Filters': the text specs are rendered into the base
        pixels and the list is cleared, so later copy/cut/selection edits work
        on the flattened glyphs instead of live text.
        """
        multi = self._multi_layer_selection()
        if multi is not None:
            self.drawing.commit_text()
            self._bake_layers(
                multi, lambda l: bool(l.texts), lambda l: l.rasterize_texts(),
                "Rasterize texts", "text", "_text_sel",
            )
            return
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer with text.")
            return
        if self._active_layer_locked():
            return
        self.drawing.commit_text()
        if not layer.texts:
            self.set_result("The active layer has no text.")
            return
        self._push_undo("Rasterize texts")
        layer.rasterize_texts()
        self._text_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Rasterized texts on '{layer.name}'.")

    def rasterize_selected_text(self):
        """Bake just the selected text object into the layer's pixels.

        Unlike 'Rasterize Texts' (which flattens every text object), this bakes
        the one selected object and drops it, leaving the others editable.
        """
        target = self._text_target()
        if target is None:
            self.set_result("Select a text object first.")
            return
        layer, index = target
        if layer.is_locked():
            self._active_layer_locked()
            return
        if not (0 <= index < len(layer.texts)):
            return
        spec = layer.texts[index]
        self.drawing.commit_text()
        if spec not in layer.texts:
            return
        self._push_undo("Rasterize text")
        layer.texts.remove(spec)
        layer.set_image(textrender.paste(layer.image, spec, layer.offset))
        layer.invalidate_texts()
        self._text_sel = None
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Rasterized text on '{layer.name}'.")

    # ------------------------------------------------------------------ masks
    def add_layer_mask(self, invert=False):
        """Add a mask to the active layer from the selection (or fully visible).

        ``invert=True`` hides the selected region instead of revealing it (or
        hides everything when there is no selection). A link layer accepts one
        too: it masks that instance alone, on top of whatever the master's own
        mask already does.
        """
        if self._active_layer_locked(mask=True):
            return
        layer = self.active_layer()
        if layer.mask is not None:
            self.set_result(f"Layer '{layer.name}' already has a mask.")
            return
        self._push_undo("Add mask")
        self.current.bake_offsets()
        layer.mask = self._mask_from_selection(layer, invert)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Added mask to '{layer.name}'.")

    def _mask_from_selection(self, layer, invert=False):
        """A fresh layer-sized mask holding the selection, or reveal/hide-all.

        The selection lives in document pixels while the mask lives in the
        layer's own buffer frame, so the selection is placed at the layer's
        offset. Layers are normally baked to the canvas first (see
        ``bake_offsets``), which makes the two frames coincide; a link is not
        baked, so this placement is what keeps its mask aligned.
        """
        selection = self.selection
        if selection is None or selection.is_empty():
            return Image.new("L", layer.size, 0 if invert else 255)
        mask = Image.new("L", layer.size, 0)
        ox, oy = layer.offset
        mask.paste(selection.mask, (-ox, -oy))
        if invert:
            mask = mask.point(lambda value: 255 - value)
        return mask

    def apply_layer_mask(self):
        """Bake the active layer's mask into its alpha channel and drop it."""
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer.")
            return
        if layer.mask is None:
            self.set_result("The active layer has no mask.")
            return
        if layer.is_link:
            self.set_result(
                f"Layer '{layer.name}' is a link layer; unlink it to apply its mask."
            )
            return
        if self._active_layer_locked():
            return
        self._push_undo("Apply mask")
        self.current.bake_offsets()
        layer.set_image(layer.effective_image())
        layer.mask = None
        self.set_mask_edit(False)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Applied mask to '{layer.name}'.")

    def delete_layer_mask(self):
        """Discard the active layer's mask, restoring full visibility."""
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer.")
            return
        if layer.mask is None:
            self.set_result("The active layer has no mask.")
            return
        self._push_undo("Delete mask")
        layer.mask = None
        self.set_mask_edit(False)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Deleted mask from '{layer.name}'.")

    def set_mask_edit(self, enabled):
        """Enter or leave mask-editing mode for the active layer."""
        enabled = bool(enabled)
        layer = self.active_layer()
        if layer is None:
            enabled = False
        elif enabled:
            if self._active_layer_locked(mask=True):
                enabled = False
            elif layer.mask is None:
                self._push_undo("Add mask")
                self.current.bake_offsets()
                layer.mask = Image.new("L", layer.size, 255)
                self._refresh_layer_list()
                self.render()
        tools = self.tools
        if enabled:
            tools.eyedropper = tools.eraser = False
            tools.fill_tool = False
            tools.select_tool = False
            tools.path_tool = False
            tools.vector_tool = False
            tools.liquify_tool = False
            tools.warp_tool = False
            tools.puppet_tool = False
            tools.paste_tool = False
            tools.arrow_tool = False
            tools.move_tool = tools.drag_tool = tools.transform_tool = False
            tools.text_tool = False
            tools.brush_tool = False
        if enabled:
            tools.inpaint_mask_edit = False
            if hasattr(self, "gen"):
                self.gen.set_inpaint_mask_edit(False)
        tools.mask_edit = enabled
        if not enabled and not tools._any_tool_active():
            tools.arrow_tool = True
        tools._sync_buttons()
        self.layer_panel.set_mask_edit(enabled)
        # This method repaints below, so record the state to keep the mode-change
        # handler from rendering a second time for the same change.
        self._mask_preview_state = (bool(tools.mask_edit),
                                    bool(tools.inpaint_mask_edit))
        self._on_mode_change()
        self.render()

    def toggle_mask_edit(self):
        self.set_mask_edit(not self.tools.mask_edit)

    def stroke_surface(self):
        """The surface the current stroke targets.

        While the inpaint mask is being painted that is the canvas's inpaint
        mask; otherwise it is the active layer.
        """
        if self.tools.inpaint_mask_edit and self.current is not None:
            return self.current.inpaint_mask_surface()
        return self.active_layer()

    def set_inpaint_mask_edit(self, enabled):
        """Enter or leave inpaint-mask painting for the active canvas.

        The mask is painted with the Mask tool's brush (value, hardness, opacity)
        and shown with the mask overlay, but it marks where generation may change
        pixels rather than how the layer is revealed.
        """
        enabled = bool(enabled) and self.current is not None
        tools = self.tools
        if enabled:
            tools.eyedropper = tools.eraser = False
            tools.fill_tool = False
            tools.select_tool = False
            tools.path_tool = False
            tools.vector_tool = False
            tools.liquify_tool = False
            tools.warp_tool = False
            tools.puppet_tool = False
            tools.paste_tool = False
            tools.arrow_tool = False
            tools.move_tool = tools.drag_tool = tools.transform_tool = False
            tools.text_tool = False
            tools.brush_tool = False
            tools.mask_edit = False
            self.current.inpaint_mask_surface()
            # Inpaint semantics are reversed from a layer mask (255 = regenerate),
            # so paint the marked value by default and restore it on exit.
            if not tools.inpaint_mask_edit:
                self._mask_value_before_inpaint = tools.mask_value
                tools.mask_value = 255
        tools.inpaint_mask_edit = enabled
        if not enabled and not tools._any_tool_active():
            tools.arrow_tool = True
        tools._sync_buttons()
        if hasattr(self, "gen"):
            self.gen.set_inpaint_mask_edit(enabled)
        # This method repaints below, so record the state to keep the mode-change
        # handler from rendering a second time for the same change.
        self._mask_preview_state = (bool(tools.mask_edit),
                                    bool(tools.inpaint_mask_edit))
        self._on_mode_change()
        self.render()
        if enabled:
            self.set_result("Painting the inpaint mask; generation replaces it.")

    def toggle_inpaint_mask_edit(self):
        self.set_inpaint_mask_edit(not self.tools.inpaint_mask_edit)

    # --------------------------------------------------------- mask settings
    def set_mask_overlay_color(self, color_hex):
        """Set the tint used to preview the mask while editing it."""
        self.mask_overlay_color = tuple(rgba(color_hex)[:3])
        self.render()

    def _on_mask_action(self, action, *args):
        """Dispatch a button from the mask settings panel."""
        inpaint = self.tools.inpaint_mask_edit
        if action == "invert":
            (self.invert_inpaint_mask if inpaint else self.invert_layer_mask)()
        elif action == "reset":
            (self.reset_inpaint_mask if inpaint else self.reset_layer_mask)()
        elif action == "feather":
            radius = args[0] if args else 4
            (self.feather_inpaint_mask if inpaint else self.feather_layer_mask)(radius)
        elif action == "overlay":
            self.render()

    def _active_mask_layer(self):
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer.")
            return None
        if layer.mask is None:
            self.set_result("The active layer has no mask.")
            return None
        return layer

    def invert_layer_mask(self):
        """Invert the active layer's mask so hidden and revealed areas swap."""
        layer = self._active_mask_layer()
        if layer is None or self._active_layer_locked(mask=True):
            return
        self._push_undo("Invert mask")
        layer.mask = ImageChops.invert(layer.mask)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Inverted mask on '{layer.name}'.")

    def reset_layer_mask(self):
        """Fill the active layer's mask with reveal-all, discarding its paint."""
        layer = self._active_mask_layer()
        if layer is None or self._active_layer_locked(mask=True):
            return
        self._push_undo("Reset mask")
        layer.mask = Image.new("L", layer.mask.size, 255)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Reset mask on '{layer.name}'.")

    def feather_layer_mask(self, radius):
        """Blur the active layer's mask to soften its edges."""
        layer = self._active_mask_layer()
        if layer is None or self._active_layer_locked(mask=True):
            return
        radius = max(1, int(radius))
        self._push_undo("Feather mask")
        layer.mask = layer.mask.filter(ImageFilter.GaussianBlur(radius))
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Feathered mask on '{layer.name}' by {radius}px.")

    def invert_inpaint_mask(self):
        """Invert the canvas's inpaint mask, swapping regenerate and keep."""
        doc = self.current
        if doc is None or doc.inpaint_mask is None:
            return
        self._push_undo("Invert inpaint mask")
        doc.inpaint_mask = ImageChops.invert(doc.inpaint_mask)
        doc._inpaint_mask_shared = False
        self.render()
        self.set_result("Inverted the inpaint mask.")

    def reset_inpaint_mask(self):
        """Clear the canvas's inpaint mask so nothing is marked for generation."""
        doc = self.current
        if doc is None or doc.inpaint_mask is None:
            return
        self._push_undo("Reset inpaint mask")
        doc.inpaint_mask = None
        doc._inpaint_mask_shared = False
        self.render()
        self.set_result("Cleared the inpaint mask.")

    def feather_inpaint_mask(self, radius):
        """Blur the canvas's inpaint mask to soften its edges."""
        doc = self.current
        if doc is None or doc.inpaint_mask is None:
            return
        radius = max(1, int(radius))
        self._push_undo("Feather inpaint mask")
        doc.inpaint_mask = doc.inpaint_mask.filter(ImageFilter.GaussianBlur(radius))
        doc._inpaint_mask_shared = False
        self.render()
        self.set_result(f"Feathered the inpaint mask by {radius}px.")

    def merge_layers(self):
        indices = self.layer_panel.selected_indices(len(self.layers))
        if len(indices) < 2:
            self.set_result("Select at least two layers to merge.")
            return
        for index in indices:
            if self.layers[index].is_locked():
                self.set_result("Cannot merge a hidden or locked layer.")
                return
        self._push_undo("Merge layers")
        target_index = indices[0]
        target = self.layers[target_index]
        # Fold any editable text, strokes, poses and paths into pixels so the
        # merged result contains them and the target keeps no duplicate (now
        # drawn) specs.
        for index in indices:
            layer = self.layers[index]
            layer.rasterize_texts()
            layer.rasterize_vectors()
            layer.rasterize_poses()
            layer.rasterize_paths()
        target.set_image(self.current.merged_image([self.layers[i] for i in indices]))
        target.offset = (0, 0)
        target.visible = True
        target.opacity = 1.0
        target.mask = None
        # The merged pixels already include every source layer's filters, so the
        # target keeps none to avoid applying them a second time.
        target.filters = []
        # Merging layers that all belong to one group keeps the result in that
        # group; merging across groups (or from the root) leaves it at the root.
        merged_group = target.group
        if merged_group is not None and any(
            self.layers[i].group is not merged_group for i in indices
        ):
            target.group = None
        for index in reversed(indices[1:]):
            del self.layers[index]
        self.current.normalize_groups()
        self.active = self.layers.index(target)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Merged {len(indices)} layers into {target.name}.")

    def clear(self):
        multi = self._multi_layer_selection()
        if multi is not None:
            layers = [self.layers[i] for i in multi]
            if any(self._layer_pixels_blocked(l) for l in layers):
                self.set_result("Cannot clear a hidden, locked or link layer.")
                return
            if not dialogs.ask_yes_no(
                self.root, "Clear Layer", f"Clear {len(layers)} layers?",
            ):
                return
            self.drawing.commit_text()
            self.drawing.finish_path()
            self._push_undo("Clear layers")
            self.current.bake_offsets()
            for layer in layers:
                self._wipe_layer(layer)
            self._filter_sel = None
            self._text_sel = None
            self._path_sel = None
            self._refresh_layer_list(multi)
            self.set_result("")
            self.render()
            return
        if self._active_layer_locked():
            return
        layer = self.active_layer()
        if not dialogs.ask_yes_no(
            self.root,
            "Clear Layer",
            f"Clear the layer '{layer.name}'?",
        ):
            return
        self.drawing.commit_text()
        self.drawing.finish_path()
        self._push_undo("Clear layer")
        self.current.bake_offsets()
        self._wipe_layer(layer)
        self._filter_sel = None
        self._text_sel = None
        self._path_sel = None
        self._refresh_layer_list()
        self.set_result("")
        self.render()

    def _wipe_layer(self, layer):
        """Erase ``layer``'s pixels and drop its filters and editable objects."""
        layer.begin_write()
        # Clear the whole buffer, not just the canvas rectangle: a layer moved
        # past the canvas edge can carry off-canvas pixels in a larger image
        # (see ``align_for_painting``), which a canvas-sized rectangle misses.
        image_width, image_height = layer.image.size
        layer.draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0, 0))
        layer.filters = []
        layer.texts = []
        layer.paths = []
        layer.strokes = []
        layer.vector = False
        layer.poses = []
        layer.invalidate_filters()
        layer.invalidate_texts()
        layer.invalidate_paths()
        layer.invalidate_vectors()
        layer.invalidate_poses()

    # ------------------------------------------------------------------ edit
    def _history(self):
        """The active canvas's undo stack, or ``None`` when there is no canvas."""
        canvas = self.current
        return canvas.history if canvas is not None else None

    @property
    def can_undo(self):
        history = self._history()
        return history is not None and history.can_undo

    @property
    def can_redo(self):
        history = self._history()
        return history is not None and history.can_redo

    def _push_undo(self, label="Edit"):
        """Snapshot the active canvas before a mutation on it."""
        canvas = self.current
        if canvas is None:
            return
        canvas.history.push(canvas, self, label)
        self._set_modified(True)
        self._refresh_history()

    def _drop_undo(self):
        """Forget the last snapshot on the active canvas (cancels a live edit)."""
        canvas = self.current
        if canvas is not None:
            canvas.history.drop_last()

    def _refresh_history(self):
        if hasattr(self, "history_panel"):
            canvas = self.current
            self.history_panel.refresh(canvas.history if canvas is not None else None)

    def _on_history_select(self, index):
        canvas = self.current
        if canvas is None:
            return
        self.drawing.commit_text()
        # A jump that runs a custom action (for example a layer moved to another
        # canvas) leaves the layer objects in place, so prompts and settings need
        # no positional capture and re-apply. A pure snapshot jump still does.
        extras = (
            None if canvas.history.crosses_action(index)
            else self._capture_canvas_extras()
        )
        if canvas.history.jump(canvas, self, index):
            if extras is not None:
                self._apply_canvas_extras(extras)
            self._after_history_change()
            self.set_result("Moved to history step.")

    def _after_history_change(self):
        # A restore re-points the views at fresh documents, so a pending
        # generation preview would no longer belong to what is shown.
        self._clear_generation_preview()
        self.area.retarget(self.canvases)
        # A restore rebuilds the layer objects, so link layers must be pointed
        # back at their masters by id.
        self._relink_masters()
        # A rebuild drops each view's shared-guide pointer; re-install it so the
        # restored shared list (or per-canvas mode) is what every tab draws.
        self._sync_shared_guides()
        self._activate_canvas(self.active_canvas)
        self._refresh_history()

    def _capture_canvas_extras(self):
        """The active canvas's generation prompts and settings, before a restore.

        These are deliberately independent of undo/redo: they live on the
        document so they follow a canvas or layer switch, but a history change
        must not roll them back. Capturing and re-applying them keeps the prompt
        box and the Generate panel untouched across the layer rebuild a restore
        performs.
        """
        canvas = self.current
        if canvas is None:
            return None
        return {
            "layers": [layer.positive_prompt for layer in canvas.layers],
            "settings": dict(canvas.gen_settings),
            "layer_settings": [dict(layer.gen_settings) for layer in canvas.layers],
        }

    def _apply_canvas_extras(self, extras):
        """Re-apply what :meth:`_capture_canvas_extras` captured after a restore.

        Matched by position: the restored layers line up with the captured ones
        in the common case, so the active prompt and settings survive.
        """
        canvas = self.current
        if canvas is None or not extras:
            return
        for layer, text in zip(canvas.layers, extras["layers"]):
            layer.positive_prompt = text
        canvas.gen_settings = dict(extras.get("settings", {}))
        for layer, settings in zip(canvas.layers, extras.get("layer_settings", [])):
            layer.gen_settings = dict(settings)

    def undo(self):
        canvas = self.current
        if canvas is None:
            self.set_result("Nothing to undo.")
            return
        self.drawing.commit_text()
        if not canvas.history.can_undo:
            self.set_result("Nothing to undo.")
            return
        # A custom action (for example a layer moved to another canvas) carries
        # its own inverse and does not rebuild this canvas from a snapshot, so
        # generation prompts and settings must not be captured and re-applied.
        is_action = canvas.history.top_undo_is_action
        extras = None if is_action else self._capture_canvas_extras()
        canvas.history.undo(canvas, self)
        if not is_action:
            self._apply_canvas_extras(extras)
        self._after_history_change()
        self.set_result("Undo.")

    def redo(self):
        canvas = self.current
        if canvas is None:
            self.set_result("Nothing to redo.")
            return
        self.drawing.commit_text()
        if not canvas.history.can_redo:
            self.set_result("Nothing to redo.")
            return
        is_action = canvas.history.top_redo_is_action
        extras = None if is_action else self._capture_canvas_extras()
        canvas.history.redo(canvas, self)
        if not is_action:
            self._apply_canvas_extras(extras)
        self._after_history_change()
        self.set_result("Redo.")

    def _text_has_focus(self):
        try:
            widget = self.root.focus_get()
        except (KeyError, tk.TclError):
            # ttk.Combobox pop-downs are internal toplevels that Tkinter
            # cannot map back to a widget. They are not text entry fields.
            return False
        if widget is None:
            return False
        return widget.winfo_class() in {
            "Text", "Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox",
        }

    def _masked_pixels(self, selection):
        """The active layer's pixels inside ``selection``, transparent outside."""
        layer = self.active_layer()
        if layer is None:
            return None
        out = Image.new("RGBA", self.current.size, (0, 0, 0, 0))
        out.alpha_composite(layer.image, dest=layer.offset)
        out.putalpha(ImageChops.multiply(out.getchannel("A"), selection.mask))
        return out

    def _clear_masked(self, image, mask):
        inverse = mask.point(lambda value: 255 - value)
        image.putalpha(ImageChops.multiply(image.getchannel("A"), inverse))

    def copy(self):
        selection = self.selection
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer to copy.")
            return
        if (selection is None or selection.is_empty()) \
                and getattr(layer, "vector", False) \
                and (layer.strokes or getattr(layer, "fills", None)):
            # A vector layer's pixels do not include its strokes, so copy the
            # editable lines themselves (as SVG the plain Paste understands).
            self.copy_vector_svg()
            return
        if selection is not None and not selection.is_empty():
            self._clipboard = self._masked_pixels(selection)
            left, top, right, bottom = selection.bbox
            self.set_result(f"Copied {right - left} x {bottom - top} selection.")
        else:
            self._clipboard = layer.image.copy()
            self.set_result(f"Copied layer '{layer.name}'.")
        self._remember_clipboard_origin()

    def cut(self):
        if self._active_layer_locked():
            return
        self._push_undo("Cut")
        self.current.bake_offsets()
        layer = self.active_layer()
        selection = self.selection
        if selection is not None and not selection.is_empty():
            self._clipboard = self._masked_pixels(selection)
            self._clear_masked(layer.image, selection.mask)
            layer.invalidate_filters()
            self.render()
            left, top, right, bottom = selection.bbox
            self.set_result(f"Cut {right - left} x {bottom - top} selection.")
        else:
            self._clipboard = layer.image.copy()
            layer.set_image(Image.new("RGBA", self.current.size, (0, 0, 0, 0)))
            self.render()
            self.set_result(f"Cut layer '{layer.name}'.")
        self._remember_clipboard_origin()

    def paste(self):
        if self.current is None:
            self.set_result("There is no canvas. Add a canvas first.")
            return
        dest = self.tools.paste_dest
        fit = self.tools.paste_fit_canvas
        if self.tools.paste_ask:
            layer = self.active_layer()
            chosen = dialogs.ask_paste_dest(
                self.root, layer.name if layer is not None else None)
            if chosen is None:
                return
            dest, fit = chosen
        # Prefer the app's own clipboard (regions/layers copied inside the app);
        # fall back to an image sitting on the OS clipboard. The in-app copy is
        # only reused while the OS clipboard still holds what it did at copy
        # time, so a freshly copied external image is not shadowed forever.
        external = self._grab_clipboard_image()
        if (self._clipboard is not None
                and self._clipboard_mark != self._clipboard_fingerprint(external)):
            self._clipboard = None
        if dest == "asset":
            self._paste_into_assets(external)
            return
        if self._clipboard is None or self._clipboard.getbbox() is None:
            if dest == "selected_layer" and self._active_layer_locked():
                self.set_result("The active layer is locked.")
                return
            if not self._paste_external_image(external, dest=dest, fit=fit):
                if dest == "asset" or not self._paste_clipboard_svg(dest, external):
                    self.set_result("Clipboard is empty.")
            return
        bounds = self._clipboard.getbbox()
        if dest == "selected_layer" and self._active_layer_locked():
            self.set_result("The active layer is locked.")
            return
        self._push_undo("Paste")
        self.current.bake_offsets()
        layer = self.active_layer()
        selection = self.selection
        if fit:
            buf = self.current.fit_image(self._clipboard)
        else:
            if selection is not None and not selection.is_empty():
                dest_x, dest_y = selection.bbox[:2]
            else:
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                dest_x = (self.current.width - width) // 2
                dest_y = (self.current.height - height) // 2
            buf = Image.new("RGBA", self.current.size, (0, 0, 0, 0))
            buf.paste(self._clipboard, (dest_x - bounds[0], dest_y - bounds[1]),
                      self._clipboard)
        if dest == "new_layer":
            layer = self.current.add_layer(
                name="Pasted",
                index=(
                    self.current.layers.index(self.active_layer()) + 1
                    if self.active_layer() is not None else len(self.current.layers)
                ),
            )
            layer.set_image(buf)
            self.active = self.current.layers.index(layer)
            self._refresh_layer_list()
            self.render()
            self.set_result(f"Pasted into new layer '{layer.name}'.")
            return
        layer.set_image(Image.alpha_composite(layer.image, buf))
        self.render()
        self.set_result(f"Pasted into layer '{layer.name}'.")

    def _paste_into_assets(self, data=None):
        """Add the clipboard image to the active canvas's assets.

        Uses the app's own in-app copy when it is still current, otherwise the
        image (or image file) on the OS clipboard.
        """
        if self.current is None:
            self.set_result("There is no canvas to paste into.")
            return
        name = "Pasted"
        image = None
        if self._clipboard is not None and self._clipboard.getbbox() is not None:
            image = self._clipboard.copy()
        else:
            if data is None:
                data = self._grab_clipboard_image()
            if isinstance(data, Image.Image):
                image = data
            elif isinstance(data, list):
                for path in data:
                    try:
                        image = Image.open(path)
                    except Exception:
                        continue
                    name = os.path.basename(path)
                    break
        if image is None:
            self.set_result("Clipboard is empty.")
            return
        self._push_undo("Paste asset")
        asset = self.current.add_asset(name=name, image=image, source="imported")
        self._refresh_asset_list()
        self._select_asset(asset)
        self.set_result(f"Pasted '{asset.name}' into assets.")

    def _paste_external_image(self, data=None, dest=None, fit=None):
        """Create a new layer from an image on the OS clipboard.

        Returns ``True`` when an image was pasted. Handles both raw image data
        (copied from a viewer or browser) and image files copied in Explorer,
        which arrive as a list of file paths. By default the image keeps its
        native pixels and the layer is centered by offset so content larger than
        the canvas survives; when ``fit`` (or the ``Fit to canvas`` paste option
        when ``fit`` is ``None``) is on, the image is scaled to the canvas
        instead, matching Open Image. With the ``Paste in selected layer``
        destination the pixels are composited onto the active layer instead of a
        fresh one.

        ``data`` is the already-grabbed clipboard contents when the caller has
        them, to avoid grabbing twice. ``fit`` overrides the stored option for
        one call, so the paste dialog can decide per paste.
        """
        if data is None:
            data = self._grab_clipboard_image()
        if dest is None:
            dest = self.tools.paste_dest
        if fit is None:
            fit = self.tools.paste_fit_canvas
        name = "Pasted"
        image = None
        if isinstance(data, Image.Image):
            image = data
        elif isinstance(data, list):
            for path in data:
                try:
                    image = Image.open(path)
                except Exception:
                    continue
                name = os.path.basename(path)
                break
        if image is None:
            return False
        image = self._import_to_working(image)
        if dest == "asset":
            self._paste_into_assets(image)
            return True
        if dest == "selected_layer" and self._active_layer_locked():
            return False
        self._push_undo("Paste image")
        if dest == "selected_layer":
            layer = self.active_layer()
            if fit:
                buf = self.current.fit_image(image).convert("RGBA")
            else:
                image = image.convert("RGBA")
                buf = Image.new("RGBA", self.current.size, (0, 0, 0, 0))
                width, height = self.current.size
                iw, ih = image.size
                buf.paste(image, ((width - iw) // 2, (height - ih) // 2))
            layer.set_image(Image.alpha_composite(layer.image, buf))
            self.render()
            self.set_result(f"Pasted {name} into layer '{layer.name}'.")
            return True
        layer = self._add_layer(name=name, index=self.active + 1)
        if fit:
            layer.set_image(self.current.fit_image(image))
        else:
            image = image.convert("RGBA")
            layer.set_image(image)
            width, height = self.current.size
            iw, ih = image.size
            layer.offset = ((width - iw) // 2, (height - ih) // 2)
        self.active = self.layers.index(layer)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Pasted {name} as a new layer.")
        return True

    def _grab_clipboard_image(self):
        """The raw OS clipboard contents, or ``None`` when unavailable."""
        try:
            return ImageGrab.grabclipboard()
        except Exception:
            return None

    @staticmethod
    def _clipboard_fingerprint(data):
        """A cheap fingerprint of OS clipboard ``data``, or ``None`` when empty.

        Used to detect whether the OS clipboard was replaced since the app put
        something on its own clipboard, so Paste can stop reusing a stale copy.
        """
        if isinstance(data, Image.Image):
            thumb = data.convert("RGBA").resize((16, 16))
            return ("image", data.size, thumb.tobytes())
        if isinstance(data, list) and data:
            return ("files", tuple(data))
        return None

    def _remember_clipboard_origin(self):
        """Record the OS clipboard state alongside a fresh in-app copy."""
        self._clipboard_mark = self._clipboard_fingerprint(
            self._grab_clipboard_image())

    def fill(self):
        if self._active_layer_locked():
            return
        self._push_undo("Fill")
        self.current.bake_offsets()
        layer = self.active_layer()
        layer.begin_write()
        layer.draw.rectangle((0, 0, self.current.width, self.current.height),
                             fill=rgba(self.color))
        layer.invalidate_filters()
        self.render()
        self.set_result(f"Filled layer '{layer.name}'.")

    # ------------------------------------------------------------------ text
    def default_text_style(self):
        """The current text-tool style, with defaults for anything unset."""
        base = {
            "family": fonts.default_family(),
            "size": textrender.DEFAULT_SIZE,
            "bold": False,
            "italic": False,
            "align": "left",
            "color": self.color,
            "line_spacing": textrender.DEFAULT_LINE_SPACING,
            "letter_spacing": 0.0,
            "stroke": None,
            "shadow": None,
            "wrap": True,
        }
        for key, value in (self._text_style or {}).items():
            if key in base and value is not None:
                base[key] = value
        # When a text object is open for editing, its own style wins so the
        # panel mirrors what is on the canvas rather than the last-used style.
        drawing = getattr(self, "drawing", None)
        spec = drawing.current_text_style() if drawing is not None else None
        if spec is not None:
            for key in base:
                if key in spec:
                    base[key] = spec[key]
        try:
            size = int(base["size"])
        except (TypeError, ValueError):
            size = textrender.DEFAULT_SIZE
        base["size"] = max(textrender.MIN_SIZE, min(size, textrender.MAX_SIZE))
        return base

    def _on_text_style(self, style):
        """Persist a Text-panel change and apply it to the text being edited."""
        self._text_style = dict(style)
        self.drawing.update_text_style(style)

    def _transform_layer(self, transform, label):
        if self._active_layer_locked():
            return
        self._push_undo(label)
        layer = self.active_layer()
        # Text objects live in document space; fold them into pixels so the
        # geometric transform moves the glyphs along with the rest of the layer.
        layer.rasterize_texts()
        image = transform(layer.image)
        layer.set_image(image)
        # Center the transformed pixels on the canvas but keep the full buffer,
        # letting ``offset`` retain anything that lands outside the canvas.
        layer.offset = (
            (self.current.width - image.width) // 2,
            (self.current.height - image.height) // 2,
        )
        if layer.mask is not None:
            layer.mask = transform(layer.mask)
        self.render()
        self.set_result(f"{label} layer '{layer.name}'.")

    def flip_horizontal(self):
        self._transform_layer(
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT), "Flipped"
        )

    def flip_vertical(self):
        self._transform_layer(
            lambda img: img.transpose(Image.FLIP_TOP_BOTTOM), "Flipped"
        )

    def rotate_cw(self):
        self._transform_layer(
            lambda img: img.transpose(Image.ROTATE_270), "Rotated"
        )

    def rotate_ccw(self):
        self._transform_layer(
            lambda img: img.transpose(Image.ROTATE_90), "Rotated"
        )

    def rotate_180(self):
        self._transform_layer(
            lambda img: img.transpose(Image.ROTATE_180), "Rotated"
        )

    # ---------------------------------------------------- canvas-level transform
    def _transform_canvas(self, kind, label):
        """Rotate or flip the whole canvas, not just the active layer.

        ``kind`` is one of ``"flip_h"``, ``"flip_v"``, ``"rotate_cw"`` or
        ``"rotate_ccw"``. Every layer's pixels and mask are transposed and its
        offset repositioned so off-canvas content stays put, the canvas inpaint
        mask and ruler guides move with the document, and the canvas width and
        height swap for a 90 degree turn. A canvas flip or rotate is the whole
        document, so it is recorded as one undo step.

        Editable path/vector geometry lives in document pixels and is re-mapped
        exactly, as are text anchors and the poses camera for a rotation. A
        horizontal/vertical flip cannot be represented for re-rendered glyphs or
        a 3D poses scene - mirroring reverses their handedness - so text and
        poses are first rasterised into the layer's pixels, exactly as the
        per-layer flip/rotate already bakes text.
        """
        if self.current is None:
            self.set_result("There is no canvas to transform.")
            return
        doc = self.current
        width, height = doc.size
        rotate = kind in ("rotate_cw", "rotate_ccw")
        mirror = not rotate
        transpose = {
            "flip_h": Image.FLIP_LEFT_RIGHT,
            "flip_v": Image.FLIP_TOP_BOTTOM,
            "rotate_cw": Image.ROTATE_270,
            "rotate_ccw": Image.ROTATE_90,
        }[kind]
        # Pillow's ROTATE_270 turns clockwise, matching ``rotate_cw``.
        theta = 90.0 if kind == "rotate_cw" else -90.0

        def map_point(x, y):
            """Map a document point through the canvas transform."""
            if kind == "flip_h":
                return (width - x, y)
            if kind == "flip_v":
                return (x, height - y)
            if kind == "rotate_cw":
                return (height - y, x)
            return (y, width - x)

        self._push_undo(label)
        for layer in doc.layers:
            if mirror:
                if layer.texts:
                    layer.rasterize_texts()
                if layer.poses:
                    layer.rasterize_poses()
            ox, oy = layer.offset
            lw, lh = layer.image.size
            layer.set_image(layer.image.transpose(transpose))
            if kind == "flip_h":
                layer.offset = (width - ox - lw, oy)
            elif kind == "flip_v":
                layer.offset = (ox, height - oy - lh)
            elif kind == "rotate_cw":
                layer.offset = (height - oy - lh, ox)
            else:
                layer.offset = (oy, width - ox - lw)
            if layer.mask is not None:
                layer.mask = layer.mask.transpose(transpose)
            for spec in layer.strokes:
                for key in ("points", "handles_in", "handles_out"):
                    for point in spec.get(key) or []:
                        point[0], point[1] = map_point(point[0], point[1])
            if layer.strokes:
                layer.invalidate_vectors()
            for spec in layer.paths:
                for key in ("anchors", "handles_in", "handles_out"):
                    for point in spec.get(key) or []:
                        point[0], point[1] = map_point(point[0], point[1])
            if layer.paths:
                layer.invalidate_paths()
            if rotate and layer.texts:
                for spec in layer.texts:
                    spec["x"], spec["y"] = map_point(
                        spec.get("x", 0.0), spec.get("y", 0.0)
                    )
                    spec["rotation"] = float(spec.get("rotation", 0.0)) + theta
                layer.invalidate_texts()
            if rotate and layer.poses:
                scene = layer.pose_scene
                scene["x"], scene["y"] = map_point(scene["x"], scene["y"])
                scene["roll"] = float(scene.get("roll", 0.0)) + theta
                layer.invalidate_poses()

        if doc.inpaint_mask is not None:
            doc.inpaint_mask = doc.inpaint_mask.transpose(transpose)
            doc._inpaint_mask_shared = False

        # Guides annotate the document, so they move with the new origin unless
        # they are shared across canvases (see ``crop_canvas``).
        if not self.guides_shared:
            moved = []
            for orient, pos in doc.guides:
                if kind == "flip_h":
                    # A left-right flip mirrors vertical guides and leaves
                    # horizontal ones where they are.
                    if orient == "v":
                        pos = width - pos
                elif kind == "flip_v":
                    if orient == "h":
                        pos = height - pos
                elif kind == "rotate_cw":
                    orient, pos = (
                        ("h", pos) if orient == "v" else ("v", height - pos)
                    )
                else:
                    orient, pos = (
                        ("h", width - pos) if orient == "v" else ("v", pos)
                    )
                moved.append((orient, int(round(pos))))
            doc.guides = moved

        if rotate:
            doc.width, doc.height = height, width
        view = self.area.current_view()
        if view is not None:
            view.set_selection(None)
        # Live overlays (a pending crop, warp or puppet mesh) address the old
        # document, so drop them rather than leave them misaligned.
        self.drawing.clear_crop()
        self.drawing.discard_warp()
        self.drawing.discard_puppet()
        self._sync_gen_size()
        self._update_title()
        self.render()
        self.set_result(f"{label} canvas.")

    def flip_canvas_horizontal(self):
        self._transform_canvas("flip_h", "Flipped")

    def flip_canvas_vertical(self):
        self._transform_canvas("flip_v", "Flipped")

    def rotate_canvas_cw(self):
        self._transform_canvas("rotate_cw", "Rotated")

    def rotate_canvas_ccw(self):
        self._transform_canvas("rotate_ccw", "Rotated")

    def _apply_transform(self, image, values):
        """Scale, rotate, and flip ``image`` per an ``ask_transform`` result."""
        img = image
        if values["flip_h"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if values["flip_v"]:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        scale_x = values["scale_x"]
        scale_y = values["scale_y"]
        if abs(scale_x - 1.0) > 1e-9 or abs(scale_y - 1.0) > 1e-9:
            width = max(1, round(img.width * scale_x))
            height = max(1, round(img.height * scale_y))
            img = img.resize((width, height), Image.LANCZOS)
        angle = values["angle"]
        if abs(angle) > 1e-9:
            # PIL rotates counter-clockwise; negate so positive turns clockwise.
            img = img.rotate(-angle, resample=Image.BICUBIC, expand=True)
        return img

    def transform_layer(self):
        """Open the transform dialog and apply the result to the active layer."""
        if self._active_layer_locked():
            return
        values = dialogs.ask_transform(self.root)
        if values is None:
            return
        self._transform_layer(
            lambda img: self._apply_transform(img, values), "Transformed"
        )

    # ------------------------------------------------------------------ select
    @property
    def selection(self):
        """The active view's :class:`selection.Selection`, or ``None``."""
        view = self.area.current_view()
        return getattr(view, "selection", None)

    def alpha_to_selection(self):
        """Select the active layer's opaque pixels (its alpha channel)."""
        layer = self.active_layer()
        if layer is None:
            self.set_result("There is no layer to select from.")
            return
        view = self.area.current_view()
        if view is None:
            return
        alpha = layer.composited_image().getchannel("A")
        mask = Image.new("L", self.current.size, 0)
        mask.paste(alpha, layer.offset)
        selection = Selection.from_mask(self.current.size, mask)
        if selection.is_empty():
            self.set_result(f"Layer '{layer.name}' has no opaque pixels.")
            return
        view.set_selection(selection)
        self.tools.select("select")
        box = selection.bbox
        self.set_result(
            f"Selected {box[2] - box[0]} x {box[3] - box[1]} "
            f"from layer '{layer.name}'."
        )

    def select_all(self):
        view = self.area.current_view()
        if view is None:
            return
        # In exploded view the editable frame is the active layer's cell, not the
        # whole canvas, so select that rectangle instead of the document.
        left, top, right, bottom = view.doc_frame_rect()
        box = (left, top, right, bottom)
        view.set_selection(
            Selection.rectangle((self.current.width, self.current.height), box)
        )
        self.tools.select("select")
        self.set_result(f"Selected {right - left} x {bottom - top}.")

    def deselect(self):
        """Drop the selection. Returns whether there was one to drop."""
        view = self.area.current_view()
        cleared = view is not None and view.selection is not None
        if cleared:
            view.set_selection(None)
            self.drawing.sync_transform()
        self.set_result("Deselected.")
        return cleared

    def invert_selection(self):
        self._apply_selection_op(lambda sel: sel.inverted(), "Inverted selection.")

    def grow_selection(self, radius=4):
        self._apply_selection_op(
            lambda sel: sel.grown(radius), f"Grew selection by {radius}px."
        )

    def shrink_selection(self, radius=4):
        self._apply_selection_op(
            lambda sel: sel.shrunk(radius), f"Shrank selection by {radius}px."
        )

    def feather_selection(self, radius=8):
        self._apply_selection_op(
            lambda sel: sel.feathered(radius),
            f"Feathered selection by {radius}px.",
        )

    def _apply_selection_op(self, op, message):
        """Replace the active selection with ``op(selection)`` when non-empty."""
        selection = self.selection
        if selection is None or selection.is_empty():
            self.set_result("No selection.")
            return
        view = self.area.current_view()
        if view is None:
            return
        view.set_selection(op(selection))
        self.set_result(message)

    def delete_selection(self):
        selection = self.selection
        if selection is None or selection.is_empty():
            self.set_result("No selection.")
            return
        if self._active_layer_locked():
            return
        self._push_undo("Delete selection")
        self.current.bake_offsets()
        self._clear_masked(self.active_layer().image, selection.mask)
        self.active_layer().invalidate_filters()
        self.render()
        self.set_result("Cleared selection.")

    def delete_forward(self):
        """The Delete key: act on the most specific current selection.

        An active pixel selection is cleared first, matching the Select menu's
        "Delete Selection". Otherwise whatever row is selected in the Layers
        panel is removed - a layer, one of its filters or text objects, or a
        group - so Delete mirrors that panel's Delete button.
        """
        selection = self.selection
        if selection is not None and not selection.is_empty():
            self.delete_selection()
            return
        if self.current is not None and self.layers:
            self.layer_panel.delete_current()
            return
        self.set_result("Nothing to delete.")

    def _escape(self):
        if self._has_running_job():
            self._cancel_current()
            return
        if self.area.gallery_visible():
            self.hide_gallery()
            return
        if self.area.gallery_side_visible():
            self.hide_gallery_side()
            return
        # Cascade: each Escape peels one layer of state. The active tool's
        # in-progress work is cancelled first and the tool is kept; only once
        # there is nothing left to cancel does Escape reset to the Arrow tool.
        if self.drawing.cancel_path():
            return
        if self.drawing.cancel_text():
            return
        if self.drawing.cancel_warp():
            return
        if self.drawing.cancel_puppet():
            return
        if self.drawing.cancel_transform():
            return
        if self.drawing.clear_crop():
            return
        if self.deselect():
            return
        self.tools.cancel_modes()

    # ------------------------------------------------------------------ color
    @property
    def color(self):
        return self.colorpicker.color

    @color.setter
    def color(self, value):
        self.colorpicker.color = value

    @property
    def palette(self):
        return self.colorpicker.palette

    @palette.setter
    def palette(self, value):
        self.colorpicker.set_palette(value)

    @property
    def selected_swatch(self):
        return self.colorpicker.selected_swatch

    @selected_swatch.setter
    def selected_swatch(self, value):
        self.colorpicker.set_selected(value)

    @property
    def swatches(self):
        return self.colorpicker.swatches

    def set_color(self, color):
        self.colorpicker.set_color(color)

    def choose_color(self):
        self.colorpicker.choose_color()

    # ------------------------------------------------------------------ tools
    @property
    def brush_size(self):
        return self.tools.brush_size

    @brush_size.setter
    def brush_size(self, value):
        self.tools.brush_size = value

    def toggle_brush(self):
        self.tools.toggle_brush()

    def toggle_eyedropper(self):
        self.tools.toggle_eyedropper()

    def toggle_eraser(self):
        self.tools.toggle_eraser()

    def toggle_fill(self):
        self.tools.toggle_fill()

    def toggle_select(self):
        self.tools.toggle_select()

    def select_rect(self):
        self.tools.set_select_mode("rect")

    def toggle_lasso(self):
        self.tools.toggle_lasso()

    def toggle_path(self):
        self.tools.toggle_path()

    def toggle_vector(self):
        self.tools.toggle_vector()

    def toggle_poses(self):
        self.tools.toggle_poses()

    def toggle_liquify(self):
        self.tools.toggle_liquify()

    def toggle_warp(self):
        self.tools.toggle_warp()

    def toggle_puppet(self):
        self.tools.toggle_puppet()

    def toggle_move(self):
        self.tools.toggle_move()

    def toggle_drag(self):
        self.tools.toggle_drag()

    def toggle_transform(self):
        self.tools.toggle_transform()

    def toggle_text(self):
        self.tools.toggle_text()

    def toggle_gradient(self):
        self.tools.toggle_gradient()

    def toggle_shape(self):
        self.tools.toggle_shape()

    def toggle_clone(self):
        self.tools.toggle_clone()

    def toggle_smudge(self):
        self.tools.toggle_smudge()

    def toggle_dodge(self):
        self.tools.toggle_dodge()

    def toggle_burn(self):
        self.tools.toggle_burn()

    def toggle_wand(self):
        self.tools.toggle_wand()

    def toggle_paste(self):
        self.tools.toggle_paste()

    def toggle_arrow(self):
        self.tools.toggle_arrow()

    def toggle_crop(self):
        self.tools.toggle_crop()

    def _apply_cursor_state(self):
        tools = self.tools
        if tools.drag_tool:
            cursor = "fleur"
        elif tools.move_tool:
            cursor = "fleur"
        elif tools.transform_tool:
            cursor = "crosshair"
        elif tools.arrow_tool:
            cursor = "arrow"
        elif tools.paste_tool:
            cursor = "arrow"
        elif tools.select_tool:
            cursor = "crosshair"
        elif tools.path_tool:
            cursor = "crosshair"
        elif tools.poses_tool:
            cursor = "arrow"
        elif tools.mask_edit:
            cursor = "crosshair"
        elif tools.text_tool:
            cursor = "xterm"
        elif tools.eyedropper:
            cursor = "dotbox"
        else:
            cursor = "crosshair"
        view = self.area.current_view()
        if view is not None:
            view.config(cursor=cursor)
        self.drawing.refresh_cursor()

    def _transform_mode_active(self):
        """True when the Transform tool or the Select tool's toggle runs."""
        tools = self.tools
        return bool(tools.transform_tool or (
            tools.select_tool
            and getattr(tools, "select_transform", False)))

    def symmetry_guide(self):
        """The active symmetry guide (see ``symmetry``), or ``None`` when off.

        The guide itself is a tool setting on :class:`ToolsPanel`; this exposes
        it to the view (which draws the axes) and the controller (which mirrors
        the dabs), in the plain dict ``symmetry.transforms`` expects. The
        exploded view mirrors in document space, which its cells do not share, so
        symmetry is off there.
        """
        if self.show_exploded:
            return None
        tools = self.tools
        mode = getattr(tools, "symmetry_mode", "off")
        if mode == "off":
            return None
        return {
            "mode": mode,
            "count": int(getattr(tools, "symmetry_count", 6)),
            "mirror": bool(getattr(tools, "symmetry_mirror", True)),
        }

    def _on_mode_change(self):
        tools = self.tools
        if tools.drag_tool:
            status = "drag: drag to pan the view"
        elif tools.move_tool:
            status = "move: drag to move the selected layers or a guide"
        elif tools.transform_tool:
            status = ("transform: drag to move; corners scale (Shift inverts "
                      "uniform); edges skew; the top handle rotates; drag the "
                      "orange origin to change the pivot. Exact values, pivot "
                      "presets, snapping and smoothing are in Tool Settings")
        elif tools.paste_tool:
            status = "paste: paste the clipboard, or set the paste options"
        elif tools.arrow_tool:
            status = "no tool: pick a tool from the toolbar"
        elif tools.select_tool:
            if getattr(tools, "select_transform", False):
                status = ("select (transform): drag the handles to scale, "
                          "rotate or skew the selection; drag inside to move")
            elif getattr(tools, "select_mode", "rect") == "wand":
                status = "select (wand): click to select a region by color"
            elif getattr(tools, "select_mode", "rect") == "segment":
                status = ("select (segment): click an object to select it with "
                          "AI segmentation")
            elif getattr(tools, "select_mode", "rect") == "lasso":
                status = ("select (lasso): drag a freeform outline; "
                          "drag inside it to move")
            else:
                status = "select: drag a rectangle; drag inside it to move"
        elif tools.path_tool:
            what = ("stroke a line (brush size)" if tools.path_mode == "stroke"
                    else "fill and stroke the shape"
                    if tools.path_mode == "both" else "fill the shape")
            status = ("path: click to add points, drag for curves; Alt-drag a "
                      "round handle to reshape, drag a point to move, "
                      "right-click to delete; click the first point, "
                      f"double-click, or Enter to {what}")
        elif tools.vector_tool:
            mode = getattr(tools, "vector_mode", "draw")
            status = {
                "draw": ("vector (draw): drag to draw an editable line on a "
                         "vector layer; a new vector layer is added when needed"),
                "edit": ("vector (edit): click a line, Shift/Ctrl-click or "
                         "marquee to select several, drag to move, corner "
                         "handles scale and the top handle rotates; arrows "
                         "nudge the active anchor; double-click adds a point, "
                         "right-click deletes one"),
                "width": ("vector (width): drag across lines to set their "
                          "width to the current Size"),
                "erase": ("vector (erase): drag across lines to cut them; a "
                          "cut through the middle splits the line in two"),
            }.get(mode, "vector")
        elif tools.poses_tool:
            status = ("poses: click a joint to select it, then drag to pose that "
                      "limb (a square handle solves the whole limb); click the "
                      "body to cycle Move/Scale/Rotate, then drag the body to "
                      "apply it; drag off the figure to turn it in 3D (Shift to "
                      "tilt); click empty space to deselect")
        elif tools.liquify_tool:
            status = {
                "push": "liquify (push): drag to push pixels along the stroke",
                "expand": "liquify (expand): drag to bloat pixels away from the brush",
                "pinch": "liquify (pinch): drag to pull pixels toward the brush",
                "twirl_cw": "liquify (twirl cw): drag to spin pixels clockwise",
                "twirl_ccw": "liquify (twirl ccw): drag to spin pixels counter-clockwise",
            }.get(getattr(tools, "liquify_mode", "push"), "liquify")
        elif tools.warp_tool:
            status = ("warp: drag a grid point to deform the layer; Apply keeps "
                      "it, Cancel restores")
        elif tools.puppet_tool:
            status = ("puppet warp: tap to add a pin, drag to move, drag its "
                      "ring to rotate, Shift/Ctrl-click to multi-select, "
                      "right-click to delete; Apply keeps it, Cancel restores")
        elif tools.layout_tool:
            status = ("layout control: drag empty canvas to add a box, drag a box "
                      "to move it, or its handles to resize; name and label items "
                      "in Tool Settings")
        elif tools.inpaint_mask_edit:
            status = ("inpaint mask: paint where generation may change pixels; "
                      "Generate with Region set to 'Painted mask' to use it")
        elif tools.mask_edit:
            status = "mask: left-click paints the value, right-click inverts it"
        elif tools.text_tool:
            status = ("text: click to add; type to edit; Enter commits, "
                      "Shift+Enter newline, Esc cancels; Alt-drag the orange "
                      "origin to change the rotation pivot (snaps to "
                      "corners/centre, Alt-double-click to reset)")
        elif tools.eyedropper:
            status = "eyedropper: click canvas to pick"
        elif tools.fill_tool:
            status = "fill: click to flood-fill the active layer"
        elif tools.gradient_tool:
            status = "gradient: drag to set the axis, release to fill"
        elif tools.shape_tool:
            status = ("shape: drag to draw (Shift constrains); fill, stroke or "
                      "both with the tool's colors")
        elif tools.clone_tool:
            status = "clone: Alt-click to set the source, then paint"
        elif tools.smudge_tool:
            status = "smudge: drag to smear colors along the stroke"
        elif tools.dodge_tool:
            status = "dodge: drag to lighten the pixels under the brush"
        elif tools.burn_tool:
            status = "burn: drag to darken the pixels under the brush"
        elif tools.crop_tool:
            status = "crop: drag a rectangle, then Apply Crop or Enter"
        elif tools.eraser:
            status = "eraser: drag to erase"
        else:
            status = "ready"
        self._apply_cursor_state()
        self.drawing.sync_text()
        self.drawing.sync_transform()
        self.drawing.sync_path()
        self.drawing.sync_vector()
        self.drawing.sync_poses()
        self.drawing.sync_symmetry()
        self.drawing.sync_warp()
        self.drawing.sync_puppet()
        self.drawing.sync_crop()
        self.drawing.sync_clone_source()
        self._sync_layout_overlay()
        if hasattr(self, "layer_panel"):
            self.layer_panel.set_mask_edit(tools.mask_edit)
        if hasattr(self, "gen"):
            self.gen.set_inpaint_mask_edit(tools.inpaint_mask_edit)
        if (not tools.inpaint_mask_edit
                and getattr(self, "_mask_value_before_inpaint", None) is not None):
            # Left inpaint-mask painting (possibly by switching tools): restore
            # the layer mask's paint value.
            tools.mask_value = self._mask_value_before_inpaint
            self._mask_value_before_inpaint = None
        # Switching tools clears the mask-edit flags directly, so repaint when
        # that changes which veil the canvas should show; otherwise the old
        # overlay lingers until the next click triggers a render.
        mask_state = (bool(tools.mask_edit), bool(tools.inpaint_mask_edit))
        if mask_state != self._mask_preview_state:
            self._mask_preview_state = mask_state
            self.render()
        self._report_layer()
        self._report_tool()
        if hasattr(self, "brush_library"):
            self.brush_library.set_tool(self.tools._active_tool())
        guide = self.symmetry_guide()
        if guide is not None:
            status = f"{status}  |  symmetry: {guide['mode']}"
            if guide["mode"] == "radial":
                status += f" x{guide['count']}" + (" (mirror)" if guide["mirror"] else "")
        self.statusbar.set_status(status)

    def _on_brush_preset(self):
        """A brush preset was applied: refresh the cursor and tool report."""
        self.drawing.refresh_cursor()
        self._report_tool()

    def prev_brush_preset(self):
        """Select the previous brush preset for the active tool."""
        if hasattr(self, "brush_library"):
            self.brush_library.cycle(-1)

    def next_brush_preset(self):
        """Select the next brush preset for the active tool."""
        if hasattr(self, "brush_library"):
            self.brush_library.cycle(1)

    def _on_color_changed(self, _color=None):
        """The main colour changed: repaint the tool previews drawn with it."""
        if hasattr(self, "tools"):
            self.tools.refresh_color_previews()

    def _on_brush_size(self):
        self.drawing.refresh_cursor()
        if hasattr(self, "brush_library"):
            self.brush_library.refresh_modified()
        if self.tools.path_tool:
            # The stroke preview draws at the brush size, so repaint it live.
            self.drawing.refresh_path()
        self._report_tool()

    def begin_canvas_pick(self, callback):
        """Arm a one-shot canvas sample for a modal color dialog.

        ``callback(color)`` receives the picked ``#rrggbb`` string after the
        next canvas click. Used by the "Pick on Canvas" button so a color can be
        sampled without leaving the dialog.
        """
        self._canvas_pick = callback
        self.statusbar.set_status("Click the canvas to pick a color.")

    def canvas_pick_pending(self):
        """Whether a dialog is waiting for a canvas color sample."""
        return self._canvas_pick is not None

    def pick_color_at(self, x, y):
        if self.current is None:
            return
        if self.show_exploded:
            # The cell under the pointer shows a single layer; sample that layer
            # at its own pixel rather than the stacked composite, which the grid
            # does not show.
            layer = self.active_layer()
            if layer is None:
                return
            ox, oy = layer.offset
            image = layer.composited_image()
            lx, ly = x - ox, y - oy
            if not (0 <= lx < image.width and 0 <= ly < image.height):
                return
            r, g, b = image.getpixel((lx, ly))[:3]
        else:
            width, height = self.current.size
            if not (0 <= x < width and 0 <= y < height):
                return
            r, g, b = self.composite().getpixel((x, y))
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.statusbar.set_status(f"picked {hex_color}")
        callback = self._canvas_pick
        if callback is not None:
            self._canvas_pick = None
            callback(hex_color)
            return
        self.colorpicker.set_rgb(r, g, b)

    # ------------------------------------------------------------------ files
    def new_document(self):
        """Start a new project: discard the current one, then prompt for a canvas."""
        self._confirm_discard_changes(self._new_document_now)

    def _new_document_now(self):
        self._reset_document()
        # A new project starts by asking for its first canvas; cancelling the
        # dialog leaves the empty workspace new_document already produced.
        self.add_canvas()

    def _reset_document(self):
        """Clear the workspace to an empty, unsaved project (no canvas)."""
        self._clear_generation_preview()
        self.canvases = []
        self.active_canvas = 0
        self.canvas_counter = 0
        self.project_path = None
        self.shared_guides = []
        self._refresh_history()
        self.area.rebuild(self.canvases)
        self._sync_shared_guides()
        self._activate_canvas(0)
        self._set_modified(False)
        self.set_result("")

    def _import_svg_path(self, path):
        """Import an SVG file as editable vector lines on a new vector layer."""
        try:
            doc = svgimport.load(path)
        except (OSError, ValueError) as exc:
            dialogs.show_error(
                self.root, "Open Image", f"Could not import SVG:\n{exc}"
            )
            return
        specs = doc.get("strokes") or []
        fill_specs = doc.get("fills") or []
        if not specs and not fill_specs:
            self.set_result("That SVG has no vector geometry to import.")
            return
        self._push_undo("Import vector")
        if self.current is None:
            self._new_canvas()
            self.area.add(self.canvases[-1])
            self.active_canvas = len(self.canvases) - 1
        placed, placed_fills = svgimport.fit_to(
            specs, self.current.size, fills=fill_specs)
        layer = self._add_layer(name=os.path.basename(path), index=self.active + 1)
        layer.vector = True
        layer.strokes = placed
        layer.fills = placed_fills
        layer.invalidate_vectors()
        self.active = self.layers.index(layer)
        self._refresh_layer_list()
        self.render()
        total = len(placed) + len(placed_fills)
        self.set_result(
            f"Imported {total} vector shape"
            f"{'s' if total != 1 else ''} from {os.path.basename(path)}."
        )

    def copy_vector_svg(self):
        """Copy the active layer's vector lines to the clipboard as SVG text."""
        layer = self.active_layer()
        if layer is None or not (
                getattr(layer, "strokes", None) or getattr(layer, "fills", None)):
            self.set_result("The active layer has no vector lines to copy.")
            return
        text = svgimport.export(
            layer.strokes, fills=getattr(layer, "fills", None),
            size=self.current.size)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        # Drop any in-app pixel copy so a following Paste uses this text.
        self._clipboard = None
        count = len(layer.strokes) + len(getattr(layer, "fills", []))
        self.set_result(
            f"Copied {count} vector shape"
            f"{'s' if count != 1 else ''} as SVG.")

    def _paste_clipboard_svg(self, dest="selected_layer", data=None):
        """Paste SVG on the clipboard as editable vector lines.

        Returns ``True`` when something was pasted. Understands both SVG *text*
        (copied from an editor or another app) and one or more ``.svg`` *files*
        copied in the file manager, which arrive as a list of paths in ``data``.
        :meth:`paste` calls this once the image paths come up empty, so the plain
        Paste command handles SVG without a separate menu entry.
        """
        text = None
        name = "SVG"
        if isinstance(data, list):
            for path in data:
                if not str(path).lower().endswith(".svg"):
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as handle:
                        text = handle.read()
                except OSError:
                    continue
                name = os.path.basename(path)
                break
        if text is None:
            try:
                clipboard_text = self.root.clipboard_get()
            except tk.TclError:
                return False
            stripped = clipboard_text.strip().strip('"')
            if "<svg" in stripped.lower():
                text = clipboard_text
            elif stripped.lower().endswith(".svg"):
                try:
                    with open(stripped, "r", encoding="utf-8", errors="replace") as handle:
                        text = handle.read()
                except OSError:
                    return False
                name = os.path.basename(stripped)
            else:
                return False
        try:
            doc = svgimport.parse(text)
        except ValueError:
            return False
        specs = doc.get("strokes") or []
        fill_specs = doc.get("fills") or []
        if not specs and not fill_specs:
            return False
        layer = self.active_layer() if dest == "selected_layer" else None
        self._push_undo("Paste SVG")
        if layer is None or not getattr(layer, "vector", False):
            layer = self._add_layer(name="Vector Layer", index=self.active + 1)
            layer.vector = True
            self.active = self.layers.index(layer)
        specs, fill_specs = svgimport.fit_within(
            specs, fill_specs, self.current.size)
        layer.strokes.extend(specs)
        layer.fills.extend(fill_specs)
        layer.vector = True
        layer.invalidate_vectors()
        self._refresh_layer_list()
        self.render()
        total = len(specs) + len(fill_specs)
        self.set_result(
            f"Pasted {total} vector shape"
            f"{'s' if total != 1 else ''} from {name}.")
        return True

    def export_vector(self):
        """Export the active layer's vector lines as an SVG file."""
        layer = self.active_layer()
        if layer is None or not getattr(layer, "strokes", None):
            self.set_result("The active layer has no vector lines to export.")
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Export Vector (SVG)",
            defaultextension=".svg",
            filetypes=[("SVG", "*.svg"), ("All files", "*.*")],
        )
        if not path:
            return
        text = svgimport.export(
            layer.strokes, fills=getattr(layer, "fills", None),
            size=self.current.size)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            dialogs.show_error(
                self.root, "Export Vector", f"Could not write SVG:\n{exc}"
            )
            return
        count = len(layer.strokes) + len(getattr(layer, "fills", []))
        self.set_result(
            f"Exported {count} vector shape"
            f"{'s' if count != 1 else ''} to {os.path.basename(path)}."
        )

    def convert_paths_to_vector(self):
        """Convert the active layer's editable paths into editable vector lines."""
        layer = self.active_layer()
        if layer is None or not getattr(layer, "paths", None):
            self.set_result("The active layer has no paths to convert.")
            return
        self._push_undo("Convert paths to vector")
        converted = []
        for spec in layer.paths:
            mode = spec.get("mode", "fill")
            closed = bool(spec.get("closed", False))
            if mode == "stroke":
                color = spec.get("stroke_color", "#000000")
                width = spec.get("width", vector.DEFAULT_WIDTH)
                fill = None
            else:
                color = spec.get("fill_color", "#000000")
                width = spec.get("outline") or spec.get("width", vector.DEFAULT_WIDTH)
                fill = (spec.get("fill_color")
                        if closed and mode in ("fill", "both") else None)
            stroke = vector.new_stroke(
                [list(point) for point in spec.get("anchors", [])],
                width, color,
                opacity=float(spec.get("opacity", 100)) / 100.0,
                closed=closed, cap=spec.get("cap", "round"),
                join=spec.get("join", "round"),
                handles_in=[list(point) for point in spec.get("handles_in", [])],
                handles_out=[list(point) for point in spec.get("handles_out", [])],
                fill=fill)
            if stroke is not None:
                converted.append(stroke)
        layer.paths = []
        layer.strokes.extend(converted)
        layer.vector = True
        layer.invalidate_paths()
        layer.invalidate_vectors()
        self._refresh_layer_list()
        self.render()
        self.set_result(
            f"Converted {len(converted)} path"
            f"{'s' if len(converted) != 1 else ''} into vector lines."
        )

    def open_image(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("SVG vector", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        if path.lower().endswith(".svg"):
            self._import_svg_path(path)
            return
        try:
            image = Image.open(path)
        except Exception as exc:
            dialogs.show_error(
                self.root, "Open Image", f"Could not open image:\n{exc}"
            )
            return
        self._push_undo("Open image")
        if self.current is None:
            self._new_canvas()
            self.area.add(self.canvases[-1])
            self.active_canvas = len(self.canvases) - 1
        image = self._import_to_working(image)
        layer = self._add_layer(name=os.path.basename(path), index=self.active + 1)
        layer.set_image(self.current.fit_image(image))
        self.active = self.layers.index(layer)
        self._refresh_layer_list()
        self.render()
        self.set_result(f"Opened {os.path.basename(path)} as a new layer.")

    def _import_to_working(self, image):
        """Convert an opened image into the active canvas's working space.

        Uses the image's embedded ICC profile when it has one, and is a no-op
        otherwise (or when the working space is sRGB and the image is untagged
        or already sRGB).
        """
        if self.current is None:
            return image
        working = (
            getattr(self.current, "working_profile", None) or colormanagement.SRGB
        )
        return colormanagement.convert_from_embedded(image, working)

    def save_image(self):
        """Save the current canvas as an image, with export options.

        The canvas is flattened here, then the (potentially slow) resize/encode
        and file write run on the project worker so the status bar shows
        progress and the window stays responsive.
        """
        if self.current is None:
            self.set_result("There is no canvas to save.")
            return
        options = dialogs.ask_export(
            self.root, title="Save Image As", action="Save",
            base_size=self.current.size, estimate=self._estimate_export,
            initial=self._export_defaults, canvas_name=self.current.name,
        )
        if options is None:
            return
        self._remember_export(options)
        fmt = options["format"]
        ext = self._EXPORT_EXT[fmt]
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Image As",
            defaultextension=ext,
            filetypes=[(fmt, f"*{ext}"), ("All files", "*.*")],
        )
        if not path:
            return
        # Flatten on the Tk thread (it reads the document), then hand the
        # standalone image to the worker to encode and write.
        try:
            rgba = self._flatten_canvas(self.current, options)
        except Exception as exc:
            self._report_export_failure("Save Image", str(exc))
            return
        name = os.path.basename(path)
        working = getattr(self.current, "working_profile", colormanagement.SRGB)

        def runner(progress, cancel):
            cancel.throw_if_cancelled()
            progress(0.1, "preparing", None)
            image = self._encode_export_image(rgba, options, working)
            cancel.throw_if_cancelled()
            progress(0.7, "writing", None)
            self._save_export(image, path, fmt, options, working)
            progress(1.0, "done", None)
            return name

        def finisher(_result, error, cancelled):
            if cancelled:
                self.statusbar.set_status("cancelled")
                self.set_result(f"Save {name} cancelled.", "warn")
            elif error is not None:
                self._report_export_failure("Save Image", error)
            else:
                self.statusbar.set_status(f"saved {name}")
                self.set_result(f"Saved {name}.")

        self.submit_project_job("export", f"Save {name}", runner, finisher)
        self.set_result(f"Saving {name}\u2026")

    def _report_export_failure(self, title, message):
        """Report a failed export/save in the status bar, log, and a dialog."""
        # The queue wraps a runner's exception as "Error: ..."; show the
        # explanation itself rather than doubling the prefix.
        if message.startswith("Error: "):
            message = message[len("Error: "):]
        self.statusbar.set_status("failed")
        self.set_result(f"Could not {title.lower()}: {message}", "error")
        dialogs.show_error(self.root, title, f"Could not {title.lower()}:\n{message}")

    _EXPORT_EXT = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp",
                   "GIF": ".gif", "TIFF": ".tif"}

    def export_images(self):
        """Export the current canvas, or every canvas, with format options."""
        if not self.canvases:
            self.set_result("There is no canvas to export.")
            return
        options = dialogs.ask_export(
            self.root, multiple_canvases=len(self.canvases) > 1,
            base_size=self.current.size, estimate=self._estimate_export,
            initial=self._export_defaults, canvas_name=self.current.name,
            canvas_count=len(self.canvases),
        )
        if options is None:
            return
        self._remember_export(options)
        fmt = options["format"]
        ext = self._EXPORT_EXT[fmt]
        if options["scope"] == "all":
            if options.get("animate"):
                self._export_animation(options, ext)
            else:
                self._export_all_canvases(options, ext)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Image",
            defaultextension=ext,
            filetypes=[(fmt, f"*{ext}"), ("All files", "*.*")],
        )
        if not path:
            return
        # Flatten on the Tk thread, then encode/write on the worker.
        try:
            rgba = self._flatten_canvas(self.current, options)
        except Exception as exc:
            self._report_export_failure("Export Image", str(exc))
            return
        name = os.path.basename(path)
        working = getattr(self.current, "working_profile", colormanagement.SRGB)

        def runner(progress, cancel):
            cancel.throw_if_cancelled()
            progress(0.1, "preparing", None)
            image = self._encode_export_image(rgba, options, working)
            cancel.throw_if_cancelled()
            progress(0.7, "writing", None)
            self._save_export(image, path, fmt, options, working)
            progress(1.0, "done", None)
            return name

        def finisher(_result, error, cancelled):
            if cancelled:
                self.statusbar.set_status("cancelled")
                self.set_result(f"Export {name} cancelled.", "warn")
            elif error is not None:
                self._report_export_failure("Export Image", error)
            else:
                self.statusbar.set_status(f"exported {name}")
                self.set_result(f"Exported {name}.")

        self.submit_project_job("export", f"Export {name}", runner, finisher)
        self.set_result(f"Exporting {name}\u2026")

    def _load_export_defaults(self):
        """The last-used export options persisted in settings, or ``{}``."""
        stored = theme.read_settings().get("export")
        return stored if isinstance(stored, dict) else {}

    def _remember_export(self, options):
        """Persist the export options for the next dialog (minus scope/size)."""
        stored = {
            key: value for key, value in options.items()
            if key not in ("scope", "size")
        }
        settings = theme.read_settings()
        settings["export"] = stored
        theme.write_settings(settings)
        self._export_defaults = stored

    def _export_all_canvases(self, options, ext):
        directory = filedialog.askdirectory(
            parent=self.root, title="Export All Canvases"
        )
        if not directory:
            return
        # Flatten every canvas on the Tk thread, then encode and write them on
        # the worker so the status bar reports per-canvas progress.
        fmt = options["format"]
        try:
            targets = [
                (self._safe_name(canvas.name, index + 1),
                 self._flatten_canvas(canvas, options),
                 getattr(canvas, "working_profile", colormanagement.SRGB))
                for index, canvas in enumerate(self.canvases)
            ]
        except Exception as exc:
            self._report_export_failure("Export Images", str(exc))
            return

        def runner(progress, cancel):
            used = set()
            count = 0
            total = len(targets)
            for index, (base, rgba, working) in enumerate(targets):
                cancel.throw_if_cancelled()
                name = base
                suffix = 2
                while name.lower() in used:
                    name = f"{base}_{suffix}"
                    suffix += 1
                used.add(name.lower())
                image = self._encode_export_image(rgba, options, working)
                self._save_export(image, os.path.join(directory, name + ext),
                                  fmt, options, working)
                count += 1
                progress(count / total if total else 1.0,
                         f"exporting {count}/{total}", count / total if total else 1.0)
            return count

        def finisher(result, error, cancelled):
            if cancelled:
                self.statusbar.set_status("cancelled")
                self.set_result("Export cancelled.", "warn")
            elif error is not None:
                self._report_export_failure("Export Images", error)
            else:
                self.statusbar.set_status(f"exported {result} image(s)")
                self.set_result(f"Exported {result} image(s) to {directory}.")

        self.submit_project_job("export", "Export images", runner, finisher)
        self.set_result(f"Exporting {len(targets)} image(s)\u2026")

    def _export_animation(self, options, ext):
        """Export every canvas as frames of one animated PNG/WEBP/GIF."""
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Animation",
            defaultextension=ext,
            filetypes=[(options["format"], f"*{ext}"), ("All files", "*.*")],
        )
        if not path:
            return
        # Flatten every frame on the Tk thread, then build and encode the
        # animation on the worker with per-frame progress.
        try:
            snapshots = [
                (self._flatten_canvas(canvas, options),
                 getattr(canvas, "working_profile", colormanagement.SRGB))
                for canvas in self.canvases
            ]
        except Exception as exc:
            self._report_export_failure("Export Animation", str(exc))
            return
        name = os.path.basename(path)

        def runner(progress, cancel):
            total = len(snapshots)
            frames = []
            for index, (rgba, working) in enumerate(snapshots):
                cancel.throw_if_cancelled()
                frames.append(self._encode_export_image(rgba, options, working))
                progress(0.7 * (index + 1) / total if total else 0.7,
                         f"frame {index + 1}/{total}",
                         (index + 1) / total if total else 1.0)
            frames = self._pad_frames(frames, options)
            cancel.throw_if_cancelled()
            progress(0.9, "encoding", 0.9)
            self._save_animation(frames, path, options)
            progress(1.0, "done", 1.0)
            return len(frames)

        def finisher(result, error, cancelled):
            if cancelled:
                self.statusbar.set_status("cancelled")
                self.set_result("Export cancelled.", "warn")
            elif error is not None:
                self._report_export_failure("Export Animation", error)
            else:
                self.statusbar.set_status(f"exported {name}")
                self.set_result(
                    f"Exported a {result}-frame animation to {name}."
                )

        self.submit_project_job("export", "Export animation", runner, finisher)
        self.set_result("Exporting animation\u2026")

    def _pad_frames(self, frames, options):
        """Center frames of differing sizes on a common background.

        Animation frames must share one size, so each is composited over the
        export background (or kept transparent) on a buffer as large as the
        biggest frame.
        """
        width = max(frame.width for frame in frames)
        height = max(frame.height for frame in frames)
        if all(frame.size == (width, height) for frame in frames):
            return frames
        transparent = options["transparent"] and options["format"] != "JPEG"
        fill = (0, 0, 0, 0) if transparent else _hex_to_rgba(options["background"])
        padded = []
        for frame in frames:
            base = Image.new("RGBA", (width, height), fill)
            base.alpha_composite(
                frame.convert("RGBA"),
                ((width - frame.width) // 2, (height - frame.height) // 2),
            )
            padded.append(base if transparent else base.convert("RGB"))
        return padded

    def _save_animation(self, frames, path, options):
        """Write ``frames`` as one animated file in the selected format."""
        fmt = options["format"]
        kwargs = {
            "format": fmt,
            "save_all": True,
            "append_images": frames[1:],
            "duration": int(options.get("frame_delay", 100)),
            "loop": 0 if options.get("loop", True) else 1,
        }
        dpi = options.get("dpi")
        if dpi and fmt != "GIF":
            kwargs["dpi"] = (dpi, dpi)
        metadata = _export_metadata(options)
        xmp = _export_xmp(metadata)
        if fmt == "WEBP":
            opts = options.get("webp", {})
            if opts.get("lossless"):
                kwargs["lossless"] = True
            else:
                kwargs["quality"] = options["quality"]
            kwargs["method"] = int(opts.get("method", 4))
            if metadata:
                kwargs["exif"] = _export_exif(metadata)
                if xmp:
                    kwargs["xmp"] = xmp
        elif fmt == "GIF":
            opts = options.get("gif", {})
            kwargs["optimize"] = bool(opts.get("optimize", False))
            kwargs["disposal"] = 2
        else:
            opts = options.get("png", {})
            kwargs["compress_level"] = int(opts.get("compress_level", 6))
            kwargs["optimize"] = bool(opts.get("optimize", False))
            if metadata:
                kwargs["pnginfo"] = _export_pnginfo(metadata)
        frames[0].save(path, **kwargs)

    def _export_image(self, canvas, options):
        """Flatten ``canvas`` for export and size it as the options request.

        Alpha is kept only when ``options["transparent"]`` is set (JPEG has no
        alpha channel); otherwise the layers are flattened over the chosen matte
        color. ``options["trim"]`` crops fully transparent borders first. A print
        ``options["page"]`` composes the artwork onto a page with bleed and crop
        marks (which overrides Scale/Size); otherwise an explicit
        ``options["size"]`` wins over ``options["scale"]``. A CMYK
        ``options["cmyk"]`` converts the result for JPEG/TIFF, and the working
        pixels are converted to ``options["profile_target"]`` when set.
        """
        return self._encode_export_image(
            self._flatten_canvas(canvas, options), options,
            getattr(canvas, "working_profile", colormanagement.SRGB),
        )

    def _flatten_canvas(self, canvas, options):
        """Flatten ``canvas`` to RGBA and apply Trim.

        This reads the document and its layer caches, so it runs on the Tk
        thread; the resulting standalone image is then safe to encode off-thread
        (see :meth:`_encode_export_image`).
        """
        rgba = canvas.composite_transparent()
        if options.get("trim"):
            bounds = rgba.getchannel("A").getbbox()
            if bounds:
                rgba = rgba.crop(bounds)
        return rgba

    def _encode_export_image(self, rgba, options, working_profile=None):
        """Background/page/resize/CMYK a flattened ``rgba`` image for export.

        Operates only on the passed image, so it can run on a worker thread once
        :meth:`_flatten_canvas` has produced the snapshot. The pixels are in
        ``working_profile`` and are converted to ``options["profile_target"]``
        (when set) before sizing; CMYK output uses an ICC transform when
        ``options["cmyk_profile"]`` names a CMYK profile.
        """
        fmt = options["format"]
        working = working_profile or colormanagement.SRGB
        target = options.get("profile_target") or ""
        if options["transparent"] and fmt != "JPEG":
            image = rgba
        else:
            base = Image.new("RGBA", rgba.size, _hex_to_rgba(options["background"]))
            base.alpha_composite(rgba)
            image = base.convert("RGB")
        conv_source = target or working
        if target and target != working:
            image = colormanagement.convert(image, working, target)
        if options.get("page") not in (None, "none"):
            image = self._compose_print_page(image, options)
        else:
            target_size = options.get("size")
            if target_size is None and options["scale"] != 1.0:
                target_size = (
                    max(1, round(rgba.width * options["scale"])),
                    max(1, round(rgba.height * options["scale"])),
                )
            if target_size and tuple(target_size) != image.size:
                resample = _RESAMPLE_FILTERS.get(options.get("resample"),
                                                 Image.LANCZOS)
                image = image.resize(tuple(target_size), resample)
        if options.get("cmyk") and fmt in ("JPEG", "TIFF"):
            cmyk = colormanagement.to_cmyk(
                image, conv_source, options.get("cmyk_profile") or ""
            )
            image = cmyk if cmyk is not None else image.convert("CMYK")
        return image

    def _compose_print_page(self, image, options):
        """Place the flattened artwork on a print page with bleed and crop marks.

        The page is sized from ``options["page_mm"]`` at ``options["dpi"]``
        (300 when blank), the artwork is fitted inside the trim area and centered,
        and ``options["bleed_mm"]`` adds a margin around the trim. Crop marks are
        drawn in that margin when requested.
        """
        page_mm = options.get("page_mm")
        if not page_mm:
            return image
        width_mm, height_mm = float(page_mm[0]), float(page_mm[1])
        if width_mm <= 0 or height_mm <= 0:
            return image
        try:
            dpi = int(options.get("dpi") or _PRINT_FALLBACK_DPI)
        except (TypeError, ValueError):
            dpi = _PRINT_FALLBACK_DPI
        dpi = max(1, min(100000, dpi))
        bleed = max(0.0, float(options.get("bleed_mm") or 0.0))
        per_mm = dpi / 25.4
        trim_w = max(1, int(round(width_mm * per_mm)))
        trim_h = max(1, int(round(height_mm * per_mm)))
        bleed_px = int(round(bleed * per_mm))
        page = Image.new(
            "RGBA", (trim_w + bleed_px * 2, trim_h + bleed_px * 2),
            _hex_to_rgba(options["background"]),
        )
        art = image.convert("RGBA")
        scale = min(trim_w / art.width, trim_h / art.height)
        art_w = max(1, round(art.width * scale))
        art_h = max(1, round(art.height * scale))
        resample = _RESAMPLE_FILTERS.get(options.get("resample"), Image.LANCZOS)
        art = art.resize((art_w, art_h), resample)
        page.alpha_composite(
            art,
            (bleed_px + (trim_w - art_w) // 2, bleed_px + (trim_h - art_h) // 2),
        )
        if options.get("crop_marks") and bleed_px >= 3:
            self._draw_crop_marks(page, bleed_px, trim_w, trim_h)
        if options["transparent"] and options["format"] != "JPEG":
            return page
        return page.convert("RGB")

    @staticmethod
    def _draw_crop_marks(page, bleed_px, trim_w, trim_h):
        """Draw corner crop marks in the bleed margin of an RGBA ``page``."""
        draw = ImageDraw.Draw(page)
        gap = max(1, bleed_px // 4)
        arm = max(1, bleed_px - gap)
        left, top = bleed_px, bleed_px
        right, bottom = bleed_px + trim_w - 1, bleed_px + trim_h - 1
        ink = (0, 0, 0, 255)
        for x, y, sx, sy in ((left, top, -1, -1), (right, top, 1, -1),
                             (left, bottom, -1, 1), (right, bottom, 1, 1)):
            hx = x + sx * gap
            draw.line((hx, y, hx + sx * arm, y), fill=ink, width=1)
            vy = y + sy * gap
            draw.line((x, vy, x, vy + sy * arm), fill=ink, width=1)

    def _estimate_export(self, options):
        """Encode the current canvas in memory for the dialog's size read-out."""
        if self.current is None:
            return None
        image = self._export_image(self.current, options)
        buffer = io.BytesIO()
        self._save_export(
            image, buffer, options["format"], options,
            getattr(self.current, "working_profile", colormanagement.SRGB),
        )
        return buffer.tell(), image.size

    def _export_icc(self, image, options, working_profile):
        """Deprecated shim; see module-level :func:`_export_icc`."""
        return _export_icc(image, options, working_profile)

    def _save_export(self, image, path, fmt, options, working_profile=None):
        """Save ``image`` to ``path`` honouring the per-format export options."""
        kwargs = {"format": fmt}
        dpi = options.get("dpi")
        if dpi and fmt != "GIF":
            kwargs["dpi"] = (dpi, dpi)
        # Only JPEG and TIFF can hold CMYK; anything else falls back to RGB so a
        # stray CMYK image cannot reach an encoder that rejects it.
        if image.mode == "CMYK" and fmt not in ("JPEG", "TIFF"):
            image = image.convert("RGB")
        # GIF cannot carry an ICC profile; every other format can.
        if fmt != "GIF":
            icc = _export_icc(image, options, working_profile)
            if icc:
                kwargs["icc_profile"] = icc
        metadata = _export_metadata(options)
        xmp = _export_xmp(metadata)
        if fmt == "JPEG":
            opts = options.get("jpeg", {})
            kwargs["quality"] = options["quality"]
            kwargs["optimize"] = bool(opts.get("optimize", True))
            kwargs["progressive"] = bool(opts.get("progressive", False))
            kwargs["subsampling"] = int(opts.get("subsampling", 2))
            if image.mode not in ("RGB", "CMYK"):
                image = image.convert("RGB")
            if metadata:
                kwargs["exif"] = _export_exif(metadata)
                if xmp:
                    kwargs["xmp"] = xmp
        elif fmt == "TIFF":
            if image.mode not in ("RGB", "CMYK", "RGBA", "L"):
                image = image.convert("RGB")
            if metadata:
                exif = _export_exif(metadata)
                if xmp:
                    exif[700] = xmp  # XMP tag in IFD0
                kwargs["exif"] = exif
        elif fmt == "WEBP":
            opts = options.get("webp", {})
            if opts.get("lossless"):
                kwargs["lossless"] = True
            else:
                kwargs["quality"] = options["quality"]
            kwargs["method"] = int(opts.get("method", 4))
            if metadata:
                kwargs["exif"] = _export_exif(metadata)
                if xmp:
                    kwargs["xmp"] = xmp
        elif fmt == "GIF":
            kwargs["optimize"] = bool(options.get("gif", {}).get("optimize", False))
        else:
            opts = options.get("png", {})
            kwargs["compress_level"] = int(opts.get("compress_level", 6))
            kwargs["optimize"] = bool(opts.get("optimize", False))
            if opts.get("interlace"):
                kwargs["interlace"] = True
            if metadata:
                kwargs["pnginfo"] = _export_pnginfo(metadata)
        image.save(path, **kwargs)

    @staticmethod
    def _safe_name(name, fallback_index):
        cleaned = "".join(
            ch for ch in (name or "") if ch not in '<>:"/\\|?*'
        ).strip()
        return cleaned or f"canvas_{fallback_index}"

    # ------------------------------------------------------------------ projects
    def _canvas_size_text(self):
        if self.current is None:
            return "-- x --"
        return f"{self.current.width} x {self.current.height}"

    def _update_title(self):
        name = os.path.basename(self.project_path) if self.project_path else "Untitled"
        canvas = self.current.name if self.current is not None else "No Canvas"
        self.root.title(f"{APP_NAME} {APP_VERSION} - {name} - {canvas}")
        self.statusbar.set_canvas_size(self._canvas_size_text())

    def open_project(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open Project",
            filetypes=[("Pai project", "*.pai"), ("All files", "*.*")],
        )
        if not path:
            return
        self._confirm_discard_changes(lambda: self._start_project_load(path))

    def _start_project_load(self, path):
        """Queue a project load so its decode reports progress and can cancel.

        The archive is decoded on the queue's worker thread (``project.read``)
        and installed on the Tk thread (``project.apply``), so opening a large
        project no longer freezes the window while its layer images decode. The
        load shows on the status bar's progress bar and in the Queue panel, and
        Escape or a row's stop button cancels it.
        """
        name = os.path.basename(path)

        def runner(progress, cancel):
            return project.read(self, path, progress=progress, cancel=cancel)

        def finisher(payload, error, cancelled):
            self._finish_project_load(path, payload, error, cancelled)

        self.submit_project_job("project", f"Open {name}", runner, finisher)
        self.set_result(f"Opening {name}\u2026")

    def _finish_project_load(self, path, payload, error, cancelled):
        """Install a finished project load, or report why it did not happen."""
        if cancelled:
            self.statusbar.set_status("cancelled")
            self.set_result("Open project cancelled.")
            return
        if error is not None:
            self._report_project_failure(error)
            return
        try:
            project.apply(self, payload)
        except Exception as exc:
            self._report_project_failure(str(exc))
            return
        self.project_path = path
        self._clear_generation_preview()
        self._refresh_history()
        self.area.rebuild(self.canvases)
        self._sync_shared_guides()
        self._activate_canvas(self.active_canvas)
        self._set_modified(False)
        self.set_result(f"Opened project {os.path.basename(path)}.")
        self.statusbar.set_status(f"opened {os.path.basename(path)}")

    def _report_project_failure(self, message):
        """Report a failed project load in the status bar, the log, and a dialog."""
        # The queue wraps a runner's exception as "Error: ..."; show the
        # explanation itself rather than doubling the prefix.
        if message.startswith("Error: "):
            message = message[len("Error: "):]
        self.statusbar.set_status("failed")
        self.set_result(f"Could not open project: {message}", "error")
        dialogs.show_error(
            self.root, "Open Project", f"Could not open project:\n{message}"
        )

    def close_project(self):
        """Close the open project, leaving an empty workspace.

        Unsaved changes are confirmed first; the canvases, guides and project
        path are then dropped, so the next New or Open starts from a clean
        state. The workspace is left with no canvas rather than a fresh blank
        one, which is what :meth:`new_document` is for.
        """
        self._confirm_discard_changes(self._close_project_now)

    def _close_project_now(self):
        self._reset_document()
        self.set_result("Project closed.")

    def save_project(self, on_done=None):
        """Save the project, prompting for a path when it has none yet.

        The save runs on the project worker: the current state is snapshotted on
        the Tk thread and encoded off-thread, so a large project no longer
        freezes the window. ``on_done(ok)`` runs on the Tk thread once the save
        finishes (``ok`` is ``False`` when it was cancelled or failed), so a
        caller can continue only after a successful write.
        """
        if not self.project_path:
            self.save_project_as(on_done=on_done)
            return
        self._write_project(self.project_path, on_done=on_done)

    def save_project_as(self, on_done=None):
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Project As",
            defaultextension=".pai",
            filetypes=[("Pai project", "*.pai"), ("All files", "*.*")],
        )
        if not path:
            if on_done is not None:
                on_done(False)
            return
        self._write_project(path, on_done=on_done)

    def _write_project(self, path, on_done=None):
        """Queue a project save on the project worker.

        The snapshot is taken here, on the Tk thread (it reads the tkinter
        variables and copies the layer images); ``project.write`` then encodes
        it off-thread, with the queue driving the status-bar progress.
        """
        name = os.path.basename(path)
        payload = project.snapshot(self)

        def runner(progress, cancel):
            return project.write(payload, path, progress=progress, cancel=cancel)

        def finisher(manifest, error, cancelled):
            if cancelled:
                self.statusbar.set_status("cancelled")
                self.set_result(f"Save {name} cancelled.", "warn")
                if on_done is not None:
                    on_done(False)
                return
            if error is not None:
                self._report_save_failure(error)
                if on_done is not None:
                    on_done(False)
                return
            self.project_path = path
            self._update_title()
            self._set_modified(False)
            self.set_result(f"Saved project {name}.")
            self.statusbar.set_status(f"saved {name}")
            if on_done is not None:
                on_done(True)

        self.submit_project_job("project", f"Save {name}", runner, finisher)
        self.set_result(f"Saving {name}\u2026")

    def _report_save_failure(self, message):
        """Report a failed project save in the status bar, the log, and a dialog."""
        # The queue wraps a runner's exception as "Error: ..."; show the
        # explanation itself rather than doubling the prefix.
        if message.startswith("Error: "):
            message = message[len("Error: "):]
        self.statusbar.set_status("failed")
        self.set_result(f"Could not save project: {message}", "error")
        dialogs.show_error(
            self.root, "Save Project", f"Could not save project:\n{message}"
        )

    def _confirm_discard_changes(self, on_ready):
        """Offer to save pending changes, then run ``on_ready`` when safe.

        Because saving is asynchronous, "safe to proceed" is delivered through
        ``on_ready`` rather than a return value: it is called once the project
        is written, or immediately when there is nothing to save or the user
        chooses to discard.
        """
        if not self.modified:
            on_ready()
            return
        choice = dialogs.ask_save_changes(
            self.root,
            "Unsaved Changes",
            "This project has unsaved changes.\n\n"
            "Save them before continuing?",
        )
        if choice == "save":
            self.save_project(on_done=lambda ok: on_ready() if ok else None)
        elif choice == "discard":
            on_ready()

    def show_about(self):
        dialogs.show_info(
            self.root,
            f"About {APP_NAME}",
            f"{APP_NAME} {APP_VERSION}\n\n"
            "This is a painting app with local image generation written in Python. "
            "It exists because I got tired of jumping between editors and AI tools.\n\n"
            "Anyway, go paint and edit something!",
        )

    def show_donate(self):
        dialogs.show_donate(
            self.root,
            DONATE_URL,
            "Pai is free: no ads, no telemetry, no SAAS. "
            "If it's been useful to you, a small tip helps keep the updates coming. "
            "And if not, no worries, it stays free either way."
        )

    def show_contribute(self):
        dialogs.show_contribute(
            self.root,
            CONTRIBUTE_URL,
            "Pai only gets better with other people poking at it. Bug reports, "
            "feature ideas, code, docs, translations, or just telling me what "
            "broke: all welcome. The link below has the details on how to get "
            "involved.",
        )

    def open_discord(self):
        dialogs.open_url(DISCORD_URL)

    # ------------------------------------------------------------------ theme
    def open_settings(self):
        """Open the Settings dialog (hotkeys, interface and canvas defaults)."""
        settingsdialog.AppSettingsDialog(self.root, self)

    def open_color_management(self):
        """Edit the active canvas's working and soft-proof ICC profiles."""
        if self.current is None:
            self.set_result("There is no canvas to configure.")
            return
        canvas = self.current
        default = getattr(self, "default_working_profile", colormanagement.SRGB)
        working = getattr(canvas, "working_profile", colormanagement.SRGB)
        result = dialogs.ask_color_management(
            self.root,
            working=working,
            proof=getattr(canvas, "proof_profile", None),
            proof_intent=getattr(canvas, "proof_intent", "relative"),
            make_default=(working == default),
        )
        if result is None:
            return
        canvas.working_profile = result["working_profile"] or colormanagement.SRGB
        canvas.proof_profile = result["proof_profile"] or None
        canvas.proof_intent = result["proof_intent"]
        if result.get("make_default"):
            self.default_working_profile = canvas.working_profile
            settings = theme.read_settings()
            color = settings.get("color")
            color = color if isinstance(color, dict) else {}
            color["working_profile"] = canvas.working_profile
            settings["color"] = color
            theme.write_settings(settings)
        self._set_modified(True)
        self.render()
        label = colormanagement.profile_label(canvas.working_profile)
        proof = colormanagement.profile_label(canvas.proof_profile) if canvas.proof_profile else None
        self.set_result(
            f"Working space: {label}." if not proof
            else f"Working space: {label}; proofing {proof}."
        )

    def open_model_settings(self):
        """Edit where model files live; refresh the panels when saved."""
        modelsettings.ModelSettingsDialog(
            self.root, on_saved=self._on_models_changed,
        )

    def _on_models_changed(self):
        self.gen.refresh_models()
        self.controlnet_panel.refresh_models()
        self.bg.refresh_models()
        self.upscale.refresh_models()
        self.loaded_models_panel.refresh()
        self.set_result("Model settings saved.")

    def toggle_theme(self):
        """Switch between the light and dark themes."""
        self.set_theme("light" if theme.is_dark() else "dark")

    def set_theme(self, name):
        """Apply ``name`` (``"light"``, ``"dark"`` or ``"system"``) and remember it."""
        if name not in ("light", "dark", "system"):
            name = "light"
        resolved = theme.os_preference() if name == "system" else name
        if hasattr(self, "dark_var"):
            self.dark_var.set(resolved == "dark")
        theme.apply(resolved)
        theme.save_preference(name)
        self.set_result(f"{name.capitalize()} theme.")

    # ------------------------------------------------------------------ results
    def set_result(self, text, kind=None):
        """Show ``text`` in the status bar and append it to the Logs panel.

        ``kind`` forces a severity (``"info"``, ``"success"``, ``"warn"`` or
        ``"error"``); when omitted the message is classified from its wording
        (see :func:`_classify_result`). An empty ``text`` just clears the bar
        without adding a line to the log.
        """
        if kind is None:
            kind = _classify_result(text)
        self.statusbar.set_result(text, kind)
        log_panel = getattr(self, "log_panel", None)
        if text and log_panel is not None:
            log_panel.append(kind, text)

    def clear_logs(self):
        """Empty the Logs panel."""
        log_panel = getattr(self, "log_panel", None)
        if log_panel is not None:
            log_panel.clear()

    def _poll_queue(self):
        try:
            while True:
                message = self.queue.get_nowait()
                # Project jobs are posted through the project worker, which tags
                # its messages so the job handlers know which queue to look in.
                source = self.job_queue
                if message[0] == "project":
                    source = self.project_queue
                    message = message[1:]
                tag = message[0]
                if tag == "sysstats":
                    _tag, cpu, gpu, ram = message
                    self.statusbar.set_system_stats(cpu, gpu, ram)
                elif tag == "jobstate":
                    _tag, job_id = message
                    self._on_job_state(source, job_id)
                elif tag == "jobprogress":
                    _tag, job_id, frac, label = message[:4]
                    sub = message[4] if len(message) > 4 else None
                    self._on_job_progress(source, job_id, frac, label, sub)
                elif tag == "genpreview":
                    _tag, doc, image = message
                    self._on_gen_preview(doc, image)
                elif tag == "jobdone":
                    _tag, job_id, result, error, cancelled = message
                    self._on_job_done(source, job_id, result, error, cancelled)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_close(self):
        self._confirm_discard_changes(self._quit)

    def _quit(self):
        self.monitor.stop()
        # Persist any kinds detected this session (e.g. the startup control
        # adapter scan) so the next session opens with them warm.
        from engine import component_kinds as engine_kinds

        engine_kinds.save_cache()
        self.root.destroy()

    # ------------------------------------------------------------------ generate
    def _refresh_job_state(self):
        """Sync the busy flag and the form Cancel buttons with the queue."""
        self.busy = self.job_queue.has_running()
        jobs = self.job_queue.jobs()

        def panel_state(kinds):
            # A job still counts as active while it is cancelling, but then it is
            # no longer cancellable: the button greys out so a second click
            # cannot pile up (and the status text says "Cancelling...").
            active = any(job.active and job.kind in kinds for job in jobs)
            cancellable = any(job.cancellable and job.kind in kinds for job in jobs)
            return active, cancellable

        gen_active, gen_cancellable = panel_state(("generate", "preload"))
        bg_active, bg_cancellable = panel_state(("background",))
        up_active, up_cancellable = panel_state(("upscale",))
        # A job is cancellable whether it is running or still waiting in line,
        # so the Cancel button follows "is there anything of this kind active".
        self.gen.set_busy(gen_active, cancellable=gen_cancellable)
        self.bg.set_busy(bg_active, cancellable=bg_cancellable)
        self.upscale.set_busy(up_active, cancellable=up_cancellable)

    def _all_jobs(self):
        """Every job across both workers, oldest first."""
        return sorted(
            self.job_queue.jobs() + self.project_queue.jobs(),
            key=lambda job: job.id,
        )

    def _queue_for(self, job_id):
        """The queue that owns ``job_id`` (ids are unique across both)."""
        if self.project_queue.get(job_id) is not None:
            return self.project_queue
        return self.job_queue

    def _cancel_current(self):
        """Cancel whichever job is running (used by Escape)."""
        job = self.job_queue.current or self.project_queue.current
        if job is not None:
            self._queue_for(job.id).cancel(job.id)

    def _dismiss_job(self, job_id):
        """Row button: cancel a job that is still active, drop it once finished."""
        queue_ = self._queue_for(job_id)
        job = queue_.get(job_id)
        if job is None:
            return
        if job.active:
            queue_.cancel(job_id)
            return
        queue_.remove(job_id)
        self.job_panel.rebuild(self._all_jobs())

    def _cancel_all_jobs(self):
        self.job_queue.cancel_all()
        self.project_queue.cancel_all()

    def _clear_finished_jobs(self):
        self.job_queue.clear_finished()
        self.project_queue.clear_finished()
        self.job_panel.rebuild(self._all_jobs())

    def submit_job(self, kind, label, runner, finisher, detail=""):
        """Queue a long-running job; the worker runs it once earlier jobs finish."""
        self._touch_model_activity()
        job = Job(kind, label, runner, finisher, detail=detail)
        self.job_queue.submit(job)
        return job

    def _touch_model_activity(self):
        """Restart the auto-unload idle clock (any engine use counts)."""
        self._last_model_activity = time.monotonic()

    def _auto_unload_tick(self):
        """Periodic check that drops cached models after the configured idle time."""
        try:
            self._maybe_auto_unload()
        except Exception:
            pass
        self.root.after(30_000, self._auto_unload_tick)

    def _maybe_auto_unload(self):
        from engine import loaded_models as engine_models
        from engine import tuning

        seconds = tuning.auto_unload_seconds()
        if not seconds or self._has_running_job():
            return
        if time.monotonic() - self._last_model_activity < seconds:
            return
        if not engine_models.loaded():
            return
        engine_models.unload(None)
        if hasattr(self, "loaded_models_panel"):
            self.loaded_models_panel.refresh()
            self.loaded_models_panel.set_load_status(
                "Models unloaded after inactivity.", "muted",
            )
        self._touch_model_activity()

    def submit_project_job(self, kind, label, runner, finisher, detail=""):
        """Queue a project open/save on its own worker (see ``project_queue``)."""
        job = Job(kind, label, runner, finisher, detail=detail)
        self.project_queue.submit(job)
        return job

    def _on_job_state(self, source, job_id):
        """A job was queued, started, or cancelled while waiting."""
        job = source.get(job_id)
        if job is not None:
            self.job_panel.update(job)
        self._refresh_job_state()
        self._sync_preload_status(job)
        # Immediate feedback while a running job winds down, so a Cancel that
        # takes a moment to reach the engine still reads as acknowledged.
        if job is not None and job.cancelling and job.active:
            self.statusbar.set_status("Cancelling\u2026")

    def _has_running_job(self):
        """True while either worker runs a job, so Escape can cancel it."""
        return (
            self.job_queue.current is not None
            or self.project_queue.current is not None
        )

    def _on_job_progress(self, source, job_id, frac, label, sub=None):
        self._touch_model_activity()
        job = source.get(job_id)
        kind = job.kind if job is not None else ""
        # Once a cancel has been requested the engine may still emit progress
        # until it reaches a safe point; keep the "cancelling" feedback rather
        # than letting late ticks repaint the status back to a running label.
        if job is not None and job.cancelling:
            return
        self.job_panel.set_progress(job_id, frac, label)
        # The status bar has two meters: the far-left one for project
        # open/save/export and the one next to the status label for the running
        # job (a generation, upscale, removal or model preload). They serve
        # different sources, so they never fight over the same bar.
        if source is self.project_queue:
            self.statusbar.set_load_progress(frac)
        else:
            self.statusbar.set_job_progress(frac)
        if label:
            self.statusbar.set_status(f"{label}...")
        if job is not None and kind == "preload":
            self._preload_progress(label, frac)

    def _on_job_done(self, source, job_id, result, error, cancelled):
        job = source.get(job_id)
        if job is not None:
            self.job_panel.update(job)
            try:
                job.finisher(result, error, cancelled)
            except Exception as exc:
                self.set_result(f"Error: {exc}")
        # A finished job marks engine activity, so the auto-unload idle clock
        # measures time since the *last* job ended, not since it was submitted.
        self._touch_model_activity()
        # Clear the meter this job was driving.
        if source is self.project_queue:
            self.statusbar.reset_load_progress()
        else:
            self.statusbar.reset_job_progress()
        self._refresh_job_state()
        self.loaded_models_panel.refresh()
        self._finish_preload_status(job, result, error, cancelled)

    def _sync_preload_status(self, job):
        """Mirror a preload job's queued/running state onto the two panels."""
        if job is None or job.kind != "preload":
            return
        if job.status == RUNNING:
            if job.cancelling:
                self._set_preload_status("cancelling\u2026", "Cancelling\u2026", "muted")
            else:
                self._preload_progress(job.progress_label, job.progress or 0.0)
        elif job.status == QUEUED:
            self._set_preload_status("queued\u2026", "Load queued\u2026", "muted")
        elif job.status == CANCELLED:
            self._set_preload_status("cancelled", "Load cancelled.", "muted")

    def _preload_progress(self, label, frac):
        """Show a preload's progress on the Generate hint and the Loaded Models panel."""
        label = label or "loading"
        percent = frac * 100
        self.gen.set_preload_status(f"{label}\u2026 {percent:.0f}%", "accent")
        self.loaded_models_panel.set_load_status(
            f"{_capitalize(label)}\u2026 {percent:.0f}%", "accent"
        )

    def _finish_preload_status(self, job, result, error, cancelled):
        """Settle a finished preload job's status on the two panels."""
        if job is None or job.kind != "preload":
            return
        if cancelled:
            self._set_preload_status("cancelled", "Load cancelled.", "muted")
        elif error is not None:
            self._set_preload_status("failed", f"Load failed: {error}", "error")
        else:
            self._set_preload_status("loaded", f"Loaded {result}.", "text")

    def _set_preload_status(self, gen_text, mem_text, token="muted"):
        """Show a preload status on the Generate hint and the Loaded Models panel."""
        self.gen.set_preload_status(gen_text, token)
        self.loaded_models_panel.set_load_status(mem_text, token)

    def unload_models(self, key=None, label=None):
        """Drop the engine's cached models to free RAM/VRAM (Loaded Models panel).

        With no ``key`` every cached model is dropped; with a record ``key`` only
        that one model is dropped. ``label`` names the model in the status line.
        """
        if self.job_queue.has_running():
            self.set_result("Cannot unload models while a job is running.", "warn")
            return
        from engine import loaded_models

        try:
            dropped = loaded_models.unload(key)
        except Exception as exc:
            self.set_result(f"Unload failed: {exc}", "error")
            return
        self.loaded_models_panel.refresh()
        if not dropped:
            self.loaded_models_panel.set_load_status("No models were loaded.", "muted")
            return
        if key and label:
            message = f"Unloaded {label}."
        else:
            message = f"Unloaded {dropped} model{'s' if dropped != 1 else ''}."
        self.loaded_models_panel.set_load_status(message, "muted")
        self.set_result(message)

    def _show_doc(self, doc):
        """Refresh the view when ``doc`` is the active canvas."""
        if doc is self.current:
            self._refresh_layer_list()
            self.render()

    def _layer_init_image(self, layer):
        """A single layer's rendered pixels over white, at its canvas position.

        Uses the layer's composited contribution (filters, text and mask) so the
        init image matches what the layer looks like, placed at ``layer.offset``
        so it lines up with the canvas. Visibility is ignored, so a hidden layer
        can still serve as a reference.
        """
        doc = self.current
        base = Image.new("RGBA", doc.size, (255, 255, 255, 255))
        contrib = layer.composited_image()
        if contrib.mode != "RGBA":
            contrib = contrib.convert("RGBA")
        ox, oy = int(layer.offset[0]), int(layer.offset[1])
        sx0 = max(0, -ox)
        sy0 = max(0, -oy)
        sx1 = min(contrib.width, doc.width - ox)
        sy1 = min(contrib.height, doc.height - oy)
        if sx1 > sx0 and sy1 > sy0:
            sub = contrib.crop((sx0, sy0, sx1, sy1))
            base.alpha_composite(sub, dest=(ox + sx0, oy + sy0))
        return base.convert("RGB")

    def _asset_init_image(self, asset):
        """The selected asset's pixels over white, as an init image.

        The asset is used at its own size and the sampler scales it to the
        requested output size, so an imported asset of any size works as the
        source. Transparency is flattened onto white, matching the layer source,
        so a shaped asset reads against the sampler's RGB input.
        """
        image = asset.image
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        base = Image.new("RGBA", image.size, (255, 255, 255, 255))
        base.alpha_composite(image)
        return base.convert("RGB")

    @staticmethod
    def _control_entries(rows):
        """The engine config entries for a list of ControlNet panel rows."""
        return [
            {
                "kind": row.get("kind"),
                "model": row.get("model"),
                "preprocessor": row.get("preprocessor"),
                "strength": row.get("strength", 1.0),
                "start": row.get("start", 0.0),
                "end": row.get("end", 1.0),
            }
            for row in rows
        ]

    def _resolve_control_hints(self, rows, init_image):
        """One control hint image per adapter row, or ``None`` after an error.

        Each row picks its own source, so different adapters can steer from
        different images (the 3D-pose mesh and the regional colour map, for
        instance). ``Source`` reuses the run's own source image, matching the
        historical single-ControlNet behaviour.
        """
        images = []
        for row in rows:
            hint = row.get("hint") or HINT_SOURCE
            if hint == HINT_CANVAS:
                image = self.composite()
            elif hint == HINT_LAYER:
                layer = self.active_layer()
                if layer is None:
                    self.set_result("Select a layer for the control hint.", "warn")
                    return None
                image = self._layer_init_image(layer)
            elif hint == HINT_ASSET:
                asset = self.selected_asset()
                if asset is None:
                    self.set_result("Select an asset for the control hint.", "warn")
                    return None
                image = self._asset_init_image(asset)
            else:
                if init_image is None:
                    self.set_result(
                        "A control adapter set to Source needs a Source image.",
                        "warn",
                    )
                    return None
                image = init_image
            images.append(image)
        return images

    def _capture_reference_selection(self, kind):
        """Snapshot the current selection of ``kind`` into a reference binding.

        ``kind`` is ``"asset"``, ``"layer"`` or ``"canvas"``. The returned dict
        pins a specific asset id or layer name (or the canvas) together with a
        display label, so two reference rows can hold two different assets.
        ``None`` (with a warning) means there was nothing of that kind selected.
        """
        if kind == "canvas":
            if self.current is None:
                self.set_result("There is no canvas to assign.", "warn")
                return None
            return {"source": "canvas", "label": "Canvas"}
        if kind == "asset":
            asset = self.selected_asset()
            if asset is None:
                self.set_result("Select an asset to assign.", "warn")
                return None
            return {"source": "asset", "asset": asset.id, "label": f"Asset: {asset.name}"}
        if kind == "layer":
            layer = self.active_layer()
            if layer is None:
                self.set_result("Select a layer to assign.", "warn")
                return None
            return {"source": "layer", "layer": layer.name, "label": f"Layer: {layer.name}"}
        self.set_result("Unknown reference kind.", "warn")
        return None

    def _binding_image(self, binding):
        """Resolve one reference binding to a plain RGB image, or ``None``.

        A binding pinned to an asset id or layer name resolves that exact item;
        a binding with only a kind (an unassigned row, or an old project) falls
        back to the current selection of that kind. The image is captured now, so
        later selection, layer or canvas edits cannot retarget a queued job.
        """
        source = binding.get("source")
        if source == "canvas":
            return self.composite()
        if source == "layer":
            name = binding.get("layer")
            layer = None
            if name is not None and self.current is not None:
                layer = next(
                    (l for l in self.current.layers if l.name == name), None
                )
            if layer is None:
                layer = self.active_layer()
            if layer is None:
                self.set_result("Select a layer for the reference image.", "warn")
                return None
            return self._layer_init_image(layer)
        if source == "asset":
            asset_id = binding.get("asset")
            asset = None
            if asset_id is not None and self.current is not None:
                asset = next(
                    (a for a in self.current.assets if a.id == asset_id), None
                )
            if asset is None:
                asset = self.selected_asset()
            if asset is None:
                self.set_result("Select an asset for the reference image.", "warn")
                return None
            return self._asset_init_image(asset)
        self.set_result("Set the reference source first.", "warn")
        return None

    def _resolve_anima_reference(self, bindings):
        """The Anima reference image: the rows' images stitched side by side.

        Two characters in one image are the adapter's convention, so several
        assigned references are laid out left-to-right into a single reference.
        Returns ``None`` after an error (a warning is shown).
        """
        images = []
        for binding in bindings:
            image = self._binding_image(binding)
            if image is None:
                return None
            images.append(image)
        if not images:
            return None
        return _stitch_side_by_side(images)

    def _resolve_qwen_references(self, bindings):
        """One image per Qwen reference row, in order, or ``None`` after an error."""
        images = []
        for binding in bindings:
            image = self._binding_image(binding)
            if image is None:
                return None
            images.append(image)
        return images

    def _capture_region_mask(self, source, warn=True):
        """A copy of the mask a region row should snapshot, or ``None``.

        The Regions panel asks for this when a row captures its mask: the
        current selection, or the canvas' painted inpaint mask. The copy is
        taken now, so painting or re-selecting later cannot change what the row
        holds. ``None`` (with an optional warning) means there was nothing of
        that kind to capture.
        """
        if self.current is None:
            return None
        if source == MASK_PAINTED:
            if not self.current.has_inpaint_mask():
                if warn:
                    self.set_result("Paint an inpaint mask for the region first.", "warn")
                return None
            return self.current.inpaint_mask.copy()
        selection = self.selection
        if selection is None or selection.is_empty():
            if warn:
                self.set_result("Select a region for the region prompt first.", "warn")
            return None
        return selection.mask.copy()

    # ----------------------------------------------------------------- layout
    def _layout_canvas_size(self):
        """The active canvas size, the coordinate space layout boxes are drawn in."""
        if self.current is None:
            return (1024, 1024)
        return self.current.size

    def _layout_output_size(self):
        """The generation size the layout boxes are scaled into."""
        values = self.gen.settings() if hasattr(self, "gen") else None
        if not values:
            return self._layout_canvas_size()
        return (values["width"], values["height"])

    def _layout_selection_box(self):
        """The current rectangular selection's box, or ``None`` after a warning."""
        selection = self.selection
        if selection is None or selection.is_empty():
            self.set_result("Select a rectangle first.", "warn")
            return None
        return selection.bbox

    def add_layout_box(self, box):
        """Add a layout item for a rectangle drawn with the Layout tool."""
        if not hasattr(self, "tools"):
            return
        self.tools.layout_model.enabled = True
        self.tools.layout_model.add_box(box)
        self.tools.refresh_layout_settings()
        self._on_layout_change()
        self.set_result("Added a layout item; fill in its fields.", "success")

    def _sync_layout_overlay(self):
        """Show the layout boxes (with names/captions) while the Layout tool runs."""
        if not hasattr(self, "tools") or not hasattr(self, "area"):
            return
        view = self.area.current_view()
        if view is None:
            return
        model = self.tools.layout_model
        if self.tools.layout_tool and model.enabled:
            view.set_layout_boxes(model.overlay_items())
        else:
            view.set_layout_boxes([])

    def _resolve_region_masks(self, rows):
        """One mask image per region row, or ``None`` after an error.

        Each row carries its own captured snapshot (a base64 PNG); decode it
        here. A row the user added but never gave a mask to cannot be applied,
        so the run is refused rather than silently conditioning the whole image.
        """
        regions = []
        for row in rows:
            prompt = (row.get("prompt") or "").strip()
            mask = decode_mask(row.get("mask"))
            if mask is None:
                self.set_result("Give every region prompt a mask first.", "warn")
                return None
            try:
                strength = float(row.get("strength", 1.0))
            except (TypeError, ValueError):
                strength = 1.0
            regions.append({"prompt": prompt, "mask": mask, "strength": strength})
        return regions

    # Long edge a cropped inpaint region is rendered at, in pixels. Families
    # compress the image 8x-16x, so a region smaller than this otherwise lands
    # on only a handful of latent cells and comes back as a smudge.
    _INPAINT_CROP_TARGET = 768
    # Never magnify a region more than this when reframing it for the run.
    _INPAINT_CROP_MAX_SCALE = 4.0

    def _plan_inpaint_crop(self, image, mask, feather, panel_w, panel_h):
        """Crop a small inpaint region and upscale it for the run.

        Renders the region ("inpaint crop then stitch") so its mask spans most
        of the frame the model sees instead of a few latent cells. Returns
        ``(crop_image, crop_mask, width, height, stitch)``, or ``None`` when the
        region already fills the run frame (the plain full-frame path is then
        the better context). ``stitch`` carries the base image and the feathered
        crop mask used to composite the finished crop back (see
        :meth:`_stitch_inpaint_crop`).
        """
        box = mask.getbbox()
        if box is None:
            return None
        left, top, right, bottom = box
        pad = max(32, int(round(0.5 * max(right - left, bottom - top))))
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(mask.width, right + pad)
        bottom = min(mask.height, bottom + pad)
        crop_w, crop_h = right - left, bottom - top
        if crop_w < 16 or crop_h < 16:
            return None
        if max(crop_w, crop_h) >= max(panel_w, panel_h):
            return None
        scale = min(
            self._INPAINT_CROP_MAX_SCALE,
            max(1.0, self._INPAINT_CROP_TARGET / max(crop_w, crop_h)),
        )
        width = min(MAX_CANVAS_SIZE, max(64, int(round(crop_w * scale / 16)) * 16))
        height = min(MAX_CANVAS_SIZE, max(64, int(round(crop_h * scale / 16)) * 16))
        box = (left, top, right, bottom)
        crop_image = image.crop(box).resize((width, height), Image.LANCZOS)
        crop_mask = mask.crop(box).resize((width, height), Image.BILINEAR)
        # The sampler runs with a hard region (the model pins the source outside
        # it); the seam is softened here, once, at composite time.
        stitch_mask = mask.crop(box)
        if feather > 0:
            stitch_mask = stitch_mask.filter(ImageFilter.GaussianBlur(feather))
        stitch = {"box": box, "mask": stitch_mask, "base": image}
        return crop_image, crop_mask, width, height, stitch

    def _stitch_inpaint_crop(self, result, stitch):
        """Composite a finished crop back into its base image through the mask."""
        left, top, right, bottom = stitch["box"]
        patch = result.convert("RGBA").resize(
            (right - left, bottom - top), Image.LANCZOS
        )
        base = stitch["base"].convert("RGBA").copy()
        base.paste(patch, (left, top), stitch["mask"])
        return base.convert("RGB")

    def start_generate(self):
        if self.current is None:
            self.set_result("There is no canvas. Add a canvas first.")
            return
        prompt = self.gen.prompt()
        if not prompt:
            self.set_result("Enter a prompt first.")
            return
        config = self.gen.get_config()
        self.references_panel.apply_config(config)
        doc = self.current
        init_image = None
        inpaint_mask = None
        source, region = self.gen.mode()
        # Where the finished image goes, and the layer it targets. The layer is
        # captured now so a run queued against one layer cannot retarget another
        # if the user switches layers while it runs. With preview on, the
        # destination is chosen from the preview's buttons instead, so no target
        # is needed up front.
        output = self.gen.output()
        preview = self.gen.preview()
        target_layer = None
        if not preview and output in (OUTPUT_LAYER, OUTPUT_NEW_LAYER):
            target_layer = self.active_layer()
            if target_layer is None:
                self.set_result("There is no layer to generate into.")
                return
            if output == OUTPUT_LAYER and target_layer.is_locked():
                self._active_layer_locked()
                return
        if region == REGION_MASK:
            # Use the canvas's painted inpaint mask; capture a copy now so later
            # painting cannot retarget a queued job.
            if not doc.has_inpaint_mask():
                self.set_result("Paint an inpaint mask first.", "warn")
                return
            inpaint_mask = doc.inpaint_mask.copy()
        elif region == REGION_SELECTION:
            # Inpainting captures the selection now, so a later selection change
            # cannot retarget a queued job.
            selection = self.selection
            if selection is None or selection.is_empty():
                self.set_result("Select a region to inpaint first.")
                return
            inpaint_mask = selection.mask.copy()
        if source == SOURCE_LAYER:
            layer = self.active_layer()
            if layer is None:
                self.set_result("Select a layer to use as the init image.")
                return
            init_image = self._layer_init_image(layer)
        elif source == SOURCE_CANVAS:
            init_image = self.composite()
        elif source == SOURCE_ASSET:
            asset = self.selected_asset()
            if asset is None:
                self.set_result("Select an asset to use as the init image.")
                return
            init_image = self._asset_init_image(asset)
        if inpaint_mask is not None and init_image is None:
            # A region narrows a source edit, so a region with no source (an
            # unrecognised value) still falls back to the canvas.
            init_image = self.composite()
        # Krea 2 AnyPaint is a masked edit driven by the Source and Region inputs
        # (see engine.krea2_anypaint): it needs both a source and a mask.
        anypaint_on = self.references_panel.anypaint_enabled()
        if anypaint_on:
            if inpaint_mask is None:
                self.set_result(
                    "AnyPaint needs a painted Region mask (or a selection).", "warn"
                )
                return
            if init_image is None:
                self.set_result("AnyPaint needs a Source image.", "warn")
                return
            if not self.references_panel.anypaint_lora_path():
                self.set_result(
                    "Select an AnyPaint adapter in the References panel.", "warn"
                )
                return
        # Control adapters steer the run from a control hint. Each adapter picks
        # its own hint image, so a single run can combine several (for example an
        # Anima 3D-pose mesh and a regional colour map). The images are captured
        # now so a later change cannot retarget a queued job.
        control_rows = self.controlnet_panel.settings()
        control_on = self.controlnet_panel.enabled()
        control_images = None
        if control_rows and control_on:
            control_images = self._resolve_control_hints(control_rows, init_image)
            if control_images is None:
                return
            config["controlnets"] = self._control_entries(control_rows)
        # Regional prompting: each region pairs a prompt with a mask, captured
        # now so later selection/mask edits cannot retarget a queued job.
        region_rows = self.regions_panel.settings()
        region_on = self.regions_panel.enabled()
        region_entries = None
        if region_rows and region_on:
            region_entries = self._resolve_region_masks(region_rows)
            if region_entries is None:
                return
        if anypaint_on and (control_images or region_entries):
            self.set_result(
                "Turn off ControlNet and Region prompts when using AnyPaint.", "warn"
            )
            return
        # Anima reference adapter: capture the assigned reference images now (the
        # rows' bindings) so a later selection/canvas change cannot retarget a
        # queued job. Several rows are stitched into the one reference image the
        # adapter reads. Separate from the img2img Source.
        reference_image = None
        if self.gen.model_family() == "anima":
            bindings = self.references_panel.reference_bindings()
            if bindings:
                reference_image = self._resolve_anima_reference(bindings)
                if reference_image is None:
                    return
        # Reference images for multi-image edit (Qwen-Image and MiniMax H3 Image):
        # capture each now so a later selection/layer/canvas change cannot
        # retarget a queued job. They are separate from the img2img Source and
        # from the Anima reference.
        qwen_ref_images = None
        if self.gen.model_family() in ("qwen_image", "h3_image"):
            bindings = self.references_panel.qwen_reference_bindings()
            if bindings:
                if region_entries:
                    self.set_result(
                        "Turn off Region prompts when using reference images.", "warn"
                    )
                    return
                qwen_ref_images = self._resolve_qwen_references(bindings)
                if qwen_ref_images is None:
                    return
        # Layout Control: with the Layout panel enabled the run's prompt becomes
        # the layout prompt (the MOSS JSON layout, or the Krea 2 bbox/grounding
        # DSL) and the Positive prompt is its scene caption.
        layout_on = self.tools.layout_supported() and self.tools.layout_model.enabled
        layout_text = None
        if layout_on:
            if region_entries:
                self.set_result(
                    "Turn off Region prompts when generating from a layout.", "warn"
                )
                return
            if not self.tools.layout_model.has_items():
                self.set_result("Add at least one layout box first.", "warn")
                return
            layout_text = self.tools.layout_model.layout_text(
                self.gen.model_family(), prompt,
                self._layout_canvas_size(), self._layout_output_size(),
            )
        run_prompt = layout_text or prompt
        feather = config.get("feather") or 0.0
        # A small region is reframed ("inpaint crop") so its mask spans most of
        # the run frame rather than a few latent cells; the finished crop is
        # composited back through the feathered mask once the run returns. The
        # full-frame path is kept when the region already fills the frame, and
        # when ControlNet, regional prompts or a layout prompt steer the run (their
        # inputs are built for the full frame).
        stitch = None
        crop_ok = (
            self.gen.crop_region()
            and inpaint_mask is not None and init_image is not None
            and init_image.size == inpaint_mask.size
            and control_images is None and region_entries is None
            and not layout_on and not anypaint_on
        )
        crop = self._plan_inpaint_crop(
            init_image, inpaint_mask, feather,
            config["width"], config["height"],
        ) if crop_ok else None
        if crop is not None:
            init_image, inpaint_mask, config["width"], config["height"], stitch = crop
            config["feather"] = 0.0
        elif inpaint_mask is not None and feather > 0 and not anypaint_on:
            # Feather the region once, in document space, so the sampler's latent
            # mask and the final composite share the same soft edge. The sampler
            # is then left unfeathered to avoid doubling the blur.
            inpaint_mask = inpaint_mask.filter(ImageFilter.GaussianBlur(feather))
            config["feather"] = 0.0
        # The panel's fields at queue time, kept on a generated asset so selecting
        # it later can restore the run's settings. Captured now so a later panel
        # edit cannot rewrite what a queued job records.
        asset_settings = self.gen.asset_settings()
        if asset_settings is not None:
            asset_settings = dict(asset_settings)
            refs = self.references_panel.settings()
            if refs:
                asset_settings.update(refs)
            asset_settings["controlnets"] = control_rows
            asset_settings["controlnet_enabled"] = control_on
            asset_settings["region_prompts"] = region_rows
            asset_settings["region_prompts_enabled"] = region_on
            asset_settings["layout"] = self.tools.layout_model.settings()
            asset_settings["layout_meta"] = self.tools.layout_model.meta_tags()
            asset_settings["layout_enabled"] = self.tools.layout_model.enabled
            asset_settings["layout_show_caption"] = (
                self.tools.layout_model.show_caption)
        width = config["width"]
        height = config["height"]

        if not self._confirm_load_another(config, self.gen.model_var.get()):
            return

        cancel_state = {"token": None}

        def runner(progress, cancel):
            from engine.pipeline import generate as generate_image
            cancel_state["token"] = cancel
            return generate_image(
                run_prompt, config, image=init_image, mask=inpaint_mask,
                progress=progress, cancel=cancel, control=control_images,
                regions_list=region_entries, reference=reference_image,
                ref_images=qwen_ref_images, preview=preview_frame,
            )

        def preview_frame(image):
            self.queue.put(("genpreview", doc, image))

        def finisher(result, error, cancelled):
            self._clear_live_preview()
            token = cancel_state["token"]
            # A "stop and keep" run finished early on purpose: always show its
            # partial image in the accept/reject preview, even when the panel's
            # Preview toggle is off, so the user chooses what to do with it.
            kept = bool(token is not None and token.keep_requested)
            if stitch is not None and error is None and not cancelled:
                # The run produced the cropped frame; composite it back into the
                # full image before it is previewed or applied.
                result = self._stitch_inpaint_crop(result, stitch)
            self._apply_generated(
                doc, result, error, cancelled, prompt,
                output, target_layer, preview=(preview or kept),
                settings=asset_settings,
            )

        # A new run takes over the canvas preview: close any result still
        # awaiting accept/reject, so its stale overlay (and action bar) does not
        # block this run's live preview and is not left behind if this run lands
        # its pixels directly.
        if self._preview is not None:
            self._clear_generation_preview()

        self.submit_job(
            "generate", f"Generate {width}\u00d7{height}", runner, finisher,
            detail=_summarize(prompt),
        )
        self.set_result(
            "Generation queued for preview." if preview else "Generation queued."
        )

    def cancel_generate(self):
        """Cancel the active generation, or the model preload if none is running.

        The Generate panel's Cancel serves both jobs it starts (Generate and
        Preload); a generation takes priority so an in-flight preload behind it
        is left alone.
        """
        job = self.job_queue.running_or_last("generate")
        if job is None:
            job = self.job_queue.running_or_last("preload")
        if job is not None:
            self.job_queue.cancel(job.id)

    def stop_generate_keep(self):
        """Stop the running generation early and keep the image as denoised so far.

        The sampler returns its current ``x0`` estimate instead of finishing, so
        the partial image flows through the normal finisher (preview or apply).
        Unlike Cancel, this is not an abort: the job completes with a result.
        """
        job = self.job_queue.running_or_last("generate")
        if job is not None:
            self.job_queue.keep(job.id)

    def start_preload(self):
        """Load the Generate panel's current model into memory, without generating.

        Runs through the job queue so the load reports progress and can be
        cancelled from the Queue panel; the Loaded Models panel then lists what is
        cached.
        """
        config = self.gen.get_config()
        self.references_panel.apply_config(config)
        control_rows = self.controlnet_panel.settings()
        if control_rows and self.controlnet_panel.enabled():
            config["controlnets"] = self._control_entries(control_rows)
        label = self.model_var.get()

        if not self._confirm_load_another(config, label):
            return

        def runner(progress, cancel):
            from engine.pipeline import preload
            return preload(config, progress=progress, cancel=cancel)

        def finisher(result, error, cancelled):
            if cancelled:
                self.statusbar.set_status("cancelled")
                self.set_result("Preload cancelled.")
            elif error is not None:
                self.statusbar.set_status("failed")
                self.set_result(error, "error")
            else:
                self.statusbar.set_status("ready")
                self.set_result(f"Preloaded {result}.")

        self.submit_job("preload", f"Preload {label}", runner, finisher, detail=label)
        self.set_result("Preloading model...")

    def _confirm_load_another(self, config, label):
        """Confirm loading ``label`` when a different model is already cached.

        Returns ``True`` when it is safe to proceed (nothing loaded, or the
        selected model is already the loaded one), and ``False`` when the user
        declines to load another model alongside what is already in memory.
        """
        already = self._loaded_model_summary(config)
        if not already:
            return True
        listing = "\n".join(f"  \u2022 {item}" for item in already)
        return bool(dialogs.ask_yes_no(
            self.root,
            "Model Already Loaded",
            f"Models are already loaded:\n\n{listing}\n\n"
            f"Do you want to load another model ({label})?",
        ))

    def _loaded_model_summary(self, config=None):
        """``"Family \u2014 source"`` lines for models already cached in memory.

        Empty when nothing is loaded, or when the model ``config`` selects is
        itself already the loaded one (there is nothing new to load then).
        """
        try:
            from engine import loaded_models
            records = loaded_models.loaded()
        except Exception:
            return []
        if not records:
            return []
        selected = self._model_source(config)
        if selected and any(
            record.get("source") == selected for record in records
        ):
            return []
        lines = []
        for record in records:
            name = record.get("family") or record.get("source") or "Model"
            source = record.get("source")
            lines.append(f"{name} \u2014 {source}" if source else str(name))
        return lines

    @staticmethod
    def _model_source(config):
        """The file name of the generation model ``config`` selects, or ``None``."""
        if not config:
            return None
        for key in ("dit", "model", "checkpoint"):
            path = config.get(key)
            if path:
                return os.path.basename(path)
        return None

    def _apply_generated(self, doc, result, error, cancelled, prompt,
                         output=OUTPUT_ASSETS, target_layer=None, preview=False,
                         settings=None):
        """Deliver a finished generation to the destination it was queued for.

        ``output`` picks where the pixels land: ``In assets`` keeps them as a
        reference image to insert later (the default, so a run never overwrites
        existing work),         ``In layer`` writes them straight into ``target_layer``,
        and ``New layer`` inserts them as a new layer at the top of the stack.
        ``target_layer``
        is the layer captured when the job was queued, so a later layer switch
        cannot retarget the run. ``prompt`` is kept on an asset as a record of
        what produced it (and as a new layer's positive prompt), and ``settings``
        records the panel's fields so selecting the asset restores them. With
        ``preview`` the run is shown in the canvas viewport first and nothing is
        written until the user accepts it from the preview's buttons.
        """
        if cancelled:
            self.statusbar.set_status("cancelled")
            self.set_result("Generation cancelled.")
            return
        if error is not None:
            self.statusbar.set_status("failed")
            self.set_result(error, "error")
            return
        if preview:
            self._show_generation_preview(doc, result, prompt, settings)
            return
        # Snapshot the canvas the job was queued against, which may no longer be
        # the active one if the user switched tabs while it ran.
        doc.history.push(doc, self, "Generate")
        self._set_modified(True)
        if doc is self.current:
            self._refresh_history()
        image = result.convert("RGBA")
        name = _summarize(prompt) or "Generated"
        if output == OUTPUT_ASSETS:
            asset = doc.add_asset(
                name=name, image=image, source="generated", prompt=prompt,
                settings=settings,
            )
            self._show_doc(doc)
            self._select_asset(asset)
            self.statusbar.set_status("generated")
            self.set_result(
                f"Generated {result.width} x {result.height} image into assets."
            )
            return
        # In-layer and new-layer both land on a layer; the layer is missing only
        # if the canvas was rebuilt while the job ran, in which case a new layer
        # is the safe fallback.
        origin = ((doc.width - image.width) // 2,
                  (doc.height - image.height) // 2)
        if output == OUTPUT_LAYER and target_layer in doc.layers:
            target_layer.set_image(image)
            target_layer.offset = origin
            self._show_doc(doc)
            self.statusbar.set_status("generated")
            self.set_result(
                f"Generated {result.width} x {result.height} into '{target_layer.name}'."
            )
            return
        # A new layer goes on top of the stack so the freshly generated pixels are
        # visible over everything, regardless of which layer was selected.
        layer = doc.add_layer(name=name, index=len(doc.layers))
        layer.set_image(image)
        # Centre the generated pixels on the canvas, matching the preview, so a
        # run smaller (or larger) than the canvas is not anchored to the top-left.
        layer.offset = origin
        # Carry the run's positive prompt and settings onto the new layer, so
        # selecting it restores the prompt, negative prompt and sampler fields
        # that produced the image, matching what a generated asset records. The
        # positive prompt travels on the layer itself, so it is dropped from the
        # stored settings dict.
        layer.positive_prompt = prompt or ""
        if settings:
            layer.gen_settings = {
                key: value for key, value in settings.items()
                if key != "positive_prompt"
            }
        if doc is self.current:
            self.active = doc.layers.index(layer)
        self._show_doc(doc)
        self.statusbar.set_status("generated")
        self.set_result(
            f"Generated {result.width} x {result.height} into a new layer."
        )

    # --------------------------------------------------------- gen preview
    def _canvas_index(self, doc):
        """Index of ``doc`` in the canvas list, or ``None`` when it is gone."""
        for index, canvas in enumerate(self.canvases):
            if canvas is doc:
                return index
        return None

    def _on_gen_preview(self, doc, image):
        """Paint one live denoising-step frame onto its canvas.

        Frames use the same surface as the finished result: the target canvas is
        brought forward, the frame is dimmed and framed to fit (only on the
        first frame, so the user's zoom and pan are not reset every step), and
        the floating action bar is shown with the run's own actions. A preview
        still awaiting accept/reject on this same canvas is closed so this run
        can take over; a preview on another canvas is left alone.
        """
        pending = self._preview
        if pending is not None:
            if pending["doc"] is not doc:
                # A preview on another canvas is open; leave it and this frame.
                return
            # A run for the same canvas takes over: close the stale preview so
            # this run's frames (and result) are not blocked by it.
            self._clear_generation_preview()
        first = doc is not self._live_preview_doc
        if first:
            self._live_preview_doc = doc
        view = getattr(doc, "widget", None)
        if view is None or not view.winfo_exists():
            return
        index = self._canvas_index(doc)
        if first and index is not None and index != self.active_canvas:
            self._activate_canvas(index)
        origin = ((doc.width - image.width) // 2, (doc.height - image.height) // 2)
        try:
            # Never re-frame: the frame repaints in place so the zoom and pan the
            # user set are kept while the image forms.
            view.set_generation_preview(image, origin, fit=False)
        except tk.TclError:
            return
        if first:
            self._show_live_preview_bar()

    def _show_live_preview_bar(self):
        """Show the preview's floating action bar for a running generation.

        This is the same bar the finished preview uses, carrying the actions a
        still-running run can offer instead of the accept/reject ones.
        """
        self.area.show_preview_bar([
            ("Stop & Keep", self.stop_generate_keep),
            ("Cancel", self.cancel_generate),
        ])

    def _clear_live_preview(self):
        """Drop any live denoising-step frame (a run settled)."""
        doc = self._live_preview_doc
        self._live_preview_doc = None
        if doc is None:
            return
        view = getattr(doc, "widget", None)
        if view is None:
            return
        try:
            view.clear_generation_preview()
        except tk.TclError:
            pass
        self.area.hide_preview_bar()

    def _show_generation_preview(self, doc, result, prompt, settings=None):
        """Show a finished run in the canvas viewport for review.

        Nothing is written to the document until the user accepts the preview:
        the action bar offers rejecting it, or landing it in the selected layer,
        a new layer, or the canvas assets. The image is centred on the canvas at
        its own size so it lines up with the pixels it would be applied to.
        ``settings`` records the panel's fields, carried to the asset if the run
        is accepted into the canvas assets.
        """
        index = self._canvas_index(doc)
        view = getattr(doc, "widget", None)
        if index is None or view is None or not view.winfo_exists():
            # The canvas was replaced while the job ran; deliver as usual.
            self._apply_generated(
                doc, result, None, False, prompt, OUTPUT_ASSETS,
                settings=settings,
            )
            return
        image = result.convert("RGBA")
        origin = ((doc.width - image.width) // 2,
                  (doc.height - image.height) // 2)
        self._clear_generation_preview()
        self._preview = {
            "kind": "generation",
            "doc": doc,
            "image": image,
            "origin": origin,
            "prompt": prompt,
            "settings": settings,
        }
        if index != self.active_canvas:
            self._activate_canvas(index)
        # Draw at the current zoom and scroll rather than re-framing, so
        # generating does not throw away the user's view.
        view.set_generation_preview(image, origin, fit=False)
        self._refresh_preview_bar()
        self.report_zoom(view)
        self.statusbar.set_status("preview")
        self.set_result(
            f"Preview of {result.width} x {result.height} ready: "
            "accept or reject it."
        )

    def _clear_generation_preview(self):
        """Drop any pending preview (generation or asset) and its action bar."""
        preview = self._preview
        self._preview = None
        self.area.hide_preview_bar()
        if preview is None:
            return
        view = getattr(preview["doc"], "widget", None)
        if view is None:
            return
        try:
            view.clear_generation_preview()
        except tk.TclError:
            pass

    def _refresh_preview_bar(self):
        """Show the preview's action bar for the previewed canvas's layer."""
        preview = self._preview
        if preview is None:
            self.area.hide_preview_bar()
            return
        doc = preview["doc"]
        if doc is not self.current:
            # Another canvas is in front; its own page has no bar to show.
            self.area.hide_preview_bar()
            return
        layer = doc.active_layer()
        layer_label = (
            f"Add in '{layer.name}'" if layer is not None else "Add in layer"
        )
        if preview.get("kind") == "asset":
            self.area.show_preview_bar([
                ("Close", self.dismiss_asset_preview),
                (layer_label, self.accept_asset_preview_in_layer,
                 layer is not None),
                ("Insert as new layer", self.accept_asset_preview_as_layer),
            ])
            return
        self.area.show_preview_bar([
            ("Reject", self.reject_generation_preview),
            (layer_label, self.accept_generation_preview_in_layer,
             layer is not None),
            ("Add as new layer", self.accept_generation_preview_as_layer),
            ("Add in assets", self.accept_generation_preview_in_assets),
        ])

    def reject_generation_preview(self):
        """Discard the pending generation without changing the document."""
        if self._preview is None:
            return
        self._clear_generation_preview()
        self.statusbar.set_status("ready")
        self.set_result("Generation preview rejected.")

    def accept_generation_preview_in_layer(self):
        """Write the pending generation into the selected layer."""
        preview = self._preview
        if preview is None:
            return
        doc = preview["doc"]
        layer = doc.active_layer()
        if layer is None or layer.is_locked():
            self.set_result("Select an unlocked layer to add the preview into.")
            return
        self._deliver_generation_preview(OUTPUT_LAYER, layer)

    def accept_generation_preview_as_layer(self):
        """Write the pending generation onto a new layer at the top of the stack."""
        if self._preview is None:
            return
        self._deliver_generation_preview(
            OUTPUT_NEW_LAYER, self._preview["doc"].active_layer()
        )

    def accept_generation_preview_in_assets(self):
        """Keep the pending generation as a reference image on the canvas."""
        if self._preview is None:
            return
        self._deliver_generation_preview(OUTPUT_ASSETS, None)

    def _deliver_generation_preview(self, output, target_layer):
        """Apply the pending preview to ``output``, then drop the preview."""
        preview = self._preview
        if preview is None:
            return
        doc = preview["doc"]
        image = preview["image"]
        prompt = preview["prompt"]
        settings = preview.get("settings")
        self._clear_generation_preview()
        self._apply_generated(
            doc, image, None, False, prompt, output, target_layer,
            settings=settings,
        )

    # --------------------------------------------------------- asset preview
    def preview_asset(self, index):
        """Show an asset in the canvas viewport for review before inserting it.

        Nothing is written to the document until the user accepts the preview:
        the action bar offers closing it, or landing the asset in the selected
        layer or a new layer. The image is centred on the canvas at its own size
        so it lines up with the pixels it would be inserted against, letting the
        user check it before it goes onto the canvas.
        """
        if self.current is None:
            self.set_result("There is no canvas to preview an asset on.")
            return
        asset = self._asset_at(index)
        if asset is None:
            return
        doc = self.current
        view = getattr(doc, "widget", None)
        if view is None or not view.winfo_exists():
            return
        image = asset.image
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        origin = ((doc.width - image.width) // 2,
                  (doc.height - image.height) // 2)
        self._clear_generation_preview()
        self._preview = {
            "kind": "asset",
            "doc": doc,
            "image": image,
            "origin": origin,
            "asset": asset,
        }
        view.set_generation_preview(image, origin)
        self._refresh_preview_bar()
        self.report_zoom(view)
        self.statusbar.set_status("preview")
        self.set_result(
            f"Preview of '{asset.name}': insert it or close the preview."
        )

    def dismiss_asset_preview(self):
        """Close a pending asset preview without changing the document."""
        preview = self._preview
        if preview is None or preview.get("kind") != "asset":
            return
        self._clear_generation_preview()
        self.statusbar.set_status("ready")
        self.set_result("Asset preview closed.")

    def accept_asset_preview_in_layer(self):
        """Write the pending asset into the selected layer."""
        preview = self._preview
        if preview is None or preview.get("kind") != "asset":
            return
        doc = preview["doc"]
        layer = doc.active_layer()
        if layer is None or layer.is_locked():
            self.set_result("Select an unlocked layer to add the asset into.")
            return
        asset = preview["asset"]
        image = self._centered_asset_image(asset)
        self._clear_generation_preview()
        self._push_undo("Insert asset")
        layer.set_image(image)
        self._show_doc(doc)
        self.set_result(f"Added '{asset.name}' to '{layer.name}'.")

    def accept_asset_preview_as_layer(self):
        """Insert the pending asset as a new layer above the selected one."""
        preview = self._preview
        if preview is None or preview.get("kind") != "asset":
            return
        doc = preview["doc"]
        asset = preview["asset"]
        self._clear_generation_preview()
        if asset not in doc.assets:
            return
        self.insert_asset(doc.assets.index(asset))

    # ------------------------------------------------------------ segment (AI)
    def start_segment(self, x, y):
        """Select the object under ``(x, y)`` with a local SAM ONNX model.

        The mask is produced on the job queue (an image encoder then a point
        prompt), so the selection is installed by :meth:`_apply_segment` when
        the job finishes rather than here.
        """
        doc = self.current
        if doc is None:
            return
        tools = self.tools
        model = getattr(tools, "segment_model", None)
        if not model:
            self.set_result(
                "No segmentation model configured. Add a SAM ONNX "
                "'<name>_encoder.onnx' and '<name>_decoder.onnx' pair to the "
                "segmentation folder.",
                "warn",
            )
            return
        image = self._segment_source(doc, tools)
        if image is None:
            return

        def runner(progress, cancel):
            from engine.segmentation import segment
            return segment(
                image, [(x, y)], model_name=model, progress=progress, cancel=cancel
            )

        def finisher(mask, error, cancelled):
            self._apply_segment(doc, mask, error, cancelled)

        self.submit_job(
            "segment", "Segment object", runner, finisher, detail=model,
        )
        self.set_result("Segmenting object...")

    def _segment_source(self, doc, tools):
        """The document-space RGB image the Segment tool samples, or ``None``.

        The active layer is sampled over a transparent document-sized canvas so
        a click on it maps to document pixels; the composited canvas (the
        default) already is document-sized.
        """
        sample = getattr(tools, "segment_sample", "image")
        if sample == "layer":
            layer = self.active_layer()
            if layer is None:
                self._active_layer_locked()
                return None
            if layer.image.getbbox() is None:
                self.set_result("The active layer is empty.")
                return None
            canvas = Image.new("RGBA", doc.size, (0, 0, 0, 0))
            ox, oy = layer.offset
            canvas.paste(layer.composited_image(), (ox, oy))
            image = canvas
        else:
            image = doc.composite_transparent()
        if image.mode != "RGB":
            if image.mode == "RGBA":
                backdrop = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(backdrop, image).convert("RGB")
            else:
                image = image.convert("RGB")
        return image

    def cancel_segment(self):
        job = self.job_queue.running_or_last("segment")
        if job is not None:
            self.job_queue.cancel(job.id)

    def _apply_segment(self, doc, mask, error, cancelled):
        """Install a finished segmentation mask as the active selection."""
        if cancelled:
            self.statusbar.set_status("cancelled")
            self.set_result("Segmentation cancelled.")
            return
        if error is not None:
            self.statusbar.set_status("failed")
            self.set_result(error, "error")
            return
        if mask is None or self.current is not doc:
            return
        view = self.area.current_view()
        if view is None:
            return
        selection = Selection.from_mask(doc.size, mask)
        if selection.is_empty():
            self.set_result("Nothing to select there.")
            return
        view.set_selection(selection)
        box = selection.bbox
        self.set_result(f"Selected {box[2] - box[0]} x {box[3] - box[1]}.")

    # --------------------------------------------------------- remove background
    def start_remove_background(self):
        target = self.active_layer()
        if target is None:
            self._active_layer_locked()
            return
        if target.is_locked():
            self._active_layer_locked()
            return
        if target.image.getbbox() is None:
            self.set_result("The active layer is empty.")
            return
        model = self.bg.model()
        new_layer = self.bg.new_layer()
        doc = self.current
        source = target.image.copy()

        def runner(progress, cancel):
            from engine.background import remove_background
            return remove_background(
                source, model_name=model, progress=progress, cancel=cancel
            )

        def finisher(result, error, cancelled):
            self._apply_background(doc, target, result, error, cancelled, new_layer)

        self.submit_job(
            "background", "Remove background", runner, finisher, detail=model,
        )
        self.set_result("Background removal queued.")

    def cancel_remove_background(self):
        job = self.job_queue.running_or_last("background")
        if job is not None:
            self.job_queue.cancel(job.id)

    def _apply_background(self, doc, target, result, error, cancelled, new_layer):
        """Apply a finished cutout to the layer it was queued against."""
        if cancelled:
            self.statusbar.set_status("cancelled")
            self.set_result("Background removal cancelled.")
            return
        if error is not None:
            self.statusbar.set_status("failed")
            self.set_result(error, "error")
            return
        self._push_undo("Remove background")
        if new_layer:
            index = doc.layers.index(target) + 1 if target in doc.layers else len(doc.layers)
            layer = doc.add_layer(name=f"{target.name} (cutout)", index=index)
            layer.set_image(result)
            if doc is self.current:
                self.active = doc.layers.index(layer)
        else:
            target.set_image(result)
        self._show_doc(doc)
        self.statusbar.set_status("background removed")
        if new_layer:
            self.set_result(
                f"Removed background from '{target.name}' into a new layer."
            )
        else:
            self.set_result(f"Removed background from '{target.name}'.")

    # ------------------------------------------------------------------ upscale
    def start_upscale(self):
        target = self.active_layer()
        if target is None:
            self._active_layer_locked()
            return
        if target.is_locked():
            self._active_layer_locked()
            return
        if target.image.getbbox() is None:
            self.set_result("The active layer is empty.")
            return
        model = self.upscale.model()
        new_layer = self.upscale.new_layer()
        doc = self.current
        source = target.image.copy()

        def runner(progress, cancel):
            from engine.upscale import upscale
            return upscale(
                source, model_name=model, progress=progress, cancel=cancel
            )

        def finisher(result, error, cancelled):
            self._apply_upscale(doc, target, result, error, cancelled, new_layer)

        self.submit_job(
            "upscale", "Upscale", runner, finisher, detail=model,
        )
        self.set_result("Upscale queued.")

    def cancel_upscale(self):
        job = self.job_queue.running_or_last("upscale")
        if job is not None:
            self.job_queue.cancel(job.id)

    def _apply_upscale(self, doc, target, result, error, cancelled, new_layer):
        """Apply a finished upscale to the layer it was queued against."""
        if cancelled:
            self.statusbar.set_status("cancelled")
            self.set_result("Upscale cancelled.")
            return
        if error is not None:
            self.statusbar.set_status("failed")
            self.set_result(error, "error")
            return
        self._push_undo("Upscale")
        if new_layer:
            index = doc.layers.index(target) + 1 if target in doc.layers else len(doc.layers)
            layer = doc.add_layer(name=f"{target.name} (upscaled)", index=index)
            layer.set_image(result)
            if doc is self.current:
                self.active = doc.layers.index(layer)
        else:
            target.set_image(result)
        self._show_doc(doc)
        self.statusbar.set_status("upscaled")
        if new_layer:
            self.set_result(f"Upscaled '{target.name}' into a new layer.")
        else:
            self.set_result(f"Upscaled '{target.name}'.")


def _summarize(text, limit=40):
    """A one-line prompt snippet for the queue panel."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"
