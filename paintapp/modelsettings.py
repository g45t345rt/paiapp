"""The Model Settings dialog.

Points the engine at the user's own model files instead of a fixed ComfyUI
layout. The dialog edits the models root and per-kind folder paths (the
**Folders** tab), the curated model entries and their component pools; all of it
is stored in the ``models`` section of ``settings.json``.

On open the dialog reads the stored rows and builds itself at once, then warms
the kind-detection cache (:mod:`engine.component_kinds`) on a worker thread while
a strip above the buttons shows that pass's progress. Detection reads file
headers, so doing it during the row build froze the dialog before it appeared;
now the rows render from warm results only (see ``_kind``) and the two
kind-labelled tabs are rebuilt when the pass fills the cache.

The settings are laid out as tabs in a notebook:

* **Folders** — where new downloads land, and the engine's fallback when a model
  or component is not curated: the models root plus a per-kind override for
  Checkpoints, Diffusion models, Unet / GGUF, Text encoders, VAE, LoRAs, Upscale
  and Background models. Each field is pre-filled with the path in use and
  defaults to a subfolder of the models root, so every folder lives under the
  root until it is pointed elsewhere; folders are created on demand (a download
  makes its destination), never pre-created.
* **Models** — the picker's models, grouped by family, with per-row
  Label/Family/Path fields, Add / Up / Down / Duplicate / Remove controls and a
  component strip that assigns each model its Text encoder / VAE from the
  component pools. Every family also gets a LoRA list — any number of LoRAs, each
  with its own strength — applied to the model's diffusion transformer; a model
  has no LoRAs until one is added here. Duplicate (the layers panel's \u29c9
  glyph) copies a row (same file, components and LoRAs) under a new identity and
  a numbered label, so the same model can be listed more than once with different
  LoRA stacks. "(default)" leaves a component slot on the family default. "Scan
  folder…" browses a folder (and its subfolders, bounded) and appends the model
  files it finds; component files, which normally share a models root, are filed
  into their pools instead of becoming rows. "Browse Hugging Face…" searches the
  Hub and downloads a repo's files into a chosen folder (a picker-model folder
  becomes a new row, a component folder fills the matching pool). Each tab
  scrolls on its own.
  Each component slot gets a **Kind** picker beside its file picker. The Kind
  list holds every implementation that pool type can hold (CLIP-L, CLIP-G,
  T5-XXL, Qwen3-0.6B/4B, Qwen3-VL-8B; Qwen-Image / Flux VAEs), with the family's
  intended one marked "(recommended)" and preselected, and hides files whose
  detected kind does not match — so a slot shows only the handful of files that
  fit it. Files are labelled with their detected kind, the model row shows its
  own detected kind and format, and **Detect family** lines a row's group up with
  the file it points at (see :mod:`engine.component_kinds`).
* **Components** — Text encoders, VAEs, LoRAs, control adapters and Background
  models are editable lists (add / remove / reorder) of candidate files; each row
  shows the detected kind beside its path. "Scan folder…"
  browses a folder (and its subfolders, bounded) and appends its component
  files to the matching pool.
* **Recipes** — the saved per-model presets (steps / CFG / sampler / scheduler /
  denoise, plus the output width and height) the Generate panel offers, grouped
  by model. Add a recipe for any model — a new one starts from that model's
  family generation defaults — or edit and remove the existing ones; a recipe is
  picked from the Generate panel's dropdown, which also lists the read-only
  per-family recipes (see :func:`engine.paths.family_recipes`). A recipe is filed
  under its model row's identity (see :func:`engine.paths.entry_key`), so two
  rows over one file keep their own.

Group headers in the Models and Components tabs are clickable: clicking one
collapses or expands its rows. Groups start collapsed, and a group's Add button
is shown only while it is expanded. Each header also shows how many rows the
group holds (``\u25bc Anima (12)``), kept up to date as rows are added or
removed.

Saving writes the ``models`` section of ``settings.json`` (see
:mod:`engine.settings`) — the folder paths (``root`` and the per-kind
``*_dir`` overrides), ``components`` pools and ``entries`` (each carrying an
identity ``id``, its ``components`` assignment and its ``loras`` list) — plus the
``generation`` section's
recipes (see :func:`engine.paths.set_model_recipes`), reloads
:mod:`engine.paths`, and calls ``on_saved`` so open panels can rebuild their
model lists.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from engine import component_kinds as engine_kinds
from engine import paths as engine_paths
from engine import settings as engine_settings

from . import dialogs, hubdialog, modelwizard, theme, toolwizard
from .constants import MAX_CANVAS_SIZE
from .scrollbar import AutoScrollbar

_MODEL_FILETYPES = [
    ("Model files", "*.safetensors *.ckpt *.gguf *.onnx"),
    ("All files", "*.*"),
]

# Picker families a curated model row can belong to, shared with the engine so
# the Models tab and the Generate picker agree on grouping and order. A family is
# one generation pipeline: Stable Diffusion covers SD 1.5 and SDXL checkpoints
# (the arch is detected per file), Qwen-Image covers the 2.1 checkpoints.
_FAMILY_CHOICES = engine_paths.MODEL_GROUPS
_GROUP_ORDER = [group for group, _family in _FAMILY_CHOICES]
_GROUP_TO_FAMILY = {group: family for group, family in _FAMILY_CHOICES}

# Component pools, in display order: (title, settings key). Each is a curated
# list a model's components can be assigned from.
_POOL_CATEGORIES = (
    ("Text encoders", "text_encoders"),
    ("VAEs", "vaes"),
    ("LoRAs", "loras"),
    ("Adapters", "adapters"),
    ("ControlNet", "controlnet"),
    ("Background models", "background"),
    ("Upscale models", "upscale"),
)

# The model folders the Folders tab can point somewhere other than the models
# root, in display order: (label, settings key, default subfolder). A field left
# at its default follows the models root plus that subfolder.
_FOLDER_KINDS = (
    ("Checkpoints", "checkpoints_dir", "checkpoints"),
    ("Diffusion models", "diffusion_dir", "diffusion_models"),
    ("Unet / GGUF", "unet_dir", "unet"),
    ("Text encoders", "text_encoders_dir", "text_encoders"),
    ("VAE", "vae_dir", "vae"),
    ("LoRAs", "loras_dir", "loras"),
    ("Adapters", "adapters_dir", "adapters"),
    ("ControlNet", "controlnet_dir", "controlnet"),
    ("Upscale models", "upscale_dir", "upscale_models"),
    ("Background models", "background_dir", "background"),
    ("Segmentation models", "segmentation_dir", "segmentation"),
)

# The LoRA picker's "nothing selected" value. A LoRA slot has no family
# default (unlike a component slot), so it reads differently from "(default)".
_LORA_NONE = "(pick a LoRA)"

# The remove glyph, matching the layers panel's delete button (layers.ICON_DELETE).
_REMOVE_GLYPH = "\u2715"

# The duplicate glyph, matching the layers panel's duplicate button
# (layers.ICON_DUPLICATE).
_DUPLICATE_GLYPH = "\u29c9"

# Per-family component slots a model row can assign: (slot, label, pool key).
# Every family also gets a LoRA list (see ``_render_loras``); it is kept out of
# this table because it is a repeatable, strength-bearing list. The table lives
# in the engine so the download wizard and the Model Settings rows agree on what
# a family needs.
_FAMILY_SLOTS = engine_kinds.FAMILY_SLOTS

# The kind (see :mod:`engine.component_kinds`) each family's component slot is
# meant to hold. It seeds the slot's "Kind" dropdown and is marked
# "(recommended)" there, so a user who does not know which file to assign sees
# the intended implementation first. The mapping lives in the engine beside the
# detector, which also uses it to pick a slot's default from the component pools.
_SLOT_KIND = engine_kinds.SLOT_KIND

# A slot's "Kind" dropdown value that lists every pool file, unfiltered.
_ANY_KIND = "Any kind"

# Detected diffusion-model kind -> picker group, so the model row's "Detect
# family" button can line the file up with the pipeline that loads it.
_MODEL_KIND_GROUP = {
    "anima": "Anima",
    "qwen_image": "Qwen-Image",
    "flux": "Flux",
    "z_image": "Z-Image",
    "krea2": "Krea 2",
    "moss_image": "MOSS Image",
    "h3": "MiniMax H3",
    "h3_image": "MiniMax H3 Image",
    "fibo": "Fibo 2.0",
    "sd15": "Stable Diffusion",
    "sdxl": "Stable Diffusion",
    "stable_diffusion": "Stable Diffusion",
}

# Suffixes the "Scan folder…" button treats as picker models.
_SCAN_SUFFIXES = (".safetensors", ".ckpt", ".gguf")

# Recursion bounds shared by both "Scan folder…" buttons: descend at most this
# many subfolder levels below the browsed folder and stop after this many
# matched files, so a huge or misplaced folder cannot stall the dialog. Hidden
# files and folders are skipped. Nothing is written until Save, so a scan is
# always safe to redo.
_SCAN_MAX_DEPTH = 3
_SCAN_MAX_FILES = 2000

# Suffixes the component "Scan folder…" button treats as component files.
_COMPONENT_SUFFIXES = (
    ".safetensors", ".ckpt", ".gguf", ".onnx", ".pth", ".pt", ".bin",
)

# Known component folders (relative to the models root) -> pool name, so a scan
# of one of the engine's own folders files everything into the right pool.
_COMPONENT_DIRS = (
    ("text_encoders", ("text_encoders",)),
    ("vaes", ("vae",)),
    ("loras", ("loras",)),
    ("adapters", ("adapters",)),
    ("controlnet", ("controlnet",)),
    ("upscale", ("upscale_models",)),
    ("background", ("background",)),
)

# Pool -> its engine folder, as a ``component_path`` parts tuple. A model's
# component slot also offers the files sitting in the matching folder, not just
# the curated pool entries, so a file downloaded (or copied) into the folder is
# selectable without curating it first (see ``_pool_items``).
_POOL_FOLDERS = dict(_COMPONENT_DIRS)

# Folder basenames that name a component pool, for folders the engine does not
# know by path (a ComfyUI-style layout, a renamed dir, etc.).
_COMPONENT_FOLDER_NAMES = {
    "text_encoders": "text_encoders", "text_encoder": "text_encoders",
    "clip": "text_encoders", "clip_vision": "text_encoders",
    "vae": "vaes", "vaes": "vaes",
    "loras": "loras", "lora": "loras",
    "adapters": "adapters", "adapter": "adapters",
    "controlnet": "controlnet", "controlnets": "controlnet",
    "controlnet_models": "controlnet",
    "lllite": "controlnet", "controlnet_lllite": "controlnet",
    "background": "background", "background_removal": "background",
    "upscale": "upscale", "upscale_models": "upscale", "esrgan": "upscale",
}

# Filename tokens that suggest a component pool (checked in order, first wins).
# Control adapters come before LoRAs so a "control-lora" file is not pooled as a
# LoRA.
_COMPONENT_TOKENS = (
    ("vaes", ("vae",)),
    ("controlnet", ("controlnet", "control_v", "control-lora", "control_lora", "lllite")),
    ("adapters", ("anima_ref", "anima-ref", "ref_adapter", "ref-adapter")),
    ("loras", ("lora", "lycoris")),
    ("background", ("birefnet", "rmbg", "rembg", "isnet", "u2net", "modnet", "background")),
    ("upscale", ("esrgan", "swinir", "upscale", "realesr", "drct", "ldsr")),
    ("text_encoders", ("clip", "t5", "text_encoder", "llava", "umt5", "qwen", "gemma")),
)

# Recipes tab columns: (title, entry width). The Remove button follows them.
_RECIPE_COLUMNS = (
    ("Name", 18), ("Steps", 5), ("CFG", 5),
    ("Sampler", 11), ("Scheduler", 12), ("Denoise", 6),
    ("Width", 6), ("Height", 6),
)

# The vertical scrollbar's thickness (see ``paintapp.scrollbar``). Each tab
# keeps a fixed-width rail this wide, so the scrollbar appearing or going away
# as content shrinks or grows never changes the canvas width -- a width change
# re-lays out every row in the tab, which is what made collapsing a large group
# stutter.
_SCROLLBAR_WIDTH = 12

def _format_path(path, full):
    """A path as shown in Model Settings: the whole thing, or just its tail.

    Model Settings prints paths so the user can point the engine at their files,
    but a full path puts the user's home folder (``C:\\Users\\...``) on screen,
    which then shows up in screenshots and recordings. Unless full paths are
    asked for, the home folder is folded to ``~`` and only the last couple of
    segments are kept (see :func:`engine.paths.shorten_path`).
    """
    return engine_paths.shorten_path(path, full)


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _path_key(path):
    """A case- and separator-insensitive key for comparing file paths.

    The same file can reach the dialog spelled several ways -- a directory
    browse yields forward slashes, an ``os.walk`` the backslash form, and
    Windows keeps whatever case it was navigated with -- so de-duplicating on
    the raw string would re-add a file already listed. Normalising both sides
    the way the engine does (see :func:`engine.paths.list_controlnets`) keeps a
    rescan from duplicating an entry.
    """
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _collapse_marker(collapsed):
    """The header triangle shown before a group title."""
    return "\u25b6" if collapsed else "\u25bc"


def _index_of(seq, item):
    """Index of ``item`` in ``seq`` by identity, or ``-1``.

    The editable rows are plain dicts, so ``list.index``/``in`` would compare by
    value and could match the wrong row; identity is what the widgets hold.
    """
    for index, value in enumerate(seq):
        if value is item:
            return index
    return -1


class ModelSettingsDialog:
    """Modal editor for the ``models`` settings section."""

    def __init__(self, root, on_saved=None):
        self.root = root
        self.on_saved = on_saved
        self.window = tk.Toplevel(root)
        self.window.title("Model Settings")
        self.window.transient(root)
        self.window.withdraw()
        # Models and Components groups start collapsed; a group is only expanded
        # once the user clicks its header (and only then is its Add shown).
        self._expanded_pools = set()
        self._expanded_groups = set()
        self._entry_col_minsize = {}
        self._pool_col_minsize = {}
        self._pool_header = None
        self._pool_trail_reserved = False
        self._recipe_labels = []
        self._recipe_header = None
        self._recipe_group_frames = {}
        # The folder fields show a shortened path unless full paths are asked
        # for (see ``_format_path``), so the real value each one holds is kept
        # here while the widgets are bound to the display form.
        self._folder_full = {}

        # Read the stored rows, then build the dialog at once. Detection reads
        # file headers, so until the background pass finishes the rows render
        # from already-warm results only (see ``_kind``) and this pass's progress
        # shows in a strip above the buttons; the pools/entries tabs are rebuilt
        # from the filled cache when it completes. Building before showing keeps
        # the whole dialog on screen with no content swap to glimpse through.
        cfg = engine_settings.section("models")
        self._show_full_paths = bool(cfg.get("show_full_paths"))
        self._load_pools(cfg)
        self._load_entries(cfg)
        self._load_recipes()
        self._pending_kinds = self._pending_detections()
        self._kinds_ready = not self._pending_kinds

        self._build_dialog()
        self.window.bind("<Escape>", lambda _e: self.window.destroy())
        theme.restyle(self.window)
        dialogs.center_window(root, self.window)
        self.window.grab_set()
        if self._pending_kinds:
            self._start_detection()

    # ------------------------------------------------------------ loading pass
    def _pending_detections(self):
        """``[(path, category), ...]`` for every file the rows will detect.

        Detection reads a file's header, so it is done off the Tk thread; this is
        that work list, deduplicated by path (case folded) and category. Pools
        without a detection category (LoRAs, backgrounds, upscale) are skipped,
        matching :meth:`_pool_item_kind`, and so is a model row's component slot
        whose pool has no category.
        """
        tasks = []
        seen = set()

        def add(path, category):
            if not path or category is None:
                return
            path = os.fspath(path)
            key = (os.path.normcase(path), category)
            if key in seen:
                return
            seen.add(key)
            tasks.append((path, category))

        for _title, name in _POOL_CATEGORIES:
            category = engine_kinds.POOL_CATEGORY.get(name)
            for item in self._pool_items(name):
                add(item.get("path"), category)
        for entry in self.entries:
            add(entry.get("path"), engine_kinds.MODEL)
            for slot, path in (entry.get("components") or {}).items():
                pool = engine_kinds.SLOT_POOL.get(slot)
                add(path, engine_kinds.POOL_CATEGORY.get(pool))
        return tasks

    def _kind(self, path, category):
        """A file's detected kind, computed now or read from the warm cache.

        While the background pass runs (``_kinds_ready`` false) a miss is left
        undetected, so building the rows never reads a header on the Tk thread;
        the pass fills the cache and :meth:`_finish_detection` rebuilds the tabs.
        """
        if self._kinds_ready:
            return engine_kinds.detect(path, category)
        return engine_kinds.cached(path, category)

    def _build_progress(self):
        """The strip above the buttons that reports the detection pass.

        Its height is fixed and it is left in place (blanked) when the pass
        ends, so removing the progress never resizes the mapped window.
        """
        strip = tk.Frame(self.window, padx=12, height=18)
        strip.pack(fill=tk.X, pady=(6, 0))
        strip.pack_propagate(False)
        self._progress_detail = tk.Label(
            strip, text="Detecting model files\u2026", anchor="w",
        )
        theme.register(self._progress_detail, fg="muted")
        self._progress_detail.pack(side=tk.LEFT)
        self._progress_bar = ttk.Progressbar(
            strip, mode="determinate", maximum=100, length=200,
        )
        self._progress_bar.pack(side=tk.RIGHT)
        self._progress_strip = strip

    def _start_detection(self):
        """Warm the kind cache on a worker thread, reporting progress to Tk."""
        self._detect_queue = queue.Queue()
        self._detect_total = len(self._pending_kinds)
        pending = self._pending_kinds

        def work():
            # ``detect`` is pure file I/O and swallows its own errors, so it is
            # safe off the Tk thread; it populates the module cache the rebuild
            # then reads. Progress messages go through the queue only.
            for index, (path, category) in enumerate(pending, 1):
                engine_kinds.detect(path, category)
                self._detect_queue.put(("progress", index))
            self._detect_queue.put(("done", None))

        threading.Thread(
            target=work, name="pai-model-kinds", daemon=True,
        ).start()
        self._poll_detection()

    def _poll_detection(self):
        """Drain the worker's progress messages and rebuild when it finishes."""
        # A pending ``after`` survives the window being destroyed (the worker
        # thread keeps running and reporting), so bail out before touching any
        # widget that Cancel/Escape has already torn down.
        if not self._window_alive():
            return
        while True:
            try:
                message = self._detect_queue.get_nowait()
            except queue.Empty:
                break
            if message[0] == "done":
                self._finish_detection()
                return
            self._show_detect_progress(message[1])
        self.window.after(60, self._poll_detection)

    def _show_detect_progress(self, done):
        self._progress_detail.config(
            text=f"Detecting model files\u2026 {done} of {self._detect_total}"
        )
        self._progress_bar.config(value=min(1.0, done / self._detect_total) * 100)

    def _finish_detection(self):
        """Blank the progress strip and redraw the kind-labelled tabs."""
        self._kinds_ready = True
        self._progress_bar.pack_forget()
        self._progress_detail.config(text="")
        # Persist the warmed kinds so a later session opens with them already
        # known and this pass has nothing left to read.
        engine_kinds.save_cache()
        # The kinds are cached now, so rebuilding the two tabs labels the files,
        # narrows each slot's file list and reseeds its Kind picker from the
        # assigned file. Collapse state lives in ``_expanded_*`` and survives.
        self._rebuild_pools()
        self._rebuild_entries()

    def _window_alive(self):
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    # ----------------------------------------------------------------- layout
    def _build_dialog(self):
        body = tk.Frame(self.window)
        body.pack(fill=tk.BOTH, expand=True)

        self._build_header(body)

        notebook = ttk.Notebook(body)
        self.notebook = notebook
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        self._build_models_tab(notebook)
        self._build_components_tab(notebook)
        self._build_recipes_tab(notebook)
        self._build_folders_tab(notebook)

        if not self._kinds_ready:
            self._build_progress()

        buttons = tk.Frame(self.window, padx=12, pady=10)
        buttons.pack(fill=tk.X)
        tk.Button(buttons, text="Save", width=10, command=self._save).pack(side=tk.RIGHT)
        tk.Button(buttons, text="Cancel", width=10, command=self.window.destroy).pack(
            side=tk.RIGHT, padx=(0, 6)
        )
        tk.Button(buttons, text="Reset", width=10, command=self._reset).pack(side=tk.LEFT)
        # Paths are shortened by default so a screen recording does not show the
        # user's home folder; this reveals them whole for local use.
        self._show_full_var = tk.BooleanVar(value=self._show_full_paths)
        tk.Checkbutton(
            buttons, text="Show full paths", variable=self._show_full_var,
            command=self._on_toggle_full_paths, anchor="w",
        ).pack(side=tk.LEFT, padx=(12, 0))

    def _build_header(self, body):
        """The shared actions and help note above the tab container.

        Scanning and downloading act on the whole dialog, so they live outside
        the tabs; ``_scan_folder_action`` picks the model or component scan from
        the active tab.
        """
        actions = tk.Frame(body)
        actions.pack(fill=tk.X, padx=10, pady=(10, 0))
        tk.Button(
            actions, text="Scan folder\u2026", command=self._scan_folder_action,
        ).pack(side=tk.LEFT)
        tk.Button(
            actions, text="Browse Hugging Face\u2026", command=self._browse_hub,
        ).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(
            actions, text="Download model wizard\u2026", command=self._download_wizard,
        ).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(
            actions, text="Download tools\u2026", command=self._download_tools,
        ).pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(
            body, text="Adds model files from a folder (and subfolders) or the Hub.",
            anchor="w", fg=theme.color("muted"),
        ).pack(fill=tk.X, padx=10, pady=(4, 0))

        info = tk.Frame(body)
        info.pack(fill=tk.X, padx=10, pady=(6, 0))
        self._info_box(
            info,
            "Already have models downloaded (ComfyUI, or anywhere else)? You do "
            "not have to move them: use \u201cScan folder\u2026\u201d above, "
            "or add a model row and type its full path. "
        )

    def _scan_folder_action(self):
        """Scan from the shared bar: components on the Components tab, else models."""
        try:
            title = self.notebook.tab(self.notebook.select(), "text")
        except tk.TclError:
            title = ""
        if title == "Components":
            self._scan_component_folder()
        else:
            self._scan_folder()

    def _new_tab(self, notebook, title):
        """Add a scrollable page to ``notebook`` and target it for building.

        Each page scrolls on its own, so a long list (the picker) no longer
        pushes the whole dialog into one tall scroll. ``self.table``/``self._row``
        point at the new page's grid for the ``_group`` helper.
        """
        page = tk.Frame(notebook)
        notebook.add(page, text=title)

        # The scrollbar lives in a fixed-width rail (see ``_SCROLLBAR_WIDTH``)
        # so it can show or hide without resizing the canvas.
        rail = tk.Frame(page, width=_SCROLLBAR_WIDTH)
        rail.pack(side=tk.RIGHT, fill=tk.Y)
        rail.pack_propagate(False)

        # The canvas is only a scroll viewport (not content), so paint it with
        # the theme background instead of Tk's default white.
        canvas = tk.Canvas(
            page, highlightthickness=0, width=640, height=340, bg=theme.color("bg"),
        )
        scroll = AutoScrollbar(rail, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(canvas, padx=12, pady=10)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        # Match the inner frame to the canvas width so the table fills the
        # viewport (no unthemed canvas showing through on the right).
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window_id, width=e.width),
        )

        self.table = tk.Frame(inner)
        self.table.pack(fill=tk.X)
        self.table.columnconfigure(1, weight=1)
        self._row = 0
        return canvas, inner

    def _build_folders_tab(self, notebook):
        """The model-location editor: the models root plus per-kind overrides.

        Every field is pre-filled with the path the engine will use -- the
        models root, then each folder under it -- so the locations are visible
        and editable. Editing a field points just that folder elsewhere; a field
        left at its default follows the models root, so changing the root moves
        every folder that was not overridden. Folders are created on demand (a
        download makes its destination), never pre-created here.
        """
        canvas, inner = self._new_tab(notebook, "Folders")
        self._folders_canvas = canvas

        cfg = engine_settings.section("models")
        self._folder_parts = {key: part for _label, key, part in _FOLDER_KINDS}
        # A folder follows the root until it is given its own path; the flag
        # tracks that so a root change updates the ones still following. The
        # entry is bound to the (possibly shortened) display form, so the real
        # path each field holds lives in ``_folder_full``.
        self._folder_following = {}
        root_full = (
            os.fspath(cfg.get("root") or "") or engine_paths.default_models_root()
        )
        self._folder_full = {"root": root_full}
        self._folder_vars = {
            "root": tk.StringVar(value=self._display_path(root_full)),
        }
        for _label, key, _part in _FOLDER_KINDS:
            stored = os.fspath(cfg.get(key) or "")
            self._folder_following[key] = not stored
            self._folder_full[key] = stored or self._folder_default(key)
            self._folder_vars[key] = tk.StringVar(
                value=self._display_path(self._folder_full[key])
            )

        self._folder_row("Models root", "root")
        for label, key, _part in _FOLDER_KINDS:
            self._folder_row(label, key)

        for key in self._folder_vars:
            self._folder_vars[key].trace_add(
                "write", lambda *_a, k=key: self._on_folder_display(k),
            )

        self._bind_wheel(inner, canvas)

    def _info_box(self, frame, text):
        """A muted note inside ``frame``, with an info icon and theme info colors."""
        frame.configure(padx=8, pady=6, highlightthickness=1)
        theme.register(frame, bg="info_bg", highlightbackground="border")
        icon = tk.Label(frame, text="\u2139", anchor="nw")
        theme.register(icon, bg="info_bg", fg="info")
        icon.pack(side=tk.LEFT, anchor="n", padx=(0, 6))
        label = tk.Label(frame, text=text, anchor="w", justify="left", wraplength=540)
        theme.register(label, bg="info_bg", fg="info")
        label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _display_path(self, path):
        """A path as shown in the dialog (whole, or tail only -- see the toggle)."""
        return _format_path(path, self._show_full_paths)

    def _folder_default(self, key):
        """The path a folder field falls back to (root-relative by default)."""
        if key == "root":
            return engine_paths.default_models_root()
        root = (
            self._folder_full.get("root", "").strip()
            or engine_paths.default_models_root()
        )
        part = self._folder_parts[key]
        return os.path.join(root, part) if part else root

    def _on_folder_display(self, key):
        """Store a folder field's edit and re-link the folders that follow it.

        The entry is bound to the display form, so a value that matches the
        shortened path is the program's own re-render (a toggle or focus-out),
        not an edit, and is left alone.
        """
        value = self._folder_vars[key].get().strip()
        if value == self._display_path(self._folder_full.get(key, "")).strip():
            return
        self._folder_full[key] = value
        if key == "root":
            self._sync_folders()
        else:
            self._mark_folder(key)

    def _set_folder(self, key, value):
        """Point a folder field at ``value`` and redraw it in the display form."""
        self._folder_full[key] = value
        self._set_var(self._folder_vars[key], self._display_path(value))
        if key == "root":
            self._sync_folders()
        else:
            self._mark_folder(key)

    def _mark_folder(self, key):
        """Record whether a folder field still follows the models root."""
        value = self._folder_full.get(key, "").strip()
        default = self._folder_default(key)
        self._folder_following[key] = (
            not value or os.path.normpath(value) == os.path.normpath(default)
        )

    def _sync_folders(self):
        """Move every folder that still follows the root to its new default."""
        root = self._folder_full.get("root", "")
        for _label, key, part in _FOLDER_KINDS:
            if self._folder_following.get(key):
                default = os.path.join(root, part) if part else root
                self._folder_full[key] = default
                self._set_var(
                    self._folder_vars[key], self._display_path(default)
                )

    @staticmethod
    def _set_var(var, value):
        """Set a StringVar only when it actually changes, to avoid extra traces."""
        if var.get() != value:
            var.set(value)

    def _folder_row(self, label, key):
        """One folder field: a label, an entry and Browse / Default buttons."""
        table = self.table
        var = self._folder_vars[key]
        get_full = lambda k=key: self._folder_full.get(k, "")
        tk.Label(table, text=label, anchor="w").grid(
            row=self._row, column=0, sticky="w", pady=2,
        )
        entry = tk.Entry(table, textvariable=var)
        entry.grid(
            row=self._row, column=1, sticky="we", padx=(6, 0), pady=2,
        )
        self._bind_path_field(entry, var, get_full)
        buttons = tk.Frame(table)
        buttons.grid(row=self._row, column=2, sticky="w", padx=(6, 0), pady=2)
        tk.Button(
            buttons, text="Browse\u2026",
            command=lambda k=key: self._browse_folder(k),
        ).pack(side=tk.LEFT)
        tk.Button(
            buttons, text="Default", width=8,
            command=lambda k=key: self._set_folder(k, self._folder_default(k)),
        ).pack(side=tk.LEFT, padx=(4, 0))
        self._row += 1

    def _bind_path_field(self, widget, var, get_full):
        """Show a path field's full value while it is edited, the tail otherwise.

        Editing a shortened path would be useless, so focusing the field expands
        it to the real path and leaving it folds it back (unless full paths are
        being shown anyway).
        """
        widget.bind(
            "<FocusIn>",
            lambda _e, v=var, g=get_full: self._expand_path_field(v, g),
        )
        widget.bind(
            "<FocusOut>",
            lambda _e, v=var, g=get_full: self._set_var(v, self._display_path(g())),
        )

    def _expand_path_field(self, var, get_full):
        if not self._show_full_paths:
            self._set_var(var, get_full())

    def _build_components_tab(self, notebook):
        canvas, inner = self._new_tab(notebook, "Components")
        self._pools_canvas = canvas
        self._build_pools()
        self._bind_wheel(inner, canvas)

    def _build_models_tab(self, notebook):
        canvas, inner = self._new_tab(notebook, "Models")
        self._entries_canvas = canvas
        self._build_model_list()
        self._bind_wheel(inner, canvas)

    def _build_recipes_tab(self, notebook):
        """The per-model recipe editor: a flat table grouped by model."""
        canvas, inner = self._new_tab(notebook, "Recipes")
        self._recipes_canvas = canvas

        bar = tk.Frame(self.table)
        bar.grid(row=self._row, column=0, columnspan=4, sticky="we")
        self._row += 1
        tk.Label(bar, text="Model:").pack(side=tk.LEFT)
        self._recipe_model_choices = self._recipe_model_labels()
        self._recipe_model_var = tk.StringVar()
        self._recipe_model_combo = ttk.Combobox(
            bar, textvariable=self._recipe_model_var,
            values=list(self._recipe_model_choices), state="readonly", width=28,
        )
        self._recipe_model_combo.pack(side=tk.LEFT, padx=(4, 6))
        # The Models tab is edited in this same dialog, so the model list is
        # rebuilt from the current rows each time the dropdown opens.
        self._recipe_model_combo.configure(
            postcommand=self._refresh_recipe_model_choices,
        )
        if self._recipe_model_choices:
            self._recipe_model_var.set(next(iter(self._recipe_model_choices)))
        tk.Button(bar, text="Add recipe", command=self._add_recipe).pack(side=tk.LEFT)
        tk.Label(
            self.table, text="Recipes are picked per model in the Generate panel.",
            anchor="w", fg=theme.color("muted"),
        ).grid(row=self._row, column=0, columnspan=4, sticky="we", pady=(0, 2))
        self._row += 1

        self._recipes_body = tk.Frame(self.table)
        self._recipes_body.grid(
            row=self._row, column=0, columnspan=4, sticky="we", pady=(4, 0),
        )
        self._row += 1
        self._rebuild_recipes()
        self._bind_wheel(inner, canvas)

    def _load_recipes(self):
        """Load every stored recipe as a flat list of editable rows.

        A recipe is filed under its model's identity key (see
        :func:`engine.paths.entry_key`): a curated row's ``id``, else the file
        ``path``. An older recipe stored under a bare path is remapped onto the
        matching row's identity, so it follows the row rather than splitting off.
        """
        self.recipes = []
        by_path = {}
        for entry in self.entries:
            path = (entry.get("path") or "").strip()
            if path and path not in by_path:
                by_path[path] = engine_paths.entry_key(entry)
        for key, by_name in engine_paths.all_model_recipes().items():
            identity = by_path.get(key, key)
            path = self._path_for_key(identity)
            for name, values in by_name.items():
                self.recipes.append({
                    "key": identity, "path": path, "name": name,
                    "values": dict(values),
                })

    def _path_for_key(self, key):
        """The model file a preset/recipe ``key`` belongs to (``""`` unknown)."""
        for entry in self.entries:
            path = (entry.get("path") or "").strip()
            if path and engine_paths.entry_key(entry) == key:
                return path
        for _group, models in engine_paths.list_models():
            for _label, _family, path, _components, _loras, model_key in models:
                if model_key == key:
                    return path
        return ""

    def _recipe_catalog_models(self):
        """``[(group, label, key, path), ...]`` from the model rows, then the picker.

        The rows being edited in the Models tab come first, so the list follows
        them live (a new row appears, an edited label is what the recipe is filed
        under). ``key`` is the row's identity (its ``id``, else its ``path``), so
        two rows over one file are listed separately. The saved picker rows then
        add any model this dialog opened without — minus any path the user
        removed from the rows — and finally any key only a saved recipe knows
        about is kept.
        """
        items = []
        seen_keys = set()
        seen_paths = set()
        for entry in self.entries:
            path = (entry.get("path") or "").strip()
            if not path:
                continue
            key = engine_paths.entry_key(entry)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            seen_paths.add(path)
            label = (entry.get("label") or "").strip() or _stem(path)
            items.append((entry.get("group") or _GROUP_ORDER[0], label, key, path))
        removed = self._loaded_entry_paths - seen_paths
        for group, models in engine_paths.list_models():
            for label, _family, path, _components, _loras, key in models:
                if key in seen_keys or path in removed:
                    continue
                seen_keys.add(key)
                seen_paths.add(path)
                items.append((group, label, key, path))
        for entry in self.recipes:
            key = entry["key"]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            path = entry.get("path") or ""
            if path:
                group, _family = engine_paths.classify_model(path)
                items.append((group, _stem(path), key, path))
            else:
                items.append((_GROUP_ORDER[0], key, key, ""))
        return items

    def _recipe_model_labels(self):
        """Display label -> identity key for the "add recipe" dropdown."""
        labels = {}
        for group, label, key, _path in self._recipe_catalog_models():
            display = label if label not in labels else f"{group}: {label}"
            labels[display] = key
        return labels

    def _refresh_recipe_model_choices(self):
        """Re-list the current models in the "add recipe" dropdown (postcommand).

        Keeps the selection when that model is still there, otherwise falls back
        to the first model (or nothing, when the Models tab has no rows yet).
        """
        self._recipe_model_choices = self._recipe_model_labels()
        self._recipe_model_combo.configure(values=list(self._recipe_model_choices))
        if self._recipe_model_var.get() not in self._recipe_model_choices:
            self._recipe_model_var.set(next(iter(self._recipe_model_choices), ""))

    def _add_recipe(self):
        # Re-derive so Add works even if the dropdown was never opened.
        key = self._recipe_model_labels().get(self._recipe_model_var.get())
        if key:
            self._add_recipe_for(key)

    def _add_recipe_for(self, key):
        entry = {
            "key": key, "path": self._path_for_key(key), "name": "",
            "values": self._default_recipe_values(key),
        }
        self.recipes.append(entry)
        self._finish(self._new_recipe_row(entry), self._recipes_canvas)

    def _family_for_key(self, key):
        """The engine family a recipe's model belongs to, or ``None``."""
        for group, _label, model_key, _path in self._recipe_catalog_models():
            if model_key == key:
                return _GROUP_TO_FAMILY.get(group)
        return None

    def _default_recipe_values(self, key):
        """A new recipe's starting fields: the model's family generation defaults.

        Taken from the same read-only family recipe the Generate panel offers (see
        :func:`engine.paths.family_recipes`), so a recipe added here starts at the
        family default -- steps, CFG, sampler, scheduler, denoise, width and
        height -- instead of blank fields. ``{}`` when the family is unknown, so
        the row still gets its fields.
        """
        family = self._family_for_key(key)
        if not family:
            return {}
        recipes = engine_paths.family_recipes(family, self._path_for_key(key))
        return dict(next(iter(recipes.values())))

    def _remove_recipe(self, entry):
        index = _index_of(self.recipes, entry)
        if index < 0:
            return
        key = entry["key"]
        del self.recipes[index]
        row = entry.pop("_row", None)
        if row is not None:
            row.destroy()
        # A model with no recipes left has nothing to show, so its group goes
        # away, exactly as it did when the whole table used to be redrawn.
        info = self._recipe_group_frames.get(key)
        if info is not None and not info["rows"].winfo_children():
            info["holder"].destroy()
            del self._recipe_group_frames[key]

    def _rebuild_recipes(self):
        """Redraw the recipe table, grouped by model identity."""
        for child in self._recipes_body.winfo_children():
            child.destroy()
        self._recipe_group_frames = {}

        _header, labels = self._column_header(
            self._recipes_body, _RECIPE_COLUMNS, weight_column=len(_RECIPE_COLUMNS),
        )
        self._recipe_labels = labels
        self._recipe_header = _header

        catalog = {
            key: (label, path)
            for _group, label, key, path in self._recipe_catalog_models()
        }
        order = []
        for entry in self.recipes:
            if entry["key"] not in order:
                order.append(entry["key"])
        for key in order:
            self._ensure_recipe_group(key, catalog)
        for entry in self.recipes:
            self._new_recipe_row(entry)

        self._bind_wheel(self._recipes_body, self._recipes_canvas)
        theme.restyle(self._recipes_body)

    def _ensure_recipe_group(self, key, catalog=None):
        """The container for one model's recipe rows, created on first use.

        Each model sits in its own frame, so adding or removing a recipe touches
        that model alone instead of redrawing every row in the tab.
        """
        info = self._recipe_group_frames.get(key)
        if info is not None:
            return info
        if catalog is None:
            catalog = {
                model_key: (label, path)
                for _group, label, model_key, path in self._recipe_catalog_models()
            }
        holder = tk.Frame(self._recipes_body)
        holder.pack(fill=tk.X)
        header = tk.Frame(holder)
        theme.register(header, bg="header_bg")
        header.grid(row=0, column=0, sticky="we", pady=(8, 0))
        label, path = catalog.get(key, (key, ""))
        title = tk.Label(
            header, text=self._recipe_title(label, path),
            anchor="w", padx=6, pady=3,
        )
        theme.register(title, bg="header_bg", fg="header_fg")
        title.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            header, text="Add", width=5,
            command=lambda k=key: self._add_recipe_for(k),
        ).pack(side=tk.RIGHT, padx=6, pady=1)
        rows = tk.Frame(holder)
        rows.grid(row=1, column=0, sticky="we")
        holder.columnconfigure(0, weight=1)
        info = {
            "holder": holder, "header": header, "title": title, "rows": rows,
            "label": label, "path": path,
        }
        self._recipe_group_frames[key] = info
        self._finish(holder, self._recipes_canvas)
        return info

    def _recipe_title(self, label, path):
        """A recipe group's header text, with the path in its display form."""
        if not path:
            return label
        return f"{label}  ({self._display_path(path)})"

    def _refresh_recipe_titles(self):
        """Redraw the recipe group headers after the full-path toggle."""
        for info in self._recipe_group_frames.values():
            title = info.get("title")
            if title is not None and title.winfo_exists():
                title.configure(text=self._recipe_title(info["label"], info["path"]))

    def _new_recipe_row(self, entry):
        """Add one editable recipe row as its own frame, so Remove can drop it."""
        info = self._ensure_recipe_group(entry["key"])
        row = tk.Frame(info["rows"])
        row.pack(fill=tk.X)
        entry["_row"] = row
        self._recipe_row(row, entry)
        return row

    def _recipe_row(self, row, entry):
        row.columnconfigure(len(_RECIPE_COLUMNS), weight=1)
        name_var = tk.StringVar(value=entry.get("name", ""))
        name_var.trace_add(
            "write",
            lambda *_a, var=name_var, item=entry: item.__setitem__("name", var.get()),
        )
        tk.Entry(row, textvariable=name_var, width=18).grid(
            row=0, column=0, sticky="we", pady=1,
        )

        values = entry.setdefault("values", {})
        # The numeric fields match the Generate panel's own spinboxes, so a
        # recipe's Steps / CFG / Denoise / Width / Height read and adjust the same
        # way there. Width and Height are optional: leaving one blank lets the
        # recipe keep whatever size the Generate panel already has.
        for column, field, width, spec in (
            (1, "steps", 5, {"from_": 1, "to": 50}),
            (2, "cfg", 5, {"from_": 0.0, "to": 10.0, "increment": 0.1}),
            (5, "denoise", 6, {"from_": 0.0, "to": 1.0, "increment": 0.05}),
            (6, "width", 6, {"from_": 64, "to": MAX_CANVAS_SIZE}),
            (7, "height", 6, {"from_": 64, "to": MAX_CANVAS_SIZE}),
        ):
            current = values.get(field)
            var = tk.StringVar(value="" if current is None else str(current))
            var.trace_add(
                "write",
                lambda *_a, v=var, vals=values, f=field: self._set_recipe_number(
                    vals, f, v.get()
                ),
            )
            tk.Spinbox(row, textvariable=var, width=width, **spec).grid(
                row=0, column=column, sticky="w", padx=(4, 0), pady=1,
            )

        for column, field, choices, width in (
            (3, "sampler", engine_paths.SAMPLERS, 11),
            (4, "scheduler", engine_paths.SCHEDULERS, 12),
        ):
            var = tk.StringVar(value=str(values.get(field, "")))
            ttk.Combobox(
                row, textvariable=var, values=list(choices),
                state="readonly", width=width,
            ).grid(row=0, column=column, sticky="w", padx=(4, 0), pady=1)
            var.trace_add(
                "write",
                lambda *_a, v=var, vals=values, f=field: self._set_recipe_text(
                    vals, f, v.get()
                ),
            )

        tk.Button(
            row, text=_REMOVE_GLYPH, width=3,
            command=lambda item=entry: self._remove_recipe(item),
        ).grid(row=0, column=len(_RECIPE_COLUMNS), sticky="w", padx=(4, 0), pady=1)

        self._match_recipe_columns(row)

    def _match_recipe_columns(self, row):
        """Give a row's columns and the header's the same widths.

        The header and the rows sit in separate frames, so each titled column is
        pinned to the wider of its title and this row's cell (plus the grid pad)
        in both frames; a trailing untitled column takes the spare width on each
        side, leaving every title directly above its cells.
        """
        for column, label in enumerate(self._recipe_labels):
            cells = row.grid_slaves(row=0, column=column)
            padx = 0 if column == 0 else 4
            width = padx + max(
                label.winfo_reqwidth(),
                cells[0].winfo_reqwidth() if cells else 0,
            )
            row.columnconfigure(column, minsize=width)
            self._recipe_header.columnconfigure(column, minsize=width)

    def _set_recipe_number(self, values, field, text):
        """Store a parsed number for a recipe field, ignoring a partial edit."""
        try:
            value = float(text.strip())
        except ValueError:
            return
        values[field] = int(value) if field in ("steps", "width", "height") else value

    def _set_recipe_text(self, values, field, text):
        if text:
            values[field] = text
        else:
            values.pop(field, None)

    def _collect_recipes(self):
        """The edited recipes as a ``{key: {name: fields}}`` map for saving."""
        recipes = {}
        for entry in self.recipes:
            key = (entry.get("key") or "").strip()
            name = (entry.get("name") or "").strip()
            values = dict(entry.get("values") or {})
            if key and name and values:
                recipes.setdefault(key, {})[name] = values
        return recipes

    def _group_header(self, parent, row, columnspan, label, toggle, add_command):
        """A clickable group header: click the bar to collapse/expand its rows.

        ``label`` is the full header text (triangle, title and item count); an
        optional Add button sits beside the title and is packed by the caller
        (see :meth:`_show_group_add`) so it is shown only while the group is
        expanded. Returns ``(header, marker_label, add_button)`` so the caller can
        flip the triangle, update the count or toggle the button without
        rebuilding the group. ``add_button`` is ``None`` without an Add action.
        """
        header = tk.Frame(parent, cursor="hand2")
        theme.register(header, bg="header_bg")
        header.grid(
            row=row, column=0, columnspan=columnspan, sticky="we", pady=(8, 0),
        )
        name = tk.Label(
            header, text=label, anchor="w", padx=6, pady=3, cursor="hand2",
        )
        theme.register(name, bg="header_bg", fg="header_fg")
        name.pack(side=tk.LEFT)
        header.bind("<Button-1>", lambda _e: toggle())
        name.bind("<Button-1>", lambda _e: toggle())
        add_button = None
        if add_command is not None:
            add_button = tk.Button(header, text="Add", width=5, command=add_command)
        return header, name, add_button

    @staticmethod
    def _group_label(title, count, collapsed):
        """The header text: ``\u25bc Title (n)`` with the current row count."""
        return f"{_collapse_marker(collapsed)} {title} ({count})"

    def _column_header(self, parent, columns, weight_column=-1):
        """The fixed column-title row above a table.

        Built once per tab (not per rebuild) and returns the header frame plus
        its cell labels: the rows live in their own frames, so these labels (the
        widest widget in their column) define the minimum widths the row frames
        reserve to keep every column lined up. ``weight_column`` is the column
        that soaks up the spare width (the one the rows stretch), the last by
        default.
        """
        header = tk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 2))
        if columns:
            column = weight_column if weight_column >= 0 else len(columns) - 1
            header.columnconfigure(column, weight=1)
        labels = []
        for column, (title, width) in enumerate(columns):
            label = tk.Label(header, text=title, anchor="w")
            if width is not None:
                label.configure(width=width)
            label.grid(
                row=0, column=column, sticky="we", padx=(0 if column == 0 else 4, 0),
            )
            labels.append(label)
        return header, labels

    @staticmethod
    def _column_minsizes(labels, weight_column=-1):
        """Per-column minimum widths from the header, minus the weighted column.

        Every other column has to reserve at least what its header cell measures,
        or the rows would sit narrower than the titles above them.
        """
        column = weight_column if weight_column >= 0 else len(labels) - 1
        return {
            index: label.winfo_reqwidth()
            for index, label in enumerate(labels)
            if index != column
        }

    def _finish(self, widget, canvas):
        """Wire a freshly built subtree: wheel scrolling plus the active theme."""
        if widget is None:
            return
        self._bind_wheel(widget, canvas)
        theme.restyle(widget)

    def _show_group_add(self, info, show):
        """Pack or hide a group's Add button (only shown while expanded).

        Repacking is skipped when the button's visibility does not change: a
        needless ``pack``/``pack_forget`` alters the header's requested width and
        makes Tk re-lay out the whole tab on every toggle.
        """
        button = info.get("add")
        if button is None:
            return
        if info.get("add_shown", False) == show:
            return
        if show:
            button.pack(side=tk.RIGHT, padx=6, pady=1)
        else:
            button.pack_forget()
        info["add_shown"] = show

    def _apply_group_state(self, info, expanded_set, key, expanded):
        """Show or hide one group's rows and Add button for ``expanded``."""
        if expanded:
            expanded_set.add(key)
        else:
            expanded_set.discard(key)
        if info is None:
            return
        rows = info["rows"]
        if expanded:
            rows.grid()
        else:
            rows.grid_remove()
        self._show_group_add(info, expanded)
        title = info.get("title", key)
        info["marker"].configure(
            text=self._group_label(title, info["count"], not expanded)
        )

    def _toggle_pool(self, name):
        info = self._pool_group_frames.get(name)
        self._apply_group_state(
            info, self._expanded_pools, name, name not in self._expanded_pools
        )

    def _toggle_group(self, group):
        info = self._entry_group_frames.get(group)
        self._apply_group_state(
            info, self._expanded_groups, group, group not in self._expanded_groups
        )

    def _expand_pool(self, name):
        """Expand a pool so freshly added rows are visible."""
        self._apply_group_state(
            self._pool_group_frames.get(name), self._expanded_pools, name, True
        )

    def _expand_group(self, group):
        """Expand a family so freshly added rows are visible."""
        self._apply_group_state(
            self._entry_group_frames.get(group), self._expanded_groups, group, True
        )

    def _pool_count(self, name):
        return len(self.pools.get(name, []))

    def _entry_count(self, group):
        return sum(1 for entry in self.entries if entry["group"] == group)

    def _refresh_pool_count(self, name):
        """Update a pool header's ``(n)`` after its rows changed."""
        info = self._pool_group_frames.get(name)
        if info is None:
            return
        info["count"] = self._pool_count(name)
        info["marker"].configure(
            text=self._group_label(
                info["title"], info["count"], name not in self._expanded_pools
            )
        )

    def _refresh_entry_count(self, group):
        """Update a family header's ``(n)`` after its rows changed."""
        info = self._entry_group_frames.get(group)
        if info is None:
            return
        info["count"] = self._entry_count(group)
        info["marker"].configure(
            text=self._group_label(
                group, info["count"], group not in self._expanded_groups
            )
        )

    # --------------------------------------------------------- component pools
    def _load_pools(self, cfg):
        """Load the component pools from the settings ``components`` section."""
        section = cfg.get("components")
        section = section if isinstance(section, dict) else {}
        self.pools = {}
        for _title, name in _POOL_CATEGORIES:
            items = section.get(name)
            pool = []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("path"):
                        path = os.fspath(item["path"])
                        pool.append({
                            "label": item.get("label") or _stem(path), "path": path,
                        })
            self.pools[name] = pool
        # A pre-merge settings file kept Anima LLLite adapters in their own
        # "lllite" pool; fold them into "controlnet" (they are control adapters,
        # distinguished by their detected kind).
        legacy = section.get("lllite")
        if isinstance(legacy, list):
            known = {
                _path_key(item["path"]) for item in self.pools.get("controlnet", [])
            }
            for item in legacy:
                if not isinstance(item, dict) or not item.get("path"):
                    continue
                path = os.fspath(item["path"])
                key = _path_key(path)
                if key in known:
                    continue
                known.add(key)
                self.pools.setdefault("controlnet", []).append({
                    "label": item.get("label") or _stem(path), "path": path,
                })

    def _pool_items(self, name):
        """A pool's curated items plus files sitting in its engine folder.

        A file downloaded into the configured component folder is not always
        curated into the pool -- a Hub download whose additions were not saved,
        or a file copied in by hand. The slot file pickers still have to offer
        it, so this lists the folder's component files too (path-de-duplicated,
        with a curated label winning). Detection for these extra paths happens
        in the same background pass as the pool (see ``_pending_detections``).
        """
        items = list(self.pools.get(name, []))
        known = {
            _path_key(item["path"]) for item in items if item.get("path")
        }
        parts = _POOL_FOLDERS.get(name)
        if parts is None:
            return items
        folder = engine_paths.component_path(parts)
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            return items
        for entry in entries:
            if entry.startswith(".") or not entry.lower().endswith(
                _COMPONENT_SUFFIXES
            ):
                continue
            path = os.path.join(folder, entry)
            if not os.path.isfile(path):
                continue
            key = _path_key(path)
            if key in known:
                continue
            known.add(key)
            items.append({"label": _stem(path), "path": path})
        return items

    def _build_pools(self):
        """The grouped add/remove/reorder pool lists."""
        holder = tk.Frame(self.table)
        holder.grid(row=self._row, column=0, columnspan=4, sticky="we")
        self._row += 1

        self._pools_body = tk.Frame(holder)
        self._pools_body.pack(fill=tk.X, pady=(4, 0))
        self._pool_group_frames = {}
        self._rebuild_pools()

    def _ensure_pool_group(self, title, name):
        """The container for one pool's header and rows, created on first use.

        Keeping each pool in its own frame lets add/remove/reorder/collapse touch
        that pool alone instead of redrawing every row in the tab.
        """
        info = self._pool_group_frames.get(name)
        if info is not None:
            return info
        holder = tk.Frame(self._pools_body)
        holder.pack(fill=tk.X)
        collapsed = name not in self._expanded_pools
        count = self._pool_count(name)
        header, marker, add_button = self._group_header(
            holder, 0, 6, self._group_label(title, count, collapsed),
            toggle=lambda n=name: self._toggle_pool(n),
            add_command=lambda n=name: self._add_pool_item(n),
        )
        rows = tk.Frame(holder)
        rows.grid(row=1, column=0, sticky="we")
        holder.columnconfigure(0, weight=1)
        if collapsed:
            rows.grid_remove()
        info = {
            "holder": holder, "header": header, "marker": marker,
            "rows": rows, "title": title, "count": count, "add": add_button,
        }
        self._show_group_add(info, not collapsed)
        self._pool_group_frames[name] = info
        self._finish(holder, self._pools_canvas)
        return info

    def _rebuild_pools(self):
        for child in self._pools_body.winfo_children():
            child.destroy()
        self._pool_group_frames = {}

        _header, labels = self._column_header(
            self._pools_body, (("Label", 18), ("Path", None), ("Kind", 14)),
            weight_column=1,
        )
        self._pool_header = _header
        self._pool_trail_reserved = False
        self._pool_col_minsize = self._column_minsizes(labels, weight_column=1)
        for title, name in _POOL_CATEGORIES:
            self._ensure_pool_group(title, name)
            for item in self.pools.get(name, []):
                self._new_pool_row(name, item)

        self._bind_wheel(self._pools_body, self._pools_canvas)
        theme.restyle(self._pools_body)

    def _new_pool_row(self, name, item):
        """Add one editable pool row as its own frame, so Remove can drop it."""
        info = self._pool_group_frames.get(name)
        if info is None:
            return None
        row = tk.Frame(info["rows"])
        row.pack(fill=tk.X)
        item["_row"] = row
        self._pool_row(row, name, item)
        return row

    def _pool_row(self, row, name, item):
        for column, minsize in self._pool_col_minsize.items():
            row.columnconfigure(column, minsize=minsize)
        label_var = tk.StringVar(value=item.get("label", ""))
        label_var.trace_add(
            "write",
            lambda *_a, var=label_var, entry=item: entry.__setitem__("label", var.get()),
        )
        tk.Entry(row, textvariable=label_var, width=18).grid(
            row=0, column=0, sticky="we", pady=1,
        )
        path_var = tk.StringVar(value=self._display_path(item.get("path", "")))
        get_full = lambda it=item: it.get("path", "")
        path_var.trace_add(
            "write",
            lambda *_a, var=path_var, it=item, n=name: self._on_pool_path(var, it, n),
        )
        item["_path_var"] = path_var
        path_entry = tk.Entry(row, textvariable=path_var)
        path_entry.grid(
            row=0, column=1, sticky="we", padx=(4, 0), pady=1,
        )
        self._bind_path_field(path_entry, path_var, get_full)
        row.columnconfigure(1, weight=1)
        kind_label = tk.Label(row, anchor="w", fg=theme.color("muted"), width=14)
        kind_label.grid(row=0, column=2, sticky="w", padx=(4, 0), pady=1)
        item["_kind_label"] = kind_label
        self._refresh_pool_kind(name, item)
        browse = tk.Button(
            row, text="Browse\u2026",
            command=lambda var=path_var, g=get_full: self._browse(var, "file", g),
        )
        browse.grid(row=0, column=3, sticky="w", padx=(4, 0), pady=1)
        up = tk.Button(
            row, text="\u25b2", width=2,
            command=lambda i=item: self._move_pool_item(name, i, -1),
        )
        up.grid(row=0, column=4, sticky="w", padx=(4, 0), pady=1)
        down = tk.Button(
            row, text="\u25bc", width=2,
            command=lambda i=item: self._move_pool_item(name, i, 1),
        )
        down.grid(row=0, column=5, sticky="w", padx=(2, 0), pady=1)
        remove = tk.Button(
            row, text=_REMOVE_GLYPH, width=3,
            command=lambda i=item: self._remove_pool_item(name, i),
        )
        remove.grid(row=0, column=6, sticky="w", padx=(2, 0), pady=1)
        # The header titles stop at "Kind"; these button columns follow it, so
        # the header has to reserve their widths or "Kind" drifts right of the
        # labels beneath it.
        self._reserve_pool_trail(
            ((3, 4, browse), (4, 4, up), (5, 2, down), (6, 2, remove)),
        )

    def _reserve_pool_trail(self, columns):
        """Pad the pool header's trailing columns to match a row's buttons.

        ``columns`` pairs each button column with its leading padx and a widget
        to measure. Every row draws the same buttons, so the first row's sizes
        are reserved once on the header (which is laid out before any row). A
        grid column only takes up space when it holds a widget, so each button
        column gets an empty spacer frame sized to the button it stands in for.
        """
        if self._pool_trail_reserved or self._pool_header is None:
            return
        self._pool_trail_reserved = True
        for column, padx, widget in columns:
            spacer = tk.Frame(
                self._pool_header, width=widget.winfo_reqwidth(), height=1,
            )
            spacer.grid(row=0, column=column, sticky="w", padx=(padx, 0))

    def _refresh_pool_kind(self, name, item):
        """Show a pool row's detected kind beside its path."""
        label = item.get("_kind_label")
        if label is None:
            return
        kind_label = engine_kinds.kind_label(self._pool_item_kind(name, item))
        label.configure(text=kind_label or "\u2014")

    def _on_pool_path(self, var, item, name):
        """Store a pool item's path and refresh its detected kind label.

        The entry shows the shortened path, so a value that matches that
        rendering is the program's own re-render and leaves the stored path
        (and its detected kind) untouched.
        """
        value = var.get()
        if value == self._display_path(item.get("path", "")):
            return
        item["path"] = value
        self._refresh_pool_kind(name, item)

    def _relayout_pool_rows(self, name):
        """Re-pack one pool's rows in their stored order (after a move)."""
        info = self._pool_group_frames.get(name)
        if info is None:
            return
        rows = info["rows"]
        for child in rows.winfo_children():
            child.pack_forget()
        for item in self.pools.get(name, []):
            row = item.get("_row")
            if row is not None:
                row.pack(fill=tk.X)

    def _add_pool_item(self, name):
        item = {"label": "", "path": ""}
        self.pools.setdefault(name, []).append(item)
        row = self._new_pool_row(name, item)
        self._finish(row, self._pools_canvas)
        self._refresh_pool_count(name)

    def _remove_pool_item(self, name, item):
        pool = self.pools.get(name, [])
        index = _index_of(pool, item)
        if index < 0:
            return
        del pool[index]
        row = item.pop("_row", None)
        if row is not None:
            row.destroy()
        self._refresh_pool_count(name)

    def _move_pool_item(self, name, item, step):
        pool = self.pools[name]
        index = _index_of(pool, item)
        target = index + step
        if index < 0 or not 0 <= target < len(pool):
            return
        pool[index], pool[target] = pool[target], pool[index]
        self._relayout_pool_rows(name)

    @staticmethod
    def _pool_display(item):
        """How a pool item reads in a model row's dropdown.

        A manually added item starts with a blank label, so fall back to the
        file name; otherwise such items show up as blank choices.
        """
        return item.get("label") or _stem(item.get("path", "")) or "(unnamed)"

    def _pool_item_kind(self, name, item):
        """A pool item's detected kind (see :mod:`engine.component_kinds`)."""
        category = engine_kinds.POOL_CATEGORY.get(name)
        if category is None:
            return None
        return self._kind(item.get("path"), category)

    def _pool_item_display(self, name, item):
        """A pool item labelled with its kind, e.g. ``clip_l  (CLIP-L)``."""
        label = self._pool_display(item)
        kind_label = engine_kinds.kind_label(self._pool_item_kind(name, item))
        return f"{label}  ({kind_label})" if kind_label else label

    def _pool_path(self, name, label):
        for item in self.pools.get(name, []):
            if self._pool_display(item) == label:
                return item.get("path", "")
        return ""

    # ------------------------------------------------------------ slot helpers
    def _slot_kind_choices(self, name, recommended):
        """A slot's "Kind" values: every pool kind plus "Any kind".

        The family's intended kind is suffixed "(recommended)" so a user who
        does not know which file to assign sees it first.
        """
        values = [_ANY_KIND]
        category = engine_kinds.POOL_CATEGORY.get(name, "")
        for kind in engine_kinds.kinds(category):
            label = engine_kinds.kind_label(kind)
            values.append(f"{label} (recommended)" if kind == recommended else label)
        return values

    @staticmethod
    def _slot_kind_value(choices, kind):
        """The dropdown value matching a kind id (recommended suffix included)."""
        if not kind:
            return _ANY_KIND
        label = engine_kinds.kind_label(kind)
        for value in choices:
            if value == label or value == f"{label} (recommended)":
                return value
        return label

    def _slot_kind_from_display(self, name, display):
        """The kind id a "Kind" value names (``None`` for "Any kind")."""
        if not display or display == _ANY_KIND:
            return None
        wanted = (
            display[: -len(" (recommended)")]
            if display.endswith(" (recommended)") else display
        )
        category = engine_kinds.POOL_CATEGORY.get(name, "")
        for kind in engine_kinds.kinds(category):
            if engine_kinds.kind_label(kind) == wanted:
                return kind
        return None

    def _slot_file_choices(self, name, kind, assigned):
        """A slot's file values, filtered to ``kind`` and labelled with each kind.

        Returns ``(choices, mapping)`` where ``mapping`` resolves a dropdown
        value back to its file path. Files come from the pool plus the pool's
        engine folder (see ``_pool_items``), so a downloaded-but-uncurated file
        is still selectable. A file whose kind is unknown is listed only
        under "Any kind"; the assigned file is always kept so the row keeps
        showing what it stores even when the current kind filter hides it.

        While the background detection pass is still running no file has a kind
        yet, so the filter is skipped and every candidate file is listed; the
        rebuild when the pass ends applies it.
        """
        if not self._kinds_ready:
            kind = None
        choices = ["(default)"]
        mapping = {"(default)": ""}

        def add(item):
            display = self._pool_item_display(name, item)
            if display in mapping:
                return
            mapping[display] = item.get("path", "")
            choices.append(display)

        items = self._pool_items(name)
        for item in items:
            if kind and self._pool_item_kind(name, item) != kind:
                continue
            add(item)
        if assigned and assigned not in mapping.values():
            assigned_item = None
            for item in items:
                if item.get("path") == assigned:
                    assigned_item = item
                    break
            if assigned_item is None:
                # A manually entered path outside the pool still has to show
                # (and survive a Kind change), so list it by itself.
                display = self._slot_display_for(name, mapping, assigned)
                if display not in mapping:
                    mapping[display] = assigned
                    choices.append(display)
            elif not self._pool_item_kind(name, assigned_item):
                # A pool file whose kind is unknown cannot be judged against the
                # filter, so keep it listed rather than silently dropping it.
                add(assigned_item)
        return choices, mapping

    def _slot_display_for(self, name, mapping, path):
        """The dropdown value for a stored slot path (``"(default)"`` when none)."""
        if not path:
            return "(default)"
        for display, item_path in mapping.items():
            if item_path == path:
                return display
        kind_label = engine_kinds.kind_label(
            self._kind(path, engine_kinds.POOL_CATEGORY.get(name))
        )
        label = _stem(path)
        return f"{label}  ({kind_label})" if kind_label else label

    def _fill_slot_files(self, entry, slot, name):
        """Re-list one slot's files for the current Kind (``postcommand``).

        Runs as the dropdown opens, so a component added, renamed or removed in
        the Components tab — or a Kind change — is reflected without redrawing
        the Models tab.
        """
        widgets = entry.get("_slot_widgets", {}).get(slot)
        if widgets is None:
            return
        assigned = (entry.get("components") or {}).get(slot)
        kind = self._slot_kind_from_display(name, widgets["kind_var"].get())
        choices, mapping = self._slot_file_choices(name, kind, assigned)
        entry.setdefault("_slot_paths", {})[slot] = mapping
        widgets["file_combo"].configure(values=choices)

    def _on_slot_kind(self, entry, slot, name):
        """Re-list a slot's files for the picked Kind, clearing a mismatch."""
        widgets = entry.get("_slot_widgets", {}).get(slot)
        if widgets is None:
            return
        assigned = (entry.get("components") or {}).get(slot)
        kind = self._slot_kind_from_display(name, widgets["kind_var"].get())
        choices, mapping = self._slot_file_choices(name, kind, assigned)
        entry.setdefault("_slot_paths", {})[slot] = mapping
        widgets["file_combo"].configure(values=choices)
        if assigned and assigned not in mapping.values():
            # The stored file is not of this kind, so the slot falls back to the
            # family default rather than keeping a hidden mismatch; setting the
            # variable runs ``_assign`` through its trace.
            widgets["file_var"].set("(default)")
        else:
            widgets["file_var"].set(self._slot_display_for(name, mapping, assigned))

    def _assign(self, entry, slot, name, display):
        """Point a model's component slot at a pool item (or back to default)."""
        components = entry.setdefault("components", {})
        if display == "(default)":
            components.pop(slot, None)
            return
        mapping = entry.get("_slot_paths", {}).get(slot, {})
        path = mapping.get(display, self._pool_path(name, display))
        if path:
            components[slot] = path

    # ------------------------------------------------------------- model list
    def _load_entries(self, cfg):
        """Load the curated picker rows from the settings ``entries`` list."""
        self.entries = []
        # The paths the dialog opened with: a model the user removes from the
        # rows must not come back through the saved picker list.
        self._loaded_entry_paths = set()
        raw = cfg.get("entries")
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            path = os.fspath(item["path"])
            # Drop a support model an earlier build curated by mistake (a SAM
            # segmentation bundle read as an Anima model), so the dialog inherits
            # the same healed list as the picker and Save writes it back clean.
            if engine_paths.is_segmentation_model(path):
                continue
            group = item.get("group") or _GROUP_ORDER[0]
            components = item.get("components")
            loras = item.get("loras")
            self.entries.append({
                "id": str(item.get("id") or "") or engine_paths.new_entry_id(),
                "label": item.get("label") or _stem(path),
                "family": _GROUP_TO_FAMILY.get(group, item.get("family") or "anima"),
                "group": group,
                "path": path,
                "components": {
                    str(slot): os.fspath(value)
                    for slot, value in (components or {}).items()
                    if value
                } if isinstance(components, dict) else {},
                "loras": self._parse_loras(loras),
            })
        self._loaded_entry_paths = {entry["path"].strip() for entry in self.entries}

    def _build_model_list(self):
        """The grouped header/rows table for entries."""
        holder = tk.Frame(self.table)
        holder.grid(row=self._row, column=0, columnspan=4, sticky="we")
        self._row += 1

        self._entries_body = tk.Frame(holder)
        self._entries_body.pack(fill=tk.X, pady=(2, 0))
        self._entry_group_frames = {}
        self._rebuild_entries()

    def _entry_group_order(self):
        """Known families first, then any other group a row happens to carry."""
        groups = list(_GROUP_ORDER)
        for entry in self.entries:
            if entry["group"] not in groups:
                groups.append(entry["group"])
        return groups

    def _ensure_entry_group(self, group):
        """The container for one family's header and rows, created on first use.

        Each family lives in its own frame, so adding, removing, reordering or
        collapsing a row touches that family alone; the rest of the table is
        left untouched instead of being destroyed and rebuilt.
        """
        info = self._entry_group_frames.get(group)
        if info is not None:
            return info
        holder = tk.Frame(self._entries_body)
        holder.pack(fill=tk.X)
        collapsed = group not in self._expanded_groups
        count = self._entry_count(group)
        header, marker, add_button = self._group_header(
            holder, 0, 8, self._group_label(group, count, collapsed),
            toggle=lambda g=group: self._toggle_group(g),
            add_command=lambda g=group: self._add_entry(g),
        )
        rows = tk.Frame(holder)
        rows.grid(row=1, column=0, sticky="we")
        holder.columnconfigure(0, weight=1)
        if collapsed:
            rows.grid_remove()
        info = {
            "holder": holder, "header": header, "marker": marker,
            "rows": rows, "count": count, "add": add_button,
        }
        self._show_group_add(info, not collapsed)
        self._entry_group_frames[group] = info
        self._finish(holder, self._entries_canvas)
        return info

    def _rebuild_entries(self):
        """Redraw the curated rows, grouped by family with Add/Move/Remove."""
        for child in self._entries_body.winfo_children():
            child.destroy()
        self._entry_group_frames = {}

        _header, labels = self._column_header(
            self._entries_body, (("Label", 18), ("Family", 16), ("Path", None)),
        )
        self._entry_col_minsize = self._column_minsizes(labels)

        for group in self._entry_group_order():
            self._ensure_entry_group(group)
        for entry in self.entries:
            self._new_entry_row(entry)

        self._bind_wheel(self._entries_body, self._entries_canvas)
        theme.restyle(self._entries_body)

    def _new_entry_row(self, entry):
        """Add one editable model row as its own frame, so Remove can drop it."""
        info = self._ensure_entry_group(entry["group"])
        row = tk.Frame(info["rows"])
        row.pack(fill=tk.X)
        entry["_row"] = row
        self._entry_row(row, entry)
        return row

    def _relayout_entry_rows(self, group):
        """Re-pack one family's rows in their stored order (after a move)."""
        info = self._entry_group_frames.get(group)
        if info is None:
            return
        rows = info["rows"]
        for child in rows.winfo_children():
            child.pack_forget()
        for entry in self.entries:
            if entry["group"] != group:
                continue
            row = entry.get("_row")
            if row is not None:
                row.pack(fill=tk.X)

    def _entry_row(self, row, entry):
        for column, minsize in self._entry_col_minsize.items():
            row.columnconfigure(column, minsize=minsize)
        label_var = tk.StringVar(value=entry.get("label", ""))
        label_var.trace_add(
            "write",
            lambda *_a, var=label_var, item=entry: item.__setitem__("label", var.get()),
        )
        tk.Entry(row, textvariable=label_var, width=18).grid(
            row=0, column=0, sticky="we", pady=1,
        )

        group_var = tk.StringVar(value=entry.get("group", _GROUP_ORDER[0]))
        group_var.trace_add(
            "write",
            lambda *_a, var=group_var, item=entry: self._set_group(item, var.get()),
        )
        entry["_group_var"] = group_var
        ttk.Combobox(
            row, textvariable=group_var, values=_GROUP_ORDER,
            state="readonly", width=16,
        ).grid(row=0, column=1, sticky="we", padx=(4, 0), pady=1)

        path_var = tk.StringVar(value=self._display_path(entry.get("path", "")))
        get_full = lambda it=entry: it.get("path", "")
        path_var.trace_add(
            "write",
            lambda *_a, var=path_var, item=entry: self._on_entry_path(var, item),
        )
        entry["_path_var"] = path_var
        path_entry = tk.Entry(row, textvariable=path_var)
        path_entry.grid(
            row=0, column=2, sticky="we", padx=(4, 0), pady=1,
        )
        self._bind_path_field(path_entry, path_var, get_full)
        row.columnconfigure(2, weight=1)
        # Fibo 2.0's base slot is a diffusers-format directory (or a Hub repo
        # id), not a single weight file, so its Browse opens a folder picker.
        browse_kind = "dir" if entry.get("family") == "fibo" else "file"
        tk.Button(
            row, text="Browse\u2026",
            command=lambda var=path_var, g=get_full, k=browse_kind: self._browse(var, k, g),
        ).grid(row=0, column=3, sticky="w", padx=(4, 0), pady=1)
        tk.Button(
            row, text="\u25b2", width=2,
            command=lambda i=entry: self._move_entry(i, -1),
        ).grid(row=0, column=4, sticky="w", padx=(4, 0), pady=1)
        tk.Button(
            row, text="\u25bc", width=2,
            command=lambda i=entry: self._move_entry(i, 1),
        ).grid(row=0, column=5, sticky="w", padx=(2, 0), pady=1)
        tk.Button(
            row, text=_DUPLICATE_GLYPH, width=3,
            command=lambda i=entry: self._duplicate_entry(i),
        ).grid(row=0, column=6, sticky="w", padx=(2, 0), pady=1)
        tk.Button(
            row, text=_REMOVE_GLYPH, width=3,
            command=lambda i=entry: self._remove_entry(i),
        ).grid(row=0, column=7, sticky="w", padx=(2, 0), pady=1)

        slots = _FAMILY_SLOTS.get(entry["family"], ())

        strip = tk.Frame(row)
        strip.grid(row=1, column=0, columnspan=8, sticky="we", pady=(0, 3))
        entry.setdefault("components", {})
        # The row is built afresh (a family change rebuilds it), so the slot
        # bookkeeping is replaced rather than kept from the destroyed widgets.
        entry["_slot_paths"] = {}
        entry["_slot_widgets"] = {}
        if slots:
            for slot, title, name in slots:
                self._slot_row(strip, entry, slot, title, name)
        else:
            tk.Label(
                strip, text="This family has no separate components.", anchor="w",
                fg=theme.color("muted"),
            ).pack(side=tk.LEFT)

        kind_line = tk.Frame(row)
        kind_line.grid(row=2, column=0, columnspan=8, sticky="we", pady=(0, 3))
        model_kind = tk.Label(kind_line, anchor="w", fg=theme.color("muted"))
        model_kind.pack(side=tk.LEFT)
        entry["_model_kind_label"] = model_kind
        tk.Button(
            kind_line, text="Detect family",
            command=lambda item=entry: self._guess_family(item),
        ).pack(side=tk.LEFT, padx=(8, 0))
        self._refresh_model_kind(entry)

        entry["_lora_box"] = tk.Frame(row)
        entry["_lora_box"].grid(row=3, column=0, columnspan=8, sticky="we", pady=(0, 3))
        self._render_loras(entry)

        # A hairline under every row separates the several models a family can
        # hold; the line rides on the row, so reordering/removing keeps it right.
        divider = tk.Frame(row, height=1, bd=0, highlightthickness=0)
        divider.grid(row=4, column=0, columnspan=8, sticky="we", pady=(6, 0))
        theme.register(divider, bg="border")

    def _slot_row(self, strip, entry, slot, title, name):
        """One component slot: a Kind picker and a kind-filtered file picker.

        The Kind list is seeded with the family's intended implementation (see
        ``_SLOT_KIND``) and narrows the file list to files detected as that kind,
        so the user gets the handful of files that actually fit the slot instead
        of every file in the pool.
        """
        line = tk.Frame(strip)
        line.pack(fill=tk.X, pady=(0, 1))
        tk.Label(line, text=title + ":", anchor="w", width=13).pack(side=tk.LEFT)

        family = entry.get("family")
        recommended = _SLOT_KIND.get((family, slot))
        kind_var = tk.StringVar()
        kind_combo = ttk.Combobox(
            line, textvariable=kind_var, state="readonly", width=16,
            values=self._slot_kind_choices(name, recommended),
        )
        kind_combo.pack(side=tk.LEFT, padx=(0, 4))

        assigned = (entry.get("components") or {}).get(slot)
        choices, mapping = self._slot_file_choices(name, None, assigned)
        entry["_slot_paths"][slot] = mapping
        file_var = tk.StringVar()
        file_combo = ttk.Combobox(
            line, textvariable=file_var, state="readonly", width=34, values=choices,
        )
        file_combo.pack(side=tk.LEFT)

        # Seed the Kind from the assigned file, else the family's recommended one.
        assigned_kind = (
            self._pool_item_kind(name, {"path": assigned}) if assigned else None
        )
        kind_var.set(self._slot_kind_value(kind_combo["values"], assigned_kind or recommended))
        file_var.set(self._slot_display_for(name, mapping, assigned))

        # The pool can change while this row is on screen (the Components tab
        # sits beside it), so both lists are rebuilt as needed instead of being
        # frozen at build time.
        file_combo.configure(
            postcommand=lambda i=entry, s=slot, n=name: self._fill_slot_files(i, s, n),
        )
        kind_var.trace_add(
            "write",
            lambda *_a, i=entry, s=slot, n=name: self._on_slot_kind(i, s, n),
        )
        file_var.trace_add(
            "write",
            lambda *_a, i=entry, s=slot, n=name: self._assign(i, s, n, file_var.get()),
        )

        entry["_slot_widgets"][slot] = {
            "kind_var": kind_var, "kind_combo": kind_combo,
            "file_var": file_var, "file_combo": file_combo,
        }
        # Narrow the file list to the seeded Kind (the assigned file stays listed).
        self._fill_slot_files(entry, slot, name)

    def _on_entry_path(self, var, entry):
        """Store a row's path and refresh its detected model-kind label.

        One callback (not two traces) so the label always reads the path that
        was just set, whatever order Tk runs traces in. The entry shows the
        shortened path, so a value matching that rendering is a re-render, not
        an edit, and leaves the stored path alone.
        """
        value = var.get()
        if value == self._display_path(entry.get("path", "")):
            return
        entry["path"] = value
        self._refresh_model_kind(entry)

    def _refresh_model_kind(self, entry):
        """Update a row's detected model-kind label (from its current path)."""
        label = entry.get("_model_kind_label")
        if label is None:
            return
        path = (entry.get("path") or "").strip()
        if not path:
            label.configure(text="Model: (no file)")
            return
        described = engine_kinds.describe_kind(path, self._kind(path, engine_kinds.MODEL))
        label.configure(text=f"Model: {described or 'unknown'}")

    def _guess_family(self, entry):
        """Point a row at the group its file's detected kind belongs to."""
        path = (entry.get("path") or "").strip()
        if not path:
            return
        group = _MODEL_KIND_GROUP.get(self._kind(path, engine_kinds.MODEL))
        group_var = entry.get("_group_var")
        if group and group_var is not None:
            group_var.set(group)

    # -------------------------------------------------------------- LoRA list
    @staticmethod
    def _parse_loras(value):
        """A saved ``loras`` list as ``[{"path", "strength"}]`` (empty if unset)."""
        loras = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("path"):
                    loras.append({
                        "path": os.fspath(item["path"]),
                        "strength": float(item.get("strength", 1.0) or 1.0),
                    })
        return loras

    def _render_loras(self, entry):
        """Draw a model row's LoRA list, one LoRA per line.

        Every family's diffusion model can take LoRAs, so the section is always
        shown. A model has no LoRAs until one is picked here; the "+ LoRA"
        button appends a blank row.
        """
        box = entry.get("_lora_box")
        if box is None:
            return
        for child in box.winfo_children():
            child.destroy()
        loras = entry.setdefault("loras", [])
        header = tk.Frame(box)
        header.pack(fill=tk.X)
        tk.Label(header, text="LoRA:", anchor="w").pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            header, text="+ LoRA", command=lambda item=entry: self._add_lora(item),
        ).pack(side=tk.LEFT)
        if not loras:
            tk.Label(
                header, text="(none)", anchor="w", fg=theme.color("muted"),
            ).pack(side=tk.LEFT, padx=(6, 0))
        for index, lora in enumerate(loras):
            self._lora_row(box, entry, index, lora)

    def _lora_row(self, box, entry, index, lora):
        """One LoRA line: picker, strength spinbox and remove button."""
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=(12, 0), pady=(1, 0))
        var = tk.StringVar(value=self._lora_label(lora.get("path")))
        combo = ttk.Combobox(
            row, textvariable=var, state="readonly", width=26,
            values=self._lora_choices(),
        )
        combo.pack(side=tk.LEFT, padx=(0, 4))
        combo.configure(
            postcommand=lambda c=combo, e=entry, i=index: self._fill_lora_choices(
                c, e, i
            ),
        )
        var.trace_add(
            "write",
            lambda *_a, v=var, item=lora, i=index: self._assign_lora(item, i, v.get()),
        )
        tk.Label(row, text="Strength:", anchor="w").pack(side=tk.LEFT, padx=(0, 2))
        strength = tk.StringVar(value=f"{float(lora.get('strength', 1.0)):g}")
        spin = tk.Spinbox(
            row, from_=0.0, to=4.0, increment=0.05, width=5,
            textvariable=strength,
        )
        spin.pack(side=tk.LEFT, padx=(0, 6))
        strength.trace_add(
            "write",
            lambda *_a, v=strength, item=lora: self._set_lora_strength(item, v.get()),
        )
        tk.Button(
            row, text=_REMOVE_GLYPH, width=3,
            command=lambda i=index: self._remove_lora(entry, i),
        ).pack(side=tk.LEFT)

    def _lora_choices(self):
        """The LoRA dropdown values: the placeholder plus the pool's items."""
        return [_LORA_NONE] + [
            self._pool_display(item) for item in self.pools.get("loras", [])
        ]

    def _lora_label(self, path):
        """A LoRA entry's dropdown label (the file, or the placeholder)."""
        if not path:
            return _LORA_NONE
        for item in self.pools.get("loras", []):
            if item.get("path") == path:
                return self._pool_display(item)
        return _stem(path)

    def _fill_lora_choices(self, combo, entry, index):
        """Refresh a LoRA picker's list from the pool as it opens (postcommand)."""
        combo.configure(values=self._lora_choices())
        loras = entry.get("loras") or []
        if index >= len(loras):
            return
        assigned = loras[index].get("path")
        if not assigned:
            return
        desired = self._lora_label(assigned)
        if desired == combo.get():
            return
        # Only rewrite the picker when the label still resolves to the same
        # pool file (a rename), or the item is gone and the slot must clear.
        if desired == _LORA_NONE or self._pool_path("loras", desired) == assigned:
            combo.set(desired)

    def _assign_lora(self, lora, index, label):
        """Point one LoRA entry at a pool item (or clear it on the placeholder)."""
        if label == _LORA_NONE:
            lora["path"] = ""
        else:
            lora["path"] = self._pool_path("loras", label)

    def _set_lora_strength(self, lora, text):
        try:
            lora["strength"] = float(text)
        except (TypeError, ValueError):
            pass

    def _add_lora(self, entry):
        entry.setdefault("loras", []).append({"path": "", "strength": 1.0})
        self._render_loras(entry)
        self._finish(entry.get("_lora_box"), self._entries_canvas)

    def _remove_lora(self, entry, index):
        loras = entry.setdefault("loras", [])
        if 0 <= index < len(loras):
            del loras[index]
        self._render_loras(entry)
        self._finish(entry.get("_lora_box"), self._entries_canvas)

    def _remake_entry_row(self, entry):
        """Rebuild a single row after its family (and so its slots) changed."""
        old = entry.pop("_row", None)
        if old is not None:
            old.destroy()
        row = self._new_entry_row(entry)
        self._finish(row, self._entries_canvas)

    def _set_group(self, entry, group):
        """Update an entry's family when its group dropdown changes."""
        previous = entry.get("group")
        if previous == group:
            return
        entry["group"] = group
        entry["family"] = _GROUP_TO_FAMILY.get(group, "anima")
        allowed = {slot for slot, _title, _name in _FAMILY_SLOTS.get(entry["family"], ())}
        entry["components"] = {
            slot: value
            for slot, value in (entry.get("components") or {}).items()
            if slot in allowed
        }
        # The row has already changed group, so both headers' counts are stale;
        # a not-yet-built group is counted when its frame is created.
        self._refresh_entry_count(previous)
        self._refresh_entry_count(group)
        # The group change moves the row under a different header and swaps its
        # component strip, so rebuild that one row; defer it so the combobox
        # event finishes before its widget goes away.
        self._entries_body.after_idle(lambda item=entry: self._remake_entry_row(item))

    def _add_entry(self, group):
        entry = {
            "id": engine_paths.new_entry_id(),
            "label": "", "family": _GROUP_TO_FAMILY.get(group, "anima"),
            "group": group, "path": "", "components": {},
            "loras": [],
        }
        self.entries.append(entry)
        row = self._new_entry_row(entry)
        self._finish(row, self._entries_canvas)
        self._refresh_entry_count(group)

    def _duplicate_entry(self, entry):
        """Copy a model row -- same file, components and LoRA stack -- with a new
        identity.

        The copy gets a fresh ``id``, so it keeps its own presets and recipes even
        though it points at the same model file, and a label numbered apart from
        the row it was copied from. It is placed right after that row.
        """
        index = _index_of(self.entries, entry)
        if index < 0:
            return
        group = entry.get("group") or _GROUP_ORDER[0]
        copy = {
            "id": engine_paths.new_entry_id(),
            "label": self._duplicate_label(entry.get("label")),
            "family": entry.get("family") or _GROUP_TO_FAMILY.get(group, "anima"),
            "group": group,
            "path": entry.get("path") or "",
            "components": dict(entry.get("components") or {}),
            "loras": [dict(lora) for lora in (entry.get("loras") or [])],
        }
        self.entries.insert(index + 1, copy)
        row = self._new_entry_row(copy)
        self._finish(row, self._entries_canvas)
        self._relayout_entry_rows(group)
        self._refresh_entry_count(group)

    def _duplicate_label(self, label):
        """A label for a copy: the source's, numbered `` (n)``.

        The number continues from the source's own suffix (``Foo (1)`` becomes
        ``Foo (2)``) and skips values already in use, so every row reads apart.
        """
        base = (label or "").strip() or "model"
        stem, number = base, 0
        if base.endswith(")"):
            head, sep, tail = base.rpartition(" (")
            if sep and head and tail[:-1].isdigit():
                stem, number = head, int(tail[:-1])
        existing = {(item.get("label") or "").strip() for item in self.entries}
        number += 1
        while f"{stem} ({number})" in existing:
            number += 1
        return f"{stem} ({number})"

    def _remove_entry(self, entry):
        index = _index_of(self.entries, entry)
        if index < 0:
            return
        group = entry["group"]
        del self.entries[index]
        row = entry.pop("_row", None)
        if row is not None:
            row.destroy()
        self._refresh_entry_count(group)

    def _move_entry(self, entry, step):
        """Swap ``entry`` with the neighbouring entry in the same group."""
        group = entry["group"]
        members = [item for item in self.entries if item["group"] == group]
        position = _index_of(members, entry)
        target = position + step
        if position < 0 or not 0 <= target < len(members):
            return
        other = members[target]
        first, second = _index_of(self.entries, entry), _index_of(self.entries, other)
        self.entries[first], self.entries[second] = (
            self.entries[second], self.entries[first],
        )
        self._relayout_entry_rows(group)

    def _scan_folder(self):
        """Browse a folder and add its model files as curated rows.

        Component files under the folder (text encoders, VAEs, LoRAs, control
        adapters and upscale/background models) are not picker models, so they
        are filed into their pools on the Components tab instead. When that
        leaves nothing to add, say so rather than appearing to do nothing.
        """
        folder = filedialog.askdirectory(
            parent=self.window, initialdir=engine_paths.models_root() or None,
        )
        if not folder:
            return
        candidates, skipped = self._scan_models(folder)
        known = {_path_key(entry["path"]) for entry in self.entries}
        added = []
        for path, group, family in candidates:
            key = _path_key(path)
            if key in known:
                continue
            entry = {
                "id": engine_paths.new_entry_id(),
                "label": _stem(path), "family": family, "group": group, "path": path,
                "loras": [],
            }
            self.entries.append(entry)
            added.append(entry)
            known.add(key)
        for entry in added:
            self._finish(self._new_entry_row(entry), self._entries_canvas)
        for group in {entry["group"] for entry in added}:
            self._refresh_entry_count(group)
            self._expand_group(group)
        filed = self._file_components(folder)
        if not added and not filed and skipped:
            dialogs.show_info(
                self.window, "Scan folder",
                f"Skipped {skipped} component file(s) that could not be filed "
                "into a pool. Add them by hand on the Components tab.",
            )

    def _file_components(self, folder):
        """File a folder's component files into the pools; returns how many."""
        added, _found, new_items = self._scan_components(folder, None)
        for name, item in new_items:
            self._finish(self._new_pool_row(name, item), self._pools_canvas)
        for name in {name for name, _item in new_items}:
            self._refresh_pool_count(name)
            self._expand_pool(name)
        return added


    def _scan_models(self, folder):
        """Classify the model files under ``folder`` into ``(path, group, family)``.

        A checkpoints folder (or file) becomes Stable Diffusion rows; elsewhere
        ``.gguf`` and qwen-named files go to Qwen-Image and other safetensors to
        Anima. Subfolders are scanned up to ``_SCAN_MAX_DEPTH`` deep, so rows
        stay editable and a wrong guess is one dropdown away from fixed.

        Component files (text encoders, VAEs, LoRAs, control adapters and upscale
        and background models) are left out: a models root normally holds both,
        and those files belong in the Components tab's pools. Returns
        ``(rows, skipped)``.
        """
        checkpoints = os.path.normpath(engine_paths.checkpoints_dir())
        found = []
        skipped = 0
        for path, _name in self._iter_scan_files(folder, _SCAN_SUFFIXES):
            if engine_paths.is_segmentation_model(path):
                # A SAM segmentation file is a support model, not a picker model.
                continue
            if self._component_pool_of(path):
                skipped += 1
                continue
            in_checkpoints = self._under_dir(path, checkpoints)
            found.append((path, *self._classify(path, in_checkpoints)))
        return found, skipped

    def _iter_scan_files(self, folder, suffixes):
        """Yield ``(path, name)`` for matching files under ``folder``, bounded.

        Recurses into subfolders up to ``_SCAN_MAX_DEPTH`` levels below
        ``folder`` and stops after ``_SCAN_MAX_FILES`` matches. Hidden files and
        directories are skipped; directory symlinks are not followed, so a scan
        cannot loop.
        """
        root = os.path.abspath(folder)
        count = 0
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if dirpath == root:
                depth = 0
            else:
                depth = os.path.relpath(dirpath, root).count(os.sep) + 1
            if depth >= _SCAN_MAX_DEPTH:
                dirnames[:] = []
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                if name.startswith(".") or not name.lower().endswith(suffixes):
                    continue
                yield os.path.join(dirpath, name), name
                count += 1
                if count >= _SCAN_MAX_FILES:
                    return

    def _under_dir(self, path, folder):
        """Whether ``path`` is ``folder`` itself or lives under it."""
        if not folder:
            return False
        path = os.path.normpath(path)
        folder = os.path.normpath(folder)
        return path == folder or path.startswith(folder + os.sep)

    def _classify(self, path, in_checkpoints):
        return engine_paths.classify_model(path, in_checkpoints)

    def _scan_component_folder(self):
        """Browse a folder and append its component files to the pools."""
        folder = filedialog.askdirectory(
            parent=self.window, initialdir=engine_paths.models_root() or None,
        )
        if not folder:
            return
        added, found, new_items = self._scan_components(folder, None)
        for name, item in new_items:
            self._finish(self._new_pool_row(name, item), self._pools_canvas)
        for name in {name for name, _item in new_items}:
            self._refresh_pool_count(name)
            self._expand_pool(name)
        if not added and found:
            dialogs.show_info(
                self.window, "Scan folder",
                "No component files matched in that folder.",
            )

    def _scan_components(self, folder, forced_pool):
        """Append the component files under ``folder`` to the matching pools.

        Returns ``(added, found, new_items)``: how many files were added or
        retargeted onto an existing row, how many candidate files were seen, and
        the ``(pool, item)`` pairs appended so the caller can draw only those
        rows. Subfolders are scanned up to ``_SCAN_MAX_DEPTH`` deep. A file goes
        to the pool named by the folder it sits in (a known component
        folder, or a folder named for a pool), else to the pool its filename
        suggests. A wrong guess is one Remove away from fixed, like the model
        scan.
        """
        known = {
            _path_key(item.get("path"))
            for pool in self.pools.values() for item in pool
        }
        # A curated model row's file is not a component: the picker already lists
        # it, so filing it into a pool would show the same model twice.
        known |= {_path_key(entry.get("path")) for entry in self.entries}
        # Model folders hold diffusion models, not components, so a file sitting
        # in one is skipped unless a pool is forced.
        model_dirs = [
            engine_paths.checkpoints_dir(),
            engine_paths.diffusion_dir(),
            engine_paths.unet_dir(),
        ]
        # A file already curated can be listed under a path it has since left
        # (the pool was filled before the file moved -- e.g. the Anima LLLite
        # folder becoming ``controlnet``). Match such an entry by name so a
        # rescan points it at the file's new home instead of adding a second row
        # for the same file.
        stale = {}
        for pool_name, pool in self.pools.items():
            for item in pool:
                old = item.get("path")
                if old and not os.path.exists(old):
                    stale.setdefault(
                        (pool_name, os.path.basename(old).lower()), item
                    )
        added = found = 0
        new_items = []
        for path, name in self._iter_scan_files(folder, _COMPONENT_SUFFIXES):
            found += 1
            key = _path_key(path)
            if key in known:
                continue
            if engine_paths.is_segmentation_model(path):
                # A SAM segmentation file belongs to no component pool.
                continue
            dir_path = os.path.dirname(path)
            if not forced_pool and any(
                self._under_dir(dir_path, d) for d in model_dirs if d
            ):
                continue
            dir_pool = self._folder_component_pool(dir_path)
            pool_name = forced_pool or dir_pool or self._component_pool_for(name)
            if not pool_name:
                continue
            moved = stale.get((pool_name, name.lower()))
            if moved is not None:
                self._retarget_pool_item(moved, path)
                known.add(key)
                added += 1
                continue
            item = {"label": _stem(path), "path": path}
            self.pools.setdefault(pool_name, []).append(item)
            new_items.append((pool_name, item))
            known.add(key)
            added += 1
        return added, found, new_items

    def _retarget_pool_item(self, item, path):
        """Point an existing pool row at a moved file, keeping one row per file."""
        var = item.get("_path_var")
        if var is not None:
            # Fires the row's trace, which stores the path and refreshes its kind,
            # then fold the entry back to its (possibly shortened) display form.
            var.set(path)
            self._set_var(var, self._display_path(path))
        else:
            item["path"] = path

    def _folder_component_pool(self, folder):
        """The pool a known component folder belongs to (``""`` if unknown)."""
        for pool_name, parts in _COMPONENT_DIRS:
            path = engine_paths.component_path(parts)
            if path and self._under_dir(folder, os.path.normpath(path)):
                return pool_name
        return _COMPONENT_FOLDER_NAMES.get(os.path.basename(folder).lower(), "")

    def _component_pool_for(self, name):
        """Guess the pool for a component file from its name (``""`` if none)."""
        lower = name.lower()
        for pool_name, tokens in _COMPONENT_TOKENS:
            if any(token in lower for token in tokens):
                return pool_name
        return ""

    def _component_pool_of(self, path):
        """The pool a scan candidate belongs to, or ``""`` for a real model.

        Used to keep the Models tab's scan from filing component files (text
        encoders, VAEs, LoRAs, control adapters and upscale and background
        models) as picker models: a file is a component when it sits in a known
        component folder — a configured one, or a folder named for a pool — or
        when its name carries an unambiguous component token. The text-encoder
        name tokens are ignored here, because a picker model's name can
        legitimately read like "qwen" or "t5"; those files are recognised by
        their folder instead.
        """
        pool = self._folder_component_pool(os.path.dirname(path))
        if pool:
            return pool
        guessed = self._component_pool_for(os.path.basename(path))
        return "" if guessed == "text_encoders" else guessed

    def _browse_hub(self):
        """Search Hugging Face and file any downloaded files into the tables."""
        hubdialog.HubBrowserDialog(self.window, on_downloaded=self._on_hub_downloaded)

    def _download_wizard(self):
        """Download a model together with the VAE/text encoders it needs."""
        modelwizard.ModelDownloadWizard(
            self.window, on_ready=self._on_wizard_ready,
        )

    def _download_tools(self):
        """Download a background, upscale or segmentation model into its folder."""
        toolwizard.ToolDownloadWizard(
            self.window, on_ready=self._on_tool_wizard_ready,
        )

    def _on_tool_wizard_ready(self, payload):
        """Curate a completed tool download into its component pool.

        Unlike a generation download there is no model row: background and
        upscale models are selected by pool label alone, so the files simply
        join the matching pool. Segmentation exports are support models that
        :mod:`engine.segmentation` lists from their folder, so they are skipped
        like :meth:`_on_hub_downloaded` does. Nothing is saved here; Save still
        commits the tables, matching the other download paths.
        """
        changed = set()
        for entry in payload.get("files") or ():
            path = os.fspath(entry.get("path") or "")
            pool_name = entry.get("pool")
            if not path or not pool_name:
                continue
            if engine_paths.is_segmentation_model(path):
                continue
            pool = self.pools.setdefault(pool_name, [])
            key = _path_key(path)
            if any(_path_key(item.get("path")) == key for item in pool):
                continue
            item = {"label": entry.get("label") or _stem(path), "path": path}
            pool.append(item)
            self._finish(self._new_pool_row(pool_name, item), self._pools_canvas)
            changed.add(pool_name)
        for name in changed:
            self._refresh_pool_count(name)
            self._expand_pool(name)

    def _on_wizard_ready(self, payload):
        """File a completed wizard download into pools and one model row.

        Unlike :meth:`_on_hub_downloaded`, which only sees paths, the wizard
        knows each file's role, so the new row is created with its component
        slots already assigned. Nothing is saved here; Save still commits the
        tables, matching the other download paths.
        """
        path = os.fspath(payload.get("path") or "")
        if not path:
            return
        components = {
            slot: os.fspath(value)
            for slot, value in (payload.get("components") or {}).items()
            if value
        }
        pool_for = {
            "clip": "text_encoders", "t5": "text_encoders",
            "text": "text_encoders", "vae": "vaes",
        }
        for slot, value in components.items():
            pool_name = pool_for.get(slot)
            if not pool_name:
                continue
            pool = self.pools.setdefault(pool_name, [])
            key = _path_key(value)
            if any(_path_key(item.get("path")) == key for item in pool):
                continue
            item = {"label": _stem(value), "path": value}
            pool.append(item)
            self._finish(self._new_pool_row(pool_name, item), self._pools_canvas)
            self._refresh_pool_count(pool_name)
            self._expand_pool(pool_name)

        group = payload.get("group") or _GROUP_ORDER[0]
        family = payload.get("family") or _GROUP_TO_FAMILY.get(group, "anima")
        label = payload.get("label") or _stem(path)
        key = _path_key(path)
        entry = next(
            (item for item in self.entries
             if _path_key(item.get("path")) == key),
            None,
        )
        if entry is None:
            entry = {
                "id": engine_paths.new_entry_id(),
                "label": label, "family": family, "group": group,
                "path": path, "components": dict(components), "loras": [],
            }
            self.entries.append(entry)
            row = self._new_entry_row(entry)
            self._finish(row, self._entries_canvas)
        else:
            entry.update({
                "label": label, "family": family, "group": group,
                "components": dict(components),
            })
            self._remake_entry_row(entry)
        self._refresh_entry_count(group)
        self._expand_group(group)

    def _folder_role(self, folder):
        """Which known model folder ``folder`` is (``""`` for an unknown one)."""
        checks = (
            ("checkpoints", engine_paths.checkpoints_dir()),
            ("diffusion", engine_paths.diffusion_dir()),
            ("unet", engine_paths.unet_dir()),
            ("text_encoders", engine_paths.text_encoders_dir()),
            ("vae", engine_paths.vae_dir()),
            ("loras", engine_paths.loras_dir()),
            ("adapters", engine_paths.adapters_dir()),
            ("controlnet", engine_paths.controlnet_dir()),
        )
        for role, path in checks:
            if path and os.path.normpath(path) == folder:
                return role
        return ""

    def _on_hub_downloaded(self, paths):
        """Add downloaded files to the model rows or the component pools.

        The folder a file landed in decides where it goes: a picker-model folder
        becomes a model row (family guessed from the name, editable as usual),
        while a text-encoder/VAE/LoRA/control-adapter folder is appended to the
        matching pool. A file whose name marks it as a component is pooled too,
        so a download into a plain folder does not become a picker model.
        """
        pool_for = {
            "text_encoders": "text_encoders", "vae": "vaes", "loras": "loras",
            "adapters": "adapters",
            "controlnet": "controlnet",
        }
        known = {_path_key(entry.get("path")) for entry in self.entries}
        new_pools = []
        new_entries = []
        for path in paths:
            if engine_paths.is_segmentation_model(path):
                # A download into the segmentation folder is a support model:
                # engine.segmentation lists it itself, so it is not curated here.
                continue
            key = _path_key(path)
            role = self._folder_role(os.path.normpath(os.path.dirname(path)))
            pool_name = (
                pool_for[role] if role in pool_for else self._component_pool_of(path)
            )
            if pool_name:
                pool = self.pools.setdefault(pool_name, [])
                if not any(_path_key(item.get("path")) == key for item in pool):
                    item = {"label": _stem(path), "path": path}
                    pool.append(item)
                    new_pools.append((pool_name, item))
                continue
            if key in known:
                continue
            group, family = engine_paths.classify_model(path, role == "checkpoints")
            entry = {
                "id": engine_paths.new_entry_id(),
                "label": _stem(path), "family": family, "group": group,
                "path": path, "components": {},
                "loras": [],
            }
            self.entries.append(entry)
            new_entries.append(entry)
            known.add(key)
        for name, item in new_pools:
            self._finish(self._new_pool_row(name, item), self._pools_canvas)
        for entry in new_entries:
            self._finish(self._new_entry_row(entry), self._entries_canvas)
        for name in {name for name, _item in new_pools}:
            self._refresh_pool_count(name)
            self._expand_pool(name)
        for group in {entry["group"] for entry in new_entries}:
            self._refresh_entry_count(group)
            self._expand_group(group)

    def _bind_wheel(self, widget, canvas):
        """Scroll ``canvas`` from anywhere over ``widget`` or its descendants.

        Tk delivers ``<MouseWheel>`` to the widget under the pointer, which here
        is an Entry/Label/Button inside the canvas window rather than the canvas
        itself, so a binding on the canvas alone never fires. Bind every
        descendant to the same handler, tied to its own tab's canvas.
        """
        widget.bind("<MouseWheel>", lambda _e, c=canvas: self._on_wheel(_e, c))
        for child in widget.winfo_children():
            self._bind_wheel(child, canvas)

    def _on_wheel(self, event, canvas):
        canvas.yview_scroll(int(-event.delta / 120) or 0, "units")
        return "break"

    # ---------------------------------------------------------------- actions
    def _browse(self, var, kind, get_full=None):
        """Pick a model file for a path field, then fold the field back down.

        ``get_full`` reads the field's real path so a shortened entry still
        starts the dialog in the right folder and stores the picked path whole.
        """
        current = (get_full() if get_full is not None else var.get()) or ""
        if kind == "dir" and os.path.isdir(current):
            initial = current
        else:
            initial = os.path.dirname(current) or engine_paths.models_root()
        if kind == "dir":
            path = filedialog.askdirectory(parent=self.window, initialdir=initial)
        else:
            path = filedialog.askopenfilename(
                parent=self.window, initialdir=initial, filetypes=_MODEL_FILETYPES,
            )
        if path:
            path = os.path.normpath(path)
            # Setting the full path fires the field's trace (which stores it),
            # then the entry is redrawn in the shortened display form.
            var.set(path)
            self._set_var(var, self._display_path(path))

    def _browse_folder(self, key):
        """Browse for a folder field, keeping its real path in ``_folder_full``."""
        current = self._folder_full.get(key) or self._folder_default(key)
        path = filedialog.askdirectory(parent=self.window, initialdir=current)
        if path:
            self._set_folder(key, os.path.normpath(path))

    def _render_path_fields(self):
        """Redraw every path field in its current display form (after a toggle)."""
        for key, var in self._folder_vars.items():
            self._set_var(var, self._display_path(self._folder_full.get(key, "")))
        for items in self.pools.values():
            for item in items:
                var = item.get("_path_var")
                if var is not None:
                    self._set_var(
                        var, self._display_path(item.get("path", ""))
                    )
        for entry in self.entries:
            var = entry.get("_path_var")
            if var is not None:
                self._set_var(var, self._display_path(entry.get("path", "")))

    def _on_toggle_full_paths(self):
        """Switch between shortened and full paths for every field and header."""
        self._show_full_paths = self._show_full_var.get()
        self._render_path_fields()
        self._refresh_recipe_titles()

    def _reset(self):
        self._show_full_paths = False
        self._show_full_var.set(False)
        self.pools = {name: [] for _title, name in _POOL_CATEGORIES}
        self._rebuild_pools()
        self.entries = []
        self._rebuild_entries()
        self.recipes = []
        self._rebuild_recipes()
        for key in self._folder_vars:
            self._set_folder(key, self._folder_default(key))

    def _collect_entries(self):
        """The curated rows as settings dicts, dropping rows without a path."""
        collected = []
        for entry in self.entries:
            path = (entry.get("path") or "").strip()
            if not path:
                continue
            group = entry.get("group") or _GROUP_ORDER[0]
            entry["id"] = entry.get("id") or engine_paths.new_entry_id()
            item = {
                "id": entry["id"],
                "label": (entry.get("label") or "").strip() or _stem(path),
                "family": _GROUP_TO_FAMILY.get(group, entry.get("family") or "anima"),
                "group": group,
                "path": path,
            }
            components = {
                slot: value
                for slot, value in (entry.get("components") or {}).items()
                if value
            }
            if components:
                item["components"] = components
            loras = [
                {
                    "path": lora["path"],
                    "strength": float(lora.get("strength", 1.0) or 1.0),
                }
                for lora in (entry.get("loras") or [])
                if lora.get("path")
            ]
            # Save the list only when it holds something; a model with no
            # LoRAs needs no stored key.
            if loras:
                item["loras"] = loras
            collected.append(item)
        return collected

    def _collect_pools(self):
        """The component pools as a settings section, dropping empty pools."""
        section = {}
        for _title, name in _POOL_CATEGORIES:
            items = []
            for item in self.pools.get(name, []):
                path = (item.get("path") or "").strip()
                if not path:
                    continue
                items.append({
                    "label": (item.get("label") or "").strip() or _stem(path),
                    "path": path,
                })
            if items:
                section[name] = items
        return section

    def _save(self):
        # Start from the existing section so legacy keys this dialog does not
        # own (the per-family fallbacks) are preserved, then replace everything
        # this dialog owns: the folder paths, the component pools and the
        # curated entries.
        models = dict(engine_settings.section("models"))
        for key in ("entries", "components", "lllite_dir"):
            models.pop(key, None)

        # Store a folder path only when it is a real override; a field left at
        # its default (the models root or a subfolder of it) is dropped, so the
        # folder keeps following the root. The fields show a shortened path, so
        # the real value is read from ``_folder_full`` (and a hand-typed ``~`` is
        # expanded).
        root_value = os.path.expanduser(self._folder_full.get("root", "")).strip()
        default_root = engine_paths.default_models_root()
        if root_value and os.path.normpath(root_value) != os.path.normpath(default_root):
            models["root"] = os.path.normpath(root_value)
        else:
            models.pop("root", None)

        root_effective = root_value or default_root
        for _label, key, part in _FOLDER_KINDS:
            value = os.path.expanduser(self._folder_full.get(key, "")).strip()
            default = os.path.join(root_effective, part) if part else root_effective
            if value and os.path.normpath(value) != os.path.normpath(default):
                models[key] = os.path.normpath(value)
            else:
                models.pop(key, None)

        # A display preference, not an engine setting: keep it beside the model
        # rows so it is remembered without leaking into the engine's own paths.
        if self._show_full_paths:
            models["show_full_paths"] = True
        else:
            models.pop("show_full_paths", None)

        pools = self._collect_pools()
        if pools:
            models["components"] = pools
        entries = self._collect_entries()
        if entries:
            models["entries"] = entries

        engine_settings.set_section("models", models)
        engine_paths.set_model_recipes(self._collect_recipes())
        engine_paths.reload()
        self.window.destroy()
        if self.on_saved is not None:
            self.on_saved()
