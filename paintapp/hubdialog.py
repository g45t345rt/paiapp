"""Browse Hugging Face and download models into the configured folders.

Opened from the Model Settings "Models" tab, this dialog searches the public
Hugging Face Hub, lists the files in a chosen repo, and streams the selected
files into one of the configured model folders. When a download finishes the
caller is handed the saved paths (see ``on_downloaded``) so the new files can be
filed into the model list or the component pools without a rescan.

A repo often names its weights generically (``model.safetensors``), which is
hard to tell apart later and collides across repos. Each file therefore has a
"Save as" name: generic names are rewritten from the repo id by default (the
"Rename generic names" check), edited per file with "Rename...", and made unique
across the listed files, so the column shows exactly what a download will write.

Three small windows do the work:

* :class:`HubBrowserDialog` - the search box, sort/filter controls and results
  table.
* :class:`HubFileDialog` - the files inside one repo with a destination picker.
* :class:`_RenameDialog` - edit the on-disk name for the selected files.
* :class:`_DownloadWindow` - a modal progress bar that runs the transfers.

Network and disk work always happens on a worker thread; each window reports
back to Tk through a small ``after`` poller, so nothing blocks the UI.
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

from engine import hub as engine_hub
from engine import paths as engine_paths
from engine import settings as engine_settings
from engine.cancel import CancelToken, Cancelled

from . import theme
from .scrollbar import AutoScrollbar

# How many results one search asks for. The Hub caps this server-side.
_LIMIT = 100

# The folders a download can land in, as ``(label, path_fn)`` pairs. ``path_fn``
# is called when needed so a just-changed models root is honored.
def _folder_targets():
    return [
        ("Checkpoints", engine_paths.checkpoints_dir),
        ("Diffusion models", engine_paths.diffusion_dir),
        ("Unet / GGUF", engine_paths.unet_dir),
        ("Text encoders", engine_paths.text_encoders_dir),
        ("VAE", engine_paths.vae_dir),
        ("LoRAs", engine_paths.loras_dir),
        ("Adapters", engine_paths.adapters_dir),
        ("ControlNet", engine_paths.controlnet_dir),
        ("Upscale models", engine_paths.upscale_dir),
        ("Background models", engine_paths.background_dir),
        ("Segmentation models", engine_paths.segmentation_dir),
        ("Models root", engine_paths.models_root),
    ]


def human_size(count):
    """A file size like ``1.2 GB``; ``7 B`` for tiny files."""
    size = float(count or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def human_count(value):
    """A popularity number like ``12.3M`` or ``4.5K``."""
    n = int(value or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _grab(window):
    """Give ``window`` the grab and return the widget that had it before."""
    try:
        previous = window.grab_current()
    except tk.TclError:
        previous = None
    try:
        window.grab_set()
    except tk.TclError:
        pass
    return previous


def _restore(previous):
    if previous is not None and previous.winfo_exists():
        try:
            previous.grab_set()
        except tk.TclError:
            pass


def _run_async(window, work, callback, interval=80):
    """Run ``work()`` on a thread and deliver its result on the Tk thread.

    ``callback(result, error)`` fires once from an ``after`` poller, so it may
    touch widgets. A window that is destroyed before the worker finishes simply
    never delivers (its pending ``after`` is dropped with it).
    """
    box = {"done": False, "result": None, "error": None}

    def runner():
        try:
            box["result"] = work()
        except Exception as exc:  # noqa: BLE001 - surfaced to the dialog
            box["error"] = exc
        box["done"] = True

    threading.Thread(target=runner, daemon=True).start()

    def poll():
        if box["done"]:
            if _alive(window):
                callback(box["result"], box["error"])
            return
        window.after(interval, poll)

    window.after(interval, poll)


def _alive(widget):
    try:
        return bool(widget.winfo_exists())
    except tk.TclError:
        return False


class HubBrowserDialog:
    """Search the Hugging Face Hub and pick a repo to download."""

    def __init__(self, root, on_downloaded=None):
        self.root = root
        self.on_downloaded = on_downloaded
        self.results = []
        self.window = tk.Toplevel(root)
        self.window.title("Browse Hugging Face")
        self.window.transient(root)
        self.window.withdraw()
        self._build()
        self.window.bind("<Escape>", lambda _e: self._close())
        self.window.bind("<Return>", lambda _e: self._search())
        theme.restyle(self.window)
        self._previous_grab = _grab(self.window)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._center()
        self._search()

    # ----------------------------------------------------------------- layout
    def _build(self):
        body = tk.Frame(self.window, padx=10, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        top = tk.Frame(body)
        top.pack(fill=tk.X)
        tk.Label(top, text="Search").pack(side=tk.LEFT)
        self.query_var = tk.StringVar()
        entry = tk.Entry(top, textvariable=self.query_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        entry.focus_set()
        tk.Button(top, text="Search", width=8, command=self._search).pack(side=tk.LEFT)

        controls = tk.Frame(body)
        controls.pack(fill=tk.X, pady=(6, 6))
        tk.Label(controls, text="Sort").pack(side=tk.LEFT)
        self.sort_var = tk.StringVar(value=engine_hub.SORTS[0][0])
        sort_box = ttk.Combobox(
            controls, textvariable=self.sort_var, state="readonly", width=15,
            values=[label for label, _key in engine_hub.SORTS],
        )
        sort_box.pack(side=tk.LEFT, padx=(6, 0))
        sort_box.bind("<<ComboboxSelected>>", lambda _e: self._search())
        self.desc_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls, text="Descending", variable=self.desc_var,
            command=self._search,
        ).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(controls, text="Filter").pack(side=tk.LEFT, padx=(16, 0))
        self.filter_var = tk.StringVar(value=engine_hub.FILTERS[0][0])
        filter_box = ttk.Combobox(
            controls, textvariable=self.filter_var, state="readonly", width=16,
            values=[label for label, _tag in engine_hub.FILTERS],
        )
        filter_box.pack(side=tk.LEFT, padx=(6, 0))
        filter_box.bind("<<ComboboxSelected>>", lambda _e: self._search())

        table = tk.Frame(body)
        table.pack(fill=tk.BOTH, expand=True)
        columns = ("model", "downloads", "likes", "updated")
        self.tree = ttk.Treeview(
            table, columns=columns, show="headings", selectmode="browse", height=14,
        )
        headings = (
            ("model", "Model", 300, "w"),
            ("downloads", "Downloads", 90, "e"),
            ("likes", "Likes", 70, "e"),
            ("updated", "Updated", 90, "w"),
        )
        for key, text, width, anchor in headings:
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, anchor=anchor, stretch=key == "model")
        scroll = AutoScrollbar(table, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self._open_selected)

        bottom = tk.Frame(body)
        bottom.pack(fill=tk.X, pady=(8, 0))
        self.download_btn = tk.Button(
            bottom, text="Download\u2026", width=12, state=tk.DISABLED,
            command=self._open_selected,
        )
        self.download_btn.pack(side=tk.RIGHT)
        tk.Button(bottom, text="Close", width=8, command=self._close).pack(
            side=tk.RIGHT, padx=(0, 6)
        )
        self.status = tk.Label(bottom, text="", anchor="w")
        theme.register(self.status, fg="muted")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _center(self):
        self.window.update_idletasks()
        from . import dialogs

        dialogs.center_window(self.root, self.window)

    # ---------------------------------------------------------------- actions
    def _search(self):
        query = self.query_var.get().strip()
        sort = dict(engine_hub.SORTS).get(self.sort_var.get(), engine_hub.DEFAULT_SORT)
        tag = dict(engine_hub.FILTERS).get(self.filter_var.get())
        descending = bool(self.desc_var.get())
        self.status.config(text="Searching\u2026")
        self.download_btn.config(state=tk.DISABLED)

        def work():
            return engine_hub.search_models(
                query=query, sort=sort, descending=descending, tag=tag, limit=_LIMIT,
            )

        _run_async(self.window, work, self._show_results)

    def _show_results(self, items, error):
        if error is not None:
            self.results = []
            self._fill_tree([])
            self.status.config(text=str(error))
            return
        self.results = list(items)
        self._fill_tree(self.results)
        self.status.config(
            text=f"{len(self.results)} result(s). Double-click one to choose files."
        )

    def _fill_tree(self, items):
        self.tree.delete(*self.tree.get_children())
        for index, raw in enumerate(items):
            info = engine_hub.model_summary(raw)
            self.tree.insert(
                "", tk.END, iid=str(index),
                values=(
                    info["id"], human_count(info["downloads"]),
                    human_count(info["likes"]), info["updated"][:10],
                ),
            )

    def _on_select(self, _event=None):
        state = tk.NORMAL if self.tree.selection() else tk.DISABLED
        self.download_btn.config(state=state)

    def _selected(self):
        selection = self.tree.selection()
        if not selection:
            return None
        try:
            return self.results[int(selection[0])]
        except (ValueError, IndexError):
            return None

    def _open_selected(self, _event=None):
        raw = self._selected()
        if raw is None:
            return
        summary = engine_hub.model_summary(raw)
        self.status.config(text=f"Reading files in {summary['id']}\u2026")
        self.download_btn.config(state=tk.DISABLED)

        def work():
            return engine_hub.list_files(summary["id"])

        _run_async(self.window, work, lambda files, error: self._open_files(summary, files, error))

    def _open_files(self, summary, files, error):
        self._on_select()
        if error is not None:
            self.status.config(text=str(error))
            return
        if not files:
            self.status.config(text="This repo has no files to download.")
            return
        self.status.config(text=f"{summary['id']}: {len(files)} file(s).")
        HubFileDialog(
            self.root, summary, files, on_downloaded=self._downloaded,
        )

    def _downloaded(self, paths):
        if self.on_downloaded is not None:
            self.on_downloaded(paths)
        self.status.config(text=f"Downloaded {len(paths)} file(s) into the model folders.")

    def _close(self):
        _restore(self._previous_grab)
        self.window.destroy()


class HubFileDialog:
    """Choose the files in one repo and the folder to download them into."""

    def __init__(self, root, summary, files, on_downloaded=None):
        self.root = root
        self.summary = summary
        self.files = files
        self.on_downloaded = on_downloaded
        self._targets = _folder_targets()
        self._names = {}
        self.window = tk.Toplevel(root)
        self.window.title(f"Files \u2014 {summary['id']}")
        self.window.transient(root)
        self.window.withdraw()
        self._build()
        self.window.bind("<Escape>", lambda _e: self._close())
        theme.restyle(self.window)
        self._previous_grab = _grab(self.window)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        from . import dialogs

        dialogs.center_window(self.root, self.window)

    def _build(self):
        body = tk.Frame(self.window, padx=10, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        header = tk.Label(
            body, text=self.summary["id"], anchor="w",
        )
        theme.register(header, fg="text")
        header.pack(fill=tk.X)
        meta = tk.Label(
            body, anchor="w",
            text="  ".join(
                part for part in (
                    self.summary["pipeline"], self.summary["library"],
                ) if part
            ) or "Hugging Face repo",
        )
        theme.register(meta, fg="muted")
        meta.pack(fill=tk.X, pady=(0, 6))

        picker = tk.Frame(body)
        picker.pack(fill=tk.X, pady=(0, 6))
        self.only_models_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            picker, text="Model files only", variable=self.only_models_var,
            command=self._fill_tree,
        ).pack(side=tk.LEFT)
        self.rename_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            picker, text="Rename generic names", variable=self.rename_var,
            command=self._fill_tree,
        ).pack(side=tk.LEFT, padx=(12, 0))

        table = tk.Frame(body)
        table.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(
            table, columns=("path", "name", "size"), show="headings",
            selectmode="extended", height=14,
        )
        self.tree.heading("path", text="File")
        self.tree.heading("name", text="Save as")
        self.tree.heading("size", text="Size")
        self.tree.column("path", width=300, anchor="w", stretch=True)
        self.tree.column("name", width=200, anchor="w", stretch=True)
        self.tree.column("size", width=80, anchor="e", stretch=False)
        scroll = AutoScrollbar(table, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._fill_tree()

        dest = tk.Frame(body)
        dest.pack(fill=tk.X, pady=(6, 0))
        tk.Label(dest, text="Download to").pack(side=tk.LEFT)
        self.dest_var = tk.StringVar(value=self._guess_target())
        dest_box = ttk.Combobox(
            dest, textvariable=self.dest_var, state="readonly", width=18,
            values=[label for label, _fn in self._targets],
        )
        dest_box.pack(side=tk.LEFT, padx=(6, 6))
        dest_box.bind("<<ComboboxSelected>>", lambda _e: self._on_dest_change())
        self.dest_path = tk.Label(dest, text="", anchor="w")
        theme.register(self.dest_path, fg="muted")
        self.dest_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(dest, text="Browse\u2026", command=self._browse).pack(side=tk.RIGHT)
        tk.Button(
            dest, text="Models folder\u2026", command=self._choose_root,
        ).pack(side=tk.RIGHT, padx=(0, 6))
        self._refresh_dest()

        bottom = tk.Frame(body)
        bottom.pack(fill=tk.X, pady=(8, 0))
        tk.Button(bottom, text="Download", width=12, command=self._download).pack(
            side=tk.RIGHT
        )
        tk.Button(bottom, text="Close", width=8, command=self._close).pack(
            side=tk.RIGHT, padx=(0, 6)
        )
        tk.Button(
            bottom, text="Rename\u2026", width=10, command=self._rename,
        ).pack(side=tk.RIGHT, padx=(0, 6))
        self.status = tk.Label(bottom, text="", anchor="w")
        theme.register(self.status, fg="muted")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ---------------------------------------------------------------- helpers
    def _visible_files(self):
        if not self.only_models_var.get():
            return list(self.files)
        return [
            item for item in self.files
            if item["path"].lower().endswith(engine_hub.MODEL_SUFFIXES)
        ]

    def _fill_tree(self):
        self.tree.delete(*self.tree.get_children())
        self._shown = self._visible_files()
        self._display = {}
        taken = []
        for index, item in enumerate(self._shown):
            name = engine_hub.safe_name(self._effective_name(item))
            if not name:
                name = os.path.basename(item["path"])
            name = engine_hub.unique_name(name, taken)
            taken.append(name)
            self._display[item["path"]] = name
            self.tree.insert(
                "", tk.END, iid=str(index),
                values=(item["path"], name, human_size(item["size"])),
            )
        status = getattr(self, "status", None)
        if status is not None:
            status.config(text="" if self._shown else "No files match the filter.")

    def _guess_target(self):
        tags = {tag.lower() for tag in self.summary["tags"]}
        if tags & {"segment-anything", "sam", "segmentation"}:
            return "Segmentation models"
        if tags & {"birefnet", "rmbg", "rembg", "isnet", "u2net", "modnet"}:
            return "Background models"
        if tags & {"esrgan", "swinir", "upscale", "realesrgan"}:
            return "Upscale models"
        if "gguf" in tags:
            return "Unet / GGUF"
        if tags & {"anima-ref-adapter", "anima-ref", "ref-adapter", "adapter"}:
            return "Adapters"
        if "lora" in tags:
            return "LoRAs"
        if "controlnet" in tags or "lllite" in tags:
            return "ControlNet"
        if self.summary["library"] == "diffusers" or self.summary["pipeline"] == "text-to-image":
            return "Diffusion models"
        return "Models root"

    def _custom_dir(self):
        return getattr(self, "_custom", None)

    def _selected_dest(self):
        return self._custom_dir() or self._dest_dir()

    def _dest_dir(self):
        for label, fn in self._targets:
            if label == self.dest_var.get():
                return fn()
        return ""

    def _on_dest_change(self):
        self._custom = None
        self._refresh_dest()

    def _refresh_dest(self):
        dest = self._selected_dest()
        shown = engine_paths.shorten_path(dest, engine_paths.show_full_paths())
        if dest and os.path.isabs(dest):
            self.dest_path.config(text=shown)
        elif dest:
            self.dest_path.config(text=f"(choose a models folder) {shown}")
        else:
            self.dest_path.config(text="(no destination chosen)")

    def _browse(self):
        initial = self._selected_dest() or engine_paths.models_root() or None
        chosen = filedialog.askdirectory(parent=self.window, initialdir=initial)
        if chosen:
            self._custom = os.path.normpath(chosen)
            self.dest_path.config(
                text=engine_paths.shorten_path(
                    self._custom, engine_paths.show_full_paths(),
                )
            )

    def _choose_root(self):
        """Set the models root from here, so a new user can organize downloads
        into the standard folders without leaving for Model Settings."""
        chosen = filedialog.askdirectory(
            parent=self.window, initialdir=engine_paths.models_root() or None,
        )
        if not chosen:
            return
        chosen = os.path.normpath(chosen)
        cfg = engine_settings.section("models")
        cfg["root"] = chosen
        engine_settings.set_section("models", cfg)
        engine_paths.reload()
        self._targets = _folder_targets()
        self._custom = None
        self._refresh_dest()
        self.status.config(
            text="Models folder set to "
            f"{engine_paths.shorten_path(chosen, engine_paths.show_full_paths())}."
        )

    def _selected_files(self):
        chosen = []
        for iid in self.tree.selection():
            try:
                chosen.append(self._shown[int(iid)])
            except (ValueError, IndexError):
                continue
        return chosen

    # --------------------------------------------------------------- renaming
    def _effective_name(self, item):
        """The name ``item`` would be saved as, before batch de-duplication."""
        override = self._names.get(item["path"])
        if override:
            return override
        if self.rename_var.get():
            return engine_hub.suggest_filename(self.summary["id"], item["path"])
        return os.path.basename(item["path"])

    def _names_for(self, files):
        """``[(item, name), ...]`` using the names shown in the "Save as" column.

        The shown names are already unique across the whole file list, so any
        selection of them is unique too; the fallback re-derives a safe name for
        a row that was somehow not displayed.
        """
        jobs = []
        for item in files:
            name = self._display.get(item["path"])
            if not name:
                name = engine_hub.safe_name(self._effective_name(item))
                name = name or os.path.basename(item["path"])
            jobs.append((item, name))
        return jobs

    def _rename(self):
        files = self._selected_files()
        if not files:
            self.status.config(text="Select the files to rename first.")
            return
        entries = [
            (item["path"], self._display.get(item["path"]) or self._effective_name(item))
            for item in files
        ]
        result = _RenameDialog(self.window, entries).run()
        if result is None:
            return
        for item, name in zip(files, result):
            self._names[item["path"]] = name
        self._fill_tree()
        self.status.config(text="Renamed the selected file(s).")

    # ---------------------------------------------------------------- actions
    def _download(self):
        files = self._selected_files()
        if not files:
            self.status.config(text="Select at least one file to download.")
            return
        dest = self._selected_dest()
        if not dest or not os.path.isabs(dest):
            self.status.config(
                text="Pick a destination folder with Browse\u2026 or Models folder\u2026."
            )
            return
        jobs = [
            {
                "repo": self.summary["id"],
                "filename": item["path"],
                "name": name,
                "dest_dir": dest,
                "size": item.get("size") or 0,
            }
            for item, name in self._names_for(files)
        ]
        window = _DownloadWindow(self.window, jobs, title=self.summary["id"])
        results = window.run()
        saved = []
        for result in results:
            if result.get("members"):
                saved.extend(result["members"])
            elif result.get("path"):
                saved.append(result["path"])
        if not saved:
            return
        if self.on_downloaded is not None:
            self.on_downloaded(saved)
        self.status.config(text=f"Saved {len(saved)} file(s) to {dest}.")

    def _close(self):
        _restore(self._previous_grab)
        self.window.destroy()


class _RenameDialog:
    """Manually set the file name each selected download is saved under.

    ``entries`` is ``[(original_path, suggested_name), ...]``. :meth:`run`
    returns the edited names in order, or ``None`` when cancelled.
    """

    def __init__(self, root, entries):
        self.root = root
        self.entries = entries
        self.result = None
        self.vars = []
        self.window = tk.Toplevel(root)
        self.window.title("Rename downloads")
        self.window.transient(root)
        self.window.withdraw()
        self._build()
        self.window.bind("<Escape>", lambda _e: self._close(None))
        theme.restyle(self.window)
        self._previous_grab = _grab(self.window)
        self.window.protocol("WM_DELETE_WINDOW", lambda: self._close(None))
        from . import dialogs

        dialogs.center_window(self.root, self.window)

    def _build(self):
        body = tk.Frame(self.window, padx=10, pady=10)
        body.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            body, text="Set the name each file is saved as:", anchor="w",
        ).pack(fill=tk.X)

        grid = tk.Frame(body)
        grid.pack(fill=tk.BOTH, expand=True, pady=(6, 6))
        grid.columnconfigure(1, weight=1)
        for row, (path, name) in enumerate(self.entries):
            tk.Label(
                grid, text=os.path.basename(path), anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=(0, 6))
            var = tk.StringVar(value=name)
            entry = tk.Entry(grid, textvariable=var, width=42)
            entry.grid(row=row, column=1, sticky="we")
            self.vars.append(var)

        buttons = tk.Frame(body)
        buttons.pack(fill=tk.X)
        tk.Button(buttons, text="OK", width=8, command=self._ok).pack(side=tk.RIGHT)
        tk.Button(
            buttons, text="Cancel", width=8, command=lambda: self._close(None),
        ).pack(side=tk.RIGHT, padx=(0, 6))
        self.status = tk.Label(buttons, text="", anchor="w")
        theme.register(self.status, fg="muted")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _ok(self):
        names = []
        taken = []
        for var in self.vars:
            name = engine_hub.safe_name(var.get())
            if not name:
                self.status.config(text="A file name cannot be empty.")
                return
            if name.lower() in {existing.lower() for existing in taken}:
                self.status.config(text=f"Duplicate name: {name}")
                return
            taken.append(name)
            names.append(name)
        self._close(names)

    def _close(self, result):
        self.result = result
        _restore(self._previous_grab)
        if _alive(self.window):
            self.window.destroy()

    def run(self):
        self.window.deiconify()
        self.window.wait_window(self.window)
        return self.result


class _DownloadWindow:
    """A modal progress bar for downloading a list of files, one at a time.

    Each job is a dict with ``repo``, ``filename``, ``dest_dir`` and optional
    ``name``, ``size`` and ``label``. Jobs may target different folders, which
    the model download wizard needs (a diffusion model, a text encoder and a VAE
    land in three folders). :meth:`run` returns one result per completed job:
    ``{"job", "path", "members"}``.
    """

    def __init__(self, root, jobs, title="Downloading"):
        self.root = root
        self.jobs = jobs
        self.saved = []
        self.results = []
        self._cancel = CancelToken()
        self._queue = queue.Queue()
        self._finished = False
        self.window = tk.Toplevel(root)
        self.window.title(title)
        self.window.transient(root)
        self.window.resizable(False, False)
        self.window.withdraw()
        self._build()
        theme.restyle(self.window)
        self._previous_grab = self._grab_self()
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)
        from . import dialogs

        dialogs.center_window(self.root, self.window)

    def _grab_self(self):
        try:
            previous = self.window.grab_current()
        except tk.TclError:
            previous = None
        try:
            self.window.grab_set()
        except tk.TclError:
            pass
        return previous

    def _build(self):
        body = tk.Frame(self.window, padx=14, pady=12)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.name = tk.Label(body, text="Preparing\u2026", anchor="w")
        theme.register(self.name, fg="text")
        self.name.pack(fill=tk.X)
        self.detail = tk.Label(body, text="", anchor="w")
        theme.register(self.detail, fg="muted")
        self.detail.pack(fill=tk.X, pady=(2, 6))
        self.bar = ttk.Progressbar(body, mode="determinate", maximum=100, length=340)
        self.bar.pack(fill=tk.X)
        self.status = tk.Label(body, text="", anchor="w")
        theme.register(self.status, fg="muted")
        self.status.pack(fill=tk.X, pady=(6, 0))
        tk.Button(body, text="Cancel", width=8, command=self._on_cancel).pack(
            side=tk.RIGHT, pady=(8, 0)
        )

    def run(self):
        """Start the transfers and block until they finish.

        Returns one result dict per completed job:
        ``{"job", "path", "members"}``. ``members`` is non-empty when the
        download was an archive that was unpacked into the destination.
        """
        threading.Thread(target=self._work, daemon=True).start()
        self.window.after(80, self._poll)
        self.root.wait_window(self.window)
        return self.results

    def _work(self):
        job_count = len(self.jobs)
        speed = {"time": None, "read": 0, "value": 0.0}

        def progress(frac, label, read=0, total=0, index=0, rate=0.0):
            self._queue.put(("progress", frac, label, read, total, index, rate))

        try:
            for index, job in enumerate(self.jobs):
                self._cancel.throw_if_cancelled()
                speed.update(time=None, read=0, value=0.0)
                name = job.get("name") or job["filename"].rsplit("/", 1)[-1]
                dest_dir = job["dest_dir"]

                def on_file(frac, label, read=0, total=0, base=index):
                    now = time.monotonic()
                    if speed["time"] is None:
                        speed["time"] = now
                        speed["read"] = read
                    else:
                        elapsed = now - speed["time"]
                        if elapsed >= 0.2:
                            instant = max(0.0, (read - speed["read"]) / elapsed)
                            speed["value"] = (
                                instant if not speed["value"]
                                else speed["value"] * 0.6 + instant * 0.4
                            )
                            speed["time"] = now
                            speed["read"] = read
                    progress(
                        (base + frac) / job_count, label, read, total, base,
                        speed["value"],
                    )

                path = engine_hub.download_file(
                    job["repo"], job["filename"], dest_dir, name=name,
                    progress=on_file, cancel=self._cancel,
                )
                members = engine_hub.extract_archive(path, dest_dir, name=name)
                if members:
                    # The archive held the model files (a SAM ONNX bundle, say);
                    # keep the unpacked files and drop the archive itself.
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    self.saved.extend(members)
                else:
                    self.saved.append(path)
                self.results.append({"job": job, "path": path, "members": members})
        except Cancelled:
            self._queue.put(("cancelled", None))
            return
        except engine_hub.HubError as exc:
            self._queue.put(("error", str(exc)))
            return
        except Exception as exc:  # noqa: BLE001 - reported in the window
            self._queue.put(("error", str(exc)))
            return
        self._queue.put(("done", None))

    def _poll(self):
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            kind = message[0]
            if kind == "progress":
                frac, label, read, size, index, rate = message[1:7]
                job = self.jobs[index]
                if not size:
                    size = job.get("size") or 0
                self.bar.config(value=max(0.0, min(1.0, frac)) * 100)
                role = job.get("label")
                self.name.config(text=f"{role}: {label}" if role else label)
                detail = f"{index + 1} of {len(self.jobs)}"
                if size:
                    detail += f"  \u00b7  {human_size(read)} / {human_size(size)}"
                elif read:
                    detail += f"  \u00b7  {human_size(read)}"
                if rate:
                    detail += f"  \u00b7  {human_size(rate)}/s"
                self.detail.config(text=detail)
            elif kind == "done":
                self._finish("Finished.")
                return
            elif kind == "cancelled":
                self._finish("Cancelled.")
                return
            elif kind == "error":
                self._finish(message[1] or "Failed.", error=True)
                return
        if not self._finished and _alive(self.window):
            self.window.after(80, self._poll)

    def _finish(self, text, error=False):
        if self._finished:
            return
        self._finished = True
        if error:
            self.saved = []
            self.results = []
        self.status.config(text=text)
        if error:
            theme.register(self.status, fg="error")
        if not error:
            self.bar.config(value=100)
        if _alive(self.window):
            self.window.after(400, self._close)

    def _on_cancel(self):
        self._cancel.cancel()
        self.status.config(text="Cancelling\u2026")

    def _close(self):
        _restore(self._previous_grab)
        if _alive(self.window):
            self.window.destroy()
