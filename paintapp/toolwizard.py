"""The tool download wizard.

Installs the standalone models used by the Background, Upscale and Segment
panels. The files and their destinations live in :mod:`engine.tool_catalog`;
this module only drives the three steps:

1. Choose a tool model, with its purpose and trade-offs.
2. Review the files, destinations and anything already on disk.
3. Download, then hand the paths and their component pools back to Model
   Settings.

Nothing is persisted here: the caller receives the files and pools, exactly
like the generation wizard, and Model Settings still needs its Save button.
"""

import os
import shutil
import tkinter as tk
from tkinter import ttk

from engine import tool_catalog

from . import dialogs, theme
from .hubdialog import (
    _DownloadWindow, _alive, _grab, _restore, human_size,
)
from .scrollbar import AutoScrollbar


def _free_bytes(path):
    """Free space on the volume holding ``path`` (``None`` when unknown)."""
    probe = os.path.abspath(path)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _short_disk_folders(jobs):
    """``[(folder, needed_bytes, free_bytes), ...]`` for folders without room."""
    needed = {}
    for job in jobs:
        needed[job["dest_dir"]] = needed.get(job["dest_dir"], 0) + job["size"]
    short = []
    for folder, size in needed.items():
        free = _free_bytes(folder)
        if free is not None and free < size:
            short.append((folder, size, free))
    return short


class ToolDownloadWizard:
    """A modal, three-step downloader for background, upscale and segmentation models."""

    def __init__(self, root, on_ready=None):
        self.root = root
        self.on_ready = on_ready
        self.bundles = tool_catalog.bundles()
        self._step = 0
        self._selected = None
        self._jobs = []
        self._existing = {}
        self._downloading = False
        self._done = False
        self._warnings = []

        self.window = tk.Toplevel(root)
        self.window.title("Download a tool model")
        self.window.transient(root)
        self.window.withdraw()
        self._build_shell()
        self.window.bind("<Escape>", lambda _e: self._close())
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._show_step(0)
        theme.restyle(self.window)
        self._previous_grab = _grab(self.window)
        dialogs.center_window(root, self.window)

    # ------------------------------------------------------------------ shell
    def _build_shell(self):
        body = tk.Frame(self.window, padx=12, pady=10)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)

        title = tk.Label(
            body, text="Download a background, upscale or segmentation model", anchor="w",
        )
        theme.register(title, fg="text")
        title.grid(row=0, column=0, sticky="we")

        self.hint = tk.Label(body, text="", anchor="w")
        theme.register(self.hint, fg="muted")
        self.hint.grid(row=1, column=0, sticky="we", pady=(2, 8))

        self.content = tk.Frame(body)
        self.content.grid(row=2, column=0, sticky="nsew")

        footer = tk.Frame(body)
        footer.grid(row=3, column=0, sticky="we", pady=(10, 0))
        self.status = tk.Label(footer, text="", anchor="w")
        theme.register(self.status, fg="muted")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._next_btn = tk.Button(
            footer, text="Next", width=12, command=self._next,
        )
        self._next_btn.pack(side=tk.RIGHT)
        self._back_btn = tk.Button(
            footer, text="Back", width=8, command=self._back,
        )
        self._back_btn.pack(side=tk.RIGHT, padx=(0, 6))
        tk.Button(footer, text="Cancel", width=8, command=self._close).pack(
            side=tk.RIGHT, padx=(0, 6),
        )

    # ------------------------------------------------------------------ steps
    def _show_step(self, step):
        self._step = step
        for child in self.content.winfo_children():
            child.destroy()
        if step == 0:
            self.hint.config(
                text=(
                    "These models run on their own in the Background, Upscale "
                    "and Segment panels."
                ),
            )
            self._build_choose()
        elif step == 1:
            self.hint.config(
                text=(
                    "Review the files and folders. Anything already present is "
                    "reused."
                ),
            )
            self._build_review()
        else:
            self.hint.config(
                text="The download runs in a separate progress window.",
            )
            self._build_confirm()
        self._refresh_footer()
        self._fit_window()
        theme.restyle(self.window)

    def _fit_window(self):
        """Grow the mapped window when a step needs more room than the last."""
        if not _alive(self.window):
            return
        try:
            if not self.window.winfo_viewable():
                return
            self.window.update_idletasks()
            width = self.window.winfo_reqwidth()
            height = self.window.winfo_reqheight()
            if (width > self.window.winfo_width()
                    or height > self.window.winfo_height()):
                self.window.geometry(
                    f"{max(width, self.window.winfo_width())}x"
                    f"{max(height, self.window.winfo_height())}"
                    f"+{self.window.winfo_x()}+{self.window.winfo_y()}"
                )
        except tk.TclError:
            pass

    def _refresh_footer(self):
        self._back_btn.config(
            state=(
                tk.NORMAL if self._step > 0 and not self._downloading
                else tk.DISABLED
            ),
        )
        self._next_btn.config(command=self._next)
        if self._step == 0:
            self._next_btn.config(
                text="Next", state=tk.NORMAL if self._selected else tk.DISABLED,
            )
        elif self._step == 1:
            self._next_btn.config(text="Next", state=tk.NORMAL)
        elif self._done:
            self._next_btn.config(
                text="Close", state=tk.NORMAL, command=self._close,
            )
        else:
            self._next_btn.config(
                text="Download",
                state=tk.DISABLED if self._downloading else tk.NORMAL,
            )

    def _next(self):
        if self._step == 0:
            if self._selected is not None:
                self._show_step(1)
        elif self._step == 1:
            self._show_step(2)
        elif self._step == 2:
            self._start_download()

    def _back(self):
        if self._downloading or self._done or self._step == 0:
            return
        self._show_step(self._step - 1)

    # -------------------------------------------------------------- step 1
    def _build_choose(self):
        tk.Label(
            self.content, text="Choose a tool model", anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

        table = tk.Frame(self.content)
        table.pack(fill=tk.BOTH, expand=True)
        self._tree = ttk.Treeview(
            table, columns=("tool", "files", "size"),
            show="tree headings", selectmode="browse", height=10,
        )
        self._tree.heading("#0", text="Model")
        self._tree.heading("tool", text="Tool")
        self._tree.heading("files", text="Files")
        self._tree.heading("size", text="Download")
        self._tree.column("#0", width=260, anchor="w", stretch=True)
        self._tree.column("tool", width=140, anchor="w", stretch=False)
        self._tree.column("files", width=60, anchor="center", stretch=False)
        self._tree.column("size", width=90, anchor="e", stretch=False)
        scroll = AutoScrollbar(table, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for bundle in self.bundles:
            self._tree.insert(
                "", tk.END, iid=bundle["id"], text=bundle["label"],
                values=(
                    bundle["group"], len(bundle["files"]),
                    human_size(tool_catalog.total_bytes(bundle)),
                ),
            )

        detail_box = tk.Frame(self.content)
        detail_box.pack(fill=tk.X, pady=(6, 0))
        self.detail = tk.Text(
            detail_box, height=13, wrap="word", state=tk.DISABLED,
            borderwidth=0, highlightthickness=0, padx=4, pady=2,
        )
        theme.register(self.detail, fg="muted")
        self._detail_scroll = AutoScrollbar(
            detail_box, command=self.detail.yview,
        )
        self.detail.configure(yscrollcommand=self._detail_scroll.set)
        self._detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        note = tk.Label(
            self.content,
            text=(
                "Community tool models can still be added with Browse\u2026 or "
                "Scan folder\u2026 in their component pool."
            ),
            anchor="w", justify="left", wraplength=620,
        )
        theme.register(note, fg="muted")
        note.pack(fill=tk.X, pady=(6, 0))
        self._tree.bind("<<TreeviewSelect>>", self._on_choose)
        if self._selected is not None and self._tree.exists(self._selected["id"]):
            self._tree.selection_set(self._selected["id"])
        self._refresh_detail()

    def _on_choose(self, _event=None):
        chosen = self._tree.selection()
        self._selected = tool_catalog.bundle(chosen[0]) if chosen else None
        self._refresh_detail()
        self._refresh_footer()

    def _set_detail(self, text):
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", text)
        self.detail.configure(state=tk.DISABLED)
        self.detail.yview_moveto(0.0)

    def _refresh_detail(self):
        if self._selected is None:
            self._set_detail(
                "Select a tool model to see its purpose, strengths and what it "
                "needs."
            )
            return
        description = self._selected.get("description") or {}
        lines = []
        purpose = description.get("purpose")
        if purpose:
            lines.append("Purpose: " + purpose)
        strengths = description.get("strengths") or ()
        if strengths:
            lines.append("Strengths:")
            lines.extend("  \u2022 " + item for item in strengths)
        consider = description.get("consider") or ()
        if consider:
            lines.append("Consider:")
            lines.extend("  \u2022 " + item for item in consider)
        if lines:
            lines.append("")
        lines.append(
            f"Download: {human_size(tool_catalog.total_bytes(self._selected))} "
            f"in {len(self._selected['files'])} file(s). "
            + (self._selected.get("notes") or "")
        )
        self._set_detail("\n".join(lines))

    # -------------------------------------------------------------- step 2
    def _build_review(self):
        bundle = self._selected
        tk.Label(self.content, text=bundle["label"], anchor="w").pack(fill=tk.X)
        tk.Label(
            self.content,
            text=f"{bundle['group']}  \u00b7  {bundle['notes']}",
            anchor="w", justify="left", wraplength=620,
        ).pack(fill=tk.X, pady=(2, 8))

        if bundle.get("gated"):
            warning = tk.Label(
                self.content,
                text=(
                    "Gated on Hugging Face: accept the model license in a "
                    "browser and set HF_TOKEN in the environment before "
                    "downloading."
                ),
                anchor="w", justify="left", wraplength=620,
            )
            theme.register(warning, fg="error")
            warning.pack(fill=tk.X, pady=(0, 8))

        table = tk.Frame(self.content)
        table.pack(fill=tk.BOTH, expand=True)
        self._files = ttk.Treeview(
            table, columns=("role", "file", "folder", "size", "state"),
            show="headings", selectmode="none", height=8,
        )
        self._files.heading("role", text="Role")
        self._files.heading("file", text="File")
        self._files.heading("folder", text="Folder")
        self._files.heading("size", text="Size")
        self._files.heading("state", text="Status")
        self._files.column("role", width=90, anchor="w", stretch=False)
        self._files.column("file", width=270, anchor="w", stretch=True)
        self._files.column("folder", width=130, anchor="w", stretch=False)
        self._files.column("size", width=70, anchor="e", stretch=False)
        self._files.column("state", width=150, anchor="w", stretch=False)
        scroll = AutoScrollbar(table, command=self._files.yview)
        self._files.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._fill_files()

    def _fill_files(self):
        self._jobs = tool_catalog.jobs(self._selected)
        self._existing = {}
        self._files.delete(*self._files.get_children())
        present_count = 0
        for job in self._jobs:
            info = tool_catalog.find_existing(job)
            if info:
                self._existing[job["slot"]] = info
                present_count += 1
                if info["reason"] == "exact":
                    state = "Already present"
                else:
                    state = "Reuse: " + os.path.basename(info["path"])
            else:
                state = "Will download"
            self._files.insert(
                "", tk.END, values=(
                    job["label"],
                    job["name"],
                    os.path.basename(job["dest_dir"]),
                    human_size(job["size"]),
                    state,
                ),
            )
        total = sum(
            job["size"] for job in self._jobs
            if job["slot"] not in self._existing
        )
        self.status.config(
            text=(
                f"{len(self._jobs)} file(s), {human_size(total)} to download"
                + (f"; {present_count} already available" if present_count else ".")
            ),
        )

    # -------------------------------------------------------------- step 3
    def _build_confirm(self):
        jobs = self._jobs or tool_catalog.jobs(self._selected)
        pending = [job for job in jobs if job["slot"] not in self._existing]
        total = sum(job["size"] for job in pending)
        tk.Label(
            self.content, text=self._selected["label"], anchor="w",
        ).pack(fill=tk.X)
        if self._selected.get("tool") == "segmentation":
            destination = (
                "The download is unpacked into the segmentation folder and "
                "appears in the Segment tool."
            )
        else:
            destination = (
                "The downloads are added to the matching pool in Model "
                "Settings; press Save there to keep them."
            )
        summary = tk.Label(
            self.content,
            text=(
                f"{len(pending)} file(s) will be downloaded "
                f"({human_size(total)}).\n"
                "Files already present are reused. " + destination
            ),
            anchor="w", justify="left", wraplength=620,
        )
        theme.register(summary, fg="muted")
        summary.pack(fill=tk.X, pady=(4, 8))
        short = _short_disk_folders(pending)
        if short:
            warning = tk.Label(
                self.content,
                text="Not enough free space:\n" + "\n".join(
                    f"  \u00b7 {os.path.basename(folder)}: needs "
                    f"{human_size(size)}, has {human_size(free)}"
                    for folder, size, free in short
                ),
                anchor="w", justify="left", wraplength=620,
            )
            theme.register(warning, fg="error")
            warning.pack(fill=tk.X, pady=(0, 8))
        self._result = tk.Label(
            self.content, text="", anchor="w", justify="left", wraplength=620,
        )
        self._result.pack(fill=tk.X)
        if self._done:
            self._show_result_text()

    def _start_download(self):
        if self._downloading or self._done:
            return
        jobs = [job for job in self._jobs if job["slot"] not in self._existing]
        short = _short_disk_folders(jobs)
        if short:
            self._result.config(
                text="Not enough free space in "
                + ", ".join(os.path.basename(folder) for folder, _s, _f in short)
                + ". Free some space or choose another model.",
            )
            theme.register(self._result, fg="error")
            return
        self._downloading = True
        self._refresh_footer()
        self.status.config(text="")
        results = {
            slot: info["path"] for slot, info in self._existing.items()
        }
        if jobs:
            window = _DownloadWindow(
                self.window, jobs, title="Downloading tool model files",
            )
            for result in window.run():
                job = result.get("job") or {}
                path = result.get("members") or result.get("path")
                if isinstance(path, list):
                    path = path[0] if path else ""
                if path and job.get("slot"):
                    results[job["slot"]] = path
        self._downloading = False

        missing = [
            job for job in self._jobs if job["slot"] not in results
        ]
        if missing:
            self._done = False
            self._result.config(
                text="Download incomplete. Missing: "
                + ", ".join(job["label"] for job in missing)
                + ". Press Download to retry.",
            )
            theme.register(self._result, fg="error")
            self._refresh_footer()
            return

        files = [
            {
                "path": results[job["slot"]],
                "pool": job["pool"],
                "label": (
                    self._selected["label"] if job["slot"] == "model"
                    else f"{self._selected['label']} {job['label']}"
                ),
            }
            for job in self._jobs
        ]
        payload = {
            "tool": self._selected["tool"],
            "label": self._selected["label"],
            "path": results.get("model") or files[0]["path"],
            "files": files,
            "warnings": self._warnings,
        }
        self._done = True
        self._payload = payload
        self._refresh_footer()
        self._show_result_text()
        if self.on_ready is not None:
            try:
                self.on_ready(payload)
            except Exception as exc:  # noqa: BLE001 - keep the wizard usable
                self._result.config(
                    text=f"Downloaded, but adding the model failed: {exc}",
                )
                theme.register(self._result, fg="error")

    def _show_result_text(self):
        if self._warnings:
            text = (
                "Downloaded with warnings:\n"
                + "\n".join("  \u00b7 " + line for line in self._warnings)
            )
            theme.register(self._result, fg="error")
        else:
            if self._selected.get("tool") == "segmentation":
                text = "Downloaded. The model is ready in the Segment tool."
            else:
                text = (
                    "Downloaded. The model is ready in the matching pool in "
                    "Model Settings; press Save there to keep it."
                )
            theme.register(self._result, fg="text")
        self._result.config(text=text)
        self.status.config(text="")

    # ------------------------------------------------------------------ close
    def _close(self):
        if self._downloading:
            return
        _restore(self._previous_grab)
        if _alive(self.window):
            self.window.destroy()
