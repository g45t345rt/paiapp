"""The model download wizard.

Downloads a complete model set -- the diffusion weights plus the VAE and text
encoder a family needs -- instead of asking the user to work out which companion
files belong together. The compatibility data lives in
:mod:`engine.model_catalog`; the hardware probe in :mod:`engine.hardware` only
chooses which quantization profile to recommend.

Three steps:

1. Choose a bundle, with a fit badge for the detected GPU/RAM.
2. Review the files, destination folders and chosen profile.
3. Download, verify the file kinds, then hand the paths back to Model Settings.

Nothing is persisted here: the caller receives the paths and staged component
assignments, exactly like the existing Hub downloader.
"""

import os
import shutil
import tkinter as tk
from tkinter import ttk

from engine import component_kinds as engine_kinds
from engine import hardware as engine_hardware
from engine import model_catalog
from engine import tuning as engine_tuning

from . import dialogs, monitor, theme
from .hubdialog import (
    _DownloadWindow, _alive, _grab, _restore, _run_async, human_size,
)
from .scrollbar import AutoScrollbar

#: Shown while the hardware probe is still running.
_PENDING = {"pending": True, "gpu": False, "vram_gb": None, "ram_gb": None}


def _gb(value):
    return f"{value:.1f} GB"


def _fit_text(rec):
    """The fit badge for a recommendation result."""
    if rec.get("tier") == "unknown":
        return "Balanced"
    return model_catalog.tier_label(rec["tier"])


def _fit_detail(rec):
    """The explanatory line under a bundle/profile."""
    if rec.get("tier") == "unknown":
        return (
            "Hardware was not detected, so a balanced profile is preselected. "
            "You can pick another quantization below."
        )
    return model_catalog.tier_detail(rec["tier"], {
        "fp8_vram_gb": rec.get("need_fp8_vram_gb", 0.0),
    })


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
        if job.get("remote"):
            continue
        needed[job["dest_dir"]] = needed.get(job["dest_dir"], 0) + job["size"]
    short = []
    for folder, size in needed.items():
        free = _free_bytes(folder)
        if free is not None and free < size:
            short.append((folder, size, free))
    return short


class ModelDownloadWizard:
    """A modal, three-step model downloader for Model Settings."""

    def __init__(self, root, on_ready=None):
        self.root = root
        self.on_ready = on_ready
        self.bundles = model_catalog.bundles()
        self._hardware = dict(_PENDING)
        self._step = 0
        self._selected = None
        self._profile = None
        self._profile_auto = True
        self._jobs = []
        self._existing = {}
        self._downloading = False
        self._done = False

        self.window = tk.Toplevel(root)
        self.window.title("Download a model")
        self.window.transient(root)
        self.window.withdraw()
        self._build_shell()
        self.window.bind("<Escape>", lambda _e: self._close())
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        # Build the first step *before* centring: center_window measures the
        # requested size and locks the geometry, so an empty shell would be
        # mapped at a tiny size with the step content clipped out of view.
        self._show_step(0)
        theme.restyle(self.window)
        self._previous_grab = _grab(self.window)
        dialogs.center_window(root, self.window)
        self._probe()

    # ------------------------------------------------------------------ shell
    def _build_shell(self):
        body = tk.Frame(self.window, padx=12, pady=10)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)

        title = tk.Label(
            body, text="Download a model and its components", anchor="w",
        )
        theme.register(title, fg="text")
        title.grid(row=0, column=0, sticky="we")

        self.hardware_label = tk.Label(body, text="", anchor="w")
        theme.register(self.hardware_label, fg="muted")
        self.hardware_label.grid(row=1, column=0, sticky="we", pady=(2, 8))

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

    # --------------------------------------------------------------- hardware
    def _probe(self):
        def work():
            result = engine_hardware.probe()
            if not result.get("gpu") or not result.get("vram_gb"):
                primary = monitor.primary_gpu()
                if primary:
                    result.update({
                        "gpu": True,
                        "gpu_name": primary["name"],
                        "vram_gb": primary["vram_gb"],
                        "backend": "monitor",
                        "source": "monitor",
                    })
            if not result.get("ram_gb"):
                ram = monitor.ram_usage()
                if ram:
                    result["ram_gb"] = ram[1]
            return result

        _run_async(self.window, work, self._on_probe)

    def _on_probe(self, result, error):
        if error or not isinstance(result, dict):
            result = {"gpu": False, "vram_gb": None, "ram_gb": None}
        self._hardware = result
        self._refresh_hardware_label()
        if not self._downloading and not self._done:
            self._show_step(self._step)

    def _refresh_hardware_label(self):
        hw = self._hardware
        if hw.get("pending"):
            text = "Detecting hardware\u2026"
        elif not hw.get("gpu") and not hw.get("ram_gb"):
            text = "Hardware not detected \u2014 showing balanced defaults."
        else:
            parts = []
            if hw.get("gpu"):
                parts.append(hw.get("gpu_name") or "GPU")
                if hw.get("vram_gb"):
                    parts.append(_gb(hw["vram_gb"]) + " VRAM")
            else:
                parts.append("No GPU detected")
            if hw.get("ram_gb"):
                parts.append(_gb(hw["ram_gb"]) + " RAM")
            text = "  \u00b7  ".join(parts)
        self.hardware_label.config(text=text)

    # --------------------------------------------------------- recommendation
    def _balanced_profile(self, bundle):
        profiles = model_catalog.ordered_profiles(bundle)
        return profiles[len(profiles) // 2]

    def _hardware_unknown(self):
        """Whether the probe has not produced a usable GPU/RAM reading."""
        hw = self._hardware
        return bool(
            hw.get("pending")
            or (hw.get("gpu") and not hw.get("vram_gb"))
            or (not hw.get("gpu") and not hw.get("ram_gb"))
        )

    def _recommend(self, bundle):
        hw = self._hardware
        tuning = engine_tuning.read()
        # A profile whose model is already on disk wins over the hardware pick:
        # downloading a duplicate in another quantization is wasted space, and
        # the profile dropdown still lets the user choose a different one.
        installed = model_catalog.installed_profile(
            bundle, None if self._hardware_unknown() else hw, tuning,
        )
        if installed is not None:
            return installed
        if self._hardware_unknown():
            profile = self._balanced_profile(bundle)
            return {
                "profile": profile,
                "tier": "unknown",
                "tier_label": "Balanced",
                "total_gb": model_catalog.total_bytes(profile) / 1e9,
            }
        return model_catalog.recommend(bundle, hw, tuning)

    # ------------------------------------------------------------------ steps
    def _show_step(self, step):
        self._step = step
        for child in self.content.winfo_children():
            child.destroy()
        if step == 0:
            self._build_choose()
        elif step == 1:
            self._build_review()
        else:
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
            remote = bool(
                self._profile and model_catalog.is_remote(self._profile)
            )
            self._next_btn.config(
                text="Add model" if remote else "Download",
                state=tk.DISABLED if self._downloading else tk.NORMAL,
            )

    def _next(self):
        if self._step == 0:
            if self._selected is not None:
                rec = self._recommend(self._selected)
                self._profile = rec["profile"] if self._profile_auto else self._profile
                self._show_step(1)
        elif self._step == 1:
            self._show_step(2)
        elif self._step == 2:
            self._start_download()

    def _back(self):
        if self._downloading or self._done or self._step == 0:
            return
        # Re-run the hardware recommendation on the next review so the best
        # profile for the detected hardware is always preselected again.
        self._profile_auto = True
        self._show_step(self._step - 1)

    # -------------------------------------------------------------- step 1
    def _build_choose(self):
        self._refresh_hardware_label()
        tk.Label(
            self.content, text="Choose a model", anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

        table = tk.Frame(self.content)
        table.pack(fill=tk.BOTH, expand=True)
        self._tree = ttk.Treeview(
            table, columns=("family", "profile", "fit", "size"),
            show="tree headings", selectmode="browse", height=10,
        )
        self._tree.heading("#0", text="Model")
        self._tree.heading("family", text="Family")
        self._tree.heading("profile", text="Profile")
        self._tree.heading("fit", text="Fit")
        self._tree.heading("size", text="Download")
        self._tree.column("#0", width=230, anchor="w", stretch=True)
        self._tree.column("family", width=100, anchor="w", stretch=False)
        self._tree.column("profile", width=130, anchor="w", stretch=False)
        self._tree.column("fit", width=130, anchor="w", stretch=False)
        self._tree.column("size", width=90, anchor="e", stretch=False)
        scroll = AutoScrollbar(table, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._recs = {}
        for bundle in self.bundles:
            rec = self._recommend(bundle)
            self._recs[bundle["id"]] = rec
            self._tree.insert(
                "", tk.END, iid=bundle["id"], text=bundle["label"],
                values=(
                    bundle["group"], rec["profile"]["label"], _fit_text(rec),
                    human_size(model_catalog.total_bytes(rec["profile"])),
                ),
            )
        detail_box = tk.Frame(self.content)
        detail_box.pack(fill=tk.X, pady=(6, 0))
        self.detail = tk.Text(
            detail_box, height=14, wrap="word", state=tk.DISABLED,
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
                "Community Stable Diffusion finetunes are not listed; use "
                "Browse Hugging Face\u2026 on the Models tab for those."
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
        self._selected = model_catalog.bundle(chosen[0]) if chosen else None
        self._profile_auto = True
        self._refresh_detail()
        self._refresh_footer()

    def _set_detail(self, text):
        """Replace the read-only detail text and scroll back to the top."""
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", text)
        self.detail.configure(state=tk.DISABLED)
        self.detail.yview_moveto(0.0)

    def _refresh_detail(self):
        if self._selected is None:
            self._set_detail(
                "Select a model to see its purpose, strengths and the VAE and "
                "text encoders it needs."
            )
            return
        rec = self._recs.get(self._selected["id"]) or self._recommend(self._selected)
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
            f"Recommended download: {rec['profile']['label']} "
            f"({human_size(model_catalog.total_bytes(rec['profile']))}). "
            + _fit_detail(rec)
        )
        self._set_detail("\n".join(lines))

    # -------------------------------------------------------------- step 2
    def _build_review(self):
        bundle = self._selected
        rec = self._recommend(bundle)
        if self._profile is None or self._profile_auto:
            self._profile = rec["profile"]

        header = tk.Frame(self.content)
        header.pack(fill=tk.X, pady=(0, 4))
        tk.Label(header, text=bundle["label"], anchor="w").pack(side=tk.LEFT)
        self.profile_var = tk.StringVar(value=self._profile_option(self._profile))
        self._profile_map = {
            self._profile_option(profile): profile
            for profile in model_catalog.ordered_profiles(bundle)
        }
        combo = ttk.Combobox(
            header, textvariable=self.profile_var, state="readonly", width=48,
            values=list(self._profile_map),
        )
        combo.pack(side=tk.RIGHT)
        combo.bind("<<ComboboxSelected>>", self._on_profile_change)
        tk.Label(header, text="Profile:", anchor="w").pack(
            side=tk.RIGHT, padx=(0, 6),
        )

        self._fit_label = tk.Label(
            self.content, text="", anchor="w", justify="left", wraplength=620,
        )
        theme.register(self._fit_label, fg="muted")
        self._fit_label.pack(fill=tk.X, pady=(2, 6))
        self._refresh_fit_label()

        if not hasattr(self, "_force_profile"):
            self._force_profile = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.content,
            text=(
                "Download the exact profile files instead of reusing "
                "similar ones (another name or precision)"
            ),
            variable=self._force_profile, command=self._fill_files,
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 6))

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
        self._files.column("role", width=120, anchor="w", stretch=False)
        self._files.column("file", width=250, anchor="w", stretch=True)
        self._files.column("folder", width=120, anchor="w", stretch=False)
        self._files.column("size", width=70, anchor="e", stretch=False)
        self._files.column("state", width=110, anchor="w", stretch=False)
        scroll = AutoScrollbar(table, command=self._files.yview)
        self._files.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._fill_files()

    def _on_profile_change(self, _event=None):
        profile = self._profile_map.get(self.profile_var.get())
        if profile is None:
            return
        self._profile = profile
        self._profile_auto = False
        self._refresh_fit_label()
        self._fill_files()

    def _classify(self, profile):
        """The fit tier for one profile, or ``None`` when hardware is unknown."""
        if self._hardware_unknown():
            return None
        return model_catalog.classify_profile(
            profile, self._hardware, engine_tuning.read(),
        )

    def _profile_option(self, profile):
        """A profile dropdown label annotated with its own fit tier."""
        rec = self._classify(profile)
        if rec is None:
            return profile["label"]
        return f"{profile['label']} \u2014 {rec['tier_label']}"

    def _fit_numbers(self, rec):
        """The memory figures for a classified profile."""
        tier = rec["tier"]
        if tier == "resident":
            return f"~{rec['need_vram_gb']:.1f} GB VRAM"
        if tier == "fp8":
            return f"~{rec['need_fp8_vram_gb']:.1f} GB VRAM in fp8"
        if tier == "stream":
            return (
                f"~{rec['need_vram_gb']:.1f} GB VRAM and "
                f"~{rec['need_ram_gb']:.1f} GB RAM"
            )
        if tier == "cpu":
            return f"~{rec['need_ram_gb']:.1f} GB RAM"
        return (
            f"~{rec['need_vram_gb']:.1f} GB VRAM or "
            f"~{rec['need_ram_gb']:.1f} GB RAM"
        )

    def _refresh_fit_label(self):
        rec = self._classify(self._profile)
        if rec is None:
            need = model_catalog.footprint(
                self._profile,
                engine_tuning.read().get("text_encoder_device") == "auto",
            )
            text = (
                "Hardware not detected. This profile needs about "
                f"{need['vram'] / 1e9:.1f} GB VRAM and "
                f"{need['ram'] / 1e9:.1f} GB RAM."
            )
        else:
            text = (
                f"{rec['tier_label']}. "
                + model_catalog.tier_detail(rec["tier"], {
                    "fp8_vram_gb": rec["need_fp8_vram_gb"],
                })
                + f" Needs {self._fit_numbers(rec)}."
            )
        self._fit_label.config(text=text)

    def _fill_files(self):
        jobs = model_catalog.jobs(self._profile)
        self._jobs = jobs
        self._existing = {}
        self._files.delete(*self._files.get_children())
        if model_catalog.is_remote(self._profile):
            job = jobs[0]
            self._files.insert(
                "", tk.END, values=(
                    job["label"],
                    job["repo"],
                    "Hugging Face",
                    human_size(job["size"]),
                    "Fetched on first use",
                ),
            )
            self.status.config(
                text=(
                    "No download now: the model row keeps the repo id and "
                    "diffusers fetches it when you first generate."
                ),
            )
            return
        present_count = 0
        force = bool(
            getattr(self, "_force_profile", None) and self._force_profile.get()
        )
        for job in jobs:
            info = model_catalog.find_existing(job)
            if info and info["ok"] and not (
                force and info["reason"] in ("kind", "variant")
            ):
                self._existing[job["slot"]] = info
                present_count += 1
                if info["reason"] == "exact":
                    state = "Already present"
                else:
                    state = "Reuse: " + os.path.basename(info["path"])
            else:
                state = (
                    "Present, wrong kind"
                    if info and not info["ok"] else "Will download"
                )
            self._files.insert(
                "", tk.END, values=(
                    job["label"],
                    job["name"],
                    os.path.basename(job["dest_dir"]),
                    human_size(job["size"]),
                    state,
                ),
            )
        total = sum(job["size"] for job in jobs
                    if job["slot"] not in self._existing)
        self.status.config(
            text=(
                f"{len(jobs)} file(s), {human_size(total)} to download"
                + (f"; {present_count} already available" if present_count else ".")
            ),
        )

    # -------------------------------------------------------------- step 3
    def _build_confirm(self):
        jobs = self._jobs or model_catalog.jobs(self._profile)
        remote = model_catalog.is_remote(self._profile)
        pending = [] if remote else [
            job for job in jobs if job["slot"] not in self._existing
        ]
        total = jobs[0]["size"] if remote else sum(
            job["size"] for job in pending
        )
        tk.Label(
            self.content,
            text=f"{self._selected['label']} \u2014 {self._profile['label']}",
            anchor="w",
        ).pack(fill=tk.X)
        if remote:
            summary_text = (
                f"No download now: {jobs[0]['repo']} is kept as a Hub repo "
                f"reference and diffusers fetches about {human_size(total)} "
                "on first use.\n"
                "Accept the model license and set HF_TOKEN for gated repos. "
                "The row is added to Model Settings; press Save there to keep "
                "it."
            )
        else:
            summary_text = (
                f"{len(pending)} file(s) will be downloaded "
                f"({human_size(total)}).\n"
                "Files already present are reused. The downloaded files are "
                "added to Model Settings when this finishes; press Save there "
                "to keep them."
            )
        summary = tk.Label(
            self.content, text=summary_text,
            anchor="w", justify="left", wraplength=620,
        )
        theme.register(summary, fg="muted")
        summary.pack(fill=tk.X, pady=(4, 8))
        short = [] if remote else _short_disk_folders(pending)
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
        remote = model_catalog.is_remote(self._profile)
        jobs = [] if remote else [
            job for job in self._jobs if job["slot"] not in self._existing
        ]
        short = [] if remote else _short_disk_folders(jobs)
        if short:
            self._result.config(
                text="Not enough free space in "
                + ", ".join(os.path.basename(folder) for folder, _s, _f in short)
                + ". Free some space or choose a smaller profile.",
            )
            theme.register(self._result, fg="error")
            return
        self._downloading = True
        self._refresh_footer()
        self.status.config(text="")
        results = {}
        if remote:
            results["model"] = self._jobs[0]["name"]
        for slot, info in self._existing.items():
            results[slot] = info["path"]
        if jobs:
            window = _DownloadWindow(
                self.window, jobs, title="Downloading model files",
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
            job for job in self._jobs
            if not job.get("remote") and job["slot"] not in results
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

        warnings = []
        for job in self._jobs:
            if job.get("remote"):
                continue
            spec = self._job_spec(job)
            detected, ok = model_catalog.verify(results[job["slot"]], spec)
            if not ok:
                expected = engine_kinds.kind_label(job["kind"]) or job["kind"]
                found = engine_kinds.kind_label(detected) or (detected or "unknown")
                warnings.append(
                    f"{job['label']}: expected {expected}, detected {found}."
                )

        components = {
            job["slot"]: results[job["slot"]]
            for job in self._jobs if job["slot"] != "model"
        }
        payload = {
            "family": self._selected["family"],
            "group": self._selected["group"],
            "label": self._selected["label"],
            "path": results["model"],
            "components": components,
            "warnings": warnings,
        }
        self._done = True
        self._payload = payload
        self._warnings = warnings
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

    def _job_spec(self, job):
        """The catalog file spec behind a job (for expected-kind checks)."""
        for spec in self._profile["files"].values():
            if spec["slot"] == job["slot"]:
                return spec
        return {"kind": job["kind"]}

    def _show_result_text(self):
        remote = bool(
            self._profile and model_catalog.is_remote(self._profile)
        )
        if self._warnings:
            text = (
                "Downloaded, but the file kind did not match the expected one:\n"
                + "\n".join("  \u00b7 " + line for line in self._warnings)
                + "\nYou can still review the assignment in Model Settings."
            )
            theme.register(self._result, fg="error")
        elif remote:
            text = (
                "Added the Hub repo reference. The model row is ready in Model "
                "Settings; the weights download on first use."
            )
            theme.register(self._result, fg="text")
        else:
            text = "Downloaded and verified. The model row is ready in Model Settings."
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
