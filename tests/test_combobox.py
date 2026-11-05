"""Checks for the auto-widening combobox.

The drop-down of a ``ttk.Combobox`` is sized from the widget, so long values
(sampler names, model file names) are clipped. Building the real widget on a
withdrawn Tk root proves the pop-down is widened to the longest value and that
a picker's own ``postcommand`` still runs. Skipped (with a note) when no display
is available.

Run: .venv\\Scripts\\python.exe tests\\test_combobox.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILED = []
SKIPPED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _popdown_width(combo):
    combo.tk.call("ttk::combobox::Post", combo)
    combo.update_idletasks()
    popdown = combo.tk.call("ttk::combobox::PopdownWindow", combo)
    width = combo.tk.call("winfo", "width", popdown)
    combo.tk.call("ttk::combobox::Unpost", combo)
    return int(width)


def test_widens_dropdown():
    import tkinter as tk
    from tkinter import font as tkfont, ttk

    from paintapp import combobox

    try:
        combobox.install()
        root = tk.Tk()
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"combobox test skipped: {exc}")
        return
    try:
        # Keep the window mapped (an unmapped widget reports a width of 1) but
        # near the left edge, so the pop-down has room to grow.
        root.geometry("+40+40")
        root.deiconify()
        root.update()
        calls = []

        def postcommand():
            calls.append(1)

        values = [
            "euler_ancestral", "dpmpp_2m_cfg_pp",
            "dpmpp_sde_gpu_a_name_long_enough_to_be_clipped", "ddim",
        ]
        combo = ttk.Combobox(
            root, values=values, state="readonly", width=12,
            postcommand=postcommand,
        )
        combo.pack()
        root.update()

        font = tkfont.nametofont("TkTextFont")
        longest = max(font.measure(value) for value in values)
        check("popdown fits longest value", _popdown_width(combo) >= longest, True)
        check("postcommand still runs", calls, [1])

        # Setting a new postcommand must keep both it and the widening hook.
        combo.configure(postcommand=postcommand)
        check("reconfigured popdown fits", _popdown_width(combo) >= longest, True)
        check("reconfigured postcommand runs", calls, [1, 1])

        # Short values need no extra width: the pop-down matches the widget.
        short = ttk.Combobox(root, values=["a", "b", "c"], state="readonly", width=20)
        short.pack()
        root.update()
        check("short values leave width alone",
              _popdown_width(short), short.winfo_width())
    finally:
        root.destroy()


def test_shifts_left_at_screen_edge():
    import tkinter as tk
    from tkinter import font as tkfont, ttk

    from paintapp import combobox

    try:
        combobox.install()
        root = tk.Tk()
        root.update_idletasks()
        screen = root.winfo_screenwidth()
        if screen < 600:  # not a desktop-sized screen; skip the edge case
            root.destroy()
            SKIPPED.append("screen-edge test skipped: screen too small")
            return
        # A short window pinned near the right edge leaves no room to grow.
        root.geometry(f"300x120+{screen - 320}+80")
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"combobox test skipped: {exc}")
        return
    try:
        root.deiconify()
        values = [
            "dpmpp_sde_gpu_a_name_long_enough_to_be_clipped",
            "euler_ancestral",
        ]
        combo = ttk.Combobox(root, values=values, state="readonly", width=12)
        combo.pack(side="right", padx=4)
        root.update()

        combo.tk.call("ttk::combobox::Post", combo)
        root.update_idletasks()
        popdown = combo.tk.call("ttk::combobox::PopdownWindow", combo)
        left = int(combo.tk.call("winfo", "rootx", popdown))
        width = int(combo.tk.call("winfo", "width", popdown))
        combo.tk.call("ttk::combobox::Unpost", combo)

        longest = max(tkfont.nametofont("TkTextFont").measure(v) for v in values)
        check("edge popdown fits longest value", width >= longest, True)
        check("edge popdown stays on screen", left + width <= screen, True)
        check("edge popdown shifts left", left < combo.winfo_rootx(), True)
    finally:
        root.destroy()


def main():
    for test in (test_widens_dropdown, test_shifts_left_at_screen_edge):
        try:
            test()
        except Exception as exc:  # keep going so one failure is not fatal
            import traceback

            FAILED.append(f"{test.__name__} raised: {exc}\n{traceback.format_exc()}")

    for note in SKIPPED:
        print(f"skip {note}")
    if FAILED:
        print("\nFAILURES:")
        for line in FAILED:
            print(" -", line)
        return 1
    print("\nall combobox tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
