"""Regression test: the layer list shrinks when switching to a smaller canvas.

Run: .venv\\Scripts\\python.exe tests\\test_layer_panel_scroll.py

A layer list is a frame embedded in a scroll canvas. Tk keeps that embedded
window at the height it had when the list was last scrolled, so a return from a
canvas with many layers to one with few used to leave the list at its old
height: blank space after the remaining rows and a scrollbar still showing
overflow that no longer existed.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_list_shrinks_with_its_content():
    import tkinter as tk

    from paintapp import theme
    from paintapp.document import Canvas
    from paintapp.layers import LayerPanel

    try:
        root = tk.Tk()
        theme.init(root)
    except Exception as exc:  # no display, or Tk missing
        print(f"ok  layer list scroll (no display, skipped: {exc})")
        return
    try:
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        def callback(*_args, **_kwargs):
            pass

        panel = LayerPanel(frame, *([callback] * 13))
        panel.pack(fill=tk.BOTH, expand=True)
        root.update()

        big = Canvas("Big", (4, 4))
        for index in range(40):
            big.add_layer(name=f"Layer {index}")
        panel.refresh(big.layers, len(big.layers) - 1)
        root.update()
        panel.canvas.yview_moveto(1.0)
        root.update()
        check("big list overflows", panel.canvas.yview()[1], 1.0)
        check("big list is scrolled to the bottom",
              panel.canvas.yview()[0] > 0.0, True)

        small = Canvas("Small", (4, 4))
        for index in range(2):
            small.add_layer(name=f"Small {index}")
        panel.refresh(small.layers, 1)
        root.update()

        content = panel.inner.winfo_reqheight()
        check("embedded list shrank to its content",
              panel.inner.winfo_height(), content)
        check("scroll region matches the content",
              int(panel.canvas.cget("scrollregion").split()[-1]), content)
        check("short list shows its whole range", panel.canvas.yview(), (0.0, 1.0))

        panel.refresh(big.layers, len(big.layers) - 1)
        root.update()
        check("returning to the big list overflows again",
              panel.canvas.yview()[1] < 1.0, True)
        check("returning to the big list restores its height",
              panel.inner.winfo_height(), panel.inner.winfo_reqheight())
    finally:
        root.destroy()


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for failure in FAILED:
            print(f"    {failure}")
        raise SystemExit(1)
    print("\nall layer panel scroll tests passed")


if __name__ == "__main__":
    main()
