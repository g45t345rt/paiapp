"""Checks for floating a tab group as a real window.

Tk cannot move a widget between toplevels, so a group floats as two synced
windows: the group frame draws the tab header and resize gutter, and its active
member is promoted to a companion window covering the body area. Building the
real sidebar on a Tk root proves the companion is a top-level window that tracks
the group's move and resize and switches with the active tab. Skipped (with a
note) when no display is available.

Run: .venv\\Scripts\\python.exe tests\\test_sidebar_group_float.py
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


class _Event:
    def __init__(self, x_root=0, y_root=0):
        self.x_root = x_root
        self.y_root = y_root


def _rect(widget):
    return (
        widget.winfo_rootx(), widget.winfo_rooty(),
        widget.winfo_width(), widget.winfo_height(),
    )


def test_group_floats_as_a_synced_companion():
    import tkinter as tk

    from paintapp.sidebar import Sidebar

    try:
        root = tk.Tk()
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"group float test skipped: {exc}")
        return
    try:
        root.geometry("900x600+40+40")
        body = tk.Frame(root, width=600, height=560)
        body.pack_propagate(False)
        body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = Sidebar(body, root, "right")
        sb.mount(row=0, column=0, sticky="ns")
        c1 = sb.add_panel("gen", "Generate")
        for i in range(30):
            tk.Label(c1, text=f"g{i}").pack(fill=tk.X)
        c2 = sb.add_panel("models", "Loaded Models")
        for i in range(30):
            tk.Label(c2, text=f"m{i}").pack(fill=tk.X)

        def flush(n=4):
            for _ in range(n):
                root.update_idletasks()
                root.update()

        flush()
        hg, hm = sb.host("gen"), sb.host("models")
        sb.merge_items(hm, hg)
        flush()
        group = sb.group_of("gen")

        sb.float_item(group)
        flush()
        host = group._managed_host
        check("group floats as a window", group._windowed, True)
        check("companion is the active member", host is sb.host(group.active), True)
        check("companion is its own toplevel", host.winfo_toplevel() is host, True)

        gx, gy, gw, gh = _rect(group)
        hx, hy, hw, hh = _rect(host)
        check("companion sits under the header", hy > gy, True)
        check("companion is inside the group width", hx >= gx and hx + hw <= gx + gw, True)
        check("companion is inside the group height", hy + hh <= gy + gh, True)

        # Moving the group moves the companion with it.
        group._drag_start(_Event(gx + 40, gy + 10))
        group._drag_motion(_Event(gx + 40 + 500, gy + 10 + 300))
        flush()
        group._drag_end(_Event())
        check("companion tracks the group move", (_rect(host)[0] - _rect(group)[0],
                                                   _rect(host)[1] - _rect(group)[1]),
              (hx - gx, hy - gy))
        # The opaque group frame must not cover the companion after a drag.
        top = root.winfo_containing(host.winfo_rootx() + host.winfo_width() // 2,
                                    host.winfo_rooty() + host.winfo_height() // 2)
        inside = False
        node = top
        while node is not None:
            if node is host:
                inside = True
                break
            node = getattr(node, "master", None)
        check("companion stays on top after a drag", inside, True)

        # Resizing the group resizes the companion to stay inside.
        start = _Event(group.winfo_rootx(), group.winfo_rooty())
        group._resize_start(start)
        group._resize_motion(_Event(start.x_root - 120, start.y_root - 160))
        flush()
        hx, hy, hw, hh = _rect(host)
        gx, gy, gw, gh = _rect(group)
        check("companion stays inside after resize",
              hx + hw <= gx + gw and hy + hh <= gy + gh, True)

        # Switching tabs swaps which member the companion shows.
        sb.set_group_active(group, "models")
        flush()
        check("companion follows the active tab", group._managed_host is hm, True)
        check("old companion is no longer a toplevel", hg.winfo_toplevel() is hg, False)

        # A short companion scrolls its own content.
        group._set_window_geometry("240x200+%d+%d" % (gx, gy))
        flush()
        hm._float_scroll_by(100000)
        flush()
        check("companion scrolls its content", hm._foffset > 0, True)

        # Docking returns both to ordinary frames.
        sb.dock_item(group)
        flush()
        check("group stops floating", group.floating, False)
        check("group is a frame again", group.winfo_toplevel() is group, False)
        check("companion released", group._managed_host, None)
        check("active member packed in group body",
              sb.host(group.active).winfo_manager(), "pack")
    finally:
        root.destroy()


def main():
    test_group_floats_as_a_synced_companion()
    for note in SKIPPED:
        print(f"skip  {note}")
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        return 1
    print("\nall sidebar group float checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
