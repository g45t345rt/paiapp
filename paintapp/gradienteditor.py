"""Editable multi-stop gradient strip widget for the Gradient tool panel.

The strip shows a live preview of the gradient (over a checkerboard so alpha is
visible) with draggable stop markers. The host owns the stop data as a list of
``[position, "#rrggbb", alpha]`` rows and rebuilds the widget's copy through
:meth:`set_stops`.
"""

import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from . import gradient, theme
from .constants import rgba


class StopsEditor(tk.Frame):
    """A compact gradient strip with draggable colour stops.

    ``stops`` is a list of ``[pos, "#rrggbb", alpha]`` (``pos`` in ``0..1``,
    ``alpha`` in ``0..255``). ``on_change`` fires after any edit so the host can
    mirror the state; ``on_pick_color`` opens the app colour picker for the
    selected stop and must return the chosen ``#rrggbb`` (or ``None``).
    """

    STRIP_H = 28
    MARKER_H = 9
    MARGIN = 6
    HIT = 9

    def __init__(self, parent, stops, on_change=None, on_pick_color=None):
        super().__init__(parent)
        self._on_change = on_change
        self._on_pick_color = on_pick_color
        self._stops = gradient.normalize_stops(stops) or [
            [0.0, "#000000", 255], [1.0, "#ffffff", 255],
        ]
        self._selected = 0
        self._dragging = False
        self._photo = None
        self.canvas = tk.Canvas(
            self, height=self.STRIP_H + self.MARKER_H,
            highlightthickness=0, bd=0, bg=theme.color("bg"), takefocus=1,
        )
        theme.skip(self.canvas)
        self.canvas.pack(fill=tk.X)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double)
        # Delete removes the selected stop while the strip has focus; the
        # "break" keeps the global Delete shortcut from also acting on a layer.
        self.canvas.bind("<Delete>", self._on_delete_key)
        self.canvas.bind("<BackSpace>", self._on_delete_key)

    # ------------------------------------------------------------------ state
    def get_stops(self):
        """A deep copy of the current stops, ordered by position."""
        return [list(stop) for stop in self._stops]

    def selected_index(self):
        return self._selected

    def selected_stop(self):
        if 0 <= self._selected < len(self._stops):
            return list(self._stops[self._selected])
        return None

    def set_stops(self, stops):
        """Replace every stop without firing ``on_change``."""
        normalized = gradient.normalize_stops(stops)
        self._stops = normalized or [[0.0, "#000000", 255], [1.0, "#ffffff", 255]]
        self._selected = min(self._selected, len(self._stops) - 1)
        self._redraw()

    def set_selected(self, index):
        self._selected = max(0, min(index, len(self._stops) - 1))
        self._redraw()

    def set_selected_color(self, color):
        if not self._stops:
            return
        self._stops[self._selected][1] = str(color)
        self._emit()

    def set_selected_alpha(self, alpha):
        if not self._stops:
            return
        self._stops[self._selected][2] = max(0, min(255, int(alpha)))
        self._emit()

    def set_selected_pos(self, pos):
        if not self._stops:
            return
        stop = self._stops[self._selected]
        stop[0] = max(0.0, min(1.0, float(pos)))
        self._stops.sort(key=lambda row: row[0])
        self._selected = self._stops.index(stop)
        self._emit()

    def add_stop(self):
        """Insert a stop in the widest gap, carrying the local colour."""
        stops = self._stops
        if len(stops) >= 2:
            gaps = [
                (b[0] - a[0], (a[0] + b[0]) / 2.0)
                for a, b in zip(stops, stops[1:])
            ]
            _gap, pos = max(gaps)
        else:
            pos = 0.5
        color, alpha = self._color_at(pos)
        stop = [pos, color, alpha]
        stops.append(stop)
        stops.sort(key=lambda row: row[0])
        self._selected = stops.index(stop)
        self._emit()

    def delete_selected(self):
        """Remove the selected stop, keeping at least two."""
        if len(self._stops) <= 2:
            return
        del self._stops[self._selected]
        self._selected = max(0, min(self._selected, len(self._stops) - 1))
        self._emit()

    # ----------------------------------------------------------------- events
    def _on_press(self, event):
        index = self._nearest(event.x)
        if index is None:
            return
        self.canvas.focus_set()
        self._selected = index
        self._dragging = True
        self._emit()

    def _on_delete_key(self, _event):
        """Delete the selected stop, keeping the global Delete off a layer."""
        self.delete_selected()
        return "break"

    def _on_drag(self, event):
        if not self._dragging or not self._stops:
            return
        width = max(1, self.canvas.winfo_width())
        pos = self._pos_from_x(event.x, width)
        stop = self._stops[self._selected]
        stop[0] = pos
        self._stops.sort(key=lambda row: row[0])
        self._selected = self._stops.index(stop)
        self._emit()

    def _on_release(self, _event):
        self._dragging = False

    def _on_double(self, event):
        index = self._nearest(event.x)
        if index is not None:
            self._selected = index
            if self._on_pick_color is not None:
                chosen = self._on_pick_color(self._stops[index][1])
                if chosen is not None:
                    self.set_selected_color(chosen)
            return
        width = max(1, self.canvas.winfo_width())
        pos = self._pos_from_x(event.x, width)
        color, alpha = self._color_at(pos)
        stop = [pos, color, alpha]
        self._stops.append(stop)
        self._stops.sort(key=lambda row: row[0])
        self._selected = self._stops.index(stop)
        self._emit()

    # --------------------------------------------------------------- geometry
    def _usable(self, width):
        return max(1, width - 2 * self.MARGIN)

    def _stop_x(self, pos, width):
        return self.MARGIN + pos * self._usable(width)

    def _pos_from_x(self, x, width):
        return max(0.0, min(1.0, (x - self.MARGIN) / self._usable(width)))

    def _nearest(self, x):
        width = max(1, self.canvas.winfo_width())
        best, best_distance = None, self.HIT + 1
        for index, stop in enumerate(self._stops):
            distance = abs(self._stop_x(stop[0], width) - x)
            if distance <= self.HIT and distance < best_distance:
                best, best_distance = index, distance
        return best

    def _color_at(self, pos):
        """Interpolate the (colour, alpha) of the current stops at ``pos``."""
        stops = self._stops
        if not stops:
            return "#000000", 255
        if pos <= stops[0][0]:
            return stops[0][1], stops[0][2]
        if pos >= stops[-1][0]:
            return stops[-1][1], stops[-1][2]
        for left, right in zip(stops, stops[1:]):
            if left[0] <= pos <= right[0]:
                span = right[0] - left[0]
                frac = 0.0 if span <= 1e-9 else (pos - left[0]) / span
                rgb_a, rgb_b = rgba(left[1]), rgba(right[1])
                rgb = [
                    int(round(rgb_a[i] * (1 - frac) + rgb_b[i] * frac))
                    for i in range(3)
                ]
                alpha = int(round(left[2] * (1 - frac) + right[2] * frac))
                return "#%02x%02x%02x" % tuple(rgb), alpha
        return stops[-1][1], stops[-1][2]

    # --------------------------------------------------------------- drawing
    def _emit(self):
        self._redraw()
        if self._on_change is not None:
            self._on_change()

    def _redraw(self):
        width = max(2, self.canvas.winfo_width())
        height = self.STRIP_H + self.MARKER_H
        strip = gradient.strip_image(self._stops, width, self.STRIP_H)
        base = self._checkerboard(width, self.STRIP_H)
        base.paste(strip, (0, 0), strip)
        self._photo = ImageTk.PhotoImage(base)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        for index, stop in enumerate(self._stops):
            self._draw_marker(index, stop, width, height)

    def _draw_marker(self, index, stop, width, height):
        x = self._stop_x(stop[0], width)
        top, tip = self.STRIP_H, self.STRIP_H + self.MARKER_H
        outline = theme.color("accent") if index == self._selected else "#ffffff"
        border = theme.color("border")
        self.canvas.create_polygon(
            x - 4, top, x + 4, top, x, tip,
            fill=stop[1], outline=outline, width=1,
        )
        self.canvas.create_rectangle(
            x - 4, top - 2, x + 4, top, fill=stop[1], outline=border,
        )

    @staticmethod
    def _checkerboard(width, height, cell=6):
        xs = np.arange(width) // cell
        ys = np.arange(height) // cell
        grid = ((xs[None, :] + ys[:, None]) % 2).astype(np.uint8)
        values = np.where(grid == 1, 200, 255).astype(np.uint8)
        rgba = np.repeat(values[:, :, None], 3, axis=2)
        return Image.fromarray(rgba, "RGB")
