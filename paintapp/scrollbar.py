"""A scrollbar whose arrow buttons are always visible.

Works in both vertical and horizontal orientations. Windows draws the native
``tkinter.Scrollbar`` control itself, and current versions render it without the
up/down arrow buttons users expect. This widget draws its own permanent arrows,
trough, and thumb, and speaks the same ``command`` / ``set`` protocol as
``tkinter.Scrollbar`` so it can replace one directly.

Colors come from :mod:`paintapp.theme` and follow theme changes; callers can
still pin individual colors through the ``*color`` options.
"""

import math
import tkinter as tk

from . import theme

_TOKENS = {
    "trough": "trough",
    "thumb": "border",
    "thumb_active": "accent",
    "button": "trough",
    "button_active": "button_active",
    "arrow": "muted",
    "arrow_active": "accent",
}

MIN_THUMB = 24
REPEAT_DELAY = 300
REPEAT_INTERVAL = 50

START = "start"
END = "end"
THUMB = "thumb"
TROUGH = "trough"


class ArrowScrollbar(tk.Canvas):
    """A scrollbar that always draws its arrow buttons.

    ``orient`` selects ``tk.VERTICAL`` (the default) or ``tk.HORIZONTAL``.
    ``width`` is the thickness in pixels for both orientations, matching
    ``tkinter.Scrollbar``.
    """

    def __init__(
        self, parent, command=None, orient=tk.VERTICAL, width=12,
        troughcolor=None, background=None, activebackground=None,
        buttoncolor=None, activebutton=None, arrowcolor=None,
        activearrow=None, **kwargs,
    ):
        super().__init__(
            parent, highlightthickness=0, bd=0, takefocus=0, **kwargs
        )
        self._orient = (
            orient if orient in (tk.VERTICAL, tk.HORIZONTAL) else tk.VERTICAL
        )
        self._vertical = self._orient == tk.VERTICAL
        if self._vertical:
            self.configure(width=width)
        else:
            self.configure(height=width)
        self._command = command
        self._width = width
        self._overrides = {
            "trough": troughcolor,
            "thumb": background,
            "thumb_active": activebackground,
            "button": buttoncolor,
            "button_active": activebutton,
            "arrow": arrowcolor,
            "arrow_active": activearrow,
        }
        self._first = 0.0
        self._last = 1.0
        self._drag = None
        self._active = None
        self._hover = None
        self._repeat = None
        self._apply_theme()
        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_motion)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Motion>", self._on_hover)
        self.bind("<Leave>", self._on_leave)
        theme.subscribe(self._apply_theme)

    def set(self, first, last):
        """Update the thumb fractions (the ``yscrollcommand`` entry point)."""
        self._first = max(0.0, min(1.0, float(first)))
        self._last = max(0.0, min(1.0, float(last)))
        if self._last < self._first:
            self._last = self._first
        self._redraw()

    def get(self):
        return self._first, self._last

    def set_command(self, command):
        self._command = command

    def _apply_theme(self):
        if not self.winfo_exists():
            return
        for key, token in _TOKENS.items():
            value = self._overrides.get(key) or theme.color(token)
            if value is None:
                continue
            setattr(self, "_" + key, value)
        background = getattr(self, "_trough", None)
        if background is not None:
            try:
                self.configure(bg=background)
            except tk.TclError:
                return
        self._redraw()

    # ---------------------------------------------------------- geometry
    def _length(self):
        """Size along the scrolling axis."""
        return self.winfo_height() if self._vertical else self.winfo_width()

    def _thickness(self):
        """Size across the scrolling axis."""
        return self.winfo_width() if self._vertical else self.winfo_height()

    def _arrow_size(self):
        length = self._length()
        thickness = self._thickness() or self._width
        if length <= 1:
            return max(6, thickness)
        return max(6, min(thickness, length // 2))

    def _track(self):
        size = self._arrow_size()
        return size, max(size, self._length() - size)

    def _thumb_span(self):
        top, bottom = self._track()
        span = bottom - top
        # A fixed minimum would swallow the whole track on a short bar (such as
        # a three-line text box), leaving the thumb the same size no matter how
        # much content it holds. Scale the floor down with the track so the
        # thumb still reflects how much is hidden, while long bars keep the
        # comfortable grabbable minimum.
        minimum = max(4, min(MIN_THUMB, span // 4))
        first = top + self._first * span
        last = top + self._last * span
        if last - first < minimum:
            last = min(bottom, first + minimum)
            first = max(top, last - minimum)
        return first, last

    def _part_at(self, pos):
        size = self._arrow_size()
        if pos < size:
            return START
        if pos >= self._length() - size:
            return END
        top, bottom = self._thumb_span()
        if top <= pos <= bottom:
            return THUMB
        return TROUGH

    def _axis(self, event):
        return event.y if self._vertical else event.x

    # ------------------------------------------------------------ drawing
    def _redraw(self):
        if not self.winfo_exists():
            return
        self.delete("all")
        thickness = self._thickness() or self._width
        length = self._length()
        if length <= 1:
            return
        size = self._arrow_size()
        if self._vertical:
            self.create_rectangle(
                0, 0, thickness, length, fill=self._trough, outline=""
            )
        else:
            self.create_rectangle(
                0, 0, length, thickness, fill=self._trough, outline=""
            )
        top, bottom = self._thumb_span()
        highlighted = self._active == THUMB or self._hover == THUMB
        thumb = self._thumb_active if highlighted else self._thumb
        inset = max(2, thickness // 7)
        a0, a1 = top + 1, bottom - 1
        if a1 - a0 >= 4:
            if self._vertical:
                self._rounded_rect(
                    inset, a0, thickness - inset, a1, inset, fill=thumb
                )
            else:
                self._rounded_rect(
                    a0, inset, a1, thickness - inset, inset, fill=thumb
                )
        self._draw_arrow(thickness, 0, size, START)
        self._draw_arrow(thickness, length - size, length, END)

    def _rounded_rect(self, x0, y0, x1, y1, radius, **kwargs):
        radius = max(0, min(radius, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
        points = (
            x0 + radius, y0, x1 - radius, y0, x1, y0,
            x1, y0 + radius, x1, y1 - radius, x1, y1,
            x1 - radius, y1, x0 + radius, y1, x0, y1,
            x0, y1 - radius, x0, y0 + radius, x0, y0,
        )
        self.create_polygon(points, smooth=True, **kwargs)

    def _draw_arrow(self, thickness, top, bottom, direction):
        active = self._active == direction or self._hover == direction
        background = self._button_active if active else self._button
        color = self._arrow_active if active else self._arrow
        if background != self._trough or active:
            if self._vertical:
                self.create_rectangle(
                    0, top, thickness, bottom, fill=background, outline=""
                )
            else:
                self.create_rectangle(
                    top, 0, bottom, thickness, fill=background, outline=""
                )
        center = (top + bottom) / 2.0
        middle = thickness / 2.0
        half = max(3, min((thickness - 10) // 2, (bottom - top - 10) // 2))
        before = direction == START
        if self._vertical:
            if before:
                verts = (
                    (middle, center - half),
                    (middle + half, center + half),
                    (middle - half, center + half),
                )
            else:
                verts = (
                    (middle, center + half),
                    (middle - half, center - half),
                    (middle + half, center - half),
                )
        else:
            if before:
                verts = (
                    (center - half, middle),
                    (center + half, middle - half),
                    (center + half, middle + half),
                )
            else:
                verts = (
                    (center + half, middle),
                    (center - half, middle - half),
                    (center - half, middle + half),
                )
        self._rounded_polygon(verts, max(1.5, half * 0.55), fill=color, outline=color)

    def _rounded_polygon(self, verts, radius, **kwargs):
        points = []
        count = len(verts)
        for index, current in enumerate(verts):
            previous = verts[index - 1]
            following = verts[(index + 1) % count]
            to_prev = (previous[0] - current[0], previous[1] - current[1])
            to_next = (following[0] - current[0], following[1] - current[1])
            len_prev = math.hypot(*to_prev) or 1.0
            len_next = math.hypot(*to_next) or 1.0
            corner = min(radius, len_prev / 2.0, len_next / 2.0)
            points.append((
                current[0] + to_prev[0] / len_prev * corner,
                current[1] + to_prev[1] / len_prev * corner,
            ))
            points.append(current)
            points.append((
                current[0] + to_next[0] / len_next * corner,
                current[1] + to_next[1] / len_next * corner,
            ))
        self.create_polygon(
            [value for point in points for value in point], smooth=True, **kwargs
        )

    def _scroll(self, *args):
        if self._command is not None:
            self._command(*args)

    # ------------------------------------------------------------- events
    def _on_press(self, event):
        pos = self._axis(event)
        part = self._part_at(pos)
        self._active = part
        if part == START:
            self._scroll("scroll", -1, "units")
            self._start_repeat(part)
        elif part == END:
            self._scroll("scroll", 1, "units")
            self._start_repeat(part)
        elif part == THUMB:
            top, bottom = self._thumb_span()
            self._drag = (pos, top, bottom)
        else:
            top, _bottom = self._thumb_span()
            self._scroll("scroll", -1 if pos < top else 1, "pages")
        self._redraw()

    def _on_motion(self, event):
        pos = self._axis(event)
        if self._drag is not None:
            start, top, _bottom = self._drag
            track_top, track_bottom = self._track()
            # ``first`` maps linearly across the whole trough (see
            # ``_thumb_span``), so a pixel drag converts to a fraction by
            # dividing by the trough length. Dividing by the remaining travel
            # (trough minus the drawn thumb) would make the thumb outrun the
            # cursor once the thumb is clamped to MIN_THUMB.
            span = track_bottom - track_top
            if span > 0:
                first = (top + (pos - start) - track_top) / span
                self._scroll("moveto", max(0.0, min(1.0, first)))
            return
        if self._active in (START, END) and self._part_at(pos) != self._active:
            self._stop_repeat()
            self._active = None
            self._redraw()

    def _on_hover(self, event):
        if self._drag is not None:
            return
        part = self._part_at(self._axis(event))
        if part != self._hover:
            self._hover = part
            self._redraw()

    def _on_release(self, _event):
        self._drag = None
        self._stop_repeat()
        self._active = None
        self._redraw()

    def _on_leave(self, _event):
        if self._drag is None:
            self._stop_repeat()
            if self._active is not None or self._hover is not None:
                self._active = None
                self._hover = None
                self._redraw()

    def _start_repeat(self, part):
        self._stop_repeat()
        self._repeat = self.after(REPEAT_DELAY, lambda: self._repeat_scroll(part))

    def _repeat_scroll(self, part):
        self._scroll("scroll", -1 if part == START else 1, "units")
        self._repeat = self.after(
            REPEAT_INTERVAL, lambda: self._repeat_scroll(part)
        )

    def _stop_repeat(self):
        if self._repeat is not None:
            self.after_cancel(self._repeat)
            self._repeat = None


class AutoScrollbar(ArrowScrollbar):
    """An ``ArrowScrollbar`` that only appears while there is more to see.

    Pack it with the layout options you want (for example ``side=tk.RIGHT,
    fill=tk.Y``) and wire it to a ``yscrollcommand`` as usual. While the linked
    widget shows its whole range the bar removes itself, so a text box whose
    content fits does not give up a column to an idle scrollbar. It reappears
    the moment the content overflows.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # A bare ``tk.Canvas`` requests a large default size (about 382x269).
        # In a container sized by its children that would inflate the row, the
        # linked widget would then show every line, the bar would hide, the row
        # would shrink and the overflow would return: an endless toggle. Keep
        # the request tiny so the linked widget sets the size and ``fill``
        # stretches the bar to match.
        if self._vertical:
            self.configure(height=1)
        else:
            self.configure(width=1)
        self._managed = False
        self._shown = False
        self._pack_args = ()
        self._pack_kwargs = {}

    def pack(self, *args, **kwargs):
        # Remember the layout so the bar can restore itself after hiding.
        super().pack(*args, **kwargs)
        self._managed = True
        self._shown = True
        self._pack_args = args
        self._pack_kwargs = kwargs
        self._sync()

    def set(self, first, last):
        super().set(first, last)
        self._sync()

    def _sync(self):
        if not self._managed:
            return
        overflow = (self._last - self._first) < 1.0 - 1e-6
        if overflow == self._shown:
            return
        if overflow:
            super().pack(*self._pack_args, **self._pack_kwargs)
        else:
            super().pack_forget()
        self._shown = overflow
