"""A reusable width/height input pair with an optional proportional lock.

Width and height are entered together in more than one place: the canvas size
dialog in :mod:`paintapp.canvasarea` and the generate panel. Those callers want
different fields (plain entries whose contents they validate themselves, versus
bounded spinboxes for a fixed range), so :class:`SizeInput` shares only what they
have in common: the two labelled fields, the aspect-ratio lock button, and the
value plumbing.
"""

import tkinter as tk

# Padlock glyphs; the lock button toggles between them.
LOCKED_GLYPH = "\U0001F512"
UNLOCKED_GLYPH = "\U0001F513"


class SizeInput(tk.Frame):
    """A width field and a height field linked by a ratio lock.

    ``width``/``height`` seed the fields. ``integer`` (the default) keeps whole
    numbers; pass ``False`` for fractional values. ``minimum``/``maximum`` clip
    edits and are optional: when both are ``None`` the caller keeps full control
    and can validate :meth:`values` itself. ``spinbox`` swaps the plain entries
    for ``tk.Spinbox`` fields. ``on_change`` is called with the current
    ``(width, height)`` after every edit.

    The lock starts engaged (pass ``locked=False`` to start unlocked). It
    captures the ratio in effect when it is engaged, so editing either field
    afterwards scales the other to match. Values pushed in with
    :meth:`set_values` (for example the generate panel's "Canvas" button) are
    applied verbatim rather than being forced back onto the ratio.
    """

    def __init__(
        self,
        parent,
        width=1024,
        height=1024,
        *,
        labels=("Width", "Height"),
        integer=True,
        minimum=None,
        maximum=None,
        spinbox=False,
        field_width=None,
        lockable=True,
        locked=True,
        on_change=None,
    ):
        super().__init__(parent)
        self._integer = integer
        self._minimum = minimum
        self._maximum = maximum
        self._has_bounds = minimum is not None or maximum is not None
        self._on_change = on_change
        self._syncing = False
        self._has_lock = bool(lockable)
        self._locked = bool(lockable and locked)
        self._ratio = None
        self._last = [width, height]

        self.width_var = self._make_var(width)
        self.height_var = self._make_var(height)

        if field_width is None:
            field_width = 6 if spinbox else 10
        self.width_field = self._make_field(self.width_var, spinbox, field_width)
        self.height_field = self._make_field(self.height_var, spinbox, field_width)

        self.lock_button = None
        if self._has_lock:
            self.lock_button = tk.Button(
                self, width=2,
                relief=tk.SUNKEN if self._locked else tk.RAISED,
                text=LOCKED_GLYPH if self._locked else UNLOCKED_GLYPH,
                command=self._toggle_lock,
            )
            if self._locked and height:
                self._ratio = width / height

        self._build(labels)

        self.width_var.trace_add("write", lambda *_: self._handle_edit(True))
        self.height_var.trace_add("write", lambda *_: self._handle_edit(False))
        self.width_field.bind("<FocusOut>", lambda _e: self._normalize(True))
        self.height_field.bind("<FocusOut>", lambda _e: self._normalize(False))
        self.width_field.bind("<Return>", lambda _e: self._normalize(True))
        self.height_field.bind("<Return>", lambda _e: self._normalize(False))

    # ------------------------------------------------------------------ public
    def values(self):
        """Return the current ``(width, height)``.

        Unparseable fields come back as ``None`` so callers that validate
        themselves can report their own message. When ``minimum``/``maximum``
        were given the numbers are clipped to that range and fall back to the
        last good value, so bounded callers always get usable integers.
        """
        width = self._read(self.width_var)
        height = self._read(self.height_var)
        if self._has_bounds:
            width = self._clamp(self._last[0] if width is None else width)
            height = self._clamp(self._last[1] if height is None else height)
        return width, height

    def set_values(self, width, height):
        """Replace both fields at once without triggering the ratio lock."""
        self._syncing = True
        try:
            self._write(self.width_var, width)
            self._write(self.height_var, height)
        finally:
            self._syncing = False
        self._last = [width, height]
        if self._has_lock and self._locked and height:
            self._ratio = width / height
        self._notify()

    def is_locked(self):
        """Whether the ratio lock is shown and currently engaged."""
        return self._has_lock and self._locked

    # ------------------------------------------------------------------- build
    def _build(self, labels):
        tk.Label(self, text=labels[0], anchor="w").pack(side=tk.LEFT)
        self.width_field.pack(side=tk.LEFT, padx=(4, 4))
        if self.lock_button is not None:
            self.lock_button.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(self, text=labels[1], anchor="w").pack(side=tk.LEFT)
        self.height_field.pack(side=tk.LEFT, padx=(4, 0))

    def _make_var(self, value):
        if self._integer:
            return tk.IntVar(master=self, value=int(round(value)))
        return tk.DoubleVar(master=self, value=float(value))

    def _make_field(self, var, spinbox, field_width):
        options = {"width": field_width, "textvariable": var}
        if spinbox:
            if self._minimum is not None:
                options["from_"] = self._minimum
            if self._maximum is not None:
                options["to"] = self._maximum
            return tk.Spinbox(self, **options)
        return tk.Entry(self, **options)

    # ------------------------------------------------------------------ values
    def _read(self, var):
        try:
            return var.get()
        except (tk.TclError, ValueError):
            return None

    def _write(self, var, value):
        if self._integer:
            var.set(int(round(value)))
        else:
            var.set(float(value))

    def _clamp(self, value):
        if self._minimum is not None and value < self._minimum:
            value = self._minimum
        if self._maximum is not None and value > self._maximum:
            value = self._maximum
        return value

    def _notify(self):
        if self._on_change is not None:
            self._on_change(*self.values())

    # ------------------------------------------------------------------ events
    def _handle_edit(self, is_width):
        if self._syncing:
            return
        edited = self.width_var if is_width else self.height_var
        other = self.height_var if is_width else self.width_var
        value = self._read(edited)
        if value is not None:
            self._last[0 if is_width else 1] = value
            if self.is_locked() and self._ratio:
                target = value / self._ratio if is_width else value * self._ratio
                target = self._clamp(target)
                self._syncing = True
                try:
                    self._write(other, target)
                finally:
                    self._syncing = False
                self._last[1 if is_width else 0] = target
        self._notify()

    def _normalize(self, is_width):
        var = self.width_var if is_width else self.height_var
        index = 0 if is_width else 1
        value = self._read(var)
        if value is None:
            if self._has_bounds:
                self._write(var, self._clamp(self._last[index]))
            return
        clamped = self._clamp(value)
        self._last[index] = clamped
        if clamped != value:
            self._syncing = True
            try:
                self._write(var, clamped)
            finally:
                self._syncing = False
        self._handle_edit(is_width)

    def _toggle_lock(self):
        self._locked = not self._locked
        if self._locked:
            width = self._read(self.width_var)
            height = self._read(self.height_var)
            if width and height:
                self._ratio = width / height
            elif self._ratio is None:
                self._ratio = 1.0
            self.lock_button.config(text=LOCKED_GLYPH, relief=tk.SUNKEN)
        else:
            self.lock_button.config(text=UNLOCKED_GLYPH, relief=tk.RAISED)
