"""A ttk combobox whose drop-down widens to fit its longest value.

``ttk`` sizes a combobox's pop-down to the combobox itself, so long entries
(sampler names, model file names) are cut off even though the listbox knows
their full text. The width of the pop-down comes from the style's
``-postoffset`` option, not from the listbox, so each time a picker is posted
we measure its values and set the offset on that picker's style.
"""

import tkinter as tk
from tkinter import font as tkfont, ttk

# Extra room beyond the text itself: the vertical scrollbar that appears when
# the list is taller than the picker, plus the pop-down's border/padding.
_SCROLLBAR_PX = 22
_PADDING_PX = 12
# Never widen a drop-down by more than this, or a single pathological value
# would make the popup unusable.
_MAX_EXTRA_PX = 720


class AutoWidthCombobox(ttk.Combobox):
    """A ``ttk.Combobox`` whose pop-down is wide enough for every value."""

    def __init__(self, master=None, **kw):
        self._user_postcommand = None
        super().__init__(master, **kw)
        # Let tkinter register a Python callback, then keep the registered Tcl
        # command name; a bare Python callable cannot be passed to ``tk.call``.
        self._user_postcommand = self.cget("postcommand") or None
        super().configure({"postcommand": self._auto_width})

    def configure(self, cnf=None, **kw):
        # Keep a caller's postcommand (some pickers refill their values there)
        # and run it from our own hook instead of replacing it. Apply the
        # caller's value first so tkinter registers a Python callback.
        has_postcommand = (
            (isinstance(cnf, dict) and "postcommand" in cnf)
            or "postcommand" in kw
        )
        if not has_postcommand:
            return super().configure(cnf, **kw)
        result = super().configure(cnf, **kw)
        self._user_postcommand = self.cget("postcommand") or None
        super().configure({"postcommand": self._auto_width})
        return result

    config = configure

    def _auto_width(self):
        if self._user_postcommand:
            self.tk.call(self._user_postcommand)
        try:
            self._apply_width()
        except tk.TclError:
            # The pop-down may already be gone (a postcommand can close the
            # picker); a failed measurement must not stop it opening.
            pass

    def _apply_width(self):
        values = [str(value) for value in self.cget("values")]
        if not values:
            return
        try:
            font = tkfont.Font(root=self, font=self.cget("font"))
        except tk.TclError:
            font = tkfont.nametofont("TkTextFont")
        text_px = max(font.measure(value) for value in values)

        # The pop-down is placed at exactly ``winfo width`` of the picker plus
        # this offset, so measure the same width here (it is 1 while the widget
        # is still unmapped, which the offset then covers).
        combo_px = max(1, self.winfo_width())

        needed = text_px + _PADDING_PX
        if len(values) > int(self.cget("height")):
            needed += _SCROLLBAR_PX

        extra = max(0, min(needed - combo_px, _MAX_EXTRA_PX))
        xoff = 0
        # Keep the popup on screen when the picker sits near the right edge:
        # first shift it left by the shortfall, then, only if the screen itself
        # is too narrow, give up some width. Skip this off the primary monitor,
        # where rootx can lie outside the primary screen.
        screen = self.winfo_screenwidth()
        rootx = self.winfo_rootx()
        if 0 <= rootx < screen:
            margin = 8
            right_space = screen - (rootx + combo_px) - margin
            if extra > right_space:
                deficit = extra - right_space
                max_shift = max(0, rootx - margin)
                shift = min(deficit, max_shift)
                xoff = -shift
                extra = max(0, extra - (deficit - shift))
            extra = min(extra, max(0, screen - 2 * margin - combo_px))

        style = self.cget("style") or "TCombobox"
        self.tk.call("ttk::style", "configure", style,
                     "-postoffset", (xoff, 0, extra, 0))


_installed = False


def install():
    """Make every later ``ttk.Combobox`` widen its drop-down."""
    global _installed
    if _installed:
        return
    _installed = True
    ttk.Combobox = AutoWidthCombobox
