"""Per-canvas undo/redo history.

Each :class:`~paintapp.document.Canvas` owns a :class:`History`: a bounded
stack of snapshots of *that* canvas, so undo/redo always applies to the active
tab and editing one canvas never consumes another canvas's steps. The snapshot
and restore work lives on the canvas itself (:meth:`Canvas.snapshot` and
:meth:`Canvas.restore`); this module only holds the stack.

Snapshots are copy-on-write: a snapshot *references* each layer's image and
mask rather than copying it, and marks the layer shared. The layer forks (copies
its pixels) on its next in-place edit, so a snapshot only pays for the layers
that actually change afterwards. A push is therefore proportional to the edited
layer instead of the whole scene, and unchanged layers stay shared between
consecutive snapshots.
"""

DEFAULT_LIMIT = 30


class History:
    """A bounded stack of labeled canvas snapshots backing Undo/Redo.

    Each entry pairs a human-readable ``label`` (the action that produced the
    following state) with a snapshot of the canvas state before that action.
    """

    def __init__(self, limit=DEFAULT_LIMIT):
        self.limit = limit
        self._undo = []
        self._redo = []

    def clear(self):
        self._undo.clear()
        self._redo.clear()

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    @property
    def position(self):
        """Index of the current state in :meth:`timeline`."""
        return len(self._undo)

    def push(self, canvas, app, label="Edit"):
        """Record ``canvas``'s state before a mutation, dropping any redo tail."""
        self._undo.append({"label": label, "state": canvas.snapshot(app)})
        if len(self._undo) > self.limit:
            del self._undo[0]
        self._redo.clear()

    def push_action(self, label, undo, redo):
        """Record a custom, invertible action instead of a canvas snapshot.

        ``undo`` and ``redo`` are callables that operate on live app state. The
        action counts as one history step. Callers push the same action to each
        canvas it touches so undoing from either tab reverses the whole
        operation; the callbacks should be idempotent so duplicate entries are
        harmless.
        """
        self._undo.append({"label": label, "undo": undo, "redo": redo})
        if len(self._undo) > self.limit:
            del self._undo[0]
        self._redo.clear()

    @property
    def top_undo_is_action(self):
        """True when the next undo would run a custom action, not a restore."""
        return bool(self._undo) and "undo" in self._undo[-1]

    @property
    def top_redo_is_action(self):
        """True when the next redo would run a custom action, not a restore."""
        return bool(self._redo) and "undo" in self._redo[-1]

    def crosses_action(self, target_index):
        """True when moving to ``target_index`` in :meth:`timeline` runs a custom action.

        Used by the History panel to decide whether a jump rebuilds the canvas
        from snapshots (where generation prompts must be captured and re-applied)
        or runs custom actions that keep the layer objects themselves.
        """
        current = len(self._undo)
        if target_index < current:
            return any("undo" in entry for entry in self._undo[target_index:current])
        if target_index > current:
            steps = target_index - current
            # Redo pops from the end of ``_redo``; the timeline walks the same
            # stack in reverse, so the entries a jump crosses are its last ones.
            return any("undo" in entry for entry in self._redo[-steps:])
        return False

    def drop_last(self):
        """Forget the most recent snapshot (used to cancel a live edit)."""
        if self._undo:
            self._undo.pop()

    def undo(self, canvas, app):
        if not self._undo:
            return False
        entry = self._undo.pop()
        if "undo" in entry:
            # A custom action carries its own inverse; swapping the two
            # callbacks makes the same entry replayable as a redo.
            entry["undo"]()
            self._redo.append({
                "label": entry["label"],
                "undo": entry["redo"],
                "redo": entry["undo"],
            })
        else:
            self._redo.append({"label": entry["label"], "state": canvas.snapshot(app)})
            canvas.restore(entry["state"], app)
        return True

    def redo(self, canvas, app):
        if not self._redo:
            return False
        entry = self._redo.pop()
        if "undo" in entry:
            entry["undo"]()
            self._undo.append({
                "label": entry["label"],
                "undo": entry["redo"],
                "redo": entry["undo"],
            })
        else:
            self._undo.append({"label": entry["label"], "state": canvas.snapshot(app)})
            canvas.restore(entry["state"], app)
        return True

    def timeline(self):
        """Every reachable state, oldest first, as ``{"label", "current"}``."""
        rows = [{"label": "Start", "current": not self._undo}]
        for i in range(1, len(self._undo)):
            rows.append({"label": self._undo[i - 1]["label"], "current": False})
        if self._undo:
            rows.append({"label": self._undo[-1]["label"], "current": True})
        for entry in reversed(self._redo):
            rows.append({"label": entry["label"], "current": False})
        return rows

    def jump(self, canvas, app, index):
        """Move to the state at ``index`` in :meth:`timeline` via undo/redo."""
        if index < 0 or index >= len(self.timeline()):
            return False
        while len(self._undo) > index:
            self.undo(canvas, app)
        while len(self._undo) < index:
            self.redo(canvas, app)
        return True
