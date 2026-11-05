"""Cooperative cancellation for long-running engine jobs.

The UI creates one :class:`CancelToken` per job, hands it to an engine entry
point (``pipeline.generate`` or ``background.remove_background``) and calls
:meth:`CancelToken.cancel` from the Tk thread. The engine checks the token at
safe points and raises :class:`Cancelled` so partial work is discarded instead
of being applied to the document.
"""

import threading


class Cancelled(Exception):
    """Raised by an engine job that was cancelled through its token."""


class CancelToken:
    """A thread-safe cancellation flag shared between the UI and the engine.

    Besides a hard :meth:`cancel`, the token carries a *keep* request: the user
    can stop a run early and still keep the image as denoised so far. A keep is
    a graceful stop -- the sampler returns its current ``x0`` estimate instead of
    raising -- so the partial result is decoded and delivered like any other.
    """

    def __init__(self):
        self._event = threading.Event()
        self._keep = threading.Event()

    def cancel(self):
        self._event.set()

    def request_keep(self):
        """Ask the running sampler to stop early and keep its current estimate."""
        self._keep.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    @property
    def keep_requested(self):
        return self._keep.is_set()

    def throw_if_cancelled(self):
        if self._event.is_set():
            raise Cancelled()
