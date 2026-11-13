"""Entry point for the Paint app.

The application itself lives in the ``paintapp`` package; this module only
checks the environment and starts the Tk main loop.
"""

import faulthandler
import os
import sys


# A native crash (access violation) bypasses Python's traceback. faulthandler
# writes the faulting stack, so keep a log on disk under the app data -- a
# crash here is not reproducible headlessly, and the log names the stack.
CRASH_LOG = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")),
                         "pai", "faulthandler.log")
try:
    os.makedirs(os.path.dirname(CRASH_LOG), exist_ok=True)
    _crash_log = open(CRASH_LOG, "w")
    faulthandler.enable(_crash_log, all_threads=True)
    print(f"faulthandler log: {CRASH_LOG}", file=sys.stderr)
except OSError:
    _crash_log = None
    faulthandler.enable()


class _Tee:
    """Duplicate writes to the real stream and a log file (best effort)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


# Mirror stdout/stderr (the engine's [krea2]/[anima]/... diagnostics and any
# traceback) to a log so a run can be inspected after the fact. Native crashes
# are covered by the faulthandler log above; this covers everything Python-side.
LOG_PATH = os.path.join(os.path.dirname(CRASH_LOG), "pai.log")
try:
    _log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)
    print(f"log file: {LOG_PATH}", file=sys.stderr)
except OSError:
    _log_file = None

try:
    from PIL import Image  # noqa: F401
except ModuleNotFoundError:
    sys.exit(
        "This app needs Pillow, which your system Python does not have.\n"
        "Launch Pai through run.bat, which creates a private .venv and\n"
        "installs the dependencies, or bootstrap it directly:\n"
        "    python setup_env.py\n"
        "then run:\n"
        "    .venv\\Scripts\\python.exe paint.py"
    )

import tkinter as tk

from paintapp.app import PaintApp


def _set_app_user_model_id():
    """Give the process its own taskbar identity on Windows.

    Without an explicit AppUserModelID the shell groups the window under the
    launching interpreter (``python.exe``) and shows *its* icon on the taskbar,
    regardless of the icon the window sets. Must run before the first window
    is created. A no-op elsewhere.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Pai.PaintApp"
        )
    except Exception:
        pass


def _create_root():
    """Create the Tk root, with tkdnd loaded when it is installed.

    ``tkinterdnd2`` lets the window accept files dragged from the OS file
    manager. It is optional: when the package or its native tkdnd library is
    unavailable, a plain root is returned and the app starts as usual.
    """
    try:
        from tkinterdnd2 import TkinterDnD
    except Exception:
        return tk.Tk(className="Pai")
    try:
        return TkinterDnD.Tk(className="Pai")
    except Exception:
        return tk.Tk(className="Pai")


def main():
    _set_app_user_model_id()
    root = _create_root()
    PaintApp(root)
    root.mainloop()



if __name__ == "__main__":
    main()
