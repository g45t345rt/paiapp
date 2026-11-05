"""Persisted application settings (``settings.json``).

One JSON file holds every preference that outlives a session: the theme, panel
layout, text-picker history and the model locations. Both the engine (model
resolution, see :mod:`engine.paths`) and the UI read and write through here, so
the file location and parsing live in a single place.

The file is per-user, under ``%APPDATA%\\Pai`` on Windows and
``~/.config/Pai`` elsewhere.
"""

import json
import os

APP_DIR_NAME = "Pai"
FILE_NAME = "settings.json"


def app_dir():
    """The per-user app folder that holds the settings file and defaults.

    ``%APPDATA%\\Pai`` on Windows, ``~/.config/Pai`` elsewhere.
    """
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, APP_DIR_NAME)


def settings_path():
    return os.path.join(app_dir(), FILE_NAME)


def read():
    """Return the persisted settings dict, or ``{}`` when none is stored."""
    try:
        with open(settings_path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write(settings):
    """Persist ``settings`` (a dict) to the settings file."""
    path = settings_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)
    except OSError:
        pass


def section(name):
    """Return a copy of the ``name`` section as a dict (``{}`` when absent)."""
    value = read().get(name)
    return dict(value) if isinstance(value, dict) else {}


def set_section(name, values):
    """Replace the ``name`` section with ``values`` (``None`` removes it)."""
    settings = read()
    if values:
        settings[name] = values
    else:
        settings.pop(name, None)
    write(settings)
