"""Paint app package.

The application is split into focused components so ``paint.py`` stays a thin
entry point:

``constants``   shared sizes, colors and the ``rgba`` helper
``document``    ``Layer`` and ``Canvas`` document models
``canvasarea``  the canvas tab strip and each document's rendered view
``tabstrip``    the scrollable canvas tab row with its overflow list
``colorpicker`` the HSV wheel/square, sliders, entries and palette
``drawing``     the tool selector with per-tool settings and the event controller
``textrender``  editable text objects: specs, rasterising and layout metrics
``fontpicker``  searchable font chooser with preview rows and favorites
``selection``   rectangular and lasso selection regions (mask + outline)
``history``     per-canvas undo/redo snapshot stacks
``historypanel`` the clickable history list panel
``logpanel``    the scrollable, severity-colored status log panel
``layers``      the layer list panel
``icons``       the vendored Tabler status-icon renderer
``generation``  the image generation panel
``removal``     the background removal panel
``modelsettings`` the model-location settings dialog
``settingsdialog`` the app Settings dialog (hotkeys, interface, canvas)
``shortcuts``   the remappable keyboard-shortcut registry and manager
``hubdialog``   the Hugging Face model browser and downloader
``jobqueue``    the serial queue of long-running jobs and its sidebar panel
``sidebar``     the scrollable, resizable sidebar rail
``ui``          the status bar and menu bar
``app``         the :class:`PaintApp` coordinator that wires it all together
"""

__all__ = ["PaintApp"]


def __getattr__(name):
    if name == "PaintApp":
        from .app import PaintApp

        return PaintApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
