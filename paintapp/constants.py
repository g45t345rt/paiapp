"""Shared layout constants and small color helpers."""

import sys

# The Alt/Option modifier as reported in a Tk ``event.state`` bitmask. On X11 it
# is Mod1 (0x8); on Windows Tk reports Alt as the 0x20000 bit and additionally
# sets Mod1 on *every* mouse event, so testing 0x8 there would make Alt-click
# indistinguishable from a plain click.
ALT_STATE_MASK = 0x20000 if sys.platform == "win32" else 0x8

APP_NAME = "Pai"
APP_VERSION = "1.0.1"

# Opened by the Help > Donate dialog. Point this at whatever tip jar you use
# (GitHub Sponsors, Ko-fi, PayPal, ...).
DONATE_URL = "https://ko-fi.com/paiapp"

# Opened by the Help > Contribute dialog: repo, issues, or a contributing guide.
CONTRIBUTE_URL = "https://github.com/g45t345rt/paiapp/blob/main/CONTRIBUTING.md"

# Opened by Help > Discord: the community server invite.
DISCORD_URL = "https://discord.gg/bU6RzjwxFX"

CANVAS_W = 800
CANVAS_H = 640

MAX_CANVAS_SIZE = 8192

MAX_BRUSH_SIZE = 2000

# The brush size slider only spans this far; larger sizes are still reachable
# by typing them into the size box, which allows up to MAX_BRUSH_SIZE.
BRUSH_SLIDER_MAX = 500

# The Path tool's fill outline is a visual edge width, kept well below the
# brush size range. Its slider and its entry box share this limit so a typed
# value can never exceed what the slider can show.
PATH_OUTLINE_MAX = 200

GENERATE_SIZES = {
    "1024 x 1024": (1024, 1024),
    "896 x 1152": (896, 1152),
    "1152 x 896": (1152, 896),
    "768 x 768": (768, 768),
}

WHEEL_SIZE = 120

# Width of the vertical hue strip shown next to the saturation/value square.
HUE_STRIP_W = 20

# How many recently used colors the picker remembers.
RECENT_COLORS_MAX = 12

SIDEBAR_W = 300
SIDEBAR_MIN_W = 150
SIDEBAR_SPLITTER_W = 5
# The docked canvas-gallery rail shown beside the editor, and how narrow a
# drag of its splitter may make it before it stops.
GALLERY_DOCK_W = 260
GALLERY_DOCK_MIN_W = 160
SIDEBAR_RESIZE_MARGIN = 48
SIDEBAR_SCROLLBAR_W = 12

SPLITTER_COLOR = "#e6e6e6"
SPLITTER_LINE = "#b0b0b0"
SPLITTER_HOVER_COLOR = "#d0e4f5"
SPLITTER_HOVER_LINE = "#0078d7"

SIDEBAR_HEADER_BG = "#dcdcdc"
SIDEBAR_HEADER_FG = "#333333"
SIDEBAR_HEADER_ACTIVE_BG = "#c8dcf0"
SIDEBAR_HEADER_H = 24
PANEL_FLOAT_MIN_W = 180
PANEL_FLOAT_MIN_H = 120
# Gap between a floating container's border and its header/content. The gutter
# it leaves is also the band a floating panel is resized from, so the corner
# grip fits inside it instead of overlapping the panel's buttons.
PANEL_FLOAT_PAD = 6
# Size of a floating panel's corner resize grip, in pixels.
SIDEBAR_GRIP = 6

# The Queue panel caps its job list at this height (in pixels); a longer queue
# scrolls inside the panel instead of pushing the rest of the rail down.
QUEUE_LIST_MAX_H = 200
PANEL_FLOAT_BORDER = "#7a7a7a"
SIDEBAR_GHOST_BLEND = "#eeeeee"

# Every sidebar panel, in the order they are first built. Each entry is
# ``(key, title)``; ``key`` is the stable id used by the layout, menus and the
# saved panel-side preference, while ``title`` is the header label.
PANELS = [
    ("color", "Color"),
    ("brushes", "Brushes"),
    ("tools", "Tools"),
    ("tool_settings", "Tool Settings"),
    ("layout", "Layout Control"),
    ("layers", "Layers"),
    ("assets", "Assets"),
    ("history", "History"),
    ("generate", "Generate"),
    ("references", "References"),
    ("controlnet", "ControlNet"),
    ("region_prompts", "Region Prompts"),
    ("loaded_models", "Loaded Models"),
    ("background", "Remove Background"),
    ("upscale", "Upscale"),
    ("queue", "Queue"),
    ("logs", "Logs"),
]

# Sidebar sides a panel can be docked to.
PANEL_SIDES = ("left", "right")

# Where each panel starts. Anything not listed starts on the right rail.
DEFAULT_PANEL_SIDES = {
    "color": "left", "tools": "left", "brushes": "left", "tool_settings": "left",
}

# Panels that start collapsed (header only); the rest start expanded.
DEFAULT_COLLAPSED_PANELS = {"brushes", "layout", "assets", "history", "generate", "references", "controlnet", "region_prompts", "loaded_models", "background", "upscale", "queue", "logs"}

# Panels that start tabbed together, as ``(active_key, [members])``. Members
# must end up on the same sidebar (see DEFAULT_PANEL_SIDES); otherwise the
# group is skipped and the panels stay separate.
DEFAULT_PANEL_GROUPS = [
    ("layers", ["layers", "history", "queue", "logs"]),
    ("generate", ["generate", "loaded_models"]),
    ("controlnet", ["controlnet", "region_prompts"]),
]

PALETTE_COLORS = [
    "#000000", "#ffffff", "#ff0000", "#ff8800",
    "#ffff00", "#00bb00", "#0066ff", "#8800ff",
    "#7f7f7f", "#a52a2a", "#00cccc", "#ff66cc",
    "#003366", "#808000", "#cccccc", "#ffc0cb",
]


def rgba(hex_color, alpha=255):
    """Turn ``#rgb``/``#rrggbb`` into an ``(r, g, b, alpha)`` tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
        alpha,
    )
