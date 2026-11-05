"""Unit tests for the text tool's editable objects (run: .venv\\Scripts\\python.exe tests\\test_text.py)."""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import fonts, textrender  # noqa: E402
from paintapp.document import Layer  # noqa: E402

# Force a tiny font registry so the tests never scan the system font
# directories; textrender then falls back to Pillow's built-in face.
fonts._registry = {}

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_normalize_fills_defaults():
    spec = textrender.normalize({"text": "hi", "x": 5, "y": 6})
    check("size default", spec["size"], textrender.DEFAULT_SIZE)
    check("align default", spec["align"], "left")
    check("color default", spec["color"], "#000000")
    check("rotation default", spec["rotation"], 0.0)
    check("anchor kept", (spec["x"], spec["y"]), (5.0, 6.0))


def test_normalize_clamps_fields():
    spec = textrender.normalize({
        "text": "x", "size": 100000, "align": "bogus",
        "line_spacing": 99, "letter_spacing": 1e9,
    })
    check("size clamped", spec["size"], textrender.MAX_SIZE)
    check("align falls back", spec["align"], "left")
    check("leading clamped", spec["line_spacing"], 3.0)


def test_clone_is_independent():
    spec = textrender.new_spec("hello", 0, 0)
    copy = textrender.clone(spec)
    copy["text"] = "changed"
    check("original untouched", spec["text"], "hello")
    check("copy changed", copy["text"], "changed")


def test_word_left_and_right():
    text = "hello world"
    check("word_left mid-word", textrender.word_left(text, 8), 6)
    check("word_left at word start", textrender.word_left(text, 6), 0)
    check("word_left at start", textrender.word_left(text, 0), 0)
    check("word_right mid-word", textrender.word_right(text, 8), 11)
    check("word_right at end", textrender.word_right(text, 11), 11)
    check("word_right skips spaces", textrender.word_right(text, 5), 11)


def test_word_bounds():
    text = "hello, world!"
    check("word at start", textrender.word_bounds(text, 2), (0, 5))
    check("edge prefers left", textrender.word_bounds(text, 5), (0, 5))
    check("word after space", textrender.word_bounds(text, 8), (7, 12))
    blank = "   "
    check("punctuation run", textrender.word_bounds(blank, 1), (0, 3))


def test_line_bounds():
    text = "one\ntwo\nthree"
    check("first line", textrender.line_bounds(text, 1), (0, 3))
    check("second line", textrender.line_bounds(text, 5), (4, 7))
    check("last line", textrender.line_bounds(text, 10), (8, 13))
    check("on newline", textrender.line_bounds(text, 3), (0, 3))


def test_caret_vertical_moves_between_rows():
    spec = textrender.new_spec("ab\ncd", 0, 0, family="Default", size=40)
    start = textrender.caret_vertical(spec, 1, 1)
    check("moved to second row", start >= 3, True)
    back = textrender.caret_vertical(spec, start, -1)
    check("back to first row", 0 <= back < 3, True)
    check("no row above first", textrender.caret_vertical(spec, 1, -1), 1)


def test_caret_vertical_keeps_column_across_wrap():
    spec = textrender.new_spec(
        "alpha beta gamma delta", 0, 0, family="Default", size=40,
        width=120.0, wrap=True,
    )
    rows = textrender.visual_lines(spec)
    check("text wraps to several rows", len(rows) > 1, True)
    down = textrender.caret_vertical(spec, 0, 1)
    check("down reaches a later row", down >= rows[1]["start"], True)


def test_prefix_width_is_cached_and_stable():
    spec = textrender.new_spec("tracking", 0, 0, family="Default", size=30)
    font = textrender.load_font(spec)
    first = textrender._prefix_width(font, "track", 3.0)
    second = textrender._prefix_width(font, "track", 3.0)
    check("cache keeps width stable", round(first, 6), round(second, 6))


def test_tracking_ignores_combining_marks():
    spec = textrender.new_spec("x", 0, 0, family="Default", size=30)
    font = textrender.load_font(spec)
    base = textrender._prefix_width(font, "a", 0.0)
    marked = textrender._prefix_width(font, "a\u0301", 0.0)
    check("combining mark adds no tracking", round(marked - base, 4),
          round(textrender._char_width(font, "\u0301"), 4))
    spaced = textrender._prefix_width(font, "ab", 10.0)
    plain = textrender._prefix_width(font, "ab", 0.0)
    check("tracking applied between two glyphs", round(spaced - plain, 4), 10.0)


def test_text_rotation_origin_is_configurable():
    # The drag code pins the rotation origin by re-anchoring the block after
    # each angle change; a custom origin must stay fixed just as the centre did.
    spec = textrender.new_spec("Hi", 20, 30, family="Default", size=40)
    pivot_local = (5.0, 5.0)
    check("default pivot is the block centre",
          textrender.default_pivot_local(spec),
          ((textrender.local_rect(spec)[0] + textrender.local_rect(spec)[2]) / 2.0,
           (textrender.local_rect(spec)[1] + textrender.local_rect(spec)[3]) / 2.0))
    pivot_doc = textrender.pivot_point(spec, pivot_local)
    spec["rotation"] = 90.0
    rx, ry = textrender.rotate_point(math.radians(90.0), *pivot_local)
    spec["x"] = pivot_doc[0] - rx
    spec["y"] = pivot_doc[1] - ry
    moved = textrender.pivot_point(spec, pivot_local)
    check("rotating keeps the custom origin fixed",
          (round(moved[0], 3), round(moved[1], 3)),
          (round(pivot_doc[0], 3), round(pivot_doc[1], 3)))
    check("pivot handle is hittable",
          textrender.hit_handle(spec, moved[0], moved[1], 3.0, pivot_local),
          "pivot")


def test_paste_paints_pixels():
    layer = Layer("t", (128, 64), (0, 0, 0, 0))
    spec = textrender.new_spec("Hi", 8, 8, family="Default", size=40)
    textrender.paste(layer.image, spec, (0, 0))
    check("glyphs paint ink", layer.image.getchannel("A").getbbox() is not None, True)


def test_font_bucket_and_category():
    check("bold italic", fonts._bucket("Bold Italic"), "bolditalic")
    check("oblique", fonts._bucket("Oblique"), "italic")
    check("regular", fonts._bucket("Book"), "regular")
    check("mono", fonts.category("DejaVu Sans Mono"), "Mono")
    check("serif", fonts.category("Times New Roman"), "Serif")
    check("sans", fonts.category("Arial"), "Sans")


def test_layer_visibility_helper():
    from paintapp.drawing import DrawingController

    class Obj:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    check("visible layer", DrawingController._layer_visible(Obj(visible=True, group=None)), True)
    check("hidden layer", DrawingController._layer_visible(Obj(visible=False, group=None)), False)
    check(
        "hidden group hides member",
        DrawingController._layer_visible(Obj(visible=True, group=Obj(visible=False))),
        False,
    )
    check(
        "visible group keeps member",
        DrawingController._layer_visible(Obj(visible=True, group=Obj(visible=True))),
        True,
    )


def test_pick_text_searches_all_layers():
    from paintapp.document import Canvas
    from paintapp.drawing import DrawingController

    class FakeSelf:
        _text_at = staticmethod(DrawingController._text_at)
        _layer_visible = staticmethod(DrawingController._layer_visible)

    class FakeApp:
        pass

    doc = Canvas("c", (200, 200))
    bottom = doc.add_layer(name="bottom")
    top = doc.add_layer(name="top")
    bottom.texts.append(textrender.new_spec("bottom", 10, 10, size=40))
    top.texts.append(textrender.new_spec("top", 10, 10, size=40))

    fake = FakeSelf()
    fake.app = FakeApp()
    fake.app.current = doc

    hit = DrawingController._pick_text(fake, 12, 12)
    check("topmost layer wins", hit is not None and hit[0] is top, True)
    check("topmost index", hit[1], 0)

    top.visible = False
    hit = DrawingController._pick_text(fake, 12, 12)
    check("hidden layer skipped", hit is not None and hit[0] is bottom, True)

    check("miss returns none", DrawingController._pick_text(fake, 150, 150), None)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
