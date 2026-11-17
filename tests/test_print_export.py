"""Unit tests for print export (run: .venv\\Scripts\\python.exe tests\\test_print_export.py)."""

import io
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from paintapp import dialogs  # noqa: E402
from paintapp.app import PaintApp  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def fake_app():
    """A stand-in self: only the static crop-mark helper is used."""
    return SimpleNamespace(_draw_crop_marks=PaintApp._draw_crop_marks)


def print_options(**overrides):
    options = {
        "page": "a4",
        "page_mm": (100.0, 150.0),
        "dpi": 254,  # 10 px/mm, so millimetres map to round pixel counts
        "bleed_mm": 0.0,
        "crop_marks": False,
        "transparent": False,
        "resample": "lanczos",
        "background": "#ffffff",
        "format": "PNG",
    }
    options.update(overrides)
    return options


def test_dialog_constants():
    check("TIFF format present", "TIFF" in dialogs._EXPORT_FORMATS, True)
    check("TIFF not animatable", "TIFF" in dialogs._EXPORT_ANIMATABLE, False)
    check("CMYK formats", dialogs._EXPORT_CMYK_FORMATS, ("JPEG", "TIFF"))
    for pid in ("none", "a5", "b5", "a4", "b4", "letter", "custom"):
        check(f"page preset {pid}", pid in dialogs._EXPORT_PAGE_LABELS, True)
    check("A4 mm", dialogs._EXPORT_PAGE_MM["a4"], (210.0, 297.0))


def test_no_page_passthrough():
    art = Image.new("RGB", (40, 30), (10, 20, 30))
    out = PaintApp._compose_print_page(
        fake_app(), art, print_options(page="none", page_mm=None)
    )
    check("no page returns same image", out is art, True)


def test_page_fit_and_center():
    art = Image.new("RGB", (400, 200), (255, 0, 0))
    out = PaintApp._compose_print_page(fake_app(), art, print_options())
    # 100x150 mm at 10 px/mm -> trim 1000x1500; art fits by width -> 1000x500.
    check("page size", out.size, (1000, 1500))
    check("art centered", out.getpixel((500, 750)), (255, 0, 0))
    check("margin is matte", out.getpixel((500, 100)), (255, 255, 255))


def test_bleed_expands_page():
    art = Image.new("RGB", (100, 100), (0, 0, 0))
    out = PaintApp._compose_print_page(
        fake_app(), art, print_options(bleed_mm=2.0)
    )
    # 2 mm at 10 px/mm -> 20 px margin on every side.
    check("bleed page size", out.size, (1000 + 40, 1500 + 40))


def test_crop_marks_drawn():
    art = Image.new("RGB", (100, 100), (255, 255, 255))
    out = PaintApp._compose_print_page(
        fake_app(), art, print_options(bleed_mm=2.0, crop_marks=True)
    )
    check("crop mark ink in margin", out.getpixel((15, 20)), (0, 0, 0))
    plain = PaintApp._compose_print_page(
        fake_app(), art, print_options(bleed_mm=2.0, crop_marks=False)
    )
    check("no marks without the flag", plain.getpixel((15, 20)), (255, 255, 255))


def save_bytes(image, fmt, options):
    buffer = io.BytesIO()
    PaintApp._save_export(None, image, buffer, fmt, options)
    buffer.seek(0)
    return Image.open(buffer)


def metadata_options(**metadata):
    return {
        "quality": 90, "png": {}, "jpeg": {}, "webp": {}, "gif": {},
        "metadata": metadata,
    }


def test_metadata_roundtrip():
    options = metadata_options(
        title="A title", author="An artist", copyright="(c) 2026",
        description="A description", keywords="one, two", software="Pai",
        created="2026-10-30",
    )
    for fmt in ("PNG", "JPEG", "WEBP", "TIFF"):
        result = save_bytes(Image.new("RGB", (4, 4), (1, 2, 3)), fmt, options)
        if fmt == "PNG":
            check("PNG title", result.info.get("Title"), "A title")
            check("PNG keywords", result.info.get("Keywords"), "one, two")
        else:
            exif = result.getexif()
            check(f"{fmt} title", exif.get(269), "A title")
            check(f"{fmt} artist", exif.get(315), "An artist")
            check(f"{fmt} copyright", exif.get(33432), "(c) 2026")
            check(f"{fmt} date", exif.get(306), "2026:10:30 00:00:00")


def test_blank_metadata_is_omitted():
    result = save_bytes(
        Image.new("RGB", (4, 4)), "PNG",
        metadata_options(title="", author="   "),
    )
    check("blank title omitted", "Title" in result.info, False)
    check("blank author omitted", "Author" in result.info, False)


def test_cmyk_falls_back_to_rgb_for_png():
    result = save_bytes(Image.new("CMYK", (4, 4)),
                        "PNG", {"png": {}})
    check("PNG drops CMYK", result.mode, "RGB")


def test_cmyk_kept_for_tiff():
    result = save_bytes(Image.new("CMYK", (4, 4)), "TIFF", {})
    check("TIFF keeps CMYK", result.mode, "CMYK")


def test_cmyk_kept_for_jpeg():
    result = save_bytes(Image.new("CMYK", (4, 4)), "JPEG",
                        {"quality": 90, "jpeg": {}})
    check("JPEG keeps CMYK", result.mode, "CMYK")


def test_tiff_saves_rgba():
    result = save_bytes(Image.new("RGBA", (4, 4), (1, 2, 3, 128)), "TIFF", {})
    check("TIFF keeps alpha", result.mode, "RGBA")


def test_canvas_presets_match_print_pages():
    from paintapp import canvasarea

    sizes = {size for _label, size in canvasarea.CANVAS_PRESETS}
    for pid, (width_mm, height_mm) in dialogs._EXPORT_PAGE_MM.items():
        expected = (round(width_mm * 300 / 25.4), round(height_mm * 300 / 25.4))
        check(f"canvas preset for {pid}", expected in sizes, True)


def test_canvas_preset_metadata():
    from paintapp import canvasarea

    presets = canvasarea._CANVAS_PRESET_TABLE
    labels = [preset.label for preset in presets]
    sizes = {preset.size for preset in presets}
    check("preset labels unique", len(labels) == len(set(labels)), True)
    check("preset sizes unique", len(sizes) == len(presets), True)
    check("all presets have a ratio", all(preset.ratio for preset in presets), True)
    check("all presets have a note", all(preset.note for preset in presets), True)
    check("all presets fit the canvas limit",
          max(max(preset.size) for preset in presets) <= canvasarea.MAX_CANVAS_SIZE,
          True)
    check("B5 600 dpi preset", (4299, 6071) in sizes, True)
    check("manga web page 1600 preset", (1133, 1600) in sizes, True)
    check("A6 preset", (1240, 1748) in sizes, True)
    check("B6 preset", (1476, 2079) in sizes, True)
    check("Legal preset", (2550, 4200) in sizes, True)
    check("Tabloid preset", (3300, 5100) in sizes, True)
    check("US comic preset", (1988, 3056) in sizes, True)
    check("16:10 preset", (1920, 1200) in sizes, True)
    check("ultrawide preset", (3440, 1440) in sizes, True)
    check("social preset", (1080, 1350) in sizes, True)
    check("AI 768 preset", (768, 768) in sizes, True)
    check("8K preset", (7680, 4320) in sizes, True)
    check("A2 preset", (4961, 7016) in sizes, True)
    check("B3 preset", (4169, 5906) in sizes, True)
    check("ANSI C preset", (5100, 6600) in sizes, True)
    check("webtoon 690 preset", (690, 1280) in sizes, True)
    check("AI 512 preset", (512, 512) in sizes, True)
    check("preset count", len(presets) >= 100, True)
    check("print category present",
          "Print" in canvasarea.CANVAS_PRESET_CATEGORIES, True)
    check("photo category present",
          "Photo" in canvasarea.CANVAS_PRESET_CATEGORIES, True)
    check("manga category present",
          "Manga" in canvasarea.CANVAS_PRESET_CATEGORIES, True)
    check("web category removed",
          "Web" in canvasarea.CANVAS_PRESET_CATEGORIES, False)
    check("webtoon category present",
          "Webtoon" in canvasarea.CANVAS_PRESET_CATEGORIES, True)
    manga_labels = [
        preset.label for preset in presets
        if preset.label.startswith("Manga · ")
    ]
    check("manga category has print size",
          any("B5 print" in label for label in manga_labels), True)
    check("manga category has web size",
          any("Web page" in label for label in manga_labels), True)
    check("comic digest preset",
          "Comic · Digest · 1:1.55 · 300 dpi" in labels, True)
    check("duplicate half letter print removed",
          "Print · Half Letter · 1:1.55 · 300 dpi" in labels, False)
    check("default category is All",
          canvasarea.DEFAULT_CANVAS_CATEGORY, "All")
    check("preset category helper",
          canvasarea.preset_category("Print · A4 portrait · 1:1.41 · 300 dpi"),
          "Print")
    check("16:9 ratio", canvasarea.aspect_ratio_label(1920, 1080), "16:9")
    check("A4 ratio", canvasarea.aspect_ratio_label(2480, 3508), "1:1.41")
    check("mm ratio", canvasarea.aspect_ratio_label(215.9, 279.4), "1:1.29")
    check("extreme ratio", canvasarea.aspect_ratio_label(200, 300000), "1:1500")
    check("extreme inverse ratio", canvasarea.aspect_ratio_label(300000, 200), "1500:1")


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
