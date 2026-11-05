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
