"""Unit tests for the symmetry guides (run: .venv\\Scripts\\python.exe tests\\test_symmetry.py)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import symmetry  # noqa: E402

FAILED = []

SIZE = (100, 80)
CX, CY = 50.0, 40.0


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def close(name, got, want, tol=1e-6):
    if abs(got - want) > tol:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_off_is_identity():
    check("normalize off -> None", symmetry.normalize({"mode": "off"}), None)
    check("off maps are identity only", len(symmetry.transforms(None, SIZE)), 1)
    check("off identity holds", symmetry.transforms(None, SIZE)[0](3, 4), (3, 4))


def test_bad_mode_dropped():
    check("bogus mode -> None", symmetry.normalize({"mode": "z"}), None)
    check("non-dict -> None", symmetry.normalize("nope"), None)


def test_vertical_flips_x():
    maps = symmetry.transforms({"mode": "vertical"}, SIZE)
    check("vertical has two maps", len(maps), 2)
    x, y = maps[1](10, 20)
    close("vertical x mirrored", x, 2 * CX - 10)
    close("vertical y kept", y, 20)


def test_horizontal_flips_y():
    maps = symmetry.transforms({"mode": "horizontal"}, SIZE)
    check("horizontal has two maps", len(maps), 2)
    x, y = maps[1](10, 20)
    close("horizontal x kept", x, 10)
    close("horizontal y mirrored", y, 2 * CY - 20)


def test_both_is_four_way():
    maps = symmetry.transforms({"mode": "both"}, SIZE)
    check("both has four maps", len(maps), 4)
    images = {tuple(round(c) for c in m(10, 20)) for m in maps}
    check(
        "both covers four corners",
        images,
        {(10, 20), (90, 20), (10, 60), (90, 60)},
    )


def test_radial_rotates_about_center():
    maps = symmetry.transforms({"mode": "radial", "count": 4, "mirror": False}, SIZE)
    check("radial count maps", len(maps), 4)
    # A point one unit to the right of centre rotates to one unit below it.
    x, y = maps[1](CX + 10, CY)
    close("radial quarter-turn x", x, CX, 1e-6)
    close("radial quarter-turn y", y, CY + 10, 1e-6)


def test_radial_mirror_doubles():
    plain = symmetry.transforms({"mode": "radial", "count": 6, "mirror": False}, SIZE)
    mirrored = symmetry.transforms({"mode": "radial", "count": 6, "mirror": True}, SIZE)
    check("radial plain count", len(plain), 6)
    check("radial mirrored count", len(mirrored), 12)


def test_radial_count_clamped():
    check("count clamped high", symmetry.normalize(
        {"mode": "radial", "count": 99})["count"], symmetry.RADIAL_MAX)
    check("count clamped low", symmetry.normalize(
        {"mode": "radial", "count": 0})["count"], symmetry.RADIAL_MIN)


def test_axes_vertical():
    lines = symmetry.axes({"mode": "vertical"}, SIZE)
    check("vertical one line", len(lines), 1)
    (x0, y0), (x1, y1) = lines[0]
    close("vertical axis at center x", x0, CX)
    check("vertical axis spans height", (y0, y1), (0.0, float(SIZE[1])))


def test_axes_both_and_radial():
    check("both two lines", len(symmetry.axes({"mode": "both"}, SIZE)), 2)
    radial = symmetry.axes({"mode": "radial", "count": 6, "mirror": True}, SIZE)
    check("radial line count", len(radial), 6)
    check("off has no axes", symmetry.axes({"mode": "off"}, SIZE), [])


def test_transforms_are_symmetric_involutions():
    # Each mirror mapping applied twice returns the original point.
    for mode in ("vertical", "horizontal", "both"):
        for mapping in symmetry.transforms({"mode": mode}, SIZE):
            x, y = mapping(13.5, 61.25)
            bx, by = mapping(x, y)
            close(f"{mode} involution x", bx, 13.5)
            close(f"{mode} involution y", by, 61.25)


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
