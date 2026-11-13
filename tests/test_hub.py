"""Unit tests for the Hugging Face file-name helpers and renamed downloads.

Run: .venv\\Scripts\\python.exe tests\\test_hub.py

No network is used: the download test replaces ``engine.hub.urlopen`` with a
small in-memory response, so the on-disk naming logic is checked offline.
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import hub  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_is_generic():
    check("model generic", hub.is_generic_filename("model.safetensors"), True)
    check("numbered generic", hub.is_generic_filename("pytorch_model_00001.bin"), True)
    check("named not generic", hub.is_generic_filename("birefnet.safetensors"), False)


def test_suggest_filename():
    check(
        "generic -> repo",
        hub.suggest_filename("ZhengPeng7/BiRefNet", "model.safetensors"),
        "BiRefNet.safetensors",
    )
    check(
        "keeps real name",
        hub.suggest_filename("a/b", "vae.safetensors"),
        "vae.safetensors",
    )
    check(
        "generic repo adds owner",
        hub.suggest_filename("foo/model", "model.safetensors"),
        "foo_model.safetensors",
    )
    check(
        "nested path uses repo, keeps ext",
        hub.suggest_filename("a/b", "unet/model.bin"),
        "b.bin",
    )


def test_safe_name():
    check("strips dirs", hub.safe_name("sub/dir/file.bin"), "file.bin")
    check("cleans illegal", hub.safe_name("a*b?c.bin"), "a_b_c.bin")
    check("empty", hub.safe_name("   "), "")


def test_unique_name():
    check("free", hub.unique_name("a.bin", []), "a.bin")
    check("collision", hub.unique_name("a.bin", ["A.BIN"]), "a (1).bin")
    check("second", hub.unique_name("a.bin", ["a.bin", "a (1).bin"]), "a (2).bin")


class _Response:
    """A minimal stand-in for the ``urlopen`` context manager."""

    def __init__(self, payload):
        self._payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size=-1):
        data, self._payload = self._payload, b""
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_rename():
    real = hub.urlopen
    hub.urlopen = lambda request, timeout=60: _Response(b"weights")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = hub.download_file(
                "a/b", "model.safetensors", tmp, name="Custom.safetensors",
            )
            check("renamed base", os.path.basename(path), "Custom.safetensors")
            check("renamed exists", os.path.isfile(path), True)
            with open(path, "rb") as handle:
                check("renamed bytes", handle.read(), b"weights")

            evil = hub.download_file(
                "a/b", "model.safetensors", tmp, name="../escape.safetensors",
            )
            check("no traversal", os.path.dirname(evil), os.path.normpath(tmp))
    finally:
        hub.urlopen = real


def test_is_archive():
    check("zip true", hub.is_archive("sam.zip"), True)
    check("onnx false", hub.is_archive("sam.onnx"), False)


def test_search_direction():
    """Descending omits the direction hint; ascending is only for Newest."""
    seen = []
    real = hub._get_json

    def fake(path, timeout=30):
        seen.append(path)
        return []

    hub._get_json = fake
    try:
        hub.search_models(sort="downloads", descending=True)
        check("descending never sends 1", "direction=1" in seen[-1], False)
        check("descending sends -1", "direction=-1" in seen[-1], True)
        try:
            hub.search_models(sort="downloads", descending=False)
        except hub.HubError:
            check("ascending downloads refused", True, True)
        else:
            check("ascending downloads refused", False, True)
        hub.search_models(sort="createdAt", descending=False)
        check("ascending newest sent", "direction=1" in seen[-1], True)
    finally:
        hub._get_json = real


def test_extract_archive():
    import zipfile

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = os.path.join(tmp, "sam_vit_b.zip")
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("encoder.onnx", b"E")
            archive.writestr("nested/decoder.onnx", b"D")
            archive.writestr("../escape.onnx", b"X")
        dest = os.path.join(tmp, "segmentation")
        members = hub.extract_archive(archive_path, dest, name="sam_vit_b.zip")
        names = sorted(os.path.basename(path) for path in members)
        # A "../" member is reduced to its base name, so it lands inside the
        # bundle folder rather than above it.
        check("extracted all", names, ["decoder.onnx", "encoder.onnx", "escape.onnx"])
        folder = os.path.dirname(members[0])
        check("folder named", os.path.basename(folder), "sam_vit_b")
        check("nothing outside", os.path.isfile(os.path.join(tmp, "escape.onnx")),
              False)


def test_extract_non_archive():
    with tempfile.TemporaryDirectory() as tmp:
        plain = os.path.join(tmp, "model.onnx")
        with open(plain, "w") as handle:
            handle.write("x")
        check("non-zip ignored", hub.extract_archive(plain, tmp), [])


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
