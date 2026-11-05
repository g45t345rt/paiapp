"""Unit tests for SAM ONNX point-prompted segmentation.

Run: .venv\\Scripts\\python.exe tests\\test_segmentation.py

No model file or ``onnxruntime`` is needed: the graph I/O is read from a tiny
stub session, and the encoder/decoder lookup is monkeypatched so the prompt,
preprocessing, mask-selection and caching logic can be exercised directly.
"""

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import segmentation  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def check_close(name, got, want, tol=1e-4):
    if abs(got - want) > tol:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class _IO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _Session:
    """A minimal session stand-in that records how many times it ran."""

    def __init__(self, inputs, outputs, produce):
        self._inputs = [_IO(n, s) for n, s in inputs]
        self._outputs = [_IO(n, s) for n, s in outputs]
        self._produce = produce
        self.runs = 0

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, names, feed):
        self.runs += 1
        return self._produce(names, feed)


def test_preprocess_geometry():
    array, scale = segmentation.preprocess(Image.new("RGB", (500, 250)))
    check("preprocess shape", array.shape, (1, 3, 1024, 1024))
    check_close("preprocess scale", scale, 1024 / 500.0)


def test_preprocess_pads_bottom_right():
    # A wide image is scaled on the long side; the top-left stays the origin.
    array, scale = segmentation.preprocess(Image.new("RGB", (2000, 1000)))
    check_close("wide scale", scale, 0.512)
    check("wide row count", array.shape[2], 1024)


def test_build_prompt_scales_points():
    coords, labels = segmentation.build_prompt([(10, 20), (5, 5)], None, 0.5)
    check("coords shape", coords.shape, (1, 2, 2))
    check_close("coords x0", coords[0, 0, 0], 5.0)
    check_close("coords y0", coords[0, 0, 1], 10.0)
    check("labels default", labels.tolist(), [[1.0, 1.0]])


def test_build_prompt_keeps_labels():
    coords, labels = segmentation.build_prompt([(1, 2)], [0], 2.0)
    check("labels explicit", labels.tolist(), [[0.0]])
    check_close("coords scaled", coords[0, 0, 0], 2.0)


def test_decoder_feed_fills_declared_inputs():
    session = _Session(
        [("image_embeddings", None), ("point_coords", None), ("point_labels", None),
         ("mask_input", None), ("has_mask_input", None), ("orig_im_size", None)],
        [("masks", None), ("iou_predictions", None), ("low_res_masks", None)],
        lambda names, feed: [],
    )
    embeddings = np.zeros((1, 256, 64, 64), "float32")
    coords = np.zeros((1, 1, 2), "float32")
    labels = np.ones((1, 1), "float32")
    feed = segmentation._decoder_feed(session, embeddings, coords, labels, (40, 60))
    check("feed keys", sorted(feed), sorted([
        "image_embeddings", "point_coords", "point_labels", "mask_input",
        "has_mask_input", "orig_im_size"]))
    # orig_im_size is the encoder frame (height, width).
    check("orig_im_size", feed["orig_im_size"].tolist(), [40.0, 60.0])


def test_decoder_feed_skips_absent_inputs():
    session = _Session(
        [("image_embeddings", None), ("point_coords", None)],
        [("masks", None)],
        lambda names, feed: [],
    )
    feed = segmentation._decoder_feed(
        session, np.zeros((1, 256, 64, 64), "float32"),
        np.zeros((1, 1, 2), "float32"), np.ones((1, 1), "float32"), (10, 10),
    )
    check("feed minimal", sorted(feed), ["image_embeddings", "point_coords"])


def test_mask_image_picks_best_iou():
    geometry = (4, 4, 4, 4, 4, 4)  # square doc: content fills the frame
    raw = np.zeros((1, 3, 4, 4), "float32")
    raw[0, 0, 0, 0] = 1.0  # candidate 0: a single pixel
    raw[0, 2, :, :] = 5.0  # candidate 2: the whole tile

    # The highest predicted IoU selects the candidate, not the largest mask.
    mask = segmentation._mask_image(
        raw, np.array([[0.9, 0.1, 0.2]], "float32"), geometry)
    check("best-by-iou sparse", mask.getextrema(), (0, 255))
    check("best-by-iou pixel", mask.getpixel((0, 0)), 255)

    mask = segmentation._mask_image(
        raw, np.array([[0.1, 0.2, 0.95]], "float32"), geometry)
    check("best-by-iou full", mask.getextrema(), (255, 255))


def test_mask_image_thresholds_and_resizes():
    raw = np.array([[[[0.2, 0.9], [0.8, 0.1]]]], "float32")  # 0..1 soft
    mask = segmentation._mask_image(raw, None, (2, 2, 2, 2, 2, 2))
    check("soft threshold", list(mask.getdata()), [0, 255, 255, 0])

    raw = np.array([[[[-5.0, 3.0]]]], "float32")  # logits, 2x1 mask
    mask = segmentation._mask_image(raw, None, (4, 4, 2, 4, 4, 2))
    check("resized size", mask.size, (4, 2))


def test_mask_image_unpads_encoder_frame():
    # A portrait 4x8 image inside an 8x8 padded encoder frame: content sits in
    # the left 4 columns, the rest is right padding.
    geometry = (8, 8, 4, 8, 4, 8)  # frame_w, frame_h, new_w, new_h, W, H
    raw = np.zeros((8, 8), "float32")
    raw[2:6, 1:3] = 1.0
    mask = segmentation._mask_image(raw, None, geometry)
    check("unpadded size", mask.size, (4, 8))
    check("content kept", mask.getpixel((1, 3)), 255)
    check("padding area empty", mask.getpixel((3, 3)), 0)
    # A mask already at document size is returned untouched.
    raw_doc = np.zeros((8, 4), "float32")
    raw_doc[2:6, 1:3] = 1.0
    mask_doc = segmentation._mask_image(raw_doc, None, geometry)
    check("doc-sized kept", mask_doc.size, (4, 8))


def test_encoder_io_names():
    session = _Session(
        [("input", [1, 3, 1024, 1024])], [("image_embeddings", None)],
        lambda names, feed: [],
    )
    in_name, out_name, shape = segmentation._encoder_io(session)
    check("encoder input", in_name, "input")
    check("encoder output", out_name, "image_embeddings")
    check("encoder shape", shape, [1, 3, 1024, 1024])


def test_encoder_io_batchless_and_size():
    session = _Session(
        [("input_image", [3, 512, 512])], [("image_embeddings", None)],
        lambda names, feed: [],
    )
    in_name, _out, shape = segmentation._encoder_io(session)
    check("batchless input name", in_name, "input_image")
    check("batchless shape", shape, [3, 512, 512])


def test_encoder_frame():
    check("nchw frame", segmentation._encoder_frame([1, 3, 1024, 1024]), (1024, 1024))
    check("hwc frame", segmentation._encoder_frame([684, 1024, 3]), (1024, 684))
    check("nhwc frame", segmentation._encoder_frame([1, 684, 1024, 3]), (1024, 684))
    check("symbolic frame", segmentation._encoder_frame(["n", 3, "h", "w"]), (1024, 1024))


def test_read_simple_yaml():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "config.yaml")
        with open(path, "w") as handle:
            handle.write("type: segment_anything\ninput_size: 1024\n"
                         "max_width: 1024\nmax_height: 682\n"
                         "display_name: Segment Anything (ViT-H)\n")
        values = segmentation._read_simple_yaml(path)
        check("yaml int", values["max_height"], 682)
        check("yaml str", values["type"], "segment_anything")
        check("yaml str spaces", values["display_name"], "Segment Anything (ViT-H)")


def test_config_frame():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        encoder = os.path.join(folder, "sam.encoder.onnx")
        with open(encoder, "w") as handle:
            handle.write("")
        with open(os.path.join(folder, "config.yaml"), "w") as handle:
            handle.write("max_width: 1024\nmax_height: 682\n")
        check("config frame", segmentation._config_frame(encoder), (1024, 682))
        check("frame prefers config",
              segmentation._encoder_frame(["h", "w", 3], encoder), (1024, 682))
        plain = os.path.join(folder, "plain")
        os.makedirs(plain)
        check("frame ignores absent config",
              segmentation._encoder_frame(["h", "w", 3],
                                          os.path.join(plain, "x.onnx")),
              (1024, 1024))

        alt = os.path.join(folder, "alt")
        os.makedirs(alt)
        other = os.path.join(alt, "m.encoder.onnx")
        with open(other, "w") as handle:
            handle.write("")
        with open(os.path.join(alt, "config.yaml"), "w") as handle:
            handle.write("input_width: 512\ninput_height: 384\n")
        check("input_width/height keys",
              segmentation._config_frame(other), (512, 384))

        pair_dir = os.path.join(folder, "pair")
        os.makedirs(pair_dir)
        pair_enc = os.path.join(pair_dir, "m.encoder.onnx")
        with open(pair_enc, "w") as handle:
            handle.write("")
        with open(os.path.join(pair_dir, "config.yaml"), "w") as handle:
            handle.write("input_size: [684, 1024]\n")
        check("input_size pair", segmentation._config_frame(pair_enc), (684, 1024))


def test_encoder_input_layout():
    hwc = np.zeros((1024, 1024, 3), "float32")

    def first(shape):
        return segmentation._encoder_inputs(hwc, shape)[0].shape

    check("nchw", first([1, 3, 1024, 1024]), (1, 3, 1024, 1024))
    check("nhwc batch", first([1, 1024, 1024, 3]), (1, 1024, 1024, 3))
    check("chw", first([3, 1024, 1024]), (3, 1024, 1024))
    check("hwc", first([1024, 1024, 3]), (1024, 1024, 3))
    # Symbolic dims prefer the common NCHW layout.
    check("symbolic", first(["batch", 3, "h", "w"]), (1, 3, 1024, 1024))
    # Both layouts are offered so the caller can retry.
    check("two candidates", len(segmentation._encoder_inputs(hwc, [1, 3, 1024, 1024])), 2)


def test_encoder_rank_adapts():
    captured = {}

    class _Enc(_Session):
        def run(self, names, feed):
            captured["shape"] = feed["input_image"].shape
            return [np.zeros((1, 256, 64, 64), "float32")]

    encoder = _Enc(
        [("input_image", [3, 1024, 1024])], [("image_embeddings", None)],
        lambda names, feed: [],
    )
    old = segmentation._session_for
    segmentation._session_for = lambda path: encoder
    segmentation.unload()
    try:
        segmentation._embed(Image.new("RGB", (40, 30)), "enc")
        check("fed chw", captured["shape"], (3, 1024, 1024))
    finally:
        segmentation._session_for = old
        segmentation.unload()


def test_encoder_rank_adapts_nhwc():
    captured = {}

    class _Enc(_Session):
        def run(self, names, feed):
            captured["shape"] = feed["input_image"].shape
            return [np.zeros((1, 256, 64, 64), "float32")]

    encoder = _Enc(
        [("input_image", [1024, 1024, 3])], [("image_embeddings", None)],
        lambda names, feed: [],
    )
    old = segmentation._session_for
    segmentation._session_for = lambda path: encoder
    segmentation.unload()
    try:
        segmentation._embed(Image.new("RGB", (40, 30)), "enc")
        check("fed hwc", captured["shape"], (1024, 1024, 3))
    finally:
        segmentation._session_for = old
        segmentation.unload()


def test_decoder_path_derivation():
    check(
        "decoder sibling",
        segmentation._decoder_path(os.path.join("m", "sam_encoder.onnx")),
        os.path.join("m", "sam_decoder.onnx"),
    )
    check("decoder none", segmentation._decoder_path("sam.onnx"), None)


def test_segment_end_to_end_and_cache():
    encoder = _Session(
        [("input", [1, 3, 1024, 1024])], [("image_embeddings", None)],
        lambda names, feed: [np.zeros((1, 256, 64, 64), "float32")],
    )
    height, width = 30, 40

    def produce(names, feed):
        masks = np.zeros((1, 3, height, width), "float32")
        masks[0, 1, 5:20, 5:25] = 1.0
        return [masks, np.array([[0.1, 0.9, 0.2]], "float32"),
                np.zeros((1, 3, 256, 256), "float32")]

    decoder = _Session(
        [("image_embeddings", None), ("point_coords", None), ("point_labels", None),
         ("mask_input", None), ("has_mask_input", None), ("orig_im_size", None)],
        [("masks", None), ("iou_predictions", None), ("low_res_masks", None)],
        produce,
    )
    sessions = {"enc": encoder, "dec": decoder}

    old_resolve = segmentation._resolve
    old_session_for = segmentation._session_for
    segmentation._resolve = lambda label: ("enc", "dec")
    segmentation._session_for = lambda path: sessions[path]
    segmentation.unload()
    try:
        image = Image.new("RGB", (width, height), (10, 20, 30))
        mask = segmentation.segment(image, [(10, 10)], model_name="test")
        check("segment size", mask.size, (width, height))
        check("segment foreground", mask.getpixel((10, 10)), 255)
        check("segment background", mask.getpixel((0, 0)), 0)
        check("encoder ran once", encoder.runs, 1)

        # A second click on the same image reuses the cached embedding.
        segmentation.segment(image, [(8, 8)], model_name="test")
        check("encoder cached", encoder.runs, 1)
        check("decoder ran twice", decoder.runs, 2)
    finally:
        segmentation._resolve = old_resolve
        segmentation._session_for = old_session_for
        segmentation.unload()


def test_list_models_pairs(tmpdir=None):
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        for name in ("sam_encoder.onnx", "sam_decoder.onnx",
                     "lone_encoder.onnx", "other.txt"):
            with open(os.path.join(folder, name), "w") as handle:
                handle.write("")
        old = segmentation.paths.segmentation_dir
        segmentation.paths.segmentation_dir = lambda: folder
        try:
            check("paired listed", segmentation.list_models(), ["sam"])
        finally:
            segmentation.paths.segmentation_dir = old


def test_list_models_extracted_folder():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        bundle = os.path.join(folder, "sam_vit_b")
        os.makedirs(bundle)
        for name in ("encoder.onnx", "decoder.onnx", "README.md"):
            with open(os.path.join(bundle, name), "w") as handle:
                handle.write("")
        old = segmentation.paths.segmentation_dir
        segmentation.paths.segmentation_dir = lambda: folder
        try:
            check("folder listed", segmentation.list_models(), ["sam_vit_b"])
            encoder, decoder = segmentation._resolve("sam_vit_b")
            check("folder encoder", os.path.basename(encoder), "encoder.onnx")
            check("folder decoder", os.path.basename(decoder), "decoder.onnx")
        finally:
            segmentation.paths.segmentation_dir = old


def test_segment_requires_point():
    try:
        segmentation.segment(Image.new("RGB", (8, 8)), [])
    except ValueError:
        check("empty points rejected", True, True)
    else:
        check("empty points rejected", False, True)


def test_resolve_missing_raises():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        old = segmentation.paths.segmentation_dir
        segmentation.paths.segmentation_dir = lambda: folder
        try:
            try:
                segmentation._resolve("nope")
            except FileNotFoundError:
                check("missing model raises", True, True)
            else:
                check("missing model raises", False, True)
        finally:
            segmentation.paths.segmentation_dir = old


def test_decoder_feed_matches_roles_not_names():
    session = _Session(
        [("image_embedding", None), ("coords", None), ("labels", None),
         ("mask", None), ("has_mask", None), ("orig_size", None)],
        [("masks", None)],
        lambda names, feed: [],
    )
    feed = segmentation._decoder_feed(
        session, np.zeros((1, 256, 64, 64), "float32"),
        np.zeros((1, 1, 2), "float32"), np.ones((1, 1), "float32"), (12, 10),
    )
    check("alias feed keys", sorted(feed), sorted([
        "image_embedding", "coords", "labels", "mask", "has_mask", "orig_size"]))
    check("alias orig size", feed["orig_size"].tolist(), [12.0, 10.0])


def test_segment_reports_unmapped_inputs():
    encoder = _Session(
        [("input", [1, 3, 1024, 1024])], [("image_embeddings", None)],
        lambda names, feed: [np.zeros((1, 256, 64, 64), "float32")],
    )
    decoder = _Session(
        [("image_embeddings", None), ("point_coords", None),
         ("high_res_feats_0", None)],
        [("masks", None)],
        lambda names, feed: [np.zeros((1, 1, 8, 8), "float32")],
    )
    sessions = {"enc": encoder, "dec": decoder}
    old_r, old_s = segmentation._resolve, segmentation._session_for
    segmentation._resolve = lambda label: ("enc", "dec")
    segmentation._session_for = lambda path: sessions[path]
    segmentation.unload()
    try:
        try:
            segmentation.segment(Image.new("RGB", (8, 8)), [(1, 1)], model_name="t")
        except RuntimeError as exc:
            check("unmapped input named", "high_res_feats_0" in str(exc), True)
        else:
            check("unmapped input named", False, True)
    finally:
        segmentation._resolve, segmentation._session_for = old_r, old_s
        segmentation.unload()


def test_segmentation_files_are_not_picker_models():
    import tempfile

    from engine import paths

    with tempfile.TemporaryDirectory() as folder:
        bundle = os.path.join(folder, "sam_vit_h_quant")
        os.makedirs(bundle)
        encoder = os.path.join(bundle, "sam_vit_h_4b8939.encoder.quant.onnx")
        with open(encoder, "w") as handle:
            handle.write("")
        old_dir = paths.segmentation_dir
        paths.segmentation_dir = lambda: folder
        try:
            check("segment file flagged",
                  paths.is_segmentation_model(encoder), True)
            check("outside file not flagged",
                  paths.is_segmentation_model(os.path.join(folder, "..", "x.safetensors")),
                  False)
            # An entry an older build curated from a SAM bundle must be dropped.
            old_settings = paths._model_settings
            paths._model_settings = lambda: {"entries": [
                {"path": encoder, "group": "Anima", "family": "anima"},
                {"path": os.path.join(folder, "..", "anima.safetensors"),
                 "group": "Anima", "family": "anima"},
            ]}
            try:
                entries = paths.model_entries()
            finally:
                paths._model_settings = old_settings
            check("segmentation row dropped", len(entries), 1)
            check("real model kept",
                  os.path.basename(entries[0]["path"]), "anima.safetensors")
        finally:
            paths.segmentation_dir = old_dir


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
