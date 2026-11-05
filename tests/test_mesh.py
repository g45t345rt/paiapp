"""Unit tests for the mesh pipeline (run: .venv\\Scripts\\python.exe tests\\test_mesh.py)."""

import json
import os
import struct
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from paintapp import mesh  # noqa: E402

FAILED = []

MALE = os.path.join(mesh.MODEL_DIR, "Superhero_Male_FullBody.gltf")
FEMALE = os.path.join(mesh.MODEL_DIR, "Superhero_Female_FullBody.gltf")


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def check_true(name, got):
    check(name, bool(got), True)


def check_close(name, got, want, tol=1e-4):
    if abs(float(got) - float(want)) > tol:
        FAILED.append(f"{name}: got {got!r} want {want!r} (+-{tol})")
    else:
        print(f"ok  {name}")


def test_models_load_with_full_finger_rig():
    male = mesh.load(MALE, "male")
    female = mesh.load(FEMALE, "female")
    check("male joint count", len(male.joint_names), 65)
    check("female joint count", len(female.joint_names), 65)
    check("identical skeletons", male.joint_names, female.joint_names)
    for finger in ("index", "middle", "ring", "pinky", "thumb"):
        check_true(f"{finger} bones on both hands",
                   f"{finger}_01_l" in male.joint_index and f"{finger}_01_r" in male.joint_index)
    check("inverse bind matrices", male.inverse_bind.shape, (65, 4, 4))


def test_bind_pose_skins_to_identity():
    model = mesh.load(MALE, "male")
    positions, normals, triangles = model.skin({}, proxy=False)
    check("triangle count", len(triangles), len(model.triangles))
    check_close("rest pose equals bind pose", float(np.abs(positions - model.position).max()), 0.0, 1e-3)
    check_close("normals are unit", float(np.abs(np.linalg.norm(normals, axis=1) - 1).max()), 0.0, 1e-3)


def test_pose_deforms_mesh():
    model = mesh.load(MALE, "male")
    rest, _n, _t = model.skin({}, proxy=False)
    delta = model.aim_delta("upperarm_l", np.array([0.0, -1.0, 0.0]), {})
    check_true("aim delta exists", delta is not None)
    posed, _n, _t = model.skin({"upperarm_l": delta}, proxy=False)
    moved = float(np.abs(posed - rest).max())
    check_true("pose moves vertices", moved > 0.05)


def test_aim_solver_points_a_bone():
    model = mesh.load(MALE, "male")
    for target in (np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                   np.array([0.0, -0.5, 0.86])):
        target = target / np.linalg.norm(target)
        delta = model.aim_delta("upperarm_r", target, {})
        direction = model.joint_world_direction(model.joint_index["upperarm_r"], {"upperarm_r": delta})
        direction = direction / np.linalg.norm(direction)
        check_close(f"aims at {target.round(2)}", float(direction @ target), 1.0, 1e-3)


def test_aim_solver_respects_posed_parent():
    model = mesh.load(MALE, "male")
    parents = {"spine_02": model.aim_delta("spine_02", np.array([0.35, 0.94, 0.0]), {})}
    delta = model.aim_delta("upperarm_l", np.array([0.0, -1.0, 0.0]), parents)
    posed = dict(parents)
    posed["upperarm_l"] = delta
    direction = model.joint_world_direction(model.joint_index["upperarm_l"], posed)
    direction = direction / np.linalg.norm(direction)
    check_close("aims with a bent parent", float(direction @ np.array([0.0, -1.0, 0.0])), 1.0, 1e-2)


def test_render_paints_a_solid_figure():
    model = mesh.load(MALE, "male")
    sprite, left, top = mesh.render_sprite(model, {}, height_px=420, supersample=1)
    check("sprite mode", sprite.mode, "RGBA")
    alpha = np.asarray(sprite.getchannel("A")) > 8
    check_true("sprite has ink", alpha.any())
    check_true("sprite is figure sized", 300 < sprite.height < 500)
    rows = np.nonzero(alpha.any(axis=1))[0]
    torso = rows[0] + int((rows[-1] - rows[0]) * 0.40)
    columns = np.nonzero(alpha[torso])[0]
    check_true("torso row is one solid run",
               int(np.count_nonzero(np.diff(columns) > 1)) == 0)


def test_side_view_is_solid():
    model = mesh.load(MALE, "male")
    sprite, _l, _t = mesh.render_sprite(model, {}, height_px=420, yaw=90, supersample=1)
    alpha = np.asarray(sprite.getchannel("A")) > 8
    check_true("side view is narrow", sprite.width < sprite.height * 0.4)
    fill = alpha.sum() / (alpha.any(axis=0).sum() * alpha.any(axis=1).sum())
    check_true("side view is filled", fill > 0.45)
    rows = np.nonzero(alpha.any(axis=1))[0]
    solid = 0
    for frac in (0.2, 0.35, 0.5, 0.65):
        row = rows[0] + int((rows[-1] - rows[0]) * frac)
        columns = np.nonzero(alpha[row])[0]
        if len(columns) and np.count_nonzero(np.diff(columns) > 1) == 0:
            solid += 1
    check("every sampled side row is contiguous", solid, 4)


def test_perspective_foreshortens():
    model = mesh.load(MALE, "male")
    flat, _l, _t = mesh.render_sprite(model, {}, height_px=420, supersample=1)
    deep, _l, _t = mesh.render_sprite(model, {}, height_px=420, supersample=1,
                                     yaw=55, perspective=2.5)
    check_true("perspective changes the silhouette",
               abs(flat.width - deep.width) > 1 or abs(flat.height - deep.height) > 1)


def test_proxy_is_smaller_and_renders():
    model = mesh.load(MALE, "male")
    check_true("proxy built", model.proxy_triangles > 0)
    check_true("proxy is smaller than the mesh", model.proxy_triangles < len(model.triangles) // 2)
    sprite, _l, _t = mesh.render_sprite(model, {}, height_px=300, proxy=True, supersample=1)
    check_true("proxy renders ink", (np.asarray(sprite.getchannel("A")) > 8).any())


def test_proxy_keeps_the_silhouette():
    model = mesh.load(MALE, "male")
    full, _l, _t = mesh.render_sprite(model, {}, height_px=300, supersample=1)
    proxy, _l, _t = mesh.render_sprite(model, {}, height_px=300, proxy=True, supersample=1)
    a = np.asarray(full.getchannel("A")) > 8
    b = np.asarray(proxy.getchannel("A")) > 8
    height = max(a.shape[0], b.shape[0])
    width = max(a.shape[1], b.shape[1])
    pad = lambda m: np.pad(m, ((0, height - m.shape[0]), (0, width - m.shape[1])))
    a, b = pad(a), pad(b)
    iou = np.logical_and(a, b).sum() / max(int(np.logical_or(a, b).sum()), 1)
    check_true("decimated proxy matches the full silhouette", iou > 0.85)


def test_female_and_male_differ():
    male = mesh.load(MALE, "male")
    female = mesh.load(FEMALE, "female")
    check_true("different heights", abs(male.height - female.height) > 0.01)
    check_true("different vertex counts", len(male.position) != len(female.position))


def _glb_with_strided_vec3():
    """A tiny GLB whose POSITION accessor is interleaved (byteStride 16)."""
    values = []
    raw = bytearray()
    for index in range(3):
        xyz = (float(index), float(index) + 1.0, float(index) + 2.0)
        raw += struct.pack("<3f", *xyz)
        raw += struct.pack("<f", 999.0)  # padding so the stride is 16, not 12
        values.append(xyz)
    binary = bytes(raw)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary), "byteStride": 16}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
    }
    json_bytes = json.dumps(gltf).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = binary + b"\x00" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    out = struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    out += struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes
    return out, values


def test_reads_strided_accessor_from_glb():
    data, expected = _glb_with_strided_vec3()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "strided.glb")
        with open(path, "wb") as handle:
            handle.write(data)
        gltf, binary = mesh._load_gltf_bytes(path)
        values = mesh._read_accessor(gltf, binary, 0)
    check("strided shape", values.shape, (3, 3))
    check("strided values", [list(row) for row in values.tolist()], [list(row) for row in expected])


def test_smooth_shading_gradates_the_body():
    model = mesh.load(MALE, "male")
    smooth, _l, _t = mesh.render_sprite(
        model, {}, height_px=300, supersample=1, smooth=True, force_cpu=True)
    flat, _l, _t = mesh.render_sprite(
        model, {}, height_px=300, supersample=1, smooth=False, force_cpu=True)
    smooth_pixels = np.asarray(smooth)[..., :3][np.asarray(smooth.getchannel("A")) > 200]
    flat_pixels = np.asarray(flat)[..., :3][np.asarray(flat.getchannel("A")) > 200]
    smooth_colours = len(np.unique(smooth_pixels.reshape(-1, 3), axis=0))
    flat_colours = len(np.unique(flat_pixels.reshape(-1, 3), axis=0))
    check_true("smooth shading produces a gradient", smooth_colours > flat_colours)
    check_true("flat shading is per triangle",
               flat_colours <= len(model.triangles))


def _mask(sprite, left, top):
    return np.asarray(sprite.getchannel("A")) > 8, int(left), int(top)


def _iou_aligned(first, second):
    (a, la, ta), (b, lb, tb) = first, second
    left, top = min(la, lb), min(ta, tb)
    width = max(la + a.shape[1], lb + b.shape[1]) - left
    height = max(ta + a.shape[0], tb + b.shape[0]) - top
    canvas_a = np.zeros((height, width), dtype=bool)
    canvas_b = np.zeros((height, width), dtype=bool)
    canvas_a[ta - top:ta - top + a.shape[0], la - left:la - left + a.shape[1]] = a
    canvas_b[tb - top:tb - top + b.shape[0], lb - left:lb - left + b.shape[1]] = b
    return np.logical_and(canvas_a, canvas_b).sum() / max(
        int(np.logical_or(canvas_a, canvas_b).sum()), 1)


def test_gpu_backend_matches_the_cpu():
    model = mesh.load(MALE, "male")
    if not mesh.gpu_available():
        print("ok  gpu backend unavailable (skipped)")
        return
    for label, kwargs in (("front", {}), ("turned", {"yaw": 40}),
                          ("rolled", {"roll": 30}),
                          ("perspective", {"yaw": 55, "perspective": 7.0})):
        kwargs = dict(kwargs, height_px=300)
        gpu = mesh.render_sprite(model, {}, **kwargs)
        cpu = mesh.render_sprite(model, {}, force_cpu=True, supersample=2, **kwargs)
        check_true(f"gpu drew {label}", gpu is not None)
        iou = _iou_aligned(_mask(*gpu), _mask(*cpu))
        check_true(f"gpu matches cpu ({label}, IoU {iou:.3f})", iou > 0.9)


def _aligned_rgba(first, second):
    a, la, ta = first
    b, lb, tb = second
    la, ta, lb, tb = int(la), int(ta), int(lb), int(tb)
    left, top = min(la, lb), min(ta, tb)
    width = max(la + a.shape[1], lb + b.shape[1]) - left
    height = max(ta + a.shape[0], tb + b.shape[0]) - top
    canvas_a = np.zeros((height, width, 4), dtype=np.uint8)
    canvas_b = np.zeros((height, width, 4), dtype=np.uint8)
    canvas_a[ta - top:ta - top + a.shape[0], la - left:la - left + a.shape[1]] = a
    canvas_b[tb - top:tb - top + b.shape[0], lb - left:lb - left + b.shape[1]] = b
    return canvas_a, canvas_b


def test_gpu_shades_the_same_surfaces_as_the_cpu():
    """A depth-buffer regression: with unchecked depth the GPU draws back faces
    over front ones in stripes, which a silhouette comparison would not catch."""
    model = mesh.load(MALE, "male")
    if not mesh.gpu_available():
        print("ok  gpu backend unavailable (skipped)")
        return
    gpu, gl, gt = mesh.render_sprite(model, {}, height_px=320)
    cpu, cl, ct = mesh.render_sprite(model, {}, height_px=320, force_cpu=True,
                                     supersample=2)
    a, b = _aligned_rgba((np.asarray(gpu), gl, gt), (np.asarray(cpu), cl, ct))
    mask = (a[..., 3] > 8) & (b[..., 3] > 8)
    difference = np.abs(a[..., :3].astype(np.int16) - b[..., :3].astype(np.int16)).mean(axis=2)
    mean = float(difference[mask].mean())
    bad = float((difference[mask] > 40).mean())
    check_true(f"gpu matches the cpu surfaces (mean colour diff {mean:.1f})", mean < 15.0)
    check_true(f"few mismatched pixels ({bad * 100:.1f}%)", bad < 0.05)


def test_view_matrix_rotates():
    yaw = mesh.view_matrix(90, 0)
    moved = np.array([1.0, 0.0, 0.0]) @ yaw.T
    check_close("yaw 90 sends +X to -Z", float(moved[2]), -1.0, 1e-6)
    check_close("yaw 90 keeps +X off the screen x", float(moved[0]), 0.0, 1e-6)


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
