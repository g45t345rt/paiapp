"""Unit tests for the Poses objects (run: .venv\\Scripts\\python.exe tests\\test_poses.py)."""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import project  # noqa: E402
from paintapp import mesh, poses  # noqa: E402
from paintapp.document import Canvas, Layer  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def check_true(name, got):
    check(name, bool(got), True)


def check_close(name, got, want, tol=1e-3):
    if abs(float(got) - float(want)) > tol:
        FAILED.append(f"{name}: got {got!r} want {want!r} (+-{tol})")
    else:
        print(f"ok  {name}")


def aim_direction(spec, bone):
    model = poses.model_for(spec["body"])
    index = model.joint_index[bone]
    child = model.aim_child[index]
    world = model.world_matrices(spec["joints"])
    direction = world[child][:3, 3] - world[index][:3, 3]
    return direction / np.linalg.norm(direction)


# ------------------------------------------------------------------ specs
def test_both_bodies_available():
    check("both bodies ship", poses.available_bodies(), ["female", "male"])


def test_normalize_defaults_and_clamps():
    check("rejects non dict", poses.normalize(None), None)
    spec = poses.normalize({"x": 5, "y": 6})
    check("x kept", spec["x"], 5.0)
    check("default body", spec["body"], poses.DEFAULT_BODY)
    check("default scale", spec["scale"], poses.DEFAULT_SCALE)
    check("view default", spec["view"], [0.0, 0.0])
    check("joints default", spec["joints"], {})
    clamped = poses.normalize({"x": 0, "y": 0, "scale": 10 ** 9, "opacity": 4,
                               "body": "alien", "view": [10], "roll": "x",
                               "joints": {"head": "no", "spine_01": [1, 2, 3]}})
    check("scale clamped", clamped["scale"], poses.MAX_SCALE)
    check("opacity clamped", clamped["opacity"], 1.0)
    check("body falls back", clamped["body"], poses.DEFAULT_BODY)
    check("view padded", clamped["view"], [10.0, 0.0])
    check("roll falls back", clamped["roll"], 0.0)
    check("bad joint dropped", "head" in clamped["joints"], False)
    check("good joint kept", clamped["joints"]["spine_01"], [1.0, 2.0, 3.0])


def test_clone_is_independent():
    spec = poses.new(10, 10)
    copy = poses.clone(spec)
    copy["joints"]["spine_01"] = [10.0, 0.0, 0.0]
    copy["view"][0] = 45.0
    check("joints untouched", "spine_01" in spec["joints"], False)
    check("view untouched", spec["view"][0], 0.0)


def test_new_figure_has_a_relaxed_stance():
    spec = poses.new(0, 0, body="male")
    check_true("stance posed", len(spec["joints"]) >= 6)
    check("body respected", spec["body"], "male")
    check_true("arms are down",
               float(aim_direction(spec, "upperarm_l") @ np.array([0.0, -1.0, 0.0])) > 0.9)


def test_pose_bones_exclude_the_free_hand():
    model = poses.model_for("male")
    bones = poses.pose_bones(model)
    check_true("has upperarm", "upperarm_l" in bones)
    check_true("has chest", "spine_02" in bones)
    check_true("no digits", not any(any(d in b for d in poses.DIGITS) for b in bones))
    check_true("every handle is aimable",
               all(model.aim_child[model.joint_index[b]] >= 0 for b in bones))


# ------------------------------------------------------------------ posing
def test_pose_presets_are_geometrically_correct():
    spec = poses.new(0, 0, body="male")
    poses.apply_pose(spec, "arms_down")
    check_close("arms_down aims the arm down",
                float(aim_direction(spec, "upperarm_l") @ np.array([0.0, -1.0, 0.0])), 1.0)
    poses.apply_pose(spec, "sit")
    thigh = aim_direction(spec, "thigh_l")
    calf = aim_direction(spec, "calf_l")
    check_true("sit points the thigh forward", float(thigh[2]) > 0.8)
    check_true("sit hangs the calf down", float(calf[1]) < -0.8)
    for name in poses.POSE_ORDER:
        check_true(f"preset {name} applies", poses.apply_pose(spec, name))


def test_pose_preset_is_absolute():
    spec = poses.new(0, 0, body="male")
    poses.apply_pose(spec, "action")
    first = {b: list(a) for b, a in spec["joints"].items()}
    poses.apply_pose(spec, "action")
    check("reapplying gives the same pose",
          {b: [round(v, 4) for v in a] for b, a in spec["joints"].items()},
          {b: [round(v, 4) for v in a] for b, a in first.items()})


def test_hand_presets_curl_the_fingers():
    spec = poses.new(0, 0, body="male")
    model = poses.model_for("male")

    def tip_distance():
        positions = model.joint_positions(spec["joints"])
        return float(np.linalg.norm(
            positions[model.joint_index["index_04_leaf_l"]]
            - positions[model.joint_index["hand_l"]]))

    poses.apply_hand(spec, "open", "both")
    open_distance = tip_distance()
    poses.apply_hand(spec, "fist", "both")
    fist_distance = tip_distance()
    check_true("fist curls the index toward the palm", fist_distance < open_distance * 0.7)
    right = sum(1 for b in spec["joints"] if b.endswith("_r") and "index" in b)
    left = sum(1 for b in spec["joints"] if b.endswith("_l") and "index" in b)
    check("left index has three bones", left, 3)
    check("right index has three bones", right, 3)
    poses.apply_hand(spec, "open", "both")
    check("open clears the finger rotation",
          sum(1 for b in spec["joints"] if "index" in b), 0)
    poses.apply_hand(spec, "fist", "left")
    check("a single side only", sum(1 for b in spec["joints"] if "index" in b), 3)


def test_mirror_swaps_the_pose():
    spec = poses.new(0, 0, body="male")
    spec["pos"] = [1.5, 0.0, 0.5]
    poses.apply_pose(spec, "walk")
    before = poses.joints_projected(spec)
    pos_before = list(spec["pos"])
    poses.mirror(spec)
    after = poses.joints_projected(spec)
    check_close("hand_l takes the old hand_r x", after["hand_l"][0], -before["hand_r"][0], 1.0)
    check_close("hand_r takes the old hand_l x", after["hand_r"][0], -before["hand_l"][0], 1.0)
    check("the figure mirrors across the scene", spec["pos"][0], -pos_before[0])


def test_aim_at_points_a_limb_at_the_cursor():
    spec = poses.new(200, 200, body="male")
    start = poses.joints_projected(spec)["upperarm_l"]
    target = (start[0] - 140, start[1] + 150)
    delta = poses.aim_at(spec, "upperarm_l", *target)
    check_true("aim returns a delta", delta is not None)
    model = poses.model_for(spec["body"])
    index = model.joint_index["upperarm_l"]
    child = model.aim_child[index]
    world = model.world_matrices(spec["joints"])
    pointed = world[child][:3, 3] - world[index][:3, 3]
    pointed = pointed / np.linalg.norm(pointed)
    # The cursor is left of and below the handle: model -X and -Y.
    check_true("the bone points left", float(pointed[0]) < -0.1)
    check_true("the bone points down", float(pointed[1]) < -0.1)


def test_joint_at_and_hit_body():
    spec = poses.new(100, 100, body="male")
    hand = poses.joints_projected(spec)["hand_l"]
    check("joint_at finds the hand", poses.joint_at(spec, hand[0], hand[1], 5), "hand_l")
    check("joint_at misses far away", poses.joint_at(spec, hand[0] + 900, hand[1] + 900, 5), None)
    left, top, right, bottom = poses.bounds(spec)
    check_true("bounds wraps the figure", right > left and bottom > top)
    check_true("hit_body inside", poses.hit_body(spec, 100, 50))
    check("hit_body outside", poses.hit_body(spec, -5000, -5000), False)


# --------------------------------------------------------------- rendering
def test_render_paints_the_figure():
    spec = poses.new(0, 0, scale=200, body="male")
    sprite, left, top = poses.render_sprite(spec)
    check("sprite mode", sprite.mode, "RGBA")
    check_true("sprite has ink", sprite.getchannel("A").getbbox() is not None)
    check_true("sprite is figure sized", 150 < sprite.height < 260)


def test_render_opacity_and_proxy():
    spec = poses.new(0, 0, scale=180, body="female")
    spec["opacity"] = 0.5
    sprite, _l, _t = poses.render_sprite(spec)
    peak = sprite.getchannel("A").getextrema()[1]
    check_true("opacity halves alpha", 100 <= peak <= 140)
    spec["opacity"] = 1.0
    proxy, _l, _t = poses.render_sprite(spec, proxy=True)
    check_true("proxy renders ink", proxy.getchannel("A").getbbox() is not None)


def test_roll_and_view_change_the_silhouette():
    spec = poses.new(0, 0, scale=180, body="male")
    poses.apply_pose(spec, "t_pose")
    flat, _l, _t = poses.render_sprite(spec)
    spec["roll"] = 90.0
    rolled, _l, _t = poses.render_sprite(spec)
    check_true("roll swaps the aspect", abs(flat.width - rolled.height) < 40)
    spec["roll"] = 0.0
    spec["view"] = [90.0, 0.0]
    turned, _l, _t = poses.render_sprite(spec)
    check_true("a side view is narrower than the front", turned.width < flat.width * 0.6)


def test_preview_renders_a_proxy():
    spec = poses.new(0, 0, scale=180, body="male")
    poses.set_preview(True)
    try:
        check_true("preview active", poses.preview_active())
        sprite, _l, _t = poses.render_sprite(spec, proxy=True)
        check_true("preview sprite has ink", sprite.getchannel("A").getbbox() is not None)
    finally:
        poses.set_preview(False)
    check("preview off", poses.preview_active(), False)


def test_drag_quality_toggle():
    check("default drag quality", poses.drag_quality(), poses.DEFAULT_DRAG_QUALITY)
    poses.set_drag_quality("best")
    check("best selected", poses.drag_quality(), "best")
    poses.set_drag_quality("nonsense")
    check("invalid falls back to the default", poses.drag_quality(),
          poses.DEFAULT_DRAG_QUALITY)


def test_paste_during_a_preview_draws_the_figure():
    spec = poses.new(150, 150, scale=200, body="male")
    poses.set_preview(True)
    try:
        for quality in poses.DRAG_QUALITIES:
            poses.set_drag_quality(quality)
            layer = Layer("p", (400, 400), (0, 0, 0, 0))
            poses.paste(layer.image, spec)
            check_true(f"{quality} drag paints ink",
                       layer.image.getchannel("A").getbbox() is not None)
    finally:
        poses.set_preview(False)
        poses.set_drag_quality(poses.DEFAULT_DRAG_QUALITY)


def test_paste_composites_and_offsets():
    spec = poses.new(300, 420, scale=160, body="male")
    layer = Layer("p", (640, 640), (0, 0, 0, 0))
    poses.paste(layer.image, spec)
    check_true("paste paints ink", layer.image.getchannel("A").getbbox() is not None)
    base = layer.image.getchannel("A").getbbox()
    shifted = Layer("p", (640, 640), (0, 0, 0, 0))
    poses.paste(shifted.image, spec, (20, 20))
    moved = shifted.image.getchannel("A").getbbox()
    check("offset shifts the ink", moved[0], base[0] - 20)
    check("offset shifts the ink vertically", moved[1], base[1] - 20)


def test_paste_clips_off_the_layer():
    spec = poses.new(10, 120, scale=200, body="male")
    layer = Layer("p", (128, 128), (0, 0, 0, 0))
    poses.paste(layer.image, spec)
    check_true("partly off-layer still draws",
               layer.image.getchannel("A").getbbox() is not None)
    far = Layer("p", (128, 128), (0, 0, 0, 0))
    poses.paste(far.image, poses.new(-900, 120, scale=200, body="male"))
    check("fully off-layer draws nothing", far.image.getchannel("A").getbbox(), None)


# ---------------------------------------------------------------- document
def test_layer_composites_and_rasterises_poses():
    layer = Layer("p", (320, 320), (0, 0, 0, 0))
    layer.poses.append(poses.new(160, 60, scale=200, body="male"))
    layer.invalidate_poses()
    check_true("composite has ink", layer.composited_image().getchannel("A").getbbox() is not None)
    token = layer.pixels_token
    layer.poses.append(poses.new(40, 40, scale=120, body="female"))
    layer.invalidate_poses()
    check_true("token tracks the figures", layer.pixels_token != token)
    check("rasterise reports a change", layer.rasterize_poses(), True)
    check("figures cleared", layer.poses, [])
    check_true("pixels baked", layer.image.getchannel("A").getbbox() is not None)
    check("second rasterise is a no-op", layer.rasterize_poses(), False)


def test_snapshot_restores_poses():
    canvas = Canvas("c", (200, 200))
    layer = canvas.add_layer(name="P")
    layer.poses.append(poses.new(100, 40, scale=150, body="male"))
    canvas.restore(canvas.snapshot())
    restored = canvas.layers[0]
    check("one figure restored", len(restored.poses), 1)
    check("body restored", restored.poses[0]["body"], "male")


# ------------------------------------------------------------------ project
def fake_pose_layer(name):
    layer = SimpleNamespace(
        name=name, image=Image.new("RGBA", (4, 4), (0, 0, 0, 0)), mask=None,
        visible=True, locked=False, opacity=1.0, offset=(0, 0), group=None,
        clip=False, blend="normal", filters=[], texts=[], positive_prompt="",
        gen_settings={}, vector=False, strokes=[],
        poses=[poses.new(2, 2, scale=120, body="female")],
    )
    return layer


def test_project_save_persists_poses():
    layer = fake_pose_layer("P")
    canvas = SimpleNamespace(
        name="Canvas 1", width=4, height=4, active=0, layer_counter=1,
        group_counter=0, guides=[], groups=[], assets=[], asset_counter=0,
        inpaint_mask=None, layers=[layer],
    )
    app = SimpleNamespace(
        active_canvas=0, canvas_counter=1, color="#000000", brush_size=8,
        tools=SimpleNamespace(export_settings=lambda: {}),
        palette=["#000000"], selected_swatch=0, canvases=[canvas], _text_style=None,
    )
    with tempfile.TemporaryDirectory() as tmp:
        manifest = project.save(app, os.path.join(tmp, "poses.pai"))
    entry = manifest["canvases"][0]["layers"][0]
    check("poses persisted", len(entry["poses"]), 1)
    check("body persisted", entry["poses"][0]["body"], "female")


def test_document_restore_accepts_legacy_mannequins_key():
    canvas = Canvas("c", (64, 64))
    canvas.add_layer(name="L")
    snapshot = canvas.snapshot()
    entry = snapshot["layers"][0]
    entry.pop("poses", None)
    entry["mannequins"] = [
        {"x": 5, "y": 5, "scale": 120, "body": "male", "joints": {}},
    ]
    canvas.restore(snapshot)
    restored = canvas.layers[0]
    check("legacy key still loads", len(restored.poses), 1)
    check("legacy body kept", restored.poses[0]["body"], "male")


def test_ik_handles_are_the_end_effectors():
    model = poses.model_for("male")
    check("ik handles", sorted(poses.IK_CHAIN),
          ["ball_l", "ball_r", "foot_l", "foot_r", "hand_l", "hand_r"])
    for end, chain in poses.IK_CHAIN.items():
        check_true(f"{end} handle exists", end in poses.pose_bones(model))
        check_true(f"{end} chain has the hinge",
                   poses.HINGE_LIMITS.get(chain[0]) is not None)


def test_ik_reaches_a_reachable_target():
    model = poses.model_for("male")
    base = poses.new(200, 200, body="male")
    scale = 420 / model.height
    shoulder = model.joint_positions(base["joints"])[model.joint_index["upperarm_r"]]
    reach = float(np.linalg.norm(
        model.joint_positions(base["joints"])[model.joint_index["hand_r"]] - shoulder))
    for label, direction in (("down", (0.0, -1.0, 0.4)), ("across", (-0.5, -0.8, 0.3)),
                             ("back", (0.0, -0.7, -0.7))):
        spec = poses.new(200, 200, body="male")
        vector = np.array(direction, dtype=float)
        vector /= np.linalg.norm(vector)
        target_model = shoulder + vector * reach * 0.8
        x, y, _c = mesh.project(target_model[None, :], scale, 0, 0, poses.PERSPECTIVE)
        target = (float(spec["x"]) + float(x[0]), float(spec["y"]) + float(y[0]))
        check_true(f"ik solves {label}", poses.ik_at(spec, "hand_r", *target))
        point = model.joint_positions(spec["joints"])[model.joint_index["hand_r"]]
        ax, ay, _c = mesh.project(point[None, :], scale, 0, 0, poses.PERSPECTIVE)
        miss = float(np.hypot(float(spec["x"]) + float(ax[0]) - target[0],
                              float(spec["y"]) + float(ay[0]) - target[1]))
        check_true(f"ik reaches {label} ({miss:.1f}px)", miss < 20.0)


def test_ik_solves_a_leg():
    model = poses.model_for("male")
    spec = poses.new(200, 200, body="male")
    scale = 420 / model.height
    hip = model.joint_positions(spec["joints"])[model.joint_index["thigh_l"]]
    leg = float(np.linalg.norm(
        model.joint_positions(spec["joints"])[model.joint_index["foot_l"]] - hip))
    direction = np.array([0.1, -0.75, 0.65])
    direction /= np.linalg.norm(direction)
    target_model = hip + direction * leg * 0.8
    x, y, _c = mesh.project(target_model[None, :], scale, 0, 0, poses.PERSPECTIVE)
    target = (float(spec["x"]) + float(x[0]), float(spec["y"]) + float(y[0]))
    check_true("leg ik solves", poses.ik_at(spec, "foot_l", *target))
    point = model.joint_positions(spec["joints"])[model.joint_index["foot_l"]]
    ax, ay, _c = mesh.project(point[None, :], scale, 0, 0, poses.PERSPECTIVE)
    miss = float(np.hypot(float(spec["x"]) + float(ax[0]) - target[0],
                          float(spec["y"]) + float(ay[0]) - target[1]))
    check_true(f"leg ik reaches ({miss:.1f}px)", miss < 20.0)


def test_hinge_joints_do_not_hyperextend():
    model = poses.model_for("male")

    def hinge_degrees(spec, joint):
        return float(np.degrees(poses._hinge_component(model, spec, joint)))

    spec = poses.new(200, 200, body="male")
    poses.aim_at(spec, "lowerarm_l", 0, -300)
    check_true("elbow never bends backwards", hinge_degrees(spec, "lowerarm_l") >= -1e-6)
    check_true("elbow stops at its limit", hinge_degrees(spec, "lowerarm_l") <= 150.5)
    spec = poses.new(200, 200, body="male")
    poses.aim_at(spec, "calf_l", 200, 500)
    check_true("knee never bends forwards", hinge_degrees(spec, "calf_l") >= -1e-6)
    check_true("knee stops at its limit", hinge_degrees(spec, "calf_l") <= 150.5)


def test_gizmo_rings_rotate_a_bone():
    model = poses.model_for("male")
    spec = poses.new(200, 200, body="male")
    geometry = poses.gizmo(spec, "upperarm_l")
    check_true("gizmo offers rings", len(geometry["rings"]) >= 1)
    check_true("rings are projected circles",
               all(len(points) > 8 for points in geometry["rings"].values()))
    origin = geometry["origin"]
    name = sorted(geometry["rings"])[0]
    point = geometry["rings"][name][len(geometry["rings"][name]) // 4]
    check_true("a ring point is away from the joint",
               np.hypot(point[0] - origin[0], point[1] - origin[1]) > 8)
    check_true("ring hit test",
               poses.gizmo_axis_at(spec, geometry, point[0], point[1], 5.0)
               in geometry["rings"])
    check("the joint itself is not a ring hit",
          poses.gizmo_axis_at(spec, geometry, origin[0], origin[1], 5.0), None)
    check("ring miss far away",
          poses.gizmo_axis_at(spec, geometry, point[0] + 900, point[1] + 900, 5.0), None)
    check_true("ring angle available",
               poses.gizmo_angle(spec, "upperarm_l", name, point[0], point[1]) is not None)

    def arm_direction():
        index = model.joint_index["upperarm_l"]
        child = model.aim_child[index]
        world = model.world_matrices(spec["joints"])
        vector = world[child][:3, 3] - world[index][:3, 3]
        return vector / np.linalg.norm(vector)

    before = arm_direction()
    check_true("axis rotation applies",
               poses.apply_axis_rotation(spec, "upperarm_l", poses.GIZMO_AXES[name], 0.5))
    after = arm_direction()
    turned = float(np.degrees(np.arccos(np.clip(before @ after, -1.0, 1.0))))
    check_true(f"the bone turned ({turned:.1f} deg)", turned > 5.0)


def test_overlay_reports_ik_handles_and_gizmo():
    spec = poses.new(120, 120, body="male")
    data = poses.overlay(spec)
    check("overlay lists ik handles", sorted(data["ik_joints"]), sorted(poses.IK_CHAIN))
    check_true("overlay has bones", len(data["bones"]) > 5)
    check_true("overlay has a root", data["root"] is not None)


def test_figure_gizmo_handles():
    spec = poses.new(300, 200, body="female")
    geometry = poses.figure_gizmo(spec)
    check_true("figure gizmo exists", geometry is not None)
    for name in ("move_x", "move_y", "rotate", "scale"):
        if name == "rotate":
            point = geometry["ring"][5]
        elif name == "scale":
            point = geometry["scale"]
        else:
            point = geometry[name][1]
        check(f"{name} hit test",
              poses.figure_handle_at(spec, geometry, point[0], point[1], 8.0), name)
    centre = geometry["origin"]
    check("the pivot is not a handle (body drag wins)",
          poses.figure_handle_at(spec, geometry, centre[0] + 5, centre[1] + 5, 8.0), None)
    check("no handle far away",
          poses.figure_handle_at(spec, geometry, centre[0] + 900, centre[1] + 900, 8.0), None)
    check_true("figure angle is measurable",
               poses.figure_angle(spec, centre[0] + 50, centre[1]) is not None)


def test_figure_rotate_and_scale_pin_the_pelvis():
    scene = poses.normalize_scene({"zoom": poses.DEFAULT_ZOOM, "x": 300.0, "y": 200.0})
    pivot = poses.figure_gizmo(poses.new(300, 200, body="female"))["origin"]
    rotated = poses.new(300, 200, body="female")
    rotated["spin"] = 30.0
    poses.sync_spec(rotated, scene)
    poses.keep_pelvis(rotated, scene, pivot)
    scaled = poses.new(300, 200, body="female")
    scaled["size"] = poses.size_for_pixels(scene, scaled, 700.0)
    poses.sync_spec(scaled, scene)
    poses.keep_pelvis(scaled, scene, pivot)
    for label, spec in (("rotate", rotated), ("scale", scaled)):
        pelvis = poses.joints_projected(spec)["pelvis"]
        drift = float(np.hypot(pelvis[0] - pivot[0], pelvis[1] - pivot[1]))
        check_true(f"{label} keeps the pelvis in place ({drift:.2f}px)", drift < 1.0)


def test_hole_render_is_unaffected_by_the_gizmo():
    # The figure gizmo is drawn at screen size, so it must not depend on scale.
    small = poses.figure_gizmo(poses.new(100, 100, scale=120, body="male"))
    large = poses.figure_gizmo(poses.new(100, 100, scale=900, body="male"))
    check("gizmo radius is constant", small["radius"], large["radius"])


def test_mesh_model_exposes_the_pose_skeleton():
    model = mesh.load(poses.BODIES["male"], "male")
    check_true("mesh model has fingers", "index_03_l" in model.joint_index)
    check_true("draggable handle count", len(poses.pose_bones(model)) > 10)


# --------------------------------------------------------------------- scene
def test_scene_orbit_turns_every_figure_together():
    scene = poses.new_scene((800, 600))
    left = poses.new(0, 0, body="male")
    left["pos"] = [-0.8, 0.0, 0.0]
    poses.sync_spec(left, scene)
    right = poses.new(0, 0, body="male")
    right["pos"] = [0.8, 0.0, 0.0]
    poses.sync_spec(right, scene)
    check("both figures share one camera", left["view"], right["view"])
    check_true("figures sit at their own scene positions", left["x"] < right["x"])
    left_x0 = left["x"]
    right_x0 = right["x"]
    scene["yaw"] = 40.0
    poses.sync_scene([left, right], scene)
    check("the camera stays shared after an orbit", left["view"], right["view"])
    check("the shared yaw reached both", left["view"][0], 40.0)
    check_true("the whole scene moved, not one figure",
               left["x"] != left_x0 and right["x"] != right_x0)


def test_scene_has_a_single_ground_grid():
    scene = poses.new_scene((800, 600))
    segments = poses.scene_grid(scene)
    check_true("the scene floor has lines", len(segments) > 10)
    scene["pitch"] = 30.0
    tilted = poses.scene_grid(scene)
    check_true("the floor tilts with the scene", tilted != segments)


def test_figures_foreshorten_with_depth():
    scene = poses.new_scene((800, 600))
    near = poses.new(0, 0, body="male")
    near["pos"] = [0.0, 0.0, 2.0]
    poses.sync_spec(near, scene)
    far = poses.new(0, 0, body="male")
    far["pos"] = [0.0, 0.0, -2.0]
    poses.sync_spec(far, scene)

    def span(spec):
        ys = [p[1] for p in poses.joints_projected(spec).values()]
        return max(ys) - min(ys)

    check_true("the nearer skeleton is taller", span(near) > span(far) * 1.2)
    near_sprite = poses.render_sprite(near)[0]
    far_sprite = poses.render_sprite(far)[0]
    check_true("the nearer sprite is bigger", near_sprite.height > far_sprite.height)
    # The drawn sprite and the projected skeleton must agree on the size change.
    sprite_ratio = near_sprite.height / far_sprite.height
    check_close("sprite and skeleton foreshorten alike", sprite_ratio, span(near) / span(far), 0.15)


def test_moving_a_deep_figure_tracks_the_cursor():
    scene = poses.new_scene((900, 700))
    scene["yaw"] = 18.0
    spec = poses.new(0, 0, body="male")
    spec["pos"] = [0.0, 0.0, 1.5]
    poses.sync_spec(spec, scene)
    before = poses.joints_projected(spec)["pelvis"]
    poses.move_by_doc(spec, scene, spec["pos"], 40.0, -25.0)
    poses.sync_spec(spec, scene)
    after = poses.joints_projected(spec)["pelvis"]
    check_close("the figure follows the cursor x", after[0] - before[0], 40.0, 2.0)
    check_close("the figure follows the cursor y", after[1] - before[1], -25.0, 2.0)


def test_scene_anchor_shifts_the_whole_scene():
    scene = poses.new_scene((800, 600))
    a = poses.new(0, 0, body="male")
    a["pos"] = [-0.6, 0.0, 0.0]
    poses.sync_spec(a, scene)
    b = poses.new(0, 0, body="female")
    b["pos"] = [0.6, 0.0, 0.0]
    poses.sync_spec(b, scene)
    before = (poses.joints_projected(a)["pelvis"], poses.joints_projected(b)["pelvis"])
    scene["x"] += 40.0
    scene["y"] -= 25.0
    poses.sync_scene([a, b], scene)
    after = (poses.joints_projected(a)["pelvis"], poses.joints_projected(b)["pelvis"])
    for (ax, ay), (bx, by) in zip(before, after):
        check_close("pan moves the figure x", bx - ax, 40.0, 1e-6)
        check_close("pan moves the figure y", by - ay, -25.0, 1e-6)


def test_scene_zoom_scales_about_the_pivot():
    scene = poses.new_scene((800, 600))
    figure = poses.new(0, 0, body="male")
    figure["pos"] = [1.0, 0.0, 0.0]
    poses.sync_spec(figure, scene)
    pivot = (250.0, 180.0)
    before = poses.joints_projected(figure)["pelvis"]
    poses.set_scene_zoom(scene, scene["zoom"] * 2.0, pivot=pivot)
    poses.sync_spec(figure, scene)
    after = poses.joints_projected(figure)["pelvis"]
    check_close("zoom scales about the pivot x", after[0] - pivot[0],
                2.0 * (before[0] - pivot[0]), 0.5)
    check_close("zoom scales about the pivot y", after[1] - pivot[1],
                2.0 * (before[1] - pivot[1]), 0.5)


def test_scene_zoom_without_pivot_grows_about_the_anchor():
    scene = poses.new_scene((800, 600))
    figure = poses.new(0, 0, body="male")
    figure["pos"] = [1.0, 0.0, 0.0]
    poses.sync_spec(figure, scene)
    anchor = (scene["x"], scene["y"])
    before = poses.joints_projected(figure)["pelvis"]
    poses.set_scene_zoom(scene, scene["zoom"] * 2.0)
    poses.sync_spec(figure, scene)
    after = poses.joints_projected(figure)["pelvis"]
    check_close("the anchor stays put", scene["x"], anchor[0])
    check_close("the figure grows about the anchor", after[0] - anchor[0],
                2.0 * (before[0] - anchor[0]), 0.5)


def test_scene_zoom_clamps_to_the_limits():
    scene = poses.new_scene((800, 600))
    poses.set_scene_zoom(scene, 1e9)
    check("zoom clamps high", scene["zoom"], poses.MAX_ZOOM)
    poses.set_scene_zoom(scene, 0.0)
    check("zoom clamps low", scene["zoom"], poses.MIN_ZOOM)


def test_cursor_ray_inverts_the_projection():
    scene = poses.new_scene((800, 600))
    spec = poses.new(0, 0, body="male")
    spec["pos"] = [0.3, 0.0, 0.5]
    poses.sync_spec(spec, scene)
    model = mesh.load(poses.BODIES["male"], "male")
    local = model.joint_positions({})[model.joint_index["hand_l"]]
    (px, py), camera = poses._project(model, spec, local[None, :])
    doc = (spec["x"] + float(px[0]), spec["y"] + float(py[0]))
    camera_point = poses._camera_point(spec, model, doc[0], doc[1], float(camera[0, 2]))
    world = mesh.view_matrix(*spec["view"]).T @ camera_point
    model_point = world - np.asarray(spec["pos"])
    check_close("the ray inverts to the joint x", model_point[0], local[0], 1e-6)
    check_close("the ray inverts to the joint y", model_point[1], local[1], 1e-6)
    check_close("the ray inverts to the joint z", model_point[2], local[2], 1e-6)


def test_load_layer_poses_migrates_legacy_figures():
    legacy = [
        {"x": 300, "y": 400, "scale": 420, "body": "male",
         "view": [0.0, 0.0], "roll": 0.0, "joints": {}},
        {"x": 500, "y": 420, "scale": 420, "body": "female",
         "view": [0.0, 0.0], "roll": 0.0, "joints": {}},
    ]
    specs, scene = poses.load_layer_poses(legacy, None, (800, 600))
    check("both figures survive", len(specs), 2)
    check_true("a shared scene is built", "zoom" in scene and "yaw" in scene)
    check_close("the first figure keeps its old anchor", specs[0]["x"], 300.0, 1.0)
    check_true("the second figure keeps its place", specs[1]["x"] > specs[0]["x"])
    check("they share one camera", specs[0]["view"], specs[1]["view"])


def test_scene_round_trips_through_save():
    scene = poses.new_scene((800, 600))
    scene["yaw"] = 20.0
    figure = poses.new(0, 0, body="female")
    figure["pos"] = [0.5, 0.0, 0.3]
    poses.sync_spec(figure, scene)
    specs, restored = poses.load_layer_poses(
        [poses.clone(figure)], poses.clone_scene(scene), (800, 600),
    )
    check("the camera round-trips", restored["yaw"], 20.0)
    check_close("the world position round-trips", specs[0]["pos"][0], 0.5)
    check_close("depth round-trips", specs[0]["pos"][2], 0.3)


# --------------------------------------------------------------- primitives
def test_normalize_primitive_defaults_and_clamps():
    spec = poses.normalize({"kind": "primitive"})
    check("primitive kind", spec["kind"], poses.KIND_PRIMITIVE)
    check("primitive shape default", spec["shape"], poses.DEFAULT_SHAPE)
    check("primitive dim default", spec["dim"], [1.0, 1.0, 1.0])
    check("primitive rot default", spec["rot"], [0.0, 0.0, 0.0])
    check("primitive is primitive", poses.is_primitive(spec), True)
    bad = poses.normalize({"kind": "primitive", "shape": "blob", "size": 10 ** 9,
                           "dim": [0, -3, "x"], "opacity": 4, "color": None})
    check("shape falls back", bad["shape"], poses.DEFAULT_SHAPE)
    check("dim clamped low", bad["dim"][0], poses.MIN_DIM)
    check("dim negative becomes magnitude", bad["dim"][1], 3.0)
    check("dim non-number falls back", bad["dim"][2], 1.0)
    check("primitive opacity clamped", bad["opacity"], 1.0)
    check("primitive color default", bad["color"], poses.DEFAULT_PRIMITIVE_COLOR)


def test_figure_without_kind_is_figure():
    spec = poses.normalize({"x": 1, "y": 2})
    check("no kind means figure", spec["kind"], poses.KIND_FIGURE)
    check("figure is not primitive", poses.is_primitive(spec), False)


def test_new_and_clone_primitive():
    spec = poses.new_primitive(10, 20, shape="sphere")
    check("new primitive shape", spec["shape"], "sphere")
    check("new primitive pos", spec["pos"], [0.0, 0.0, 0.0])
    check_true("new primitive describes", "Sphere" in poses.describe(spec))
    copy = poses.clone(spec)
    check("clone is a fresh dict", copy is not spec, True)
    copy["dim"][0] = 9.0
    check("clone is deep", spec["dim"][0], 1.0)


def test_primitive_geometry_and_cache():
    for shape in poses.PRIMITIVES:
        model = mesh.primitive(shape)
        check_true(f"{shape} has triangles", model.triangles.shape[1] == 3)
        check_true(f"{shape} has vertices", len(model.position) > 0)
        check_true(f"{shape} normals match vertices",
                   len(model.normal) == len(model.position))
        check_true(f"{shape} is unit referenced", model.height == 1.0)
    box = mesh.primitive("box", dim=(2.0, 0.5, 1.0))
    check_close("box x span", float(box.bounds_max[0] - box.bounds_min[0]), 2.0)
    check_close("box y span", float(box.bounds_max[1] - box.bounds_min[1]), 0.5)
    check("cached primitive", mesh.primitive("box", (2.0, 0.5, 1.0)) is box, True)


def test_primitive_renders_and_scales_with_size():
    scene = poses.new_scene((800, 800))
    small = poses.new_primitive(0, 0, scene, shape="box", size=0.5)
    big = poses.new_primitive(0, 0, scene, shape="box", size=1.0)
    poses.sync_spec(small, scene)
    poses.sync_spec(big, scene)
    sprite_small = poses.render_sprite(small)
    sprite_big = poses.render_sprite(big)
    check_true("small primitive has ink", sprite_small is not None)
    check_true("bigger primitive is bigger", sprite_big[0].width > sprite_small[0].width)


def test_primitive_bounds_and_hit():
    scene = poses.new_scene((600, 600))
    spec = poses.new_primitive(0, 0, scene, shape="box")
    poses.sync_spec(spec, scene)
    left, top, right, bottom = poses.bounds(spec)
    check_true("bounds wrap the primitive", right > left and bottom > top)
    check_true("hit_body at the centre", poses.hit_body(spec, spec["x"], spec["y"]))
    check("hit_body far away", poses.hit_body(spec, -9000, -9000), False)
    check("no joints on a primitive", poses.joints_projected(spec), {})
    check("no bones on a primitive", poses.bones_projected(spec), [])
    check("joint_at is empty", poses.joint_at(spec, spec["x"], spec["y"], 10), None)
    check("overlay marks a primitive", poses.overlay(spec)["kind"],
          poses.KIND_PRIMITIVE)


def test_primitive_gizmo_move_scale_rotate():
    scene = poses.new_scene((600, 600))
    spec = poses.new_primitive(0, 0, scene, shape="cylinder")
    poses.sync_spec(spec, scene)
    geometry = poses.primitive_gizmo(spec, "move")
    check_true("gizmo has axes", len(geometry["axes"]) >= 2)
    check_true("gizmo has handles", len(geometry["handles"]) >= 2)
    check_true("gizmo has rings", len(geometry["rings"]) >= 1)
    first = sorted(geometry["handles"])[0]
    check("handle hit", poses.primitive_handle_at(
        spec, geometry, geometry["handles"][first][0],
        geometry["handles"][first][1], 12.0), first)
    base = poses.position(spec)
    poses.move_primitive(spec, base, "x", geometry, 40.0, 0.0)
    check_true("move changed x", abs(spec["pos"][0]) > 1e-3)
    poses.scale_primitive(spec, spec["dim"], "y", 2.0)
    check_close("scale doubled y", spec["dim"][1], 2.0)
    before = list(spec["rot"])
    poses.rotate_primitive(spec, "z", 0.5)
    check_true("rotate changed rot", spec["rot"] != before)
    ring_geometry = poses.primitive_gizmo(spec, "rotate")
    ring_axis = next(axis for axis, points in ring_geometry["rings"].items()
                     if points)
    ring_x, ring_y = ring_geometry["rings"][ring_axis][3]
    ring = poses.primitive_ring_angle(spec, ring_axis, ring_x, ring_y)
    check_true("ring angle available", ring is not None)


def test_primitive_pastes_and_composites():
    layer = Layer("P", (400, 400))
    spec = poses.new_primitive(0, 0, layer.pose_scene, shape="torus")
    poses.sync_spec(spec, layer.pose_scene)
    layer.poses.append(spec)
    layer.invalidate_poses()
    ink = np.asarray(layer.composited_image().getchannel("A"))
    check_true("composite has primitive ink", int(ink.max()) > 0)
    spec["opacity"] = 0.5
    layer.invalidate_poses()
    half = np.asarray(layer.composited_image().getchannel("A"))
    check_true("opacity halves alpha", int(half.max()) < int(ink.max()))


def test_project_save_persists_primitives():
    layer = fake_pose_layer("P")
    layer.poses = [poses.new_primitive(0, 0, shape="cone")]
    layer.poses[0]["pos"] = [0.0, 0.0, 0.0]
    layer.pose_scene = poses.new_scene((4, 4))
    canvas = SimpleNamespace(
        name="Canvas 1", width=4, height=4, active=0, layer_counter=1,
        group_counter=0, guides=[], groups=[], assets=[], asset_counter=0,
        inpaint_mask=None, layers=[layer],
    )
    app = SimpleNamespace(
        active_canvas=0, canvas_counter=1, color="#000000", brush_size=8,
        tools=SimpleNamespace(export_settings=lambda: {}),
        palette=["#000000"], selected_swatch=0, canvases=[canvas],
        _text_style=None,
    )
    with tempfile.TemporaryDirectory() as tmp:
        manifest = project.save(app, os.path.join(tmp, "prims.pai"))
    entry = manifest["canvases"][0]["layers"][0]["poses"][0]
    check("primitive kind persisted", entry["kind"], poses.KIND_PRIMITIVE)
    check("primitive shape persisted", entry["shape"], "cone")


def test_load_layer_poses_accepts_primitives():
    specs, _scene = poses.load_layer_poses(
        [{"kind": "primitive", "shape": "box", "pos": [1, 0, 0], "size": 1.0}],
        None, (200, 200),
    )
    check("kind-aware entry loads", specs[0]["kind"], poses.KIND_PRIMITIVE)
    check("kind-aware shape kept", specs[0]["shape"], "box")


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
