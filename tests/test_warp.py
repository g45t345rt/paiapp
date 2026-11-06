"""Unit tests for the warp engine (run: .venv\\Scripts\\python.exe tests\\test_warp.py)."""

import os
import sys

import math

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import warp  # noqa: E402
from paintapp.document import Layer  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def make_layer(size=(64, 64), block=(22, 22, 42, 42), color=(255, 0, 0, 255)):
    layer = Layer("t", size, (0, 0, 0, 0))
    layer.image.paste(color, block)
    return layer


def make_two_blocks():
    """A layer with a red block on the left and a green block on the right."""
    layer = Layer("t", (64, 64), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (6, 26, 16, 36))
    layer.image.paste((0, 255, 0, 255), (46, 26, 56, 36))
    return layer


def total_alpha(image):
    return int(np.asarray(image.getchannel("A")).sum())


def channel_centroid(image, channel):
    """The centre of mass of ``channel``, for tracking where content moved."""
    values = np.asarray(image.getchannel(channel), dtype=np.float64)
    total = values.sum()
    if total == 0:
        return (0.0, 0.0)
    ys, xs = np.mgrid[0:values.shape[0], 0:values.shape[1]]
    return (float((xs * values).sum() / total),
            float((ys * values).sum() / total))


def test_modes_registered():
    for mode in ("push", "expand", "pinch", "twirl_cw", "twirl_ccw"):
        check(f"mode {mode} registered", mode in warp.LIQUIFY_MODES, True)


def test_premultiply_round_trip():
    layer = make_layer()
    array = warp._premultiplied(layer.image)
    back = warp._unpremultiply(array)
    check("round trip preserves pixels", back.tobytes(), layer.image.tobytes())


def test_bilinear_at_integers():
    layer = make_layer()
    source = warp._premultiplied(layer.image)
    xs = np.array([[30.0, 10.0]])
    ys = np.array([[30.0, 10.0]])
    sampled = warp._bilinear(source, xs, ys)
    check("samples the block", sampled[0, 0, 3] > 0.9, True)
    check("samples transparency", sampled[0, 1, 3], 0.0)


def test_bilinear_outside_is_transparent():
    layer = make_layer()
    source = warp._premultiplied(layer.image)
    sampled = warp._bilinear(source, np.array([[-5.0]]), np.array([[-5.0]]))
    check("outside samples transparent", sampled[0, 0, 3], 0.0)


def test_liquify_push_moves_content():
    layer = make_layer()
    session = warp.Liquify(layer)
    before = session.image.tobytes()
    result = session.stamp(32, 32, 14, "push", 1.0, step_x=8.0, step_y=0.0)
    check("stamp returns an image", result is not None, True)
    check("push changed the image", result.tobytes() != before, True)
    check("push keeps ink", total_alpha(result) > 0, True)


def test_liquify_off_canvas_is_none():
    layer = make_layer()
    session = warp.Liquify(layer)
    check("far off-canvas dab is a no-op", session.stamp(500, 500, 10, "push", 1.0), None)


def test_liquify_modes_differ():
    outputs = {}
    for mode in ("expand", "pinch", "twirl_cw", "twirl_ccw"):
        layer = make_layer()
        session = warp.Liquify(layer)
        outputs[mode] = session.stamp(32, 32, 16, mode, 0.8).tobytes()
    check("expand differs from pinch", outputs["expand"] != outputs["pinch"], True)
    check("twirl cw differs from ccw", outputs["twirl_cw"] != outputs["twirl_ccw"], True)


def test_liquify_commit_matches_preview():
    layer = make_layer()
    session = warp.Liquify(layer)
    session.stamp(32, 32, 12, "twirl_cw", 0.5)
    check("commit returns the preview", session.commit().tobytes(), session.image.tobytes())


def test_mesh_warp_hit_and_reset():
    # A flat block resamples to itself, so give the block a gradient to make the
    # warp visible on it.
    layer = Layer("t", (64, 64), (0, 0, 0, 0))
    gradient = np.zeros((20, 20, 4), dtype=np.uint8)
    gradient[..., 0] = np.arange(20, dtype=np.uint8)[None, :] * 12
    gradient[..., 3] = 255
    layer.image.paste(Image.fromarray(gradient), (22, 22))
    box = layer.image.getbbox()
    session = warp.MeshWarp(layer, box, cols=4, rows=4)
    # The node at the top-left of the content box sits on the box corner.
    hit = session.hit_node(box[0], box[1], 3.0)
    check("hits the corner node", hit, (0, 0))
    original = session.image.tobytes()
    session.nudge_node(2, 2, 6.0, 4.0)
    check("nudge warps the image", session.image.tobytes() != original, True)
    session.reset()
    check("reset restores the layer", session.image.tobytes(), layer.image.tobytes())


def test_mesh_warp_node_position_tracks_offset():
    layer = make_layer()
    box = layer.image.getbbox()
    session = warp.MeshWarp(layer, box, cols=2, rows=2)
    rest = session.node_rest(1, 1)
    session.set_node_offset(1, 1, 5.0, -3.0)
    moved = session.node_position(1, 1)
    check("node moved in x", moved[0], rest[0] + 5.0)
    check("node moved in y", moved[1], rest[1] - 3.0)


def test_mesh_warp_grid_geometry_shape():
    layer = make_layer()
    box = layer.image.getbbox()
    session = warp.MeshWarp(layer, box, cols=3, rows=2)
    geometry = session.grid_geometry()
    check("one polyline per grid row", len(geometry["horizontal"]), 3)
    check("one polyline per grid col", len(geometry["vertical"]), 4)
    check("each row polyline length", len(geometry["horizontal"][0]), 4)


def test_puppet_pin_is_an_exact_mesh_vertex():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(31.25, 30.75)
    local = np.array([31.25, 30.75]) - session.doc_origin
    distance = np.linalg.norm(session.vertices - local, axis=1)
    check("clicked pin becomes a real mesh vertex",
          float(distance.min()) < 1e-8, True)


def test_puppet_pins_add_hit_remove():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    index = session.add_pin(30.0, 30.0)
    session.add_pin(40.0, 40.0)
    check("hits the placed pin", session.hit_pin(31.0, 30.0, 3.0), index)
    check("misses far from any pin", session.hit_pin(0.0, 0.0, 3.0), None)
    check("removes a pin", session.remove_pin(index), True)
    check("removing drops the count", len(session.pins), 1)
    check("removing a bad index is a no-op", session.remove_pin(9), False)


def test_puppet_pin_geometry_shape():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(30.0, 30.0)
    session.add_pin(40.0, 40.0)
    geometry = session.pin_geometry()
    check("one entry per pin", len(geometry), 2)
    check("rest and pos start equal", geometry[0]["rest"], geometry[0]["pos"])
    check("pin starts selected", geometry[0]["selected"], True)
    check("pin starts unrotated", geometry[0]["angle"], 0.0)


def test_arap_rotation_maps_rest_to_deformed():
    vertices, faces = warp.puppetmesh.triangular_mesh(30, 30, 10)
    angle = math.pi / 2
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    deformed = vertices @ rotation.T
    arap = warp.puppetmesh.ARAP(vertices, faces)
    rotations = arap._rotations(deformed)
    check("ARAP rotations follow the deformation direction",
          bool(np.allclose(rotations, rotation[None, :, :], atol=1e-8)), True)


def test_puppet_moved_content_does_not_leave_a_ghost():
    layer = Layer("t", (80, 60), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (10, 20, 30, 40))
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=100)
    session.add_pin(20.0, 30.0)
    session.move_pin(0, 60.0, 30.0)
    session.render()
    result = np.asarray(session.image)
    check("source pixels are cleared after moving", int(result[30, 20, 3]), 0)
    check("content arrives at the dragged pin", int(result[30, 60, 3]), 255)


def test_puppet_positive_rotation_turns_right_side_down():
    layer = Layer("t", (100, 100), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (55, 45, 75, 55))
    layer.image.paste((0, 255, 0, 255), (25, 45, 45, 55))
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=100)
    session.add_pin(50.0, 50.0)
    before = channel_centroid(layer.image, "R")
    session.set_pin_angle(0, math.pi / 2)
    after = channel_centroid(session.render(), "R")
    check("positive rotation turns right-side content downward",
          after[1] > before[1] + 10.0, True)


def test_puppet_rotation_turns_content():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(32.0, 32.0)
    flat = session.render().tobytes()
    session.set_pin_angle(0, math.pi / 2)
    check("rotation changes the image", session.render().tobytes() != flat, True)
    before = channel_centroid(layer.image, "R")
    after = channel_centroid(session.image, "R")
    check("rotation keeps the centre near the pin",
          abs(after[0] - before[0]) < 2.0 and abs(after[1] - before[1]) < 2.0, True)


def test_puppet_pin_rotates_its_region():
    layer = Layer("t", (500, 500), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (100, 190, 300, 210))
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=120)
    session.add_pin(150.0, 200.0)
    before = np.asarray(session.render())
    session.set_pin_angle(0, math.pi / 2)
    after = np.asarray(session.render())

    check("rotation changes the image",
          after.tobytes() != before.tobytes(), True)

    def vertical_extent(image):
        red = (image[..., 0] > 170) & (image[..., 3] > 0)
        ys, xs = np.nonzero(red)
        return int(ys.max() - ys.min())

    check("rotation turns the bar vertical",
          vertical_extent(after) > vertical_extent(before) + 60, True)


def test_puppet_selection_helpers():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(20.0, 20.0)
    session.add_pin(40.0, 40.0)
    session.select_only(0)
    check("select_only picks one", session.selected_indices(), [0])
    session.toggle_select(1)
    check("toggle adds to the selection", sorted(session.selected_indices()),
          [0, 1])
    session.clear_selection()
    check("clear empties the selection", session.selected_indices(), [])
    session.select_rect((0, 0, 30, 30))
    check("rect selects the pins inside", session.selected_indices(), [0])


def test_puppet_handle_follows_pin():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(32.0, 32.0)
    hx, hy = session.handle_pos(0, 10.0)
    check("handle sits above a rest pin", (round(hx), round(hy)), (32, 22))
    check("handle is hittable", session.hit_handle(hx, hy, 2.0, 10.0), 0)
    session.clear_selection()
    check("hidden handles are not hittable",
          session.hit_handle(hx, hy, 2.0, 10.0), None)


def test_puppet_mesh_clips_to_drawing():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(32.0, 32.0)
    edges = session.mesh_geometry()
    check("mesh has triangle edges", len(edges) > 0, True)
    left, top, right, bottom = layer.image.getbbox()
    near_drawing = all(
        left - 2 <= x <= right + 2 and top - 2 <= y <= bottom + 2
        for edge in edges for x, y in edge
    )
    check("mesh hugs the drawing and its antialiasing halo", near_drawing, True)


def test_puppet_mesh_skips_empty_regions():
    # An L-shaped drawing leaves the bottom-right of its bounding box empty.
    layer = Layer("t", (100, 100), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (10, 10, 30, 70))
    layer.image.paste((255, 0, 0, 255), (10, 10, 70, 30))
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    edges = session.mesh_geometry()
    in_empty = [
        (x, y) for edge in edges for x, y in edge if x > 34 and y > 34
    ]
    check("mesh skips the empty region", in_empty, [])


def test_puppet_mesh_is_a_regular_grid():
    layer = Layer("t", (200, 300), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (60, 20, 98, 280))
    layer.image.paste((255, 0, 0, 255), (102, 20, 140, 280))
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    edges = session.mesh_geometry()
    check("mesh has triangle edges", len(edges) > 0, True)
    left, top, right, bottom = layer.image.getbbox()
    inside = all(
        left <= x <= right and top <= y <= bottom
        for edge in edges for x, y in edge
    )
    check("mesh stays inside the content box", inside, True)


def test_puppet_mesh_follows_a_moved_pin():
    layer = Layer("t", (200, 300), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (60, 20, 140, 280))
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=120)
    session.add_pin(100.0, 150.0)
    before = session.mesh_geometry()
    session.move_pin(0, 160.0, 150.0)
    after = session.mesh_geometry()
    moved = [
        a for a, b in zip(before, after)
        if abs(a[0][0] - b[0][0]) > 1.0 or abs(a[0][1] - b[0][1]) > 1.0
    ]
    check("a dragged pin deforms the mesh", len(moved) > 0, True)


def test_content_mesh_respects_holes():
    alpha = np.zeros((50, 50), dtype=np.float32)
    alpha[5:45, 5:45] = 1.0
    alpha[18:32, 18:32] = 0.0
    vertices, faces = warp.puppetmesh.content_mesh(alpha, 8)
    content = alpha > 0.01
    empty_faces = [
        face for face in faces
        if not warp.puppetmesh._triangle_has_content(vertices[face], content)
    ]
    check("content mesh does not fill transparent holes", empty_faces, [])
    check("content mesh has silhouette triangles", len(faces) > 20, True)


def test_triangle_identity_does_not_duplicate_source_border():
    source = np.ones((5, 5, 4), dtype=np.float32)
    vertices, faces = warp.puppetmesh.triangular_mesh(5, 5, 5)
    result = warp.puppetmesh.warp_triangles(
        source, vertices, vertices, faces, 2)
    check("identity transfer preserves source pixels exactly",
          bool(np.array_equal(result[2:-2, 2:-2], source)), True)
    check("identity transfer does not smear the outer border",
          float(result[0, :, 3].sum()) + float(result[-1, :, 3].sum()), 0.0)


def test_content_mesh_identity_covers_thin_joints():
    source = np.zeros((60, 60, 4), dtype=np.float32)
    source[5:55, 8:14, 0] = 1.0
    source[45:55, 8:50, 0] = 1.0
    source[..., 3] = source[..., 0]
    vertices, faces = warp.puppetmesh.content_mesh(source[..., 3], 9)
    result = warp.puppetmesh.warp_triangles(
        source, vertices, vertices, faces, 0)
    missing = (source[..., 3] > 0) & (result[..., 3] == 0)
    check("identity transfer leaves no holes at thin joints",
          int(missing.sum()), 0)


def test_content_mesh_keeps_disconnected_parts_separate():
    alpha = np.zeros((30, 70), dtype=np.float32)
    alpha[5:25, 5:20] = 1.0
    alpha[5:25, 50:65] = 1.0
    vertices, faces = warp.puppetmesh.content_mesh(alpha, 6)
    centroids = vertices[faces].mean(axis=1)
    crosses_gap = (centroids[:, 0] > 25) & (centroids[:, 0] < 45)
    check("disconnected drawings have no bridging triangles",
          bool(crosses_gap.any()), False)


def test_triangle_priorities_control_real_overlap():
    source = np.zeros((4, 4, 4), dtype=np.float32)
    source[:, :2, 0] = 1.0
    source[:, 2:, 1] = 1.0
    source[..., 3] = 1.0
    rest = np.array([
        [0, 0], [2, 0], [0, 2],
        [2, 0], [4, 0], [2, 2],
    ], dtype=np.float64)
    deformed = np.array([
        [0, 0], [2, 0], [0, 2],
        [0, 0], [2, 0], [0, 2],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    green = warp.puppetmesh.warp_triangles(
        source, rest, deformed, faces, 0, priorities=[0, 1])
    red = warp.puppetmesh.warp_triangles(
        source, rest, deformed, faces, 0, priorities=[1, 0])
    check("higher-priority triangle renders on top",
          green[1, 0, 1] > 0.9 and green[1, 0, 0] < 0.1, True)
    check("reversing priority reverses the visible overlap",
          red[1, 0, 0] > 0.9 and red[1, 0, 1] < 0.1, True)


def test_delaunay_grid_triangulation():
    points = [(x, y) for x in range(0, 40, 10) for y in range(0, 40, 10)]
    triangles = warp._delaunay(points)
    check("triangulates a grid", len(triangles), 18)
    check("only uses the real points",
          all(max(t) < len(points) for t in triangles), True)


def test_puppet_order_priority_overlaps():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=40)
    first = session.add_pin(24.0, 32.0)
    second = session.add_pin(40.0, 32.0)
    check("pins start at order 0", session.pins[first]["order"], 0)
    session.raise_pins([second])
    priorities = session._face_priorities()
    check("raised pin gives its triangles render priority",
          float(priorities.max()) > 0.0, True)


def test_puppet_correct_rotation_preserves_pin_anchors():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=40)
    turned = session.add_pin(24.0, 32.0)
    neighbour = session.add_pin(40.0, 32.0)
    session.set_pin_angle(turned, 0.6)
    session.correct_rotations(hold=[turned])
    check("manipulated pin keeps its angle", session.pins[turned]["angle"], 0.6)
    check("unmoved pin remains an anchor",
          session.pins[neighbour]["angle"], 0.0)


def test_puppet_toggle_correct_flag():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    index = session.add_pin(30.0, 30.0)
    check("pins correct by default", session.pins[index]["correct"], True)
    session.toggle_correct(index)
    check("toggle_correct turns it off", session.pins[index]["correct"], False)
    session.toggle_correct(index)
    check("toggle_correct turns it back on", session.pins[index]["correct"], True)


def test_puppet_move_selected_pins_together():
    layer = make_two_blocks()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(11.0, 31.0)
    session.add_pin(51.0, 31.0)
    session.select_only(0)
    session.toggle_select(1)
    for index in session.selected_indices():
        x, y = session.pins[index]["pos"]
        session.set_pin_pos(index, x + 5.0, y)
    session.render()
    red = channel_centroid(session.image, "R")[0]
    green = channel_centroid(session.image, "G")[0]
    check("both selected blocks moved right",
          red > channel_centroid(layer.image, "R")[0] + 3.0
          and green > channel_centroid(layer.image, "G")[0] + 3.0, True)


def test_puppet_identity_preserves_pixels():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(32.0, 32.0)
    check("a pin at rest leaves pixels alone",
          session.render().tobytes(), layer.image.tobytes())


def test_puppet_single_pin_translates():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    before = channel_centroid(layer.image, "R")
    index = session.add_pin(32.0, 32.0)
    session.move_pin(index, 40.0, 32.0)
    after = channel_centroid(session.image, "R")
    check("pin drag shifts content right", after[0] > before[0] + 6.0, True)
    check("pin drag keeps ink", total_alpha(session.image) > 0, True)


def test_puppet_unmoved_pin_anchors():
    layer = make_two_blocks()
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=20)
    red_before = channel_centroid(layer.image, "R")
    green_before = channel_centroid(layer.image, "G")
    session.add_pin(11.0, 31.0)
    session.add_pin(51.0, 31.0)
    session.move_pin(0, 21.0, 31.0)
    red_after = channel_centroid(session.image, "R")
    green_after = channel_centroid(session.image, "G")
    check("moved pin drags its block", red_after[0] - red_before[0] > 5.0, True)
    check("fixed pin anchors its block",
          abs(green_after[0] - green_before[0]) < 2.0, True)


def test_puppet_inactive_disconnected_part_stays_fixed():
    layer = make_two_blocks()
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=15)
    session.add_pin(11.0, 31.0)
    session.move_pin(0, 21.0, 31.0)
    green = channel_centroid(session.image, "G")[0]
    check("a disconnected part without pins stays fixed",
          abs(green - channel_centroid(layer.image, "G")[0]) < 0.1, True)


def test_puppet_drag_bends_connected_drawing_locally():
    # A single connected bar spans the whole width, like one connected figure.
    layer = Layer("t", (400, 120), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (40, 50, 360, 70))
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=55)
    near = session.add_pin(70.0, 60.0)
    session.add_pin(330.0, 60.0)
    session.set_pin_pos(near, 70.0, 20.0)
    session.render()
    red = np.asarray(session.image.getchannel("R")) > 170
    ys, xs = np.nonzero(red)
    check("the dragged end follows its pin", int(ys.min()) < 45, True)
    check("the far end of the same component stays anchored",
          abs(int(xs.max()) - 359) < 12, True)



    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(32.0, 32.0)
    session.move_pin(0, 40.0, 32.0)
    check("move changed the image",
          session.image.tobytes() != layer.image.tobytes(), True)
    session.reset()
    check("reset restores the image", session.image.tobytes(), layer.image.tobytes())
    session.move_pin(0, 40.0, 32.0)
    session.clear_pins()
    check("clear drops the pins", len(session.pins), 0)
    check("clear restores the image", session.image.tobytes(), layer.image.tobytes())


def test_two_step_arap_hits_pin_targets():
    layer = Layer("t", (300, 300), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (60, 120, 240, 180))
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(100.0, 150.0)
    session.add_pin(200.0, 150.0)
    session.set_pin_pos(0, 140.0, 150.0)
    deformed = session._ensure_solved()
    constrained, targets = session._constraints()
    error = float(np.abs(deformed[constrained] - targets).max())
    check("two-step ARAP lands on both pin targets", error < 0.01, True)
    check("two-step ARAP deforms unpinned vertices",
          bool(np.any(np.abs(deformed - session.vertices) > 1.0)), True)


def test_puppet_long_drag_does_not_duplicate():
    layer = Layer("t", (400, 200), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (50, 80, 90, 120))
    before = total_alpha(layer.image)
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=60)
    session.add_pin(70.0, 100.0)
    session.move_pin(0, 130.0, 100.0)
    session.render()
    red = np.asarray(session.image.getchannel("R")) > 170
    ys, xs = np.nonzero(red)
    check("long drag keeps the ink (no fold duplication)",
          abs(total_alpha(session.image) - before) < before * 0.10, True)
    check("long drag does not widen the block",
          int(xs.max() - xs.min()) <= 45, True)
    check("long drag carries the block to the pin", int(xs.max()) >= 145, True)


def test_puppet_hit_ring():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(32.0, 32.0)
    check("the outer ring is hittable", session.hit_ring(12.0, 32.0, 4.0, 20.0), 0)
    check("the ring misses the pin centre",
          session.hit_ring(32.0, 32.0, 4.0, 20.0), None)
    session.clear_selection()
    check("a hidden ring is not hittable",
          session.hit_ring(12.0, 32.0, 4.0, 20.0), None)


def test_puppet_selection_limits_mesh_to_selected_drawing():
    layer = make_two_blocks()
    clip = Image.new("L", layer.image.size, 0)
    clip.paste(255, (0, 0, 32, 64))
    session = warp.PuppetWarp(layer, layer.image.getbbox(), clip=clip)
    document_x = session.vertices[:, 0] + session.doc_origin[0]
    check("selection mesh excludes unselected content",
          bool((document_x < 25).all()), True)
    check("pin hit area includes selected drawing", session.contains(11, 31), True)
    check("pin hit area excludes unselected drawing", session.contains(51, 31), False)
    check("pin hit area excludes transparent canvas", session.contains(1, 1), False)


def test_puppet_clip_limits_warp():
    layer = Layer("t", (400, 200), (0, 0, 0, 0))
    layer.image.paste((255, 0, 0, 255), (50, 80, 90, 120))
    clip = Image.new("L", (400, 200), 0)
    clip.paste(255, (0, 0, 200, 200))
    session = warp.PuppetWarp(
        layer, layer.image.getbbox(), influence=60, clip=clip)
    session.add_pin(70.0, 100.0)
    session.move_pin(0, 130.0, 100.0)
    session.render()
    result = np.asarray(session.image)
    original = np.asarray(session.original)
    check("clip keeps pixels outside the selection",
          bool((result[:, 200:] == original[:, 200:]).all()), True)
    check("clip warps pixels inside the selection",
          bool((result[:, :200] != original[:, :200]).any()), True)


def test_mesh_partial_render_matches_full():
    layer = make_layer((120, 120), (30, 30, 90, 90))
    box = layer.image.getbbox()
    partial = warp.MeshWarp(layer, box, 4, 4)
    partial.set_node_offset(2, 2, 9.0, -6.0)
    full = warp.MeshWarp(layer, box, 4, 4)
    full.nodes[2, 2, 0] = 9.0
    full.nodes[2, 2, 1] = -6.0
    full.render()
    check("partial render equals full", partial.image.tobytes(), full.image.tobytes())


def test_puppet_preview_is_reduced_and_exact_on_render():
    layer = make_layer((600, 440), (100, 100, 500, 340))
    box = layer.image.getbbox()
    session = warp.PuppetWarp(layer, box, influence=150)
    session.add_pin(200, 220)
    session.add_pin(400, 220)
    session.render()
    check("exact render clears the preview flag", session.previewed, False)
    session.move_pin(0, 260, 220)
    check("drag previews at reduced resolution", session.previewed, True)
    preview = np.asarray(session.image).astype(int)
    exact = np.asarray(session.render()).astype(int)
    check("render clears the preview flag", session.previewed, False)
    check("preview is close to the exact render",
          float(np.abs(preview - exact).mean()) < 5.0, True)


def test_puppet_small_layer_renders_exact():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=150)
    session.add_pin(32.0, 32.0)
    session.move_pin(0, 40.0, 32.0)
    check("a small layer skips the preview", session.previewed, False)


def test_puppet_rotation_never_overrides_other_pin():
    layer = make_layer()
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=40)
    session.add_pin(29, 32)
    session.add_pin(35, 32)
    session.set_pin_angle(0, 0.8)
    before = session._ensure_solved().copy()
    session.raise_pins([0])
    after = session._ensure_solved()
    check("stacking order does not alter deformation",
          bool(np.allclose(before, after)), True)
    for pin in session.pins:
        rest = np.asarray(pin["rest"]) - session.doc_origin
        nearest = np.argmin(np.linalg.norm(session.vertices - rest, axis=1))
        check("rotating neighbours preserve exact pin anchors",
              bool(np.allclose(after[nearest] + session.doc_origin,
                               pin["pos"], atol=0.01)), True)


def test_puppet_preview_geometry_and_cache():
    layer = make_layer((700, 500), (101, 103, 504, 346))
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(200, 220)
    session.add_pin(400, 220)
    session.set_pin_pos(0, 240, 240)
    session.preview()
    vertices = session.deformed.copy()
    mesh = session.mesh_geometry()
    cached = session._live
    session.set_pin_pos(0, 250, 240)
    session.preview()
    check("drag reuses the live full-resolution preview",
          session._live is cached, True)
    session.set_pin_pos(0, 240, 240)
    session.render()
    check("preview and final use identical deformation",
          bool(np.allclose(session.deformed, vertices)), True)
    check("preview and final display the same mesh", session.mesh_geometry(), mesh)


def test_puppet_preview_translation_alignment():
    layer = make_layer((700, 500), (101, 103, 504, 346))
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(200, 220)
    session.set_pin_pos(0, 232, 236)
    preview = session.preview()
    exact = session.render()
    check("odd-sized preview translation stays aligned",
          bool(np.allclose(channel_centroid(preview, "R"),
                           channel_centroid(exact, "R"), atol=2)), True)
    preview_box, exact_box = preview.getbbox(), exact.getbbox()
    check("preview does not squeeze the drawing",
          abs((preview_box[2] - preview_box[0]) -
              (exact_box[2] - exact_box[0])) <= 8, True)


def test_puppet_far_drag_is_bounded():
    layer = make_layer((700, 500), (101, 103, 504, 346))
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(200, 220)
    session.set_pin_pos(0, 1_000_000, 1_000_000)
    preview = session.preview()
    exact = session.render()
    check("preview stays on the canvas",
          preview.getbbox() is not None and preview.size == (700, 500), True)
    check("exact render stays on the canvas",
          exact.getbbox() is not None and exact.size == (700, 500), True)
    check("off-canvas drag keeps the anchored drawing",
          total_alpha(exact) > 0, True)


def test_puppet_add_pin_on_deformed_drawing():
    layer = make_layer((300, 150), (30, 40, 90, 100))
    session = warp.PuppetWarp(layer, layer.image.getbbox())
    session.add_pin(60, 70)
    session.move_pin(0, 160, 70)
    check("pins can be placed on translated artwork", session.contains(170, 80), True)
    check("empty original position is not hittable", session.contains(70, 80), False)
    index = session.add_pin(170, 80)
    check("new pin references original material position",
          bool(np.allclose(session.pins[index]["rest"], (70, 80))), True)
    session.render()
    check("adding an anchor does not snap translated art back",
          bool(np.allclose(channel_centroid(session.image, "R"),
                           (159.5, 69.5), atol=1)), True)


def test_puppet_drag_coalescing_finishes_latest_state():
    from unittest.mock import Mock
    from paintapp.drawing import DrawingController

    controller = DrawingController.__new__(DrawingController)
    controller.app = Mock()
    controller.app.root.after.return_value = "preview-job"
    controller._puppet = Mock()
    controller._puppet.previewed = False
    controller._puppet_preview_job = None
    controller._puppet_mode = "move"
    controller._update_puppet_overlay = Mock()
    session = controller._puppet
    controller._puppet_refresh()
    controller._puppet_refresh()
    check("pointer events queue only one preview",
          controller.app.root.after.call_count, 1)
    check("queued drag does not render stale positions", session.preview.call_count, 0)
    controller._puppet_release()
    check("release renders latest state even before preview runs",
          session.render.call_count, 1)
    check("release cancels queued preview",
          controller.app.root.after_cancel.call_args.args, ("preview-job",))
    check("release clears preview timer", controller._puppet_preview_job, None)
    controller._puppet_refresh()
    controller._render_puppet_preview()
    check("timer renders one latest-state preview", session.preview.call_count, 1)
    check("completed timer clears its handle", controller._puppet_preview_job, None)


def test_triangle_transparent_overlap_and_shared_edges():
    source = np.zeros((6, 12, 4), dtype=np.float32)
    source[:, :6] = (1, 0, 0, 1)
    rest = np.array([[0, 0], [5, 0], [0, 5],
                     [6, 0], [11, 0], [6, 5]], dtype=float)
    deformed = rest.copy()
    deformed[3:, 0] -= 6
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    transparent = warp.puppetmesh.warp_triangles(source, rest, deformed, faces, 0)
    check("transparent foreground does not erase overlap",
          bool(np.allclose(transparent[1, 1], (1, 0, 0, 1))), True)
    source[:, 6:] = (0, 0.5, 0, 0.5)
    translucent = warp.puppetmesh.warp_triangles(source, rest, deformed, faces, 0)
    check("translucent overlap uses premultiplied source-over",
          bool(np.allclose(translucent[1, 1], (0.5, 0.5, 0, 1))), True)
    source = np.full((6, 6, 4), 0.5, dtype=np.float32)
    rest = np.array([[0, 0], [5, 0], [0, 5], [5, 5]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    result = warp.puppetmesh.warp_triangles(source, rest, rest, faces, 0)
    check("shared edges composite only once without boundary holes",
          bool(np.allclose(result, source)), True)
    bounded = warp.puppetmesh.warp_triangles(
        source, rest, rest, faces, 0, bounds=(1, 2, 4, 5))
    check("bounded raster matches full raster crop",
          bool(np.allclose(bounded, result[2:5, 1:4])), True)


def test_triangle_batches_preserve_translucent_boundaries():
    source = np.full((400, 400, 4), 0.5, dtype=np.float32)
    rest = np.array([[0, 0], [399, 0], [0, 399], [399, 399]], dtype=float)
    # Each bounding rectangle is larger than the renderer's candidate batch.
    faces = np.array([[0, 2, 1], [1, 2, 3]])
    result = warp.puppetmesh.warp_triangles(source, rest, rest, faces)
    check("batch boundaries preserve translucent identity and reversed winding",
          bool(np.allclose(result, source)), True)
    cropped = warp.puppetmesh.warp_triangles(
        source, rest, rest, faces, bounds=(-2, -3, 201, 202))
    check("negative raster bounds preserve transparent margins",
          bool((cropped[:3] == 0).all() and (cropped[:, :2] == 0).all()), True)
    check("negative bounds preserve source pixel coordinates",
          bool(np.allclose(cropped[3:, 2:], source[:202, :201])), True)


def test_puppet_render_is_deterministic():
    layer = Layer("t", (240, 200), (0, 0, 0, 0))
    gradient = np.zeros((160, 200, 4), dtype=np.uint8)
    gradient[..., 1] = np.arange(200, dtype=np.uint8)[None, :]
    gradient[..., 2] = np.arange(160, dtype=np.uint8)[:, None]
    gradient[..., 3] = 255
    layer.image.paste(Image.fromarray(gradient), (20, 20))
    session = warp.PuppetWarp(layer, layer.image.getbbox(), influence=120)
    session.add_pin(80.0, 100.0)
    session.add_pin(160.0, 100.0)
    session.set_pin_pos(0, 110.0, 122.0)
    session.set_pin_angle(1, 0.6)
    first = session.render().tobytes()
    session._dirty = True
    second = session.render().tobytes()
    check("puppet render is deterministic", first == second, True)


def test_mesh_gpu_matches_cpu():
    from paintapp import gpuimage
    if not gpuimage.available():
        print("ok  mesh gpu unavailable (skipped)")
        return
    layer = Layer("t", (240, 200), (0, 0, 0, 0))
    gradient = np.zeros((160, 200, 4), dtype=np.uint8)
    gradient[..., 0] = np.arange(200, dtype=np.uint8)[None, :]
    gradient[..., 1] = np.arange(160, dtype=np.uint8)[:, None]
    gradient[..., 2] = 90
    gradient[..., 3] = np.clip(
        np.arange(200, dtype=np.int16)[None, :] + 120, 0, 255
    ).astype(np.uint8)
    layer.image.paste(Image.fromarray(gradient), (20, 20))
    box = layer.image.getbbox()
    session = warp.MeshWarp(layer, box, 4, 4)
    session.nodes[1, 1] = (11.0, -7.0)
    session.nodes[2, 3] = (-9.0, 5.0)
    rect = (0, 0, box[2] - box[0], box[3] - box[1])
    gpu = session._render_patch_gpu(rect)
    check("mesh gpu patch produced an image", gpu is not None, True)
    if gpu is None:
        return
    cpu = session._render_patch_cpu(rect)
    difference = np.abs(
        np.asarray(gpu, dtype=int) - np.asarray(cpu, dtype=int)
    )
    check("mesh gpu matches cpu (mean)", bool(difference.mean() < 2.0), True)
    check("mesh gpu matches cpu (max)", int(difference.max()) <= 2, True)


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
