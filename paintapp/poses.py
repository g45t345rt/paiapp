"""Poses: editable 3D characters that a layer can hold and the tool can pose.

A pose object is a plain spec dict kept on a layer and rendered to pixels at
composite time, exactly as ``vector`` keeps strokes and ``textrender`` keeps text
(see ``Layer.composited_image``). Keeping the description rather than the pixels
is what makes a figure editable: the Poses tool drags its skeleton into shape and
the project save round-trips it.

The figure is a real skinned mesh from ``paintapp/assets/models`` (CC0 Quaternius
Universal Base Characters). Two bodies ship, a male and a female, sharing an
identical 65-joint humanoid rig with full finger bones::

    {"pos": [x, y, z], "size", "spin", "body",
     "joints": {bone: [rx, ry, rz]}, "color", "opacity"}

``pos`` is the figure's world position in a shared scene (``size`` is its size
multiplier and ``spin`` turns it in the picture plane), ``body`` picks the mesh.
``joints`` holds a local Euler delta per bone, applied after the bone's bind
rotation, so an empty dict is the bind (T) pose.

All figures on a layer share one scene camera (:func:`new_scene`): orbiting or
tilting the scene (:func:`sync_scene`) turns every figure together and there is a
single ground grid (:func:`scene_grid`). Each spec also carries the derived
``x``/``y``/``scale``/``view``/``roll`` the projection helpers below read, kept
current by :func:`sync_spec`, so rendering a spec needs no scene object.

Posing is *aim based*: :func:`aim_at` and the preset tables solve for the rotation
that points a bone in a given direction, so joints can be posed without knowing
each bone's private axes, and presets are expressed as world directions that can
be checked numerically.

Rendering goes through :mod:`paintapp.mesh`, which draws on the GPU
(:mod:`paintapp.mesh_gpu`) whenever an OpenGL context is available and otherwise
falls back to the numpy rasteriser.
"""

import math
import os

import numpy as np
from PIL import Image

from . import mesh

BODIES = {
    "male": os.path.join(mesh.MODEL_DIR, "Superhero_Male_FullBody.gltf"),
    "female": os.path.join(mesh.MODEL_DIR, "Superhero_Female_FullBody.gltf"),
}
BODY_ORDER = ("female", "male")
DEFAULT_BODY = "female"

MIN_SCALE = 60
MAX_SCALE = 2000
DEFAULT_SCALE = 420
DEFAULT_COLOR = "#c9a06a"

# ------------------------------------------------------------- object kinds
# A layer's scene holds objects of two kinds. A ``figure`` is a posed mannequin
# (the Poses tool's original content); a ``primitive`` is a rigid parametric
# solid (box, sphere, ...) that the 3D viewport can add as a prop. Both share the
# layer's one scene camera, ground grid and document-space projection, so the
# tool treats them uniformly (see :func:`outline`/the gizmo helpers).
KIND_FIGURE = "figure"
KIND_PRIMITIVE = "primitive"

PRIMITIVES = tuple(mesh.PRIMITIVE_SHAPES)
PRIMITIVE_LABELS = {
    "box": "Box", "sphere": "Sphere", "cylinder": "Cylinder",
    "cone": "Cone", "plane": "Plane", "torus": "Torus",
}
DEFAULT_SHAPE = "box"
DEFAULT_PRIMITIVE_COLOR = "#8f9299"
DEFAULT_PRIMITIVE_SIZE = 1.0
# The per-axis size multiplier's clamp, so a primitive cannot be scaled to nothing
# or flipped inside out by a runaway drag.
MIN_DIM = 0.02
MAX_DIM = 50.0
# The whole-object gizmo is drawn at a fixed screen size around the object centre.
PRIMITIVE_GIZMO_RADIUS = 88.0
# The selection outline is drawn just outside the object's projected bounds.
PRIMITIVE_OUTLINE = "#8f9299"

# The scene is one shared 3D space a layer's figures live in (see
# :func:`new_scene`). ``zoom`` is the scene's document pixels per model unit,
# chosen so a figure of size 1 is about ``DEFAULT_SCALE`` tall; ``x``/``y`` are
# the document position the world origin projects to, and ``yaw``/``pitch``/
# ``roll`` orbit and tilt the whole scene.
DEFAULT_ZOOM = DEFAULT_SCALE / 1.8
MIN_ZOOM = 20.0
MAX_ZOOM = 4000.0
# The world plane the figures stand on (the model's feet level). The single
# scene ground grid is drawn here.
GROUND = 0.0

# Mild perspective so limbs foreshorten when the figure is turned; the focal
# length is in model units, where the figure is about 1.8 tall.
PERSPECTIVE = 7.0

# Interactive drags render a decimated copy of the mesh, or the full mesh when
# "best" drag quality is selected. The decimator keeps vertices on the real
# surface, so a drag silhouette is within about a pixel of the committed render.
DRAG_QUALITIES = ("fast", "best")
DEFAULT_DRAG_QUALITY = "fast"

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
_PHALANX_WEIGHTS = (1.0, 1.05, 0.85)

# Dragging one of these handles solves the whole limb so the handle reaches the
# cursor (two-bone inverse kinematics) instead of aiming a single bone. The value
# is the chain to rotate, distal joint first.
IK_CHAIN = {
    "hand_l": ("lowerarm_l", "upperarm_l"),
    "hand_r": ("lowerarm_r", "upperarm_r"),
    "foot_l": ("calf_l", "thigh_l"),
    "foot_r": ("calf_r", "thigh_r"),
    "ball_l": ("calf_l", "thigh_l"),
    "ball_r": ("calf_r", "thigh_r"),
}

# Hinge joints bend one way only: elbows toward the figure's front, knees toward
# its back. The axis comes from whatever aim is applied, so only the direction a
# bend may travel in and the maximum bend need stating.
HINGE_LIMITS = {
    "lowerarm_l": ((0.0, 0.0, 1.0), 150.0),
    "lowerarm_r": ((0.0, 0.0, 1.0), 150.0),
    "calf_l": ((0.0, 0.0, -1.0), 150.0),
    "calf_r": ((0.0, 0.0, -1.0), 150.0),
}

_preview = False
_drag_quality = DEFAULT_DRAG_QUALITY


def set_preview(active):
    """Render a cheap interactive proxy while a drag is in flight."""
    global _preview
    _preview = bool(active)


def preview_active():
    return _preview


def set_drag_quality(quality):
    """What an in-flight drag draws: the decimated proxy or the full mesh."""
    global _drag_quality
    _drag_quality = quality if quality in DRAG_QUALITIES else DEFAULT_DRAG_QUALITY


def drag_quality():
    return _drag_quality


def gpu_active():
    """True when figures are drawn on the GPU.

    With the GPU a full-detail frame costs a couple of milliseconds, so a drag
    needs neither the decimated proxy nor the redraw throttle.
    """
    return mesh.gpu_available()


def renderer_name():
    return "GPU" if mesh.gpu_available() else "CPU"


# ------------------------------------------------------------------- models
def model_for(body):
    """The loaded mesh model for a body name, or ``None`` when it is unavailable."""
    path = BODIES.get(body) or BODIES[DEFAULT_BODY]
    try:
        return mesh.load(path, body)
    except (OSError, ValueError):
        return None


def available_bodies():
    return [name for name in BODY_ORDER if os.path.exists(BODIES[name])]


def object_model(spec):
    """The render model for any object kind: a skinned body or a rigid primitive."""
    if is_primitive(spec):
        return mesh.primitive(
            spec.get("shape", DEFAULT_SHAPE),
            spec.get("dim") or (1.0, 1.0, 1.0),
            spec.get("rot") or (0.0, 0.0, 0.0),
        )
    return model_for(spec.get("body", DEFAULT_BODY))


def pose_bones(model):
    """The draggable skeleton: non-digit bones that have an aim direction."""
    return [
        bone for bone in model.joint_names
        if not any(digit in bone for digit in DIGITS)
        and model.aim_child[model.joint_index[bone]] >= 0
    ]


def _is_left(bone):
    return bone.endswith("_l")


def _swap_side(bone):
    if bone.endswith("_l"):
        return bone[:-2] + "_r"
    if bone.endswith("_r"):
        return bone[:-2] + "_l"
    return bone


def _depth_order(model, bones):
    """``bones`` sorted so every bone follows its ancestors."""
    ordered = []
    for bone in bones:
        chain = []
        index = model.joint_index.get(bone)
        while index is not None and index >= 0:
            name = model.joint_names[index]
            chain.append(name)
            index = model.joint_parent[index]
        ordered.extend(reversed(chain))
    seen = set()
    result = []
    for name in ordered:
        if name in bones and name not in seen:
            seen.add(name)
            result.append(name)
    return result


# -------------------------------------------------------------------- specs
def _vec3(raw, default=(0.0, 0.0, 0.0)):
    """Coerce ``raw`` to a 3-float list, falling back to ``default`` per slot."""
    out = [float(v) for v in default]
    if isinstance(raw, (list, tuple)):
        for slot in range(min(3, len(raw))):
            try:
                out[slot] = float(raw[slot])
            except (TypeError, ValueError):
                pass
    return out


def kind_of(spec):
    """The object kind of ``spec``: ``"primitive"`` or the default ``"figure"``."""
    if isinstance(spec, dict) and spec.get("kind") == KIND_PRIMITIVE:
        return KIND_PRIMITIVE
    return KIND_FIGURE


def is_primitive(spec):
    """True when ``spec`` is a rigid primitive object rather than a mannequin."""
    return kind_of(spec) == KIND_PRIMITIVE


def _normalize_primitive(spec):
    """Coerce a primitive spec to a complete, valid object (or ``None``)."""
    shape = spec.get("shape")
    if shape not in mesh.PRIMITIVE_SHAPES:
        shape = DEFAULT_SHAPE
    try:
        size = float(spec.get("size", DEFAULT_PRIMITIVE_SIZE))
    except (TypeError, ValueError):
        size = DEFAULT_PRIMITIVE_SIZE
    size = max(0.01, size)
    dim = [max(MIN_DIM, min(MAX_DIM, abs(v) or MIN_DIM))
           for v in _vec3(spec.get("dim"), (1.0, 1.0, 1.0))]
    rot = _vec3(spec.get("rot"))
    try:
        opacity = float(spec.get("opacity", 1.0))
    except (TypeError, ValueError):
        opacity = 1.0
    view = [0.0, 0.0]
    raw_view = spec.get("view")
    if isinstance(raw_view, (list, tuple)):
        for slot in range(min(2, len(raw_view))):
            try:
                view[slot] = float(raw_view[slot])
            except (TypeError, ValueError):
                pass
    try:
        roll = float(spec.get("roll", 0.0))
    except (TypeError, ValueError):
        roll = 0.0
    try:
        x = float(spec.get("x", 0.0))
        y = float(spec.get("y", 0.0))
    except (TypeError, ValueError):
        x = y = 0.0
    try:
        scale = float(spec.get("scale", DEFAULT_SCALE))
    except (TypeError, ValueError):
        scale = DEFAULT_SCALE
    return {
        "kind": KIND_PRIMITIVE,
        "shape": shape,
        "pos": _vec3(spec.get("pos")),
        "size": size,
        "dim": dim,
        "rot": rot,
        "x": x,
        "y": y,
        "scale": scale,
        "view": view,
        "roll": roll,
        "color": str(spec.get("color") or DEFAULT_PRIMITIVE_COLOR),
        "opacity": max(0.0, min(1.0, opacity)),
    }


def normalize(spec):
    """Coerce ``spec`` to a complete, valid pose (or ``None``).

    Accepts both the scene-native fields (``pos``/``size``/``spin``) and the
    legacy per-figure ``view``/``roll``/``scale``/``x``/``y``, so a spec read
    from an older project still loads; :func:`sync_spec` refreshes the derived
    fields once the layer's scene is known. A spec carrying ``kind:
    "primitive"`` is validated by :func:`_normalize_primitive` instead, and a
    spec without a ``kind`` is treated as the original mannequin figure.
    """
    if not isinstance(spec, dict):
        return None
    if spec.get("kind") == KIND_PRIMITIVE:
        return _normalize_primitive(spec)
    try:
        x = float(spec.get("x", 0.0))
        y = float(spec.get("y", 0.0))
    except (TypeError, ValueError):
        return None
    try:
        scale = float(spec.get("scale", DEFAULT_SCALE))
    except (TypeError, ValueError):
        scale = DEFAULT_SCALE
    scale = max(MIN_SCALE, min(MAX_SCALE, scale))
    body = spec.get("body")
    if body not in BODIES:
        body = DEFAULT_BODY
    view = [0.0, 0.0]
    raw_view = spec.get("view")
    if isinstance(raw_view, (list, tuple)):
        for slot in range(min(2, len(raw_view))):
            try:
                view[slot] = float(raw_view[slot])
            except (TypeError, ValueError):
                pass
    try:
        roll = float(spec.get("roll", 0.0))
    except (TypeError, ValueError):
        roll = 0.0
    try:
        size = float(spec.get("size", 1.0))
    except (TypeError, ValueError):
        size = 1.0
    size = max(0.01, size)
    try:
        spin = float(spec.get("spin", 0.0))
    except (TypeError, ValueError):
        spin = 0.0
    joints = {}
    raw_joints = spec.get("joints")
    if isinstance(raw_joints, dict):
        for bone, angles in raw_joints.items():
            if not isinstance(angles, (list, tuple)):
                continue
            values = [0.0, 0.0, 0.0]
            for slot in range(min(3, len(angles))):
                try:
                    values[slot] = float(angles[slot])
                except (TypeError, ValueError):
                    pass
            joints[str(bone)] = values
    try:
        opacity = float(spec.get("opacity", 1.0))
    except (TypeError, ValueError):
        opacity = 1.0
    return {
        "kind": KIND_FIGURE,
        "pos": _vec3(spec.get("pos")),
        "size": size,
        "spin": spin,
        "x": x,
        "y": y,
        "scale": scale,
        "body": body,
        "view": view,
        "roll": roll,
        "joints": joints,
        "color": str(spec.get("color") or DEFAULT_COLOR),
        "opacity": max(0.0, min(1.0, opacity)),
    }


def clone(spec):
    """A deep copy safe to keep in a snapshot or undo entry."""
    normalized = normalize(spec)
    if normalized is None:
        return None
    normalized["pos"] = list(normalized["pos"])
    normalized["view"] = list(normalized["view"])
    if is_primitive(normalized):
        normalized["dim"] = list(normalized["dim"])
        normalized["rot"] = list(normalized["rot"])
        return normalized
    normalized["joints"] = {bone: list(a) for bone, a in normalized["joints"].items()}
    return normalized


def new(x, y, scale=DEFAULT_SCALE, body=DEFAULT_BODY, color=DEFAULT_COLOR, pose="relaxed"):
    """Build a normalised figure standing at document ``(x, y)``.

    Returns ``None`` when the model for ``body`` cannot be loaded, so callers can
    report a missing install instead of adding an invisible figure.
    """
    spec = normalize({
        "x": x, "y": y, "scale": scale, "body": body, "color": color,
        "view": [0.0, 0.0], "roll": 0.0, "joints": {}, "opacity": 1.0,
    })
    if spec is None or model_for(spec["body"]) is None:
        return None
    if pose:
        apply_pose(spec, pose)
    return spec


def new_primitive(x, y, scene=None, shape=DEFAULT_SHAPE,
                  size=DEFAULT_PRIMITIVE_SIZE, color=DEFAULT_PRIMITIVE_COLOR,
                  dim=(1.0, 1.0, 1.0), rot=(0.0, 0.0, 0.0), opacity=1.0):
    """Build a normalised primitive centred at document ``(x, y)``.

    ``scene`` is optional; when given the object's world ``pos`` is placed so it
    projects to the given document point at the scene origin.
    """
    spec = normalize({
        "kind": KIND_PRIMITIVE, "shape": shape, "size": size, "color": color,
        "dim": list(dim), "rot": list(rot), "pos": [0.0, 0.0, 0.0],
        "opacity": opacity,
    })
    if spec is None:
        return None
    if scene is not None:
        spec["x"] = float(x)
        spec["y"] = float(y)
    return spec


def describe(spec):
    if is_primitive(spec):
        dim = spec.get("dim") or (1.0, 1.0, 1.0)
        return (f"{PRIMITIVE_LABELS.get(spec.get('shape'), spec.get('shape'))} "
                f"({dim[0]:.2g}x{dim[1]:.2g}x{dim[2]:.2g})")
    posed = len(spec.get("joints") or {})
    return f"{spec.get('body', DEFAULT_BODY)} pose ({posed} bones, {int(spec.get('scale', DEFAULT_SCALE))}px)"


def rest(spec):
    """Clear every bone rotation, back to the bind (T) pose."""
    if is_primitive(spec):
        return
    spec["joints"] = {}


def reset(spec, pose="relaxed"):
    """Return the figure to a named preset (its default stance)."""
    if is_primitive(spec):
        spec["rot"] = [0.0, 0.0, 0.0]
        spec["dim"] = [1.0, 1.0, 1.0]
        return
    spec["joints"] = {}
    spec["spin"] = 0.0
    if pose:
        apply_pose(spec, pose)


# --------------------------------------------------------------------- scene
def new_scene(size=None):
    """A fresh scene camera for a layer, centred on a canvas of ``size``."""
    width, height = (size if size else (1000, 1000))
    return {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "zoom": DEFAULT_ZOOM,
        "x": float(width) / 2.0,
        "y": float(height) / 2.0,
    }


def normalize_scene(scene, size=None):
    """Coerce ``scene`` to a complete, valid camera (or a fresh default)."""
    scene = scene if isinstance(scene, dict) else {}
    width, height = (size if size else (0, 0))

    def number(key, default, lo=None, hi=None):
        try:
            value = float(scene.get(key, default))
        except (TypeError, ValueError):
            value = default
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return value

    return {
        "yaw": number("yaw", 0.0),
        "pitch": number("pitch", 0.0, -89.0, 89.0),
        "roll": number("roll", 0.0),
        "zoom": number("zoom", DEFAULT_ZOOM, MIN_ZOOM, MAX_ZOOM),
        "x": number("x", width / 2.0 if width else 0.0),
        "y": number("y", height / 2.0 if height else 0.0),
    }


def clone_scene(scene):
    return dict(normalize_scene(scene))


def set_scene_zoom(scene, zoom, pivot=None):
    """Dolly the scene to ``zoom`` pixels per model unit (clamped to MIN/MAX_ZOOM).

    When ``pivot`` (a document position) is given the scene zooms about it: the
    anchor is shifted so the world point under ``pivot`` stays put. Every figure
    and the ground grid scale by the same factor about the anchor, so this is the
    scene's own camera zoom, independent of the 2D canvas zoom.
    """
    zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
    old = float(scene.get("zoom", DEFAULT_ZOOM))
    if pivot is not None and old > 0.0:
        k = zoom / old
        scene["x"] = float(pivot[0]) - k * (float(pivot[0]) - float(scene["x"]))
        scene["y"] = float(pivot[1]) - k * (float(pivot[1]) - float(scene["y"]))
    scene["zoom"] = zoom


def sync_spec(spec, scene):
    """Refresh a spec's derived projection fields from its ``pos``/``size``/``spin``.

    The derived ``x``/``y``/``scale``/``view``/``roll`` are what every projection
    helper reads, so they must be current whenever the scene camera or the
    figure's placement changes. ``x``/``y`` become the document position of the
    figure's own world origin (the scene anchor plus the projection of ``pos``),
    which is exactly what the helpers already expect. A primitive's ``scale`` is
    its pixels-per-model-unit (its reference height is one), where a figure's is
    its pixel height.
    """
    if is_primitive(spec):
        pos = np.asarray(_vec3(spec.get("pos")), dtype=np.float64)
        pixels = float(scene["zoom"]) * float(spec.get("size", 1.0))
        x, y, _camera = mesh.project(
            pos[None, :], pixels, scene["yaw"], scene["pitch"], PERSPECTIVE,
        )
        dx, dy = _apply_roll(float(x[0]), float(y[0]), scene["roll"])
        spec["x"] = float(scene["x"]) + dx
        spec["y"] = float(scene["y"]) + dy
        spec["scale"] = pixels
        spec["view"] = [float(scene["yaw"]), float(scene["pitch"])]
        spec["roll"] = float(scene["roll"])
        return
    model = model_for(spec["body"])
    height = model.height if model is not None else 1.8
    pos = np.asarray(_vec3(spec.get("pos")), dtype=np.float64)
    size = float(spec.get("size", 1.0))
    pixels = float(scene["zoom"]) * size
    x, y, _camera = mesh.project(
        pos[None, :], pixels, scene["yaw"], scene["pitch"], PERSPECTIVE,
    )
    dx, dy = _apply_roll(float(x[0]), float(y[0]), scene["roll"])
    spec["x"] = float(scene["x"]) + dx
    spec["y"] = float(scene["y"]) + dy
    spec["scale"] = pixels * height
    spec["view"] = [float(scene["yaw"]), float(scene["pitch"])]
    spec["roll"] = float(scene["roll"]) + float(spec.get("spin", 0.0))


def sync_scene(specs, scene):
    """Bring every spec's derived fields up to date with ``scene``."""
    for spec in specs:
        sync_spec(spec, scene)


def load_layer_poses(entries, scene=None, size=None):
    """Build a layer's figures and scene from saved data.

    New projects store the shared ``scene`` alongside the figures; older ones
    have only per-figure ``view``/``roll``/``scale``/``x``/``y``. In the legacy
    case a scene is derived from the first figure and every figure is rehomed
    into it from its old document position, so the layout is preserved while the
    layer gains the shared camera. Returns ``(specs, scene)``.
    """
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    if isinstance(scene, dict) and scene:
        camera = normalize_scene(scene, size)
        specs = [s for s in (normalize(e) for e in entries) if s is not None]
        sync_scene(specs, camera)
        return specs, camera
    camera = _legacy_scene(entries, size)
    converted = []
    for entry in entries:
        if not isinstance(entry.get("pos"), (list, tuple)):
            entry = _legacy_pos(entry, camera)
        converted.append(entry)
    specs = [s for s in (normalize(e) for e in converted) if s is not None]
    sync_scene(specs, camera)
    return specs, camera


def _legacy_scene(entries, size):
    """A scene matching the first legacy figure's camera and anchor."""
    camera = new_scene(size)
    if not entries:
        return camera
    first = entries[0]
    if isinstance(first.get("view"), (list, tuple)) and len(first["view"]) >= 2:
        try:
            camera["yaw"] = float(first["view"][0])
            camera["pitch"] = max(-89.0, min(89.0, float(first["view"][1])))
        except (TypeError, ValueError):
            pass
    try:
        camera["roll"] = float(first.get("roll", 0.0) or 0.0)
    except (TypeError, ValueError):
        pass
    model = model_for(first.get("body", DEFAULT_BODY))
    height = model.height if model is not None else 1.8
    try:
        scale = float(first.get("scale", DEFAULT_SCALE))
    except (TypeError, ValueError):
        scale = DEFAULT_SCALE
    camera["zoom"] = max(MIN_ZOOM, min(MAX_ZOOM, scale / height))
    for key in ("x", "y"):
        try:
            camera[key] = float(first.get(key, camera[key]))
        except (TypeError, ValueError):
            pass
    return camera


def _legacy_pos(entry, camera):
    """A copy of a legacy ``entry`` with a world ``pos`` derived from its old x/y."""
    converted = dict(entry)
    converted.setdefault("size", 1.0)
    try:
        legacy_roll = float(entry.get("roll", 0.0) or 0.0)
    except (TypeError, ValueError):
        legacy_roll = 0.0
    converted.setdefault("spin", legacy_roll - float(camera["roll"]))
    try:
        doc_x = float(entry.get("x", camera["x"]))
        doc_y = float(entry.get("y", camera["y"]))
    except (TypeError, ValueError):
        doc_x, doc_y = camera["x"], camera["y"]
    pixels = camera["zoom"] * float(converted.get("size", 1.0) or 1.0)
    rx, ry = _apply_roll(doc_x - camera["x"], doc_y - camera["y"], -camera["roll"])
    camera_point = np.array([rx / pixels, -ry / pixels, 0.0], dtype=np.float64)
    world = mesh.view_matrix(camera["yaw"], camera["pitch"]).T @ camera_point
    converted["pos"] = [float(world[0]), float(world[1]), float(world[2])]
    return converted


def size_for_pixels(scene, spec, pixels):
    """The size multiplier that makes ``spec`` about ``pixels`` tall."""
    model = model_for(spec["body"])
    height = model.height if model is not None else 1.8
    return max(0.01, float(pixels) / (float(scene["zoom"]) * height))


def size_bounds(scene, body=DEFAULT_BODY):
    """The ``(min, max)`` size multipliers keeping a figure within MIN/MAX_SCALE."""
    model = model_for(body)
    height = model.height if model is not None else 1.8
    unit = float(scene["zoom"]) * height
    return (MIN_SCALE / unit, MAX_SCALE / unit)


def _doc_to_world(scene, spec, dx, dy):
    """A world-space offset that moves a figure by ``(dx, dy)`` document pixels.

    The figure's pixels-per-unit shrink with its scene depth, so the perspective
    factor at the figure's own origin is folded in; otherwise a far figure would
    lag behind the cursor.
    """
    pixels = float(scene["zoom"]) * float(spec.get("size", 1.0))
    view = mesh.view_matrix(scene["yaw"], scene["pitch"])
    pos_camera = view @ np.asarray(_vec3(spec.get("pos")), dtype=np.float64)
    factor = 1.0
    if PERSPECTIVE and PERSPECTIVE > 0.0:
        factor = PERSPECTIVE / max(PERSPECTIVE - pos_camera[2], 1e-3)
    rx, ry = _apply_roll(dx, dy, -scene["roll"])
    camera = np.array([rx / (pixels * factor), -ry / (pixels * factor), 0.0],
                      dtype=np.float64)
    return view.T @ camera


def move_by_doc(spec, scene, base_pos, dx, dy):
    """Place ``spec`` at ``base_pos`` shifted by ``(dx, dy)`` document pixels."""
    delta = _doc_to_world(scene, spec, dx, dy)
    base = _vec3(base_pos)
    spec["pos"] = [base[0] + float(delta[0]), base[1] + float(delta[1]),
                   base[2] + float(delta[2])]


def shift_doc(spec, scene, dx, dy):
    """Move ``spec`` by ``(dx, dy)`` document pixels from where it is now."""
    move_by_doc(spec, scene, spec.get("pos"), dx, dy)


def position(spec):
    """A copy of ``spec``'s world position as a 3-list."""
    return _vec3(spec.get("pos"))


# ------------------------------------------------------------------ posing
def aim_at(spec, bone, doc_x, doc_y, model=None):
    """Solve the delta that aims ``bone`` at a document point, and store it.

    The document point is turned back into a model-space direction (the camera
    is treated as orthographic for this, which is close enough for pointing) and
    handed to :meth:`mesh.Model.aim_delta`. Returns the delta or ``None``.
    """
    model = model or model_for(spec["body"])
    if model is None or bone not in model.joint_index:
        return None
    index = model.joint_index[bone]
    child = model.aim_child[index]
    if child < 0:
        return None
    joints = spec["joints"]
    world = model.world_matrices(joints)
    start = world[index][:3, 3]
    _start_xy, camera = _project(model, spec, start[None, :])
    target_camera = _camera_point(spec, model, doc_x, doc_y, float(camera[0, 2]))
    # The cursor point is in world space; bring it into the figure's own frame.
    local = mesh.view_matrix(*spec["view"]).T @ target_camera
    local = local - np.asarray(_vec3(spec.get("pos")), dtype=np.float64)
    wanted = local - start

    rule = HINGE_LIMITS.get(bone)
    if rule is not None:
        # A hinge turns within its plane only, so the aim becomes the angle about
        # the hinge axis that best aligns the bone with the cursor. That keeps the
        # limb bending the right way instead of tearing off-axis and being
        # clamped straight.
        child = model.aim_child[index]
        if child < 0:
            return None
        axis_local = _hinge_axis_local(model, bone, rule[0])
        hinge = _joint_frame(model, spec, bone)[1] @ axis_local
        current = world[child][:3, 3] - start
        wanted = wanted - float(np.dot(wanted, hinge)) * hinge
        current = current - float(np.dot(current, hinge)) * hinge
        if (float(np.linalg.norm(wanted)) < 1e-9
                or float(np.linalg.norm(current)) < 1e-9):
            return None
        angle = math.atan2(
            float(np.dot(np.cross(current, wanted), hinge)),
            float(np.dot(current, wanted)),
        )
        _apply_world_rotation(model, spec, bone, mesh.axis_angle(hinge, angle))
        _clamp_hinge(model, spec, bone)
        return joints.get(bone)

    delta = model.aim_delta(bone, wanted, joints)
    if delta is None:
        return None
    joints[bone] = delta
    return delta


def _camera_point(spec, model, doc_x, doc_y, depth):
    """The camera-space world point under a document position, at camera ``depth``.

    Inverts the same mapping :func:`_project` applies: the document offset is
    measured from the figure origin (``spec["x"]``/``spec["y"]``), and the figure
    origin's own projected offset (which carries the scene depth) is folded back
    in, so the returned point really lies at ``depth`` rather than being an
    orthographic approximation.
    """
    scale = float(spec["scale"]) / model.height
    view = mesh.view_matrix(*spec["view"])
    pos_camera = view @ np.asarray(_vec3(spec.get("pos")), dtype=np.float64)
    factor = 1.0
    factor_pos = 1.0
    if PERSPECTIVE and PERSPECTIVE > 0.0:
        factor = PERSPECTIVE / max(PERSPECTIVE - depth, 1e-3)
        factor_pos = PERSPECTIVE / max(PERSPECTIVE - pos_camera[2], 1e-3)
    ax = doc_x - float(spec["x"])
    ay = doc_y - float(spec["y"])
    roll = math.radians(spec.get("roll", 0.0))
    if roll:
        cos, sin = math.cos(-roll), math.sin(-roll)
        ax, ay = ax * cos - ay * sin, ax * sin + ay * cos
    kx = ax / scale + pos_camera[0] * factor_pos
    ky = -ay / scale + pos_camera[1] * factor_pos
    return np.array([kx / factor, ky / factor, depth], dtype=np.float64)


def _joint_frame(model, spec, joint):
    """``(parent_world_rotation, joint_frame_before_delta)``."""
    index = model.joint_index[joint]
    parent = model.joint_parent[index]
    if parent < 0:
        parent_rotation = np.eye(3)
    else:
        parent_rotation = model.world_matrices(_deltas(spec))[parent][:3, :3]
    return parent_rotation, parent_rotation @ model.rest_rotation[index]


def _apply_world_rotation(model, spec, joint, rotation):
    """Post-multiply a joint's rotation by a world-space rotation.

    A joint's stored delta is applied after its bind rotation, so a world-space
    rotation ``Q`` becomes ``A^-1 Q A delta``, where ``A`` is the joint's frame
    before the delta. That is what lets IK and the gizmo turn a bone about a world
    axis without knowing its private axes.
    """
    _parent, frame = _joint_frame(model, spec, joint)
    current = mesh.euler_matrix(*_deltas(spec).get(joint, (0.0, 0.0, 0.0)))
    spec["joints"][joint] = mesh.matrix_euler(frame.T @ rotation @ frame @ current)


def _hinge_axis_local(model, joint, curl):
    """The hinge axis in a joint's own frame, so a positive angle curls ``curl``.

    Derived from the rest pose: for a bone whose rest direction is ``d`` and which
    should curl toward ``c``, the axis ``d x c`` turns the bone toward ``c`` for a
    positive angle. That is stable, unlike probing for the direction a rotation
    happens to move a child in (which is degenerate when the axis is close to the
    probe direction).
    """
    key = (model.path, joint, tuple(np.asarray(curl, dtype=np.float64)))
    if key in _axis_cache:
        return _axis_cache[key]
    index = model.joint_index[joint]
    child = model.aim_child[index]
    rest = model.world_matrices({})
    direction = rest[child][:3, 3] - rest[index][:3, 3]
    length = float(np.linalg.norm(direction))
    axis_world = np.cross(direction / max(length, 1e-12), np.asarray(curl, dtype=np.float64))
    norm = float(np.linalg.norm(axis_world))
    if length < 1e-9 or norm < 1e-9:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis_world = axis_world / norm
        parent = model.joint_parent[index]
        parent_rotation = np.eye(3) if parent < 0 else rest[parent][:3, :3]
        axis = model.rest_rotation[index].T @ (parent_rotation.T @ axis_world)
    _axis_cache[key] = axis
    return axis


def _clamp_hinge(model, spec, joint):
    """Keep a hinge joint bending the allowed way, and no further than its limit.

    The stored delta is reduced to its component about the joint's hinge axis, so
    a limb cannot be made to bend backwards however it is aimed or solved.
    """
    rule = HINGE_LIMITS.get(joint)
    joints = spec["joints"]
    delta = joints.get(joint)
    if rule is None or delta is None:
        return
    axis = _hinge_axis_local(model, joint, rule[0])
    limit = math.radians(rule[1])
    delta_matrix = mesh.euler_matrix(*delta)
    axis_delta, angle = mesh.matrix_axis_angle(delta_matrix)
    if angle < 1e-6:
        return
    signed = angle * float(np.dot(axis_delta, axis))
    clamped = max(0.0, min(limit, signed))
    if abs(clamped - signed) < 1e-9:
        return
    joints[joint] = mesh.matrix_euler(mesh.axis_angle(axis, clamped))


def _hinge_component(model, spec, joint):
    """The signed bend of a hinge joint about its own hinge axis, in radians."""
    rule = HINGE_LIMITS.get(joint)
    delta = _deltas(spec).get(joint)
    if rule is None or delta is None:
        return 0.0
    axis, angle = mesh.matrix_axis_angle(mesh.euler_matrix(*delta))
    if angle < 1e-9:
        return 0.0
    return angle * float(np.dot(axis, _hinge_axis_local(model, joint, rule[0])))


def ik_at(spec, end_joint, doc_x, doc_y, model=None):
    """Solve a two-bone limb so ``end_joint`` reaches the cursor.

    Closed form rather than iterative: place the middle joint by the
    law-of-cosines angle off the cursor direction, aim the proximal bone at that
    point and the distal bone at the cursor. Bowing the middle joint to the side
    that satisfies the distal joint's hinge (rather than the wrong way, which the
    limit would then straighten) keeps the pose anatomically correct. A couple of
    passes refine the assumed depth, since the cursor only pins two dimensions.
    """
    model = model or model_for(spec["body"])
    chain = IK_CHAIN.get(end_joint)
    if model is None or chain is None:
        return False
    distal, proximal = chain
    rule = HINGE_LIMITS.get(distal)
    if rule is None:
        return False
    end_index = model.joint_index[end_joint]
    pos = np.asarray(_vec3(spec.get("pos")), dtype=np.float64)
    curl = np.array(rule[0], dtype=np.float64)
    saved = spec["joints"]
    snapshot = dict(saved)

    def solve(bow):
        world = model.world_matrices(_deltas(spec))
        end = world[end_index][:3, 3]
        _end_xy, camera = _project(model, spec, end[None, :])
        target = mesh.view_matrix(*spec["view"]).T @ _camera_point(
            spec, model, doc_x, doc_y, float(camera[0, 2]),
        )
        # The cursor point is in world space; bring it into the figure's own frame.
        target = target - pos
        origin = world[model.joint_index[proximal]][:3, 3]
        elbow = world[model.joint_index[distal]][:3, 3]
        upper = float(np.linalg.norm(elbow - origin))
        lower = float(np.linalg.norm(end - elbow))
        wanted = target - origin
        distance = float(np.linalg.norm(wanted))
        if upper < 1e-6 or lower < 1e-6 or distance < 1e-6:
            return False
        direction = wanted / distance
        distance = min(max(distance, abs(upper - lower) * 1.02 + 1e-4),
                       (upper + lower) * 0.999)
        cosine = (upper * upper + distance * distance - lower * lower) / (2.0 * upper * distance)
        angle = math.acos(max(-1.0, min(1.0, cosine)))
        perp = curl - float(np.dot(curl, direction)) * direction
        if float(np.linalg.norm(perp)) < 1e-6:
            axis_local = _hinge_axis_local(model, distal, curl)
            hinge = _joint_frame(model, spec, distal)[1] @ axis_local
            perp = hinge - float(np.dot(hinge, direction)) * direction
        norm = float(np.linalg.norm(perp))
        if norm < 1e-6:
            return False
        perp = perp / norm
        elbow_target = origin + upper * (
            math.cos(angle) * direction + bow * math.sin(angle) * perp
        )
        _apply_world_rotation(
            model, spec, proximal,
            mesh.rotation_between(elbow - origin, elbow_target - origin),
        )
        _clamp_hinge(model, spec, proximal)
        world = model.world_matrices(_deltas(spec))
        elbow = world[model.joint_index[distal]][:3, 3]
        end = world[end_index][:3, 3]
        if float(np.linalg.norm(target - elbow)) > 1e-6:
            _apply_world_rotation(
                model, spec, distal, mesh.rotation_between(end - elbow, target - elbow),
            )
            _clamp_hinge(model, spec, distal)
        return True

    for _ in range(3):
        chosen = None
        for bow in (-1.0, 1.0):
            spec["joints"] = dict(snapshot)
            if not solve(bow):
                continue
            component = _hinge_component(model, spec, distal)
            if chosen is None or component > chosen[0]:
                chosen = (component, dict(spec["joints"]))
            if component >= 0.0:
                break
        if chosen is None:
            spec["joints"] = saved
            return False
        snapshot = chosen[1]
        spec["joints"] = dict(snapshot)
    spec["joints"] = dict(snapshot)
    return True


def apply_axis_rotation(spec, bone, axis, radians, model=None):
    """Rotate a bone about a world axis (the gizmo rings use this)."""
    model = model or model_for(spec["body"])
    if model is None or bone not in model.joint_index:
        return False
    _apply_world_rotation(
        model, spec, bone, mesh.axis_angle(np.asarray(axis, dtype=np.float64), radians),
    )
    _clamp_hinge(model, spec, bone)
    return True


# ------------------------------------------------------------------- gizmo
GIZMO_RADIUS = 0.30
GIZMO_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _ring_basis(axis):
    """Two unit vectors spanning the plane perpendicular to ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    u = np.cross(axis, [0.0, 1.0, 0.0])
    if float(np.linalg.norm(u)) < 1e-6:
        u = np.cross(axis, [1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    return u, np.cross(axis, u)


def gizmo(spec, bone, segments=48):
    """Rotation-ring geometry for a bone, in document pixels, per world axis.

    A ring that is nearly edge-on to the camera is dropped: it projects to a line
    through the joint, which is neither readable nor reliably draggable (and would
    swallow clicks meant for the joint handle).
    """
    model = model_for(spec["body"])
    if model is None or bone not in model.joint_index:
        return None
    origin = model.joint_positions(_deltas(spec))[model.joint_index[bone]]
    angles = np.linspace(0.0, 2.0 * math.pi, int(segments), endpoint=False)
    rings = {}
    for name, axis in GIZMO_AXES.items():
        u, v = _ring_basis(axis)
        circle = origin[None, :] + GIZMO_RADIUS * (
            np.cos(angles)[:, None] * u[None, :] + np.sin(angles)[:, None] * v[None, :]
        )
        (px, py), _camera = _project(model, spec, circle)
        extent_x = float(px.max() - px.min())
        extent_y = float(py.max() - py.min())
        major, minor = max(extent_x, extent_y), min(extent_x, extent_y)
        if major < 1e-6 or minor < 0.12 * major:
            continue
        rings[name] = [
            (float(spec["x"]) + float(px[i]), float(spec["y"]) + float(py[i]))
            for i in range(len(angles))
        ]
    (ox, oy), _camera = _project(model, spec, origin[None, :])
    return {
        "bone": bone,
        "origin": (float(spec["x"]) + float(ox[0]), float(spec["y"]) + float(oy[0])),
        "rings": rings,
    }


def gizmo_hit(spec, geometry, x, y, tolerance):
    """``(axis, distance)`` for the closest bone-ring hit, else ``(None, inf)``.

    Hits near the joint itself are ignored, so the handle at the centre stays
    clickable even when a ring is nearly edge-on and passes close to it. Only the
    joint's own pickup radius is excluded: a flat ring projects to a thin ellipse
    whose whole arc can sit well inside a fraction of its mean radius, so a wider
    exclusion would make that ring impossible to grab.
    """
    if not geometry:
        return None, float("inf")
    origin = geometry["origin"]
    reach = math.hypot(x - origin[0], y - origin[1])
    best, best_distance = None, float("inf")
    for name, points in geometry["rings"].items():
        if reach < tolerance:
            continue
        for index in range(len(points)):
            ax, ay = points[index]
            bx, by = points[(index + 1) % len(points)]
            distance = _point_segment_distance(x, y, ax, ay, bx, by)
            if distance < best_distance:
                best_distance, best = distance, name
    if best is None or best_distance > tolerance:
        return None, float("inf")
    return best, best_distance


def gizmo_axis_at(spec, geometry, x, y, tolerance):
    """The ring axis under ``(x, y)``, or ``None``."""
    return gizmo_hit(spec, geometry, x, y, tolerance)[0]


def _view_ray(spec, model, doc_x, doc_y):
    """The camera-space ray a document position looks along.

    Built from two points on the line under that document position, so it stays
    exactly consistent with :func:`_camera_point` (and therefore with the
    figure's depth offset).
    """
    origin = _camera_point(spec, model, doc_x, doc_y, 0.0)
    ahead = _camera_point(spec, model, doc_x, doc_y, 1.0)
    return origin, ahead - origin


def gizmo_angle(spec, bone, axis_name, doc_x, doc_y):
    """The cursor's angle around a ring's world axis, in radians.

    The cursor ray is intersected with the ring's plane before the angle is
    measured, so the angle stays right when the ring is seen nearly edge-on. A
    ring whose plane is parallel to the view is degenerate (its screen projection
    is a line) and returns ``None``.
    """
    model = model_for(spec["body"])
    if model is None:
        return None
    view = mesh.view_matrix(*spec["view"])
    origin = model.joint_positions(_deltas(spec))[model.joint_index[bone]]
    center = view @ (origin + np.asarray(_vec3(spec.get("pos")), dtype=np.float64))
    axis = np.asarray(GIZMO_AXES[axis_name], dtype=np.float64)
    normal = view @ axis
    ray_origin, ray_direction = _view_ray(spec, model, doc_x, doc_y)
    denominator = float(np.dot(normal, ray_direction))
    if abs(denominator) < 1e-6:
        return None
    t = float(np.dot(normal, center - ray_origin)) / denominator
    radius = (ray_origin + t * ray_direction) - center
    u, v = _ring_basis(axis)
    return math.atan2(float(np.dot(radius, view @ v)), float(np.dot(radius, view @ u)))


def _point_segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    if length < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


# ---------------------------------------------------------- whole-figure gizmo
# The figure gizmo is drawn at a fixed screen size around the pelvis, so it stays
# usable whatever the figure's scale is: arrows translate, the ring rotates the
# figure in the picture plane, and the corner square scales it.
FIGURE_GIZMO_RADIUS = 92.0
FIGURE_HANDLES = ("scale", "move_x", "move_y", "rotate")
# The whole-figure gizmo is modal: the body drives one transform at a time and a
# click cycles this order, so move/scale/rotate never fight for the same drag.
FIGURE_MODES = ("move", "scale", "rotate")


def figure_gizmo(spec, radius=FIGURE_GIZMO_RADIUS):
    """Screen-space transform gizmo for the whole figure, in document pixels."""
    if is_primitive(spec):
        return None
    model = model_for(spec["body"])
    if model is None:
        return None
    center = joints_projected(spec).get("pelvis")
    if center is None:
        return None
    cx, cy = float(center[0]), float(center[1])
    angles = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    return {
        "origin": (cx, cy),
        "radius": radius,
        "move": (cx, cy),
        "move_x": ((cx, cy), (cx + radius, cy)),
        "move_y": ((cx, cy), (cx, cy - radius)),
        "ring": [
            (cx + radius * math.cos(t), cy + radius * math.sin(t)) for t in angles
        ],
        "scale": (cx + radius * 0.71, cy - radius * 0.71),
    }


def figure_hit(spec, geometry, x, y, tolerance):
    """``(handle, distance)`` for the closest whole-figure handle, else ``(None, inf)``."""
    if not geometry:
        return None, float("inf")
    if tolerance > 22.0:
        tolerance = 22.0
    cx, cy = geometry["origin"]
    reach = math.hypot(x - cx, y - cy)
    candidates = [("scale", math.hypot(x - geometry["scale"][0], y - geometry["scale"][1]))]
    if reach > geometry["radius"] * 0.25:
        for name in ("move_x", "move_y"):
            (ax, ay), (bx, by) = geometry[name]
            candidates.append((name, _point_segment_distance(x, y, ax, ay, bx, by)))
    if reach > geometry["radius"] * 0.4:
        ring = geometry["ring"]
        for index in range(len(ring)):
            ax, ay = ring[index]
            bx, by = ring[(index + 1) % len(ring)]
            candidates.append(("rotate", _point_segment_distance(x, y, ax, ay, bx, by)))
    best, best_distance = None, float("inf")
    for name, distance in candidates:
        limit = tolerance * 1.5 if name == "scale" else tolerance
        if distance <= limit and distance < best_distance:
            best_distance, best = distance, name
    return best, best_distance


def figure_handle_at(spec, geometry, x, y, tolerance):
    """Which whole-figure handle is under ``(x, y)``, or ``None``."""
    return figure_hit(spec, geometry, x, y, tolerance)[0]


def figure_angle(spec, doc_x, doc_y):
    """The cursor's angle about the pelvis, in radians."""
    center = joints_projected(spec).get("pelvis")
    if center is None:
        return None
    return math.atan2(doc_y - center[1], doc_x - center[0])


def keep_pelvis(spec, scene, pivot):
    """Shift the figure so the pelvis returns to ``pivot`` (document pixels).

    Rotating or scaling the whole figure would otherwise swing it around its
    origin (the feet); pinning the pelvis keeps the figure where the user put it.
    The shift is applied to the figure's world ``pos`` so it stays in the scene.
    """
    center = joints_projected(spec).get("pelvis")
    if center is None:
        return
    shift_doc(spec, scene, pivot[0] - center[0], pivot[1] - center[1])
    sync_spec(spec, scene)


# --------------------------------------------------------- whole-object gizmo
# A primitive's transform gizmo is drawn at a fixed screen size around the
# object's centre, in three modes so move, scale and rotate never fight for the
# same drag: axis arrows translate along a world axis, axis handles scale that
# axis, and world-axis rings rotate about it.
SCALE_PER_PIXEL = 0.004


def _axis_doc_vector(spec, model, axis):
    """The document-pixel offset produced by moving one world unit along ``axis``.

    Evaluated by projecting the object origin shifted by the axis, so perspective
    makes an axis pointing into the screen project shorter than one across it.
    """
    pos = np.asarray(_vec3(spec.get("pos")), dtype=np.float64)
    point = (pos + np.asarray(axis, dtype=np.float64))[None, :]
    (x, y), _camera = _project(model, spec, point)
    return float(x[0]), float(y[0])


def _object_ring(spec, model, center_world, axis, radius_world, segments=48):
    """Projected document points of a world-axis circle around ``center_world``.

    Returns ``None`` for a ring nearly edge-on to the camera (it would project to
    a line through the centre, unreadable and impossible to grab).
    """
    u, v = _ring_basis(axis)
    angles = np.linspace(0.0, 2.0 * math.pi, int(segments), endpoint=False)
    circle = np.asarray(center_world, dtype=np.float64)[None, :] + radius_world * (
        np.cos(angles)[:, None] * u[None, :] + np.sin(angles)[:, None] * v[None, :]
    )
    (px, py), _camera = _project(model, spec, circle)
    extent_x = float(px.max() - px.min())
    extent_y = float(py.max() - py.min())
    major, minor = max(extent_x, extent_y), min(extent_x, extent_y)
    if major < 1e-6 or minor < 0.14 * major:
        return None
    return [(float(spec["x"]) + float(px[i]), float(spec["y"]) + float(py[i]))
            for i in range(len(angles))]


def primitive_gizmo(spec, mode="move", radius=PRIMITIVE_GIZMO_RADIUS):
    """Screen-space transform gizmo geometry for a primitive, in document pixels."""
    model = object_model(spec)
    if model is None:
        return None
    cx, cy = float(spec["x"]), float(spec["y"])
    axes, handles, rings = {}, {}, {}
    for name, axis in GIZMO_AXES.items():
        dx, dy = _axis_doc_vector(spec, model, axis)
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        axes[name] = (dx, dy)
        handles[name] = (cx + dx / length * radius, cy + dy / length * radius)
    world_radius = radius / max(float(spec.get("scale", 1.0)), 1e-3)
    center_world = _vec3(spec.get("pos"))
    for name, axis in GIZMO_AXES.items():
        ring = _object_ring(spec, model, center_world, axis, world_radius)
        if ring is not None:
            rings[name] = ring
    return {
        "mode": mode,
        "origin": (cx, cy),
        "radius": radius,
        "axes": axes,
        "handles": handles,
        "rings": rings,
    }


def primitive_hit(spec, geometry, x, y, tolerance):
    """``(axis, distance)`` for the closest object-gizmo handle, else ``(None, inf)``."""
    if not geometry:
        return None, float("inf")
    mode = geometry.get("mode", "move")
    ox, oy = geometry["origin"]
    reach = math.hypot(x - ox, y - oy)
    best, best_distance = None, float("inf")
    if mode == "rotate":
        for name, points in (geometry.get("rings") or {}).items():
            if reach < geometry["radius"] * 0.2:
                continue
            for index in range(len(points)):
                ax, ay = points[index]
                bx, by = points[(index + 1) % len(points)]
                distance = _point_segment_distance(x, y, ax, ay, bx, by)
                if distance < best_distance:
                    best_distance, best = distance, name
    else:
        for name, (dx, dy) in (geometry.get("axes") or {}).items():
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            ux, uy = dx / length, dy / length
            ex, ey = ox + ux * geometry["radius"], oy + uy * geometry["radius"]
            distance = _point_segment_distance(x, y, ox, oy, ex, ey)
            tip = math.hypot(x - ex, y - ey)
            if tip <= tolerance * 1.6:
                distance = tip
            if distance < best_distance:
                best_distance, best = distance, name
    if best is None or best_distance > tolerance:
        return None, float("inf")
    return best, best_distance


def primitive_handle_at(spec, geometry, x, y, tolerance):
    """Which object-gizmo axis is under ``(x, y)``, or ``None``."""
    return primitive_hit(spec, geometry, x, y, tolerance)[0]


def primitive_ring_angle(spec, axis_name, doc_x, doc_y):
    """The cursor's angle about a world axis through the object centre, in radians."""
    model = object_model(spec)
    if model is None:
        return None
    view = mesh.view_matrix(*spec["view"])
    center = view @ np.asarray(_vec3(spec.get("pos")), dtype=np.float64)
    axis = np.asarray(GIZMO_AXES[axis_name], dtype=np.float64)
    normal = view @ axis
    ray_origin, ray_direction = _view_ray(spec, model, doc_x, doc_y)
    denominator = float(np.dot(normal, ray_direction))
    if abs(denominator) < 1e-6:
        return None
    t = float(np.dot(normal, center - ray_origin)) / denominator
    radius = (ray_origin + t * ray_direction) - center
    u, v = _ring_basis(axis)
    return math.atan2(float(np.dot(radius, view @ v)), float(np.dot(radius, view @ u)))


def move_primitive(spec, base_pos, axis_name, geometry, dx, dy):
    """Slide ``spec`` from ``base_pos`` along a world axis by a document drag."""
    ax, ay = geometry["axes"][axis_name]
    denominator = ax * ax + ay * ay
    if denominator < 1e-9:
        return
    distance = (dx * ax + dy * ay) / denominator
    axis = np.asarray(GIZMO_AXES[axis_name], dtype=np.float64)
    base = _vec3(base_pos)
    spec["pos"] = [base[i] + distance * float(axis[i]) for i in range(3)]


def scale_primitive(spec, base_dim, axis_name, factor):
    """Set one axis of ``spec``'s ``dim`` to ``base_dim`` scaled by ``factor``."""
    index = {"x": 0, "y": 1, "z": 2}[axis_name]
    if factor < 1e-3:
        factor = 1e-3
    dim = [max(MIN_DIM, min(MAX_DIM, float(v))) for v in base_dim]
    dim[index] = max(MIN_DIM, min(MAX_DIM, float(base_dim[index]) * factor))
    spec["dim"] = dim


def rotate_primitive(spec, axis_name, angle_delta):
    """Rotate ``spec`` about a world axis by ``angle_delta`` radians.

    The stored rotation is Euler, so an incremental world-axis turn is composed
    as ``axis_angle @ current`` and converted back.
    """
    axis = np.asarray(GIZMO_AXES[axis_name], dtype=np.float64)
    current = mesh.euler_matrix(*_vec3(spec.get("rot")))
    rotated = mesh.axis_angle(axis, float(angle_delta)) @ current
    spec["rot"] = mesh.matrix_euler(rotated)


# ------------------------------------------------------------------ presets
# Full-body presets are world-space aim directions per bone, applied top-down
# through the aim solver so they are geometrically correct rather than a table
# of guessed Euler angles.
POSES = {
    "relaxed": {
        "upperarm_l": (0.20, -0.97, 0.05), "lowerarm_l": (0.08, -0.99, 0.10),
        "upperarm_r": (-0.20, -0.97, 0.05), "lowerarm_r": (-0.08, -0.99, 0.10),
        "thigh_l": (0.06, -1.0, 0.0), "thigh_r": (-0.06, -1.0, 0.0),
        "calf_l": (0.0, -1.0, 0.03), "calf_r": (0.0, -1.0, 0.03),
    },
    "t_pose": {},
    "a_pose": {
        "upperarm_l": (0.70, -0.71, 0.0), "upperarm_r": (-0.70, -0.71, 0.0),
        "lowerarm_l": (0.70, -0.71, 0.0), "lowerarm_r": (-0.70, -0.71, 0.0),
    },
    "arms_down": {
        "upperarm_l": (0.0, -1.0, 0.0), "upperarm_r": (0.0, -1.0, 0.0),
        "lowerarm_l": (0.0, -1.0, 0.0), "lowerarm_r": (0.0, -1.0, 0.0),
    },
    "walk": {
        "spine_02": (0.0, 1.0, 0.0),
        "upperarm_l": (0.25, -0.95, 0.18), "lowerarm_l": (0.2, -0.9, 0.38),
        "upperarm_r": (-0.25, -0.95, -0.18), "lowerarm_r": (-0.2, -0.9, -0.34),
        "thigh_l": (0.05, -0.55, 0.83), "calf_l": (0.05, -0.95, -0.3),
        "thigh_r": (-0.05, -0.95, -0.3), "calf_r": (-0.05, -0.75, -0.66),
    },
    "run": {
        "spine_02": (0.0, 0.96, -0.28),
        "upperarm_l": (0.2, -0.45, 0.87), "lowerarm_l": (0.2, -0.2, 0.96),
        "upperarm_r": (-0.2, -0.75, -0.63), "lowerarm_r": (-0.2, -0.5, -0.84),
        "thigh_l": (0.05, -0.35, 0.93), "calf_l": (0.05, -0.9, -0.43),
        "thigh_r": (-0.05, -0.9, -0.43), "calf_r": (-0.05, -0.45, -0.89),
    },
    "sit": {
        "spine_02": (0.0, 0.99, 0.12),
        "thigh_l": (0.10, -0.10, 0.99), "calf_l": (0.10, -0.99, -0.05),
        "thigh_r": (-0.10, -0.10, 0.99), "calf_r": (-0.10, -0.99, -0.05),
        "upperarm_l": (0.30, -0.85, 0.44), "lowerarm_l": (0.25, -0.6, 0.76),
        "upperarm_r": (-0.30, -0.85, 0.44), "lowerarm_r": (-0.25, -0.6, 0.76),
    },
    "crouch": {
        "spine_02": (0.0, 0.94, 0.34),
        "thigh_l": (0.12, -0.55, 0.83), "calf_l": (0.12, -1.0, -0.25),
        "thigh_r": (-0.12, -0.55, 0.83), "calf_r": (-0.12, -1.0, -0.25),
        "upperarm_l": (0.45, -0.75, 0.48), "lowerarm_l": (0.35, -0.35, 0.87),
        "upperarm_r": (-0.45, -0.75, 0.48), "lowerarm_r": (-0.35, -0.35, 0.87),
    },
    "wave": {
        "upperarm_l": (0.20, -0.97, 0.05), "lowerarm_l": (0.08, -0.99, 0.10),
        "upperarm_r": (-0.55, 0.62, 0.56), "lowerarm_r": (-0.30, 0.94, 0.16),
        "thigh_l": (0.06, -1.0, 0.0), "thigh_r": (-0.06, -1.0, 0.0),
    },
    "action": {
        "spine_02": (0.05, 0.92, -0.38),
        "upperarm_l": (0.35, 0.30, 0.89), "lowerarm_l": (0.25, 0.65, 0.72),
        "upperarm_r": (-0.55, -0.55, 0.63), "lowerarm_r": (-0.75, -0.25, 0.61),
        "thigh_l": (0.10, -0.30, 0.95), "calf_l": (0.10, -0.95, -0.30),
        "thigh_r": (-0.10, -0.95, -0.30), "calf_r": (-0.10, -0.40, -0.91),
    },
    "handstand": {
        "pelvis": (0.0, -1.0, 0.05), "spine_01": (0.0, -1.0, 0.05),
        "spine_02": (0.0, -0.99, 0.16), "spine_03": (0.0, -0.98, 0.18),
        "neck_01": (0.0, -0.65, 0.76),
        "clavicle_l": (0.83, -0.56, -0.06),
        "upperarm_l": (0.0, -1.0, -0.08), "lowerarm_l": (0.0, -1.0, -0.08),
        "thigh_l": (0.04, -0.08, 1.0), "calf_l": (0.06, -0.12, 0.99),
        "foot_l": (0.08, -0.37, 0.93),
        "clavicle_r": (-0.83, -0.56, -0.06),
        "upperarm_r": (0.0, -1.0, -0.08), "lowerarm_r": (0.0, -1.0, -0.08),
        "thigh_r": (0.07, 0.41, -0.91), "calf_r": (0.0, 0.43, 0.9),
        "foot_r": (0.03, 0.3, 0.95),
    },
}

POSE_ORDER = ("relaxed", "t_pose", "a_pose", "arms_down", "walk", "run",
              "sit", "crouch", "wave", "action", "handstand")

# Hand presets are a curl angle per finger; the bend axis is derived from the
# rest geometry, so these numbers are degrees of curl, not per-bone axes.
HAND_POSES = {
    "open": {},
    "relaxed": {"thumb": 12.0, "index": 22.0, "middle": 26.0, "ring": 30.0, "pinky": 32.0},
    "fist": {"thumb": 45.0, "index": 88.0, "middle": 92.0, "ring": 92.0, "pinky": 90.0},
    "grip": {"thumb": 35.0, "index": 62.0, "middle": 66.0, "ring": 66.0, "pinky": 64.0},
    "point": {"thumb": 20.0, "index": 0.0, "middle": 78.0, "ring": 82.0, "pinky": 84.0},
    "peace": {"thumb": 25.0, "index": 0.0, "middle": 0.0, "ring": 80.0, "pinky": 82.0},
    "pinch": {"thumb": 40.0, "index": 48.0, "middle": 24.0, "ring": 30.0, "pinky": 34.0},
    "thumbs_up": {"thumb": 0.0, "index": 84.0, "middle": 88.0, "ring": 88.0, "pinky": 86.0},
}

HAND_ORDER = ("open", "relaxed", "fist", "grip", "point", "peace", "pinch",
              "thumbs_up")
HAND_SIDES = {"both": ("l", "r"), "left": ("l",), "right": ("r",)}

_axis_cache = {}


def apply_pose(spec, name):
    """Reset to the bind pose and apply a named full-body preset."""
    if is_primitive(spec):
        return False
    targets = POSES.get(name)
    model = model_for(spec["body"])
    if targets is None or model is None:
        return False
    spec["joints"] = {}
    for bone in _depth_order(model, targets.keys()):
        direction = np.array(targets[bone], dtype=np.float64)
        if float(np.linalg.norm(direction)) < 1e-9:
            continue
        delta = model.aim_delta(bone, direction, spec["joints"])
        if delta is not None:
            spec["joints"][bone] = delta
    return True


def apply_hand(spec, name, side="both"):
    """Curl the fingers of one or both hands into a named preset."""
    if is_primitive(spec):
        return False
    curls = HAND_POSES.get(name)
    sides = HAND_SIDES.get(side)
    model = model_for(spec["body"])
    if curls is None or sides is None or model is None:
        return False
    rest_world = model.world_matrices({})
    for hand in sides:
        # Presets are absolute per hand, so clear that hand's fingers first and
        # an "open" preset really opens it.
        for finger in DIGITS:
            for suffix in ("01", "02", "03", "04_leaf"):
                spec["joints"].pop(f"{finger}_{suffix}_{hand}", None)
        for finger in DIGITS:
            angle = float(curls.get(finger, 0.0))
            if abs(angle) < 1e-6:
                continue
            axis = _finger_axis(model, hand, finger, rest_world)
            if axis is None:
                continue
            for step, suffix in enumerate(("01", "02", "03")):
                bone = f"{finger}_{suffix}_{hand}"
                index = model.joint_index.get(bone)
                if index is None:
                    continue
                local_axis = rest_world[index][:3, :3].T @ axis
                delta = mesh.axis_angle(
                    local_axis, math.radians(angle * _PHALANX_WEIGHTS[step]),
                )
                spec["joints"][bone] = mesh.matrix_euler(delta)
    return True


def _finger_axis(model, hand, finger, rest_world):
    """The world axis a finger curls about, oriented so a positive angle curls in."""
    key = (model.name, hand, finger)
    if key in _axis_cache:
        return _axis_cache[key]
    positions = rest_world[:, :3, 3]
    try:
        base = positions[model.joint_index[f"{finger}_01_{hand}"]]
        knuckle = positions[model.joint_index[f"{finger}_02_{hand}"]]
        tip = positions[model.joint_index[f"{finger}_04_leaf_{hand}"]]
        d_index = (positions[model.joint_index[f"index_02_{hand}"]]
                   - positions[model.joint_index[f"index_01_{hand}"]])
        d_pinky = (positions[model.joint_index[f"pinky_02_{hand}"]]
                   - positions[model.joint_index[f"pinky_01_{hand}"]])
    except KeyError:
        return None
    direction = knuckle - base
    normal = np.cross(d_index, d_pinky)
    norm = np.linalg.norm(normal)
    if norm < 1e-9 or np.linalg.norm(direction) < 1e-9:
        return None
    normal = normal / norm
    axis = np.cross(direction, normal)
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return None
    axis = axis / length
    motion = np.cross(axis, tip - base)
    if float(np.dot(motion, normal)) > 0.0:
        axis = -axis
    _axis_cache[key] = axis
    return axis


def mirror(spec):
    """Mirror the pose left-to-right, and its placement in the scene with it."""
    if is_primitive(spec):
        pos = _vec3(spec.get("pos"))
        spec["pos"] = [-pos[0], pos[1], pos[2]]
        return True
    model = model_for(spec["body"])
    if model is None:
        return False
    joints = spec["joints"]
    world = model.world_matrices(joints)
    aims = {}
    for bone in joints:
        index = model.joint_index.get(bone)
        if index is None:
            continue
        child = model.aim_child[index]
        if child < 0:
            continue
        direction = world[child][:3, 3] - world[index][:3, 3]
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        aims[_swap_side(bone)] = (-direction[0] / length, direction[1] / length,
                                 direction[2] / length)
    spec["joints"] = {}
    for bone in _depth_order(model, aims.keys()):
        delta = model.aim_delta(bone, np.array(aims[bone]), spec["joints"])
        if delta is not None:
            spec["joints"][bone] = delta
    pos = _vec3(spec.get("pos"))
    spec["pos"] = [-pos[0], pos[1], pos[2]]
    spec["spin"] = -float(spec.get("spin", 0.0))
    return True


# ----------------------------------------------------------------- geometry
def _deltas(spec):
    return spec.get("joints") or {}


def _project(model, spec, points):
    """Project figure-local points to rolled offsets relative to the figure origin.

    The figure's world ``pos`` is added before the camera and the figure origin's
    own projection is subtracted, so the offsets stay relative to the figure
    origin (which the caller adds as ``spec["x"]``/``spec["y"]``) while the
    perspective divide is evaluated at the figure's real depth. That is what makes
    a figure placed farther from the camera draw smaller.
    """
    scale = float(spec["scale"]) / model.height
    view = spec["view"]
    pos = np.asarray(_vec3(spec.get("pos")), dtype=np.float64)
    x, y, camera = mesh.project(
        np.asarray(points, dtype=np.float64) + pos[None, :],
        scale, view[0], view[1], PERSPECTIVE,
    )
    origin_x, origin_y, _origin_camera = mesh.project(
        pos[None, :], scale, view[0], view[1], PERSPECTIVE,
    )
    x = np.asarray(x) - float(origin_x[0])
    y = np.asarray(y) - float(origin_y[0])
    return _apply_roll(x, y, spec.get("roll", 0.0)), camera


def _apply_roll(x, y, roll):
    if not roll:
        return x, y
    angle = math.radians(roll)
    cos, sin = math.cos(angle), math.sin(angle)
    return x * cos - y * sin, x * sin + y * cos


def joints_projected(spec):
    """``{bone: (doc_x, doc_y)}`` for every bone (no rasterisation)."""
    if is_primitive(spec):
        return {}
    model = model_for(spec["body"])
    if model is None:
        return {}
    positions = model.joint_positions(_deltas(spec))
    (x, y), _camera = _project(model, spec, positions)
    return {
        bone: (float(spec["x"]) + float(x[i]), float(spec["y"]) + float(y[i]))
        for i, bone in enumerate(model.joint_names)
    }


def bones_projected(spec):
    """Document-space ``(x1, y1, x2, y2)`` segments for the draggable skeleton."""
    if is_primitive(spec):
        return []
    model = model_for(spec["body"])
    if model is None:
        return []
    joints = joints_projected(spec)
    segments = []
    for bone in pose_bones(model):
        parent = model.joint_parent[model.joint_index[bone]]
        if parent < 0:
            continue
        ax, ay = joints[model.joint_names[parent]]
        bx, by = joints[bone]
        segments.append((ax, ay, bx, by))
    return segments


def bounds(spec):
    """The object's document-space bounding box ``(l, t, r, b)``.

    A figure is framed by its projected joints; a primitive by the eight corners
    of its (already dim/rot-baked) geometry bounds, so a long box or a rotated
    shape still yields a tight box.
    """
    model = object_model(spec)
    if model is None:
        return (spec["x"], spec["y"], spec["x"], spec["y"])
    if is_primitive(spec):
        low, high = model.bounds_min, model.bounds_max
        corners = np.array([
            [low[0] if i & 1 else high[0],
             low[1] if i & 2 else high[1],
             low[2] if i & 4 else high[2]]
            for i in range(8)
        ], dtype=np.float64)
        (x, y), _camera = _project(model, spec, corners)
        left = float(spec["x"]) + float(x.min())
        right = float(spec["x"]) + float(x.max())
        top = float(spec["y"]) + float(y.min())
        bottom = float(spec["y"]) + float(y.max())
        return (left, top, right, bottom)
    positions = model.joint_positions(_deltas(spec))
    (x, y), _camera = _project(model, spec, positions)
    margin = 0.10 * float(spec["scale"])
    left = float(spec["x"]) + float(x.min()) - margin
    right = float(spec["x"]) + float(x.max()) + margin
    top = float(spec["y"]) + float(y.min()) - margin
    bottom = float(spec["y"]) + float(y.max()) + margin
    return (left, top, right, bottom)


def joint_at(spec, x, y, tolerance):
    """The nearest draggable bone handle within ``tolerance``, or ``None``."""
    if is_primitive(spec):
        return None
    model = model_for(spec["body"])
    if model is None:
        return None
    best = None
    best_distance = tolerance
    joints = joints_projected(spec)
    for bone in pose_bones(model):
        jx, jy = joints[bone]
        distance = math.hypot(x - jx, y - jy)
        if distance <= best_distance:
            best_distance = distance
            best = bone
    return best


def hit_body(spec, x, y):
    left, top, right, bottom = bounds(spec)
    return left <= x <= right and top <= y <= bottom


def overlay(spec):
    """Handle geometry (document pixels) for the tool's overlay layer.

    Only the draggable bones are listed as joints, so every handle drawn is one
    the tool can actually grab (the fingers, toes and leaf bones would otherwise
    add unclickable dots around the hands and feet). A primitive has no skeleton;
    its overlay is just the projected outline box and centre the tool's gizmo and
    selection chrome hang off.
    """
    if is_primitive(spec):
        left, top, right, bottom = bounds(spec)
        return {
            "kind": KIND_PRIMITIVE,
            "outline": (left, top, right, bottom),
            "center": (float(spec["x"]), float(spec["y"])),
            "joints": {},
            "bones": [],
            "root": None,
            "ik_joints": [],
            "bones_count": 0,
        }
    model = model_for(spec["body"])
    joints = joints_projected(spec)
    root = joints.get("pelvis") or joints.get("root")
    if model is None:
        handles = {}
        bones_count = 0
    else:
        draggable = pose_bones(model)
        handles = {name: joints[name] for name in draggable if name in joints}
        bones_count = len(draggable)
    return {
        "joints": handles,
        "bones": bones_projected(spec),
        "root": root,
        "ik_joints": sorted(IK_CHAIN),
        "bones_count": bones_count,
    }


# ---------------------------------------------------------------- ground grid
# One perspective ground plane drawn under the whole scene, so figures read
# against a single floor. It follows the scene camera (yaw/pitch/roll/zoom),
# so it turns and tilts with everything else.
GRID_EXTENT = 8.0  # model units from the world origin
GRID_STEP = 0.5


def _clip_near(view, limit, a, b):
    """Cut a model-space segment at the camera's near plane (``camera z = limit``).

    Both points past the limit are dropped; one point past the limit is pulled
    back to it so the projected line cannot shoot off to infinity.
    """
    if limit is None:
        return a, b
    za = float((view @ a)[2])
    zb = float((view @ b)[2])
    if za >= limit and zb >= limit:
        return None, None
    if za >= limit:
        a = a + (limit - za) / (zb - za) * (b - a)
    elif zb >= limit:
        b = b + (limit - zb) / (za - zb) * (a - b)
    return a, b


def _project_scene(scene, points):
    """Project world-space points with the scene camera to document offsets."""
    x, y, camera = mesh.project(
        np.asarray(points, dtype=np.float64), float(scene["zoom"]),
        scene["yaw"], scene["pitch"], PERSPECTIVE,
    )
    return _apply_roll(np.asarray(x), np.asarray(y), scene["roll"]), camera


def scene_grid(scene, step=GRID_STEP, extent=GRID_EXTENT):
    """The single scene ground grid as ``(x0, y0, x1, y1)`` document segments.

    The lines lie on the world ground plane (the feet level) and are projected
    with the scene camera, so the whole floor turns and tilts with the scene.
    Segments that cross the camera's near plane are cut there; the caller clips
    the rest to the viewport.
    """
    view = mesh.view_matrix(scene["yaw"], scene["pitch"])
    limit = PERSPECTIVE * 0.9 if PERSPECTIVE and PERSPECTIVE > 0.0 else None
    count = max(1, int(round(extent / step)))
    lines = []
    for i in range(-count, count + 1):
        v = i * step
        lines.append((np.array([v, GROUND, -extent]), np.array([v, GROUND, extent])))
        lines.append((np.array([-extent, GROUND, v]), np.array([extent, GROUND, v])))
    ox, oy = float(scene["x"]), float(scene["y"])
    segments = []
    for a, b in lines:
        a, b = _clip_near(view, limit, a, b)
        if a is None:
            continue
        (ax, ay), _camera = _project_scene(scene, a[None, :])
        (bx, by), _camera = _project_scene(scene, b[None, :])
        segments.append((ox + float(ax[0]), oy + float(ay[0]),
                         ox + float(bx[0]), oy + float(by[0])))
    return segments


# ---------------------------------------------------------------- rendering
def render_sprite(spec, quality=1.0, proxy=False):
    """Render ``spec``; ``(sprite, left, top)`` with the offsets relative to (x, y)."""
    model = object_model(spec)
    if model is None:
        return None
    if is_primitive(spec):
        return mesh.render_sprite(
            model, {}, height_px=spec["scale"], color=spec["color"],
            yaw=spec["view"][0], pitch=spec["view"][1], roll=spec.get("roll", 0.0),
            perspective=PERSPECTIVE, proxy=proxy, opacity=spec["opacity"],
            quality=quality, supersample=1 if (proxy or quality < 1.0) else None,
            offset=_vec3(spec.get("pos")),
        )
    return mesh.render_sprite(
        model, _deltas(spec), height_px=spec["scale"], color=spec["color"],
        yaw=spec["view"][0], pitch=spec["view"][1], roll=spec.get("roll", 0.0),
        perspective=PERSPECTIVE, proxy=proxy, opacity=spec["opacity"],
        quality=quality, supersample=1 if (proxy or quality < 1.0) else None,
        offset=_vec3(spec.get("pos")),
    )


def paste(image, spec, offset=(0, 0)):
    """Composite ``spec`` onto ``image`` (minus the layer ``offset``).

    During an interactive drag the decimated proxy is used only on the CPU
    backend; with a GPU the drag already draws the full mesh at full quality, as
    does the committed render and the rasterise.
    """
    fast_drag = (_preview and _drag_quality == "fast"
                 and not mesh.gpu_available())
    result = render_sprite(spec, proxy=fast_drag)
    if result is None:
        return image
    sprite, left, top = result
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    ox, oy = offset
    dest_x = int(round(float(spec["x"]) + left - ox))
    dest_y = int(round(float(spec["y"]) + top - oy))
    ix0 = max(0, dest_x)
    iy0 = max(0, dest_y)
    ix1 = min(image.width, dest_x + sprite.width)
    iy1 = min(image.height, dest_y + sprite.height)
    if ix1 <= ix0 or iy1 <= iy0:
        return image
    crop = sprite.crop((ix0 - dest_x, iy0 - dest_y, ix1 - dest_x, iy1 - dest_y))
    image.alpha_composite(crop, dest=(ix0, iy0))
    return image
