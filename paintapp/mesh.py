"""Skinned glTF model loading and software rendering.

The Poses tool draws real 3D characters, so the app needs a small mesh pipeline:
read a rigged ``.gltf``/``.glb``, drive its skeleton, skin the mesh on the CPU and
rasterise it with a depth buffer. Everything here is Pillow + numpy only, to keep
the core install dependency-free, and it reads just the subset of glTF a skinned
humanoid needs (nodes, one skin with inverse bind matrices, triangle primitives
with POSITION/NORMAL/JOINTS_0/WEIGHTS_0 and an index buffer). Textures,
materials, UVs, vertex colours and animations are ignored: the figure is shaded
flat from its normals, which suits a pose reference.

Two decisions make it fast enough for a Tk paint app:

* Linear blend skinning is a few 4x4 matrix multiplies, vectorised over every
  vertex, instead of a per-vertex Python loop.
* The full mesh is ~14k triangles (~0.25s to rasterise, because numpy triangle
  filling is dominated by per-triangle overhead), so :meth:`Model.build_proxy`
  decimates the mesh once (edge collapse) and the tool uses that smaller index
  buffer for interactive drags, falling back to the full mesh for the cached
  commit.

Rasterisation is a z-buffer: triangles are rotated into camera space, culled when
back-facing, projected, then filled over their bounding box with barycentric
coverage and a depth test.

This module is the self-contained CPU fallback. When an OpenGL context is
available the figures are drawn by :mod:`paintapp.mesh_gpu` instead, which is
roughly a hundred times faster; :func:`gpu_available` reports which path is live.
"""

import json
import math
import os
import struct

import numpy as np
from PIL import Image

from . import antialias

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "models")

_COMPONENT = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
    5125: np.uint32, 5126: np.float32,
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9,
          "MAT4": 16}
_NORMALIZE = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}

# Flat shading (camera space; light from the viewer's upper left, slightly front).
AMBIENT = 0.36
DIFFUSE = 0.74
LIGHT = np.array([-0.40, 0.52, 0.76], dtype=np.float64)
LIGHT /= np.linalg.norm(LIGHT)

PROXY_TRIANGLES = 5000
MAX_SPRITE_PIXELS = 4_000_000

_CACHE = {}
_GPU = None
_GPU_TRIED = False


def _gpu():
    """The GPU backend module when an offscreen GL context is usable, else None."""
    global _GPU, _GPU_TRIED
    if not _GPU_TRIED:
        _GPU_TRIED = True
        try:
            from . import mesh_gpu
            _GPU = mesh_gpu if mesh_gpu.available() else None
        except Exception:
            _GPU = None
    return _GPU


def gpu_available():
    """True when figures are being drawn on the GPU rather than by the CPU."""
    return _gpu() is not None


def _render_gpu(backend, model, deltas, *, height_px, color, yaw, pitch, roll,
                perspective, opacity, offset):
    """Draw with the GPU backend. Any failure returns ``None`` so the caller can
    fall back to the CPU renderer instead of failing the whole composite.

    Multisampling does the anti-aliasing, so no supersampled downscale is needed
    and a full-quality frame stays in the low milliseconds.
    """
    try:
        return backend.render(
            model, deltas, height_px=height_px, color=color, yaw=yaw, pitch=pitch,
            roll=roll, perspective=perspective, opacity=opacity, supersample=1,
            samples=4, offset=offset,
        )
    except Exception:
        return None


# --------------------------------------------------------------- glTF reading
def _read_accessor(gltf, buffer, index):
    """Read one accessor into an ``(count, ncomp)`` array, honouring byteStride."""
    accessor = gltf["accessors"][index]
    count = accessor["count"]
    ncomp = _NCOMP[accessor["type"]]
    dtype = np.dtype(_COMPONENT[accessor["componentType"]]).newbyteorder("<")
    itemsize = dtype.itemsize
    view = gltf["bufferViews"][accessor["bufferView"]]
    stride = view.get("byteStride") or itemsize * ncomp
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    if stride == itemsize * ncomp:
        out = np.frombuffer(
            buffer, dtype=dtype, count=count * ncomp, offset=start,
        ).reshape(count, ncomp).astype(dtype, copy=True)
    else:
        window = np.frombuffer(buffer, dtype=np.uint8)[start:]
        raw = np.lib.stride_tricks.as_strided(
            window, shape=(count, ncomp * itemsize), strides=(stride, 1),
            writeable=False,
        )
        out = raw.copy().view(dtype).reshape(count, ncomp)
    if accessor.get("normalized") and dtype.kind != "f":
        out = out.astype(np.float32) / _NORMALIZE[accessor["componentType"]]
    return out


def _load_gltf_bytes(path):
    """Return ``(gltf dict, binary buffer)`` for a ``.gltf`` (external .bin) or ``.glb``."""
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw[:4] == b"glTF":
        gltf = None
        binary = b""
        offset = 12
        while offset < len(raw):
            chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
            offset += 8
            chunk = raw[offset:offset + chunk_len]
            offset += chunk_len
            if chunk_type == 0x4E4F534A:
                gltf = json.loads(chunk.decode("utf-8"))
            elif chunk_type == 0x004E4942:
                binary = chunk
        if gltf is None:
            raise ValueError("GLB has no JSON chunk")
        return gltf, binary
    gltf = json.loads(raw.decode("utf-8"))
    binary = b""
    buffers = gltf.get("buffers") or []
    if buffers:
        uri = buffers[0].get("uri")
        if uri and not uri.startswith("data:"):
            with open(os.path.join(os.path.dirname(path), uri), "rb") as handle:
                binary = handle.read()
    return gltf, binary


# ------------------------------------------------------------------- math
def _quat_matrix(x, y, z, w):
    """A 3x3 rotation matrix from a glTF ``[x, y, z, w]`` quaternion."""
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def _split_transform(matrix):
    """Split a 4x4 into ``(translation, rotation3x3, scale)``."""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    translation = matrix[:3, 3].copy()
    basis = matrix[:3, :3].copy()
    scale = np.linalg.norm(basis, axis=0)
    scale[scale == 0.0] = 1.0
    return translation, basis / scale, scale


def euler_matrix(rx, ry, rz):
    """A rotation from Euler degrees, applied X then Y then Z."""
    cx, sx = math.cos(math.radians(rx)), math.sin(math.radians(rx))
    cy, sy = math.cos(math.radians(ry)), math.sin(math.radians(ry))
    cz, sz = math.cos(math.radians(rz)), math.sin(math.radians(rz))
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rzm @ rym @ rxm


def matrix_euler(matrix):
    """Euler degrees (X then Y then Z) for a rotation matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    sy = max(-1.0, min(1.0, -m[2, 0]))
    ry = math.asin(sy)
    if abs(sy) < 0.999999:
        rx = math.atan2(m[2, 1], m[2, 2])
        rz = math.atan2(m[1, 0], m[0, 0])
    else:
        rx = math.atan2(-m[1, 2], m[1, 1])
        rz = 0.0
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def matrix_axis_angle(matrix):
    """An ``(axis, angle)`` pair for a rotation matrix (angle in radians)."""
    m = np.asarray(matrix, dtype=np.float64)
    cosine = max(-1.0, min(1.0, (m[0, 0] + m[1, 1] + m[2, 2] - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(angle - math.pi) < 1e-6:
        diagonal = np.array([m[0, 0], m[1, 1], m[2, 2]])
        index = int(np.argmax(diagonal))
        column = m[:, index]
        column[index] += 1.0
        length = np.linalg.norm(column)
        if length < 1e-9:
            return np.array([0.0, 0.0, 1.0]), angle
        return column / length, angle
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return axis / np.linalg.norm(axis), angle


def axis_angle(axis, angle):
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ], dtype=np.float64)


def rotation_between(source, target):
    """The shortest rotation matrix taking ``source`` onto ``target``.

    The arguments are normalised first, so callers may pass raw offsets.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source / max(float(np.linalg.norm(source)), 1e-12)
    target = target / max(float(np.linalg.norm(target)), 1e-12)
    dot = float(np.dot(source, target))
    if dot > 0.999999:
        return np.eye(3)
    if dot < -0.999999:
        axis = np.cross(source, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, [0.0, 1.0, 0.0])
        return axis_angle(axis / np.linalg.norm(axis), math.pi)
    axis = np.cross(source, target)
    return axis_angle(axis / np.linalg.norm(axis), math.acos(max(-1.0, min(1.0, dot))))


def _homogeneous(translation, rotation, scale):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.asarray(scale, dtype=np.float64)
    matrix[:3, 3] = translation
    return matrix


# ------------------------------------------------------------------------ model
class Model:
    """A loaded riggable model: skeleton, bind pose, skinnable mesh and proxy."""

    def __init__(self, name, gltf, binary, path):
        self.name = name
        self.path = path
        nodes = gltf["nodes"]
        parent = [-1] * len(nodes)
        for index, node in enumerate(nodes):
            for child in node.get("children", []):
                parent[child] = index

        skin = (gltf.get("skins") or [None])[0]
        if skin is None:
            raise ValueError(f"model {name!r} has no skin")
        self.joint_nodes = list(skin["joints"])
        self.joint_names = [nodes[index].get("name", "") for index in self.joint_nodes]
        self.joint_index = {joint: i for i, joint in enumerate(self.joint_names)}
        self.joint_parent = []
        for node in self.joint_nodes:
            walk = parent[node]
            while walk != -1 and walk not in self.joint_nodes:
                walk = parent[walk]
            self.joint_parent.append(self.joint_nodes.index(walk) if walk != -1 else -1)

        ibm = skin.get("inverseBindMatrices")
        if ibm is None:
            self.inverse_bind = np.tile(np.eye(4), (len(self.joint_nodes), 1, 1))
        else:
            self.inverse_bind = _read_accessor(
                gltf, binary, ibm,
            ).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)

        # Rest (bind) transforms. glTF matrices are column-major (hence the
        # transposes) and the joint chain can have non-joint nodes between joints
        # (an Armature node, a scale node), so the authoritative source for the
        # bind pose is the inverse bind matrices: the bind world matrix is their
        # inverse and a joint's local transform is the difference from its
        # parent's. Only a skin without inverse bind matrices falls back to the
        # node hierarchy, with the intermediate transforms folded in.
        rest_world = self._rest_world(nodes, parent, ibm is not None)
        rest_t, rest_r, rest_s = [], [], []
        for index, joint_parent in enumerate(self.joint_parent):
            if joint_parent < 0:
                local = rest_world[index]
            else:
                local = np.linalg.inv(rest_world[joint_parent]) @ rest_world[index]
            translation, rotation, scale = _split_transform(local)
            rest_t.append(translation)
            rest_r.append(rotation)
            rest_s.append(scale)
        self.rest_translation = np.array(rest_t)
        self.rest_rotation = np.array(rest_r)
        self.rest_scale = np.array(rest_s)

        positions, normals, joint_ids, weights, triangles = [], [], [], [], []
        vertex_base = 0
        for mesh in gltf.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                attributes = primitive.get("attributes", {})
                if "POSITION" not in attributes or primitive.get("indices") is None:
                    continue
                pos = _read_accessor(gltf, binary, attributes["POSITION"]).astype(np.float32)
                nrm = _read_accessor(gltf, binary, attributes["NORMAL"]).astype(np.float32)
                idx = _read_accessor(gltf, binary, primitive["indices"]).astype(np.int64).reshape(-1)
                if "JOINTS_0" in attributes:
                    jnt = _read_accessor(gltf, binary, attributes["JOINTS_0"]).astype(np.int64)
                else:
                    jnt = np.zeros((len(pos), 4), dtype=np.int64)
                if "WEIGHTS_0" in attributes:
                    wgt = _read_accessor(gltf, binary, attributes["WEIGHTS_0"]).astype(np.float32)
                else:
                    wgt = np.zeros((len(pos), 4), dtype=np.float32)
                    wgt[:, 0] = 1.0
                positions.append(pos)
                normals.append(nrm)
                joint_ids.append(jnt)
                weights.append(wgt)
                triangles.append(idx.reshape(-1, 3) + vertex_base)
                vertex_base += len(pos)

        self.position = np.concatenate(positions).astype(np.float32)
        self.normal = np.concatenate(normals).astype(np.float32)
        self.joint_ids = np.concatenate(joint_ids).astype(np.int32)
        self.weights = np.concatenate(weights).astype(np.float32)
        self.triangles = np.concatenate(triangles).astype(np.int32)

        low, high = self.position.min(0), self.position.max(0)
        self.bounds_min = low.astype(np.float64)
        self.bounds_max = high.astype(np.float64)
        self.height = float(high[1] - low[1])

        # Largest rest distance from a vertex to its dominant joint. A sprite
        # bounding box can then be derived from the joints alone, which is a
        # forward-kinematics pass, instead of skinning every vertex just to
        # frame the render.
        dominant = self.world_matrices({})[:, :3, 3][self.joint_ids[:, 0]]
        self.skin_margin = float(
            np.linalg.norm(self.position - dominant.astype(np.float32), axis=1).max()
        )

        self.aim_child = self._aim_children()
        self.aim_local = self._aim_directions()
        self._proxy = None
        self.proxy_triangles = 0

    def _rest_world(self, nodes, parent, has_inverse_bind):
        """World matrix per joint for the bind pose, in joint order."""
        if has_inverse_bind:
            return np.linalg.inv(self.inverse_bind)
        local = []
        for spec in nodes:
            if "matrix" in spec:
                local.append(np.array(spec["matrix"], dtype=np.float64).reshape(4, 4).T)
            else:
                translation = np.array(spec.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
                quat = spec.get("rotation", [0.0, 0.0, 0.0, 1.0])
                rotation = _quat_matrix(quat[0], quat[1], quat[2], quat[3])
                scale = np.array(spec.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
                local.append(_homogeneous(translation, rotation, scale))
        world = [None] * len(nodes)

        def resolve(index):
            if world[index] is None:
                up = parent[index]
                world[index] = local[index] if up < 0 else resolve(up) @ local[index]
            return world[index]

        return np.array([resolve(node) for node in self.joint_nodes])

    # ---- skeleton
    def _aim_children(self):
        """For each joint, the child that defines its aim direction.

        A joint with several children (a hand, the chest, the pelvis) still gets
        one so its handle is draggable; the first child is the natural pick.
        """
        children = [[] for _ in self.joint_names]
        for index, parent in enumerate(self.joint_parent):
            if parent >= 0:
                children[parent].append(index)
        return [kids[0] if kids else -1 for kids in children]

    def _aim_directions(self):
        """Each joint's direction to its aim child, in the joint's local frame."""
        directions = np.zeros((len(self.joint_names), 3))
        for index, child in enumerate(self.aim_child):
            if child < 0:
                continue
            offset = self.rest_translation[child]
            length = float(np.linalg.norm(offset))
            if length > 1e-9:
                directions[index] = offset / length
        return directions

    def local_matrices(self, deltas):
        """Local matrices for a pose; ``deltas`` is ``{joint: (rx, ry, rz)}`` degrees."""
        count = len(self.joint_names)
        out = np.empty((count, 4, 4))
        for index in range(count):
            rotation = self.rest_rotation[index]
            delta = deltas.get(self.joint_names[index]) if deltas else None
            if delta is not None:
                rotation = rotation @ euler_matrix(delta[0], delta[1], delta[2])
            out[index] = _homogeneous(
                self.rest_translation[index], rotation, self.rest_scale[index],
            )
        return out

    def world_matrices(self, deltas):
        """World matrix per joint for the given pose."""
        local = self.local_matrices(deltas)
        world = np.empty_like(local)
        for index, parent in enumerate(self.joint_parent):
            world[index] = local[index] if parent < 0 else world[parent] @ local[index]
        return world

    def joint_positions(self, deltas):
        """World position of every joint for the given pose."""
        return self.world_matrices(deltas)[:, :3, 3]

    def joint_world_direction(self, index, deltas):
        """World-space direction from joint ``index`` toward its aim child."""
        child = self.aim_child[index]
        if child < 0:
            return None
        world = self.world_matrices(deltas)
        start = world[index][:3, 3]
        end = world[child][:3, 3]
        return end - start

    def aim_delta(self, joint, target_world_direction, deltas):
        """Euler delta that aims ``joint``'s bone along ``target_world_direction``.

        The delta is the minimal rotation taking the bone's rest direction (in the
        parent frame) onto the target direction (also expressed in the parent
        frame), so joints can be posed by pointing them rather than by guessing
        per-bone axes. Returns ``None`` when the joint cannot be aimed.
        """
        index = self.joint_index.get(joint)
        if index is None:
            return None
        local_direction = self.aim_local[index]
        if not np.any(local_direction):
            return None
        target = np.asarray(target_world_direction, dtype=np.float64)
        length = float(np.linalg.norm(target))
        if length < 1e-9:
            return None
        target = target / length
        parent = self.joint_parent[index]
        if parent < 0:
            parent_rotation = np.eye(3)
        else:
            parent_world = self.world_matrices(deltas)[parent]
            parent_rotation = parent_world[:3, :3]
        target_parent = parent_rotation.T @ target
        desired_local = self.rest_rotation[index].T @ target_parent
        delta = rotation_between(local_direction, desired_local)
        return matrix_euler(delta)

    # ---- skinning
    def _skin_matrices(self, deltas):
        world = self.world_matrices(deltas)
        return np.einsum("nij,njk->nik", world, self.inverse_bind)

    def _skin_full(self, deltas):
        skin = self._skin_matrices(deltas)
        position = np.zeros((len(self.position), 3), dtype=np.float64)
        normal = np.zeros((len(self.position), 3), dtype=np.float64)
        for k in range(4):
            matrix = skin[self.joint_ids[:, k]]
            basis = matrix[:, :3, :3]
            weight = self.weights[:, k][:, None]
            position += weight * (
                np.einsum("nij,nj->ni", basis, self.position) + matrix[:, :3, 3]
            )
            normal += weight * np.einsum("nij,nj->ni", basis, self.normal)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        return position, normal

    def skin(self, deltas, proxy=False):
        """Skin the mesh (or its proxy) for a pose: positions, normals, triangles."""
        position, normal = self._skin_full(deltas)
        if proxy and self._proxy is not None:
            survivors, _count, triangles = self._proxy
            return (position[survivors].astype(np.float32),
                    normal[survivors].astype(np.float32), triangles)
        return position.astype(np.float32), normal.astype(np.float32), self.triangles

    # ---- proxy
    def build_proxy(self, target=PROXY_TRIANGLES):
        """Decimate the mesh to ``target`` triangles for interactive drags.

        Edges are collapsed shortest-and-flattest first, which is a cheap edge
        collapse decimator. A collapsed vertex snaps onto a surviving neighbour
        rather than onto a cell average, so the surface keeps its shape far
        better than grid clustering at the same triangle count, and rendering the
        proxy is just gathering the surviving vertices.
        """
        triangles = self.triangles.astype(np.int64, copy=True)
        count = len(self.position)
        parent = np.arange(count, dtype=np.int64)
        alive = np.ones(len(triangles), dtype=bool)
        incident = [[] for _ in range(count)]
        for face in range(len(triangles)):
            for vertex in triangles[face]:
                incident[vertex].append(face)

        def find(vertex):
            root = vertex
            while parent[root] != root:
                root = parent[root]
            while parent[vertex] != root:
                parent[vertex], vertex = root, parent[vertex]
            return root

        edges = np.unique(
            np.sort(np.concatenate((
                triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
            )), axis=1), axis=0)
        offsets = self.position[edges[:, 0]] - self.position[edges[:, 1]]
        lengths = np.linalg.norm(offsets.astype(np.float64), axis=1)
        flatness = np.clip(
            (self.normal[edges[:, 0]] * self.normal[edges[:, 1]]).sum(1), -1.0, 1.0,
        )
        order = np.argsort(lengths * (1.0 - flatness))

        count_live = len(triangles)
        for index in order:
            if count_live <= target:
                break
            a = find(int(edges[index, 0]))
            b = find(int(edges[index, 1]))
            if a == b:
                continue
            parent[b] = a
            for face in incident[b]:
                if not alive[face]:
                    continue
                row = triangles[face]
                row[row == b] = a
                if row[0] == row[1] or row[1] == row[2] or row[0] == row[2]:
                    alive[face] = False
                    count_live -= 1
                else:
                    incident[a].append(face)
            incident[b] = []

        triangles = triangles[alive]
        survivors = np.unique(triangles)
        remap = np.zeros(count, dtype=np.int64)
        remap[survivors] = np.arange(len(survivors))
        self._proxy = (survivors, len(survivors), remap[triangles])
        self.proxy_triangles = len(triangles)
        return self.proxy_triangles


def load(path, name=None):
    """Load and cache a model from ``path``."""
    key = os.path.abspath(path)
    if key not in _CACHE:
        gltf, binary = _load_gltf_bytes(key)
        model = Model(name or os.path.splitext(os.path.basename(key))[0], gltf, binary, key)
        model.build_proxy()
        _CACHE[key] = model
    return _CACHE[key]


# ---------------------------------------------------------------- primitives
# Parametric solids the 3D viewport can add beside the mannequin. Each shape is
# generated around the origin at a unit size (about one model unit, where a
# figure is ~1.8 tall), then an object's per-axis ``dim`` and its Euler ``rot``
# are baked into the vertices. Baking keeps the primitives flowing through the
# existing CPU rasteriser and GPU skinned shader unchanged: a :class:`Primitive`
# duck-types a one-joint :class:`Model`, so both backends treat it as a rigid
# body (see ``poses.render_sprite``).
PRIMITIVE_SHAPES = ("box", "sphere", "cylinder", "cone", "plane", "torus")
PRIMITIVE_SEGMENTS = 24
PRIMITIVE_TUBE_SEGMENTS = 12
PRIMITIVE_CACHE_LIMIT = 512

_PRIMITIVES = {}
_PRIMITIVE_MODELS = {}


def _orient(positions, normals, triangles):
    """Make every triangle wind outwards, i.e. counter-clockwise as seen from the
    side its vertex normals face.

    The rasteriser culls a triangle whose winding-derived normal points away from
    the camera, so the winding has to agree with the shading normal. Deriving it
    from the average vertex normal keeps each generator free to emit triangles in
    whatever order is convenient.
    """
    pos = np.asarray(positions, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    out = []
    for i, j, k in triangles:
        face = np.cross(pos[j] - pos[i], pos[k] - pos[i])
        if float(np.dot(face, nrm[i] + nrm[j] + nrm[k])) < 0.0:
            out.append((i, k, j))
        else:
            out.append((i, j, k))
    return out


def _box_mesh():
    h = 0.5
    faces = [
        ((0, 0, -1), [(-h, h, -h), (h, h, -h), (h, -h, -h), (-h, -h, -h)]),
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]),
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)]),
        ((0, 1, 0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)]),
        ((1, 0, 0), [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)]),
        ((-1, 0, 0), [(-h, -h, h), (-h, h, h), (-h, h, -h), (-h, -h, -h)]),
    ]
    positions, normals, triangles = [], [], []
    for normal, quad in faces:
        base = len(positions)
        positions.extend(quad)
        normals.extend([normal] * 4)
        triangles.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    return _orient(positions, normals, triangles), positions, normals


def _sphere_mesh(segments=PRIMITIVE_SEGMENTS, rings=None):
    rings = rings or max(8, segments // 2)
    positions, normals, triangles = [], [], []

    def point(i, j):
        theta = math.pi * i / rings
        phi = 2.0 * math.pi * j / segments
        return (
            math.sin(theta) * math.cos(phi) * 0.5,
            math.cos(theta) * 0.5,
            math.sin(theta) * math.sin(phi) * 0.5,
        )

    for i in range(rings):
        for j in range(segments):
            quad = (point(i, j), point(i + 1, j),
                    point(i + 1, j + 1), point(i, j + 1))
            base = len(positions)
            positions.extend(quad)
            for x, y, z in quad:
                length = math.sqrt(x * x + y * y + z * z) or 1.0
                normals.append((x / length, y / length, z / length))
            triangles.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    return _orient(positions, normals, triangles), positions, normals


def _cylinder_mesh(segments=PRIMITIVE_SEGMENTS):
    positions, normals, triangles = [], [], []
    for j in range(segments):
        a = 2.0 * math.pi * j / segments
        b = 2.0 * math.pi * (j + 1) / segments
        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)
        base = len(positions)
        positions.extend([
            (ca * 0.5, -0.5, sa * 0.5), (cb * 0.5, -0.5, sb * 0.5),
            (cb * 0.5, 0.5, sb * 0.5), (ca * 0.5, 0.5, sa * 0.5),
        ])
        normals.extend([(ca, 0.0, sa), (cb, 0.0, sb), (cb, 0.0, sb), (ca, 0.0, sa)])
        triangles.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
        top = len(positions)
        positions.extend([(0.0, 0.5, 0.0), (ca * 0.5, 0.5, sa * 0.5), (cb * 0.5, 0.5, sb * 0.5)])
        normals.extend([(0.0, 1.0, 0.0)] * 3)
        triangles.append((top, top + 1, top + 2))
        bottom = len(positions)
        positions.extend([(0.0, -0.5, 0.0), (ca * 0.5, -0.5, sa * 0.5), (cb * 0.5, -0.5, sb * 0.5)])
        normals.extend([(0.0, -1.0, 0.0)] * 3)
        triangles.append((bottom, bottom + 1, bottom + 2))
    return _orient(positions, normals, triangles), positions, normals


def _cone_mesh(segments=PRIMITIVE_SEGMENTS):
    positions, normals, triangles = [], [], []
    for j in range(segments):
        a = 2.0 * math.pi * j / segments
        b = 2.0 * math.pi * (j + 1) / segments
        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)
        normal = ((ca + cb) * 0.5, 0.5, (sa + sb) * 0.5)
        length = math.sqrt(sum(v * v for v in normal)) or 1.0
        normal = tuple(v / length for v in normal)
        base = len(positions)
        positions.extend([(0.0, 0.5, 0.0), (ca * 0.5, -0.5, sa * 0.5), (cb * 0.5, -0.5, sb * 0.5)])
        normals.extend([normal] * 3)
        triangles.append((base, base + 1, base + 2))
        bottom = len(positions)
        positions.extend([(0.0, -0.5, 0.0), (ca * 0.5, -0.5, sa * 0.5), (cb * 0.5, -0.5, sb * 0.5)])
        normals.extend([(0.0, -1.0, 0.0)] * 3)
        triangles.append((bottom, bottom + 1, bottom + 2))
    return _orient(positions, normals, triangles), positions, normals


def _plane_mesh():
    h = 0.5
    positions = [(-h, 0.0, -h), (h, 0.0, -h), (h, 0.0, h), (-h, 0.0, h)]
    normals = [(0.0, 1.0, 0.0)] * 4 + [(0.0, -1.0, 0.0)] * 4
    positions = positions + positions
    triangles = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
    return _orient(positions, normals, triangles), positions, normals


def _torus_mesh(segments=32, tube=PRIMITIVE_TUBE_SEGMENTS):
    ring, thickness = 0.35, 0.15
    positions, normals, triangles = [], [], []

    def point(i, j):
        u = 2.0 * math.pi * i / segments
        v = 2.0 * math.pi * j / tube
        normal = (math.cos(v) * math.cos(u), math.sin(v), math.cos(v) * math.sin(u))
        center = (ring * math.cos(u), 0.0, ring * math.sin(u))
        return tuple(center[k] + thickness * normal[k] for k in range(3)), normal

    for i in range(segments):
        for j in range(tube):
            quad = (point(i, j), point(i + 1, j),
                    point(i + 1, j + 1), point(i, j + 1))
            base = len(positions)
            for position, normal in quad:
                positions.append(position)
                normals.append(normal)
            triangles.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    return _orient(positions, normals, triangles), positions, normals


_PRIMITIVE_BUILDERS = {
    "box": _box_mesh,
    "sphere": _sphere_mesh,
    "cylinder": _cylinder_mesh,
    "cone": _cone_mesh,
    "plane": _plane_mesh,
    "torus": _torus_mesh,
}


def primitive_geometry(shape, dim=(1.0, 1.0, 1.0), rot=(0.0, 0.0, 0.0)):
    """Vertex positions, normals and triangles for a shaped primitive.

    The base shape is scaled per axis by ``dim`` and turned by the Euler ``rot``
    (degrees), then cached so repeated frames during a drag reuse the arrays.
    """
    shape = shape if shape in _PRIMITIVE_BUILDERS else "box"
    dim = [float(v) for v in dim]
    rot = [float(v) for v in rot]
    key = (shape,) + tuple(round(v, 4) for v in dim) + tuple(round(v, 4) for v in rot)
    cached = _PRIMITIVES.get(key)
    if cached is not None:
        return cached
    triangles, positions, normals = _PRIMITIVE_BUILDERS[shape]()
    pos = np.asarray(positions, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    safe = [v if abs(v) > 1e-6 else 1e-6 for v in dim]
    pos = pos * np.asarray(dim, dtype=np.float64)[None, :]
    nrm = nrm / np.asarray(safe, dtype=np.float64)[None, :]
    if any(abs(v) > 1e-9 for v in rot):
        rotation = euler_matrix(rot[0], rot[1], rot[2])
        pos = pos @ rotation.T
        nrm = nrm @ rotation.T
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    result = (
        pos.astype(np.float32), nrm.astype(np.float32),
        np.asarray(triangles, dtype=np.int32),
    )
    if len(_PRIMITIVES) >= PRIMITIVE_CACHE_LIMIT:
        _PRIMITIVES.clear()
    _PRIMITIVES[key] = result
    return result


class Primitive:
    """A rigid primitive that duck-types :class:`Model` for both render backends.

    The skinned pipeline expects joints and weights; a primitive supplies a
    single identity joint with every vertex fully bound to it, so the skinning
    math (and the GPU vertex shader) is a no-op and the shape draws unchanged.
    Its :attr:`height` is a fixed reference unit rather than the geometry's own
    height, so the caller's pixels-per-model-unit scale stays independent of the
    per-axis ``dim``.
    """

    def __init__(self, shape="box", dim=(1.0, 1.0, 1.0), rot=(0.0, 0.0, 0.0)):
        self.shape = shape
        self.path = ("primitive", shape)
        self.name = shape
        position, normal, triangles = primitive_geometry(shape, dim, rot)
        self.position = position
        self.normal = normal
        self.triangles = triangles
        low, high = position.min(0), position.max(0)
        self.bounds_min = low.astype(np.float64)
        self.bounds_max = high.astype(np.float64)
        self.height = 1.0
        self.joint_names = ["root"]
        self.joint_index = {"root": 0}
        self.joint_parent = [-1]
        count = len(position)
        self.joint_ids = np.zeros((count, 4), dtype=np.int32)
        weights = np.zeros((count, 4), dtype=np.float32)
        weights[:, 0] = 1.0
        self.weights = weights
        self.inverse_bind = np.eye(4, dtype=np.float64).reshape(1, 4, 4)
        self.rest_translation = np.zeros((1, 3), dtype=np.float64)
        self.rest_rotation = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
        self.rest_scale = np.ones((1, 3), dtype=np.float64)
        self.skin_margin = (
            float(np.linalg.norm(position, axis=1).max()) if count else 0.0
        )
        self.aim_child = [-1]
        self.aim_local = np.zeros((1, 3), dtype=np.float64)
        self._proxy = None
        self.proxy_triangles = 0

    def world_matrices(self, _deltas):
        return self.inverse_bind

    def joint_positions(self, _deltas):
        return np.zeros((1, 3), dtype=np.float64)

    def _skin_matrices(self, _deltas):
        return self.inverse_bind

    def skin(self, _deltas, proxy=False):
        return self.position, self.normal, self.triangles


def primitive(shape, dim=(1.0, 1.0, 1.0), rot=(0.0, 0.0, 0.0)):
    """A cached :class:`Primitive` for ``shape`` at the given dim/rot.

    Caching matters for the GPU backend, which keys its vertex buffers on the
    model object's identity: a fresh object every frame would leak a buffer per
    frame.
    """
    shape = shape if shape in _PRIMITIVE_BUILDERS else "box"
    key = (shape, tuple(round(float(v), 3) for v in dim),
           tuple(round(float(v), 3) for v in rot))
    cached = _PRIMITIVE_MODELS.get(key)
    if cached is None:
        if len(_PRIMITIVE_MODELS) >= PRIMITIVE_CACHE_LIMIT:
            _PRIMITIVE_MODELS.clear()
        cached = Primitive(shape, key[1], key[2])
        _PRIMITIVE_MODELS[key] = cached
    return cached


# ------------------------------------------------------------------- camera
def view_matrix(yaw, pitch):
    """A camera rotation; the figure turns ``yaw`` about Y then ``pitch`` about X."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    return rot_x @ rot_y


def project(points, scale, yaw=0.0, pitch=0.0, perspective=0.0):
    """Project model-space points. Returns ``(x, y, camera_space_points)``.

    ``perspective`` of 0 is orthographic; a positive value is a focal length in
    model units, which foreshortens nearer geometry (the eye sits at that
    distance along +Z).
    """
    camera = np.asarray(points, dtype=np.float64) @ view_matrix(yaw, pitch).T
    if perspective and perspective > 0.0:
        factor = perspective / np.maximum(perspective - camera[:, 2], 1e-3)
    else:
        factor = 1.0
    x = camera[:, 0] * scale * factor
    y = -camera[:, 1] * scale * factor
    return x, y, camera


# ---------------------------------------------------------------- rasteriser
def rasterize(x, y, camera, normals, triangles, width, height, base_rgb,
              smooth=True):
    """Depth-tested triangle fill with smooth (or flat) Lambert shading.

    Filling triangles in numpy costs far more in per-call overhead than in
    arithmetic, so the loop does the minimum: the projected pixel rows are shared
    instead of re-allocated, a triangle entirely behind the depth already written
    is rejected before any coverage work, and shading is evaluated only on the
    pixels the triangle actually wins (via boolean indexing) rather than across
    its whole bounding box. With ``smooth`` the three vertex normals are
    interpolated per pixel, so the body reads as a curved surface; otherwise one
    Lambert term from the face normal is used.

    ``x``/``y`` are sprite-space pixel coordinates, ``camera`` the camera-space
    3D positions and ``normals`` the rotated vertex normals. Returns
    ``(color, alpha)`` float arrays of shape ``(height, width, 3)`` and
    ``(height, width)``.
    """
    color = np.zeros((height, width, 3), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)
    depth = np.full((height, width), -1e18, dtype=np.float32)

    ax, ay, az = x[triangles[:, 0]], y[triangles[:, 0]], camera[triangles[:, 0], 2]
    bx, by, bz = x[triangles[:, 1]], y[triangles[:, 1]], camera[triangles[:, 1], 2]
    cx, cy, cz = x[triangles[:, 2]], y[triangles[:, 2]], camera[triangles[:, 2], 2]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    ca, cb, cc = camera[triangles[:, 0]], camera[triangles[:, 1]], camera[triangles[:, 2]]
    face = np.cross(cb - ca, cc - ca)
    face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)
    n0 = normals[triangles[:, 0]]
    n1 = normals[triangles[:, 1]]
    n2 = normals[triangles[:, 2]]
    if smooth:
        vertex = np.empty((len(triangles), 3, 3), dtype=np.float32)
        vertex[:, 0], vertex[:, 1], vertex[:, 2] = n0, n1, n2
        uniform = None
    else:
        uniform = base_rgb[None, :] * (
            AMBIENT + DIFFUSE * np.clip(face @ LIGHT, 0.0, 1.0)
        )[:, None]

    visible = (np.abs(area) > 1e-9) & (face[:, 2] > 0.0)
    candidates = np.nonzero(visible)[0]
    deepest = np.maximum(np.maximum(az, bz), cz)[candidates]
    candidates = candidates[np.argsort(-deepest)]

    xrow = np.arange(width, dtype=np.float32)
    ycol = np.arange(height, dtype=np.float32)

    for i in candidates:
        x0 = int(max(0, math.floor(min(ax[i], bx[i], cx[i]))))
        x1 = int(min(width, math.ceil(max(ax[i], bx[i], cx[i])) + 1))
        y0 = int(max(0, math.floor(min(ay[i], by[i], cy[i]))))
        y1 = int(min(height, math.ceil(max(ay[i], by[i], cy[i])) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        zmax = max(az[i], bz[i], cz[i])
        region_depth = depth[y0:y1, x0:x1]
        if zmax <= region_depth.min():
            continue
        xs = xrow[x0:x1]
        ys = ycol[y0:y1, None]
        e0 = (bx[i] - ax[i]) * (ys - ay[i]) - (by[i] - ay[i]) * (xs - ax[i])
        e1 = (cx[i] - bx[i]) * (ys - by[i]) - (cy[i] - by[i]) * (xs - bx[i])
        e2 = (ax[i] - cx[i]) * (ys - cy[i]) - (ay[i] - cy[i]) * (xs - cx[i])
        a = area[i]
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        d = (az[i] * e1 + bz[i] * e2 + cz[i] * e0) / a
        take = inside & (d > region_depth)
        if not take.any():
            continue
        rows, cols = np.nonzero(take)
        if smooth:
            w0 = (e1[rows, cols] / a)[:, None]
            w1 = (e2[rows, cols] / a)[:, None]
            w2 = (e0[rows, cols] / a)[:, None]
            normal = n0[i][None, :] * w0 + n1[i][None, :] * w1 + n2[i][None, :] * w2
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
            lambert = np.clip(normal @ LIGHT, 0.0, 1.0)
            rgb = base_rgb[None, :] * (AMBIENT + DIFFUSE * lambert)[:, None]
        else:
            rgb = uniform[i]
        color[y0:y1, x0:x1][rows, cols] = rgb
        alpha[y0:y1, x0:x1][rows, cols] = 1.0
        region_depth[rows, cols] = d[rows, cols]
    return color, alpha


def render_sprite(model, deltas, *, height_px, color="#c9a06a", yaw=0.0, pitch=0.0,
                  roll=0.0, perspective=0.0, proxy=False, opacity=1.0,
                  supersample=None, quality=1.0, smooth=True, force_cpu=False,
                  offset=(0.0, 0.0, 0.0)):
    """Render a posed model to an RGBA sprite; returns ``(sprite, left, top)``.

    ``left``/``top`` are the sprite's top-left corner relative to the model
    origin's projected position in the owning document, so the caller composites
    the sprite at ``(anchor_x + left, anchor_y + top)``.

    ``offset`` is the model's position in a shared scene (model units). It is
    added to every vertex before the camera, so a figure placed farther from the
    camera is drawn smaller (true per-figure foreshortening) while ``left``/``top``
    stay relative to the figure's own origin.

    ``quality`` below 1 renders at a reduced internal resolution and scales the
    result back up with nearest-neighbour, which is the interactive drag path:
    rasterisation cost is dominated by pixel work, so a drag renders at a
    fraction of the size and commits at full quality.

    When a GPU context is available the figure is drawn by ``mesh_gpu`` at full
    mesh detail instead; ``proxy`` and ``quality`` apply only to the CPU
    fallback, which is what makes the CPU-only install still work.
    """
    offset = np.asarray(offset, dtype=np.float64).reshape(3)
    backend = None if force_cpu else _gpu()
    if backend is not None:
        rendered = _render_gpu(
            backend, model, deltas, height_px=height_px, color=color, yaw=yaw,
            pitch=pitch, roll=roll, perspective=perspective, opacity=opacity,
            offset=offset,
        )
        if rendered is not None:
            return rendered

    positions, normals, triangles = model.skin(deltas, proxy=proxy)
    positions = positions + offset[None, :]
    scale = float(height_px) * quality / model.height
    x, y, camera = project(positions, scale, yaw, pitch, perspective)
    # ``left``/``top`` are relative to the figure's own origin, so undo the
    # offset's projection (the perspective divide already applied the depth).
    origin_x, origin_y, _origin_camera = project(offset[None, :], scale, yaw, pitch, perspective)
    x = x - float(origin_x[0])
    y = y - float(origin_y[0])
    if roll:
        angle = math.radians(roll)
        cos, sin = math.cos(angle), math.sin(angle)
        x, y = x * cos - y * sin, x * sin + y * cos
    rotated_normals = normals @ view_matrix(yaw, pitch).T

    left = int(math.floor(float(x.min()))) - 1
    top = int(math.floor(float(y.min()))) - 1
    width = max(1, int(math.ceil(float(x.max()))) + 1 - left)
    height = max(1, int(math.ceil(float(y.max()))) + 1 - top)

    if supersample is not None:
        ss = max(1, int(supersample))
    else:
        ss = 1
        for candidate in (2, 1):
            if width * height * candidate * candidate <= MAX_SPRITE_PIXELS:
                ss = candidate
                break

    base_rgb = np.array(_hex_rgb(color), dtype=np.float32) / 255.0
    color_buf, alpha_buf = rasterize(
        (x - left) * ss, (y - top) * ss, camera, rotated_normals, triangles,
        width * ss, height * ss, base_rgb, smooth=smooth,
    )
    out = np.zeros((height * ss, width * ss, 4), dtype=np.uint8)
    out[..., :3] = np.clip(color_buf * 255.0, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(alpha_buf * 255.0 * opacity, 0, 255).astype(np.uint8)
    sprite = Image.fromarray(out, "RGBA")
    if ss > 1:
        sprite = antialias.downscale(sprite, ss)
    if quality < 1.0:
        sprite = sprite.resize(
            (max(1, int(round(width))), max(1, int(round(height)))), Image.NEAREST,
        )
        left = left / quality
        top = top / quality
    return sprite, left, top


def _hex_rgb(value):
    value = (value or "#000000").lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        return [int(value[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return [0, 0, 0]
