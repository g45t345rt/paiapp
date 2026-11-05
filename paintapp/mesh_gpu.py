"""GPU rendering backend for the Poses figures (optional).

The CPU rasteriser in :mod:`paintapp.mesh` exists so the core install stays
Pillow + numpy only, but filling 14k triangles in a Python loop costs a few
hundred milliseconds a frame, which is far too slow for interactive posing. This
module does the same job on the GPU: the skinned mesh is drawn into an offscreen
framebuffer by an OpenGL 3.3 shader (four-bone skinning in the vertex shader,
per-pixel Lambert shading in the fragment shader), read back and returned as a
Pillow image. A full-quality frame costs about a millisecond, so a drag needs no
decimated proxy at all.

It is entirely optional: :func:`available` reports whether a context could be
created and :func:`render` returns ``None`` when it cannot, so callers fall back
to the CPU path. ``moderngl`` is the only extra dependency (see
``requirements-gpu.txt``).
"""

import math

import numpy as np
from PIL import Image

from . import antialias

AMBIENT = 0.36
DIFFUSE = 0.74
LIGHT = np.array([-0.40, 0.52, 0.76], dtype=np.float32)
LIGHT /= np.linalg.norm(LIGHT)

# The largest framebuffer we will ask for, per side.
MAX_SIZE = 4096

# Depth is linear in camera-space z, normalised over this many model units. The
# figure is under two units tall, so this keeps every pose inside the clip range
# while giving the depth buffer plenty of resolution.
DEPTH_RANGE = 4.0

_VERTEX = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in ivec4 in_joints;
in vec4 in_weights;
uniform mat4 u_bones[{joints}];
uniform mat3 u_view;
uniform mat4 u_projection;
uniform vec3 u_offset;
uniform float u_depth_range;
out vec3 v_normal;
void main() {{
    vec4 position = vec4(0.0);
    vec3 normal = vec3(0.0);
    for (int i = 0; i < 4; ++i) {{
        mat4 bone = u_bones[in_joints[i]];
        float weight = in_weights[i];
        position += weight * (bone * vec4(in_position, 1.0));
        normal += weight * (mat3(bone) * in_normal);
    }}
    v_normal = u_view * normal;
    vec3 camera = u_view * position.xyz + u_offset;
    gl_Position = u_projection * vec4(camera, 1.0);
    // Depth has to come from the camera-space z. The projection matrix only
    // carries x/y (the perspective divide lives in w), so writing it here keeps
    // the depth buffer linear and lets nearer surfaces win instead of the
    // last-drawn triangle. Camera z grows toward the viewer, and OpenGL depth
    // grows away from it, hence the negation and division by w.
    gl_Position.z = -(camera.z / u_depth_range) * gl_Position.w;
}}
"""

_FRAGMENT = """
#version 330
in vec3 v_normal;
uniform vec3 u_color;
uniform vec3 u_light;
uniform float u_opacity;
out vec4 f_color;
void main() {
    vec3 normal = normalize(v_normal);
    float lambert = clamp(dot(normal, u_light), 0.0, 1.0);
    f_color = vec4(u_color * (0.36 + 0.74 * lambert), u_opacity);
}
"""

_CONTEXT = None
_CONTEXT_TRIED = False
_GL_MODELS = {}


def available():
    """True when an offscreen GL context can be created and used."""
    return _context() is not None


def _context():
    global _CONTEXT, _CONTEXT_TRIED
    if _CONTEXT is not None:
        return _CONTEXT
    if _CONTEXT_TRIED:
        return None
    _CONTEXT_TRIED = True
    try:
        import moderngl
        _CONTEXT = moderngl.create_standalone_context(require=330)
    except Exception:
        _CONTEXT = None
    return _CONTEXT


def _projection_matrix(scale, width, height, left, top, roll, perspective):
    """The camera-space -> clip matrix, matching ``mesh.project`` exactly.

    ``mesh.project`` maps a camera-space point to sprite pixels with
    ``x = cam.x * scale * focal / (focal - cam.z)`` and ``y = -cam.y * scale *
    focal / (focal - cam.z)``, then subtracts the sprite origin and rotates by
    ``roll`` in pixel space. That whole map is affine in ``(cam, 1)`` once the
    perspective divide is folded into the clip ``w`` (``focal - cam.z``), so it
    becomes one 4x4 matrix and the shader stays trivial.
    """
    cos = math.cos(math.radians(roll))
    sin = math.sin(math.radians(roll))
    perspective = float(perspective or 0.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    if perspective > 0.0:
        factor = scale * perspective
        constant = perspective
        matrix[3, 2] = -1.0
        matrix[3, 3] = perspective
    else:
        factor = scale
        constant = 1.0
        matrix[3, 3] = 1.0
    offset_x = 2.0 * (-left) / width - 1.0
    offset_y = 2.0 * (-top) / height - 1.0
    ax = 2.0 * factor / width
    ay = 2.0 * factor / height
    matrix[0, 0] = ax * cos
    matrix[0, 1] = ax * sin
    matrix[0, 3] = offset_x * constant
    matrix[1, 0] = ay * sin
    matrix[1, 1] = -ay * cos
    matrix[1, 3] = offset_y * constant
    if perspective > 0.0:
        matrix[0, 2] = -offset_x
        matrix[1, 2] = -offset_y
    return matrix


def _gl_model(ctx, model):
    """Per-model GPU resources: the skinned vertex buffer and index buffer."""
    key = id(model)
    cached = _GL_MODELS.get(key)
    if cached is not None:
        return cached
    import moderngl
    joints = len(model.joint_names)
    program = ctx.program(
        vertex_shader=_VERTEX.format(joints=joints),
        fragment_shader=_FRAGMENT,
    )

    interleaved = np.empty(
        len(model.position),
        dtype=np.dtype([
            ("position", "f4", 3), ("normal", "f4", 3),
            ("joints", "i4", 4), ("weights", "f4", 4),
        ]),
    )
    interleaved["position"] = model.position
    interleaved["normal"] = model.normal
    interleaved["joints"] = model.joint_ids
    interleaved["weights"] = model.weights
    vbo = ctx.buffer(interleaved.tobytes())
    ibo = ctx.buffer(model.triangles.astype("i4").tobytes())
    vao = ctx.vertex_array(
        program,
        [(vbo, "3f 3f 4i 4f", "in_position", "in_normal", "in_joints", "in_weights")],
        ibo,
    )
    cached = (program, vao, len(model.triangles) * 3, joints)
    _GL_MODELS[key] = cached
    return cached


def render(model, deltas, *, height_px, color="#c9a06a", yaw=0.0, pitch=0.0,
           roll=0.0, perspective=0.0, opacity=1.0, supersample=1, samples=4,
           offset=(0.0, 0.0, 0.0)):
    """Render a posed model on the GPU; returns ``(sprite, left, top)``.

    ``left``/``top`` are the sprite's top-left corner relative to the model
    origin's projected position, exactly like :func:`mesh.render_sprite`, so the
    two backends are interchangeable. ``offset`` is the model's position in a
    shared scene (model units), added to every vertex in camera space so a figure
    farther from the camera is drawn smaller. Anti-aliasing comes from
    multisampling (``samples``), so no supersampled downscale is needed;
    ``supersample`` above one is still honoured for extra-fine committed frames.
    """
    ctx = _context()
    if ctx is None:
        return None
    from . import mesh as mesh_module

    program, vao, vertex_count, _joints = _gl_model(ctx, model)
    offset = np.asarray(offset, dtype=np.float64).reshape(3)
    view = mesh_module.view_matrix(yaw, pitch)

    # Frame the sprite from the joints plus the largest vertex-to-joint offset at
    # rest, which costs a forward-kinematics pass instead of skinning 8.5k
    # vertices. The sprite is trimmed to its ink after readback, so a slightly
    # generous box only costs a few transparent pixels. The returned ``left``/``top``
    # stay relative to the figure's own origin (like the CPU renderer), so the
    # framing is taken after subtracting the origin's projection and rolling.
    scale = float(height_px) / model.height
    positions = model.joint_positions(deltas) + offset[None, :]
    x, y, _camera = mesh_module.project(positions, scale, yaw, pitch, perspective)
    origin_x, origin_y, _origin_camera = mesh_module.project(
        offset[None, :], scale, yaw, pitch, perspective,
    )
    origin_x = float(origin_x[0])
    origin_y = float(origin_y[0])
    x = x - origin_x
    y = y - origin_y
    if roll:
        angle = math.radians(roll)
        cos, sin = math.cos(angle), math.sin(angle)
        x, y = x * cos - y * sin, x * sin + y * cos
        origin_x, origin_y = (origin_x * cos - origin_y * sin,
                              origin_x * sin + origin_y * cos)
    margin = model.skin_margin * scale * 1.15
    left = int(math.floor(float(x.min()) - margin)) - 1
    top = int(math.floor(float(y.min()) - margin)) - 1
    width = max(1, int(math.ceil(float(x.max()) + margin)) + 1 - left)
    height = max(1, int(math.ceil(float(y.max()) + margin)) + 1 - top)
    # The projection matrix subtracts its own ``left``/``top`` in world-projected
    # space, so hand it the frame origin plus the rolled figure origin to cancel
    # the offset; the returned values stay figure-relative.
    frame_left = left + origin_x
    frame_top = top + origin_y

    ss = max(1, int(supersample))
    if (width * ss > MAX_SIZE) or (height * ss > MAX_SIZE):
        ss = max(1, MAX_SIZE // max(width, height))
    buffer_width = width * ss
    buffer_height = height * ss

    import moderngl
    multisample = int(samples) if samples else 0
    target = ctx.framebuffer(color_attachments=[ctx.texture((buffer_width, buffer_height), 4)])
    if multisample > 1:
        attachment = ctx.renderbuffer(
            (buffer_width, buffer_height), 4, samples=multisample,
        )
        depth = ctx.depth_renderbuffer((buffer_width, buffer_height), samples=multisample)
        draw = ctx.framebuffer(color_attachments=[attachment], depth_attachment=depth)
    else:
        depth = ctx.depth_texture((buffer_width, buffer_height))
        draw = ctx.framebuffer(
            color_attachments=[ctx.texture((buffer_width, buffer_height), 4)],
            depth_attachment=depth,
        )
    try:
        draw.use()
        ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        view = mesh_module.view_matrix(yaw, pitch).astype("f4")
        program["u_view"].write(view.T.tobytes())
        program["u_offset"].value = tuple(
            float(v) for v in (mesh_module.view_matrix(yaw, pitch) @ offset)
        )
        program["u_projection"].write(_projection_matrix(
            scale * ss, buffer_width, buffer_height, frame_left * ss,
            frame_top * ss, roll, perspective,
        ).T.tobytes())
        program["u_color"].value = tuple(_rgb(color))
        program["u_light"].value = tuple(float(v) for v in LIGHT)
        program["u_opacity"].value = float(opacity)
        program["u_depth_range"].value = DEPTH_RANGE
        bones = model._skin_matrices(deltas)
        program["u_bones"].write(bones.transpose(0, 2, 1).astype("f4").tobytes())
        vao.render(moderngl.TRIANGLES, vertices=vertex_count)
        if multisample > 1:
            ctx.copy_framebuffer(target, draw)
        raw = target.read(components=4)
    finally:
        if multisample > 1:
            draw.release()
            attachment.release()
        else:
            draw.release()
        depth.release()
        target.release()

    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(buffer_height, buffer_width, 4)
    sprite = Image.fromarray(pixels.copy(), "RGBA")
    if ss > 1:
        sprite = antialias.downscale(sprite, ss)
    return _trim(sprite, left, top)


def _trim(sprite, left, top):
    """Trim a rendered sprite to its ink, adjusting the document offset."""
    mask = np.asarray(sprite.getchannel("A")) > 0
    rows = np.nonzero(mask.any(axis=1))[0]
    columns = np.nonzero(mask.any(axis=0))[0]
    if not len(rows) or not len(columns):
        return None
    top_row, bottom_row = int(rows[0]), int(rows[-1]) + 1
    left_column, right_column = int(columns[0]), int(columns[-1]) + 1
    trimmed = sprite.crop((left_column, top_row, right_column, bottom_row))
    return trimmed, left + left_column, top + top_row


def _rgb(value):
    value = (value or "#000000").lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        return [int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        return [0.0, 0.0, 0.0]
