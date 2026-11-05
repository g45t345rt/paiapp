"""Offscreen-GL pixel processing shared by the raster tools (optional).

The Poses figures already draw through an OpenGL 3.3 context (see
:mod:`paintapp.mesh_gpu`). The same offscreen pass - a texture in, a framebuffer
out, read back - turns any Pillow image into another Pillow image with a
fragment shader, which is what the warp, filter and selection tools use to move
their per-pixel work onto the GPU.

Like :mod:`paintapp.mesh_gpu` this is optional. :func:`available` reports whether
a GL context could be created, and every entry point returns ``None`` when it
could not, so callers keep their numpy/Pillow path and the app still starts on a
machine with no GL. ``moderngl`` is the only dependency and it is already a core
requirement.

:func:`render` uploads whatever images it is handed and frees them again, which
suits a one-off pass. A tool that runs the same source through many passes - the
transform gizmo resamples a layer on every drag event - uploads it once with
:func:`texture` instead and passes the handle in ``textures``; see
:class:`Texture`.
"""

import numpy as np
from PIL import Image

# Full-screen triangle: one vertex over each screen edge, so the fragment shader
# runs once per output pixel with no attribute interpolation to worry about.
_VERTEX = """#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_VERTICES = np.array(
    (-1.0, -1.0, 3.0, -1.0, -1.0, 3.0), dtype="f4",
)

_CONTEXT = None
_CONTEXT_TRIED = False
_PROGRAMS = {}
_VAOS = {}


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


def _program(fragment):
    program = _PROGRAMS.get(fragment)
    if program is None:
        program = _context().program(vertex_shader=_VERTEX, fragment_shader=fragment)
        _PROGRAMS[fragment] = program
    return program


def _vao(fragment):
    cached = _VAOS.get(fragment)
    if cached is None:
        import moderngl
        ctx = _context()
        program = _program(fragment)
        vbo = ctx.buffer(_VERTICES.tobytes())
        vertex_array = ctx.vertex_array(program, [(vbo, "2f", "in_pos")])
        cached = (vertex_array, vbo)
        _VAOS[fragment] = cached
    return cached[0]


def _has(program, name):
    """Whether ``name`` is an active uniform (it may have been optimised out)."""
    try:
        program[name]
        return True
    except KeyError:
        return False


def _set_uniform(program, name, value):
    import moderngl
    uniform = program[name]
    if isinstance(value, np.ndarray):
        uniform.write(np.ascontiguousarray(value, dtype="f4").tobytes())
    elif isinstance(value, (tuple, list)):
        if len(value) == 1:
            uniform.value = float(value[0])
        elif len(value) == 2:
            uniform.value = (float(value[0]), float(value[1]))
        elif len(value) == 3:
            uniform.value = tuple(float(v) for v in value)
        else:
            uniform.value = tuple(float(v) for v in value)
    elif isinstance(value, bool):
        uniform.value = bool(value)
    elif isinstance(value, int):
        uniform.value = int(value)
    elif isinstance(value, float):
        uniform.value = float(value)
    else:  # pragma: no cover - internal misuse
        raise TypeError(f"unsupported uniform {name!r}: {type(value)!r}")


def _upload(ctx, entry, storage="f4"):
    """An input ``entry`` becomes a texture and its pixel size.

    ``entry`` is either a Pillow image (normalised to ``0..1``) or a float RGBA
    array already in ``0..1`` (used by the warps, which keep the source
    premultiplied). Float storage keeps the sampling linear, so a warp reads the
    same values the numpy resampler would. ``storage="f1"`` stores the bytes as
    normalised 8-bit instead: a quarter of the texture memory, and no float
    conversion of the whole image, which is worth it for a caller whose source
    is bytes already and whose result is only a preview (the transform gizmo).
    """
    import moderngl
    byte = storage == "f1"
    if isinstance(entry, np.ndarray):
        if byte:
            array = np.clip(entry * 255.0 + 0.5, 0.0, 255.0)
            array = np.ascontiguousarray(array.astype("u1"))
        else:
            array = np.ascontiguousarray(entry, dtype="f4")
        height, width = array.shape[:2]
    elif byte:
        array = np.ascontiguousarray(np.asarray(entry.convert("RGBA")))
        height, width = array.shape[:2]
    else:
        height, width = entry.height, entry.width
        array = np.asarray(entry.convert("RGBA"), dtype="f4") / 255.0
    texture = ctx.texture(
        (width, height), 4, array.tobytes(), dtype="f1" if byte else "f4")
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False
    return texture, width, height


class Texture:
    """A source image kept uploaded in one GL texture across many passes.

    :func:`render` uploads a bare Pillow image for a single pass and releases it
    again, which is what a one-off filter or warp wants. A tool that resamples
    the same source on every event of a drag (the transform gizmo) would pay that
    upload - a float conversion of the whole image - on each one, so it holds the
    upload open instead. Pass the handle in the ``textures`` list exactly like a
    Pillow image and :meth:`release` it when the session ends.

    ``mipmaps`` builds the full mip chain and samples it with trilinear
    filtering. A tool that scales *down* needs it: a fixed-size filter kernel
    aliases the detail it skips, while the hardware picks the level whose
    footprint matches the scale (it derives the level from the affine mapping's
    own derivatives, so no shader-side level maths is needed).
    """

    def __init__(self, gl_texture, size):
        self._texture = gl_texture
        self.size = size

    def release(self):
        """Free the GL texture; safe to call more than once."""
        if self._texture is not None:
            self._texture.release()
            self._texture = None


def texture(image, mipmaps=False, storage="f4"):
    """Upload ``image`` once for reuse across passes; ``None`` when no GL.

    See :class:`Texture`. ``storage`` is :func:`_upload`'s, ``"f4"`` by default
    and ``"f1"`` for a byte source, which is what a caller whose input is bytes
    already wants: a quarter of the texture memory and no float conversion of
    the whole image. Like every entry point here it never raises: a machine
    without a usable context gets ``None`` back and the caller keeps its CPU
    path.
    """
    ctx = _context()
    if ctx is None:
        return None
    try:
        gl_texture, width, height = _upload(ctx, image, storage)
        if mipmaps:
            import moderngl
            gl_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            gl_texture.build_mipmaps()
    except Exception:
        return None
    return Texture(gl_texture, (width, height))


def render(fragment, size, textures=(), uniforms=None, dtype="u1",
           target_dtype=None):
    """Run ``fragment`` once per pixel over an empty ``size`` ``(w, h)`` target.

    Returns the framebuffer in top-down image order: an ``(h, w, 4)`` uint8
    array by default, or the raw ``(h, w, 4)`` float32 values when ``dtype`` is
    ``"f4"`` (which lets a caller quantise exactly as its numpy path does). It
    returns ``None`` if no context is available or the pass failed, in which case
    the caller falls back to its CPU path. ``textures`` are bound to ``u_tex0``,
    ``u_tex1`` ... in order and may be Pillow images or float RGBA arrays in
    ``0..1``; a :class:`Texture` handle is used as-is instead of being uploaded
    and released. The shader sees the pixel coordinate in ``gl_FragCoord`` with
    ``u_size`` set to ``(w, h)`` and the origin at the bottom-left.

    ``target_dtype`` is the storage of the render target, ``"f4"`` (the default)
    or an 8-bit ``"f1"``. An ``"f1"`` target lets the readback skip the float
    conversion of the whole buffer *and* the float write, which is most of the
    cost of a full-canvas pass - worth it for a caller whose pixels are bytes
    anyway (the transform preview), and avoided by the warps and filters, whose
    numpy paths agree with the float one to the byte.
    """
    ctx = _context()
    if ctx is None:
        return None
    import moderngl
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    textures = list(textures)
    uploaded = []
    target = None
    framebuffer = None
    storage = target_dtype or "f4"
    try:
        program = _program(fragment)
        for index, entry in enumerate(textures):
            if isinstance(entry, Texture):
                if entry._texture is None:
                    return None
                texture = entry._texture
                _w, _h = entry.size
            else:
                texture, _w, _h = _upload(ctx, entry)
                uploaded.append(texture)
            name = f"u_tex{index}"
            if _has(program, name):
                texture.use(index)
                program[name].value = index
        if _has(program, "u_size"):
            program["u_size"].value = (float(width), float(height))
        for name, value in (uniforms or {}).items():
            if _has(program, name):
                _set_uniform(program, name, value)
        target = ctx.texture((width, height), 4, dtype=storage)
        framebuffer = ctx.framebuffer(color_attachments=[target])
        framebuffer.use()
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        _vao(fragment).render(moderngl.TRIANGLES, vertices=3)
        raw = framebuffer.read(components=4, dtype=storage)
        # The upload treats row 0 as v=0 at gl_FragCoord.y=0 and the readback
        # returns row 0 at gl_FragCoord.y=0, so the two cancel: reading back
        # top-down already, with gl_FragCoord.x/y matching image x/y (y downward)
        # and the first uploaded row at the top.
        if storage == "f4":
            floats = np.frombuffer(raw, dtype=np.float32).reshape(
                height, width, 4)
            if dtype == "f4":
                return floats
            return np.clip(floats * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
        if dtype == "f4":
            return pixels.astype("f4") / 255.0
        return pixels
    except Exception:
        return None
    finally:
        for texture in uploaded:
            texture.release()
        if framebuffer is not None:
            framebuffer.release()
        if target is not None:
            target.release()


def image(fragment, image_or_size, textures=(), uniforms=None):
    """Like :func:`render` but returning a Pillow ``RGBA`` image, or ``None``."""
    if hasattr(image_or_size, "size"):
        size = image_or_size.size
    else:
        size = image_or_size
    pixels = render(fragment, size, textures=textures, uniforms=uniforms)
    if pixels is None:
        return None
    return Image.fromarray(pixels, "RGBA")