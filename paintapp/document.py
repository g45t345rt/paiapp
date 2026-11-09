"""Document models: a :class:`Canvas` holds a stack of :class:`Layer` images."""

import math

from PIL import Image, ImageChops, ImageDraw

from . import brush, filters, path, poses, textrender, vector
from .history import History

# Largest content area (in pixels) for which the below/above backdrop cache is
# kept at document resolution. The cache lets a view transform (zoom or pan)
# resample two cached buffers instead of re-compositing every layer, so a
# many-layer document stays responsive while zooming. The buffers are
# content-sized, so a canvas larger than this falls back to a whole-stack
# flatten per render rather than holding hundreds of megabytes.
MAX_BACKDROP_PIXELS = 12_000_000

# Layer blend modes, in menu order. "normal" is plain over-compositing; the
# others mix the layer's color with the pixels already composited beneath it.
# Pillow's ImageChops provide the C implementations for :data:`BLEND_OPS`; the
# remaining modes (dodge/burn, exclusion and the HSL hue/saturation/colour/
# luminosity set) are computed with numpy by ``brush._blend``, the same code the
# paint tools use, so a layer and a brush share one blend implementation.
BLEND_MODES = [
    ("normal", "Normal"),
    ("darken", "Darken"),
    ("multiply", "Multiply"),
    ("color_burn", "Color Burn"),
    ("linear_burn", "Linear Burn"),
    ("darker_color", "Darker Color"),
    ("lighten", "Lighten"),
    ("screen", "Screen"),
    ("color_dodge", "Color Dodge"),
    ("add", "Add"),
    ("lighter_color", "Lighter Color"),
    ("overlay", "Overlay"),
    ("soft_light", "Soft Light"),
    ("hard_light", "Hard Light"),
    ("vivid_light", "Vivid Light"),
    ("linear_light", "Linear Light"),
    ("pin_light", "Pin Light"),
    ("hard_mix", "Hard Mix"),
    ("difference", "Difference"),
    ("exclusion", "Exclusion"),
    ("subtract", "Subtract"),
    ("divide", "Divide"),
    ("hue", "Hue"),
    ("saturation", "Saturation"),
    ("color", "Color"),
    ("luminosity", "Luminosity"),
]
BLEND_LABELS = {mode: label for mode, label in BLEND_MODES}

# Blend modes grouped the way image editors present them. The layer blend
# combobox inserts a display-only divider between groups so the long list reads
# as categories instead of one flat column. Divider rows use the same
# ``— Group —`` style as the generation panel's model picker
# (:func:`paintapp.generation.GenerationPanel._model_catalog`).
BLEND_GROUPS = [
    ("Normal", ["normal"]),
    ("Darken", ["darken", "multiply", "color_burn", "linear_burn",
                "darker_color"]),
    ("Lighten", ["lighten", "screen", "color_dodge", "add", "lighter_color"]),
    ("Contrast", ["overlay", "soft_light", "hard_light", "vivid_light",
                  "linear_light", "pin_light", "hard_mix"]),
    ("Comparative", ["difference", "exclusion", "subtract", "divide"]),
    ("Component", ["hue", "saturation", "color", "luminosity"]),
]
BLEND_DIVIDER_PREFIX = "\u2014 "


def blend_divider_label(group):
    """A display-only group header in the models-combobox divider style."""
    return BLEND_DIVIDER_PREFIX + group + " \u2014"


def is_blend_divider(label):
    """True when ``label`` is a display-only divider in the blend menu."""
    return str(label).startswith(BLEND_DIVIDER_PREFIX)


def blend_mode_labels():
    """Return layer blend menu values, grouped with display-only dividers."""
    labels = []
    for group, modes in BLEND_GROUPS:
        labels.append(blend_divider_label(group))
        for mode in modes:
            labels.append(BLEND_LABELS[mode])
    return labels


def blend_headers():
    """The set of display-only divider labels in the blend menu."""
    return {blend_divider_label(group) for group, _modes in BLEND_GROUPS}
BLEND_OPS = {
    "multiply": ImageChops.multiply,
    "screen": ImageChops.screen,
    "overlay": ImageChops.overlay,
    "darken": ImageChops.darker,
    "lighten": ImageChops.lighter,
    "difference": ImageChops.difference,
    "add": ImageChops.add,
    "subtract": ImageChops.subtract,
    "hard_light": ImageChops.hard_light,
    "soft_light": ImageChops.soft_light,
}


def normalize_blend(mode):
    """Coerce ``mode`` to a known blend id, falling back to ``"normal"``."""
    return mode if mode in BLEND_LABELS else "normal"


def _with_opacity(image, opacity):
    """Return ``image`` with its alpha channel scaled by ``opacity``."""
    if opacity >= 1.0:
        return image
    scaled = image.copy()
    alpha = scaled.getchannel("A").point(lambda a: int(a * opacity))
    scaled.putalpha(alpha)
    return scaled


# ------------------------------------------------------------------- affine
# A 2-D affine transform is a 6-tuple ``(a, b, c, d, e, f)`` mapping
# ``(x, y)`` to ``(a*x + c*y + e, b*x + d*y + f)``. Link layers use one to
# place their master's content independently of the master (see
# :meth:`Layer.set_transform`).
IDENTITY_TRANSFORM = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _affine_mul(first, second):
    """Compose two affine transforms; ``second`` is applied first."""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _affine_translate(tx, ty):
    return (1.0, 0.0, 0.0, 1.0, float(tx), float(ty))


def _affine_invert(matrix):
    """Return the inverse of an affine transform (identity when singular)."""
    a, b, c, d, e, f = matrix
    det = a * d - b * c
    if not det:
        return IDENTITY_TRANSFORM
    return (
        d / det, -b / det, -c / det, a / det,
        (c * f - d * e) / det, (b * e - a * f) / det,
    )


def _affine_bounds(matrix, size):
    """The ``(left, top, right, bottom)`` box a transform maps ``size`` into."""
    width, height = size
    points = (
        (0.0, 0.0), (float(width), 0.0),
        (float(width), float(height)), (0.0, float(height)),
    )
    xs = []
    ys = []
    for x, y in points:
        xs.append(matrix[0] * x + matrix[2] * y + matrix[4])
        ys.append(matrix[1] * x + matrix[3] * y + matrix[5])
    return (min(xs), min(ys), max(xs), max(ys))


def _paste_layer(base, layer, left, top, clip=None, scale=1.0, offset=None):
    """Alpha-composite one layer into ``base``, whose top-left is document
    ``(left, top)``, cropping the layer to the overlap first.

    Cropping matters: PIL's ``alpha_composite(dest=...)`` crops the *background*
    to the overlay's box, so handing it a full-size layer over a small region
    buffer silently expands the work back to the whole layer (the viewport-only
    render would then cost as much as a full-canvas flatten).

    ``base`` holds ``scale`` pixels per document pixel. When ``scale < 1`` (a
    zoomed-out view) the layer's contribution is cropped and then downscaled
    before compositing, so the work tracks the viewport's pixels rather than the
    document rectangle it covers.

    ``clip`` is an optional region-sized ``L`` image (the clip base's alpha);
    where it is given the layer's own alpha is multiplied by it, so the layer
    only shows through the base's opaque pixels.

    ``offset`` overrides the position of the layer image's top-left in document
    space (defaulting to :attr:`Layer.offset`); the exploded-grid view passes a
    layout position here so the same compositing code lays a layer out in a cell
    without changing the document.
    """
    ox, oy = offset if offset is not None else layer.offset
    lw, lh = layer.size
    region_w = base.width / scale
    region_h = base.height / scale
    ix0 = max(0, ox - left)
    iy0 = max(0, oy - top)
    ix1 = min(region_w, ox + lw - left)
    iy1 = min(region_h, oy + lh - top)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    sx0 = ix0 + left - ox
    sy0 = iy0 + top - oy
    box = (
        int(round(sx0)), int(round(sy0)),
        int(round(sx0 + (ix1 - ix0))), int(round(sy0 + (iy1 - iy0))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    # Fetch only the overlap, so a masked/filtered layer is not copied and
    # re-multiplied at full size on every dab. ``ix0``/``iy0`` are already
    # relative to the region's top-left.
    sub = layer.composited_region(box)
    dest = (int(round(ix0 * scale)), int(round(iy0 * scale)))
    if scale != 1.0:
        width = max(1, int(round(sub.width * scale)))
        height = max(1, int(round(sub.height * scale)))
        sub = sub.resize((width, height), Image.BOX)
    # The overlay must be RGBA to composite: a layer's own image can be a
    # non-alpha mode (e.g. an RGB result stored by a tool), and PIL's
    # ``alpha_composite`` raises "images do not match" on a mode mismatch.
    if sub.mode != "RGBA":
        sub = sub.convert("RGBA")
    blend_mode = layer.blend
    if blend_mode != "normal":
        # Mix the layer's color with the pixels already under it (the backdrop),
        # keeping the layer's own alpha; the over-composite below then applies
        # that alpha, which is the standard "blend then composite" formula.
        backdrop = base.crop((dest[0], dest[1],
                              dest[0] + sub.width, dest[1] + sub.height))
        backdrop_rgb = backdrop.convert("RGB")
        sub_rgb = sub.convert("RGB")
        blend_op = BLEND_OPS.get(blend_mode)
        if blend_op is not None:
            mixed = blend_op(backdrop_rgb, sub_rgb)
        else:
            # Dodge/burn, exclusion and the HSL modes: share the tools' numpy
            # implementation rather than duplicating the channel maths.
            mixed = brush._blend(blend_mode, backdrop_rgb, sub_rgb)
        # A fully transparent backdrop pixel has no color to blend against, so a
        # bare area would otherwise read as black (e.g. multiply over alpha 0);
        # fall back to the layer's own pixels there, matching the brush tools.
        # Threshold the mask first: feeding the raw alpha to ``Image.composite``
        # would linearly mix the layer's raw color into *partially* transparent
        # pixels, washing out the blend for alpha 1..254. Only alpha 0 has no
        # color to blend with; the final ``alpha_composite`` below applies the
        # layer's alpha and the backdrop's coverage correctly.
        if backdrop.mode == "RGBA":
            opaque = backdrop.getchannel("A").point(lambda a: 255 if a else 0)
            mixed = Image.composite(mixed, sub_rgb, opaque)
        mixed = mixed.convert("RGBA")
        mixed.putalpha(sub.getchannel("A"))
        sub = mixed
    if clip is not None:
        clip_sub = clip.crop((dest[0], dest[1],
                              dest[0] + sub.width, dest[1] + sub.height))
        sub = sub.copy()
        sub.putalpha(ImageChops.multiply(sub.getchannel("A"), clip_sub))
    if layer.opacity < 1.0:
        sub = _with_opacity(sub, layer.opacity)
    base.alpha_composite(sub, dest=dest)


class Group:
    """A named set of layers treated as one unit at composite time.

    Members are the layers whose :attr:`Layer.group` is this object; they are
    kept adjacent in :attr:`Canvas.layers` (see :meth:`Canvas.normalize_groups`)
    so the stack order stays unambiguous. The group's ``visible`` and
    ``opacity`` apply to the members composited together, so a partially
    transparent group fades as a whole instead of letting overlaps show through.

    ``filters`` is a non-destructive filter chain (see ``filters``) applied to
    the group's members composited together, so one filter edits every member at
    once and a blur/glow bleeds across member edges the way a single layer's
    would. Like a layer's chain it is stored as specs and applied at composite
    time; unlike a layer it has no pixels of its own.
    """

    def __init__(self, name, group_id):
        self.name = name
        self.id = group_id
        self.visible = True
        self.opacity = 1.0
        self.collapsed = False
        # Non-destructive filter chain over the group's combined pixels; see the
        # class docstring and ``Canvas._flatten``.
        self.filters = []
        # An empty group created on its own (see :meth:`Canvas.add_group`)
        # survives without members; a group only ever held together by its
        # members is dropped once the last one leaves (see ``normalize_groups``).
        self.keep_empty = False

    def members(self, layers):
        """The layers in ``layers`` that belong to this group, in stack order."""
        return [layer for layer in layers if getattr(layer, "group", None) is self]

    def preview(self, layers):
        """Composite this group's member layers into one RGBA image, or ``None``.

        Mirrors how the canvas flattens the group as a single unit: its visible
        members are composited in stack order over the group's content bounds
        (with clipping and blend modes honoured), then the group's own filter
        chain and opacity are applied. Returns ``None`` when the group has no
        visible member. Used by the layer panel for a group row's preview
        thumbnail, so an organized group still reads at a glance.
        """
        members = self.members(layers)
        visible = [
            layer for layer in members
            if layer.visible and layer.opacity > 0.0
        ]
        if not visible:
            return None
        left = min(layer.offset[0] for layer in visible)
        top = min(layer.offset[1] for layer in visible)
        right = max(layer.offset[0] + layer.size[0] for layer in visible)
        bottom = max(layer.offset[1] + layer.size[1] for layer in visible)
        width = max(1, right - left)
        height = max(1, bottom - top)
        base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # Resolve each clipped member's clip base: the nearest member below it
        # that is not itself clipped, matching the canvas compositor.
        clip_bases = {}
        previous = None
        for layer in members:
            if getattr(layer, "clip", False):
                clip_bases[id(layer)] = previous
            else:
                previous = layer
        for layer in visible:
            clip = None
            if getattr(layer, "clip", False):
                clip_base = clip_bases.get(id(layer))
                if clip_base is None or not clip_base.visible:
                    continue
                image = clip_base.composited_image()
                ox, oy = clip_base.offset
                clip = image.crop(
                    (left - ox, top - oy, left - ox + width, top - oy + height)
                ).getchannel("A")
            _paste_layer(base, layer, left, top, clip)
        active = [spec for spec in self.filters if spec.get("enabled", True)]
        if active:
            base = filters.apply_chain(base, active)
        if self.opacity < 1.0:
            base = _with_opacity(base, self.opacity)
        return base


class Layer:
    def __init__(self, name, size, fill=(0, 0, 0, 0)):
        self.name = name
        self.image = Image.new("RGBA", size, fill)
        self.draw = ImageDraw.Draw(self.image)
        self.visible = True
        self.locked = False
        self.opacity = 1.0
        # Optional 8-bit mask, aligned to ``image`` in the layer's own pixel
        # space, whose value scales the layer's alpha at composite time (255 =
        # keep, 0 = hide). ``None`` means no mask (fully visible).
        self._mask = None
        # Copy-on-write markers: set while a history snapshot still references
        # the current ``image``/``mask`` object (see ``Layer.begin_write``).
        # Any in-place edit must fork first so the snapshot stays intact.
        self._image_shared = False
        self._mask_shared = False
        # Position of ``image``'s top-left corner in document pixels. Moving a
        # layer only changes this offset, so content dragged past the canvas
        # edge is retained and reappears when the layer is moved back.
        self.offset = (0, 0)
        # A transient ``(dx, dy)`` applied by the Move tool *while a drag is in
        # progress*: the document-space content (texts, vectors, poses, paths)
        # is drawn at ``spec - content_offset`` rather than ``spec - offset``, so
        # a drag can slide the layer's placement without translating every spec
        # (which would re-rasterise every overlay on each motion event). It is
        # held at the drag's start offset and baked into the specs on release
        # (see ``DrawingController._move_to`` / ``_end_move``).
        self._content_delta = (0, 0)
        # Non-destructive filter chain, applied in order at composite time.
        # Each entry is a spec dict (see ``filters``); the base ``image`` is
        # never modified by them.
        self.filters = []
        # Editable text objects (see ``textrender``) drawn over the filtered
        # pixels at composite time. Keeping the specs rather than only pixels
        # lets the text tool re-edit them; ``rasterize_texts`` folds them in.
        self.texts = []
        # Editable vector strokes (see ``vector``) drawn over the filtered pixels
        # at composite time. ``vector`` marks a layer created (or converted) as a
        # vector layer; the strokes are its content, kept editable so the vector
        # tools can move anchors, correct a line's width and erase across it.
        self.vector = False
        self.strokes = []
        # Editable compound fills (see ``vector``) drawn under the strokes at
        # composite time. Each paints one or more closed contours with a single
        # colour and an even-odd / nonzero rule, so an imported filled glyph
        # (whose holes come from overlapping contours) keeps its shape. They are
        # not per-anchor editable; the vector eraser/width tools act on strokes.
        self.fills = []
        # Editable path objects (see ``path``) drawn over the filtered pixels at
        # composite time, like texts and strokes. They are the Path tool's
        # committed output kept as specs so a path stays re-editable after it is
        # finished; ``rasterize_paths`` folds them into the pixels.
        self.paths = []
        # Editable 3D poses figures (see ``poses``) drawn over the
        # filtered pixels at composite time. Like texts and strokes they are
        # kept as specs, so the Poses tool can pose them and project
        # save/load can round-trip them.
        self.poses = []
        # The single 3D scene the layer's figures live in (shared camera and
        # ground grid; see ``poses.new_scene``). Orbit, tilt and roll act on this
        # one camera, so every figure on the layer moves as one scene.
        self.pose_scene = poses.new_scene(size)
        # Positive generation prompt that belongs to this layer alone, kept when
        # the active layer changes and shown in the Generate panel's Positive
        # prompt box. The negative prompt travels with the layer's gen_settings.
        self.positive_prompt = ""
        # Generation settings (the Generate panel's fields) that belong to this
        # layer, kept when the active layer changes and restored when it is
        # selected again. Empty until the panel is used on this layer, in which
        # case the canvas's settings (then the model's defaults) apply.
        self.gen_settings = {}
        # The :class:`Group` this layer belongs to, or ``None``. A group's
        # visibility and opacity apply to all of its members as a unit.
        self.group = None
        # When set, the layer is clipped to the alpha of the layer below it (its
        # clip base): it shows only through the base's opaque pixels. Set via
        # ``Clip to Layer Below`` in the Layers panel.
        self.clip = False
        # How the layer's color mixes with the pixels beneath it; see
        # :data:`BLEND_MODES`.
        self.blend = "normal"
        # Link Layer support. A *master* is a normal layer promoted to be a
        # shared source (``is_master`` plus a project-stable ``master_id``); a
        # *link* stores ``link_id`` (its master's id) and renders the master's
        # content at its own offset while keeping its own opacity/blend/
        # visibility. ``master`` is the resolved source object, rebuilt by the
        # app's relink pass after a project load or an undo restore.
        self.is_master = False
        self.master_id = None
        self.link_id = None
        self.master = None
        # Per-instance transform for a link layer: a 6-tuple mapping the
        # master's *content* pixels into this layer's own buffer. ``None`` means
        # the content is shown untransformed. ``set_transform`` normalises the
        # matrix and moves ``offset`` to the transformed bounding box, so the
        # renderer can keep using ``offset`` as the buffer's document position.
        self.transform = None
        self._transform_cache = None
        self._transform_key = None
        self._filter_cache = None
        self._filter_key = None
        self._image_token = 0
        self._text_token = 0
        self._text_cache = None
        self._text_key = None
        # Version of the stroke list and its rendered overlay, bumped whenever
        # the strokes change so the composited-image cache invalidates.
        self._vector_token = 0
        self._vector_cache = None
        self._vector_key = None
        # Version of the path list and its rendered overlay, bumped whenever the
        # paths change so the composited-image cache invalidates.
        self._path_token = 0
        self._path_cache = None
        self._path_key = None
        # Version of the poses list and its rendered overlay, bumped
        # whenever the figures change so the composited-image cache invalidates.
        self._poses_token = 0
        self._poses_cache = None
        self._poses_key = None
        # Version of the mask's pixels, bumped whenever the mask is replaced or
        # handed out for an in-place edit. The composited-image cache keys on it
        # so a mask edit invalidates the cache without a full pixel comparison.
        self._mask_token = 0
        self._composite_cache = None
        self._composite_key = None

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        # Reassigning the mask installs a fresh object no snapshot references,
        # so the copy-on-write marker clears.
        self._mask = value
        self._mask_shared = False
        self._mask_token += 1

    def set_image(self, image):
        """Replace the layer's pixels and rebind the draw handle."""
        self.image = image
        self.draw = ImageDraw.Draw(image)
        self._image_shared = False
        self.invalidate_filters()

    def begin_write(self):
        """Return the layer pixels for an in-place edit, forking if shared.

        History snapshots reference layer images instead of copying them, so a
        layer must fork before any in-place edit; otherwise the edit would also
        rewrite the snapshot and corrupt undo. The fork copies the layer once,
        on the first edit after a snapshot, and later edits reuse it.
        """
        if self._image_shared:
            self.set_image(self.image.copy())
        return self.image

    def begin_mask_write(self):
        """Return the layer mask for an in-place edit, forking if shared.

        The caller mutates the returned mask in place, so the mask version is
        bumped here to invalidate the composited-image cache for the edit.
        """
        self._mask_token += 1
        if self._mask is not None and self._mask_shared:
            self.mask = self._mask.copy()
        return self._mask

    def copy(self, name=None):
        """Return an independent duplicate of this layer preserving its pixels."""
        duplicate = Layer(name if name is not None else self.name, self.size)
        duplicate.set_image(self.image.copy())
        if self.mask is not None:
            duplicate.mask = self.mask.copy()
        duplicate.visible = self.visible
        duplicate.locked = self.locked
        duplicate.opacity = self.opacity
        duplicate.offset = tuple(self.offset)
        duplicate.filters = [filters.clone(spec) for spec in self.filters]
        duplicate.texts = [
            clone for clone in (textrender.clone(spec) for spec in self.texts)
            if clone is not None
        ]
        duplicate.vector = self.vector
        duplicate.strokes = [
            clone for clone in (vector.clone(spec) for spec in self.strokes)
            if clone is not None
        ]
        duplicate.fills = [
            clone for clone in (vector.clone_fill(spec) for spec in self.fills)
            if clone is not None
        ]
        duplicate.paths = [
            clone for clone in (path.clone(spec) for spec in self.paths)
            if clone is not None
        ]
        duplicate.poses = [
            clone for clone in (poses.clone(spec) for spec in self.poses)
            if clone is not None
        ]
        duplicate.pose_scene = poses.clone_scene(self.pose_scene)
        duplicate.positive_prompt = self.positive_prompt
        duplicate.gen_settings = dict(self.gen_settings)
        duplicate.group = self.group
        duplicate.clip = self.clip
        duplicate.blend = self.blend
        # Link Layer identity travels with the copy; the app's relink pass (or
        # Canvas.copy for a same-canvas master) re-points ``master``.
        duplicate.is_master = self.is_master
        duplicate.master_id = self.master_id
        duplicate.link_id = self.link_id
        duplicate.master = self.master
        duplicate.transform = (
            tuple(self.transform) if self.transform is not None else None
        )
        return duplicate

    def invalidate_filters(self):
        """Drop the cached filtered image (call after editing ``image``)."""
        self._image_token += 1
        self._filter_cache = None
        self._filter_key = None

    def adopt_content(self, other):
        """Take a copy of ``other``'s content, becoming an independent layer.

        Used when a link layer is unlinked, or when its master is deleted: the
        layer keeps showing what it did, but now owns the pixels, mask, effect
        chain and editable objects. The layer's own transform and appearance
        (``offset``/``opacity``/``blend``/``visible``/``locked``/``group``) are
        left untouched.
        """
        self.set_image(other.image.copy())
        self.mask = other.mask.copy() if other.mask is not None else None
        self.filters = [filters.clone(spec) for spec in other.filters]
        self.texts = [
            clone for clone in (textrender.clone(spec) for spec in other.texts)
            if clone is not None
        ]
        self.vector = bool(getattr(other, "vector", False))
        self.strokes = [
            clone for clone in (vector.clone(spec) for spec in other.strokes)
            if clone is not None
        ]
        self.fills = [
            clone for clone in (
                vector.clone_fill(spec) for spec in getattr(other, "fills", [])
            )
            if clone is not None
        ]
        self.fills = [
            clone for clone in (
                vector.clone_fill(spec) for spec in getattr(other, "fills", [])
            )
            if clone is not None
        ]
        self.paths = [
            clone for clone in (path.clone(spec) for spec in other.paths)
            if clone is not None
        ]
        self.poses = [
            clone for clone in (poses.clone(spec) for spec in other.poses)
            if clone is not None
        ]
        self.pose_scene = poses.clone_scene(getattr(other, "pose_scene", None))
        self.invalidate_filters()
        self.invalidate_texts()
        self.invalidate_vectors()
        self.invalidate_poses()
        self.invalidate_paths()

    @property
    def pixels_token(self):
        """A value that changes whenever this layer's pixels do.

        Views that mirror the layer's pixels without holding them - the layer
        panel's row previews - key their caches on it, so they can tell a real
        edit from an unrelated redraw without comparing pixels. A link layer
        reports its master's tokens, so the preview follows the master.
        """
        source = self.content_source()
        return (
            id(source.image), source._image_token, source._mask_token,
            source._text_token, source._vector_token, source._poses_token,
            source._path_token,
        )

    def content_offset(self):
        """``offset`` minus the in-progress move delta, for drawing content.

        Document-space content is rendered at ``spec - content_offset`` so a
        Move drag can slide the layer's placement (via ``offset``) while the
        content appears to travel with it, without translating the specs. During
        a drag ``content_offset`` stays at the drag's start, so the layer's own
        image (and its overlay caches) is unchanged and only its placement
        moves; with no drag in progress it is simply :attr:`offset`.
        """
        dx, dy = self._content_delta
        if not dx and not dy:
            return self.offset
        ox, oy = self.offset
        return (ox - dx, oy - dy)

    def invalidate_texts(self):
        """Drop the cached text overlay (call after editing ``texts``)."""
        self._text_token += 1
        self._text_cache = None
        self._text_key = None

    def invalidate_vectors(self):
        """Drop the cached stroke overlay (call after editing ``strokes``)."""
        self._vector_token += 1
        self._vector_cache = None
        self._vector_key = None

    def invalidate_fills(self):
        """Drop the cached fill overlay (call after editing ``fills``).

        Fills share the vector overlay cache, so this simply drops it too.
        """
        self.invalidate_vectors()

    def invalidate_poses(self):
        """Drop the cached figure overlay (call after editing ``poses``)."""
        self._poses_token += 1
        self._poses_cache = None
        self._poses_key = None

    def invalidate_paths(self):
        """Drop the cached path overlay (call after editing ``paths``)."""
        self._path_token += 1
        self._path_cache = None
        self._path_key = None

    def effective_image(self):
        """The layer's base pixels with the mask folded into the alpha channel."""
        source = self.content_source()
        if source.mask is None:
            return source.image
        masked = source.image.copy()
        masked.putalpha(ImageChops.multiply(masked.getchannel("A"), source.mask))
        return masked

    def filtered_image(self):
        """The base pixels with the non-destructive filter chain applied.

        Returns ``self.image`` unchanged when no filter is enabled. The result
        is cached until the base pixels or the chain change.
        """
        active = [spec for spec in self.filters if spec.get("enabled", True)]
        if not active:
            return self.image
        key = (self._image_token, filters.chain_key(self.filters))
        if self._filter_cache is not None and self._filter_key == key:
            return self._filter_cache
        result = filters.apply_chain(self.image, active)
        self._filter_cache = result
        self._filter_key = key
        return result

    def composited_image(self, include_mask=True):
        """What the layer contributes to a composite: filtered pixels, texts, masked.

        The masked result is cached and reused while the base pixels, filter
        chain, texts and mask are unchanged, so a repaint (viewport render, undo
        push, gallery refresh) does not re-copy and re-multiply a masked layer's
        full-size pixels every time.         A link renders its master's content at the
        link's own offset (see :meth:`content_source`); with a per-instance
        transform, the content is resampled through that transform first.

        ``include_mask=False`` returns the same content without folding the mask
        into its alpha. The layer panel uses it for the row thumbnail, which
        shows the layer and its mask as two separate previews.
        """
        source = self.content_source()
        if self.transform is not None:
            return self._transformed_image(source, self.transform, include_mask)
        return source._composited_image_at(self.content_offset(), include_mask)

    def _transformed_image(self, source, matrix, include_mask=True):
        """The source's composite resampled through ``matrix`` (cached).

        ``matrix`` maps the source's own pixel space into this layer's buffer
        space. The result is cropped to the matrix's bounding box, so its
        top-left is the box's top-left and the layer's ``offset`` places it.
        ``include_mask`` is forwarded to the source's composite.
        """
        base = source._composited_image_at(source.content_offset(), include_mask)
        left, top, right, bottom = _affine_bounds(matrix, base.size)
        left_i = math.floor(left)
        top_i = math.floor(top)
        size = (
            max(1, math.ceil(right - left_i)),
            max(1, math.ceil(bottom - top_i)),
        )
        key = (
            id(source),
            source._image_token,
            filters.chain_key(source.filters),
            source._text_token,
            source._vector_token,
            source._poses_token,
            source._path_token,
            source._mask_token,
            tuple(matrix),
            left_i,
            top_i,
            include_mask,
        )
        if self._transform_cache is not None and self._transform_key == key:
            return self._transform_cache
        inverse = _affine_invert(
            _affine_mul(_affine_translate(-left_i, -top_i), matrix)
        )
        data = (
            inverse[0], inverse[2], inverse[4],
            inverse[1], inverse[3], inverse[5],
        )
        result = base.transform(size, Image.AFFINE, data, resample=Image.BICUBIC)
        self._transform_cache = result
        self._transform_key = key
        return result

    def _composited_image_at(self, offset, include_mask=True):
        """``composited_image`` for this layer's own content drawn at ``offset``.

        Split out so a link layer can render its master's content at the link's
        offset; the cache keys on that offset as well as the content tokens.
        With ``include_mask=False`` the mask is left out of the alpha channel.
        """
        image = self.filtered_image()
        if self.texts:
            image = self._text_overlay(image, offset)
        if self.strokes or self.fills:
            image = self._vector_overlay(image, offset)
        if self.poses:
            image = self._poses_overlay(image, offset)
        if self.paths:
            image = self._path_overlay(image, offset)
        if self.mask is None or not include_mask:
            return image
        key = (
            self._image_token,
            filters.chain_key(self.filters),
            self._text_token,
            self._vector_token,
            self._poses_token,
            self._path_token,
            self._mask_token,
            tuple(offset),
        )
        if self._composite_cache is not None and self._composite_key == key:
            return self._composite_cache
        masked = image.copy()
        masked.putalpha(ImageChops.multiply(masked.getchannel("A"), self.mask))
        self._composite_cache = masked
        self._composite_key = key
        return masked

    def composited_region(self, box):
        """What the layer contributes to a composite, cropped to ``box``.

        ``box`` is ``(left, top, right, bottom)`` in the layer's own pixel
        space. The mask is folded into the crop's alpha only, so painting on a
        masked layer recomposites the dab's region instead of copying and
        re-multiplying the whole layer on every commit. Filters and text are
        still resolved through their full-image caches (so they are correct),
        then cropped. A link resolves the master's content at the link's offset;
        with a per-instance transform, the content is resampled first.
        """
        source = self.content_source()
        if self.transform is not None:
            return self._transformed_image(source, self.transform).crop(box)
        return source._composited_region_at(box, self.content_offset())

    def _composited_region_at(self, box, offset):
        """``composited_region`` for this layer's own content drawn at ``offset``."""
        image = self.filtered_image()
        if self.texts:
            image = self._text_overlay(image, offset)
        if self.strokes or self.fills:
            image = self._vector_overlay(image, offset)
        if self.poses:
            image = self._poses_overlay(image, offset)
        if self.paths:
            image = self._path_overlay(image, offset)
        sub = image.crop(box)
        if self.mask is not None:
            sub.putalpha(
                ImageChops.multiply(sub.getchannel("A"), self.mask.crop(box))
            )
        return sub

    def _text_overlay(self, image, offset=None):
        """``image`` with every text object drawn over it (cached)."""
        offset = self.content_offset() if offset is None else offset
        key = (
            self._image_token, filters.chain_key(self.filters), self._text_token,
            tuple(offset),
        )
        if self._text_cache is not None and self._text_key == key:
            return self._text_cache
        result = image.copy()
        for spec in self.texts:
            result = textrender.paste(result, spec, offset)
        self._text_cache = result
        self._text_key = key
        return result

    def _vector_overlay(self, image, offset=None):
        """``image`` with every stroke drawn over it (cached)."""
        offset = self.content_offset() if offset is None else offset
        key = (
            self._image_token, filters.chain_key(self.filters),
            self._text_token, self._vector_token, tuple(offset),
        )
        if self._vector_cache is not None and self._vector_key == key:
            return self._vector_cache
        result = image.copy()
        for spec in self.fills:
            result = vector.paste_fill(result, spec, offset)
        for spec in self.strokes:
            result = vector.paste(result, spec, offset)
        self._vector_cache = result
        self._vector_key = key
        return result

    def _poses_overlay(self, image, offset=None):
        """``image`` with every poses figure drawn over it (cached)."""
        offset = self.content_offset() if offset is None else offset
        key = (
            self._image_token, filters.chain_key(self.filters),
            self._text_token, self._vector_token, self._poses_token,
            tuple(offset),
        )
        if self._poses_cache is not None and self._poses_key == key:
            return self._poses_cache
        poses.sync_scene(self.poses, self.pose_scene)
        result = image.copy()
        for spec in self.poses:
            result = poses.paste(result, spec, offset)
        self._poses_cache = result
        self._poses_key = key
        return result

    def _path_overlay(self, image, offset=None):
        """``image`` with every path object drawn over it (cached)."""
        offset = self.content_offset() if offset is None else offset
        key = (
            self._image_token, filters.chain_key(self.filters),
            self._text_token, self._vector_token, self._poses_token,
            self._path_token, tuple(offset),
        )
        if self._path_cache is not None and self._path_key == key:
            return self._path_cache
        result = image.copy()
        for spec in self.paths:
            result = path.paste(result, spec, offset)
        self._path_cache = result
        self._path_key = key
        return result

    def rasterize_texts(self):
        """Fold every text object into the base pixels and clear the list.

        Returns ``True`` when something was rasterized, so callers can decide
        whether to report the layer as changed.
        """
        if not self.texts:
            return False
        offset = self.offset
        image = self.image
        for spec in self.texts:
            image = textrender.paste(image, spec, offset)
        self.texts = []
        # set_image rebinds the draw handle and drops the filter cache.
        self.set_image(image)
        self.invalidate_texts()
        return True

    def rasterize_vectors(self):
        """Fold every stroke into the base pixels, clearing the vector content.

        The layer stops being a vector layer once its geometry is baked, so the
        strokes are drawn into the pixels (forking first, in case a snapshot
        still shares them) and ``vector`` is cleared. Returns ``True`` when
        something was rasterized.
        """
        if not self.strokes and not self.fills:
            return False
        offset = self.offset
        image = self.begin_write()
        for spec in self.fills:
            image = vector.paste_fill(image, spec, offset)
        for spec in self.strokes:
            image = vector.paste(image, spec, offset)
        self.strokes = []
        self.fills = []
        self.vector = False
        # set_image rebinds the draw handle and drops the filter cache.
        self.set_image(image)
        self.invalidate_vectors()
        return True

    def rasterize_poses(self):
        """Fold every poses figure into the base pixels, clearing the list.

        Returns ``True`` when something was rasterized, so callers can decide
        whether to report the layer as changed.
        """
        if not self.poses:
            return False
        offset = self.offset
        poses.sync_scene(self.poses, self.pose_scene)
        image = self.begin_write()
        for spec in self.poses:
            image = poses.paste(image, spec, offset)
        self.poses = []
        # set_image rebinds the draw handle and drops the filter cache.
        self.set_image(image)
        self.invalidate_poses()
        return True

    def rasterize_paths(self):
        """Fold every path object into the base pixels and clear the list.

        Returns ``True`` when something was rasterized, so callers can decide
        whether to report the layer as changed.
        """
        if not self.paths:
            return False
        offset = self.offset
        image = self.begin_write()
        for spec in self.paths:
            image = path.paste(image, spec, offset)
        self.paths = []
        # set_image rebinds the draw handle and drops the filter cache.
        self.set_image(image)
        self.invalidate_paths()
        return True

    def is_locked(self):
        """A layer is locked when explicitly locked, hidden, or a link.

        Hidden layers are always locked so nothing can affect pixels you cannot
        see; the explicit flag locks a visible layer in place. A link layer is
        always locked because its content belongs to its master: it can only be
        changed by editing the master (see :meth:`content_source`).
        """
        return self.locked or not self.visible or self.is_link

    @property
    def is_link(self):
        """True when this layer renders another (master) layer's content."""
        return self.link_id is not None

    def content_source(self):
        """The layer holding the content this layer shows.

        A resolved link returns its master; every other layer (a normal layer,
        a master, or an unresolved link) returns itself, so callers can treat
        content uniformly.
        """
        if self.link_id is not None and self.master is not None:
            return self.master
        return self

    @property
    def size(self):
        """The size of this layer's composite buffer.

        A link with a per-instance transform renders into the transform's
        bounding box rather than the master's pixel size.
        """
        source = self.content_source()
        if self.transform is not None:
            left, top, right, bottom = _affine_bounds(
                self.transform, source.image.size
            )
            return (
                max(1, math.ceil(right - math.floor(left))),
                max(1, math.ceil(bottom - math.floor(top))),
            )
        return source.image.size

    def set_transform(self, matrix):
        """Place this layer's content under an affine transform.

        ``matrix`` maps the content's own pixel space into *document* space
        (``None`` restores the untransformed layer). The matrix is normalised so
        its bounding box starts at ``(0, 0)`` in the layer's buffer, and
        :attr:`offset` is moved to where that box sits in the document. The
        renderer then places the transformed buffer at ``offset`` as usual, so
        an arbitrary rotation or scale keeps its position instead of being
        cropped to its top-left corner.
        """
        if matrix is None:
            self.transform = None
            self._transform_cache = None
            self._transform_key = None
            return
        source = self.content_source()
        left, top, right, bottom = _affine_bounds(matrix, source.image.size)
        left_i = math.floor(left)
        top_i = math.floor(top)
        self.transform = _affine_mul(_affine_translate(-left_i, -top_i), matrix)
        self.offset = (left_i, top_i)
        self._transform_cache = None
        self._transform_key = None

    def clear_transform(self):
        """Drop the per-instance transform (back to the master's own pixels)."""
        self.set_transform(None)


class Asset:
    """A reference image kept with a canvas, outside the layer stack.

    Assets are not composited: they are a per-canvas collection of images (a
    finished generation, or a file the user added) shown in the Assets panel and
    inserted onto a layer on demand. Each has a stable per-canvas ``id``, a
    ``name`` and the ``image`` pixels; ``source`` records where it came from
    (``"generated"`` or ``"imported"``) and ``prompt`` keeps the prompt a
    generation used, for the panel's tooltip. ``settings`` keeps the Generate
    panel's fields (and both prompts) from the run that produced the image, so
    selecting a generated asset can restore them; it is empty for an imported
    asset.
    """

    def __init__(self, name, image, asset_id, source="", prompt="", settings=None):
        self.name = name
        self.image = image
        self.id = asset_id
        self.source = source
        self.prompt = prompt
        self.settings = dict(settings) if settings else {}

    def copy(self, name=None):
        """Return an independent duplicate sharing no pixels with this asset."""
        return Asset(
            name if name is not None else self.name,
            self.image.copy(),
            self.id,
            self.source,
            self.prompt,
            dict(self.settings),
        )


class MaskSurface:
    """A stand-alone 8-bit mask the mask brush can paint without a layer behind it.

    Provides the slice of the layer interface :class:`brush.Stroke` needs for a
    ``"mask"`` stroke (``size``, ``mask``, ``offset``, ``begin_mask_write`` and
    ``invalidate_filters``), so the per-canvas inpaint mask can be painted with
    the same brush, hardness and overlay as a layer mask while not existing as a
    layer. ``begin_mask_write`` forks the mask when a history snapshot shares it,
    exactly like :meth:`Layer.begin_mask_write`, so an unpainted snapshot is not
    rewritten.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self.offset = (0, 0)

    @property
    def size(self):
        return self.canvas.size

    @property
    def mask(self):
        return self.canvas.inpaint_mask

    def begin_mask_write(self):
        canvas = self.canvas
        if canvas.inpaint_mask is not None and canvas._inpaint_mask_shared:
            canvas.inpaint_mask = canvas.inpaint_mask.copy()
            canvas._inpaint_mask_shared = False
        return canvas.inpaint_mask

    def invalidate_filters(self):
        """No layer to invalidate; the overlay redraws from the mask directly."""


class Canvas:
    """A single document: one sized pixel buffer with its own layer stack."""

    def __init__(self, name, size):
        self.name = name
        self.width = int(size[0])
        self.height = int(size[1])
        self.layers = []
        self.active = 0
        self.layer_counter = 0
        # Layer groups, in no particular order; membership lives on each layer
        # via ``Layer.group``. ``group_counter`` names new groups.
        self.groups = []
        self.group_counter = 0
        self.widget = None
        # Exploded-grid view state: whether the grid is shown, and the cached
        # layout plus the stack fingerprint it was built for, rebuilt only when
        # the layers are added, removed, reordered or resized - not while
        # painting. It is a per-canvas view preference, persisted in the .pai
        # project (the cached layout is not).
        self.show_exploded = False
        self._exploded_layout = None
        self._exploded_key = None
        # Cache for :meth:`composite_region` and :meth:`composite_overview`: the
        # document-anchored flatten of the layers below and above the active
        # layer. Keeping it at a fixed region (the content bounds) rather than
        # the last requested viewport lets a zoom or pan reuse it: the view only
        # changes the region/scale, so the cached buffers are cropped and
        # resampled instead of every layer being re-composited. It also serves a
        # stroke, which changes only the active layer and so recomposites one
        # layer over the cached backdrop. ``_region_cache_signature`` keys every
        # layer's state; ``_last_render`` decides when to promote to it (see
        # :meth:`_flatten_cached`).
        self._region_cache_signature = None
        self._region_cache_images = None
        # Signature of the layer stack the last render used. A second render with
        # the same signature after the view or the active layer changed promotes
        # the below/above split to the content-sized backdrop cache (see
        # :meth:`_flatten_cached`); a one-off or unchanged render leaves it on the
        # cheap whole-stack path.
        self._last_render = None
        # Generation settings shared by every layer of this canvas, restored when
        # the canvas is activated. A layer's own ``gen_settings`` takes precedence
        # when it has them; an empty dict falls back to the model's defaults.
        self.gen_settings = {}
        # Ruler guides drawn over the canvas, as ``(orient, pos)`` pairs where
        # ``orient`` is ``"v"`` (a vertical line at document x ``pos``) or
        # ``"h"`` (a horizontal line at document y ``pos``). They annotate the
        # document without altering pixels, so they travel with the canvas.
        # When the app shares guides across canvases, the view draws the
        # app-level list instead (see ``CanvasView.set_shared_guides``) and this
        # per-canvas list is left untouched.
        self.guides = []
        # A canvas-sized 8-bit mask painted with the Mask tool that marks where a
        # generation may change pixels (255 = regenerate, 0 = keep). ``None``
        # until first painted; ``_inpaint_mask_shared`` is the copy-on-write flag
        # (see :class:`MaskSurface`).
        self.inpaint_mask = None
        self._inpaint_mask_shared = False
        # This canvas's own undo/redo stack (see ``paintapp.history``); its
        # snapshots capture only this document, so history is per canvas.
        self.history = History()
        # Reference images kept with this canvas (generation results and added
        # files), each an :class:`Asset`. They are not composited; they live
        # beside the layer stack for the Assets panel to insert on demand.
        self.assets = []
        self.asset_counter = 0
        # ICC colour management (see ``paintapp.colormanagement``). The layers
        # are composited in the ``working_profile`` space, and the view converts
        # that to the monitor space for display. ``proof_profile``, when set,
        # soft-proofs the preview against a print/output profile with
        # ``proof_intent``. Identifiers are ``"sRGB"`` (the default), a
        # ``.icc``/``.icm`` path, or an ``"icc:<sha1>"`` embedded id.
        self.working_profile = "sRGB"
        self.proof_profile = None
        self.proof_intent = "relative"

    def inpaint_mask_surface(self):
        """A :class:`MaskSurface` for this canvas's inpaint mask, creating it.

        The mask starts fully transparent (0 = nothing to regenerate) so the
        first paint reveals what generation is allowed to touch.
        """
        if self.inpaint_mask is None:
            self.inpaint_mask = Image.new("L", self.size, 0)
            self._inpaint_mask_shared = False
        return MaskSurface(self)

    def has_inpaint_mask(self):
        """True when a painted inpaint mask selects at least one pixel."""
        return self.inpaint_mask is not None and self.inpaint_mask.getbbox() is not None

    @property
    def size(self):
        return (self.width, self.height)

    def active_layer(self):
        if not (0 <= self.active < len(self.layers)):
            return None
        return self.layers[self.active]

    def add_layer(self, name=None, fill=(0, 0, 0, 0), index=None, vector=False):
        self.layer_counter += 1
        if name is None:
            name = f"Layer {self.layer_counter}"
        layer = Layer(name, self.size, fill)
        layer.vector = bool(vector)
        if index is None:
            index = len(self.layers)
        self.layers.insert(index, layer)
        return layer

    def remove_layer(self, layer):
        """Remove ``layer`` from this canvas's stack and return it.

        Group membership is cleared so the layer can be inserted into another
        canvas without carrying a group that belongs to this one. The active
        layer is kept when it still exists; the region cache is invalidated.
        Returns ``None`` when the layer is not in this canvas.
        """
        if layer not in self.layers:
            return None
        active = self.active_layer()
        self.layers.remove(layer)
        layer.group = None
        self.normalize_groups()
        if active in self.layers:
            self.active = self.layers.index(active)
        elif self.layers:
            self.active = min(max(self.active, 0), len(self.layers) - 1)
        else:
            self.active = 0
        self._region_cache_signature = None
        self._region_cache_images = None
        self._last_render = None
        self._exploded_key = None
        return layer

    def insert_layer(self, layer, index=None, group=None):
        """Insert an existing ``layer`` into this canvas's stack and return it.

        ``index`` defaults to the top of the stack. ``group`` is restored only
        when that group already belongs to this canvas; groups are canvas-local,
        so a foreign group is dropped rather than carried across. The inserted
        layer becomes active and the region cache is invalidated.
        """
        if layer is None:
            return None
        if index is None:
            index = len(self.layers)
        index = max(0, min(index, len(self.layers)))
        layer.group = group if group in self.groups else None
        self.layers.insert(index, layer)
        self.normalize_groups()
        self.active = self.layers.index(layer)
        self._region_cache_signature = None
        self._region_cache_images = None
        self._last_render = None
        self._exploded_key = None
        return layer

    def add_asset(self, name=None, image=None, source="", prompt="", settings=None,
                  index=None):
        """Add a reference image to this canvas's assets and return it.

        The image is stored as-is (copied into RGBA), independent of any layer
        or canvas size, so it keeps its own dimensions. A ``None`` image yields
        a transparent canvas-sized asset, for callers that only need a slot.
        ``settings`` records the Generate panel fields that produced a generated
        image, so selecting the asset can restore them.

        Assets are kept newest first, so the new asset goes to the front by
        default and the Assets panel shows the most recent addition at the top.
        Pass ``index`` to place it at a specific position instead, which loading
        uses to rebuild a saved list in its original order.
        """
        self.asset_counter += 1
        if name is None:
            name = f"Asset {self.asset_counter}"
        if image is None:
            image = Image.new("RGBA", self.size, (0, 0, 0, 0))
        elif image.mode != "RGBA":
            image = image.convert("RGBA")
        asset = Asset(name, image, self.asset_counter, source, prompt, settings)
        if index is None:
            self.assets.insert(0, asset)
        else:
            self.assets.insert(index, asset)
        return asset

    def remove_asset(self, asset):
        """Drop ``asset`` from this canvas's list (a no-op if it is absent)."""
        if asset in self.assets:
            self.assets.remove(asset)

    def snapshot(self, app=None):
        """Capture this canvas's state for undo, sharing pixels copy-on-write.

        Every layer's image and mask is *referenced* rather than copied and
        flagged shared, so the layer forks on its next in-place edit (see
        ``Layer.begin_write``). The app-level shared guide list rides along so
        guide edits stay undoable in shared mode even though it is not stored
        on this canvas.
        """
        layers = []
        for layer in self.layers:
            layer._image_shared = True
            if layer.mask is not None:
                layer._mask_shared = True
            layers.append({
                "name": layer.name,
                "image": layer.image,
                "mask": layer.mask,
                "visible": layer.visible,
                "locked": layer.locked,
                "opacity": layer.opacity,
                "offset": tuple(layer.offset),
                "transform": (
                    tuple(layer.transform) if layer.transform is not None else None
                ),
                "group_id": layer.group.id if layer.group is not None else None,
                "clip": bool(layer.clip),
                "blend": layer.blend,
                # Link Layer identity; the app's relink pass re-points the
                # resolved master after a restore (it may live in another canvas).
                "is_master": bool(layer.is_master),
                "master_id": layer.master_id,
                "link_id": layer.link_id,
                "filters": [filters.clone(spec) for spec in layer.filters],
                "texts": [
                    clone for clone in (textrender.clone(spec) for spec in layer.texts)
                    if clone is not None
                ],
                "vector": bool(layer.vector),
                "strokes": [
                    clone for clone in (vector.clone(spec) for spec in layer.strokes)
                    if clone is not None
                ],
                "fills": [
                    clone for clone in (
                        vector.clone_fill(spec) for spec in layer.fills
                    )
                    if clone is not None
                ],
                "paths": [
                    clone for clone in (path.clone(spec) for spec in layer.paths)
                    if clone is not None
                ],
                "poses": [
                    clone for clone in (
                        poses.clone(spec) for spec in layer.poses
                    )
                    if clone is not None
                ],
                "pose_scene": poses.clone_scene(getattr(layer, "pose_scene", None)),
            })
        if self.inpaint_mask is not None:
            self._inpaint_mask_shared = True
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "active": self.active,
            "layer_counter": self.layer_counter,
            "group_counter": self.group_counter,
            "guides": list(self.guides),
            "inpaint_mask": self.inpaint_mask,
            "groups": [
                {
                    "id": group.id,
                    "name": group.name,
                    "visible": group.visible,
                    "opacity": group.opacity,
                    "collapsed": group.collapsed,
                    "keep_empty": bool(group.keep_empty),
                    "filters": [filters.clone(spec) for spec in group.filters],
                }
                for group in self.groups
            ],
            "assets": [
                {
                    "id": asset.id,
                    "name": asset.name,
                    "source": asset.source,
                    "prompt": asset.prompt,
                    "settings": dict(asset.settings),
                    # Assets are never edited in place, so the snapshot can share
                    # the pixels instead of copying them.
                    "image": asset.image,
                }
                for asset in self.assets
            ],
            "asset_counter": self.asset_counter,
            "layers": layers,
            "guides_shared": bool(getattr(app, "guides_shared", False)),
            "shared_guides": [
                tuple(guide) for guide in getattr(app, "shared_guides", [])
            ],
        }

    def restore(self, snapshot, app=None):
        """Rebuild this canvas's contents in place from a :meth:`snapshot`.

        The canvas object itself (and thus its tab and its history) is kept;
        only its size, layers, groups, guides and masks are replaced, so the
        per-canvas undo stack survives a restore.
        """
        self.name = snapshot["name"]
        self.width = snapshot["width"]
        self.height = snapshot["height"]
        self.groups = []
        self.layers = []
        groups = {}
        for group_entry in snapshot.get("groups", []):
            group = Group(group_entry["name"], group_entry["id"])
            group.visible = group_entry.get("visible", True)
            group.opacity = group_entry.get("opacity", 1.0)
            group.collapsed = group_entry.get("collapsed", False)
            group.keep_empty = bool(group_entry.get("keep_empty", False))
            group.filters = [
                spec
                for spec in (
                    filters.normalize_spec(entry)
                    for entry in group_entry.get("filters", [])
                )
                if spec is not None
            ]
            groups[group.id] = group
            self.groups.append(group)
        for layer_entry in snapshot["layers"]:
            layer = self.add_layer(name=layer_entry["name"])
            # Share the snapshot's pixels (copy-on-write) instead of copying
            # them; the restored layer forks before any in-place edit.
            layer.set_image(layer_entry["image"])
            layer._image_shared = True
            mask = layer_entry.get("mask")
            if mask is not None:
                layer.mask = mask
                layer._mask_shared = True
            layer.visible = layer_entry["visible"]
            layer.locked = layer_entry.get("locked", False)
            layer.opacity = layer_entry["opacity"]
            layer.offset = tuple(layer_entry.get("offset", (0, 0)))
            transform = layer_entry.get("transform")
            layer.transform = tuple(transform) if transform is not None else None
            layer.group = groups.get(layer_entry.get("group_id"))
            layer.clip = bool(layer_entry.get("clip", False))
            layer.blend = normalize_blend(layer_entry.get("blend"))
            layer.is_master = bool(layer_entry.get("is_master", False))
            layer.master_id = layer_entry.get("master_id")
            layer.link_id = layer_entry.get("link_id")
            layer.master = None
            layer.filters = [
                spec
                for spec in (
                    filters.normalize_spec(entry)
                    for entry in layer_entry.get("filters", [])
                )
                if spec is not None
            ]
            layer.texts = [
                spec
                for spec in (
                    textrender.normalize(entry)
                    for entry in layer_entry.get("texts", [])
                )
                if spec is not None
            ]
            layer.vector = bool(layer_entry.get("vector", False))
            layer.strokes = [
                spec
                for spec in (
                    vector.normalize(entry)
                    for entry in layer_entry.get("strokes", [])
                )
                if spec is not None
            ]
            layer.fills = [
                spec
                for spec in (
                    vector.normalize_fill(entry)
                    for entry in layer_entry.get("fills", [])
                )
                if spec is not None
            ]
            layer.paths = [
                spec
                for spec in (
                    path.normalize(entry)
                    for entry in layer_entry.get("paths", [])
                )
                if spec is not None
            ]
            layer.poses, layer.pose_scene = poses.load_layer_poses(
                layer_entry.get("poses", layer_entry.get("mannequins", [])),
                layer_entry.get("pose_scene"),
                layer.size,
            )
        # ``add_layer`` bumps the counter while rebuilding; reset it to the
        # snapshotted value so future layer names stay stable.
        self.layer_counter = snapshot["layer_counter"]
        self.assets = [
            Asset(
                entry["name"], entry["image"], entry["id"],
                entry.get("source", ""), entry.get("prompt", ""),
                entry.get("settings", {}),
            )
            for entry in snapshot.get("assets", [])
        ]
        self.asset_counter = snapshot.get("asset_counter", len(self.assets))
        self.group_counter = snapshot.get("group_counter", len(self.groups))
        self.guides = [tuple(guide) for guide in snapshot.get("guides", [])]
        mask = snapshot.get("inpaint_mask")
        if mask is not None:
            self.inpaint_mask = mask
            self._inpaint_mask_shared = True
        else:
            self.inpaint_mask = None
            self._inpaint_mask_shared = False
        self.active = (
            max(0, min(snapshot["active"], len(self.layers) - 1)) if self.layers else 0
        )
        # The rebuilt layer stack invalidates the region-composite cache.
        self._region_cache_signature = None
        self._region_cache_images = None
        self._last_render = None
        if app is not None:
            app.shared_guides = [
                tuple(guide) for guide in snapshot.get("shared_guides", [])
            ]
            app.guides_shared = bool(snapshot.get("guides_shared", False))

    def copy(self, name=None):
        """Return an independent duplicate of this canvas, layers and all.

        Every layer is deep-copied through :meth:`Layer.copy`, groups are
        cloned so the copy's membership does not point back at the original,
        and the document-level state (guides, inpaint mask, generation prompts
        and settings, and the layer/group counters) is carried over. The result
        is a fully separate document that can be edited without affecting this
        one.
        """
        duplicate = Canvas(name if name is not None else self.name, self.size)
        groups = {}
        for group in self.groups:
            clone = Group(group.name, group.id)
            clone.visible = group.visible
            clone.opacity = group.opacity
            clone.collapsed = group.collapsed
            clone.keep_empty = group.keep_empty
            clone.filters = [filters.clone(spec) for spec in group.filters]
            groups[id(group)] = clone
            duplicate.groups.append(clone)
        layer_copies = {}
        for layer in self.layers:
            layer_copy = layer.copy()
            # Layer.copy shares the group reference; re-point it at the clone.
            layer_copy.group = (
                groups.get(id(layer.group)) if layer.group is not None else None
            )
            layer_copies[id(layer)] = layer_copy
            duplicate.layers.append(layer_copy)
        # A link whose master lives in this canvas must point at the master's
        # copy, not the original; a cross-canvas master is left shared.
        for layer in self.layers:
            if layer.is_link and layer.master is not None:
                layer_copies[id(layer)].master = layer_copies.get(
                    id(layer.master), layer.master
                )
        duplicate.active = self.active
        duplicate.layer_counter = self.layer_counter
        duplicate.group_counter = self.group_counter
        duplicate.guides = [tuple(guide) for guide in self.guides]
        duplicate.gen_settings = dict(self.gen_settings)
        duplicate.assets = [asset.copy() for asset in self.assets]
        duplicate.asset_counter = self.asset_counter
        duplicate.working_profile = self.working_profile
        duplicate.proof_profile = self.proof_profile
        duplicate.proof_intent = self.proof_intent
        if self.inpaint_mask is not None:
            duplicate.inpaint_mask = self.inpaint_mask.copy()
        return duplicate

    def composite(self):
        """Flatten visible layers over white into an RGB image."""
        rgba = self._flatten(self.layers, (0, 0, self.width, self.height))
        base = Image.new("RGBA", self.size, (255, 255, 255, 255))
        base.alpha_composite(rgba)
        return base.convert("RGB")

    def composite_transparent(self):
        """Flatten visible layers into one transparent RGBA image."""
        return self._flatten(self.layers, (0, 0, self.width, self.height))

    def new_group(self, name=None, members=()):
        """Create a group containing ``members`` and return it.

        Every member's :attr:`Layer.group` is set, the group is registered on
        the canvas, and the layer stack is normalised so the members sit next to
        each other.
        """
        self.group_counter += 1
        if name is None:
            name = f"Group {self.group_counter}"
        group = Group(name, self.group_counter)
        for layer in members:
            layer.group = group
        self.groups.append(group)
        self.normalize_groups()
        return group

    def add_group(self, name=None):
        """Create a new empty group and return it.

        Unlike :meth:`new_group`, the group has no members yet, so it is flagged
        to survive :meth:`normalize_groups` until it is removed explicitly.
        """
        self.group_counter += 1
        if name is None:
            name = f"Group {self.group_counter}"
        group = Group(name, self.group_counter)
        group.keep_empty = True
        self.groups.append(group)
        return group

    def remove_group(self, group):
        """Dissolve ``group``: detach its members and drop it from the canvas."""
        for layer in self.layers:
            if layer.group is group:
                layer.group = None
        if group in self.groups:
            self.groups.remove(group)

    def normalize_groups(self):
        """Reorder ``layers`` so each group's members are contiguous.

        Group membership is stored per layer, so operations elsewhere only need
        to set :attr:`Layer.group`; this re-establishes the invariant the
        compositor relies on. Layers keep their relative order within each run,
        and any group whose members vanished is dropped - except an empty group
        created on its own (``keep_empty``), which stays until removed.
        """
        present = {getattr(layer, "group", None) for layer in self.layers}
        self.groups = [
            group for group in self.groups
            if group is not None and (group in present or group.keep_empty)
        ]
        for layer in self.layers:
            if layer.group is not None and layer.group not in self.groups:
                # Member of a group that was removed: promote it to the root.
                layer.group = None
        ordered = []
        emitted = set()
        for layer in self.layers:
            group = layer.group
            if group is None:
                ordered.append(layer)
            elif id(group) not in emitted:
                emitted.add(id(group))
                ordered.extend(group.members(self.layers))
        self.layers = ordered

    def _units(self, layers, plain=False):
        """Group ``layers`` into render units: a layer, or a contiguous group.

        Yields ``("layer", layer, None)`` or ``("group", group, members)`` in
        stack order (bottom-to-top), so the compositor can flatten each unit and
        apply the unit's own opacity/visibility exactly once. ``plain`` treats
        every entry as a bare layer, used when recursing into a group's members
        so their shared membership does not group them again.
        """
        if plain:
            return [("layer", layer, None) for layer in layers]
        units = []
        index = 0
        count = len(layers)
        while index < count:
            layer = layers[index]
            group = getattr(layer, "group", None)
            if group is None:
                units.append(("layer", layer, None))
                index += 1
                continue
            members = []
            while index < count and getattr(layers[index], "group", None) is group:
                members.append(layers[index])
                index += 1
            units.append(("group", group, members))
        return units

    def _flatten(self, layers, region, plain=False, scale=1.0):
        """Composite ``layers`` over ``region`` into a straight-alpha RGBA image.

        This is the document-level flatten: it honours layer groups as units
        (see :meth:`_units`) so a group's own opacity and visibility apply once,
        to the group's combined pixels, rather than to each member separately.
        ``plain`` composites every entry as a bare layer, used when recursing
        into a group's members. ``scale`` is the pixels-per-document-pixel of the
        returned buffer; a zoomed-out view passes ``scale < 1`` so the flatten is
        bounded by the viewport's pixels.
        """
        left, top, right, bottom = (
            int(region[0]), int(region[1]), int(region[2]), int(region[3])
        )
        width = max(0, right - left)
        height = max(0, bottom - top)
        base = Image.new(
            "RGBA", (int(round(width * scale)), int(round(height * scale))),
            (0, 0, 0, 0),
        )
        # Resolve each clipped layer's clip base: the nearest layer below it
        # that is not itself clipped. A clipped layer with no such base (or a
        # hidden base) is fully hidden, matching the usual clipping behaviour.
        clip_bases = {}
        previous = None
        for layer in layers:
            if getattr(layer, "clip", False):
                clip_bases[id(layer)] = previous
            else:
                previous = layer
        for kind, unit, members in self._units(layers, plain):
            if not unit.visible or unit.opacity <= 0.0:
                continue
            if kind == "layer":
                clip = None
                if getattr(unit, "clip", False):
                    clip_base = clip_bases.get(id(unit))
                    if clip_base is None or not clip_base.visible:
                        continue
                    clip = self._layer_alpha_region(
                        clip_base, left, top, width, height, scale
                    )
                _paste_layer(base, unit, left, top, clip, scale)
            else:
                if unit.filters:
                    # A group filter acts on the members' combined pixels, so
                    # flatten them at document resolution and filter there; a
                    # zoomed-out view filters the full-size result and only then
                    # downsamples, so the effect does not change with zoom the
                    # way filtering an already-scaled buffer would.
                    sub = self._flatten(
                        members, (left, top, right, bottom), plain=True
                    )
                    sub = filters.apply_chain(sub, unit.filters)
                    if scale != 1.0:
                        sub = sub.resize(
                            (max(1, int(round(width * scale))),
                             max(1, int(round(height * scale)))),
                            Image.BOX,
                        )
                else:
                    sub = self._flatten(
                        members, (left, top, right, bottom), plain=True, scale=scale
                    )
                if unit.opacity < 1.0:
                    sub = _with_opacity(sub, unit.opacity)
                base.alpha_composite(sub)
        return base

    def _layer_alpha_region(self, layer, left, top, width, height, scale=1.0):
        """A region-sized ``L`` image of ``layer``'s alpha at its own offset."""
        image = layer.composited_image()
        ox, oy = layer.offset
        box = (left - ox, top - oy, left - ox + width, top - oy + height)
        alpha = image.crop(box).getchannel("A")
        if scale != 1.0:
            alpha = alpha.resize(
                (max(1, int(round(width * scale))),
                 max(1, int(round(height * scale)))),
                Image.BOX,
            )
        return alpha

    def content_bounds(self, exclude=None):
        """Union rectangle of the canvas and every visible layer, in doc pixels.

        Returns ``(left, top, right, bottom)``. Moves keep content outside the
        canvas in each layer's own buffer, so this is the region that actually
        holds pixels. ``exclude`` drops one layer index from the union, which the
        below/above backdrop cache uses to anchor itself on the layers that a
        move of the active layer does not touch (see :meth:`_flatten_cached`).
        """
        left, top = 0, 0
        right, bottom = self.size
        for index, layer in enumerate(self.layers):
            if index == exclude:
                continue
            if not layer.visible or layer.opacity <= 0.0:
                continue
            ox, oy = layer.offset
            width, height = layer.size
            left = min(left, ox)
            top = min(top, oy)
            right = max(right, ox + width)
            bottom = max(bottom, oy + height)
        return (left, top, right, bottom)

    def composite_overview(self, offscreen_opacity=0.35, region=None, scale=1.0,
                           active_index=None):
        """Flatten visible layers over a buffer covering all off-canvas content.

        Returns ``(image, origin)`` where ``origin`` is the document coordinate
        of the buffer's top-left corner. Pixels outside the canvas rectangle have
        their alpha scaled by ``offscreen_opacity`` so off-canvas content reads
        as a faint preview while the canvas itself stays fully opaque.

        ``region`` limits the work to a ``(left, top, right, bottom)`` document
        rectangle; ``origin`` is then that rectangle's top-left corner. The view
        passes the visible viewport so a large canvas is not flattened whole.
        ``scale`` is the returned buffer's pixels per document pixel.

        ``active_index`` opts into the same below/above cache as
        :meth:`composite_region`: when the stack has no groups or clipped layers
        and the region repeats (a stroke, a mask edit, a tool preview), only the
        active layer is recomposited over the cached backdrop, instead of every
        layer being flattened again.
        """
        content = self.content_bounds()
        if region is None:
            left, top, right, bottom = content
        else:
            rl, rt, rr, rb = (int(region[0]), int(region[1]),
                              int(region[2]), int(region[3]))
            # Clip the requested rectangle to the content bounds. The view asks
            # for the visible document rectangle, which at low zoom is many times
            # the canvas; flattening that whole rectangle would allocate a
            # document-resolution buffer far larger than anything on the canvas,
            # and cropping it later can exceed Pillow's decompression-bomb limit.
            left = max(content[0], rl)
            top = max(content[1], rt)
            right = min(content[2], rr)
            bottom = min(content[3], rb)
            if right <= left or bottom <= top:
                return Image.new("RGBA", (0, 0), (0, 0, 0, 0)), (rl, rt)
        base = self._flatten_cached(
            (left, top, right, bottom), active_index, scale
        )
        if offscreen_opacity < 1.0 and (left < 0 or top < 0
                                        or right > self.width
                                        or bottom > self.height):
            veil = Image.new("L", base.size, int(round(offscreen_opacity * 255)))
            cl = max(left, 0)
            ct = max(top, 0)
            cr = min(right, self.width)
            cb = min(bottom, self.height)
            if cr > cl and cb > ct:
                veil.paste(
                    Image.new("L", (max(1, int(round((cr - cl) * scale))),
                                     max(1, int(round((cb - ct) * scale)))), 255),
                    (int(round((cl - left) * scale)),
                     int(round((ct - top) * scale))),
                )
            base.putalpha(ImageChops.multiply(base.getchannel("A"), veil))
        return base, (left, top)

    def merged_image(self, layers, region=None):
        """Composite ``layers`` (bottom-to-top) into one transparent RGBA image.

        ``region`` limits the composite to a ``(left, top, right, bottom)``
        document rectangle; the returned buffer then starts at that rectangle's
        top-left corner instead of the canvas origin.
        """
        if region is None:
            left = top = 0
            size = self.size
        else:
            left, top, right, bottom = region
            size = (max(0, right - left), max(0, bottom - top))
        base = Image.new("RGBA", size, (0, 0, 0, 0))
        for layer in layers:
            if not layer.visible or layer.opacity <= 0.0:
                continue
            _paste_layer(base, layer, left, top)
        return base

    def composite_region(self, region, active_index=None, scale=1.0):
        """Flatten visible layers over the part of ``region`` inside the canvas.

        Returns ``(image, origin)`` for the view's viewport-only rendering; the
        cost scales with the visible rectangle, not the whole canvas. Off-canvas
        pixels are excluded, matching :meth:`composite_transparent`. ``scale`` is
        the returned buffer's pixels per document pixel, so a zoomed-out view
        composites the viewport's pixels rather than the document rectangle it
        covers.

        When ``active_index`` is given and the stack has no groups or clipped
        layers, the layers below and above the active one are cached and reused,
        so a stroke (which only changes the active layer) recomposites a single
        layer over the cached backdrop rather than every layer.
        """
        rl, rt, rr, rb = (int(region[0]), int(region[1]),
                          int(region[2]), int(region[3]))
        left = max(0, rl)
        top = max(0, rt)
        right = min(self.width, rr)
        bottom = min(self.height, rb)
        if right <= left or bottom <= top:
            # The region misses the canvas entirely. Hand back an empty buffer at
            # the requested origin rather than a region-sized blank, which at low
            # zoom can dwarf the canvas (and trip Pillow's bomb guard downstream).
            return Image.new("RGBA", (0, 0), (0, 0, 0, 0)), (rl, rt)
        bounds = (left, top, right, bottom)
        return self._flatten_cached(bounds, active_index, scale), (left, top)

    def composite_layout(self, layout, region, scale=1.0):
        """Flatten the layer stack laid out in an exploded grid ``layout``.

        Unlike :meth:`composite_region`, each layer is placed at its layout cell
        (not its document offset) and clipped to that cell, so the layers sit
        side by side instead of on top of one another. Grouping, clipping and
        blend modes are ignored in this view: it is a contact-sheet
        representation, one cell per layer. ``region`` is a
        ``(left, top, right, bottom)`` rectangle in layout space and ``scale``
        its pixels per layout pixel (a zoomed-out view passes a value below 1).
        Returns ``(image, origin)`` where ``origin`` is the region's top-left.
        """
        left, top, right, bottom = (
            int(region[0]), int(region[1]), int(region[2]), int(region[3])
        )
        width = max(0, right - left)
        height = max(0, bottom - top)
        base = Image.new(
            "RGBA", (int(round(width * scale)), int(round(height * scale))),
            (0, 0, 0, 0),
        )
        for index, layer in enumerate(self.layers):
            if index >= len(layout.cells):
                break
            # Visibility is deliberately ignored here: the exploded view always
            # shows every layer so a hidden one can still be found and edited.
            if layer.opacity <= 0.0:
                continue
            cx0, cy0, cx1, cy1 = layout.cells[index]
            il = max(cx0, left)
            it = max(cy0, top)
            ir = min(cx1, right)
            ib = min(cy1, bottom)
            if ir <= il or ib <= it:
                continue
            ox, oy = layout.offsets[index]
            sub = layer.composited_region(
                (il - ox, it - oy, ir - ox, ib - oy)
            )
            if sub.mode != "RGBA":
                sub = sub.convert("RGBA")
            if layer.opacity < 1.0:
                sub = _with_opacity(sub, layer.opacity)
            dest = (int(round((il - left) * scale)), int(round((it - top) * scale)))
            if scale != 1.0:
                sub = sub.resize(
                    (max(1, int(round(sub.width * scale))),
                     max(1, int(round(sub.height * scale)))),
                    Image.BOX,
                )
            base.alpha_composite(sub, dest=dest)
        return base, (left, top)

    def _flatten_cached(self, bounds, active_index, scale):
        """Flatten the layer stack over ``bounds``, reusing the below/above cache.

        Shared by :meth:`composite_region` and :meth:`composite_overview` so both
        the on-canvas and the off-canvas preview paths benefit from the cache.

        When ``active_index`` is given and the stack has no groups or clipped
        layers, the layers below the active one are flattened once over a fixed,
        document-anchored region (the content bounds) and cached. A view
        transform - zooming or panning - only changes ``bounds`` and ``scale``,
        never the layers, so the cached backdrop is cropped and resampled rather
        than re-composited: a zoom step costs a couple of crops and a resize, so
        it stays fast however many layers the document holds. The active layer is
        still composited live, so a stroke (which changes only the active layer)
        recomposites one layer over the cached backdrop.

        The layers above the active one are cached as one flattened image only
        when none of them blends (a flattened-away backdrop would leave a blend
        mode with nothing to mix into); otherwise they are composited live so
        their blend modes still see the pixels beneath them.

        The backdrop is promoted from the cheap whole-stack path only once the
        same layer stack is rendered again *after a change* - a zoom or pan, or an
        edit to the active layer - so a one-off render (a tab switch, a lone
        repaint) never pays for a content-sized cache it would not reuse.

        A canvas whose content exceeds :data:`MAX_BACKDROP_PIXELS` skips the
        cache and flattens the whole stack per call, since the cached buffers are
        content-sized.
        """
        rl, rt, rr, rb = (
            int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
        )
        cacheable = (
            active_index is not None
            and 0 <= active_index < len(self.layers)
            and not any(getattr(layer, "group", None) is not None
                        for layer in self.layers)
            and not any(getattr(layer, "clip", False) for layer in self.layers)
        )
        # Anchor the backdrop on every visible layer except the active one. The
        # active layer is composited live over the crop, and a move only changes
        # its offset, so excluding it keeps the cache (and its origin) stable
        # through a whole drag instead of rebuilding - and re-anchoring - it on
        # every event.
        cache_bounds = (
            self.content_bounds(exclude=active_index) if cacheable else None
        )
        if cache_bounds is not None:
            width = max(0, cache_bounds[2] - cache_bounds[0])
            height = max(0, cache_bounds[3] - cache_bounds[1])
            if width * height > MAX_BACKDROP_PIXELS:
                cache_bounds = None
        if cache_bounds is None:
            self._region_cache_signature = None
            self._region_cache_images = None
            self._last_render = None
            return self._flatten(self.layers, bounds, scale=scale)
        above_layers = self.layers[active_index + 1:]
        # A flattened "above" image cannot carry blend modes, since it is
        # composited over the live backdrop rather than the pixels it was
        # flattened with; only cache it when every above layer is a plain over.
        above_cacheable = all(
            getattr(layer, "blend", "normal") == "normal"
            for layer in above_layers
        )
        # Key only the layers around the active one: the active layer is
        # recomposited live on every call, so its own edits must not evict the
        # cached backdrop (that is what makes a stroke cheap).
        signature = (
            active_index,
            tuple(
                self._layer_signature(layer)
                for layer in self.layers[:active_index]
            ),
            tuple(
                self._layer_signature(layer)
                for layer in above_layers
            ),
        )
        active_layer = self.layers[active_index]
        current = (
            signature, (rl, rt, rr, rb), scale,
            self._layer_signature(active_layer),
        )
        last = self._last_render
        if (self._region_cache_images is not None
                and self._region_cache_signature == signature):
            below, above = self._region_cache_images
        elif (last is not None and last[0] == signature
                and last[1:] != current[1:]):
            # The same layer stack as the previous render, but the view or the
            # active layer changed since: this is a transform gesture (zoom or
            # pan) or a stroke, so promote to the content-sized backdrop.
            # Building it costs more than one viewport-sized flatten, so an
            # unchanged or one-off render stays on the cheap whole-stack path and
            # never pays for a cache it will not reuse.
            below = self._flatten(self.layers[:active_index], cache_bounds)
            above = (self._flatten(above_layers, cache_bounds)
                     if above_cacheable else None)
            self._region_cache_images = (below, above)
            self._region_cache_signature = signature
        else:
            self._last_render = current
            self._region_cache_signature = None
            self._region_cache_images = None
            # Render this one-off frame the way the cached path does - composite
            # at document resolution, then resample once - instead of downscaling
            # each layer separately. At ``scale < 1`` the two differ, so a drag
            # that promotes to the cache mid-gesture would switch the whole frame
            # (a zoomed-out flicker). Matching the scaling keeps it seamless.
            full = self._flatten(self.layers, bounds, scale=1.0)
            if scale != 1.0:
                full = full.resize(
                    (max(1, int(round((rr - rl) * scale))),
                     max(1, int(round((rb - rt) * scale)))),
                    Image.BOX,
                )
            return full
        self._last_render = current
        # Crop the fixed-resolution backdrop to the requested region, composite
        # the active layer at document resolution, then resample once. Cropping a
        # zoomed-out view's large region and resizing the combined result is
        # cheaper than downscaling every layer separately, and sharpens the
        # zoomed-in case (composite at full resolution, then resample).
        cl, ct = int(cache_bounds[0]), int(cache_bounds[1])
        box = (rl - cl, rt - ct, rr - cl, rb - ct)
        base = below.crop(box)
        if active_layer.visible and active_layer.opacity > 0.0:
            _paste_layer(base, active_layer, rl, rt)
        if above is not None:
            base.alpha_composite(above.crop(box))
        else:
            # Blend layers above the active one need the backdrop, so paste them
            # in stack order instead of compositing a flattened image.
            for layer in above_layers:
                if layer.visible and layer.opacity > 0.0:
                    _paste_layer(base, layer, rl, rt)
        if scale != 1.0:
            base = base.resize(
                (max(1, int(round((rr - rl) * scale))),
                 max(1, int(round((rb - rt) * scale)))),
                Image.BOX,
            )
        return base

    @staticmethod
    def _layer_signature(layer):
        """A cheap key for every input that changes a layer's composite.

        A link keys on its master's content tokens (via ``content_source``) so a
        change to the master invalidates every canvas that shows a link to it.
        """
        content = layer.content_source()
        return (
            id(layer),
            content._image_token,
            filters.chain_key(content.filters),
            content._text_token,
            content._mask_token,
            tuple(layer.offset),
            None if layer.transform is None else tuple(layer.transform),
            layer.opacity,
            layer.visible,
            layer.blend,
        )

    def bake_offsets(self):
        """Fold every layer's offset into its pixels, clipping at the canvas.

        Moves keep a layer's content by only shifting :attr:`Layer.offset`; this
        is called before an edit that works in document coordinates so those
        operations keep seeing a canvas-aligned image. A mask, which shares the
        layer's pixel space, is shifted by the same offset.
        """
        for layer in self.layers:
            if layer.offset == (0, 0):
                continue
            # A link's pixels belong to its master, so only its offset moves.
            if layer.is_link:
                continue
            aligned = Image.new("RGBA", self.size, (0, 0, 0, 0))
            aligned.alpha_composite(layer.image, dest=layer.offset)
            layer.set_image(aligned)
            if layer.mask is not None:
                aligned_mask = Image.new("L", self.size, 0)
                aligned_mask.paste(layer.mask, layer.offset)
                layer.mask = aligned_mask
            layer.offset = (0, 0)

    def align_for_painting(self, layer, bounds=None):
        """Grow ``layer``'s buffer so painting can reach the whole canvas.

        Strokes map document coordinates into the layer's own buffer via
        :attr:`Layer.offset`, so a layer moved off the canvas would otherwise
        clip paint at its buffer edge. Unlike :meth:`bake_offsets`, the buffer
        is expanded to hold both the canvas rectangle and the layer's existing
        pixels, keeping off-canvas content, and ``offset`` is shifted to the
        buffer's new origin.

        ``bounds`` is an optional ``(left, top, right, bottom)`` document-space
        rectangle that must also fit, so a one-shot primitive drawn past the
        canvas edge (the Shape tool) is not clipped at that edge.
        """
        # A link layer has no pixels of its own to grow (it is read-only).
        if layer.is_link:
            return
        width, height = self.size
        ox, oy = layer.offset
        left = min(0, ox)
        top = min(0, oy)
        right = max(width, ox + layer.image.width)
        bottom = max(height, oy + layer.image.height)
        if bounds is not None:
            left = min(left, math.floor(bounds[0]))
            top = min(top, math.floor(bounds[1]))
            right = max(right, math.ceil(bounds[2]))
            bottom = max(bottom, math.ceil(bounds[3]))
        # Nothing to do when the buffer already covers the target rectangle at
        # its current origin.
        if (left == ox and top == oy
                and right - left == layer.image.width
                and bottom - top == layer.image.height):
            return
        size = (right - left, bottom - top)
        aligned = Image.new("RGBA", size, (0, 0, 0, 0))
        aligned.alpha_composite(layer.image, dest=(ox - left, oy - top))
        layer.set_image(aligned)
        if layer.mask is not None:
            aligned_mask = Image.new("L", size, 0)
            aligned_mask.paste(layer.mask, (ox - left, oy - top))
            layer.mask = aligned_mask
        layer.offset = (left, top)

    def fit_image(self, image):
        """Scale ``image`` to fit this canvas, centered on a transparent buffer."""
        width, height = self.size
        img = image.convert("RGBA")
        iw, ih = img.size
        scale = min(width / iw, height / ih)
        nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
        resized = img.resize((nw, nh), Image.LANCZOS)
        out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        out.paste(resized, ((width - nw) // 2, (height - nh) // 2), resized)
        return out
