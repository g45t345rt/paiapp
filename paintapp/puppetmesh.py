"""As-Rigid-As-Possible triangle-mesh deformation for the puppet warp.

A puppet warp deforms a drawing like a rubber sheet pinned at a few points. The
robust way to do that (Photoshop's, and Clip Studio's) is to cover the drawing
with a triangular mesh, pin some of its vertices, and solve for the rest of the
vertices with the *As-Rigid-As-Possible* energy: every triangle tries to keep its
shape (translate and rotate, but not stretch or shear), so the mesh bends
smoothly instead of folding over itself and duplicating pixels.

This module implements the pieces directly on numpy (no scipy/opencv):

* :func:`triangular_mesh` - a regular grid triangulated into a mesh,
* :func:`content_mesh` - that grid clipped to rendered/selected pixels,
* :class:`ARAP` - the precomputed mesh plus the local/global solver,
* :func:`warp_triangles` - resamples an image by mapping each triangle from its
  rest position to its deformed position.

The solver follows Sorkine & Alexa, "As-Rigid-As-Possible Surface Modeling"
(2007) with uniform edge weights.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay


def triangular_mesh(width, height, delta):
    """A regular triangular mesh over a ``width`` x ``height`` rectangle.

    Returns ``(vertices, faces)`` in local pixel coordinates: a ``(cols + 1) x
    (rows + 1)`` grid of vertices (``delta`` apart) with each grid cell split
    into two triangles, like the mesh Clip Studio's puppet warp lays over an
    image.
    """
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    delta = max(4.0, float(delta))
    cols = max(1, int(round(width / delta)))
    rows = max(1, int(round(height / delta)))
    xs = np.linspace(0.0, width, cols + 1)
    ys = np.linspace(0.0, height, rows + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    vertices = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    faces = []
    for row in range(rows):
        for col in range(cols):
            top_left = row * (cols + 1) + col
            top_right = top_left + 1
            bottom_left = top_left + (cols + 1)
            bottom_right = bottom_left + 1
            faces.append((top_left, top_right, bottom_left))
            faces.append((top_right, bottom_right, bottom_left))
    return vertices, np.asarray(faces, dtype=np.int64)


def _in_circumcircle(a, b, c, p):
    """Whether point ``p`` lies inside the circumcircle of triangle ``abc``."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    px, py = p
    if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0.0:
        bx, by, cx, cy = cx, cy, bx, by
    adx, ady = ax - px, ay - py
    bdx, bdy = bx - px, by - py
    cdx, cdy = cx - px, cy - py
    abdet = adx * bdy - bdx * ady
    bcdet = bdx * cdy - cdx * bdy
    cadet = cdx * ady - adx * cdy
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    # Points on a circumcircle belong to the cavity too. Excluding them leaves
    # holes in regular grids and along densely sampled pixel outlines.
    return alift * bcdet + blift * cadet + clift * abdet >= -1e-9


def delaunay(points):
    """Dependency-free Bowyer-Watson triangulation of a small point set."""
    count = len(points)
    if count < 3:
        return []
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    span = max(max_x - min_x, max_y - min_y) or 1.0
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    verts = list(points) + [
        (mid_x - 20 * span, mid_y - span),
        (mid_x, mid_y + 20 * span),
        (mid_x + 20 * span, mid_y - span),
    ]
    triangles = [(count, count + 1, count + 2)]
    for i in range(count):
        point = verts[i]
        bad = [
            t for t in triangles
            if _in_circumcircle(verts[t[0]], verts[t[1]], verts[t[2]], point)
        ]
        if not bad:
            continue
        edges = {}
        for tri in bad:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (a, b) if a < b else (b, a)
                edges[key] = edges.get(key, 0) + 1
        bad_set = set(bad)
        triangles = [t for t in triangles if t not in bad_set]
        for a, b in edges:
            if edges[(a, b)] == 1:
                triangles.append((i, a, b))
    return [t for t in triangles if max(t) < count]


def _triangle_has_content(tri, mask):
    """Whether a rest triangle covers at least one opaque source pixel."""
    height, width = mask.shape
    x0 = max(0, int(np.floor(tri[:, 0].min())))
    x1 = min(width - 1, int(np.ceil(tri[:, 0].max())))
    y0 = max(0, int(np.floor(tri[:, 1].min())))
    y1 = min(height - 1, int(np.ceil(tri[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return False
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    den = v0[0] * v1[1] - v0[1] * v1[0]
    if abs(den) < 1e-12:
        return False
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    px = xs - tri[0, 0]
    py = ys - tri[0, 1]
    l1 = (px * v1[1] - py * v1[0]) / den
    l2 = (v0[0] * py - v0[1] * px) / den
    inside = (l1 >= -1e-7) & (l2 >= -1e-7) & (l1 + l2 <= 1.0 + 1e-7)
    return bool((inside & mask[y0:y1 + 1, x0:x1 + 1]).any())


def content_mesh(alpha, delta, extra_points=()):
    """Delaunay mesh over a silhouette, with optional exact pin positions.

    A regular interior sampling supplies well-conditioned deformation cells;
    opaque pixel-edge samples pull the outer mesh onto thin limbs, hair and
    clothing. Empty triangles are removed, while every triangle touching ink is
    retained. ``extra_points`` inserts clicked pin locations as real vertices so
    an arm pin never snaps to a nearby torso or background grid vertex.
    """
    alpha = np.asarray(alpha)
    if alpha.ndim != 2:
        raise ValueError("content_mesh expects a two-dimensional alpha mask")
    height, width = alpha.shape
    if not height or not width or not np.any(alpha > 0.01):
        return triangular_mesh(width, height, delta)
    mask = alpha > 0.01
    delta = max(4.0, float(delta))
    xs = np.unique(np.concatenate((np.arange(0.0, width, delta), [float(width)])))
    ys = np.unique(np.concatenate((np.arange(0.0, height, delta), [float(height)])))
    gx, gy = np.meshgrid(xs, ys)
    points = list(map(tuple, np.stack((gx.ravel(), gy.ravel()), axis=1)))

    padded = np.pad(mask, 1)
    vertical = np.nonzero(padded[1:-1, 1:-1] != padded[1:-1, :-2])
    horizontal = np.nonzero(padded[1:-1, 1:-1] != padded[:-2, 1:-1])
    vertical_count = len(vertical[0])
    stride = max(1, int(np.ceil((vertical_count + len(horizontal[0])) / 1200.0)))
    for y, x in zip(vertical[0][::stride], vertical[1][::stride]):
        points.extend(((int(x), int(y)), (int(x), int(y + 1))))
    start = (-vertical_count) % stride
    for y, x in zip(horizontal[0][start::stride], horizontal[1][start::stride]):
        points.extend(((int(x), int(y)), (int(x + 1), int(y))))
    points.extend((float(x), float(y)) for x, y in extra_points)
    points = list(dict.fromkeys(points))
    if len(points) < 3:
        return triangular_mesh(width, height, delta)

    vertices = np.asarray(points, dtype=np.float64)
    faces = Delaunay(vertices).simplices.astype(np.int64)
    # Cheaply reject the many triangles whose bounding box holds no content at
    # all (using a summed-area table of the mask) before the exact, per-pixel
    # inside test. On a sparse silhouette this skips a large share of faces.
    triangles = vertices[faces]
    low_x = np.clip(np.floor(triangles[:, :, 0].min(axis=1)).astype(np.int64),
                    0, width - 1)
    high_x = np.clip(np.ceil(triangles[:, :, 0].max(axis=1)).astype(np.int64),
                     0, width - 1)
    low_y = np.clip(np.floor(triangles[:, :, 1].min(axis=1)).astype(np.int64),
                    0, height - 1)
    high_y = np.clip(np.ceil(triangles[:, :, 1].max(axis=1)).astype(np.int64),
                     0, height - 1)
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0), axis=1)
    box_sum = (integral[high_y + 1, high_x + 1] - integral[low_y, high_x + 1]
               - integral[high_y + 1, low_x] + integral[low_y, low_x])
    kept = [
        faces[index] for index in np.nonzero(box_sum > 0)[0]
        if _triangle_has_content(vertices[faces[index]], mask)
    ]
    if not kept:
        return triangular_mesh(width, height, delta)
    faces = np.asarray(kept, dtype=np.int64)
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def uniform_edges(faces):
    """The unique undirected ``(i, j)`` edge list of a triangulation."""
    edges = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add((i, j) if i < j else (j, i))
    return np.asarray(sorted(edges), dtype=np.int64)


class ARAP:
    """Solves an ARAP deformation of a fixed triangular mesh.

    The mesh topology (edges, adjacency) is fixed, so it is precomputed once;
    each :meth:`deformation` call solves for new vertex positions given the
    vertices that a pin has fixed (the control points) and their target
    positions.
    """

    def __init__(self, vertices, faces):
        self.rest = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.n = len(self.rest)
        self.edges = uniform_edges(self.faces)
        self.e0 = self.edges[:, 0]
        self.e1 = self.edges[:, 1]
        degree = np.zeros(self.n, dtype=np.float64)
        np.add.at(degree, self.e0, 1.0)
        np.add.at(degree, self.e1, 1.0)
        self.degree = np.maximum(degree, 1.0)

    def _rotations(self, current):
        """The best-fit local rotation at each vertex (the ARAP local step)."""
        rest = self.rest
        e = rest[self.e0] - rest[self.e1]
        d = current[self.e0] - current[self.e1]
        # For covariance S = sum(e * d.T), the closest rotation mapping rest
        # edges onto current edges is V * U.T, not U * V.T. The latter is its
        # inverse and makes a clockwise pin drag rotate the drawing backwards.
        outer = e[:, :, None] * d[:, None, :]
        cov = np.zeros((self.n, 2, 2), dtype=np.float64)
        np.add.at(cov, self.e0, outer)
        np.add.at(cov, self.e1, outer)
        u, _s, vt = np.linalg.svd(cov)
        r = np.swapaxes(vt, 1, 2) @ np.swapaxes(u, 1, 2)
        det = r[:, 0, 0] * r[:, 1, 1] - r[:, 0, 1] * r[:, 1, 0]
        flip = det < 0.0
        if flip.any():
            v = np.swapaxes(vt, 1, 2).copy()
            v[flip, :, 1] *= -1.0
            r = v @ np.swapaxes(u, 1, 2)
        # A vertex with no real edges (or a degenerate covariance) stays still.
        return r

    def _laplacian(self, vector):
        """Apply the uniform Laplacian to a full-length ``(n, 2)`` vector."""
        neighbour = np.zeros_like(vector)
        np.add.at(neighbour, self.e0, vector[self.e1])
        np.add.at(neighbour, self.e1, vector[self.e0])
        return self.degree[:, None] * vector - neighbour

    def _apply_free(self, y, free):
        full = np.zeros((self.n, 2), dtype=np.float64)
        full[free] = y
        return self._laplacian(full)[free]

    def _conjugate_gradient(self, rhs, free, x0, iterations, tol):
        """Solve ``L_ff x = rhs`` (SPD) by Jacobi-preconditioned CG."""
        inv = 1.0 / self.degree[free][:, None]
        x = x0
        residual = rhs - self._apply_free(x, free)
        z = residual * inv
        direction = z.copy()
        rz = float(np.sum(residual * z))
        rhs_norm = float(np.sum(rhs * rhs)) + 1e-12
        for _ in range(iterations):
            ap = self._apply_free(direction, free)
            denom = float(np.sum(direction * ap))
            if denom <= 1e-20:
                break
            alpha = rz / denom
            x = x + alpha * direction
            residual = residual - alpha * ap
            if float(np.sum(residual * residual)) <= tol * rhs_norm:
                break
            z = residual * inv
            rz_new = float(np.sum(residual * z))
            if rz <= 1e-20:
                break
            direction = z + (rz_new / rz) * direction
            rz = rz_new
        return x

    def deformation(self, constrained, targets, iterations=8, init=None,
                    cg_iterations=80, tol=1e-6):
        """New vertex positions with ``constrained`` pinned to ``targets``.

        ``constrained`` is an index array and ``targets`` the matching ``(k, 2)``
        positions. The local/global iterations alternate best-fit rotations with
        a Laplacian solve so the free vertices follow smoothly and rigidly.
        """
        constrained = np.asarray(constrained, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
        free_mask = np.ones(self.n, dtype=bool)
        free_mask[constrained] = False
        free = np.nonzero(free_mask)[0]
        if free.size == 0:
            current = self.rest.copy()
            current[constrained] = targets
            return current

        pinned = np.zeros((self.n, 2), dtype=np.float64)
        pinned[constrained] = targets
        neighbour_target = np.zeros((self.n, 2), dtype=np.float64)
        np.add.at(neighbour_target, self.e0, pinned[self.e1])
        np.add.at(neighbour_target, self.e1, pinned[self.e0])

        current = self.rest.copy() if init is None else np.asarray(init, float).copy()
        current[constrained] = targets

        for _ in range(max(1, iterations)):
            rotations = self._rotations(current)
            blended = 0.5 * (rotations[self.e0] + rotations[self.e1])
            rest_edge = self.rest[self.e0] - self.rest[self.e1]
            rotated = np.einsum("mij,mj->mi", blended, rest_edge)
            rhs_full = np.zeros((self.n, 2), dtype=np.float64)
            np.add.at(rhs_full, self.e0, rotated)
            np.add.at(rhs_full, self.e1, -rotated)
            rhs = rhs_full[free] + neighbour_target[free]
            current[free] = self._conjugate_gradient(
                rhs, free, current[free], cg_iterations, tol)
            current[constrained] = targets
        return current


class TwoStepARAP:
    """Igarashi-style ARAP with per-edge similarity transformations.

    Unlike a vertex-Laplacian approximation, the first solve estimates a local
    similarity frame from the three or four vertices around every edge. The
    second solve removes its scale and fits globally rotated edge vectors. Pin
    positions are soft constraints with a high weight, as in the reference
    algorithm; no artificial bounding-box anchors are needed.
    """

    def __init__(self, vertices, faces):
        self.rest = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.n = len(self.rest)
        self.edges = uniform_edges(self.faces)
        opposites = {}
        for a, b, c in self.faces:
            for i, j, other in ((a, b, c), (b, c, a), (c, a, b)):
                key = (int(min(i, j)), int(max(i, j)))
                opposites.setdefault(key, []).append(int(other))

        count_edges = len(self.edges)
        # The three or four vertices around every edge: its two ends plus the
        # one or two opposite triangle corners.
        self.gi = np.full((count_edges, 4), -1, dtype=np.int64)
        self.gi[:, :2] = self.edges
        for k in range(count_edges):
            i, j = int(self.edges[k, 0]), int(self.edges[k, 1])
            opposite = opposites.get((min(i, j), max(i, j)), ())
            for slot, other in enumerate(opposite[:2]):
                self.gi[k, 2 + slot] = other

        # Build the per-edge least-squares frames and the sparse systems for all
        # edges at once. Filling a lil_matrix element by element here used to
        # dominate pin placement (tens of thousands of scipy setitem calls);
        # vectorised construction is roughly an order of magnitude faster.
        valid = self.gi >= 0
        coords = self.rest[np.maximum(self.gi, 0)]
        x = np.where(valid, coords[..., 0], 0.0)
        y = np.where(valid, coords[..., 1], 0.0)
        ones = valid.astype(np.float64)
        g = np.zeros((count_edges, 8, 4), dtype=np.float64)
        g[:, 0::2, 0] = x
        g[:, 0::2, 1] = y
        g[:, 0::2, 2] = ones
        g[:, 1::2, 0] = y
        g[:, 1::2, 1] = -x
        g[:, 1::2, 3] = ones
        gt = np.swapaxes(g, 1, 2)
        product = np.linalg.pinv(gt @ g) @ gt
        self.g_product = product[:, :2, :]

        self.rest_edges = (
            self.rest[self.edges[:, 1]] - self.rest[self.edges[:, 0]])
        edge_frame = np.stack(
            (self.rest_edges, self.rest_edges[:, ::-1] * (1.0, -1.0)), axis=1)
        oz = np.zeros((1, 2, 8), dtype=np.float64)
        oz[0, 0, 0] = oz[0, 1, 1] = -1.0
        oz[0, 0, 2] = oz[0, 1, 3] = 1.0
        h = oz - edge_frame @ self.g_product

        edge_index = np.repeat(
            np.arange(count_edges)[:, None], 4, axis=1)
        neighbour = self.gi
        selected = neighbour >= 0
        row0 = (2 * edge_index)[selected]
        col0 = (2 * neighbour)[selected]
        first = h[:, 0, :].reshape(count_edges, 4, 2)[selected]
        second = h[:, 1, :].reshape(count_edges, 4, 2)[selected]
        a1 = sparse.coo_matrix((
            np.concatenate((first[:, 0], first[:, 1],
                            second[:, 0], second[:, 1])),
            (np.concatenate((row0, row0, row0 + 1, row0 + 1)),
             np.concatenate((col0, col0 + 1, col0, col0 + 1))),
        ), shape=(count_edges * 2, self.n * 2)).tocsr()

        rows2 = np.repeat(np.arange(count_edges), 2)
        data2 = np.empty(count_edges * 2, dtype=np.float64)
        data2[0::2] = -1.0
        data2[1::2] = 1.0
        a2 = sparse.coo_matrix(
            (data2, (rows2, self.edges.reshape(-1))),
            shape=(count_edges, self.n)).tocsr()

        self.a1 = a1
        self.a1_normal = (self.a1.T @ self.a1).tocsr()
        self.a2 = a2
        self.a2_normal = (self.a2.T @ self.a2).tocsr()
        self._cache_key = None
        self._solver1 = None
        self._solver2 = None

    def _prepare_constraints(self, constrained, weight):
        key = constrained.tobytes()
        if key == self._cache_key:
            return
        diagonal1 = np.zeros(self.n * 2, dtype=np.float64)
        diagonal = constrained * 2
        diagonal1[diagonal] = weight * weight
        diagonal1[diagonal + 1] = weight * weight
        diagonal2 = np.zeros(self.n, dtype=np.float64)
        diagonal2[constrained] = weight * weight
        matrix1 = self.a1_normal + sparse.diags(diagonal1, format="csr")
        matrix2 = self.a2_normal + sparse.diags(diagonal2, format="csr")
        self._solver1 = sparse.linalg.splu(matrix1.tocsc())
        self._solver2 = sparse.linalg.splu(matrix2.tocsc())
        self._cache_key = key

    def _solve(self, constrained, targets, weight):
        diagonal = constrained * 2
        rhs = np.zeros(self.n * 2, dtype=np.float64)
        rhs[diagonal] = weight * weight * targets[:, 0]
        rhs[diagonal + 1] = weight * weight * targets[:, 1]
        return self._solver1.solve(rhs).reshape(-1, 2)

    def deformation(self, constrained, targets, iterations=1, init=None,
                    cg_iterations=None, tol=None):
        """Solve both ARAP steps for control vertex indices and target positions."""
        constrained = np.asarray(constrained, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
        if not len(constrained):
            return self.rest.copy()
        weight = 1000.0
        self._prepare_constraints(constrained, weight)
        prime = self._solve(constrained, targets, weight)

        # Boundary edges have a missing fourth neighbour; never gather -1 as
        # the last vertex, even though its product columns are normally zero.
        vectors = prime[np.maximum(self.gi, 0)]
        vectors[self.gi < 0] = 0.0
        t = np.einsum("eij,ej->ei", self.g_product, vectors.reshape(-1, 8))
        norm = np.hypot(t[:, 0], t[:, 1])
        degenerate = norm < 1e-12
        t /= np.where(degenerate, 1.0, norm)[:, None]
        t[degenerate] = (1.0, 0.0)
        rotations = np.empty((len(self.edges), 2, 2), dtype=np.float64)
        rotations[:, 0, 0] = rotations[:, 1, 1] = t[:, 0]
        rotations[:, 0, 1] = t[:, 1]
        rotations[:, 1, 0] = -t[:, 1]
        rotated = np.einsum("eij,ej->ei", rotations, self.rest_edges)

        rhs = self.a2.T @ rotated
        rhs[constrained] += weight * weight * targets
        return self._solver2.solve(rhs)


def _sample_bilinear(source, x, y):
    """Bilinear sample of an ``(H, W, C)`` float array at float ``(x, y)``."""
    height, width = source.shape[:2]
    # Affine inversion can put an exact border sample just outside the image.
    epsilon = 1e-7
    valid = ((x >= -epsilon) & (x <= width - 1.0 + epsilon) &
             (y >= -epsilon) & (y <= height - 1.0 + epsilon))
    x = np.clip(x, 0.0, width - 1.0)
    y = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    top = source[y0, x0] * (1 - fx) + source[y0, x1] * fx
    bottom = source[y1, x0] * (1 - fx) + source[y1, x1] * fx
    result = top * (1 - fy) + bottom * fy
    return result * valid[:, None]


def warp_triangles(source, rest, deformed, faces, margin=0, priorities=None,
                   bounds=None):
    """Resample ``source`` from the rest mesh onto the deformed mesh.

    ``source`` is a premultiplied float ``(H, W, 4)`` array; ``rest``/``deformed``
    are local-pixel vertex positions and ``faces`` the triangle indices. The
    result is a premultiplied float array of size ``(H + 2m, W + 2m)`` (``m =
    margin``); output pixel ``(x, y)`` corresponds to local coordinate
    ``(x - m, y - m)``. Each triangle is filled by inverting its deformation
    (a single affine map per triangle), so the transfer is exact. When triangles
    overlap, ``priorities`` controls their stacking: larger values are rendered
    last using premultiplied source-over. Shared mesh edges and vertices are
    sampled only once, while outer boundaries remain inclusive.

    Optional ``bounds=(left, top, right, bottom)`` overrides ``margin`` with an
    integer, half-open destination rectangle in local source pixel coordinates.
    The result then has shape ``(bottom - top, right - left, 4)`` and output
    pixel ``(x, y)`` maps to local coordinate ``(x + left, y + top)``.
    """
    height, width = source.shape[:2]
    if bounds is None:
        margin = int(margin)
        left, top, right, bottom = -margin, -margin, width + margin, height + margin
    else:
        left, top, right, bottom = (int(value) for value in bounds)
        if (left, top, right, bottom) != tuple(bounds):
            raise ValueError("destination bounds must be integer pixel coordinates")
    if right < left or bottom < top:
        raise ValueError("destination bounds must have nonnegative dimensions")
    out = np.zeros((bottom - top, right - left, source.shape[2]), dtype=np.float32)
    if priorities is None:
        priorities = np.zeros(len(faces), dtype=np.float64)
    if len(priorities) != len(faces):
        raise ValueError("one overlap priority is required per triangle")
    if not len(faces) or not out.size:
        return out
    # Stable face order is also the order of candidates within every batch.
    faces = np.asarray(faces, dtype=np.int64)[np.argsort(priorities, kind="stable")]
    dst = np.asarray(deformed, dtype=np.float64)[faces]
    v0, v1 = dst[:, 1] - dst[:, 0], dst[:, 2] - dst[:, 0]
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    active = np.abs(den) >= 1e-9
    faces, dst, v0, v1, den = (array[active] for array in (faces, dst, v0, v1, den))
    if not len(faces):
        return out
    src = np.asarray(rest)[faces]
    ranks = np.arange(len(faces))

    # Topological ownership keeps outer boundaries inclusive. Separate vertex
    # owners prevent holes where different shared-edge owners meet.
    edges = np.sort(faces[:, ((1, 2), (2, 0), (0, 1))], axis=2)
    keys = edges[:, :, 0] * len(deformed) + edges[:, :, 1]
    _, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    edge_owner = np.full(inverse.max() + 1, -1, dtype=np.int64)
    np.maximum.at(edge_owner, inverse, np.repeat(ranks, 3))
    owns_edge = edge_owner[inverse].reshape(-1, 3) == ranks[:, None]
    vertex_owner = np.full(len(deformed), -1, dtype=np.int64)
    np.maximum.at(vertex_owner, faces.ravel(), np.repeat(ranks, 3))
    owns_vertex = vertex_owner[faces] == ranks[:, None]

    origin = np.array((left, top))
    lower = np.maximum(np.floor(dst.min(axis=1)) - origin, 0).astype(np.int64)
    upper = np.minimum(np.ceil(dst.max(axis=1)) - origin,
                       (out.shape[1] - 1, out.shape[0] - 1)).astype(np.int64)
    sizes = np.maximum(upper - lower + 1, 0)
    counts = sizes[:, 0] * sizes[:, 1]
    ends = np.cumsum(counts)
    starts = ends - counts
    flat_out = out.reshape(-1, source.shape[2])

    # Flatten the varying bounding rectangles without allocating a grid for
    # each face. Even one enormous triangle is split into bounded batches.
    for start in range(0, int(ends[-1]), 131072):
        candidate = np.arange(start, min(start + 131072, int(ends[-1])))
        face = np.searchsorted(ends, candidate, side="right")
        offset = candidate - starts[face]
        x = lower[face, 0] + offset % sizes[face, 0]
        y = lower[face, 1] + offset // sizes[face, 0]
        px = x + left - dst[face, 0, 0]
        py = y + top - dst[face, 0, 1]
        l1 = (px * v1[face, 1] - py * v1[face, 0]) / den[face]
        l2 = (v0[face, 0] * py - v0[face, 1] * px) / den[face]
        l0 = 1.0 - l1 - l2
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        on_edge = (np.abs(l0) <= 1e-6, np.abs(l1) <= 1e-6,
                   np.abs(l2) <= 1e-6)
        at_vertex = ((on_edge[0] & on_edge[1]) |
                     (on_edge[1] & on_edge[2]) |
                     (on_edge[2] & on_edge[0]))
        for index in range(3):
            inside &= ~on_edge[index] | at_vertex | owns_edge[face, index]
            inside &= (~(on_edge[(index + 1) % 3] & on_edge[(index + 2) % 3]) |
                       owns_vertex[face, index])
        if not inside.any():
            continue
        face = face[inside]
        l0, l1, l2 = l0[inside], l1[inside], l2[inside]
        src_x = l0 * src[face, 0, 0] + l1 * src[face, 1, 0] + l2 * src[face, 2, 0]
        src_y = l0 * src[face, 0, 1] + l1 * src[face, 1, 1] + l2 * src[face, 2, 1]
        sampled = _sample_bilinear(source, src_x, src_y)
        pixels = y[inside] * out.shape[1] + x[inside]

        # Group equal destination pixels stably, then composite one sample per
        # pixel at each overlap depth. No repeated advanced-index writes race,
        # and foreground transparency cannot erase lower-priority triangles.
        order = np.argsort(pixels, kind="stable")
        pixels, sampled = pixels[order], sampled[order]
        first = np.r_[0, np.flatnonzero(pixels[1:] != pixels[:-1]) + 1]
        lengths = np.diff(np.r_[first, len(pixels)])
        for depth in range(int(lengths.max())):
            indices = first[lengths > depth] + depth
            target = pixels[indices]
            values = sampled[indices]
            flat_out[target] = values + flat_out[target] * (1.0 - values[:, 3:4])
    return out
