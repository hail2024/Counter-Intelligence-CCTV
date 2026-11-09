"""Counter geometry: signed side of a polyline boundary, polygons, and coordinate scaling.

The counter boundary is a polyline drawn on the camera image. For a point we compute a *signed*
distance to it: the magnitude is the distance to the nearest point of the polyline and the sign
says which side the point is on. The first and last segments are extended to infinity so that
every image point has a well-defined side (a wrist slightly past the end of a drawn line is still
classified). At an interior vertex the sign uses the averaged normal of the two adjacent segments,
which avoids sign flips around concave corners.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

Point = Sequence[float]


def _as_array(points) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {arr.shape}")
    return arr


def _unit_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    n = np.array([-d[1], d[0]])  # left-hand normal of a->b (image coords: y down)
    norm = np.hypot(*n)
    if norm == 0:
        raise ValueError("boundary contains a zero-length segment")
    return n / norm


def signed_distance_to_polyline(p: Point, polyline) -> float:
    """Signed distance from ``p`` to the polyline (positive on the left-hand normal side).

    The polyline's first and last segments are treated as infinite rays beyond their end points.
    """
    pts = _as_array(polyline)
    if len(pts) < 2:
        raise ValueError("a boundary needs at least two points")
    p = np.asarray(p, dtype=float)
    n_seg = len(pts) - 1
    best = None  # (dist, seg_index, t)
    for i in range(n_seg):
        a, b = pts[i], pts[i + 1]
        d = b - a
        L2 = float(d @ d)
        if L2 == 0:
            raise ValueError("boundary contains a zero-length segment")
        t = float((p - a) @ d) / L2
        lo = -np.inf if i == 0 else 0.0
        hi = np.inf if i == n_seg - 1 else 1.0
        t_c = min(max(t, lo), hi)
        closest = a + t_c * d
        dist = float(np.hypot(*(p - closest)))
        if best is None or dist < best[0] - 1e-9:
            best = (dist, i, t_c)
    dist, i, t_c = best
    a, b = pts[i], pts[i + 1]
    if t_c <= 0.0 and i > 0:  # nearest point is the vertex shared with the previous segment
        normal = _unit_normal(pts[i - 1], a) + _unit_normal(a, b)
        anchor = a
    elif t_c >= 1.0 and i < n_seg - 1:  # vertex shared with the next segment
        normal = _unit_normal(a, b) + _unit_normal(b, pts[i + 2])
        anchor = b
    else:
        normal = _unit_normal(a, b)
        anchor = a + t_c * (b - a)
    side = float((p - anchor) @ normal)
    if dist == 0.0 or side == 0.0:
        return 0.0
    return dist if side > 0 else -dist


def point_in_polygon(p: Point, polygon) -> bool:
    """Even-odd ray casting test. Points exactly on an edge may fall either way."""
    poly = _as_array(polygon)
    x, y = float(p[0]), float(p[1])
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def polygon_area(polygon) -> float:
    poly = _as_array(polygon)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(polygon) -> np.ndarray:
    poly = _as_array(polygon)
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return poly.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * a)
    return np.array([cx, cy])


def scale_points(points, from_size: tuple[int, int], to_size: tuple[int, int]) -> list[list[float]]:
    """Map points between two resolutions of the same view (sizes are (width, height))."""
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    return [[float(x) * sx, float(y) * sy] for x, y in points]


def clip_polygon_halfplane(polygon, a: Point, b: Point, keep_left: bool) -> list[list[float]]:
    """Sutherland-Hodgman clip of ``polygon`` by the infinite line a->b.

    Keeps the left-hand-normal side (same sign convention as signed_distance_to_polyline)
    when ``keep_left`` is True, otherwise the right side.
    """
    a = np.asarray(a, float)
    n = _unit_normal(a, np.asarray(b, float))
    sgn = 1.0 if keep_left else -1.0

    def side(p):
        return sgn * float((np.asarray(p) - a) @ n)

    poly = [np.asarray(p, float) for p in polygon]
    out: list[np.ndarray] = []
    for i in range(len(poly)):
        cur, nxt = poly[i], poly[(i + 1) % len(poly)]
        sc, sn = side(cur), side(nxt)
        if sc >= 0:
            out.append(cur)
        if (sc >= 0) != (sn >= 0):
            t = sc / (sc - sn)
            out.append(cur + t * (nxt - cur))
    return [[float(p[0]), float(p[1])] for p in out]


def frame_rect(width: int, height: int) -> list[list[float]]:
    return [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]


def segment_closest(a0: Point, a1: Point, b0: Point, b1: Point) -> tuple[float, float, float]:
    """Shortest distance between segments a0-a1 and b0-b1, and WHERE it occurs on each: (distance, s, t) with s, t in
    [0, 1] measured from a0 and b0 (for hand segments wrist -> fingertip: 1 = at the fingertip)."""
    a0, a1, b0, b1 = (np.asarray(v, float) for v in (a0, a1, b0, b1))
    da, db = a1 - a0, b1 - b0

    def proj(p, s0, d):
        L2 = float(d @ d)
        return 0.0 if L2 < 1e-12 else float(np.clip((p - s0) @ d / L2, 0.0, 1.0))

    den = float(da[0] * db[1] - da[1] * db[0])
    if abs(den) > 1e-12:   # proper intersection?
        w = b0 - a0
        s = float(w[0] * db[1] - w[1] * db[0]) / den
        t = float(w[0] * da[1] - w[1] * da[0]) / den
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            return 0.0, s, t
    best = None
    for s_fix, t_fix in ((0.0, None), (1.0, None), (None, 0.0), (None, 1.0)):
        if s_fix is not None:
            p = a0 + s_fix * da
            t = proj(p, b0, db)
            s = s_fix
        else:
            q = b0 + t_fix * db
            s = proj(q, a0, da)
            t = t_fix
        d = float(np.hypot(*((a0 + s * da) - (b0 + t * db))))
        if best is None or d < best[0]:
            best = (d, s, t)
    return best


def segment_distance(a0: Point, a1: Point, b0: Point, b1: Point) -> float:
    """Shortest distance between segments a0-a1 and b0-b1 (0 when they intersect)."""
    return segment_closest(a0, a1, b0, b1)[0]


def distance(p: Point, q: Point) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))
