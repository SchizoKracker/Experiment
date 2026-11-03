"""Signed-distance-field sculpting toolkit.

Shapes are described as a tree-less ordered list of primitives combined with
smooth boolean operators.  Evaluation is vectorised over (N, 3) point arrays
and culls primitives whose (blend-expanded) bounds do not touch the query box,
which keeps dense evaluation of 80+ primitive bodies cheap.

Conventions: metres, Z up, characters face -Y (Blender front view).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

F = np.float32


# --------------------------------------------------------------------------
# smooth operators (polynomial, iq)
# --------------------------------------------------------------------------
def smin(a, b, k):
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


# --------------------------------------------------------------------------
# rotation helpers
# --------------------------------------------------------------------------
def frame_z(direction, up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """3x3 matrix whose rows are local axes; local Z aligned to `direction`."""
    z = np.asarray(direction, dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1e-12)
    up = np.asarray(up_hint, dtype=np.float64)
    if abs(np.dot(up, z)) > 0.95:
        up = np.array([0.0, -1.0, 0.0]) if abs(z[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return np.stack([x, y, z]).astype(F)


def euler(rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    """Rotation matrix (rows = local axes in world) from XYZ euler degrees."""
    rx, ry, rz = (math.radians(v) for v in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    world_from_local = rzm @ rym @ rxm
    return world_from_local.T.astype(F)  # rows: local axes expressed in world


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
@dataclass
class Prim:
    fn: Callable[[np.ndarray], np.ndarray]
    lo: np.ndarray
    hi: np.ndarray
    op: str = "union"        # union | sub | inter
    k: float = 0.0           # blend radius
    tag: str = ""            # region label for vertex masks
    name: str = ""
    tag_eps: float = 1e9     # only claim vertices within this distance of the primitive
    side: str = ""           # "", "L" or "R": left/right parts blend with the centre, never each other

    def touches(self, lo, hi) -> bool:
        pad = self.k + 1e-4
        return bool(np.all(self.lo - pad <= hi) and np.all(self.hi + pad >= lo))


def _bounds(center, ext):
    c = np.asarray(center, dtype=F)
    e = np.asarray(ext, dtype=F)
    return c - e, c + e


def sphere(c, r):
    c = np.asarray(c, F)

    def f(p):
        return np.linalg.norm(p - c, axis=1) - r

    return f, *_bounds(c, [r, r, r])


def ellipsoid(c, radii, rot: Optional[np.ndarray] = None):
    c = np.asarray(c, F)
    r = np.asarray(radii, F)
    R = rot

    def f(p):
        q = p - c
        if R is not None:
            q = q @ R.T
        k0 = np.linalg.norm(q / r, axis=1)
        k1 = np.linalg.norm(q / (r * r), axis=1)
        return k0 * (k0 - 1.0) / (k1 + 1e-9)

    m = float(r.max())
    return f, *_bounds(c, [m, m, m])


def round_cone(a, b, ra, rb):
    """Capsule with different end radii (exact SDF, iq)."""
    a = np.asarray(a, F)
    b = np.asarray(b, F)
    ba = b - a
    l2 = float(np.dot(ba, ba))
    rr = ra - rb
    a2 = l2 - rr * rr
    il2 = 1.0 / max(l2, 1e-12)

    def f(p):
        pa = p - a
        y = pa @ ba
        z = y - l2
        x = pa * l2 - np.outer(y, ba)
        x2 = np.einsum("ij,ij->i", x, x)
        y2 = y * y * l2
        z2 = z * z * l2
        k = math.copysign(1.0, rr) * rr * rr * x2
        out = np.empty(len(p), dtype=F)
        c1 = np.sign(z) * a2 * z2 > k
        c2 = (~c1) & (np.sign(y) * a2 * y2 < k)
        c3 = ~(c1 | c2)
        out[c1] = np.sqrt(x2[c1] + z2[c1]) * il2 - rb
        out[c2] = np.sqrt(x2[c2] + y2[c2]) * il2 - ra
        out[c3] = (np.sqrt(x2[c3] * a2 * il2) + y[c3] * rr) * il2 - ra
        return out

    lo = np.minimum(a - ra, b - rb)
    hi = np.maximum(a + ra, b + rb)
    return f, lo.astype(F), hi.astype(F)


def capsule(a, b, r):
    return round_cone(a, b, r, r)


def round_box(c, half, rot: Optional[np.ndarray] = None, radius=0.0):
    c = np.asarray(c, F)
    h = np.asarray(half, F) - radius
    R = rot

    def f(p):
        q = p - c
        if R is not None:
            q = q @ R.T
        q = np.abs(q) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside - radius

    m = float(np.linalg.norm(np.asarray(half)))
    return f, *_bounds(c, [m, m, m])


def torus(c, R_major, r_minor, rot: Optional[np.ndarray] = None):
    c = np.asarray(c, F)
    Rm = rot

    def f(p):
        q = p - c
        if Rm is not None:
            q = q @ Rm.T
        xy = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) - R_major
        return np.sqrt(xy * xy + q[:, 2] ** 2) - r_minor

    m = R_major + r_minor
    return f, *_bounds(c, [m, m, m])


def bent_cone(points, radii):
    """Poly-line of round cones -> list of (fn, lo, hi) (horns, tails, hair)."""
    out = []
    for i in range(len(points) - 1):
        out.append(round_cone(points[i], points[i + 1], radii[i], radii[i + 1]))
    return out


def half_space(point, normal):
    """Signed distance to plane; negative on the side opposite `normal`."""
    p0 = np.asarray(point, F)
    n = np.asarray(normal, F)
    n = n / np.linalg.norm(n)

    def f(p):
        return (p - p0) @ n

    big = np.full(3, 50.0, F)
    return f, -big, big


# --------------------------------------------------------------------------
# procedural noise (vectorised value noise + fbm + worley)
# --------------------------------------------------------------------------
def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(F) / F(0xFFFFFF)


def value_noise(p, freq=1.0, seed=0):
    q = p * freq
    i = np.floor(q).astype(np.int64)
    f = q - i
    u = f * f * (3.0 - 2.0 * f)
    res = np.zeros(len(p), F)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1]) * (u[:, 2] if dz else 1 - u[:, 2])
                res += w * _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return res * 2.0 - 1.0


def fbm(p, freq=1.0, octaves=4, seed=0, gain=0.5):
    total = np.zeros(len(p), F)
    amp = 1.0
    norm = 0.0
    for o in range(octaves):
        total += amp * value_noise(p, freq * (2.0 ** o), seed + o * 17)
        norm += amp
        amp *= gain
    return total / norm


def worley(p, freq=1.0, seed=0):
    """Returns (F1, F2) distances to nearest feature points (cell scales)."""
    q = p * freq
    i = np.floor(q).astype(np.int64)
    f1 = np.full(len(p), 9.0, F)
    f2 = np.full(len(p), 9.0, F)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cx, cy, cz = i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz
                fp = np.stack([
                    cx + _hash3(cx, cy, cz, seed),
                    cy + _hash3(cx, cy, cz, seed + 1),
                    cz + _hash3(cx, cy, cz, seed + 2)], axis=1)
                d = np.linalg.norm(fp - q, axis=1)
                f2 = np.where(d < f1, f1, np.minimum(f2, d))
                f1 = np.minimum(f1, d)
    return f1 / freq, f2 / freq


# --------------------------------------------------------------------------
# shape: ordered primitive stack + optional displacement
# --------------------------------------------------------------------------
@dataclass
class Shape:
    prims: list = field(default_factory=list)
    displace: list = field(default_factory=list)   # callables p, d -> delta
    side: str = ""                                  # side assigned to newly added prims

    def add(self, prim_tuple, k=0.0, tag="", op="union", name="", tag_eps=1e9):
        fn, lo, hi = prim_tuple
        self.prims.append(Prim(fn, np.asarray(lo, F), np.asarray(hi, F), op, k, tag, name, tag_eps, self.side))
        return self

    def union(self, prim_tuple, k=0.0, tag="", name="", tag_eps=1e9):
        return self.add(prim_tuple, k, tag, "union", name, tag_eps)

    def sub(self, prim_tuple, k=0.0, tag="", name=""):
        return self.add(prim_tuple, k, tag, "sub", name)

    def inter(self, prim_tuple, k=0.0, name=""):
        return self.add(prim_tuple, k, "", "inter", name)

    def bounds(self):
        us = [p for p in self.prims if p.op == "union"]
        lo = np.min([p.lo - p.k for p in us], axis=0)
        hi = np.max([p.hi + p.k for p in us], axis=0)
        return lo, hi

    def eval(self, p: np.ndarray, lo=None, hi=None, with_tags=False):
        p = np.asarray(p, F)
        if lo is None:
            lo, hi = p.min(axis=0), p.max(axis=0)
        tag_d = None
        tags = None
        if with_tags:
            tag_d = np.full(len(p), 10.0, F)
            tags = np.full(len(p), -1, np.int32)

        def fold(d, side):
            nonlocal tag_d, tags
            for idx, pr in enumerate(self.prims):
                if pr.side != side:
                    continue
                if pr.op != "inter" and not pr.touches(lo, hi):
                    continue
                v = pr.fn(p)
                if pr.op == "union":
                    d = smin(d, v, pr.k)
                    if with_tags and pr.tag:
                        closer = (v < tag_d) & (v < pr.tag_eps)
                        tag_d = np.where(closer, v, tag_d)
                        tags = np.where(closer, idx, tags)
                elif pr.op == "sub":
                    d = smax(d, -v, pr.k)
                else:
                    d = smax(d, v, pr.k)
            return d

        d = fold(np.full(len(p), 10.0, F), "")
        has_sides = any(pr.side for pr in self.prims)
        if has_sides:
            d = np.minimum(fold(d.copy(), "L"), fold(d.copy(), "R"))
        for disp in self.displace:
            d = d + disp(p, d)
        if with_tags:
            return d, tags
        return d

    def tag_names(self, tag_idx: np.ndarray) -> np.ndarray:
        names = np.array([p.tag for p in self.prims] + [""], dtype=object)
        return names[np.where(tag_idx < 0, len(self.prims), tag_idx)]


# --------------------------------------------------------------------------
# swept profile ("loft") primitive: one continuous surface per body part
# --------------------------------------------------------------------------
def smooth_profile(keys, n=96):
    """keys: [(t, value), ...] -> dense Catmull-Rom resample on t in [0,1]."""
    keys = sorted(keys)
    ts = np.array([k[0] for k in keys], np.float64)
    vs = np.array([k[1] for k in keys], np.float64)
    out_t = np.linspace(0.0, 1.0, n)
    out = np.empty(n)
    for i, t in enumerate(out_t):
        j = np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2)
        t0, t1 = ts[j], ts[j + 1]
        u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        p0 = vs[max(j - 1, 0)]
        p1, p2 = vs[j], vs[j + 1]
        p3 = vs[min(j + 2, len(vs) - 1)]
        # tangents scaled for non-uniform spacing
        m1 = (p2 - p0) / max(ts[min(j + 1, len(ts) - 1)] - ts[max(j - 1, 0)], 1e-6) * (t1 - t0)
        m2 = (p3 - p1) / max(ts[min(j + 2, len(ts) - 1)] - ts[j], 1e-6) * (t1 - t0)
        u2, u3 = u * u, u * u * u
        out[i] = (2 * u3 - 3 * u2 + 1) * p1 + (u3 - 2 * u2 + u) * m1 + (-2 * u3 + 3 * u2) * p2 + (u3 - u2) * m2
    return out.astype(F)


def _interp(table, t):
    n = len(table)
    x = np.clip(t, 0.0, 1.0) * (n - 1)
    i = np.clip(x.astype(np.int64), 0, n - 2)
    f = x - i
    return table[i] * (1 - f) + table[i + 1] * f


def loft(a, b, w_out, w_in, front, back, outer_dir=(1, 0, 0), front_dir=(0, -1, 0),
         n_front=2.0, n_back=2.0, front_shift=None, cap_a=True, cap_b=True, cap_round=0.7,
         ext_a=0.0, ext_b=0.0):
    """Generalised cylinder from a to b.

    Profiles (keyframe lists [(t, v)] or dense arrays) give the cross-section's
    half-extents toward the outer side, inner side, front and back.  Quadrants
    of a superellipse join with matching tangents, so the surface is smooth.
    """
    a = np.asarray(a, F)
    b = np.asarray(b, F)
    L = float(np.linalg.norm(b - a))
    ax = (b - a) / L
    fr = np.asarray(front_dir, F)
    fr = fr - ax * np.dot(fr, ax)
    fr /= np.linalg.norm(fr)
    sd = np.cross(ax, fr)
    od = np.asarray(outer_dir, F)
    osign = 1.0 if np.dot(sd, od) >= 0 else -1.0
    prof = {}
    for key, val in (("wo", w_out), ("wi", w_in), ("f", front), ("b", back)):
        prof[key] = smooth_profile(val) if isinstance(val, (list, tuple)) else np.asarray(val, F)
    nf = smooth_profile(n_front) if isinstance(n_front, (list, tuple)) else None
    nb = smooth_profile(n_back) if isinstance(n_back, (list, tuple)) else None
    fs = smooth_profile(front_shift) if isinstance(front_shift, (list, tuple)) else None

    def f(p):
        q = p - a
        tl = q @ ax
        t = tl / L
        u = (q @ sd) * osign
        v = q @ fr
        tc = np.clip(t, 0, 1)
        if fs is not None:
            v = v - _interp(fs, tc)
        rx = np.where(u > 0, _interp(prof["wo"], tc), _interp(prof["wi"], tc))
        ry = np.where(v > 0, _interp(prof["f"], tc), _interp(prof["b"], tc))
        rx = np.maximum(rx, 1e-4)
        ry = np.maximum(ry, 1e-4)
        if nf is None and nb is None and n_front == 2.0 and n_back == 2.0:
            n = np.full(len(p), 2.0, F)
        else:
            nfv = _interp(nf, tc) if nf is not None else np.full(len(p), n_front, F)
            nbv = _interp(nb, tc) if nb is not None else np.full(len(p), n_back, F)
            n = np.where(v > 0, nfv, nbv)
        au = np.abs(u) / rx + 1e-9
        av = np.abs(v) / ry + 1e-9
        s = au ** n + av ** n
        Fv = s ** (1.0 / n)
        # gradient of Fv w.r.t (u, v) for distance normalisation
        gcoef = s ** (1.0 / n - 1.0)
        gu = gcoef * au ** (n - 1) / rx
        gv = gcoef * av ** (n - 1) / ry
        g = np.sqrt(gu * gu + gv * gv) + 1e-9
        d2 = (Fv - 1.0) / g
        # rounded caps: shrink-then-inflate (rounded cylinder) with radius tied to the end section
        rr = cap_round * np.minimum(rx, ry)
        e = np.full_like(t, -1e3)
        if cap_a:
            e = np.maximum(e, -tl - ext_a)
        if cap_b:
            e = np.maximum(e, tl - L - ext_b)
        a1 = d2 + rr
        b1 = e + rr
        out = np.sqrt(np.maximum(a1, 0) ** 2 + np.maximum(b1, 0) ** 2) + np.minimum(np.maximum(a1, b1), 0) - rr
        return out.astype(F)

    rmax = float(max(prof["wo"].max(), prof["wi"].max(), prof["f"].max(), prof["b"].max()))
    if fs is not None:
        rmax += float(np.abs(fs).max())
    a_e = a - ax * ext_a
    b_e = b + ax * ext_b
    lo = np.minimum(a_e, b_e) - rmax
    hi = np.maximum(a_e, b_e) + rmax
    return f, lo.astype(F), hi.astype(F)
