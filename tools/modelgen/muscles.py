"""Surface anatomy: every muscle/bone landmark as sculpt-brush strokes.

Each stroke is placed by sphere-tracing onto the real skin surface of the base
lofts, so landmarks sit exactly on the body no matter the race/proportions.
Amplitudes are fractions of height H, scaled by muscle definition (md).
"""
from __future__ import annotations

import numpy as np

import sdf
from humanoid import v3
from sdf import F, Shape


def _n(v):
    v = np.asarray(v, F)
    return v / (np.linalg.norm(v) + 1e-9)


class Painter:
    def __init__(self, s: Shape, H: float):
        self.base = Shape(prims=s.prims, displace=[])
        self.D = s.displace
        self.H = H

    def surf(self, origin, direction):
        """Sphere-trace from outside toward `origin` along -direction; return skin point."""
        d = _n(direction)
        p = np.asarray(origin, F) + d * 0.2 * self.H
        for _ in range(200):
            dist = float(self.base.eval(p[None, :])[0])
            if abs(dist) < 2e-4:
                break
            step = float(np.clip(dist * 0.9, -0.01 * self.H, 0.02 * self.H))
            p = p - d * step
        return p

    def bump(self, origin, direction, along, r_along, r_across, amp, r_normal=None):
        """Buildup (amp>0) / hollow (amp<0) on the skin; ellipse elongated along `along`."""
        n = _n(direction)
        c = self.surf(origin, n)
        a = _n(np.asarray(along, F) - n * np.dot(along, n))
        b = np.cross(n, a)
        R = np.stack([a, b, n])
        rn = r_normal if r_normal is not None else max(r_along, r_across) * 0.9
        self.D.append(sdf.brush(c, [r_along, r_across, rn], amp, R))
        return c

    def line(self, o1, d1, o2, d2, radius, amp):
        a = self.surf(o1, d1)
        b = self.surf(o2, d2)
        self.D.append(sdf.stroke(a, b, radius, amp))


def apply(s: Shape, p, J):
    H = p["height"]
    fem, mus = p["fem"], p["muscle"]
    md = (0.5 + 0.9 * mus) * (1 - 0.3 * fem)
    P = Painter(s, H)
    h = H
    _torso_front(P, p, h, md, fem)
    _torso_back(P, p, h, md, fem)
    _pelvis(P, p, h, md, fem)
    for side, sx in (("L", 1.0), ("R", -1.0)):
        _arm(P, J, side, sx, h, md)
        _leg(P, J, side, sx, h, md)


# ------------------------------------------------------------------ torso --
def _torso_front(P, p, H, md, fem):
    FWD = v3(0, -1, 0)
    UP = v3(0, 0, 1)
    for sx in (-1, 1):
        X = v3(sx, 0, 0)
        if fem < 0.5:
            # pectoralis major: high, broad fan converging on the armpit
            P.bump(v3(sx * 0.05 * H, 0, 0.738 * H), FWD, X, 0.044 * H, 0.03 * H, 0.0078 * H * md)
            P.bump(v3(sx * 0.072 * H, 0, 0.727 * H), FWD + X * 0.3, X, 0.026 * H, 0.022 * H, 0.0042 * H * md)
            # lower pec border: diagonal from sternum up to the armpit
            P.line(v3(sx * 0.03 * H, 0, 0.703 * H), FWD, v3(sx * 0.094 * H, 0, 0.738 * H), FWD + X * 0.5, 0.0075 * H, -0.0026 * H * md)
            # anterior axillary fold
            P.line(v3(sx * 0.088 * H, 0, 0.742 * H), FWD + X * 0.6, v3(sx * 0.112 * H, 0, 0.752 * H), FWD + X * 0.8, 0.009 * H, 0.003 * H * md)
            # clavicular head (upper pec) fills the chest up to the collarbone
            P.bump(v3(sx * 0.05 * H, 0, 0.772 * H), FWD + UP * 0.25, X, 0.04 * H, 0.018 * H, 0.0042 * H * md)
            # clavicular / sternal head separation: soft line sternum -> armpit
            P.line(v3(sx * 0.016 * H, 0, 0.758 * H), FWD, v3(sx * 0.085 * H, 0, 0.752 * H), FWD + X * 0.5, 0.006 * H, -0.0007 * H * md)
            # sternal border of each pec
            P.line(v3(sx * 0.011 * H, 0, 0.79 * H), FWD, v3(sx * 0.013 * H, 0, 0.705 * H), FWD, 0.0055 * H, -0.0011 * H * md)
            # deltopectoral groove: clavicle -> armpit, separating pec from front deltoid
            P.line(v3(sx * 0.078 * H, 0, 0.805 * H), FWD + X * 0.4 + UP * 0.3, v3(sx * 0.104 * H, 0, 0.752 * H), FWD + X * 0.8, 0.0055 * H, -0.0022 * H * md)
            # nipple / areola hint
            P.bump(v3(sx * 0.062 * H, 0, 0.716 * H), FWD, X, 0.006 * H, 0.006 * H, 0.0009 * H, 0.004 * H)
        # sternoclavicular head of the clavicle and the acromion at the top of the shoulder
        P.bump(v3(sx * 0.017 * H, 0, 0.823 * H), FWD + UP * 0.4, X, 0.007 * H, 0.006 * H, 0.0013 * H)
        P.bump(v3(sx * 0.1 * H, 0, 0.818 * H), UP + X * 0.3, X, 0.009 * H, 0.008 * H, 0.0011 * H)
        # clavicle: a defined ridge (gentle S: sternal end -> slight rise -> acromion), the supraclavicular hollow just
        # above/behind it, so the chest plane, the collarbone and the trapezius slope read as separate layers
        cl = [(0.018, 0.8235), (0.04, 0.8255), (0.062, 0.826), (0.082, 0.823), (0.097, 0.82)]
        for (x0, z0), (x1, z1) in zip(cl[:-1], cl[1:]):
            P.line(v3(sx * x0 * H, 0, z0 * H), FWD, v3(sx * x1 * H, 0, z1 * H), FWD, 0.0045 * H, 0.0024 * H)
        sc = [(0.03, 0.8335), (0.052, 0.8355), (0.074, 0.832)]
        for (x0, z0), (x1, z1) in zip(sc[:-1], sc[1:]):
            P.line(v3(sx * x0 * H, 0, z0 * H), FWD, v3(sx * x1 * H, 0, z1 * H), FWD, 0.0075 * H, -0.0026 * H)
        P.line(v3(sx * 0.035 * H, 0, 0.797 * H), FWD, v3(sx * 0.075 * H, 0, 0.792 * H), FWD, 0.008 * H, -0.0011 * H)
        # costal arch: lower ribcage edge, ridge then soft hollow beneath (drives the V)
        P.line(v3(sx * 0.014 * H, 0, 0.688 * H), FWD, v3(sx * 0.086 * H, 0, 0.632 * H), FWD + X * 0.8, 0.007 * H, 0.0011 * H * md)
        P.line(v3(sx * 0.03 * H, 0, 0.665 * H), FWD, v3(sx * 0.08 * H, 0, 0.615 * H), FWD + X * 0.8, 0.008 * H, -0.0009 * H * md)
        # external obliques: fleshy flank above the iliac crest
        P.bump(v3(sx * 0.082 * H, 0, 0.575 * H), FWD * 0.7 + X, UP, 0.032 * H, 0.02 * H, 0.0032 * H * md * (1 - 0.7 * fem))
        # serratus anterior: interdigitations on the ribs below the armpit
        for i, z in enumerate((0.708, 0.684, 0.66)):
            P.bump(v3(sx * 0.096 * H, -0.01 * H * i, z * H), FWD * (0.5 + 0.2 * i) + X, FWD + UP * 0.4, 0.011 * H, 0.0065 * H, 0.0015 * H * md)
    # sternum valley
    P.line(v3(0, 0, 0.765 * H), FWD, v3(0, 0, 0.712 * H), FWD, 0.012 * H, -0.0009 * H * md)
    # rectus abdominis: segmented pads, linea alba, tendinous intersections
    a = md * (1.0 - 0.6 * fem)
    for sx in (-1, 1):
        for z, rz in ((0.668, 0.016), (0.634, 0.016), (0.6, 0.017), (0.553, 0.03)):
            P.bump(v3(sx * 0.021 * H, 0, z * H), FWD, UP, rz * H, 0.016 * H, (0.0022 if z > 0.56 else 0.0016) * H * a)
    P.line(v3(0, 0, 0.69 * H), FWD, v3(0, 0, 0.53 * H), FWD, 0.0042 * H, -0.0012 * H * a)
    for z in (0.651, 0.617, 0.582):
        P.line(v3(-0.034 * H, 0, z * H), FWD, v3(0.034 * H, 0, z * H), FWD, 0.0042 * H, -0.0008 * H * a)
    # navel
    P.bump(v3(0, 0, 0.585 * H), FWD, UP, 0.0045 * H, 0.0035 * H, -0.0045 * H, 0.004 * H)


def trapezius_field(H, md, fem):
    """The trapezius as one continuous soft sheet (a displacement field), not separate strokes:
    origin along the midline from the occiput (z ~0.90 H) down to T12 (~0.66 H); insertion along the
    lateral clavicle, acromion and spine of the scapula.  Lateral extent L(z) and thickness T(z) follow
    the muscle: thickest along the upper slope (neck -> acromion), thin over the mid back, a thin
    lower diamond; feathered edges.  Weighted to the back of the body; over the top of the shoulders
    it wraps to the front border of the upper trapezius."""
    ZL = np.array([0.66, 0.70, 0.74, 0.78, 0.80, 0.815, 0.83, 0.845, 0.86, 0.88, 0.90])
    LL = np.array([0.0, 0.025, 0.048, 0.07, 0.085, 0.095, 0.092, 0.06, 0.036, 0.03, 0.022])
    TT = np.array([0.0, 0.0012, 0.0018, 0.0024, 0.0034, 0.0046, 0.0056, 0.005, 0.003, 0.0015, 0.0])
    a = md * (1 - 0.2 * fem)
    lo = np.array([-0.11 * H, -0.06 * H, 0.64 * H], np.float32)
    hi = np.array([0.11 * H, 0.14 * H, 0.92 * H], np.float32)

    def f(q, d):
        out = np.zeros(len(q), np.float32)
        m = np.all((q > lo) & (q < hi), axis=1)
        if not np.any(m):
            return out
        x, y, z = np.abs(q[m, 0]) / H, q[m, 1] / H, q[m, 2] / H
        L = np.interp(z, ZL, LL)
        T = np.interp(z, ZL, TT)
        u = np.clip(x / np.maximum(L, 1e-4), 0, 1.5)
        prof = np.clip(1 - u ** 2.2, 0, 1) ** 1.6                    # full near the spine/neck, feathered to the edge
        # back of the body only; the upper trapezius wraps over the top of the shoulder to its front border
        y0 = np.interp(z, [0.78, 0.82, 0.86], [0.012, 0.007, 0.003])    # on the back/top only: it sits behind the clavicle
        wb = np.clip((y - y0) / 0.02 + 0.5, 0, 1)
        wb = wb * wb * (3 - 2 * wb)
        out[m] = -(T * a * prof * wb * H)
        return out
    f.lo, f.hi = lo, hi
    return f


def scapula_field(H, md, fem, sx):
    """Shoulder blade under the back's soft tissue: a low plate (scapula + infraspinatus/teres) with a
    slightly raised medial border and inferior angle, feathered into the surrounding tissue."""
    import sdf
    a = md * (1 - 0.25 * fem)
    lo = np.array([min(sx * 0.02, sx * 0.13) * H, -0.02 * H, 0.66 * H], np.float32)
    hi = np.array([max(sx * 0.02, sx * 0.13) * H, 0.14 * H, 0.83 * H], np.float32)
    # scapula triangle in the (x, z) plane (fractions of H): superior angle, inferior angle, glenoid
    sup, inf, gle = np.array([0.045, 0.80]), np.array([0.058, 0.70]), np.array([0.105, 0.785])

    def f(q, d):
        out = np.zeros(len(q), np.float32)
        m = np.all((q > lo) & (q < hi), axis=1) & (q[:, 1] > -0.02 * H)
        if not np.any(m):
            return out
        P2 = np.stack([q[m, 0] * sx / H, q[m, 2] / H], 1)
        # barycentric-ish inside measure via distance to the triangle edges
        def edge(a_, b_):
            ab = b_ - a_
            t = np.clip(((P2 - a_) @ ab) / (ab @ ab), 0, 1)
            return np.linalg.norm(P2 - (a_ + t[:, None] * ab), axis=1)
        dm = edge(sup, inf)                 # medial border
        dl = edge(inf, gle)                 # lateral border
        ds = edge(sup, gle)                 # spine side
        # inside test (same side of all three edges)
        def side(a_, b_, c_):
            return np.sign((b_[0] - a_[0]) * (P2[:, 1] - a_[1]) - (b_[1] - a_[1]) * (P2[:, 0] - a_[0])) == \
                   np.sign((b_[0] - a_[0]) * (c_[1] - a_[1]) - (b_[1] - a_[1]) * (c_[0] - a_[0]))
        inside = side(sup, inf, gle) & side(inf, gle, sup) & side(gle, sup, inf)
        dist_out = np.where(inside, 0.0, np.minimum(np.minimum(dm, dl), ds))
        plate = np.exp(-(dist_out / 0.012) ** 2) * 0.0022
        border = np.exp(-(dm / 0.006) ** 2) * 0.0010 + np.exp(-(np.linalg.norm(P2 - inf, axis=1) / 0.01) ** 2) * 0.0008
        # fade in over the side of the body (a hard cut at y = 0 left a crease down the shoulder)
        wy = np.clip((q[m, 1] / H + 0.005) / 0.02, 0, 1)
        wy = wy * wy * (3 - 2 * wy)
        out[m] = -((plate + border) * a * H * wy)
        return out
    f.lo, f.hi = lo, hi
    return f


def _torso_back(P, p, H, md, fem):
    BK = v3(0, 1, 0)
    UP = v3(0, 0, 1)
    P.D.append(trapezius_field(H, md, fem))
    for sx in (-1, 1):
        X = v3(sx, 0, 0)
        P.D.append(scapula_field(H, md, fem, sx))
        # lower trapezius border: a very soft line (the edge is mostly carried by the sheet's feathering)
        P.line(v3(sx * 0.05 * H, 0, 0.765 * H), BK, v3(sx * 0.008 * H, 0, 0.67 * H), BK, 0.009 * H, -0.0004 * H * md)
        # spine of the scapula and the acromion
        P.line(v3(sx * 0.035 * H, 0, 0.782 * H), BK, v3(sx * 0.095 * H, 0, 0.805 * H), BK + X * 0.6 + UP * 0.5, 0.0055 * H, 0.0014 * H)
        # infraspinatus and teres major over/below the scapula
        P.bump(v3(sx * 0.062 * H, 0, 0.752 * H), BK + X * 0.2, UP, 0.024 * H, 0.022 * H, 0.0016 * H * md)
        P.bump(v3(sx * 0.088 * H, 0, 0.728 * H), BK + X * 0.6, X + UP * 0.6, 0.02 * H, 0.012 * H, 0.0032 * H * md)
        # medial scapular border (soft crease)
        P.line(v3(sx * 0.03 * H, 0, 0.775 * H), BK, v3(sx * 0.045 * H, 0, 0.712 * H), BK, 0.009 * H, -0.0006 * H * md)
        # latissimus dorsi: wide sweep from the armpit down toward the lumbar fascia (the back's V)
        P.bump(v3(sx * 0.085 * H, 0, 0.69 * H), BK * 0.7 + X, UP, 0.06 * H, 0.022 * H, 0.0058 * H * md)
        P.line(v3(sx * 0.095 * H, 0, 0.705 * H), BK * 0.6 + X, v3(sx * 0.06 * H, 0, 0.6 * H), BK + X * 0.4, 0.007 * H, -0.0009 * H * md)
        # erector spinae columns
        P.line(v3(sx * 0.02 * H, 0, 0.665 * H), BK, v3(sx * 0.02 * H, 0, 0.56 * H), BK, 0.018 * H, 0.0032 * H * md)
        # lumbar dimples
        P.bump(v3(sx * 0.026 * H, 0, 0.543 * H), BK, UP, 0.006 * H, 0.006 * H, -0.0012 * H)
    # spinal groove
    P.line(v3(0, 0, 0.8 * H), BK, v3(0, 0, 0.55 * H), BK, 0.0085 * H, -0.0016 * H)
    # C7 vertebra prominence
    P.bump(v3(0, 0, 0.835 * H), BK + UP * 0.3, UP, 0.006 * H, 0.006 * H, 0.0006 * H)


def _pelvis(P, p, H, md, fem):
    FWD = v3(0, -1, 0)
    BK = v3(0, 1, 0)
    UP = v3(0, 0, 1)
    gl = 1 + 0.05 * fem + 0.3 * (p["fat"] - 0.2)
    for sx in (-1, 1):
        X = v3(sx, 0, 0)
        # inguinal V-lines and the hip crease
        P.line(v3(sx * 0.085 * H, 0, 0.547 * H), FWD + X * 0.5, v3(sx * 0.02 * H, 0, 0.475 * H), FWD, 0.008 * H, -0.0022 * H * md)
        # iliac crest ridge and gluteus medius / TFL mass on the upper outer hip
        P.line(v3(sx * 0.09 * H, 0, 0.553 * H), X + BK * 0.5, v3(sx * 0.08 * H, 0, 0.548 * H), X + FWD * 0.6, 0.007 * H, 0.0014 * H * (1 - 0.8 * fem))
        P.bump(v3(sx * 0.097 * H, 0, 0.515 * H), X, UP, 0.03 * H, 0.016 * H, 0.0032 * H * (1 - 0.7 * fem))
        # glutes: rounded cheeks, gluteal fold
        P.bump(v3(sx * (0.052 - 0.004 * fem) * H, 0, (0.49 - 0.006 * fem) * H), BK, UP, (0.048 + 0.008 * fem) * H, (0.044 - 0.008 * fem) * H, 0.0175 * H * gl, 0.04 * H)
        P.bump(v3(sx * 0.07 * H, 0, 0.51 * H), BK * (0.6 + 0.6 * fem) + X, UP, 0.03 * H, 0.026 * H, 0.0055 * H * (1 - 0.85 * fem))
        P.line(v3(sx * 0.02 * H, 0, 0.458 * H), BK, v3(sx * 0.085 * H, 0, 0.462 * H), BK + X * 0.3, 0.008 * H, -0.0018 * H)
    P.line(v3(0, 0, 0.54 * H), BK, v3(0, 0, 0.455 * H), BK, 0.006 * H, -0.006 * H)     # natal cleft
    P.bump(v3(0, 0, 0.478 * H), FWD, UP, 0.018 * H, 0.03 * H, 0.0035 * H)              # pubic mound


# ------------------------------------------------------------------- arms --
def _arm_frame(J, side, sx):
    sh, el, wr = J[f"upper_arm.{side}"], J[f"forearm.{side}"], J[f"hand.{side}"]
    ax = _n(el - sh)
    fr = _n(v3(0, -1, 0) - ax * np.dot(v3(0, -1, 0), ax))
    lat = _n(np.cross(fr, ax))
    if np.dot(lat, v3(sx, 0, 1.0)) < 0:
        lat = -lat
    ax2 = _n(wr - el)
    fr2 = _n(v3(0, -1, 0) - ax2 * np.dot(v3(0, -1, 0), ax2))
    lat2 = _n(np.cross(fr2, ax2))
    if np.dot(lat2, lat) < 0:
        lat2 = -lat2
    return sh, el, wr, ax, fr, lat, ax2, fr2, lat2


def _arm(P, J, side, sx, H, md):
    sh, el, wr, ax, fr, lat, ax2, fr2, lat2 = _arm_frame(J, side, sx)
    L1 = float(np.linalg.norm(el - sh))
    L2 = float(np.linalg.norm(wr - el))

    def U(t):
        return sh + ax * (t * L1)

    def Fo(t):
        return el + ax2 * (t * L2)

    bk, med = -fr, -lat
    # ---- deltoid: anterior, lateral, posterior heads + insertion V
    P.bump(U(0.1), _n(fr + lat * 0.5), ax, 0.05 * H, 0.016 * H, 0.0032 * H * md)
    P.bump(U(0.14), lat, ax, 0.055 * H, 0.018 * H, 0.0036 * H * md)
    P.bump(U(0.1), _n(bk + lat * 0.5), ax, 0.05 * H, 0.016 * H, 0.003 * H * md)
    P.line(U(0.05), _n(fr + lat * 0.25), U(0.35), _n(fr + lat * 0.6), 0.0045 * H, -0.0008 * H * md)
    P.line(U(0.05), _n(bk + lat * 0.25), U(0.35), _n(bk + lat * 0.6), 0.0045 * H, -0.0007 * H * md)
    # ---- biceps brachii: two heads, peak at ~55%, tapering into the elbow
    P.bump(U(0.56), _n(fr + med * 0.15), ax, 0.075 * H, 0.016 * H, 0.0048 * H * md)
    P.line(U(0.38), fr, U(0.72), fr, 0.0035 * H, -0.0006 * H * md)
    # brachialis peeking out laterally above the elbow
    P.bump(U(0.74), _n(lat + fr * 0.5), ax, 0.035 * H, 0.01 * H, 0.0022 * H * md)
    # ---- triceps: long head (medial-back), lateral head (horseshoe), flat tendon plate
    P.bump(U(0.48), _n(bk + med * 0.6), ax, 0.075 * H, 0.016 * H, 0.0042 * H * md)
    P.bump(U(0.42), _n(bk + lat * 0.6), ax, 0.06 * H, 0.014 * H, 0.0038 * H * md)
    P.bump(U(0.82), bk, ax, 0.028 * H, 0.016 * H, -0.0016 * H * md)
    # grooves: lateral intermuscular (deltoid insertion -> elbow), medial bicipital groove
    P.line(U(0.42), lat, U(0.86), _n(lat + fr * 0.2), 0.005 * H, -0.0012 * H * md)
    P.line(U(0.22), med, U(0.8), med, 0.006 * H, -0.0011 * H * md)
    # ---- elbow: epicondyles, cubital fossa
    P.bump(U(0.98), med, ax, 0.008 * H, 0.008 * H, 0.0018 * H)
    P.bump(U(0.98), lat, ax, 0.007 * H, 0.007 * H, 0.0012 * H)
    P.bump(Fo(0.04), fr2, ax2, 0.016 * H, 0.012 * H, -0.0022 * H)
    # ---- forearm
    # brachioradialis: lateral/front bulge from above the elbow toward the thumb
    P.line(U(0.8), _n(lat + fr * 0.3), Fo(0.5), _n(lat2 + fr2 * 0.4), 0.014 * H, 0.0038 * H * md)
    P.bump(Fo(0.17), _n(lat2 + fr2 * 0.5), ax2, 0.045 * H, 0.014 * H, 0.0025 * H * md)
    # extensor group (back-lateral) and flexor group (front-medial)
    P.bump(Fo(0.3), _n(-fr2 + lat2 * 0.5), ax2, 0.06 * H, 0.013 * H, 0.0026 * H * md)
    P.bump(Fo(0.27), _n(fr2 - lat2 * 0.6), ax2, 0.06 * H, 0.016 * H, 0.0032 * H * md)
    # ulna ridge (subcutaneous border on the back-medial side)
    P.line(Fo(0.05), _n(-fr2 - lat2 * 0.7), Fo(0.92), _n(-fr2 - lat2 * 0.7), 0.0045 * H, -0.001 * H * md)
    # wrist: ulnar styloid, flexor tendons
    P.bump(Fo(0.97), _n(-fr2 - lat2 * 0.8), ax2, 0.005 * H, 0.005 * H, 0.0014 * H)
    P.line(Fo(0.65), fr2, Fo(0.98), fr2, 0.0035 * H, 0.0007 * H)


# ------------------------------------------------------------------- legs --
def _leg(P, J, side, sx, H, md):
    hj, kn, an = J[f"thigh.{side}"], J[f"shin.{side}"], J[f"foot.{side}"]
    ax = _n(kn - hj)
    sk = _n(an - kn)
    FWD = v3(0, -1, 0)
    BK = v3(0, 1, 0)
    X = v3(sx, 0, 0)
    L1 = float(np.linalg.norm(kn - hj))
    L2 = float(np.linalg.norm(an - kn))

    def T(t):
        return hj + ax * (t * L1)

    def S(t):
        return kn + sk * (t * L2)

    # pelvic landmarks the thigh muscles originate from
    asis = v3(sx * 0.082 * H, 0, 0.548 * H)        # anterior superior iliac spine
    pubis = v3(sx * 0.018 * H, 0, 0.472 * H)
    # ---- hip flexors / quadriceps flowing out of the pelvis
    P.line(asis, _n(FWD + X * 0.5), T(0.45), _n(FWD + X * 0.2), 0.016 * H, 0.0026 * H * md)          # rectus femoris origin
    P.line(T(0.3), FWD, T(0.86), FWD, 0.02 * H, 0.0034 * H * md)                                     # rectus femoris belly
    P.line(asis, _n(X + FWD * 0.6), T(0.3), _n(X + FWD * 0.3), 0.014 * H, 0.0028 * H * md)           # tensor fasciae latae
    P.bump(T(0.5), _n(X + FWD * 0.6), ax, 0.09 * H, 0.022 * H, 0.0042 * H * md)                       # vastus lateralis
    P.bump(T(0.86), _n(-X + FWD * 0.8), ax, 0.035 * H, 0.016 * H, 0.0045 * H * md)                    # vastus medialis
    P.line(asis, _n(FWD + X * 0.2), T(0.88), _n(-X + FWD * 0.3), 0.008 * H, -0.0013 * H * md)        # sartorius groove
    P.line(T(0.25), _n(FWD + X * 0.45), T(0.85), _n(FWD + X * 0.3), 0.009 * H, -0.0012 * H * md)
    P.line(T(0.35), _n(FWD - X * 0.4), T(0.8), _n(FWD - X * 0.2), 0.009 * H, -0.001 * H * md)
    P.line(T(0.25), X, T(0.9), _n(X + BK * 0.2), 0.007 * H, -0.0009 * H * md)                        # IT band
    # adductors: fan from the pubis down the inner thigh (builds the inner V)
    P.line(pubis, _n(FWD - X * 0.6), T(0.55), _n(-X + FWD * 0.2), 0.02 * H, 0.0034 * H * md)
    P.bump(T(0.25), _n(-X + FWD * 0.3), ax, 0.05 * H, 0.02 * H, 0.0028 * H * md)
    # hamstrings: biceps femoris (lateral), semitendinosus/membranosus (medial)
    P.line(T(0.2), _n(BK + X * 0.4), T(0.85), _n(BK + X * 0.6), 0.016 * H, 0.0022 * H * md)
    P.line(T(0.2), _n(BK - X * 0.4), T(0.85), _n(BK - X * 0.6), 0.016 * H, 0.0022 * H * md)
    # ---- knee
    P.bump(kn, FWD, ax, 0.012 * H, 0.011 * H, 0.0026 * H)                                         # patella
    P.line(S(0.04), FWD, S(0.12), FWD, 0.006 * H, 0.0011 * H)                                      # patellar tendon
    P.bump(S(0.02), _n(FWD - X), ax, 0.008 * H, 0.008 * H, -0.001 * H)                             # medial knee hollow
    P.bump(S(-0.02), BK, ax, 0.012 * H, 0.012 * H, -0.0018 * H)                                    # popliteal fossa
    # ---- lower leg
    P.bump(S(0.28), _n(BK - X * 0.55), sk, 0.055 * H, 0.016 * H, 0.0042 * H * md)                  # gastrocnemius medial
    P.bump(S(0.24), _n(BK + X * 0.55), sk, 0.045 * H, 0.014 * H, 0.0034 * H * md)                  # gastrocnemius lateral
    P.line(S(0.18), BK, S(0.36), BK, 0.005 * H, -0.0011 * H * md)
    P.bump(S(0.5), _n(X + BK * 0.4), sk, 0.06 * H, 0.012 * H, 0.0018 * H * md)                     # soleus (lateral)
    P.bump(S(0.46), _n(-X + BK * 0.3), sk, 0.05 * H, 0.012 * H, 0.0016 * H * md)                   # soleus (medial)
    P.line(S(0.12), _n(FWD + X * 0.55), S(0.7), _n(FWD + X * 0.45), 0.011 * H, 0.0024 * H * md)    # tibialis anterior
    P.line(S(0.12), _n(FWD - X * 0.15), S(0.85), _n(FWD - X * 0.2), 0.005 * H, 0.0008 * H)         # tibia crest
    P.line(S(0.2), _n(X + BK * 0.1), S(0.8), X, 0.006 * H, -0.0008 * H * md)                       # peroneal groove
    for o in (-1, 1):                                                                              # achilles tendon edges
        P.line(S(0.72), _n(BK + X * o * 0.6), S(0.97), _n(BK + X * o * 0.6), 0.005 * H, -0.001 * H)
