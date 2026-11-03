"""External ear (auricle) built from the measured side-view outline of the reference (4.webp).

The ear is a relief plate standing off the side of the skull:
  * 2D outline (helix edge) traced on the side view -> signed 2D distance d2
  * the plate's outer surface = head surface + flare (ear stands off toward the back rim) + relief
  * relief: helix roll along the rim, scapha groove inside it, antihelix ridge with its two crura,
    triangular fossa between the crura, deep concha bowl, tragus, fleshy rim-less lobe
  * the front (concha / tragus) is attached to the head; the back rim stands free with a thin back.
All 2D coordinates are in head units: f = forward from the head centre plane, zn = down from the top.
"""
from __future__ import annotations

import numpy as np

import sdf
from sdf import F


def _crop(pts):
    """side-view crop (880,230)+, 4x zoom  ->  (f, zn) head units"""
    pts = np.asarray(pts, np.float64)
    return np.stack([(7.0 - pts[:, 0] / 4) / 552, (220.0 + pts[:, 1] / 4) / 552], 1)


OUTLINE = _crop([(110, 170), (150, 90), (230, 52), (320, 62), (385, 120), (412, 210), (405, 320), (370, 420),
                 (320, 500), (285, 570), (230, 612), (160, 610), (90, 575), (55, 525), (75, 470), (105, 430),
                 (95, 330), (100, 240)])
ANTIHELIX = _crop([(205, 455), (265, 405), (305, 320), (308, 235), (292, 150)])      # stem + superior crus
INF_CRUS = _crop([(306, 250), (240, 215), (165, 235)])
CONCHA = _crop([(175, 375)])[0], np.array([62 / 4 / 552, 82 / 4 / 552])
FOSSA = _crop([(245, 175)])[0]
TRAGUS = _crop([(92, 440)])[0]
LOBE_ZN = _crop([(0, 500)])[0][1]


def _closed_spline(P, n=160):
    P = np.asarray(P, np.float64)
    m = len(P)
    out = []
    for i in range(m):
        p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        for t in np.linspace(0, 1, n // m, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def _open_spline(P, n=40):
    P = np.asarray(P, np.float64)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def _seg_dist(q, P, closed):
    """unsigned distance from 2D points q to a polyline P"""
    A = P if closed else P[:-1]
    B = np.roll(P, -1, 0) if closed else P[1:]
    best = np.full(len(q), np.inf)
    for a, b in zip(A, B):
        ab = b - a
        t = np.clip(((q - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.hypot(*(q - (a + np.outer(t, ab))).T)
        best = np.minimum(best, d)
    return best


def _inside(q, P):
    x, y = q[:, 0], q[:, 1]
    ins = np.zeros(len(q), bool)
    A, B = P, np.roll(P, -1, 0)
    for a, b in zip(A, B):
        c = ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
        ins ^= c
    return ins


def build(s, p, J, hh, P_, scale=1.0):
    """Adds both ears to shape s.  P_(lat, fwd, zn) -> world point."""
    top = float(J["head_top"][2])
    yc = float(J["head"][1])
    ctr = OUTLINE.mean(0)
    outline = _closed_spline(ctr + (OUTLINE - ctr) * scale)
    anti = _open_spline(ctr + (ANTIHELIX - ctr) * scale)
    crus = _open_spline(ctr + (INF_CRUS - ctr) * scale)
    cc, cr = ctr + (CONCHA[0] - ctr) * scale, CONCHA[1] * scale
    fossa = ctr + (FOSSA - ctr) * scale
    trag = ctr + (TRAGUS - ctr) * scale
    lobe_z = ctr[1] + (LOBE_ZN - ctr[1]) * scale
    fmin, zmin = outline.min(0) - 0.03
    fmax, zmax = outline.max(0) + 0.03

    # lateral position of the head surface on a (f, zn) grid, found by bisection on the current shape
    gf = np.linspace(fmin, fmax, 28)
    gz = np.linspace(zmin, zmax, 32)
    GF, GZ = np.meshgrid(gf, gz, indexing="ij")
    lo_u, hi_u = np.full(GF.size, 0.15), np.full(GF.size, 0.6)
    for _ in range(26):
        mid = (lo_u + hi_u) / 2
        pts = np.stack([mid * hh, yc - GF.ravel() * hh, top - GZ.ravel() * hh], 1).astype(F)
        ins = s.eval(pts) < 0
        lo_u = np.where(ins, mid, lo_u)
        hi_u = np.where(ins, hi_u, mid)
    XB = ((lo_u + hi_u) / 2).reshape(GF.shape)

    def xbase(f, zn):
        i = np.clip((f - fmin) / (fmax - fmin) * (len(gf) - 1), 0, len(gf) - 1.001)
        j = np.clip((zn - zmin) / (zmax - zmin) * (len(gz) - 1), 0, len(gz) - 1.001)
        i0, j0 = i.astype(int), j.astype(int)
        a, b = i - i0, j - j0
        return (XB[i0, j0] * (1 - a) * (1 - b) + XB[i0 + 1, j0] * a * (1 - b)
                + XB[i0, j0 + 1] * (1 - a) * b + XB[i0 + 1, j0 + 1] * a * b)

    f_front, f_back = outline[:, 0].max(), outline[:, 0].min()

    def ear_sdf(q):
        u = np.abs(q[:, 0]) / hh
        f = (yc - q[:, 1]) / hh
        zn = (top - q[:, 2]) / hh
        q2 = np.stack([f, zn], 1).astype(np.float64)
        d2 = _seg_dist(q2, outline, True)
        d2 = np.where(_inside(q2, outline), -d2, d2)
        xb = xbase(f, zn)
        back_t = np.clip((f_front - f) / (f_front - f_back), 0, 1)          # 0 at the front, 1 at the back rim
        top_t = np.clip((lobe_z - zn) / (lobe_z - outline[:, 1].min()), 0, 1)
        flare = 0.012 + 0.048 * back_t ** 1.2 * (0.6 + 0.4 * top_t)
        lobe = np.clip((zn - lobe_z) / 0.03, 0, 1)
        rimw = 0.022 * scale
        # helix roll (not on the lobe)
        rr = np.clip(1 - ((d2 + rimw / 2) / (rimw / 2)) ** 2, 0, 1)
        relief = 0.008 * np.sqrt(rr) * (1 - lobe)
        relief -= 0.009 * np.exp(-((d2 + rimw * 1.45) / (0.008 * scale)) ** 2) * (1 - lobe) * np.clip(back_t * 3, 0, 1)
        # antihelix + inferior crus
        da = _seg_dist(q2, anti, False)
        dc = _seg_dist(q2, crus, False)
        relief += 0.009 * np.exp(-(da / (0.011 * scale)) ** 2) + 0.006 * np.exp(-(dc / (0.008 * scale)) ** 2)
        relief -= 0.007 * np.exp(-(np.hypot(f - fossa[0], zn - fossa[1]) / (0.016 * scale)) ** 2)
        # concha bowl: sinks back to (and into) the head surface
        e = ((f - cc[0]) / cr[0]) ** 2 + ((zn - cc[1]) / cr[1]) ** 2
        bowl = np.clip(1 - e, 0, 1) ** 0.7
        relief -= (flare + 0.008) * bowl
        # tragus: small flap covering the front of the concha
        relief += 0.010 * np.exp(-(np.hypot(f - trag[0], zn - trag[1]) / (0.012 * scale)) ** 2)
        # lobe: soft and full
        relief += 0.006 * lobe * np.clip(-d2 / 0.02, 0, 1) ** 0.5
        xo = xb + flare + relief
        # back of the auricle: attached at the front, a ~4 mm plate at the free back rim
        thick = 0.018 + 0.05 * (1 - np.clip((back_t - 0.35) / 0.3, 0, 1))
        xback = np.maximum(xo - thick, xb - 0.03)
        d = sdf.smax(sdf.smax(d2, u - xo, 0.006), xback - u, 0.006)
        return (d * hh * 0.7).astype(F)

    for sx in (-1, 1):
        lo = np.array([min(sx * 0.2, sx * 0.6) * hh, yc - fmax * hh, top - zmax * hh], F)
        hi = np.array([max(sx * 0.2, sx * 0.6) * hh, yc - fmin * hh, top - zmin * hh], F)

        def side(q, sx=sx):
            out = ear_sdf(q)
            return np.where(np.sign(q[:, 0]) == sx, out, 1.0).astype(F)
        s.union((side, lo, hi), k=0.008 * hh, tag="skin", name="ear")
