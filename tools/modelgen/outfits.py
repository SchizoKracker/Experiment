"""Garments, hair and accessories built as shells around the body SDF.

A garment = (body distance - thickness) intersected with a region volume;
region boundaries are hard planes, so hems/cuffs come out crisp.
Because every piece is derived from the body field, it fits any race/build.
"""
from __future__ import annotations

import math

import numpy as np

import sdf
from humanoid import HeadSpace, hand_frame, v3
from sdf import F, Shape, capsule, ellipsoid, frame_z, round_box, round_cone, sphere


# --------------------------------------------------------------------------
# region helpers
# --------------------------------------------------------------------------
def plane_cut(point, normal):
    """Keep the side the normal points AWAY from (negative half-space)."""
    return sdf.half_space(point, normal)


def limb_segment(a, b, r, cut_a=True, cut_b=True):
    """Capsule between a and b with flat (hard) caps -> crisp sleeve/leg hems."""
    a = np.asarray(a, F)
    b = np.asarray(b, F)
    d = (b - a) / np.linalg.norm(b - a)
    cap = capsule(a, b, r)
    pa = sdf.half_space(a, -d) if cut_a else None
    pb = sdf.half_space(b, d) if cut_b else None

    def f(p, cap=cap[0], pa=pa, pb=pb):
        v = cap(p)
        if pa is not None:
            v = np.maximum(v, pa[0](p))
        if pb is not None:
            v = np.maximum(v, pb[0](p))
        return v
    return f, cap[1], cap[2]


class Garment:
    def __init__(self, body: Shape, thickness: float, name: str):
        self.body = body
        self.t = thickness
        self.name = name
        self.regions = []   # hard-unioned region volumes
        self.cuts = []      # extra hard half-space intersections
        self.adds = []      # extra solid details (buttons, buckles, soles)
        self.folds = 0.0
        self.fold_freq = 18.0
        self.seed = 3

    def region(self, prim):
        self.regions.append(prim)
        return self

    def cut(self, prim):
        self.cuts.append(prim)
        return self

    def add(self, prim, k=0.0):
        self.adds.append((prim, k))
        return self

    def shape(self) -> Shape:
        body, t = self.body, self.t
        regions, cuts, adds = self.regions, self.cuts, self.adds
        folds, ff, seed = self.folds, self.fold_freq, self.seed

        lo = np.min([r[1] for r in regions], axis=0) - 0.05
        hi = np.max([r[2] for r in regions], axis=0) + 0.05

        def f(p):
            reg = np.full(len(p), 10.0, F)
            for r in regions:
                reg = np.minimum(reg, r[0](p))
            out = np.full(len(p), 10.0, F)
            m = reg < 0.03
            if np.any(m):
                q = p[m]
                d = body.eval(q) - t
                if folds > 0:
                    # anisotropic wrinkles: stretched noise along Z (gravity) + soft bulges
                    qq = q * np.array([1.0, 1.0, 0.35], F)
                    d = d - folds * (0.6 * sdf.fbm(qq, ff, 3, seed) + 0.4 * np.abs(sdf.value_noise(q, ff * 0.5, seed + 7)))
                d = np.maximum(d, reg[m])
                for c in cuts:
                    d = np.maximum(d, c[0](q))
                out[m] = d
            for prim, k in adds:
                out = sdf.smin(out, prim[0](p), k)
            return out

        s = Shape()
        s.union((f, lo, hi), tag=self.name)
        return s


# --------------------------------------------------------------------------
# outfit pieces
# --------------------------------------------------------------------------
def torso_volume(J, p, top_z=None, bottom_z=None):
    sc = p["height"] / 1.78
    hips, chest, neck = J["hips"], J["chest"], J["neck"]
    top = neck[2] - 0.01 * sc if top_z is None else top_z
    bot = hips[2] - 0.06 * sc if bottom_z is None else bottom_z
    w = 0.26 * sc * max(p["shoulder"], p["hip"]) * (1.0 + 0.25 * p["fat"])
    body = capsule(v3(0, 0.0, bot - 0.2), v3(0, 0.0, top + 0.2), w)
    up = sdf.half_space(v3(0, 0, top), [0, 0, 1])
    dn = sdf.half_space(v3(0, 0, bot), [0, 0, -1])

    def f(q, b=body[0], u=up[0], d=dn[0]):
        return np.maximum(np.maximum(b(q), u(q)), d(q))
    return f, body[1], body[2]


def shirt(body, J, p, sleeve="short", hem=0.0, collar="v", thickness=0.006, folds=0.0035, name="shirt"):
    sc = p["height"] / 1.78
    g = Garment(body, thickness * sc, name)
    g.folds = folds * sc
    hem_z = J["hips"][2] - (0.03 - hem) * sc
    g.region(torso_volume(J, p, top_z=J["neck"][2] - 0.012 * sc, bottom_z=hem_z))
    for side in ("L", "R"):
        sh, el, wr = J[f"upper_arm.{side}"], J[f"forearm.{side}"], J[f"hand.{side}"]
        if sleeve == "none":
            continue
        end = sh + (el - sh) * 0.55 if sleeve == "short" else (el + (wr - el) * 0.4 if sleeve == "three_q" else wr - (wr - el) * 0.06)
        start = sh - (el - sh) * 0.3
        if sleeve == "short" or sleeve == "three_q":
            g.region(limb_segment(start, end if sleeve == "short" else el, 0.11 * sc, cut_a=False))
            if sleeve == "three_q":
                g.region(limb_segment(el - (el - sh) * 0.1, end, 0.09 * sc, cut_a=False))
        else:
            g.region(limb_segment(start, el, 0.11 * sc, cut_a=False))
            g.region(limb_segment(el - (el - sh) * 0.1, end, 0.09 * sc, cut_a=False))
    # keep the shirt off the head and neck entirely
    hd = HeadSpace(p, J)
    g.cut(_neg(sphere(hd(0, -0.01, -0.01), hd.r(0.13))))
    g.cut(sdf.half_space(v3(0, 0, J["neck"][2] - 0.005 * sc), [0, 0, 1]))
    # neckline
    n = J["neck"]
    if collar == "v":
        g.cut(_neg(ellipsoid(n + v3(0, -0.08 * sc, 0.005 * sc), [0.05 * sc, 0.09 * sc, 0.085 * sc])))
    else:
        g.cut(_neg(ellipsoid(n + v3(0, -0.02 * sc, 0.03 * sc), [0.065 * sc, 0.08 * sc, 0.05 * sc])))
    g.cut(_neg(ellipsoid(n + v3(0, 0.0, 0.07 * sc), [0.075 * sc, 0.075 * sc, 0.06 * sc])))
    return g.shape()


def _neg(prim):
    fn = prim[0]
    return (lambda q, fn=fn: -fn(q)), prim[1], prim[2]


def trousers(body, J, p, length="full", thickness=0.007, folds=0.004, name="trousers"):
    sc = p["height"] / 1.78
    g = Garment(body, thickness * sc, name)
    g.folds = folds * sc
    waist_z = J["spine"][2] + 0.01 * sc
    g.region(torso_volume(J, p, top_z=waist_z, bottom_z=J["hips"][2] - 0.25 * sc))
    for side in ("L", "R"):
        hj, kn, an = J[f"thigh.{side}"], J[f"shin.{side}"], J[f"foot.{side}"]
        if length == "shorts":
            end = kn + (hj - kn) * 0.12
        elif length == "knee":
            end = kn + (an - kn) * 0.18
        else:
            end = an + v3(0, 0, 0.02 * sc)
        g.region(limb_segment(hj + v3(0, 0, 0.05 * sc), kn, 0.13 * sc, cut_a=False, cut_b=False))
        g.region(limb_segment(kn - (kn - hj) * 0.05, end, 0.11 * sc, cut_a=False))
        if length == "shorts":
            g.cut(sdf.half_space(end, (end - hj) / np.linalg.norm(end - hj)))
    return g.shape()


def belt(body, J, p, buckle=True, name="belt"):
    sc = p["height"] / 1.78
    g = Garment(body, 0.016 * sc, name)
    z = J["spine"][2] - 0.005 * sc
    g.region(torso_volume(J, p, top_z=z + 0.02 * sc, bottom_z=z - 0.02 * sc))
    if buckle:
        front_y = -0.13 * sc * (1 + 0.6 * p["fat"])
        # buckle sits on the belt surface: find it by marching from the front
        q = v3(0, front_y, z)
        for _ in range(12):
            d = body.eval(q[None, :])[0] - 0.016 * sc
            q = q + v3(0, d, 0)
        g.add(round_box(q + v3(0, -0.004 * sc, 0), [0.026 * sc, 0.006 * sc, 0.024 * sc], None, 0.004 * sc), 0.002)
    return g.shape()


def boots(body, J, p, height="mid", thickness=0.012, name="boots"):
    sc = p["height"] / 1.78
    g = Garment(body, thickness * sc, name)
    g.folds = 0.002 * sc
    for side in ("L", "R"):
        kn, an, toe = J[f"shin.{side}"], J[f"foot.{side}"], J[f"toe_end.{side}"]
        top = kn + (an - kn) * (0.25 if height == "high" else (0.55 if height == "mid" else 0.85))
        g.region(limb_segment(an + v3(0, 0, -0.2 * sc), top, 0.12 * sc, cut_a=False))
        g.region(capsule(an + v3(0, 0.06 * sc, -0.03 * sc), toe + v3(0, -0.03 * sc, 0), 0.08 * sc))
        # flat sole with a heel
        heel = an + v3(0, 0.035 * sc, -0.045 * sc)
        g.add(round_box(v3((heel[0] + toe[0]) / 2, (heel[1] + toe[1]) / 2 - 0.004 * sc, 0.009 * sc),
                        [0.047 * sc, np.linalg.norm(toe[:2] - heel[:2]) * 0.5 + 0.035 * sc, 0.009 * sc], None, 0.006 * sc), 0.004 * sc)
    s = g.shape()
    return s


def sandals_wraps(body, J, p, name="wraps"):
    """Cloth leg wraps (the reference's socks) – banded shells on the shins."""
    sc = p["height"] / 1.78
    g = Garment(body, 0.006 * sc, name)
    for side in ("L", "R"):
        kn, an = J[f"shin.{side}"], J[f"foot.{side}"]
        g.region(limb_segment(an + v3(0, 0, -0.02 * sc), kn + (an - kn) * 0.45, 0.1 * sc))
    base = g.shape()
    # spiral wrap ridges
    prim = base.prims[0]

    def f(q, f0=prim.fn):
        d = f0(q)
        ang = np.arctan2(q[:, 1], np.abs(q[:, 0]) - 0.09 * sc)
        return d - 0.0018 * sc * np.sin(q[:, 2] * 220.0 / sc + ang * 2.0)
    s = Shape()
    s.union((f, prim.lo, prim.hi), tag=name)
    return s


def gloves(body, J, p, cuff=0.35, name="gloves"):
    sc = p["height"] / 1.78
    g = Garment(body, 0.004 * sc, name)
    for side in ("L", "R"):
        el, wr = J[f"forearm.{side}"], J[f"hand.{side}"]
        g.region(limb_segment(wr + (el - wr) * cuff, J[f"hand_end.{side}"] + (J[f"hand_end.{side}"] - wr) * 0.8, 0.12 * sc))
    return g.shape()


def tunic_skirt(body, J, p, length=0.22, name="tunic"):
    """Flared skirt/tunic tail hanging from the hips (separate from legs)."""
    sc = p["height"] / 1.78
    hips = J["hips"]
    top = J["spine"][2]
    bottom = hips[2] - length * sc
    r_top = 0.17 * sc * (1 + 0.3 * p["fat"]) * max(1.0, p["hip"])
    r_bot = r_top * 1.35
    cone = round_cone(v3(0, 0.01 * sc, bottom), v3(0, 0.0, top), r_bot, r_top)
    shell = cone[0]

    def f(q):
        d = shell(q)
        ang = np.arctan2(q[:, 1], q[:, 0])
        d = d - 0.006 * sc * np.sin(ang * 14.0) * np.clip((top - q[:, 2]) / (top - bottom), 0, 1)
        inner = -(d + 0.006 * sc)
        d = np.maximum(d, inner)
        d = np.maximum(d, q[:, 2] - top)
        d = np.maximum(d, bottom - q[:, 2])
        return d
    s = Shape()
    s.union((f, cone[1], cone[2]), tag=name)
    return s


# --------------------------------------------------------------------------
# hair & face hair
# --------------------------------------------------------------------------
HAIR_STYLES = ("short", "crop", "long", "ponytail", "bun", "braids", "mohawk", "bald", "swept")
BEARD_STYLES = ("none", "stubble", "short", "full", "braided", "goatee")


def hair(body, J, p, style="short", volume=1.0, name="hair"):
    if style == "bald":
        return None
    h = HeadSpace(p, J)
    vol = volume
    cr_c = h(0, 0.006, 0.03)
    cr_r = np.array(h.r(0.075, 0.098, 0.092), F)
    thick = {"crop": 0.006, "short": 0.012, "swept": 0.016, "mohawk": 0.004}.get(style, 0.014) * vol
    shell = ellipsoid(cr_c + v3(0, 0.004 * h.k, 0.004 * h.k), cr_r + thick * h.k)
    parts = [(shell, 0.0)]
    # hairline: cut away face area & below ears
    cuts = [
        sdf.half_space(h(0, -0.06, 0.04), [0, -0.55, -0.83]),          # forehead hairline (tilted plane)
    ]
    if style in ("long", "braids"):
        parts.append((round_box(h(0, 0.06, -0.08), h.r(0.08 * vol, 0.045 * vol, 0.16), None, h.r(0.035)), h.r(0.04)))
        for sx in (-1, 1):
            parts.append((round_cone(h(sx * 0.07, -0.01, 0.02), h(sx * 0.075, 0.02, -0.2), h.r(0.025 * vol), h.r(0.02)), h.r(0.03)))
    elif style == "ponytail":
        base = h(0, 0.095, 0.04)
        pts = [base, h(0, 0.13, -0.02), h(0, 0.135, -0.12), h(0, 0.12, -0.22)]
        for seg in sdf.bent_cone(pts, [h.r(0.025), h.r(0.03), h.r(0.024), h.r(0.008)]):
            parts.append((seg, h.r(0.01)))
        cuts.append(sdf.half_space(h(0, 0.0, -0.03), [0, -0.5, -1.0]))
    elif style == "bun":
        parts.append((sphere(h(0, 0.09, 0.09), h.r(0.04)), h.r(0.012)))
        cuts.append(sdf.half_space(h(0, 0.0, -0.03), [0, -0.5, -1.0]))
    elif style == "mohawk":
        parts.append((round_box(h(0, 0.0, 0.11), h.r(0.012, 0.09, 0.04), None, h.r(0.01)), h.r(0.01)))
        cuts.append(sdf.half_space(h(0, 0.0, -0.01), [0, 0.0, -1.0]))
    else:
        cuts.append(sdf.half_space(h(0, 0.0, -0.04), [0, -0.35, -1.0]))   # nape
    if style in ("short", "swept", "long", "ponytail"):
        for i, (x, ln) in enumerate([(-0.04, 0.04), (-0.012, 0.05), (0.018, 0.045), (0.045, 0.035)]):
            a = h(x, -0.075, 0.075)
            b = h(x + 0.012, -0.096, 0.075 - ln)
            parts.append((round_cone(a, b, h.r(0.014), h.r(0.004)), h.r(0.012)))
    if style == "braids":
        for sx in (-1, 1):
            pts = [h(sx * 0.06, 0.07, -0.05), h(sx * 0.07, 0.08, -0.18), h(sx * 0.07, 0.07, -0.32)]
            for seg in sdf.bent_cone(pts, [h.r(0.02), h.r(0.017), h.r(0.012)]):
                parts.append((seg, h.r(0.006)))
    # ear windows for short styles
    ear_cut = []
    if style in ("short", "crop", "swept", "mohawk"):
        for sx in (-1, 1):
            ear_cut.append(ellipsoid(h(sx * 0.085, 0.0, -0.01), h.r(0.03, 0.03, 0.04)))
    strand_freq = 260.0 / h.k
    k_body = body

    def f(q):
        d = np.full(len(q), 10.0, F)
        for prim, k in parts:
            d = sdf.smin(d, prim[0](q), k)
        for c in cuts:
            d = np.maximum(d, c[0](q))
        for e in ear_cut:
            d = np.maximum(d, -e[0](q))
        # strand grooves: flow from crown backwards/downwards
        rel = q - cr_c
        ang = np.arctan2(rel[:, 0], -rel[:, 1])
        flow = np.arctan2(rel[:, 0], rel[:, 2] + 0.05 * h.k)          # strands run front -> back
        clump = sdf.fbm(q, 22.0 / h.k, 3, 11)
        d = d + 0.0013 * h.k * np.abs(np.sin(flow * 46.0 + clump * 2.5)) - 0.0045 * h.k * clump
        # never intersect the scalp
        return np.maximum(d, -(k_body.eval(q) - 0.0015))
    lo = cr_c - 0.4
    hi = cr_c + 0.4
    s = Shape()
    s.union((f, lo, hi), tag=name)
    return s


def beard(body, J, p, style="full", name="hair"):
    if style == "none" or p["snout"] > 0:
        return None
    h = HeadSpace(p, J)
    t = {"stubble": 0.0025, "short": 0.008, "goatee": 0.01, "full": 0.016, "braided": 0.014}[style]
    g = Garment(body, t * h.k, name)
    if style == "goatee":
        g.region(ellipsoid(h(0, -0.09, -0.09), h.r(0.026, 0.04, 0.04)))
    else:
        g.region(ellipsoid(h(0, -0.06, -0.085), h.r(0.07, 0.07, 0.055)))
        g.cut(sdf.half_space(h(0, 0, -0.035), [0, 0.0, 1.0]))  # stay below the cheekbones
    # keep the lips visible
    g.cut(_neg(ellipsoid(h(0, -0.1, -0.07), h.r(0.024, 0.03, 0.013))))
    s = g.shape()
    if style in ("full", "braided"):
        pts = [h(0, -0.1, -0.11), h(0, -0.11, -0.16 if style == "full" else -0.2), h(0, -0.1, -0.22 if style == "braided" else -0.18)]
        for seg in sdf.bent_cone(pts, [h.r(0.035), h.r(0.022), h.r(0.008)]):
            s.union(seg, k=h.r(0.02), tag=name)
    return s


def eyebrows(J, p, name="hair"):
    if p["snout"] > 0:
        return None
    h = HeadSpace(p, J)
    s = Shape()
    for sx in (-1, 1):
        pts = [h(sx * 0.014, -0.093, 0.019), h(sx * 0.033, -0.093, 0.025), h(sx * 0.05, -0.086, 0.019)]
        for seg in sdf.bent_cone(pts, [h.r(0.0034), h.r(0.0032), h.r(0.0018)]):
            s.union(seg, k=h.r(0.002), tag=name)
    return s
