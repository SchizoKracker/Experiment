"""Parametric humanoid sculptor.

`build_skeleton(params)` produces joint positions (A-pose, Z up, facing -Y).
`build_body(params, skel)` hangs anatomical SDF forms on those joints.
The same skeleton later becomes the Blender armature, so the mesh and rig
always agree.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

import sdf
from sdf import F, Shape, capsule, ellipsoid, frame_z, round_box, round_cone, sphere

# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
BASE = {
    "height": 1.78,
    "fem": 0.0,          # 0 masculine .. 1 feminine body shape
    "muscle": 0.5,
    "fat": 0.2,
    "head_scale": 1.0,   # relative to height-derived head size
    "shoulder": 1.0,
    "hip": 1.0,
    "leg": 1.0,          # leg length multiplier
    "arm": 1.0,
    "limb_thick": 1.0,
    "hand": 1.0,
    "foot": 1.07,
    "neck": 1.0,
    # face
    "jaw": 1.0, "chin": 1.0, "brow": 1.0, "cheek": 1.0, "nose_len": 1.0, "nose_w": 1.0,
    "nose_bridge": 1.0, "lips": 1.0, "eye": 1.0, "eye_tilt": 0.0, "face_long": 1.0,
    "ears": "round", "ear_size": 1.0,
    "tusks": 0.0,
    "horns": None,       # None | "ram" | "straight" | "swept" | "crown"
    "tail": 0.0,         # length in metres (0 none)
    "snout": 0.0,        # dragonborn-style muzzle
    "antennae": 0.0,     # mothkin
    "wick": 0.0,         # tallowborn
    "seed": 1,
}


def P(**kw):
    p = deepcopy(BASE)
    p.update(kw)
    return p


def lerp(a, b, t):
    return a + (b - a) * t


def v3(*a):
    return np.array(a, dtype=F)


# --------------------------------------------------------------------------
# skeleton
# --------------------------------------------------------------------------
ARM_DROP_DEG = 42.0   # A-pose: arm angle below horizontal


def build_skeleton(p) -> dict:
    H = p["height"]
    fem = p["fem"]
    J = {}
    # head size: a 7.5-head human; races override via head_scale
    head_h = H / 7.6 * p["head_scale"]
    leg_len = 0.47 * H * p["leg"]
    hip_z = leg_len + 0.035 * H
    J["root"] = v3(0, 0, 0)
    J["hips"] = v3(0, 0.0, hip_z + 0.02 * H)
    shoulder_z_raw = H - head_h * 1.45
    # torso spans hips -> shoulders; spine joints interpolated with a gentle S-curve
    J["spine"] = v3(0, 0.0, lerp(J["hips"][2], shoulder_z_raw, 0.25))
    J["spine1"] = v3(0, 0.01 * H, lerp(J["hips"][2], shoulder_z_raw, 0.5))
    J["chest"] = v3(0, 0.005 * H, lerp(J["hips"][2], shoulder_z_raw, 0.78))
    J["neck"] = v3(0, 0.012 * H, shoulder_z_raw + 0.02 * H)
    J["head"] = v3(0, 0.002 * H, H - head_h * 0.78)
    J["head_top"] = v3(0, 0.0, H)
    sw = 0.102 * H * p["shoulder"] * lerp(1.0, 0.9, fem)
    hw = 0.052 * H * p["hip"] * lerp(1.0, 1.12, fem)
    arm_len = H * p["arm"]
    up_len, fore_len, hand_len = 0.172 * arm_len, 0.148 * arm_len, 0.105 * H * p["hand"]
    a = math.radians(ARM_DROP_DEG)
    for side, sx in (("L", 1.0), ("R", -1.0)):
        clav = v3(sx * 0.02 * H, 0.0, shoulder_z_raw - 0.005 * H)
        sh = v3(sx * sw, 0.008 * H, shoulder_z_raw - 0.012 * H)
        d_up = v3(sx * math.cos(a), 0.0, -math.sin(a))
        el = sh + d_up * up_len
        # elbow slightly bent forward
        d_fo = v3(sx * math.cos(a) * 0.93, -0.30, -math.sin(a) * 0.93)
        d_fo /= np.linalg.norm(d_fo)
        wr = el + d_fo * fore_len
        J[f"clavicle.{side}"] = clav
        J[f"upper_arm.{side}"] = sh
        J[f"forearm.{side}"] = el
        J[f"hand.{side}"] = wr
        J[f"hand_end.{side}"] = wr + d_fo * hand_len
        # legs
        hj = v3(sx * hw, 0.0, hip_z)
        kn = v3(sx * (hw + 0.016 * H), -0.008 * H, hip_z - leg_len * 0.48)
        an = v3(sx * (hw + 0.022 * H), 0.012 * H, 0.045 * H)
        J[f"thigh.{side}"] = hj
        J[f"shin.{side}"] = kn
        J[f"foot.{side}"] = an
        fl = 0.152 * H * p["foot"]
        J[f"toe.{side}"] = an + v3(sx * 0.008 * H, -fl * 0.68, -0.032 * H)
        J[f"toe_end.{side}"] = an + v3(sx * 0.012 * H, -fl * 0.92, -0.036 * H)
    _build_hand_joints(J, p)
    if p["tail"] > 0:
        base = J["hips"] + v3(0, 0.09 * H * p["hip"], -0.02 * H)
        L = p["tail"]
        pts = [base]
        for i in range(1, 6):
            t = i / 5
            pts.append(base + v3(0.08 * math.sin(t * 2.2) * L, L * (0.35 * t + 0.1 * t * t), -L * (0.8 * t - 0.25 * t * t)))
        for i, q in enumerate(pts):
            J[f"tail{i}"] = q.astype(F)
    J["_head_h"] = head_h
    return J


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def hand_frame(J, side):
    wr = J[f"hand.{side}"]
    he = J[f"hand_end.{side}"]
    fwd = (he - wr) / np.linalg.norm(he - wr)       # along the hand
    sx = 1.0 if side == "L" else -1.0
    palm_n = np.array([-sx, 0.0, 0.0], F)            # palm faces the thigh
    palm_n = palm_n - fwd * np.dot(palm_n, fwd)
    palm_n /= np.linalg.norm(palm_n)
    across = np.cross(fwd, palm_n)                     # thumb side direction
    # thumb points forward (-Y) for both hands
    if across[1] > 0:
        across = -across
    return wr, fwd, palm_n, across


def _build_hand_joints(J, p):
    H = p["height"]
    hs = H * p["hand"] / 1.78
    for side in ("L", "R"):
        wr, fwd, n, ac = hand_frame(J, side)
        palm_len = 0.095 * hs
        spread = [0.0, 0.026, 0.009, -0.009, -0.025]
        lens = {"thumb": (0.042, 0.032, 0.028), "index": (0.044, 0.026, 0.022), "middle": (0.048, 0.030, 0.024),
                "ring": (0.045, 0.028, 0.023), "pinky": (0.036, 0.021, 0.019)}
        for fi, fname in enumerate(FINGERS):
            if fname == "thumb":
                base = wr + fwd * 0.024 * hs + ac * 0.022 * hs + n * 0.012 * hs
                d = (fwd * 0.55 + ac * 0.72 + n * 0.42)
            else:
                base = wr + fwd * (palm_len - 0.004 * hs * abs(fi - 2.2)) + ac * spread[fi] * hs
                d = fwd + ac * spread[fi] * 0.6 + n * 0.12
            d = d / np.linalg.norm(d)
            q = base
            for si in range(3):
                J[f"{fname}{si + 1}.{side}"] = q.astype(F)
                L = lens[fname][si] * hs
                # gentle natural curl toward the palm
                curl = 0.16 + 0.07 * si
                d = d * math.cos(curl) + n * math.sin(curl)
                d /= np.linalg.norm(d)
                q = q + d * L
            J[f"{fname}_end.{side}"] = q.astype(F)


# --------------------------------------------------------------------------
# body
# --------------------------------------------------------------------------
def build_body(p, J) -> Shape:
    s = Shape()
    H = p["height"]
    fem, mus, fat = p["fem"], p["muscle"], p["fat"]
    th = p["limb_thick"] * H / 1.78
    hips, spine, spine1, chest, neck, head = (J[k] for k in ("hips", "spine", "spine1", "chest", "neck", "head"))
    sc = H / 1.78

    # ---------------- torso ----------------
    pel_w = 0.148 * sc * p["hip"] * lerp(1.0, 1.14, fem) * lerp(0.96, 1.1, fat)
    s.union(ellipsoid(hips + v3(0, 0.005 * sc, -0.01 * sc), [pel_w, 0.105 * sc * lerp(1, 1.05, fat), 0.11 * sc]), tag="skin", name="pelvis")
    belly = 0.06 * fat * sc
    waist = 0.122 * sc * lerp(1.0, 0.84, fem) * lerp(0.92, 1.22, fat)
    s.union(ellipsoid(spine + v3(0, 0.0, 0.0), [waist * 1.05, waist * 0.78 + belly * 0.3, 0.13 * sc]), k=0.07 * sc, tag="skin", name="abdomen")
    s.union(ellipsoid(spine1 + v3(0, 0.0, 0.0), [waist * 1.02, waist * 0.8, 0.11 * sc]), k=0.07 * sc, tag="skin", name="waist")
    s.union(ellipsoid(spine + v3(0, -0.035 * sc - belly * 0.6, 0.0), [waist * 0.8 + belly * 0.4, 0.06 * sc + belly, 0.09 * sc]), k=0.06 * sc, tag="skin", name="belly")
    rib_w = 0.158 * sc * p["shoulder"] * lerp(1.0, 0.86, fem) * lerp(0.95, 1.08, mus)
    s.union(ellipsoid(chest + v3(0, 0.004 * sc, -0.035 * sc), [rib_w, 0.105 * sc * lerp(1, 1.06, fat), 0.155 * sc]), k=0.07 * sc, tag="skin", name="ribcage")
    # back / lats
    s.union(ellipsoid(chest + v3(0, 0.045 * sc, -0.05 * sc), [rib_w * 0.95 * lerp(0.95, 1.1, mus), 0.06 * sc, 0.13 * sc]), k=0.05 * sc, tag="skin", name="lats")
    # pectorals (masc) / breasts (fem)
    for sx in (-1, 1):
        pec_c = chest + v3(sx * 0.065 * sc, -0.072 * sc, 0.005 * sc)
        pr = lerp(0.85, 1.25, mus) * (1 - fem * 0.5)
        s.union(ellipsoid(pec_c, [0.068 * sc * pr, 0.03 * sc * pr, 0.052 * sc * pr], sdf.euler(0, 0, sx * -8)), k=0.035 * sc, tag="skin", name="pec")
        if fem > 0.05:
            b = fem * lerp(0.85, 1.25, fat) * sc
            bc = chest + v3(sx * 0.072 * sc, -0.085 * sc - 0.012 * b, -0.04 * sc)
            s.union(ellipsoid(bc, [0.058 * b, 0.05 * b, 0.056 * b], sdf.euler(15, 0, sx * -12)), k=0.035 * sc, tag="skin", name="breast")
    # trapezius
    for sx in (-1, 1):
        shp = J["upper_arm.L" if sx > 0 else "upper_arm.R"]
        s.union(round_cone(neck + v3(sx * 0.02 * sc, 0.03 * sc, 0.03 * sc), shp + v3(-sx * 0.045 * sc, 0.02 * sc, 0.03 * sc),
                           0.03 * sc * lerp(0.8, 1.25, mus), 0.024 * sc), k=0.06 * sc, tag="skin", name="trap")
        # clavicle ridge
        s.union(capsule(neck + v3(sx * 0.02 * sc, -0.035 * sc, -0.035 * sc), J["upper_arm.L" if sx > 0 else "upper_arm.R"] + v3(-sx * 0.02 * sc, -0.03 * sc, 0.012 * sc), 0.014 * sc),
                k=0.03 * sc, tag="skin", name="clavicle")
    # glutes
    for sx in (-1, 1):
        g = lerp(1.0, 1.2, fem) * lerp(0.95, 1.1, fat) * sc
        s.union(ellipsoid(hips + v3(sx * 0.07 * sc, 0.065 * sc, -0.07 * sc), [0.075 * g, 0.06 * g, 0.085 * g]), k=0.05 * sc, tag="skin", name="glute")
    # neck
    nk = 0.05 * sc * p["neck"] * lerp(1.0, 0.82, fem) * lerp(0.92, 1.18, mus)
    s.union(round_cone(neck + v3(0, 0.0, -0.03 * sc), head + v3(0, 0.012 * sc, -0.07 * sc), nk * 1.18, nk * 0.92), k=0.045 * sc, tag="skin", name="neck")
    for sx in (-1, 1):   # sternocleidomastoid
        s.union(capsule(neck + v3(sx * 0.018 * sc, -0.04 * sc, -0.02 * sc), head + v3(sx * 0.05 * sc, 0.015 * sc, -0.05 * sc), 0.012 * sc), k=0.025 * sc, tag="skin", name="scm")

    # ---------------- arms ----------------
    for side, sx in (("L", 1), ("R", -1)):
        sh, el, wr = J[f"upper_arm.{side}"], J[f"forearm.{side}"], J[f"hand.{side}"]
        r_up = 0.047 * th * lerp(1.0, 0.86, fem) * lerp(0.9, 1.22, mus) * lerp(0.95, 1.18, fat)
        s.union(ellipsoid(sh + (el - sh) * 0.12 + v3(0, 0.0, 0.004 * sc), [0.047 * th * lerp(0.9, 1.2, mus), 0.05 * th * lerp(0.9, 1.15, mus), 0.07 * th], frame_z(el - sh)),
                k=0.04 * sc, tag="skin", name="deltoid")
        s.union(round_cone(sh, el, r_up, r_up * 0.78), k=0.03 * sc, tag="skin", name="upper_arm")
        mid = sh + (el - sh) * 0.55
        ax = frame_z(el - sh)
        s.union(ellipsoid(mid + v3(0, -0.018 * th, 0), [0.03 * th * lerp(0.8, 1.3, mus), 0.03 * th * lerp(0.8, 1.3, mus), 0.065 * th], ax), k=0.025 * sc, tag="skin", name="bicep")
        s.union(ellipsoid(sh + (el - sh) * 0.45 + v3(0, 0.022 * th, 0), [0.032 * th, 0.03 * th * lerp(0.9, 1.25, mus), 0.075 * th], ax), k=0.025 * sc, tag="skin", name="tricep")
        r_fo = 0.039 * th * lerp(1.0, 0.86, fem) * lerp(0.92, 1.18, mus)
        s.union(round_cone(el, wr, r_fo, 0.025 * th), k=0.025 * sc, tag="skin", name="forearm")
        s.union(ellipsoid(el + (wr - el) * 0.25, [0.036 * th * lerp(0.9, 1.2, mus), 0.032 * th, 0.06 * th], frame_z(wr - el)), k=0.025 * sc, tag="skin", name="forearm_mass")
        s.union(sphere(el + v3(0, 0.022 * th, 0.004 * th), 0.02 * th), k=0.02 * sc, tag="skin", name="elbow")
        _hand(s, p, J, side)

    # ---------------- legs ----------------
    for side, sx in (("L", 1), ("R", -1)):
        hj, kn, an = J[f"thigh.{side}"], J[f"shin.{side}"], J[f"foot.{side}"]
        r_th = 0.074 * th * lerp(1.0, 1.06, fem) * lerp(0.92, 1.15, mus) * lerp(0.92, 1.22, fat)
        s.union(round_cone(hj + v3(sx * 0.008 * sc, 0, 0.02 * sc), kn, r_th, 0.047 * th), k=0.035 * sc, tag="skin", name="thigh")
        tax = frame_z(kn - hj)
        s.union(ellipsoid(hj + (kn - hj) * 0.45 + v3(sx * 0.005 * sc, -0.03 * th, 0), [0.05 * th * lerp(0.9, 1.2, mus), 0.045 * th, 0.14 * th], tax), k=0.04 * sc, tag="skin", name="quad")
        s.union(ellipsoid(hj + (kn - hj) * 0.35 + v3(-sx * 0.022 * th, 0.005, 0), [0.036 * th, 0.042 * th, 0.11 * th], tax), k=0.025 * sc, tag="skin", name="adductor")
        s.union(ellipsoid(kn + v3(0, -0.032 * th, 0.008 * th), [0.026 * th, 0.018 * th, 0.03 * th]), k=0.02 * sc, tag="skin", name="patella")
        r_kn = 0.047 * th
        s.union(round_cone(kn, an, r_kn, 0.03 * th), k=0.03 * sc, tag="skin", name="shin")
        s.union(ellipsoid(kn + (an - kn) * 0.27 + v3(0, 0.025 * th, 0), [0.042 * th * lerp(0.9, 1.15, mus), 0.04 * th * lerp(0.9, 1.15, mus), 0.09 * th], frame_z(an - kn)),
                k=0.035 * sc, tag="skin", name="calf")
        # ankle bones
        for ox in (-1, 1):
            s.union(sphere(an + v3(ox * 0.026 * th, 0.004 * th, 0.006 * th), 0.013 * th), k=0.015 * sc, tag="skin", name="malleolus")
        _foot(s, p, J, side)

    _head(s, p, J)
    _race_extras(s, p, J)
    return s


def _hand(s: Shape, p, J, side):
    sc = p["height"] / 1.78 * p["hand"]
    wr, fwd, n, ac = hand_frame(J, side)
    R = np.stack([ac, n, fwd])  # local x=across, y=normal(palm side +), z=along hand
    palm_c = wr + fwd * 0.05 * sc
    s.union(round_box(palm_c, [0.04 * sc, 0.0135 * sc, 0.048 * sc], R, 0.012 * sc), k=0.018 * sc, tag="skin", name="palm")
    s.union(ellipsoid(wr + fwd * 0.035 * sc + ac * 0.02 * sc + n * 0.012 * sc, [0.017 * sc, 0.016 * sc, 0.03 * sc], R), k=0.012 * sc, tag="skin", name="thenar")
    radii = {"thumb": (0.0115, 0.0098, 0.0085), "index": (0.0098, 0.0088, 0.0076), "middle": (0.0102, 0.009, 0.0078),
             "ring": (0.0096, 0.0085, 0.0074), "pinky": (0.0085, 0.0075, 0.0066)}
    for fname in FINGERS:
        pts = [J[f"{fname}{i}.{side}"] for i in (1, 2, 3)] + [J[f"{fname}_end.{side}"]]
        rr = [r * sc for r in radii[fname]]
        for i in range(3):
            s.union(round_cone(pts[i], pts[i + 1], rr[i], rr[min(i + 1, 2)] * (0.92 if i == 2 else 1.0)), k=0.004 * sc, tag="skin", name=f"{fname}")
            # knuckle
            s.union(sphere(pts[i] - n * rr[i] * 0.15, rr[i] * 1.06), k=0.004 * sc, tag="skin", name="knuckle")
        # fingernail on the back of the last phalanx
        tip_dir = pts[3] - pts[2]
        tip_dir /= np.linalg.norm(tip_dir)
        nail_c = pts[2] + tip_dir * np.linalg.norm(pts[3] - pts[2]) * 0.62 - n * rr[2] * 0.62
        s.union(ellipsoid(nail_c, [rr[2] * 0.7, rr[2] * 0.36, rr[2] * 0.95], np.stack([ac, n, tip_dir])), k=0.002 * sc, tag="nail", name="nail")


def _foot(s: Shape, p, J, side):
    sc = p["height"] / 1.78 * p["foot"]
    an, toe, te = J[f"foot.{side}"], J[f"toe.{side}"], J[f"toe_end.{side}"]
    heel = an + v3(0, 0.035 * sc, -0.045 * sc)
    s.union(sphere(heel, 0.032 * sc), k=0.02 * sc, tag="skin", name="heel")
    s.union(round_cone(heel + v3(0, -0.01 * sc, 0.012 * sc), toe + v3(0, 0, 0.004 * sc), 0.033 * sc, 0.026 * sc), k=0.025 * sc, tag="skin", name="foot")
    s.union(round_box((heel + toe) * 0.5 + v3(0, 0, -0.01 * sc), [0.034 * sc, (np.linalg.norm(toe - heel)) * 0.5, 0.016 * sc], None, 0.014 * sc), k=0.02 * sc, tag="skin", name="sole")
    sx = 1 if side == "L" else -1
    for i, (off, r, l) in enumerate([(-0.022, 0.0115, 0.032), (-0.006, 0.0085, 0.03), (0.007, 0.008, 0.027), (0.018, 0.0075, 0.024), (0.028, 0.007, 0.02)]):
        b = toe + v3(-sx * off * sc, 0.004 * sc, 0.0)
        e = b + v3(-sx * off * 0.15 * sc, -l * sc, -0.004 * sc)
        s.union(round_cone(b, e, r * sc, r * 0.9 * sc), k=0.006 * sc, tag="skin", name="toe")


# --------------------------------------------------------------------------
# head
# --------------------------------------------------------------------------
class HeadSpace:
    """Maps canonical head coordinates (human head ~0.23 tall) to world."""

    def __init__(self, p, J):
        self.o = J["head"]
        self.k = J["_head_h"] / 0.23
        self.p = p

    def __call__(self, x, y, z):
        return self.o + v3(x, y, z) * self.k

    def r(self, *radii):
        return [r * self.k for r in radii] if len(radii) > 1 else radii[0] * self.k


def _head(s: Shape, p, J):
    if p["snout"] > 0:
        return _draconic_head(s, p, J)
    h = HeadSpace(p, J)
    fem = p["fem"]
    fl = p["face_long"]
    # cranium
    s.union(ellipsoid(h(0, 0.006, 0.03), h.r(0.075, 0.098, 0.092)), tag="skin", name="cranium")
    # face mass + jaw
    s.union(ellipsoid(h(0, -0.045, -0.04 * fl), h.r(0.051, 0.05, 0.062 * fl)), k=h.r(0.028), tag="skin", name="face")
    jw = p["jaw"] * lerp(1.0, 0.88, fem)
    s.union(ellipsoid(h(0, -0.035, -0.083 * fl), h.r(0.052 * jw, 0.05, 0.034)), k=h.r(0.03), tag="skin", name="jaw")
    for sx in (-1, 1):  # jaw angle
        s.union(sphere(h(sx * 0.047 * jw, -0.008, -0.074 * fl), h.r(0.015 * jw)), k=h.r(0.02), tag="skin", name="jaw_angle")
    ch = p["chin"]
    s.union(ellipsoid(h(0, -0.08, -0.108 * fl), h.r(0.022 * ch * jw, 0.017, 0.019 * ch)), k=h.r(0.022), tag="skin", name="chin")
    # cheekbones
    for sx in (-1, 1):
        s.union(ellipsoid(h(sx * 0.046, -0.07, -0.024), h.r(0.019 * p["cheek"], 0.012, 0.012)), k=h.r(0.018), tag="skin", name="cheekbone")
    # brow ridge
    b = p["brow"]
    s.union(capsule(h(-0.042, -0.086, 0.017), h(0.042, -0.086, 0.017), h.r(0.0105 * b)), k=h.r(0.022), tag="skin", name="brow")
    # eyes: socket, lid shell, almond opening
    er = 0.0128 * p["eye"]
    for sx in (-1, 1):
        ec = h(sx * 0.032, -0.074, 0.002)
        s.sub(sphere(h(sx * 0.032, -0.094, 0.009), h.r(0.0145)), k=h.r(0.01), name="socket")
        s.union(sphere(ec, h.r(er + 0.0022)), k=h.r(0.006), tag="lid", name="lidshell")
    for sx in (-1, 1):
        ec = h(sx * 0.032, -0.074, 0.002)
        R, a = 0.025 * p["eye"], 0.0068 * p["eye"]
        tilt = sdf.euler(0, sx * p["eye_tilt"] * 10.0, 0)
        up = np.array([0, 0, 1], F) @ tilt
        c1 = ec + up * h.r(-(R - a))
        c2 = ec + up * h.r(R - a)
        f1 = sphere(c1, h.r(R))
        f2 = sphere(c2, h.r(R))
        front = sdf.half_space(ec + v3(0, -h.r(0.004), 0), [0, 1, 0])

        def almond(q, f1=f1[0], f2=f2[0], fr=front[0]):
            return sdf.smax(sdf.smax(f1(q), f2(q), 0.0), fr(q), 0.0)
        s.sub((almond, ec - h.r(0.03), ec + h.r(0.03)), k=h.r(0.0015), name="eye_opening")
    # nose
    nl, nw, nb = p["nose_len"], p["nose_w"], p["nose_bridge"]
    s.union(round_cone(h(0, -0.084, 0.004), h(0, -0.103 - 0.006 * (nl - 1), -0.034 * nl), h.r(0.0075 * nb), h.r(0.0105 * nw)), k=h.r(0.012), tag="skin", name="nose_bridge")
    tip = h(0, -0.106 - 0.008 * (nl - 1), -0.039 * nl)
    s.union(sphere(tip, h.r(0.0118 * nw)), k=h.r(0.008), tag="skin", name="nose_tip")
    for sx in (-1, 1):
        s.union(sphere(h(sx * 0.0125 * nw, -0.096, -0.043 * nl), h.r(0.0085 * nw)), k=h.r(0.007), tag="skin", name="ala")
        s.sub(ellipsoid(h(sx * 0.0068 * nw, -0.101, -0.0505 * nl), h.r(0.0042, 0.0055, 0.0026)), k=h.r(0.002), name="nostril")
    # lips & mouth
    lp = p["lips"] * lerp(1.0, 1.15, fem)
    s.union(ellipsoid(h(0, -0.0905, -0.064 * fl), h.r(0.021, 0.0085, 0.0055 * lp)), k=h.r(0.006), tag="lips", name="upper_lip", tag_eps=h.r(0.0012))
    s.union(ellipsoid(h(0, -0.089, -0.0745 * fl), h.r(0.0185, 0.0085, 0.0062 * lp)), k=h.r(0.006), tag="lips", name="lower_lip", tag_eps=h.r(0.0012))
    s.union(ellipsoid(h(0, -0.092, -0.055 * fl), h.r(0.005, 0.005, 0.006)), k=h.r(0.007), tag="skin", name="philtrum")
    s.sub(ellipsoid(h(0, -0.099, -0.0695 * fl), h.r(0.0215, 0.012, 0.0011)), k=h.r(0.0012), name="mouth_line")
    # ears
    _ears(s, p, h)
    if p["tusks"] > 0:
        for sx in (-1, 1):
            base = h(sx * 0.016, -0.09, -0.074 * fl)
            tipp = h(sx * 0.021, -0.096, -0.048 * fl + 0.012 * p["tusks"])
            s.union(round_cone(base, tipp, h.r(0.0052 * p["tusks"]), h.r(0.0012)), k=h.r(0.001), tag="tusk", name="tusk")


def _ears(s, p, h):
    kind = p["ears"]
    if kind == "none":
        return
    es = p["ear_size"]
    for sx in (-1, 1):
        c = h(sx * 0.073, 0.004, -0.008)
        rot = sdf.euler(0, 0, sx * 12) @ sdf.euler(-12, 0, 0)
        s.union(ellipsoid(c, h.r(0.009, 0.019 * es, 0.03 * es), rot), k=h.r(0.006), tag="skin", name="ear")
        s.sub(ellipsoid(c + v3(sx * h.r(0.007), -h.r(0.002), h.r(0.003)), h.r(0.0055, 0.012 * es, 0.019 * es), rot), k=h.r(0.003), name="ear_bowl")
        if kind in ("pointed", "long"):
            ln = 0.045 if kind == "long" else 0.024
            a = c + v3(0, h.r(0.004), h.r(0.012 * es))
            b = c + v3(sx * h.r(ln * 0.55), h.r(ln * 0.7), h.r(ln * 0.95)) * 1.0
            s.union(round_cone(a, b, h.r(0.0105 * es), h.r(0.0015)), k=h.r(0.008), tag="skin", name="ear_tip")


def _draconic_head(s: Shape, p, J):
    h = HeadSpace(p, J)
    sn = p["snout"]
    s.union(ellipsoid(h(0, 0.02, 0.025), h.r(0.07, 0.09, 0.075)), tag="skin", name="cranium")
    s.union(round_cone(h(0, -0.03, -0.01), h(0, -0.13 * sn, -0.035), h.r(0.055), h.r(0.03)), k=h.r(0.04), tag="skin", name="muzzle")
    s.union(round_cone(h(0, 0.0, -0.06), h(0, -0.12 * sn, -0.058), h.r(0.045), h.r(0.024)), k=h.r(0.03), tag="skin", name="lower_jaw")
    s.sub(round_cone(h(0, -0.04, -0.047), h(0, -0.15 * sn, -0.047), h.r(0.003), h.r(0.002)), k=h.r(0.002), name="mouth")
    for sx in (-1, 1):
        s.union(capsule(h(sx * 0.035, -0.06, 0.02), h(sx * 0.03, -0.11 * sn, -0.015), h.r(0.009)), k=h.r(0.02), tag="skin", name="snout_ridge")
        s.sub(sphere(h(sx * 0.012, -0.135 * sn, -0.022), h.r(0.0045)), k=h.r(0.002), name="nostril")
        ec = h(sx * 0.046, -0.05, 0.012)
        s.sub(sphere(h(sx * 0.05, -0.064, 0.014), h.r(0.017)), k=h.r(0.01), name="socket")
        s.union(sphere(ec, h.r(0.0145)), k=h.r(0.005), tag="lid", name="lidshell")
        s.sub(ellipsoid(ec + v3(sx * h.r(0.006), -h.r(0.012), 0), h.r(0.012, 0.01, 0.006), sdf.euler(0, sx * 20, sx * -25)), k=h.r(0.002), name="eye_opening")
        s.union(capsule(h(sx * 0.04, -0.06, 0.03), h(sx * 0.045, -0.035, 0.034), h.r(0.012)), k=h.r(0.012), tag="skin", name="brow_scale")
        # frills / cheek spines
        for i in range(3):
            a = h(sx * 0.06, 0.03 + 0.02 * i, -0.02 - 0.012 * i)
            b = a + v3(sx * h.r(0.03), h.r(0.04), h.r(-0.01))
            s.union(round_cone(a, b, h.r(0.008), h.r(0.0015)), k=h.r(0.006), tag="horn", name="frill")


def _race_extras(s: Shape, p, J):
    h = HeadSpace(p, J)
    horns = p["horns"]
    if horns:
        for sx in (-1, 1):
            if horns == "ram":
                pts, rad = [], []
                for i in range(9):
                    t = i / 8
                    ang = t * 4.2
                    r = 0.045 * (1 - 0.35 * t)
                    pts.append(h(sx * (0.045 + 0.03 * t), 0.02 - math.sin(ang) * r * 1.2 + 0.03 * t, 0.07 + math.cos(ang) * r - 0.02))
                    rad.append(h.r(0.016 * (1 - 0.78 * t) + 0.002))
            elif horns == "straight":
                pts = [h(sx * 0.04, -0.035, 0.085), h(sx * 0.055, -0.02, 0.13), h(sx * 0.06, 0.0, 0.17)]
                rad = [h.r(0.012), h.r(0.008), h.r(0.0015)]
            elif horns == "swept":
                pts = [h(sx * 0.042, -0.03, 0.08), h(sx * 0.055, 0.03, 0.11), h(sx * 0.06, 0.1, 0.12), h(sx * 0.058, 0.16, 0.105)]
                rad = [h.r(0.0125), h.r(0.01), h.r(0.006), h.r(0.0012)]
            else:  # crown – several small nubs
                pts = None
                for i in range(3):
                    a = h(sx * (0.025 + 0.018 * i), -0.04 + 0.03 * i, 0.09 - 0.006 * i)
                    b = a + v3(sx * h.r(0.006), h.r(0.008), h.r(0.03 - 0.006 * i))
                    s.union(round_cone(a, b, h.r(0.008), h.r(0.0015)), k=h.r(0.004), tag="horn", name="horn")
            if pts is not None:
                for seg in sdf.bent_cone(pts, rad):
                    s.union(seg, k=h.r(0.004), tag="horn", name="horn")
    if p["tail"] > 0:
        pts = [J[f"tail{i}"] for i in range(6)]
        r0 = 0.035 * p["height"] / 1.78
        rad = [r0 * (1 - i / 5 * 0.85) for i in range(6)]
        for i, seg in enumerate(sdf.bent_cone(pts, rad)):
            s.union(seg, k=0.03 if i == 0 else 0.008, tag="skin", name="tail")
        tip = pts[-1]
        d = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
        s.union(ellipsoid(tip + d * 0.02, [0.018, 0.006, 0.03], frame_z(d)), k=0.004, tag="skin", name="tail_spade")
    if p["antennae"] > 0:
        for sx in (-1, 1):
            pts = [h(sx * 0.02, -0.06, 0.07), h(sx * 0.04, -0.075, 0.12), h(sx * 0.075, -0.06, 0.17), h(sx * 0.1, -0.03, 0.2)]
            rad = [h.r(0.004), h.r(0.0032), h.r(0.0025), h.r(0.002)]
            for seg in sdf.bent_cone(pts, rad):
                s.union(seg, k=h.r(0.002), tag="horn", name="antenna")
            # feathery plume: a flattened ellipsoid along the antenna
            mid = (pts[1] + pts[3]) * 0.5
            s.union(ellipsoid(mid, h.r(0.012, 0.0025, 0.05), frame_z(pts[3] - pts[1])), k=h.r(0.002), tag="horn", name="plume")


def eye_shape(p, J) -> Shape:
    """Separate eyeball mesh (gets its own material)."""
    s = Shape()
    h = HeadSpace(p, J)
    if p["snout"] > 0:
        for sx in (-1, 1):
            s.union(sphere(h(sx * 0.046, -0.05, 0.012), h.r(0.0125)), tag="eye")
        return s
    er = 0.0128 * p["eye"]
    for sx in (-1, 1):
        s.union(sphere(h(sx * 0.032, -0.074, 0.002), h.r(er)), tag="eye")
    return s
