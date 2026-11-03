"""Body (everything below the jaw) – lofted base + individually shaped muscle forms.

The head/face lives in anatomy.py and is intentionally not touched here.
All values are fractions of body height H.
"""
from __future__ import annotations

import numpy as np

import anatomy
import sdf
from humanoid import _foot, _hand, _race_extras, v3
from sdf import F, Shape, capsule, ellipsoid, frame_z, loft, sphere

# ------------------------------------------------------------------ torso --
TORSO_Z = (0.455, 0.865)
T_W_M = [(0.455, 0.062), (0.47, 0.094), (0.5, 0.103), (0.52, 0.1), (0.54, 0.095), (0.56, 0.09), (0.58, 0.087), (0.6, 0.086), (0.62, 0.086), (0.64, 0.087), (0.66, 0.087), (0.68, 0.09), (0.7, 0.094), (0.72, 0.105), (0.74, 0.108), (0.76, 0.108), (0.78, 0.106), (0.8, 0.106), (0.815, 0.104), (0.83, 0.094), (0.845, 0.076), (0.855, 0.06), (0.865, 0.042)]
T_W_F = [(0.455, 0.066), (0.47, 0.102), (0.5, 0.112), (0.52, 0.108), (0.54, 0.098), (0.56, 0.089), (0.58, 0.082), (0.6, 0.076), (0.62, 0.074), (0.64, 0.075), (0.66, 0.078), (0.68, 0.083), (0.7, 0.087), (0.72, 0.09), (0.74, 0.092), (0.76, 0.092), (0.78, 0.088), (0.8, 0.092), (0.815, 0.09), (0.83, 0.081), (0.845, 0.065), (0.855, 0.052), (0.865, 0.036)]
T_F_M = [(0.455, 0.03), (0.47, 0.048), (0.5, 0.054), (0.53, 0.054), (0.56, 0.055), (0.58, 0.055), (0.6, 0.054), (0.62, 0.054), (0.64, 0.056), (0.66, 0.06), (0.68, 0.067), (0.7, 0.073), (0.72, 0.075), (0.74, 0.074), (0.76, 0.07), (0.78, 0.064), (0.8, 0.058), (0.815, 0.05), (0.83, 0.04), (0.845, 0.026), (0.855, 0.018), (0.865, 0.01)]
T_F_F = [(0.455, 0.03), (0.47, 0.045), (0.5, 0.051), (0.53, 0.052), (0.56, 0.052), (0.58, 0.051), (0.6, 0.05), (0.62, 0.049), (0.64, 0.05), (0.66, 0.053), (0.68, 0.058), (0.7, 0.062), (0.72, 0.064), (0.74, 0.064), (0.76, 0.062), (0.78, 0.058), (0.8, 0.051), (0.815, 0.043), (0.83, 0.036), (0.845, 0.025), (0.855, 0.017), (0.865, 0.01)]
T_B_M = [(0.455, 0.04), (0.47, 0.058), (0.5, 0.066), (0.52, 0.065), (0.54, 0.06), (0.56, 0.053), (0.58, 0.047), (0.6, 0.045), (0.62, 0.046), (0.64, 0.049), (0.66, 0.053), (0.68, 0.057), (0.7, 0.06), (0.72, 0.062), (0.74, 0.063), (0.76, 0.063), (0.78, 0.061), (0.8, 0.056), (0.815, 0.05), (0.83, 0.044), (0.845, 0.042), (0.855, 0.041), (0.865, 0.036)]
T_B_F = [(0.455, 0.042), (0.47, 0.062), (0.5, 0.072), (0.52, 0.071), (0.54, 0.064), (0.56, 0.055), (0.58, 0.046), (0.6, 0.043), (0.62, 0.043), (0.64, 0.045), (0.66, 0.049), (0.68, 0.052), (0.7, 0.054), (0.72, 0.056), (0.74, 0.057), (0.76, 0.057), (0.78, 0.055), (0.8, 0.051), (0.815, 0.046), (0.83, 0.041), (0.845, 0.039), (0.855, 0.038), (0.865, 0.033)]
T_N = [(0.455, 2), (0.52, 2.1), (0.62, 2.1), (0.72, 2.4), (0.8, 2.25), (0.845, 2), (0.865, 2)]

# ------------------------------------------------------------------- limbs --
# t: proximal joint (0) -> distal joint (1)
UPPER_ARM = dict(wo=[(0, 0.031), (0.25, 0.0305), (0.5, 0.0285), (0.75, 0.0258), (1, 0.0235)],
                 wi=[(0, 0.029), (0.3, 0.0275), (0.6, 0.0255), (1, 0.0228)],
                 f=[(0, 0.029), (0.3, 0.030), (0.55, 0.0318), (0.8, 0.0268), (1, 0.021)],
                 b=[(0, 0.032), (0.3, 0.0335), (0.6, 0.0295), (0.85, 0.0248), (1, 0.0228)])
DELTOID = dict(wo=[(0, 0.024), (0.28, 0.0405), (0.6, 0.034), (0.85, 0.022), (1, 0.011)],
               wi=[(0, 0.02), (0.4, 0.024), (1, 0.01)],
               f=[(0, 0.028), (0.32, 0.0365), (0.7, 0.026), (1, 0.013)],
               b=[(0, 0.029), (0.32, 0.0375), (0.7, 0.027), (1, 0.013)])
FOREARM = dict(wo=[(0, 0.0245), (0.2, 0.027), (0.45, 0.0228), (0.7, 0.017), (1, 0.0125)],
               wi=[(0, 0.025), (0.2, 0.0262), (0.45, 0.0215), (0.7, 0.0162), (1, 0.012)],
               f=[(0, 0.0255), (0.2, 0.0285), (0.45, 0.0245), (0.7, 0.019), (1, 0.0165)],
               b=[(0, 0.0265), (0.2, 0.0262), (0.5, 0.0215), (1, 0.016)])
THIGH = dict(wo=[(0, 0.04), (0.1, 0.047), (0.22, 0.0535), (0.35, 0.0525), (0.6, 0.0465), (0.85, 0.0352), (1, 0.0295)],
             wi=[(0, 0.045), (0.15, 0.046), (0.4, 0.0425), (0.7, 0.0358), (0.84, 0.0355), (1, 0.0295)],
             f=[(0, 0.047), (0.3, 0.0495), (0.6, 0.0465), (0.85, 0.0368), (1, 0.0285)],
             b=[(0, 0.045), (0.12, 0.051), (0.3, 0.0495), (0.6, 0.0435), (0.9, 0.032), (1, 0.029)])
SHIN = dict(wo=[(0, 0.0305), (0.1, 0.0305), (0.25, 0.0328), (0.4, 0.0305), (0.6, 0.0238), (0.8, 0.0188), (1, 0.0165)],
            wi=[(0, 0.031), (0.1, 0.031), (0.22, 0.0352), (0.38, 0.0325), (0.6, 0.0238), (0.8, 0.0188), (1, 0.017)],
            f=[(0, 0.0295), (0.12, 0.027), (0.4, 0.0238), (0.7, 0.020), (1, 0.019)],
            b=[(0, 0.027), (0.15, 0.0375), (0.28, 0.0425), (0.45, 0.0355), (0.65, 0.0238), (0.85, 0.019), (1, 0.021)])


def _mix(m, f, fem):
    """Blend two (z, v) tables that may have different keys (interpolated on the union)."""
    zs = sorted(set(z for z, _ in m) | set(z for z, _ in f))
    mz, mv = zip(*m)
    fz, fv = zip(*f)
    return [(z, float(np.interp(z, mz, mv) + (np.interp(z, fz, fv) - np.interp(z, mz, mv)) * fem)) for z in zs]


def _sc(keys, s):
    return [(t, v * s) for t, v in keys]


def _zk(pairs, z0, z1):
    return [((z - z0) / (z1 - z0), v) for z, v in pairs]


def _limb(s, a, b, prof, H, outer, k, sw=1.0, sd=1.0, name="limb", ext_a=0.0, ext_b=0.0, cap_round=0.7):
    s.union(loft(a, b, _sc(prof["wo"], H * sw), _sc(prof["wi"], H * sw), _sc(prof["f"], H * sd), _sc(prof["b"], H * sd),
                 outer_dir=outer, ext_a=ext_a * H, ext_b=ext_b * H, cap_round=cap_round), k=k, tag="skin", name=name)


def torso_surface_front(z, x, fem, H):
    """Front surface depth (positive) of the torso loft at height z/H and lateral x/H."""
    mw = _mix(T_W_M, T_W_F, fem)
    mf = _mix(T_F_M, T_F_F, fem)
    w = np.interp(z, [p[0] for p in mw], [p[1] for p in mw])
    f = np.interp(z, [p[0] for p in mf], [p[1] for p in mf])
    n = np.interp(z, [p[0] for p in T_N], [p[1] for p in T_N])
    return f * max(0.0, 1 - (abs(x) / w) ** n) ** (1 / n)


def build_body(p, J) -> Shape:
    s = Shape()
    H = p["height"]
    fem, mus, fat = p["fem"], p["muscle"], p["fat"]
    thick = p["limb_thick"]
    m_arm = (1 + 0.3 * (mus - 0.5)) * (1 + 0.25 * (fat - 0.2)) * thick
    m_leg = (1 + 0.2 * (mus - 0.5)) * (1 + 0.3 * (fat - 0.2)) * thick
    torso_w = 1 + 0.35 * (fat - 0.2)
    belly = max(0.0, fat - 0.3) * 0.25

    # ---------------- torso ----------------
    z0, z1 = TORSO_Z
    shp, hipm = p["shoulder"], p["hip"]

    def wscale(z):
        u = np.clip((z - 0.56) / 0.2, 0, 1)
        return (hipm * (1 - u) + shp * u) * torso_w

    W = [(z, v * H * wscale(z)) for z, v in _mix(T_W_M, T_W_F, fem)]
    Fr = [(z, v * H * (1 + 0.3 * (fat - 0.2) + (belly if 0.52 < z < 0.66 else 0))) for z, v in _mix(T_F_M, T_F_F, fem)]
    Bk = [(z, v * H * (1 + 0.25 * (fat - 0.2))) for z, v in _mix(T_B_M, T_B_F, fem)]
    s.union(loft(v3(0, 0, z0 * H), v3(0, 0.004 * H, z1 * H), _zk(W, z0, z1), _zk(W, z0, z1), _zk(Fr, z0, z1), _zk(Bk, z0, z1),
                 n_front=_zk(T_N, z0, z1), n_back=_zk(T_N, z0, z1)), tag="skin", name="torso")

    # ---------------- neck ----------------
    hh = J["_head_h"]
    nw = p["neck"] * (1 - 0.1 * fem) * (1 + 0.2 * (mus - 0.5))
    neck_base = v3(0, 0.008 * H, 0.825 * H)
    neck_top = v3(0, J["head"][1] + 0.07 * hh, J["head_top"][2] - 0.9 * hh)
    s.union(loft(neck_base, neck_top, [(0, 0.036 * H * nw), (0.5, 0.033 * H * nw), (1, 0.033 * H * nw)],
                 [(0, 0.036 * H * nw), (0.5, 0.033 * H * nw), (1, 0.033 * H * nw)],
                 [(0, 0.028 * H * nw), (1, 0.026 * H * nw)], [(0, 0.034 * H * nw), (0.6, 0.031 * H * nw), (1, 0.033 * H * nw)]),
            k=0.012 * H, tag="skin", name="neck")
    for sx in (-1, 1):   # sternocleidomastoid: behind the ear -> sternal notch
        s.union(capsule(neck_top + v3(sx * 0.024 * H, 0.008 * H, 0.0), v3(sx * 0.008 * H, -0.022 * H, 0.828 * H), 0.0048 * H * nw),
                k=0.008 * H, tag="skin", name="scm")
    if fem < 0.5:
        s.union(ellipsoid(neck_base + (neck_top - neck_base) * 0.5 + v3(0, -0.026 * H * nw, 0), [0.006 * H, 0.006 * H, 0.009 * H]),
                k=0.008 * H, tag="skin", name="adams_apple")

    _muscles_torso(s, p, J)
    _muscles_limbs(s, p, J)

    # ---------------- arms ----------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        s.side = side
        sh, el, wr = J[f"upper_arm.{side}"], J[f"forearm.{side}"], J[f"hand.{side}"]
        up = (el - sh) / np.linalg.norm(el - sh)
        outer = v3(sx, 0, 0) + v3(0, 0, 1.2)
        aw = m_arm * (1 - 0.15 * fem)
        _limb(s, sh, el, UPPER_ARM, H, outer, 0.02 * H, aw, aw, "upper_arm", ext_a=0.01, ext_b=0.02)
        dm = (1 + 0.35 * (mus - 0.5)) * (1 - 0.2 * fem) * thick
        _limb(s, sh - up * 0.022 * H + v3(sx * 0.004 * H, 0, -0.004 * H), sh + up * 0.105 * H, DELTOID, H, outer, 0.016 * H, dm, dm, "deltoid",
              cap_round=0.95)
        # deltoid / biceps border and the deltoid's front edge
        s.displace.append(sdf.stroke(sh + up * 0.06 * H + v3(0, -0.03 * H, -0.004 * H), sh + up * 0.105 * H + v3(0, -0.02 * H, 0), 0.006 * H, -0.0015 * H))
        fo = wr + (wr - el) / np.linalg.norm(wr - el) * 0.006 * H
        _limb(s, el, fo, FOREARM, H, outer, 0.005 * H, aw * (1 - 0.04 * fem), aw * (1 - 0.04 * fem), "forearm", ext_a=0.02, ext_b=0.01)
        s.union(sphere(el + v3(0, 0.021 * H * aw, 0.002 * H), 0.009 * H), k=0.01 * H, tag="skin", name="olecranon")
        _hand(s, p, J, side)

    # ---------------- legs ----------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        s.side = side
        hj, kn, an = J[f"thigh.{side}"], J[f"shin.{side}"], J[f"foot.{side}"]
        outer = v3(sx, 0, 0)
        lw = m_leg * (1 + 0.07 * fem) * (hipm ** 0.5)
        top = hj + v3(sx * 0.0 * H, 0.004 * H, 0.012 * H)
        tk = float(np.linalg.norm(kn - top) / (np.linalg.norm(kn - top) + np.linalg.norm(an - kn)))
        leg = {}
        for key in ("wo", "wi", "f", "b"):
            th = [(t * tk, v * lw) for t, v in THIGH[key]]
            sh_ = [(tk + t * (1 - tk), v * m_leg * (1 - 0.06 * fem)) for t, v in SHIN[key]]
            merged = dict(th)
            for t, v in sh_:
                merged[t] = (merged[t] + v) * 0.5 if t in merged else v
            leg[key] = sorted(merged.items())
        _limb(s, top, an, leg, H, outer, 0.024 * H, 1.0, 1.0, "leg", ext_b=0.012)
        s.displace.append(sdf.brush(kn + v3(0, -0.028 * H, 0.008 * H), [0.011 * H, 0.012 * H, 0.012 * H], 0.0028 * H))
        s.displace.append(sdf.stroke(kn + v3(0, -0.028 * H, -0.006 * H), kn + v3(0, -0.025 * H, -0.03 * H), 0.006 * H, 0.0012 * H))
        for ox in (-1, 1):
            s.union(sphere(an + v3(ox * 0.0145 * H, 0.002 * H, 0.004 * H), 0.0068 * H), k=0.007 * H, tag="skin", name="malleolus")
        _foot(s, p, J, side)
    s.side = ""

    anatomy._head(s, p, J)
    _race_extras(s, p, J)
    return s


def _muscles_torso(s, p, J):
    """Surface anatomy sculpted with displacement brushes (no stuck-on volumes)."""
    H = p["height"]
    fem, mus = p["fem"], p["muscle"]
    md = 0.55 + 0.9 * mus
    D = s.displace
    br = sdf.brush
    st = sdf.stroke

    def front(z, x):
        return -torso_surface_front(z, x, fem, H) * H

    if fem < 0.5:
        for sx in (-1, 1):
            # pectoralis major: broad buildup, strongest along the lower-outer mass
            D.append(br(v3(sx * 0.055 * H, front(0.728, 0.055), 0.728 * H), [0.046 * H, 0.03 * H, 0.032 * H], 0.0115 * H * md))
            D.append(br(v3(sx * 0.07 * H, front(0.715, 0.07), 0.715 * H), [0.03 * H, 0.03 * H, 0.02 * H], 0.005 * H * md))
            # lower pec border crease, sweeping up toward the armpit
            D.append(st(v3(sx * 0.012 * H, front(0.699, 0.012) + 0.004 * H, 0.699 * H), v3(sx * 0.085 * H, front(0.72, 0.085) + 0.004 * H, 0.722 * H),
                        0.0065 * H, -0.0035 * H * md))
            # oblique / serratus softness on the flank
            D.append(br(v3(sx * 0.082 * H, front(0.63, 0.08) * 0.6, 0.63 * H), [0.02 * H, 0.03 * H, 0.05 * H], 0.0018 * H * md))
        # sternum valley, linea alba and two tendinous lines (very soft)
        D.append(st(v3(0, front(0.76, 0) , 0.765 * H), v3(0, front(0.69, 0), 0.69 * H), 0.009 * H, -0.0012 * H * md))
        for sx in (-1, 1):   # inguinal "V" lines: iliac crest -> pubis
            D.append(st(v3(sx * 0.085 * H, front(0.545, 0.085) * 0.95, 0.545 * H), v3(sx * 0.02 * H, front(0.475, 0.02), 0.475 * H), 0.008 * H, -0.0022 * H * md))
        D.append(st(v3(0, front(0.67, 0), 0.67 * H), v3(0, front(0.565, 0), 0.565 * H), 0.0045 * H, -0.0011 * H * md))
        for z in (0.633, 0.598):
            D.append(st(v3(-0.034 * H, front(z, 0.034), z * H), v3(0.034 * H, front(z, 0.034), z * H), 0.004 * H, -0.0007 * H * md))
        for sx in (-1, 1):
            D.append(br(v3(sx * 0.02 * H, front(0.615, 0.02), 0.615 * H), [0.016 * H, 0.02 * H, 0.05 * H], 0.0014 * H * md))
    else:
        for sx in (-1, 1):
            bs = 0.9 + 0.6 * p["fat"]
            z, x = 0.712, 0.047
            surf = torso_surface_front(z, x, fem, H)
            s.union(ellipsoid(v3(sx * x * H, -(surf - 0.022 * bs) * H, z * H), [0.036 * H * bs, 0.03 * H * bs, 0.034 * H * bs], sdf.euler(14, 0, sx * -12)),
                    k=0.016 * H, tag="skin", name="breast")
    for sx in (-1, 1):
        # clavicle ridge + the hollow beneath it
        D.append(st(v3(sx * 0.014 * H, -0.028 * H, 0.818 * H), v3(sx * 0.088 * H, -0.02 * H, 0.81 * H), 0.0055 * H, 0.0022 * H))
        D.append(st(v3(sx * 0.03 * H, -0.04 * H, 0.796 * H), v3(sx * 0.08 * H, -0.035 * H, 0.792 * H), 0.008 * H, -0.0012 * H))
        # trapezius: slope from neck to shoulder, mass on the upper back
        D.append(st(v3(sx * 0.026 * H, 0.022 * H, 0.858 * H), v3(sx * 0.095 * H, 0.008 * H, 0.812 * H), 0.012 * H, 0.0042 * H * md))
        D.append(br(v3(sx * 0.035 * H, 0.06 * H, 0.77 * H), [0.035 * H, 0.03 * H, 0.05 * H], 0.005 * H * md))
        # latissimus: soft widening of the back below the armpit
        D.append(br(v3(sx * 0.075 * H, 0.04 * H, 0.715 * H), [0.026 * H, 0.03 * H, 0.06 * H], 0.0055 * H * md))
        # scapula edge (very soft)
        D.append(st(v3(sx * 0.03 * H, 0.064 * H, 0.775 * H), v3(sx * 0.05 * H, 0.06 * H, 0.705 * H), 0.008 * H, -0.0022 * H * md))
        # erector columns -> spinal groove
        D.append(st(v3(sx * 0.016 * H, 0.05 * H, 0.66 * H), v3(sx * 0.016 * H, 0.05 * H, 0.56 * H), 0.012 * H, 0.0045 * H * md))
    D.append(st(v3(0, 0.07 * H, 0.78 * H), v3(0, 0.05 * H, 0.55 * H), 0.005 * H, -0.0015 * H))
    gl = 1 + 0.25 * fem + 0.3 * (p["fat"] - 0.2)
    for sx in (-1, 1):
        # gluteus medius / tensor fasciae latae: the upper outer hip mass
        D.append(br(v3(sx * 0.097 * H, 0.005 * H, 0.515 * H), [0.014 * H, 0.03 * H, 0.03 * H], 0.0032 * H))
        # iliac crest: soft ridge where the waist meets the hip
        D.append(st(v3(sx * 0.09 * H, 0.03 * H, 0.552 * H), v3(sx * 0.082 * H, -0.035 * H, 0.548 * H), 0.007 * H, 0.0014 * H))
        # hip crease where the thigh leaves the pelvis
        D.append(st(v3(sx * 0.09 * H, -0.025 * H, 0.488 * H), v3(sx * 0.03 * H, -0.045 * H, 0.462 * H), 0.007 * H, -0.0018 * H))
    # lower abdomen / pubic mound
    D.append(br(v3(0, -0.05 * H, 0.478 * H), [0.03 * H, 0.02 * H, 0.018 * H], 0.0035 * H))
    for sx in (-1, 1):
        D.append(br(v3(sx * 0.05 * H, 0.06 * H, 0.488 * H), [0.04 * H, 0.03 * H, 0.05 * H], 0.011 * H * gl))
        D.append(br(v3(sx * 0.075 * H, 0.04 * H, 0.515 * H), [0.025 * H, 0.03 * H, 0.03 * H], 0.004 * H * gl))
        # gluteal fold
        D.append(st(v3(sx * 0.02 * H, 0.06 * H, 0.458 * H), v3(sx * 0.085 * H, 0.04 * H, 0.462 * H), 0.008 * H, -0.0018 * H))
    D.append(st(v3(0, 0.09 * H, 0.54 * H), v3(0, 0.07 * H, 0.455 * H), 0.006 * H, -0.006 * H))   # cleft
    s.sub(ellipsoid(v3(0, -0.056 * H * (1 + 0.3 * (p["fat"] - 0.2)), 0.585 * H), [0.0035 * H, 0.007 * H, 0.0045 * H]), k=0.003 * H, name="navel")


def _muscles_limbs(s, p, J):
    H = p["height"]
    mus = p["muscle"]
    md = 0.55 + 0.9 * mus
    D = s.displace
    st = sdf.stroke
    br = sdf.brush
    for side, sx in (("L", 1.0), ("R", -1.0)):
        hj, kn, an = J[f"thigh.{side}"], J[f"shin.{side}"], J[f"foot.{side}"]
        ax = (kn - hj) / np.linalg.norm(kn - hj)
        # --- quadriceps: rectus femoris (centre), vastus lateralis (outer sweep), vastus medialis (teardrop)
        fwd = v3(0, -1, 0)
        rf_a = hj + ax * 0.04 * H + fwd * 0.045 * H
        rf_b = kn + fwd * 0.03 * H - ax * 0.07 * H
        D.append(st(rf_a, rf_b, 0.02 * H, 0.0034 * H * md))
        D.append(br(hj + ax * 0.13 * H + v3(sx * 0.042 * H, -0.018 * H, 0), [0.022 * H, 0.028 * H, 0.075 * H], 0.0042 * H * md))
        D.append(br(kn - ax * 0.05 * H + v3(-sx * 0.024 * H, -0.02 * H, 0), [0.017 * H, 0.018 * H, 0.03 * H], 0.0042 * H * md))
        # grooves: lateralis | rectus, medialis | rectus (converging on the patella), sartorius diagonal, IT band
        D.append(st(hj + ax * 0.08 * H + v3(sx * 0.022 * H, -0.05 * H, 0), kn - ax * 0.045 * H + v3(sx * 0.016 * H, -0.03 * H, 0), 0.009 * H, -0.0012 * H * md))
        D.append(st(hj + ax * 0.12 * H + v3(-sx * 0.012 * H, -0.05 * H, 0), kn - ax * 0.06 * H + v3(-sx * 0.012 * H, -0.032 * H, 0), 0.009 * H, -0.001 * H * md))
        D.append(st(hj + v3(sx * 0.03 * H, -0.045 * H, 0.03 * H), kn + v3(-sx * 0.03 * H, 0.0, 0.02 * H), 0.006 * H, -0.0012 * H * md))
        D.append(st(hj + ax * 0.06 * H + v3(sx * 0.055 * H, 0.0, 0), kn - ax * 0.03 * H + v3(sx * 0.034 * H, 0.004 * H, 0), 0.006 * H, -0.001 * H * md))
        # hamstrings: two soft columns on the back of the thigh
        for ox in (-1, 1):
            D.append(st(hj + ax * 0.05 * H + v3(ox * 0.018 * H, 0.05 * H, 0), kn - ax * 0.06 * H + v3(ox * 0.02 * H, 0.032 * H, 0), 0.016 * H, 0.0018 * H * md))
        # calf: gastrocnemius heads with a soft split
        sk = (an - kn) / np.linalg.norm(an - kn)
        for ox in (-1, 1):
            D.append(br(kn + sk * 0.07 * H + v3(ox * 0.014 * H, 0.03 * H, 0), [0.014 * H, 0.016 * H, 0.04 * H], 0.0025 * H * md))
        D.append(st(kn + sk * 0.04 * H + v3(0, 0.044 * H, 0), kn + sk * 0.11 * H + v3(0, 0.04 * H, 0), 0.005 * H, -0.0012 * H * md))
        # shin bone edge
        D.append(st(kn + sk * 0.04 * H + v3(0, -0.026 * H, 0), an + v3(0, -0.019 * H, 0.03 * H), 0.006 * H, 0.0008 * H))
    for side, sx in (("L", 1.0), ("R", -1.0)):
        sh, el = J[f"upper_arm.{side}"], J[f"forearm.{side}"]
        ax = (el - sh) / np.linalg.norm(el - sh)
        # biceps peak and the groove between biceps and triceps
        D.append(br(sh + ax * 0.11 * H + v3(0, -0.022 * H, 0), [0.016 * H, 0.016 * H, 0.04 * H], 0.0025 * H * md))
        D.append(br(sh + ax * 0.08 * H + v3(0, 0.025 * H, 0), [0.016 * H, 0.016 * H, 0.05 * H], 0.0022 * H * md))
