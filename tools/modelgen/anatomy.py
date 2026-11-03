"""Reference-calibrated anatomy built from continuous lofted surfaces.

Every major body part is ONE swept cross-section surface whose half-extents
come from measured reference silhouettes (front widths) and standard
anthropometric depth ratios.  Values are fractions of body height (H) or of
head height (top of skull -> chin) for the head.  This replaces blob unions:
no lumps, clean silhouettes, and the edge-flow mirrors real body topology.
"""
from __future__ import annotations

import json
import os

import numpy as np

import sdf
from humanoid import HeadSpace, _foot, _hand, _race_extras, hand_frame, v3
from sdf import F, Shape, capsule, ellipsoid, loft, round_cone, sphere

HERE = os.path.dirname(os.path.abspath(__file__))
REF_HEAD = json.load(open(os.path.join(HERE, "ref_head.json")))


def _mix(male, female, fem):
    """Blend two keyframe lists (same t positions) by `fem`."""
    return [(tm, vm + (vf - vm) * fem) for (tm, vm), (_, vf) in zip(male, female)]


def _scale(keys, s):
    return [(t, v * s) for t, v in keys]


def _zkeys(pairs, z0, z1):
    """[(z/H, value)] -> [(t, value)] for a loft running z0 -> z1."""
    return [((z - z0) / (z1 - z0), v) for z, v in pairs]


# ---------------------------------------------------------------- torso ----
# lateral half-width (front silhouette), front & back extents; all / H
TORSO_Z = (0.455, 0.848)
T_W_M = [(0.455, 0.070), (0.47, 0.098), (0.50, 0.106), (0.52, 0.103), (0.54, 0.097), (0.56, 0.091), (0.58, 0.087),
         (0.60, 0.087), (0.62, 0.087), (0.64, 0.087), (0.66, 0.090), (0.68, 0.094), (0.70, 0.099), (0.72, 0.104),
         (0.74, 0.108), (0.76, 0.110), (0.78, 0.107), (0.80, 0.097), (0.82, 0.079), (0.835, 0.06), (0.848, 0.044)]
T_W_F = [(0.455, 0.074), (0.47, 0.104), (0.50, 0.114), (0.52, 0.110), (0.54, 0.100), (0.56, 0.088), (0.58, 0.078),
         (0.60, 0.073), (0.62, 0.072), (0.64, 0.073), (0.66, 0.077), (0.68, 0.082), (0.70, 0.087), (0.72, 0.091),
         (0.74, 0.094), (0.76, 0.096), (0.78, 0.094), (0.80, 0.086), (0.82, 0.071), (0.835, 0.054), (0.848, 0.04)]
T_F_M = [(0.455, 0.035), (0.47, 0.050), (0.50, 0.057), (0.52, 0.058), (0.54, 0.059), (0.56, 0.060), (0.58, 0.061),
         (0.60, 0.060), (0.62, 0.059), (0.64, 0.059), (0.66, 0.062), (0.68, 0.067), (0.70, 0.072), (0.72, 0.076),
         (0.74, 0.076), (0.76, 0.072), (0.78, 0.066), (0.80, 0.056), (0.82, 0.046), (0.835, 0.038), (0.848, 0.031)]
T_F_F = [(0.455, 0.035), (0.47, 0.050), (0.50, 0.056), (0.52, 0.056), (0.54, 0.056), (0.56, 0.055), (0.58, 0.054),
         (0.60, 0.053), (0.62, 0.052), (0.64, 0.053), (0.66, 0.056), (0.68, 0.060), (0.70, 0.064), (0.72, 0.066),
         (0.74, 0.066), (0.76, 0.064), (0.78, 0.060), (0.80, 0.052), (0.82, 0.043), (0.835, 0.036), (0.848, 0.03)]
T_B_M = [(0.455, 0.050), (0.47, 0.070), (0.50, 0.079), (0.52, 0.075), (0.54, 0.065), (0.56, 0.057), (0.58, 0.052),
         (0.60, 0.050), (0.62, 0.050), (0.64, 0.051), (0.66, 0.054), (0.68, 0.057), (0.70, 0.059), (0.72, 0.061),
         (0.74, 0.062), (0.76, 0.063), (0.78, 0.062), (0.80, 0.058), (0.82, 0.051), (0.835, 0.045), (0.848, 0.04)]
T_B_F = [(0.455, 0.052), (0.47, 0.074), (0.50, 0.085), (0.52, 0.080), (0.54, 0.068), (0.56, 0.057), (0.58, 0.050),
         (0.60, 0.047), (0.62, 0.046), (0.64, 0.047), (0.66, 0.050), (0.68, 0.052), (0.70, 0.054), (0.72, 0.056),
         (0.74, 0.057), (0.76, 0.058), (0.78, 0.057), (0.80, 0.053), (0.82, 0.047), (0.835, 0.042), (0.848, 0.037)]
T_N = [(0.455, 2.0), (0.52, 2.15), (0.62, 2.1), (0.72, 2.45), (0.80, 2.3), (0.848, 2.0)]

# ----------------------------------------------------------------- limbs ---
# t from proximal joint (0) to distal joint (1); values / H
UPPER_ARM = dict(
    wo=[(0, 0.044), (0.15, 0.041), (0.35, 0.032), (0.6, 0.0285), (0.85, 0.0262), (1, 0.025)],
    wi=[(0, 0.031), (0.2, 0.030), (0.5, 0.0275), (0.85, 0.0245), (1, 0.024)],
    f=[(0, 0.037), (0.2, 0.032), (0.5, 0.0305), (0.75, 0.028), (1, 0.022)],
    b=[(0, 0.039), (0.2, 0.034), (0.45, 0.0325), (0.8, 0.026), (1, 0.024)],
)
FOREARM = dict(
    wo=[(0, 0.025), (0.25, 0.0262), (0.6, 0.0205), (1, 0.0125)],
    wi=[(0, 0.025), (0.25, 0.026), (0.6, 0.0195), (1, 0.012)],
    f=[(0, 0.027), (0.2, 0.0295), (0.6, 0.022), (1, 0.0168)],
    b=[(0, 0.028), (0.2, 0.0265), (0.6, 0.0205), (1, 0.0162)],
)
THIGH = dict(
    wo=[(0, 0.060), (0.1, 0.058), (0.3, 0.0525), (0.55, 0.0465), (0.8, 0.0375), (1, 0.032)],
    wi=[(0, 0.048), (0.1, 0.049), (0.3, 0.0475), (0.55, 0.043), (0.8, 0.0365), (1, 0.0335)],
    f=[(0, 0.050), (0.3, 0.0515), (0.6, 0.0475), (0.85, 0.040), (1, 0.034)],
    b=[(0, 0.057), (0.2, 0.0535), (0.5, 0.047), (0.8, 0.038), (1, 0.0325)],
)
SHIN = dict(
    wo=[(0, 0.032), (0.1, 0.0305), (0.25, 0.0335), (0.4, 0.0325), (0.6, 0.0265), (0.8, 0.0195), (1, 0.0172)],
    wi=[(0, 0.0335), (0.1, 0.031), (0.25, 0.0355), (0.4, 0.0345), (0.6, 0.0275), (0.8, 0.0195), (1, 0.0178)],
    f=[(0, 0.034), (0.12, 0.0285), (0.3, 0.026), (0.6, 0.022), (1, 0.0195)],
    b=[(0, 0.030), (0.15, 0.040), (0.3, 0.0445), (0.5, 0.0375), (0.7, 0.026), (0.9, 0.0195), (1, 0.021)],
)


def _limb(s, a, b, prof, H, outer_dir, k, scale_w=1.0, scale_d=1.0, tag="skin", name="limb", ext_a=0.0, ext_b=0.0):
    s.union(loft(a, b, _scale(prof["wo"], H * scale_w), _scale(prof["wi"], H * scale_w),
                 _scale(prof["f"], H * scale_d), _scale(prof["b"], H * scale_d), outer_dir=outer_dir,
                 ext_a=ext_a * H, ext_b=ext_b * H),
            k=k, tag=tag, name=name)


def build_body(p, J) -> Shape:
    s = Shape()
    H = p["height"]
    fem, mus, fat = p["fem"], p["muscle"], p["fat"]
    thick = p["limb_thick"]
    # muscle / fat scale factors (reference body ~ muscle .55, fat .2)
    m_arm = (1 + 0.28 * (mus - 0.55)) * (1 + 0.25 * (fat - 0.2)) * thick
    m_leg = (1 + 0.18 * (mus - 0.55)) * (1 + 0.3 * (fat - 0.2)) * thick
    torso_w = 1 + 0.35 * (fat - 0.2)
    belly = max(0.0, fat - 0.25) * 0.09

    # ---------------- torso: one loft hips -> neck ----------------
    z0, z1 = TORSO_Z
    shp = p["shoulder"]
    hipm = p["hip"]

    def wscale(z):
        # shoulders scale the upper torso, hips the lower torso
        u = np.clip((z - 0.56) / 0.2, 0, 1)
        return (hipm * (1 - u) + shp * u) * torso_w

    W = [(z, v * H * wscale(z)) for z, v in _mix(T_W_M, T_W_F, fem)]
    Fr = [(z, v * H * (1 + (belly * 6 if 0.52 < z < 0.66 else 0) + 0.3 * (fat - 0.2))) for z, v in _mix(T_F_M, T_F_F, fem)]
    Bk = [(z, v * H * (1 + 0.25 * (fat - 0.2))) for z, v in _mix(T_B_M, T_B_F, fem)]
    a, b = v3(0, 0, z0 * H), v3(0, 0.004 * H, z1 * H)
    s.union(loft(a, b, _zkeys(W, z0, z1), _zkeys(W, z0, z1), _zkeys(Fr, z0, z1), _zkeys(Bk, z0, z1),
                 n_front=_zkeys(T_N, z0, z1), n_back=_zkeys(T_N, z0, z1)),
            tag="skin", name="torso")

    # ---------------- neck ----------------
    neck_base = v3(0, 0.006 * H, 0.83 * H)
    hh = J["_head_h"]
    neck_top = v3(0, J["head"][1] + 0.07 * hh, J["head_top"][2] - 0.9 * hh)
    nw = p["neck"] * (1 - 0.12 * fem) * (1 + 0.2 * (mus - 0.5))
    s.union(loft(neck_base, neck_top, [(0, 0.043 * H * nw), (0.5, 0.036 * H * nw), (1, 0.034 * H * nw)],
                 [(0, 0.043 * H * nw), (0.5, 0.036 * H * nw), (1, 0.034 * H * nw)],
                 [(0, 0.032 * H * nw), (1, 0.03 * H * nw)], [(0, 0.04 * H * nw), (0.6, 0.034 * H * nw), (1, 0.036 * H * nw)],
), k=0.02 * H, tag="skin", name="neck")
    if fem < 0.5:   # laryngeal prominence
        s.union(ellipsoid(neck_base + (neck_top - neck_base) * 0.55 + v3(0, -0.028 * H * nw, 0), [0.008 * H, 0.008 * H, 0.012 * H]),
                k=0.01 * H, tag="skin", name="adams_apple")

    # ---------------- arms ----------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        s.side = side
        sh, el, wr = J[f"upper_arm.{side}"], J[f"forearm.{side}"], J[f"hand.{side}"]
        up = (el - sh) / np.linalg.norm(el - sh)
        outer = v3(sx, 0, 0) + v3(0, 0, 1) * 1.2
        aw = m_arm * (1 - 0.14 * fem)
        _limb(s, sh - up * 0.012 * H, el, UPPER_ARM, H, outer, 0.028 * H, aw, aw, name="upper_arm", ext_b=0.02)
        _limb(s, el, wr + (wr - el) / np.linalg.norm(wr - el) * 0.006 * H, FOREARM, H, outer, 0.012 * H,
              aw * (1 - 0.05 * fem), aw * (1 - 0.05 * fem), name="forearm", ext_a=0.02, ext_b=0.01)
        # elbow point (olecranon)
        s.union(sphere(el + v3(0, 0.024 * H * aw, 0.002 * H), 0.011 * H), k=0.012 * H, tag="skin", name="elbow")
        _hand(s, p, J, side)

    # ---------------- legs ----------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        s.side = side
        hj, kn, an = J[f"thigh.{side}"], J[f"shin.{side}"], J[f"foot.{side}"]
        outer = v3(sx, 0, 0)
        lw = m_leg * (1 + 0.06 * fem) * (hipm ** 0.5)
        _limb(s, hj + v3(0, 0, 0.035 * H), kn, THIGH, H, outer, 0.022 * H, lw, lw, name="thigh", ext_b=0.025)
        _limb(s, kn, an, SHIN, H, outer, 0.012 * H, m_leg * (1 - 0.06 * fem), m_leg * (1 - 0.04 * fem), name="shin", ext_a=0.025, ext_b=0.012)
        # patella + malleoli: small, tightly blended
        s.union(ellipsoid(kn + v3(0, -0.03 * H, 0.004 * H), [0.014 * H, 0.008 * H, 0.016 * H]), k=0.01 * H, tag="skin", name="patella")
        for ox in (-1, 1):
            s.union(sphere(an + v3(ox * 0.015 * H, 0.002 * H, 0.004 * H), 0.0075 * H), k=0.008 * H, tag="skin", name="malleolus")
        _foot(s, p, J, side)
    s.side = ""

    # ---------------- subtle surface anatomy ----------------
    _surface_detail(s, p, J)
    _head(s, p, J)
    _race_extras(s, p, J)
    return s


def _surface_detail(s, p, J):
    H = p["height"]
    fem, mus = p["fem"], p["muscle"]
    chest = 0.72 * H
    if fem < 0.5:
        for sx in (-1, 1):   # pectoral plates: flat, wide, low protrusion
            c = v3(sx * 0.052 * H, -0.06 * H, chest + 0.012 * H)
            s.union(ellipsoid(c, [0.05 * H, 0.016 * H * (0.8 + 0.5 * mus), 0.034 * H], sdf.euler(0, sx * 10, sx * -6)),
                    k=0.018 * H, tag="skin", name="pec")
    else:
        for sx in (-1, 1):   # breasts
            bs = 0.85 + 0.6 * p["fat"]
            c = v3(sx * 0.048 * H, -0.058 * H, chest - 0.008 * H)
            s.union(ellipsoid(c, [0.04 * H * bs, 0.035 * H * bs, 0.038 * H * bs], sdf.euler(12, 0, sx * -10)),
                    k=0.02 * H, tag="skin", name="breast")
    for sx in (-1, 1):       # clavicles
        s.union(capsule(v3(sx * 0.014 * H, -0.03 * H, 0.818 * H), v3(sx * 0.09 * H, -0.022 * H, 0.81 * H), 0.0055 * H),
                k=0.012 * H, tag="skin", name="clavicle")
        # scapulae: very soft
        s.union(ellipsoid(v3(sx * 0.055 * H, 0.05 * H, 0.75 * H), [0.035 * H, 0.012 * H, 0.04 * H]), k=0.02 * H, tag="skin", name="scapula")
    # navel
    s.sub(ellipsoid(v3(0, -0.062 * H * (1 + 0.3 * (p["fat"] - 0.2)), 0.585 * H), [0.004 * H, 0.008 * H, 0.005 * H]), k=0.004 * H, name="navel")


# ------------------------------------------------------------------ head ---
def _head(s, p, J):
    if p["snout"] > 0:
        from humanoid import _draconic_head
        return _draconic_head(s, p, J)
    hh = J["_head_h"]
    top = J["head_top"]
    yc = J["head"][1]
    R = REF_HEAD
    z = np.array(R["z"])
    zmax = z[-1]
    w = np.array(R["w"])
    fr = np.array(R["front"])
    bk = np.array(R["back"])
    lm = R["landmarks"]
    # remove the nose from the front profile (rebuilt as its own form)
    n0, n1 = lm["nasion"] - 0.01, lm["subnasale"] + 0.012
    m = (z > n0) & (z < n1)
    f0, f1 = np.interp(n0, z, fr), np.interp(n1, z, fr)
    sm = (z[m] - n0) / (n1 - n0)
    fr = fr.copy()
    fr[m] = f0 + (f1 - f0) * sm + 0.012 * np.sin(np.pi * sm)
    # race / face sliders
    jaw = p["jaw"] * (1 - 0.08 * p["fem"])
    wmod = np.where(z > 0.72, 1 + (jaw - 1) * np.clip((z - 0.72) / 0.2, 0, 1), 1.0)
    w = w * wmod
    fw = p.get("face_width", 1.0)
    w = w * fw
    # the front silhouette below the jaw angle is the NECK, not the face: taper jaw -> chin
    jz, cz = 0.80, 1.0
    wj = np.interp(jz, z, w)
    sel = z > jz
    u = np.clip((z[sel] - jz) / (cz - jz), 0, 1)
    w[sel] = wj + (0.105 * p["chin"] - wj) * (u ** 1.6)
    keep = z <= 1.02
    z, w, fr, bk = z[keep], w[keep], fr[keep], bk[keep]
    zmax = z[-1]
    t = z / zmax
    keys_w = list(zip(t, w * hh))
    keys_f = list(zip(t, fr * hh))
    keys_b = list(zip(t, bk * hh))
    nfr = [(0, 2.0), (lm["brow"] / zmax, 2.1), (lm["eye"] / zmax, 1.95), (lm["mouth"] / zmax, 1.6), (0.92 / zmax, 1.75), (1, 2.0)]
    a = v3(0, yc, top[2])
    b = v3(0, yc, top[2] - zmax * hh)
    s.union(loft(a, b, keys_w, keys_w, keys_f, keys_b, n_front=nfr, n_back=2.1), k=0.015 * hh, tag="skin", name="head")
    # rounded crown (loft caps are flat; this restores the dome exactly)
    s.union(ellipsoid(v3(0, yc + (np.interp(0.33, z, bk) - np.interp(0.33, z, fr)) * 0.5 * hh, top[2] - 0.33 * hh),
                      [np.interp(0.33, z, w) * hh * 0.995, (np.interp(0.33, z, fr) + np.interp(0.33, z, bk)) * 0.5 * hh * 0.995, 0.33 * hh]),
            k=0.01 * hh, tag="skin", name="crown")

    def P_(lat, frontv, zn):
        """head-unit point: lateral, forward (+ = toward face), z from top"""
        return v3(lat * hh, yc - frontv * hh, top[2] - zn * hh)

    def face_front(zn, lat=0.0):
        ww = np.interp(zn, z, w)
        ff = np.interp(zn, z, fr)
        n = 1.9
        return ff * max(0.0, 1 - (abs(lat) / ww) ** n) ** (1 / n)

    # ---- eyes: socket, lid shell, almond opening
    ez = lm["eye"]
    ex = 0.128
    er = 0.053 * p["eye"]
    for sx in (-1, 1):
        surf = face_front(ez, ex)
        ec = P_(sx * ex, surf - er * 0.62, ez)
        s.sub(sphere(P_(sx * ex, surf + 0.02, ez - 0.012), 0.058 * hh), k=0.03 * hh, name="socket")
        s.union(sphere(ec, (er + 0.008) * hh), k=0.022 * hh, tag="lid", name="lidshell")
        Rr, aa = 0.105 * p["eye"], 0.03 * p["eye"]
        up = np.array([0, 0, 1], F) @ sdf.euler(0, sx * p["eye_tilt"] * 10.0, 0)
        c1 = ec + up * (-(Rr - aa) * hh)
        c2 = ec + up * ((Rr - aa) * hh)
        f1 = sphere(c1, Rr * hh)[0]
        f2 = sphere(c2, Rr * hh)[0]
        fy = ec[1] - 0.012 * hh

        def almond(q, f1=f1, f2=f2, fy=fy):
            return np.maximum(np.maximum(f1(q), f2(q)), q[:, 1] - fy)
        s.sub((almond, ec - 0.15 * hh, ec + 0.15 * hh), k=0.004 * hh, name="eye_opening")
        # brow ridge: soft, wide
        s.union(ellipsoid(P_(sx * 0.11, face_front(lm["brow"], 0.11) - 0.03, lm["brow"]), [0.085 * hh, 0.03 * hh, 0.025 * hh * p["brow"]], sdf.euler(0, sx * -8, 0)),
                k=0.03 * hh, tag="skin", name="brow")
    # ---- nose
    nl, nw, nb = p["nose_len"], p["nose_w"], p["nose_bridge"]
    nas_z, tip_z, sub_z = lm["nasion"], lm["nose_tip"], lm["subnasale"]
    tip_z = nas_z + (tip_z - nas_z) * nl
    tip_front = np.interp(lm["nose_tip"], np.array(R["z"]), np.array(R["front"])) * (0.97 + 0.03 * nl)
    nas_front = face_front(nas_z) - 0.01
    s.union(round_cone(P_(0, nas_front - 0.02, nas_z + 0.01), P_(0, tip_front - 0.03, tip_z - 0.01), 0.026 * hh * nb, 0.032 * hh * nw),
            k=0.02 * hh, tag="skin", name="nose_bridge")
    s.union(sphere(P_(0, tip_front - 0.034, tip_z + 0.006), 0.036 * hh * nw), k=0.014 * hh, tag="skin", name="nose_tip")
    for sx in (-1, 1):
        s.union(sphere(P_(sx * 0.044 * nw, tip_front - 0.075, tip_z + 0.032), 0.03 * hh * nw), k=0.014 * hh, tag="skin", name="ala")
        s.sub(ellipsoid(P_(sx * 0.022 * nw, tip_front - 0.06, sub_z - 0.006), [0.013 * hh, 0.022 * hh, 0.0095 * hh]), k=0.006 * hh, name="nostril")
    # ---- mouth & lips
    mz = lm["mouth"]
    lip_f = face_front(mz)
    s.sub(ellipsoid(P_(0, lip_f + 0.01, mz), [0.095 * hh, 0.05 * hh, 0.0042 * hh]), k=0.006 * hh, name="mouth_line")
    for sx in (-1, 1):
        s.sub(sphere(P_(sx * 0.098, lip_f - 0.04, mz), 0.012 * hh), k=0.012 * hh, name="mouth_corner")
    lp = p["lips"] * (1 + 0.12 * p["fem"])
    s.union(ellipsoid(P_(0, lip_f - 0.03, mz - 0.026), [0.085 * hh, 0.03 * hh, 0.022 * hh * lp]), k=0.012 * hh, tag="lips", name="upper_lip", tag_eps=0.006 * hh)
    s.union(ellipsoid(P_(0, lip_f - 0.035, mz + 0.03), [0.075 * hh, 0.03 * hh, 0.026 * hh * lp]), k=0.012 * hh, tag="lips", name="lower_lip", tag_eps=0.006 * hh)
    # philtrum columns
    for sx in (-1, 1):
        s.union(capsule(P_(sx * 0.018, lip_f - 0.012, sub_z + 0.01), P_(sx * 0.022, lip_f - 0.012, mz - 0.04), 0.008 * hh), k=0.01 * hh, tag="skin", name="philtrum")
    # ---- ears
    _ears(s, p, J, P_, lm, z, w)
    if p["tusks"] > 0:
        for sx in (-1, 1):
            base = P_(sx * 0.07, lip_f - 0.02, mz + 0.03)
            tipp = P_(sx * 0.09, lip_f + 0.02, mz - 0.07 * p["tusks"])
            s.union(round_cone(base, tipp, 0.022 * hh * p["tusks"], 0.004 * hh), k=0.004 * hh, tag="tusk", name="tusk")


def _ears(s, p, J, P_, lm, z, w):
    kind = p["ears"]
    if kind == "none":
        return
    hh = J["_head_h"]
    es = p["ear_size"]
    zt, zb = lm["ear_top"], lm["ear_bottom"]
    zc = (zt + zb) * 0.5
    hgt = (zb - zt) * 0.5 * es
    for sx in (-1, 1):
        lat = np.interp(zc, z, w) - 0.01
        c = P_(sx * lat, -0.085, zc)
        rot = sdf.euler(0, 0, sx * 18) @ sdf.euler(-10, 0, 0)
        # helix rim + concha bowl + lobe
        s.union(ellipsoid(c + v3(sx * 0.03 * hh, 0, 0), [0.03 * hh, 0.078 * hh * es, hgt * hh], rot), k=0.02 * hh, tag="skin", name="ear")
        s.sub(ellipsoid(c + v3(sx * 0.055 * hh, -0.01 * hh, 0.01 * hh), [0.022 * hh, 0.05 * hh * es, hgt * 0.68 * hh], rot), k=0.01 * hh, name="ear_bowl")
        s.union(ellipsoid(c + v3(sx * 0.035 * hh, -0.005 * hh, -hgt * 0.78 * hh), [0.02 * hh, 0.035 * hh, 0.035 * hh]), k=0.015 * hh, tag="skin", name="lobe")
        if kind in ("pointed", "long"):
            ln = 0.2 if kind == "long" else 0.1
            a = c + v3(sx * 0.03 * hh, 0.01 * hh, hgt * 0.6 * hh)
            b = a + v3(sx * ln * 0.45 * hh, ln * 0.55 * hh, ln * hh)
            s.union(round_cone(a, b, 0.04 * hh * es, 0.006 * hh), k=0.025 * hh, tag="skin", name="ear_tip")


def eye_shape(p, J) -> Shape:
    if p["snout"] > 0:
        from humanoid import eye_shape as e2
        return e2(p, J)
    hh = J["_head_h"]
    top = J["head_top"]
    yc = J["head"][1]
    R = REF_HEAD
    z = np.array(R["z"])
    ez = R["landmarks"]["eye"]
    er = 0.053 * p["eye"]
    s = Shape()
    for sx in (-1, 1):
        ww = np.interp(ez, z, R["w"])
        ff = np.interp(ez, z, R["front"])
        surf = ff * max(0.0, 1 - (0.128 / ww) ** 1.9) ** (1 / 1.9)
        c = v3(sx * 0.128 * hh, yc - (surf - er * 0.62) * hh, top[2] - ez * hh)
        s.union(sphere(c, er * hh), tag="eye")
    return s


def eye_centers(p, J):
    hh = J["_head_h"]
    top = J["head_top"]
    yc = J["head"][1]
    R = REF_HEAD
    z = np.array(R["z"])
    ez = R["landmarks"]["eye"]
    er = 0.053 * p["eye"]
    ww = np.interp(ez, z, R["w"])
    ff = np.interp(ez, z, R["front"])
    surf = ff * max(0.0, 1 - (0.128 / ww) ** 1.9) ** (1 / 1.9)
    return [v3(sx * 0.128 * hh, yc - (surf - er * 0.62) * hh, top[2] - ez * hh) for sx in (-1, 1)]
