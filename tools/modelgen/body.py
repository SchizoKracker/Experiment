"""Body (everything below the jaw) – lofted base + individually shaped muscle forms.

The head/face lives in anatomy.py and is intentionally not touched here.
All values are fractions of body height H.
"""
from __future__ import annotations

import numpy as np

import anatomy
import extremities
import sdf
from humanoid import _foot, _hand, _race_extras, v3
from sdf import F, Shape, capsule, ellipsoid, frame_z, loft, sphere

# ------------------------------------------------------------------ torso --
TORSO_Z = (0.455, 0.865)
T_W_M = [(0.455, 0.036), (0.465, 0.048), (0.47, 0.054), (0.48, 0.064), (0.5, 0.078), (0.52, 0.085), (0.54, 0.087), (0.56, 0.086), (0.58, 0.086), (0.6, 0.085), (0.62, 0.085), (0.64, 0.086), (0.66, 0.091), (0.68, 0.095), (0.7, 0.099), (0.72, 0.105), (0.74, 0.108), (0.76, 0.108), (0.78, 0.106), (0.8, 0.106), (0.815, 0.104), (0.83, 0.094), (0.845, 0.076), (0.855, 0.06), (0.865, 0.042)]
T_W_F = [(0.455, 0.04), (0.465, 0.056), (0.47, 0.064), (0.48, 0.074), (0.49, 0.08), (0.5, 0.085), (0.52, 0.088), (0.53, 0.087), (0.54, 0.085), (0.56, 0.08), (0.58, 0.075), (0.6, 0.071), (0.62, 0.068), (0.64, 0.068), (0.66, 0.07), (0.68, 0.073), (0.7, 0.076), (0.72, 0.079), (0.74, 0.082), (0.76, 0.084), (0.78, 0.086), (0.8, 0.089), (0.815, 0.09), (0.83, 0.081), (0.845, 0.065), (0.855, 0.052), (0.865, 0.036)]
T_F_M = [(0.455, 0.03), (0.47, 0.048), (0.5, 0.054), (0.53, 0.054), (0.56, 0.055), (0.58, 0.055), (0.6, 0.054), (0.62, 0.054), (0.64, 0.056), (0.66, 0.06), (0.68, 0.067), (0.7, 0.073), (0.72, 0.075), (0.74, 0.074), (0.76, 0.07), (0.78, 0.064), (0.8, 0.058), (0.815, 0.05), (0.83, 0.04), (0.845, 0.026), (0.855, 0.018), (0.865, 0.01)]
T_F_F = [(0.455, 0.03), (0.465, 0.042), (0.47, 0.046), (0.48, 0.05), (0.5, 0.054), (0.52, 0.054), (0.53, 0.0535), (0.54, 0.053), (0.56, 0.052), (0.58, 0.051), (0.6, 0.05), (0.62, 0.049), (0.64, 0.05), (0.66, 0.053), (0.68, 0.058), (0.7, 0.062), (0.72, 0.064), (0.74, 0.064), (0.76, 0.062), (0.78, 0.058), (0.8, 0.051), (0.815, 0.043), (0.83, 0.036), (0.845, 0.025), (0.855, 0.017), (0.865, 0.01)]
T_B_M = [(0.455, 0.04), (0.47, 0.058), (0.5, 0.066), (0.52, 0.065), (0.54, 0.06), (0.56, 0.053), (0.58, 0.047), (0.6, 0.045), (0.62, 0.046), (0.64, 0.05), (0.66, 0.055), (0.68, 0.06), (0.7, 0.064), (0.72, 0.067), (0.74, 0.069), (0.76, 0.069), (0.78, 0.066), (0.8, 0.06), (0.815, 0.05), (0.83, 0.044), (0.845, 0.042), (0.855, 0.041), (0.865, 0.036)]
T_B_F = [(0.455, 0.045), (0.465, 0.058), (0.47, 0.063), (0.48, 0.068), (0.5, 0.071), (0.52, 0.07), (0.54, 0.066), (0.56, 0.058), (0.58, 0.046), (0.6, 0.043), (0.62, 0.043), (0.64, 0.045), (0.66, 0.051), (0.68, 0.055), (0.7, 0.058), (0.72, 0.061), (0.74, 0.062), (0.76, 0.062), (0.78, 0.06), (0.8, 0.055), (0.815, 0.046), (0.83, 0.041), (0.845, 0.039), (0.855, 0.038), (0.865, 0.033)]
T_N = [(0.455, 2), (0.52, 2.1), (0.62, 2.1), (0.72, 2.2), (0.8, 2.15), (0.845, 2), (0.865, 2)]
T_NB = [(0.455, 2.0), (0.52, 2.05), (0.62, 2.0), (0.72, 1.92), (0.8, 1.95), (0.845, 2.0), (0.865, 2.0)]

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
THIGH = dict(wo=[(0, 0.042), (0.12, 0.05), (0.24, 0.054), (0.49, 0.051), (0.66, 0.047), (0.83, 0.038), (0.92, 0.035), (1, 0.031)],
             wi=[(0, 0.05), (0.24, 0.052), (0.49, 0.045), (0.66, 0.036), (0.8, 0.0305), (0.88, 0.0315), (1, 0.029)],
             f=[(0, 0.047), (0.3, 0.05), (0.6, 0.044), (0.85, 0.035), (1, 0.029)],
             b=[(0, 0.045), (0.12, 0.052), (0.3, 0.049), (0.6, 0.041), (0.85, 0.032), (1, 0.029)])
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
                 n_front=_zk(T_N, z0, z1), n_back=_zk(T_NB, z0, z1)), tag="skin", name="torso")

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

    if fem > 0.05:
        for sx in (-1, 1):   # breasts (volume, not surface relief)
            bs = (0.9 + 0.6 * p["fat"]) * fem
            z, x = 0.722, 0.05
            surf = torso_surface_front(z, x, fem, H)
            s.union(ellipsoid(v3(sx * x * H, -(surf - 0.026 * bs) * H, z * H), [0.034 * H * bs, 0.032 * H * bs, 0.033 * H * bs], sdf.euler(8, 0, sx * -14)),
                    k=0.014 * H, tag="skin", name="breast")

    # ---------------- arms ----------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        s.side = side
        sh, el, wr = J[f"upper_arm.{side}"], J[f"forearm.{side}"], J[f"hand.{side}"]
        up = (el - sh) / np.linalg.norm(el - sh)
        outer = v3(sx, 0, 0) + v3(0, 0, 1.2)
        aw = m_arm * (1 - 0.15 * fem)
        _limb(s, sh, el, UPPER_ARM, H, outer, 0.02 * H, aw, aw, "upper_arm", ext_a=0.01, ext_b=0.02)
        dm = (1 + 0.35 * (mus - 0.5)) * (1 - 0.08 * fem) * thick
        _limb(s, sh - up * 0.022 * H + v3(sx * 0.004 * H, 0, -0.004 * H), sh + up * 0.105 * H, DELTOID, H, outer, 0.016 * H, dm, dm, "deltoid",
              cap_round=0.95)
        # deltoid / biceps border and the deltoid's front edge
        fo = wr + (wr - el) / np.linalg.norm(wr - el) * 0.006 * H
        _limb(s, el, fo, FOREARM, H, outer, 0.005 * H, aw * (1 - 0.04 * fem), aw * (1 - 0.04 * fem), "forearm", ext_a=0.02, ext_b=0.01)
        s.union(sphere(el + v3(0, 0.021 * H * aw, 0.002 * H), 0.009 * H), k=0.01 * H, tag="skin", name="olecranon")
        extremities.build_hand(s, p, J, side)

    # ---------------- legs ----------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        s.side = side
        hj, kn, an = J[f"thigh.{side}"], J[f"shin.{side}"], J[f"foot.{side}"]
        outer = v3(sx, 0, 0)
        lw = m_leg * (1 - 0.15 * fem) * (hipm ** 0.5)
        top = hj + v3(sx * 0.0 * H, 0.004 * H, 0.012 * H)
        top_hi_ = hj + v3(0.0, 0.004 * H, 0.06 * H)
        L_all = float(np.linalg.norm(an - top_hi_))
        t0 = float(np.linalg.norm(top - top_hi_) / L_all)
        tk = float(t0 + np.linalg.norm(kn - top) / L_all)
        leg = {}
        for key in ("wo", "wi", "f", "b"):
            def fboost(t, key=key):
                # female: outer thigh/hip and rear (glute-thigh) volume near the top, fading by mid-thigh
                f = float(np.exp(-((t - 0.16) / 0.17) ** 2))      # peaks at the greater trochanter
                return 1.0 + fem * f * {"wo": 0.0, "b": 0.0, "wi": 0.1, "f": 0.0}[key]
            th = [(0.0, THIGH[key][0][1] * lw * (0.78 if key == "wo" else 0.95))] + \
                 [(t0 + t * (tk - t0), v * lw * fboost(t)) for t, v in THIGH[key]]
            sh_ = [(tk + t * (1 - tk), v * m_leg * (1 - 0.06 * fem)) for t, v in SHIN[key]]
            merged = dict(th)
            for t, v in sh_:
                merged[t] = (merged[t] + v) * 0.5 if t in merged else v
            leg[key] = sorted(merged.items())
        # leg rises into the pelvis on the outside; its top is sliced along the inguinal line
        # (crotch -> hip bone), so torso and thighs meet in a V instead of a horizontal seam.
        top_hi = hj + v3(0.0, 0.004 * H, 0.06 * H)
        legf, llo, lhi = loft(top_hi, an, _sc(leg["wo"], H), _sc(leg["wi"], H), _sc(leg["f"], H), _sc(leg["b"], H),
                              outer_dir=outer, ext_b=0.03 * H)
        A = v3(sx * 0.016 * H, 0, 0.462 * H)            # crotch
        B = v3(sx * 0.1 * H, 0, 0.565 * H)  # above the greater trochanter / hip bone
        dline = (B - A) / np.linalg.norm(B - A)
        nrm = np.cross(dline, v3(0, 1, 0))
        if nrm[2] < 0:
            nrm = -nrm
        kc = 0.014 * H

        def leg_cut(q, legf=legf, A=A, nrm=nrm, kc=kc):
            return sdf.smax(legf(q), (q - A) @ nrm, kc)
        s.union((leg_cut, llo, lhi), k=0.02 * H, tag="skin", name="leg")
        # hip mass (gluteus medius + trochanter): a smooth Gaussian buildup, no rims
        s.displace.append(sdf.brush(v3(sx * 0.098 * H, 0.008 * H, (0.492 - 0.004 * fem) * H),
                                    [0.03 * H, 0.04 * H, (0.052 + 0.006 * fem) * H], (0.004 + 0.001 * fem) * H,
                                    np.stack([v3(sx, 0, 0), v3(0, 1, 0), v3(0, 0, 1)])))
        # malleoli: medial higher and further forward, lateral lower and further back (surface relief only)
        s.displace.append(sdf.brush(an + v3(-sx * 0.019 * H, -0.003 * H, 0.006 * H), [0.006 * H, 0.007 * H, 0.008 * H], 0.0022 * H))
        s.displace.append(sdf.brush(an + v3(sx * 0.02 * H, 0.004 * H, -0.002 * H), [0.006 * H, 0.007 * H, 0.008 * H], 0.0022 * H))
        extremities.build_foot(s, p, J, side)
    s.side = ""

    anatomy._head(s, p, J)
    import muscles
    muscles.apply(s, p, J)
    _race_extras(s, p, J)
    return s
