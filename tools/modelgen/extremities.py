"""Hands and feet: anatomically proportioned, lofted, with pads, knuckles, nails and creases.

Dimensions are for a 1.78 m adult male and scale with height * hand/foot params.
Hand: palm 0.098 (wrist->MCP), hand length ~0.19.  Foot: length ~0.265, ball width ~0.10.
"""
from __future__ import annotations

import math

import numpy as np

import sdf
from sdf import F, Shape, ellipsoid, loft

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# phalanx lengths (proximal, middle, distal); thumb = metacarpal, proximal, distal
F_LEN = {"thumb": (0.040, 0.031, 0.026), "index": (0.040, 0.024, 0.020), "middle": (0.045, 0.028, 0.021),
         "ring": (0.042, 0.027, 0.021), "pinky": (0.033, 0.019, 0.018)}
# half-width (across) at the three joints + tip, and depth ratio
F_RAD = {"thumb": (0.0125, 0.0112, 0.0102, 0.0085), "index": (0.0096, 0.0088, 0.0078, 0.0068),
         "middle": (0.0099, 0.009, 0.008, 0.007), "ring": (0.0093, 0.0085, 0.0076, 0.0066),
         "pinky": (0.0083, 0.0076, 0.0069, 0.006)}
# MCP knuckle positions along the knuckle arc: (across offset, palm-length offset)
MCP = {"index": (0.027, -0.004), "middle": (0.008, 0.0), "ring": (-0.011, -0.004), "pinky": (-0.028, -0.014)}


def _n(v):
    v = np.asarray(v, F)
    return v / (np.linalg.norm(v) + 1e-9)


def v3(*a):
    return np.array(a, dtype=F)


def hand_frame(J, side):
    from humanoid import hand_frame as hf
    return hf(J, side)


def hand_joints(J, p):
    H = p["height"]
    hs = H * p["hand"] / 1.78
    for side in ("L", "R"):
        wr, fwd, n, ac = hand_frame(J, side)
        palm = 0.098 * hs
        for fname in FINGERS:
            if fname == "thumb":
                base = wr + fwd * 0.022 * hs + ac * 0.02 * hs + n * 0.01 * hs
                d = _n(fwd * 0.8 + ac * 0.42 + n * 0.5)
                curls = (0.18, 0.12, 0.1)
            else:
                ao, lo = MCP[fname]
                base = wr + fwd * (palm + lo * hs) + ac * ao * hs
                spread = {"index": 0.1, "middle": 0.02, "ring": -0.07, "pinky": -0.16}[fname]
                d = _n(fwd + ac * spread + n * 0.05)
                curls = (0.08, 0.12, 0.1)
            q = base
            for si in range(3):
                J[f"{fname}{si + 1}.{side}"] = q.astype(F)
                d = _n(d * math.cos(curls[si]) + n * math.sin(curls[si]))
                q = q + d * F_LEN[fname][si] * hs
            J[f"{fname}_end.{side}"] = q.astype(F)


def foot_frame(J, side):
    an = J[f"foot.{side}"]
    sx = 1.0 if side == "L" else -1.0
    return an, sx


def FP(J, side, fs, x_lat, y_fwd, z):
    """Foot-space point: lateral (outward), forward, absolute height."""
    an, sx = foot_frame(J, side)
    return v3(an[0] + sx * x_lat * fs, an[1] - y_fwd * fs, z * fs)


def foot_joints(J, p, side):
    fs = p["height"] * p["foot"] / 1.78
    J[f"toe.{side}"] = FP(J, side, fs, 0.003, 0.138, 0.021)
    J[f"toe_end.{side}"] = FP(J, side, fs, -0.002, 0.205, 0.012)


# ------------------------------------------------------------------- hand --
def build_hand(s: Shape, p, J, side):
    hs = p["height"] * p["hand"] / 1.78
    wr, fwd, n, ac = hand_frame(J, side)
    D = s.displace
    palm_len = 0.098 * hs
    # palm: flat dorsum (high superellipse exponent), fuller palmar side
    mcp_mid = wr + fwd * palm_len
    s.union(loft(wr - fwd * 0.012 * hs, mcp_mid + fwd * 0.004 * hs,
                 [(0, 0.0255 * hs), (0.3, 0.035 * hs), (0.75, 0.0405 * hs), (1, 0.039 * hs)],
                 [(0, 0.0255 * hs), (0.3, 0.033 * hs), (0.75, 0.0395 * hs), (1, 0.038 * hs)],
                 [(0, 0.0135 * hs), (0.35, 0.0155 * hs), (0.8, 0.0135 * hs), (1, 0.0125 * hs)],
                 [(0, 0.0125 * hs), (0.4, 0.0115 * hs), (1, 0.0105 * hs)],
                 outer_dir=ac, front_dir=n, n_front=2.3, n_back=3.0, cap_round=0.8),
            k=0.012 * hs, tag="skin", name="palm")
    R_ = np.stack([ac, n, fwd])
    # thenar (thumb ball) and hypothenar pads
    th_dir = _n(fwd * 0.6 + ac * 0.7)
    s.union(ellipsoid(wr + fwd * 0.036 * hs + ac * 0.021 * hs + n * 0.009 * hs, [0.015 * hs, 0.012 * hs, 0.03 * hs],
                      np.stack([_n(np.cross(n, th_dir)), n, th_dir])), k=0.012 * hs, tag="skin", name="thenar")
    s.union(ellipsoid(wr + fwd * 0.05 * hs - ac * 0.028 * hs + n * 0.008 * hs, [0.011 * hs, 0.009 * hs, 0.034 * hs], R_),
            k=0.01 * hs, tag="skin", name="hypothenar")
    # fingers: three lofted phalanges each, elliptical section (wider than deep)
    for fname in FINGERS:
        pts = [J[f"{fname}{i}.{side}"] for i in (1, 2, 3)] + [J[f"{fname}_end.{side}"]]
        rad = F_RAD[fname]
        for i in range(3):
            a, b = pts[i], pts[i + 1]
            w0, w1 = rad[i] * hs, rad[i + 1] * hs
            last = i == 2
            prof_w = [(0, w0), (0.5, (w0 + w1) * 0.5 * (1.03 if not last else 1.0)), (1, w1)]
            prof_f = [(0, w0 * 0.92), (0.55, (w0 + w1) * 0.5 * (0.98 if not last else 1.02)), (1, w1 * 0.86)]
            prof_b = [(0, w0 * 0.82), (1, w1 * 0.78)]
            s.union(loft(a, b, prof_w, prof_w, prof_f, prof_b, outer_dir=ac, front_dir=n,
                         ext_a=w0 * 0.6, ext_b=(w1 * 0.9 if last else w1 * 0.5), cap_round=1.0 if last else 0.8),
                    k=(0.006 if i == 0 else 0.0025) * hs, tag="skin", name=fname)
            # joint: dorsal knuckle bump + palmar crease
            if i > 0 or fname != "thumb":
                D.append(sdf.brush(a - n * w0 * 0.85, [w0 * 0.9, w0 * 0.7, w0 * 0.7], 0.0007 * hs * (1.6 if i == 0 else 1.0)))
            if i > 0:
                D.append(sdf.stroke(a + n * w0 * 0.95 - ac * w0 * 0.7, a + n * w0 * 0.95 + ac * w0 * 0.7, 0.0011 * hs, -0.0004 * hs))
        # nail: flattened plate on the dorsal distal phalanx, with a soft cuticle groove
        a, b = pts[2], pts[3]
        dd = _n(b - a)
        L = float(np.linalg.norm(b - a))
        w = rad[3] * hs
        nail_c = a + dd * L * 0.66 - n * w * 0.88
        bn = _n(np.cross(dd, -n))
        s.union(ellipsoid(nail_c + n * w * 0.05, [w * 0.72, w * 0.16, L * 0.38], np.stack([bn, -n, dd])), k=0.0016 * hs, tag="nail", name="nail")
        # fingertip pad
        D.append(sdf.brush(a + dd * L * 0.62 + n * w * 0.95, [w * 0.7, w * 0.5, L * 0.35], 0.0009 * hs, np.stack([bn, n, dd])))
    # dorsal: extensor tendons fanning to the knuckles, metacarpal heads
    for fname in ("index", "middle", "ring", "pinky"):
        k1 = J[f"{fname}1.{side}"]
        D.append(sdf.stroke(wr + fwd * 0.02 * hs - n * 0.012 * hs, k1 - fwd * 0.012 * hs - n * 0.0115 * hs, 0.0022 * hs, 0.0006 * hs))
        D.append(sdf.brush(k1 - n * 0.0112 * hs - fwd * 0.003 * hs, [0.0075 * hs, 0.0065 * hs, 0.007 * hs], 0.0016 * hs))
    # palm creases (distal transverse, proximal transverse, thenar "life line")
    pc = lambda a_, f_: wr + ac * a_ * hs + fwd * f_ * hs + n * 0.0155 * hs
    D.append(sdf.stroke(pc(-0.036, 0.082), pc(0.012, 0.088), 0.0012 * hs, -0.0007 * hs))
    D.append(sdf.stroke(pc(-0.02, 0.07), pc(0.03, 0.074), 0.0012 * hs, -0.0006 * hs))
    D.append(sdf.stroke(pc(0.03, 0.068), pc(0.008, 0.025), 0.0013 * hs, -0.0007 * hs))
    # wrist crease
    D.append(sdf.stroke(wr + n * 0.014 * hs - ac * 0.02 * hs, wr + n * 0.014 * hs + ac * 0.02 * hs, 0.0014 * hs, -0.0006 * hs))
    # thumb web
    t1 = J[f"thumb2.{side}"]
    i1 = J[f"index1.{side}"]
    s.union(sdf.capsule(t1 + (i1 - t1) * 0.15, i1 - fwd * 0.012 * hs, 0.0055 * hs), k=0.008 * hs, tag="skin", name="thumb_web")


# ------------------------------------------------------------------- foot --
TOES = [  # (lateral, forward, z) of MTP joint, phalanx lengths, radii
    (-0.021, 0.13, 0.025, (0.036, 0.029), (0.0158, 0.0149, 0.0132)),
    (-0.0005, 0.133, 0.023, (0.027, 0.014, 0.011), (0.0104, 0.0098, 0.0092, 0.0087)),
    (0.0145, 0.127, 0.022, (0.024, 0.012, 0.01), (0.01, 0.0094, 0.0089, 0.0084)),
    (0.0275, 0.118, 0.021, (0.021, 0.011, 0.009), (0.0096, 0.009, 0.0085, 0.008)),
    (0.0385, 0.106, 0.02, (0.018, 0.009, 0.008), (0.0092, 0.0086, 0.0081, 0.0076)),
]


def build_foot(s: Shape, p, J, side):
    fs = p["height"] * p["foot"] / 1.78
    an, sx = foot_frame(J, side)
    D = s.displace
    P = lambda x, y, z: FP(J, side, fs, x, y, z)
    up = v3(0, 0, 1)
    lat = v3(sx, 0, 0)
    ground = 0.0

    def floor_cut(fn):
        return lambda q, fn=fn: sdf.smax(fn(q), ground - q[:, 2], 0.004 * fs)

    # main foot body: heel -> ball, dorsal height high at the instep, flat-ish plantar side
    a = P(0.0, -0.028, 0.032)
    b = P(0.004, 0.12, 0.02)
    ff, flo, fhi = loft(a, b,
                        [(0, 0.029 * fs), (0.25, 0.031 * fs), (0.55, 0.036 * fs), (0.85, 0.043 * fs), (1, 0.045 * fs)],
                        [(0, 0.027 * fs), (0.25, 0.028 * fs), (0.55, 0.033 * fs), (0.85, 0.043 * fs), (1, 0.047 * fs)],
                        [(0, 0.03 * fs), (0.2, 0.052 * fs), (0.45, 0.044 * fs), (0.75, 0.03 * fs), (1, 0.017 * fs)],
                        [(0, 0.034 * fs), (0.3, 0.024 * fs), (0.6, 0.021 * fs), (1, 0.022 * fs)],
                        outer_dir=lat, front_dir=up, n_front=2.2, n_back=2.8, cap_round=0.45, ext_b=0.004 * fs)
    s.union((floor_cut(ff), flo, fhi), k=0.014 * fs, tag="skin", name="foot")
    # medial longitudinal arch: lift the inner sole
    s.sub(ellipsoid(P(-0.03, 0.05, 0.0), [0.022 * fs, 0.055 * fs, 0.014 * fs]), k=0.01 * fs, name="arch")
    # heel pad and achilles insertion
    D.append(sdf.brush(P(0.0, -0.05, 0.03), [0.022 * fs, 0.02 * fs, 0.03 * fs], 0.0016 * fs))
    for o in (-1, 1):
        D.append(sdf.stroke(P(o * 0.016, -0.04, 0.075), P(o * 0.014, -0.045, 0.045), 0.005 * fs, -0.0012 * fs))
    # toes
    for ti, (x, y, z, lens, rad) in enumerate(TOES):
        q = P(x, y, z)
        d = _n(P(x * 1.04, y + 0.01, z - 0.0012) - q)
        pts = [q]
        downs = [-0.06, 0.42, 0.3] if ti else [0.04, 0.1]
        for li, L in enumerate(lens):
            d = _n(d * math.cos(downs[li]) - up * math.sin(downs[li]))
            pts.append(pts[-1] + d * L * fs)
        for i in range(len(pts) - 1):
            w0, w1 = rad[i] * fs, rad[i + 1] * fs if i + 1 < len(rad) else rad[-1] * fs * 0.9
            last = i == len(pts) - 2
            prof_w = [(0, w0), (1, w1)]
            prof_up = [(0, w0 * 0.85), (1, w1 * 0.8)]
            prof_dn = [(0, w0 * 0.9), (1, w1 * 0.95)]
            tf, tlo, thi = loft(pts[i], pts[i + 1], prof_w, prof_w, prof_up, prof_dn, outer_dir=lat, front_dir=up,
                                ext_a=w0 * 0.5, ext_b=w1 * (0.95 if last else 0.4), cap_round=1.0 if last else 0.8)
            s.union((floor_cut(tf), tlo, thi), k=(0.012 if i == 0 else 0.0022) * fs, tag="skin", name="toe")
        # dorsal knuckle (MTP head) and the toe pad
        D.append(sdf.brush(pts[0] + up * rad[0] * fs * 0.8, [rad[0] * fs, rad[0] * fs * 0.8, rad[0] * fs], 0.0012 * fs))
        # toenail
        a_, b_ = pts[-2], pts[-1]
        dd = _n(b_ - a_)
        L = float(np.linalg.norm(b_ - a_))
        w = rad[-1] * fs
        bn = _n(np.cross(up, dd))
        nup = _n(np.cross(dd, bn))
        s.union(ellipsoid(a_ + dd * L * 0.62 + nup * w * 0.78, [w * 0.7, w * 0.18, L * 0.36], np.stack([bn, nup, dd])),
                k=0.001 * fs, tag="nail", name="toenail")
    # dorsal extensor tendons and the ball of the foot
    for x in (-0.019, -0.001, 0.013, 0.025):
        D.append(sdf.stroke(P(0.0, 0.03, 0.07), P(x, 0.128, 0.034), 0.0028 * fs, 0.0006 * fs))
    D.append(sdf.brush(P(-0.012, 0.128, 0.008), [0.03 * fs, 0.02 * fs, 0.012 * fs], 0.0012 * fs))
