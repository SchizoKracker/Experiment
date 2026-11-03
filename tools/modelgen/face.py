"""Facial features placed from measured landmarks (headfit.LANDMARKS), built from measured profiles.

Principles (learned the hard way):
  * nothing may stick out past the measured front/side silhouettes except where the reference does
  * the brow line defines the top of the eye socket (the socket carve is clipped by it)
  * nose and lips get their projection from the measured side profile, their widths from the front view
  * a few broad, low-amplitude forms only (cheekbone plane, chin) - no muscle painting
"""
from __future__ import annotations

import numpy as np

import sdf
from headfit import LANDMARKS as LMK
from humanoid import v3
from muscles import Painter
from sdf import F, ellipsoid, sphere

CHIN_UNDER = [0.25, 1.0, 0.016]
EYE_R_HU = 0.052                       # eyeball radius in head units (~12.2 mm)


def eye_center(p, J, sx, face_front, ez, k, P_):
    """Eyeball centre: the cornea sits just under the skin line so the open lids show the whole iris."""
    lat = LMK["pupil_lat"]
    return P_(sx * lat, face_front(LMK["pupil_zn"], lat) - 0.006 - EYE_R_HU * p["eye"], LMK["pupil_zn"])


def _n(v):
    v = np.asarray(v, F)
    return v / (np.linalg.norm(v) + 1e-9)


def features(s, p, J, P_, face_front, lm, hh):
    import anatomy
    fem = p["fem"]
    Pn = Painter(s, p["height"])
    D = s.displace
    top = float(J["head_top"][2])
    yc = float(J["head"][1])

    def local(q):
        return q[:, 0] / hh, (yc - q[:, 1]) / hh, (top - q[:, 2]) / hh

    # ============================================================ EYES
    er = EYE_R_HU * p["eye"] * hh
    lat = LMK["pupil_lat"]
    hw = (LMK["canthus_out_lat"] - LMK["canthus_in_lat"]) / 2 * hh * (0.95 + 0.05 * p["eye"])
    ha = LMK["eye_open_h"] / 2 * hh
    Rr = (hw * hw + ha * ha) / (2 * ha)              # circle through the canthi and the lid apex
    brow_z = top - LMK["brow_bone_zn"] * hh
    for sx in (-1, 1):
        c = eye_center(p, J, sx, face_front, 0, 0, P_)
        # socket: carved around the eye, clipped by the brow line (the brow bone defines its top)
        # shallow socket hollow around the lids, clipped by the brow line
        E = ellipsoid(c + v3(0, -(er + 0.03 * hh), 0.006 * hh), [0.08 * hh, 0.032 * hh, 0.045 * hh])

        def socket(q, E=E[0], bz=brow_z):
            return sdf.smax(E(q), q[:, 2] - bz, 0.012 * hh)
        s.sub((socket, E[1], E[2]), k=0.016 * hh, name="socket")
        # lids hug the eyeball; the almond opening exposes it
        s.union(sphere(c, er + 0.006 * hh), k=0.01 * hh, tag="lid", name="lids")
        tilt = np.radians(3.0 + 4.0 * p.get("eye_tilt", 0.0)) * sx
        upv = np.array([-np.sin(tilt), 0, np.cos(tilt)], F)
        oc = c + v3(sx * ((LMK["canthus_in_lat"] + LMK["canthus_out_lat"]) / 2 - lat) * hh, 0, 0)
        c1, c2 = oc - upv * (Rr - ha), oc + upv * (Rr - ha)
        fy = c[1] - er * 0.45

        def almond(q, c1=c1, c2=c2, Rr=Rr, fy=fy):
            # 2D almond (in the x-z plane) extruded front-to-back: the opening keeps its full size at the lid surface
            d1 = np.hypot(q[:, 0] - c1[0], q[:, 2] - c1[2]) - Rr
            d2 = np.hypot(q[:, 0] - c2[0], q[:, 2] - c2[2]) - Rr
            return np.maximum(np.maximum(d1, d2), q[:, 1] - fy)
        s.sub((almond, c - 0.2 * hh, c + 0.2 * hh), k=0.004 * hh, name="eye_opening")
        # upper-lid crease (one soft stroke following the opening, ~5 mm above it)
        pts = [oc + v3(sx * t * hw * 0.95, -er * (0.9 - 0.25 * t * t), ha + 0.02 * hh - 0.012 * hh * t * t) for t in (-1, -0.4, 0.3, 1)]
        for a_, b_ in zip(pts[:-1], pts[1:]):
            D.append(sdf.stroke(a_, b_, 0.009 * hh, -0.0045 * hh * (1 - 0.3 * fem)))

    # ============================================================ NOSE (profile projection + measured widths)
    zf, frf = anatomy.head_profile.fr_full
    nw = p["nose_w"]
    z0, z1 = lm["nasion"] - 0.012, LMK["subnasale_zn"] + 0.008
    zs = np.linspace(z0, z1, 48)
    plane = np.array([face_front(zz) for zz in zs])
    proj = np.maximum(np.interp(zs, zf, frf) - plane, 0.0) * (0.85 + 0.15 * p["nose_len"])
    half_w = np.interp(zs, [z0, 0.6, 0.65, LMK["nose_tip_zn"], LMK["alae_zn"], z1],
                       [LMK["bridge_lat"], LMK["bridge_lat"] * 0.95, 0.044, 0.056, LMK["alae_lat"] * 1.08, 0.07]) * nw
    t = (zs - z0) / (z1 - z0)
    a_pt = P_(0, plane[0] - 0.02, z0)
    b_pt = P_(0, plane[-1] - 0.02, z1)
    axis_front = np.linspace(plane[0] - 0.02, plane[-1] - 0.02, len(zs))
    front_ext = (plane + proj - axis_front) * hh
    s.union(sdf.loft(a_pt, b_pt, list(zip(t, half_w * hh)), list(zip(t, half_w * hh)), list(zip(t, np.maximum(front_ext, 0.02 * hh))),
                     [(0, 0.02 * hh), (1, 0.02 * hh)], outer_dir=(1, 0, 0), front_dir=(0, -1, 0), n_front=2.4, cap_round=0.9),
            k=0.025 * hh, tag="skin", name="nose")
    tip_i = int(np.argmax(proj))
    tip_f = float(plane[tip_i] + proj[tip_i])
    for sx in (-1, 1):
        # alar crease: groove separating each wing from the cheek, following the measured ala position
        ac = P_(sx * LMK["alae_lat"], LMK["alae_fwd"], LMK["alae_zn"])
        D.append(sdf.stroke(ac + v3(-sx * 0.012 * hh, 0, 0.03 * hh), ac + v3(sx * 0.004 * hh, 0.004 * hh, 0.004 * hh), 0.006 * hh, -0.006 * hh))
        D.append(sdf.stroke(ac + v3(sx * 0.004 * hh, 0.004 * hh, 0.004 * hh), ac + v3(-sx * 0.004 * hh, 0.0, -0.024 * hh), 0.006 * hh, -0.006 * hh))
        # nostril openings on the underside
        s.sub(ellipsoid(P_(sx * 0.028 * nw, tip_f - 0.08, LMK["subnasale_zn"] + 0.004), [0.012 * hh, 0.022 * hh, 0.006 * hh], sdf.euler(0, 0, sx * 22)),
              k=0.004 * hh, name="nostril")
    # nasion dip between brow and bridge
    D.append(sdf.brush(P_(0, face_front(lm["nasion"]) + 0.006, lm["nasion"]), [0.035 * hh, 0.025 * hh, 0.03 * hh], -0.012 * hh))

    # ============================================================ LIPS (measured profile projection, measured width)
    uz, lz, sz = LMK["upper_lip_top_zn"], LMK["lower_lip_bottom_zn"], LMK["stomion_zn"]
    mcl = LMK["mouth_corner_lat"]
    zgrid = np.linspace(LMK["subnasale_zn"], 0.92, 120)
    base = np.array([face_front(zz) for zz in zgrid])
    amp_mid = np.maximum(np.interp(zgrid, zf, frf) - base, 0.0)
    lp = p["lips"] * (1 + 0.1 * fem)

    def lips(q, d):
        u, f, zn = local(q)
        out = np.zeros(len(q), F)
        m = (f > 0.15) & (zn > uz - 0.03) & (zn < lz + 0.04) & (np.abs(u) < mcl * 1.15)
        if not np.any(m):
            return out
        uu, zz = np.abs(u[m]), zn[m]
        A = np.interp(zz, zgrid, amp_mid) * lp
        # lateral falloff toward the corners; the lower lip ends a little inside the upper
        lat_ext = np.where(zz < sz, mcl, mcl * 0.9)
        L = np.clip(1 - (uu / lat_ext) ** 2.4, 0, 1) ** 0.5
        # vermilion edges: crisp upper border (with a slight cupid's bow dip at the midline), soft lower edge
        bow = uz + 0.004 * np.exp(-(uu / 0.012) ** 2) - 0.004 * np.exp(-((uu - 0.03) / 0.02) ** 2)
        edge = np.clip((zz - bow) / 0.006, 0, 1) * np.clip((lz - zz) / 0.012, 0, 1)
        stom = 0.016 * np.exp(-((zz - sz) / 0.0055) ** 2) * (0.4 + 0.6 * np.clip(1 - uu / (mcl * 1.1), 0, 1))
        band = np.clip((zz - bow) / 0.004, 0, 1) * np.clip((lz - zz) / 0.008, 0, 1)
        out[m] = -((A * 0.75 + 0.022 * band) * L * (0.45 + 0.55 * edge) - stom) * hh
        return out
    c_m = P_(0, face_front(sz), sz)
    lips.lo = c_m - np.array([0.2, 0.2, 0.15], F) * hh
    lips.hi = c_m + np.array([0.2, 0.2, 0.15], F) * hh
    D.append(lips)
    for sx in (-1, 1):
        # commissures sit on the dental arch, measured depth
        D.append(sdf.brush(P_(sx * mcl, LMK["mouth_corner_fwd"], sz), [0.018 * hh, 0.018 * hh, 0.018 * hh], -0.006 * hh))
        # philtral columns
        D.append(sdf.stroke(Pn.surf(P_(sx * 0.02, 0, LMK["subnasale_zn"] + 0.012), v3(0, -1, 0)),
                            Pn.surf(P_(sx * 0.024, 0, uz - 0.004), v3(0, -1, 0)), 0.008 * hh, 0.0025 * hh))

    # ============================================================ BROAD FORMS
    for sx in (-1, 1):
        cb = P_(sx * LMK["cheekbone_lat"], LMK["cheekbone_fwd"], LMK["cheekbone_zn"])
        cb = Pn.surf(cb, _n(v3(sx * 0.7, -1, 0)))
        D.append(sdf.brush(cb, [0.09 * hh, 0.06 * hh, 0.07 * hh], 0.008 * hh * (1 - 0.3 * fem)))
    s.prims.append(sdf.Prim(*ellipsoid(P_(0.0, face_front(0.98) - CHIN_UNDER[0], CHIN_UNDER[1]), [0.103 * hh, CHIN_UNDER[2] / 0.234 * hh, 0.047 * hh]),
                            op="union", k=0.05 * hh, tag="skin", name="chin_under"))
