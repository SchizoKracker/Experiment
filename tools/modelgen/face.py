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
# brow arch (front view of 4.webp: brow inner end lat .080 zn .471, peak .157/.426, tail .238/.435; the bone
# ridge sits just under the hair): lat, zn, forward bulge (hu).  Medial end fades into the nose bridge.
ARCH_LAT = [0.030, 0.060, 0.100, 0.155, 0.200, 0.238]
ARCH_ZN = [0.490, 0.474, 0.455, 0.437, 0.442, 0.468]
ARCH_AMP = [0.000, 0.004, 0.009, 0.011, 0.009, 0.004]


LID_EDGES = {}
CARUNCLE = {}
LID_ARCS = {}                          # sx -> (upper margin pts, lower margin pts, eye centre), filled by features()
# eyebrow hair arch measured on the front view of 4.webp (lat, zn)
BROW_HAIR_LAT = [0.060, 0.080, 0.110, 0.157, 0.200, 0.240]
BROW_HAIR_ZN = [0.478, 0.470, 0.450, 0.427, 0.427, 0.438]


def tube(pts, r):
    """solid tube along a polyline (SDF)"""
    P = np.asarray(pts, F)
    lo, hi = P.min(0) - 2 * r, P.max(0) + 2 * r

    def f(q):
        best = np.full(len(q), np.inf, F)
        for i in range(len(P) - 1):
            ab = P[i + 1] - P[i]
            t = np.clip(((q - P[i]) @ ab) / float(ab @ ab), 0, 1)
            best = np.minimum(best, np.linalg.norm(q - (P[i] + np.outer(t, ab)), axis=1))
        return best - r
    return f, lo, hi


def ridge(pts, amps, r):
    """Smooth ridge along a polyline: Gaussian in the distance to the line, amplitude interpolated
    along it (no double-counting at the joints, unlike chained strokes)."""
    P = np.asarray(pts, F)
    A = np.asarray(amps, F)
    sgn = -1.0 if A.min() < 0 else 1.0
    A = np.abs(A)
    lo, hi = P.min(0) - 3.2 * r, P.max(0) + 3.2 * r

    def f(p, d):
        out = np.zeros(len(p), F)
        m = np.all((p > lo) & (p < hi), axis=1)
        if not np.any(m):
            return out
        q = p[m]
        v = np.zeros(len(q), F)
        for i in range(len(P) - 1):
            ab = P[i + 1] - P[i]
            t = np.clip(((q - P[i]) @ ab) / float(ab @ ab), 0, 1)
            dist = np.linalg.norm(q - (P[i] + np.outer(t, ab)), axis=1)
            v = np.maximum(v, (A[i] + (A[i + 1] - A[i]) * t) * np.exp(-(dist / r) ** 2))   # continuous at joints
        out[m] = -v * sgn
        return out
    f.lo, f.hi = lo, hi
    return f


def eye_center(p, J, sx, face_front, ez, k, P_):
    """Eyeball centre: the cornea sits just under the skin line so the open lids show the whole iris."""
    lat = LMK["pupil_lat"]
    # seated ~1 mm deeper than the skin line so the lids overlap the eyeball instead of framing it
    return P_(sx * lat, face_front(LMK["pupil_zn"], lat) - 0.010 - EYE_R_HU * p["eye"], LMK["pupil_zn"])


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
    # lid apex heights measured on the front view (4.webp): upper 0.029, lower 0.033 above/below the pupil line
    ha_u, ha_l = LMK["eye_open_up"] * hh, LMK["eye_open_lo"] * hh
    ha = ha_u
    Ru = (hw * hw + ha_u * ha_u) / (2 * ha_u)        # circle through the canthi and the upper lid apex
    Rl = (hw * hw + ha_l * ha_l) / (2 * ha_l)        # ... and the lower lid apex
    brow_z = top - LMK["brow_bone_zn"] * hh
    for sx in (-1, 1):
        c = eye_center(p, J, sx, face_front, 0, 0, P_)
        # socket: carved around the eye, clipped by the brow line (the brow bone defines its top)
        # shallow socket hollow around the lids, clipped by the brow line
        E = ellipsoid(c + v3(0, -(er + 0.03 * hh), 0.006 * hh), [0.08 * hh, 0.032 * hh, 0.045 * hh])

        def socket(q, E=E[0]):
            # the socket's roof is the underside of the brow arch (measured arch + ridge half-thickness)
            bz = top - (np.interp(np.abs(q[:, 0]) / hh, ARCH_LAT, ARCH_ZN) + 0.032) * hh
            return sdf.smax(E(q), q[:, 2] - bz, 0.012 * hh)
        s.sub((socket, E[1], E[2]), k=0.016 * hh, name="socket")
        # lids hug the eyeball; the almond opening exposes it
        s.union(sphere(c, er + 0.006 * hh), k=0.01 * hh, tag="lid", name="lids")
        tilt = np.radians(3.0 + 4.0 * p.get("eye_tilt", 0.0)) * sx
        upv = np.array([-np.sin(tilt), 0, np.cos(tilt)], F)
        oc = c + v3(sx * ((LMK["canthus_in_lat"] + LMK["canthus_out_lat"]) / 2 - lat) * hh, 0, 0)
        c1, c2 = oc - upv * (Ru - ha_u), oc + upv * (Rl - ha_l)
        fy = c[1] - er * 0.45

        def almond(q, c1=c1, c2=c2, Ru=Ru, Rl=Rl, fy=fy):
            # 2D almond (in the x-z plane) extruded front-to-back: the opening keeps its full size at the lid surface
            d1 = np.hypot(q[:, 0] - c1[0], q[:, 2] - c1[2]) - Ru
            d2 = np.hypot(q[:, 0] - c2[0], q[:, 2] - c2[2]) - Rl
            return np.maximum(np.maximum(d1, d2), q[:, 1] - fy)
        s.sub((almond, c - 0.2 * hh, c + 0.2 * hh), k=0.004 * hh, name="eye_opening")
        # ---- eyelids: rounded lid margins that hug the eyeball, upper-lid fold over the crease
        rv = np.array([upv[2], 0, -upv[0]], F)
        ts = np.linspace(-1, 1, 15)

        def arc(off_u, off_l, rad):
            up, lo_ = [], []
            for t_ in ts:
                x = t_ * hw
                zu = np.sqrt(max(Ru * Ru - x * x, 0)) - (Ru - ha_u)
                zl = -(np.sqrt(max(Rl * Rl - x * x, 0)) - (Rl - ha_l))
                for z_, off, L in ((zu, off_u, up), (zl, -off_l, lo_)):
                    q = oc + rv * x + upv * (z_ + off)
                    # project forward (-y) onto the sphere of radius rad: the front of the eye, not its top
                    dx, dz = q[0] - c[0], q[2] - c[2]
                    dy = np.sqrt(max(rad * rad - dx * dx - dz * dz, (0.25 * rad) ** 2))
                    L.append(np.array([q[0], c[1] - dy, q[2]], F))
            return up, lo_
        mr = 0.0055 * hh                                   # lid margin radius (~1.3 mm)
        up_m, lo_m = arc(mr * 0.7, mr * 0.6, er + 0.0035 * hh)
        LID_ARCS[sx] = (np.array(up_m), np.array(lo_m), c.copy())
        # lacrimal caruncle: small pink mound filling the medial corner of the opening
        cpos = c + v3(-sx * (lat - LMK["canthus_in_lat"] - 0.009) * hh, -er * 0.62, 0.002 * hh)
        CARUNCLE[sx] = cpos
        s.union(ellipsoid(cpos, [0.0075 * hh, 0.006 * hh, 0.0065 * hh]), k=0.004 * hh, tag="lid", name="caruncle")
        # the opening edge itself (just inside the lid), on the eyeball surface: lash roots start here
        up_e, lo_e = arc(0.0012 * hh, 0.0012 * hh, er)
        LID_EDGES[sx] = (np.array(up_e), np.array(lo_e))
        s.union(tube(up_m, mr), k=0.004 * hh, tag="lid", name="upper_lid_margin")
        s.union(tube(lo_m, mr * 0.85), k=0.004 * hh, tag="lid", name="lower_lid_margin")
        # crease (groove) and the fold of skin above it, following the opening; lower-lid groove below
        up_c, lo_c = arc(0.022 * hh, 0.02 * hh, er + 0.008 * hh)
        up_f, _ = arc(0.034 * hh, 0.0, er + 0.012 * hh)
        taper = list(np.clip(1 - np.abs(ts) ** 3, 0, 1))
        D.append(ridge(up_c, [-0.005 * hh * a for a in taper], 0.008 * hh))
        D.append(ridge(up_f, [0.0045 * hh * a for a in taper], 0.011 * hh))
        D.append(ridge(lo_c, [-0.003 * hh * a for a in taper], 0.009 * hh))

    # ============================================================ BROW RIDGE
    # supraorbital arch measured on the front view (follows the brow, ~2 mm under the hair line), running
    # down medially into the side wall of the nose bridge; rounded forward bulge, strongest over the iris,
    # nearly zero at the midline so the measured glabella/nasion profile is kept.
    for sx in (-1, 1):
        pts = [Pn.surf(P_(sx * la, 0, zn), v3(0, -1, 0)) for la, zn in zip(ARCH_LAT, ARCH_ZN)]
        D.append(ridge(pts, [a * hh * (1 - 0.25 * fem) for a in ARCH_AMP], 0.028 * hh))

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
    # tip lobule: the paired domal cartilages give the tip a soft ball with a faint central division
    tz = LMK["nose_tip_zn"] + 0.004
    for sx in (-1, 1):
        s.union(ellipsoid(P_(sx * 0.014 * nw, tip_f - 0.026, tz), [0.027 * hh, 0.026 * hh, 0.026 * hh]),
                k=0.012 * hh, tag="skin", name="nose_tip")
        # alar wings: rounded masses beside the tip, framed by the alar crease
        s.union(ellipsoid(P_(sx * (LMK["alae_lat"] - 0.017) * nw, LMK["alae_fwd"] - 0.02, LMK["alae_zn"] + 0.002),
                          [0.021 * hh, 0.03 * hh, 0.02 * hh], sdf.euler(0, 0, sx * 15)), k=0.014 * hh, tag="skin", name="ala")
    for sx in (-1, 1):
        # alar crease: groove separating each wing from the cheek, following the measured ala position
        ac = P_(sx * LMK["alae_lat"], LMK["alae_fwd"], LMK["alae_zn"])
        D.append(sdf.stroke(ac + v3(-sx * 0.012 * hh, 0, 0.03 * hh), ac + v3(sx * 0.004 * hh, 0.004 * hh, 0.004 * hh), 0.006 * hh, -0.006 * hh))
        D.append(sdf.stroke(ac + v3(sx * 0.004 * hh, 0.004 * hh, 0.004 * hh), ac + v3(-sx * 0.004 * hh, 0.0, -0.024 * hh), 0.006 * hh, -0.006 * hh))
        # nostril openings on the underside
        s.sub(ellipsoid(P_(sx * 0.024 * nw, tip_f - 0.075, LMK["subnasale_zn"] + 0.002), [0.011 * hh, 0.021 * hh, 0.006 * hh], sdf.euler(0, 0, sx * 22)),
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

    # measured midline lip profile (skin upper lip slope -> vermilion -> stomion -> lower lip -> labiomental)
    lz0 = float(LMK["subnasale_zn"]) + 0.02
    zl = zgrid[(zgrid > lz0) & (zgrid < 0.905)]
    al = np.maximum(np.interp(zl, zgrid, amp_mid), 0.0)
    w_ = np.clip(np.minimum((zl - lz0) / 0.012, (0.905 - zl) / 0.012), 0, 1)
    al = al * w_ * w_ * (3 - 2 * w_)
    seg = (zl > 0.82) & (zl < 0.86)
    szp = float(zl[seg][np.argmin(al[seg])]) if np.any(seg) else sz

    def lips(q, d):
        u, f, zn = local(q)
        out = np.zeros(len(q), F)
        m = (f > 0.15) & (zn > lz0 - 0.01) & (zn < 0.92) & (np.abs(u) < mcl * 1.35)
        if not np.any(m):
            return out
        uu, zz = np.abs(u[m]), zn[m]
        t = np.clip(uu / mcl, 0, 1.35)
        # the lips thin toward the corners: compress the profile toward the mouth line
        hs = np.maximum(1 - t ** 2.4, 0.0) ** 0.45 * 0.8 + 0.2
        # cupid's bow: the upper border rises over the philtral columns, dips at the midline
        bow = np.where(zz < szp, 0.004 * np.exp(-((uu - 0.026) / 0.014) ** 2) - 0.003 * np.exp(-(uu / 0.01) ** 2), 0.0)
        z_eff = szp + (zz + bow - szp) / hs
        a = np.interp(z_eff, zl, al, left=0.0, right=0.0)
        lat_t = np.clip(1 - (t / 1.08) ** 2, 0, 1) ** 0.7
        stom = 0.011 * np.exp(-((zz - szp) / 0.0045) ** 2) * np.clip(1 - t / 1.1, 0, 1) ** 0.5
        out[m] = -(a * lat_t * lp * 1.08 - stom) * hh
        return out
    c_m = P_(0, face_front(sz), sz)
    lips.lo = c_m - np.array([0.2, 0.2, 0.15], F) * hh
    lips.hi = c_m + np.array([0.2, 0.2, 0.15], F) * hh
    D.append(lips)
    for sx in (-1, 1):
        # commissures sit on the dental arch, measured depth
        D.append(sdf.brush(P_(sx * mcl, LMK["mouth_corner_fwd"], sz), [0.018 * hh, 0.018 * hh, 0.018 * hh], -0.006 * hh))
        # philtral columns
        D.append(sdf.stroke(Pn.surf(P_(sx * 0.018, 0, LMK["subnasale_zn"] + 0.01), v3(0, -1, 0)),
                            Pn.surf(P_(sx * 0.026, 0, uz - 0.002), v3(0, -1, 0)), 0.01 * hh, 0.009 * hh))

    for sx in (-1, 1):
        # modiolus: the small muscular knot just lateral to each mouth corner the lips tuck into
        mpt = Pn.surf(P_(sx * (mcl + 0.02), face_front(sz, mcl + 0.02) - 0.012, sz - 0.003), _n(v3(sx * 0.5, -1, 0)))
        D.append(sdf.brush(mpt, [0.018 * hh, 0.016 * hh, 0.02 * hh], 0.004 * hh))
        # nasolabial transition: a very soft groove from the alar base toward the mouth corner
        nl = [Pn.surf(P_(sx * la, face_front(zn, la) - 0.012, zn), _n(v3(sx * 0.5, -1, 0))) for la, zn in
              ((LMK["alae_lat"] + 0.012, LMK["alae_zn"] + 0.008), (LMK["alae_lat"] + 0.03, 0.77), (mcl + 0.03, sz + 0.012))]
        D.append(ridge(nl, [-0.0028 * hh, -0.0022 * hh, -0.0008 * hh], 0.012 * hh))

    # philtrum groove between the columns, ending in the cupid's-bow dip
    D.append(sdf.stroke(Pn.surf(P_(0, 0, LMK["subnasale_zn"] + 0.008), v3(0, -1, 0)), Pn.surf(P_(0, 0, uz + 0.002), v3(0, -1, 0)),
                        0.011 * hh, -0.008 * hh))

    # ============================================================ SIDE PLANES
    # the face turns from its front plane to the side plane along the line temple -> outer orbital rim ->
    # cheekbone -> cheek; a soft ridge on that line, the temple and the cheek below the zygoma slightly hollow
    SIDE = [(0.285, 0.40, 0.004), (0.262, 0.50, 0.006), (0.245, 0.60, 0.009), (0.238, 0.68, 0.006), (0.24, 0.78, 0.002)]
    for sx in (-1, 1):
        dirn = _n(v3(sx * 0.75, -1, 0))
        pts = [Pn.surf(P_(sx * la, face_front(zn, la) - 0.012, zn), dirn) for la, zn, _ in SIDE]
        D.append(ridge(pts, [a * hh for _, _, a in SIDE], 0.03 * hh))
        D.append(sdf.brush(Pn.surf(P_(sx * 0.30, face_front(0.47, 0.30) - 0.012, 0.47), _n(v3(sx, -0.6, 0))), [0.04 * hh, 0.04 * hh, 0.045 * hh], -0.006 * hh))
        D.append(sdf.brush(Pn.surf(P_(sx * 0.20, face_front(0.75, 0.20) - 0.012, 0.75), _n(v3(sx * 0.6, -1, 0))), [0.035 * hh, 0.03 * hh, 0.04 * hh], -0.005 * hh))

    # ============================================================ BROAD FORMS
    for sx in (-1, 1):
        cb = P_(sx * LMK["cheekbone_lat"], LMK["cheekbone_fwd"], LMK["cheekbone_zn"])
        cb = Pn.surf(cb, _n(v3(sx * 0.7, -1, 0)))
        D.append(sdf.brush(cb, [0.09 * hh, 0.06 * hh, 0.07 * hh], 0.008 * hh * (1 - 0.3 * fem)))

