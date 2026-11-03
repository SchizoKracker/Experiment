"""Facial features with real depth, placed by Loomis proportions + measured landmarks.

All dimensions are in metres for a 0.234 m head (top of skull -> chin) and scale with
the head.  Volumes give the big forms (orbits, brow ridge, nose, lips, chin); sculpt brushes
(projected onto the skin) add folds, creases and planes.

Loomis checks used for placement (fractions of head height from the top):
  brow line ~0.51, eye line ~0.56, nose base ~0.74, mouth ~0.82, chin 1.0;
  eye width ~1/5 head width, one eye-width between the eyes; nose width ~= inner-canthus
  distance; mouth width ~= inter-pupil distance; ears between brow line and nose base.
"""
from __future__ import annotations

import numpy as np

import sdf
from humanoid import v3
from muscles import Painter
from sdf import F, ellipsoid, round_cone, sphere

CHIN_UNDER = [0.25, 1.0, 0.016]
EYE_X = 0.0325          # half inter-pupillary distance (m)
EYE_R = 0.0122        # eyeball radius (m)


def eye_center(p, J, sx, face_front, ez, k, P_):
    ex = EYE_X * p["eye_spacing"] if "eye_spacing" in p else EYE_X
    lat = ex / (0.234 * k) * 1.0
    surf = face_front(ez, lat)
    # cornea sits ~7 mm behind the facial surface line at that lateral position
    return P_(sx * lat, surf - (0.0055 + EYE_R) / (0.234 * k), ez)


def features(s, p, J, P_, face_front, lm, hh):
    k = hh / 0.234
    m = lambda v: v * k                 # metres for a 0.234 m head -> this head
    hu = lambda v: v * k / hh           # metres -> head units (for P_)
    fem = p["fem"]
    Pn = Painter(s, p["height"])
    D = s.displace
    FWD = v3(0, -1, 0)
    UP = v3(0, 0, 1)
    br = sdf.brush
    st = sdf.stroke
    ez, bz = lm["eye"], lm["brow"]
    nas, tipz, sub, mz = lm["nasion"], lm["nose_tip"], lm["subnasale"], lm["mouth"]
    er = m(EYE_R) * p["eye"]
    brow_k = p["brow"] * (1 - 0.45 * fem)

    def surfpt(lat, zn, dirv=FWD, back=0.0):
        """Skin point on the face at (lateral head-units, zn)."""
        c = Pn.surf(P_(lat, 0.0, zn), dirv)
        return c - _n(dirv) * back

    # ================================================================ EYES
    for sx in (-1, 1):
        X = v3(sx, 0, 0)
        c = eye_center(p, J, sx, face_front, ez, k, P_)
        lat = abs(c[0]) / hh
        # brow bone: a soft shelf along the supraorbital rim; its underside IS the top of the socket
        bf = face_front(bz, lat)
        s.union(ellipsoid(P_(sx * (lat - 0.008), bf - 0.045, bz + 0.012), [m(0.027), m(0.016), m(0.0085) * (0.85 + 0.3 * brow_k)],
                          sdf.euler(0, 0, sx * -6)), k=m(0.009), tag="skin", name="brow_bone")
        # socket: carved below the brow shelf (its top stays under the brow line)
        s.sub(ellipsoid(c + v3(sx * m(0.001), -m(0.013), -m(0.0035)), [m(0.021), m(0.013), m(0.0125)]), k=m(0.008), name="orbit")
        # lids: a shell hugging the eyeball, with an almond-shaped opening
        s.union(sphere(c, er + m(0.0013)), k=m(0.0035), tag="lid", name="lids")
        Rr, aa = m(0.024), m(0.0066) * (1 + 0.08 * fem)
        tilt = np.radians(4 + 6 * p.get("eye_tilt", 0.0)) * sx
        upv = np.array([np.sin(tilt) * -1, 0, np.cos(tilt)], F)
        c1, c2 = c - upv * (Rr - aa), c + upv * (Rr - aa)
        fy = c[1] - er * 0.35

        def almond(q, c1=c1, c2=c2, Rr=Rr, fy=fy):
            d1 = np.linalg.norm(q - c1, axis=1) - Rr
            d2 = np.linalg.norm(q - c2, axis=1) - Rr
            return np.maximum(np.maximum(d1, d2), q[:, 1] - fy)
        s.sub((almond, c - m(0.04), c + m(0.04)), k=m(0.0012), name="eye_opening")
        # upper lid fold & crease, lower lid, tear trough, canthi
        a0 = c + v3(sx * -m(0.012), -er * 0.6, m(0.0085))
        a1 = c + v3(sx * m(0.002), -er * 0.78, m(0.0105))
        a2 = c + v3(sx * m(0.013), -er * 0.55, m(0.0075))
        D.append(st(a0, a1, m(0.0022), -m(0.0011) * (1.2 - 0.4 * fem)))
        D.append(st(a1, a2, m(0.0022), -m(0.0011) * (1.2 - 0.4 * fem)))
        D.append(st(c + v3(-sx * m(0.01), -er * 0.88, -m(0.0058)), c + v3(sx * m(0.011), -er * 0.82, -m(0.006)), m(0.0024), m(0.0007)))
        D.append(st(c + v3(-sx * m(0.012), -er * 0.7, -m(0.011)), c + v3(sx * m(0.004), -er * 0.6, -m(0.016)), m(0.004), -m(0.0012)))
        D.append(br(c + v3(-sx * m(0.0135), -er * 0.75, -m(0.0005)), [m(0.003), m(0.003), m(0.003)], -m(0.0012)))


    # ================================================================ NOSE (driven by the measured profile)
    import anatomy
    zf, frf = anatomy.head_profile.fr_full
    nl, nw, nb = p["nose_len"], p["nose_w"], p["nose_bridge"]
    z0, z1 = nas - 0.012, sub - 0.004
    zs = np.linspace(z0, z1, 40)
    plane = np.array([face_front(zz) for zz in zs])            # face line with the nose removed
    prof = np.interp(zs, zf, frf)                               # measured profile incl. the nose
    proj = np.maximum(prof - plane, 0.0)                        # how far the nose stands off the face
    proj = proj * (0.85 + 0.15 * nl)
    # nose half-width (head units): bridge -> tip -> alae (anatomical ratios)
    t = (zs - z0) / (z1 - z0)
    half_w = np.interp(t, [0, 0.25, 0.55, 0.72, 0.86, 1.0], [0.026, 0.024, 0.032, 0.048, 0.08, 0.062]) * nw
    a_pt = P_(0, plane[0] - 0.012, z0)
    b_pt = P_(0, plane[-1] - 0.012, z1)
    L = float(np.linalg.norm(b_pt - a_pt))
    # loft parameter t runs along a->b; front extent measured from that axis
    axis_front = np.linspace(plane[0] - 0.012, plane[-1] - 0.012, len(zs))
    front_ext = (plane + proj - axis_front) * hh
    s.union(sdf.loft(a_pt, b_pt, list(zip(t, half_w * hh)), list(zip(t, half_w * hh)), list(zip(t, np.maximum(front_ext, 0.004 * hh))),
                     [(0, 0.012 * hh), (1, 0.012 * hh)], outer_dir=(1, 0, 0), front_dir=(0, -1, 0), n_front=2.3, cap_round=0.9),
            k=m(0.006), tag="skin", name="nose")
    tip_i = int(np.argmax(proj))
    tipz_ = float(zs[tip_i])
    tip_f = float(plane[tip_i] + proj[tip_i])
    for sx in (-1, 1):
        # alar crease wrapping each wing, nostrils from below, sidewall valley
        ac = P_(sx * 0.06 * nw, tip_f - hu(0.017), sub - 0.012)
        D.append(st(ac + v3(sx * -m(0.004), m(0.001), m(0.008)), ac + v3(sx * m(0.004), m(0.004), m(0.001)), m(0.0013), -m(0.0011)))
        D.append(st(ac + v3(sx * m(0.004), m(0.004), m(0.001)), ac + v3(sx * m(0.001), m(0.003), -m(0.006)), m(0.0013), -m(0.0011)))
        s.sub(ellipsoid(P_(sx * hu(0.0062) * nw, tip_f - hu(0.021), sub - 0.001), [m(0.0032), m(0.0056), m(0.0022)], sdf.euler(0, 0, sx * 25)),
              k=m(0.001), name="nostril")
        D.append(br(P_(sx * hu(0.0042), tip_f - hu(0.002), tipz_), [m(0.004), m(0.004), m(0.004)], m(0.0006)))   # tip domes
    # nasion: the dip between brow and nose
    D.append(br(P_(0, face_front(nas) + hu(0.002), nas), [m(0.009), m(0.006), m(0.007)], -m(0.0035)))

    # ================================================================ MUSCLES & BONES (brush-built)
    _face_anatomy(Pn, D, p, P_, face_front, lm, hh, m, fem)


def _face_anatomy(Pn, D, p, P_, face_front, lm, hh, m, fem):
    """The few broad forms that shape a face (planes, not individual muscle fibres)."""
    FWD = v3(0, -1, 0)
    bone = (1.0 - 0.4 * fem)

    def dirv(lat, side=0.0):
        sx = np.sign(lat) if lat != 0 else 0.0
        return _n(FWD + v3(sx * (abs(lat) * 2.6 + side), 0, 0))

    def S(lat, zn, side=0.0):
        return Pn.surf(P_(lat, 0.0, zn), dirv(lat, side))

    def bump(lat, zn, r_along, r_across, amp, along=(0, 0, 1), side=0.0, mirror=True):
        for sx in ((-1, 1) if mirror and lat != 0 else (1,)):
            l = lat * sx
            c = S(l, zn, side)
            al = np.asarray(along, F) * np.array([sx, 1, 1], F)
            D.append(sdf.brush(c, [m(r_along), m(r_across), max(m(r_along), m(r_across))], m(amp), _rot_from(dirv(l, side), _n(al))))

    def line(pts, radius, amp, side=0.0, mirror=True):
        for sx in ((-1, 1) if mirror else (1,)):
            q = [S(l * sx, z, side) for l, z in pts]
            for a, b in zip(q[:-1], q[1:]):
                D.append(sdf.stroke(a, b, m(radius), m(amp)))

    ez, mz, sub = lm["eye"], lm["mouth"], lm["subnasale"]
    # cheekbones (front-lateral corner of the face) and the plane change beneath them
    bump(0.215, 0.625, 0.022, 0.014, 0.003 * bone, along=(1, 0, 0.35), side=0.35)
    bump(0.235, 0.77, 0.024, 0.022, -0.0022 * (1 - 0.6 * fem), side=0.5)
    # temples
    bump(0.285, 0.42, 0.024, 0.026, -0.002, side=0.8)
    # tear trough, nasolabial fold (soft)
    line([(0.075, ez + 0.035), (0.11, ez + 0.065), (0.145, ez + 0.075)], 0.007, -0.0009)
    line([(0.075, 0.72), (0.1, 0.78), (0.125, 0.85)], 0.0055, -0.0011 * (1 - 0.3 * fem))
    # jaw angle and chin
    bump(0.285, 0.9, 0.016, 0.016, 0.0016 * bone * p["jaw"], side=1.2)
    bump(0.0, 0.95, 0.013, 0.012, 0.0018 * p["chin"], mirror=False)

    # ---- lips: two clean forms + the mouth line; no puckering mound
    lp = p["lips"] * (1 + 0.15 * fem)
    line([(0.0, mz - 0.026), (0.03, mz - 0.024), (0.07, mz - 0.014), (0.105, mz - 0.003)], 0.0048, 0.0034 * lp)   # upper lip
    line([(0.0, mz + 0.022), (0.04, mz + 0.021), (0.08, mz + 0.012), (0.105, mz + 0.003)], 0.0055, 0.0033 * lp)  # lower lip
    line([(0.0, mz + 0.001), (0.04, mz), (0.08, mz + 0.002), (0.112, mz + 0.007)], 0.0022, -0.0034)             # mouth line
    line([(0.0, mz - 0.036), (0.022, mz - 0.04), (0.055, mz - 0.03), (0.1, mz - 0.008)], 0.0022, 0.0006)        # vermilion border / cupid's bow
    for sx in (-1, 1):
        D.append(sdf.stroke(S(sx * 0.021, sub + 0.006), S(sx * 0.023, mz - 0.038), m(0.002), m(0.0005)))       # philtral columns
    bump(0.115, mz + 0.006, 0.005, 0.005, -0.0012)                                                          # mouth corners
    line([(-0.06, mz + 0.058), (0.0, mz + 0.063), (0.06, mz + 0.058)], 0.0055, -0.002, mirror=False)        # under-lip groove
    s_ = Pn.base
    s_.prims.append(sdf.Prim(*_with_bounds(ellipsoid(P_(0.0, face_front(0.98) - CHIN_UNDER[0], CHIN_UNDER[1]), [m(0.024), m(CHIN_UNDER[2]), m(0.011)])), op="union", k=m(0.012), tag="skin", name="chin_under"))


def _with_bounds(prim):
    return prim[0], prim[1], prim[2]


def _n(v):
    v = np.asarray(v, F)
    return v / (np.linalg.norm(v) + 1e-9)


def _rot_from(nrm, along):
    a = _n(along - nrm * np.dot(along, nrm))
    b = np.cross(nrm, a)
    return np.stack([a, b, nrm])
