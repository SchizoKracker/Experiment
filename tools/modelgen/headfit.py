"""Measured facial landmarks and fitted head forms (derived from the orthographic reference, 4.webp).

All values are in head units: lat = lateral from the midline, fwd = forward from the head's
centre plane (x0 = midpoint of forehead and back of skull in the side view), zn = down from the
top of the skull (chin = 1).  Measured by reading feature pixels on the front and side views:
  front: x_orig = 150 + X/2, y_orig = 180 + Y/2 (2x zoom); midline at x = 386.5 (pupil midpoint)
  side:  x_orig = 620 + X/2 ...; fwd = (887 - x) / 552;  zn = (y - 10) / 552
"""
from __future__ import annotations

import numpy as np

LANDMARKS = dict(
    pupil_lat=0.140, pupil_zn=0.52, cornea_fwd=0.339,
    canthus_in_lat=0.066, canthus_out_lat=0.216, eye_open_h=0.062, eye_open_up=0.029, eye_open_lo=0.033,
    brow_inner_zn=0.475, brow_arch_zn=0.42, brow_bone_zn=0.48,
    nose_tip_zn=0.68, alae_lat=0.074, alae_zn=0.697, alae_fwd=0.357, subnasale_zn=0.72,
    bridge_lat=0.034,
    mouth_corner_lat=0.107, mouth_corner_fwd=0.339, stomion_zn=0.832,
    upper_lip_top_zn=0.806, lower_lip_bottom_zn=0.873,
    chin_front_fwd=0.393, chin_front_zn=0.924, chin_bottom_zn=0.985,
    jaw_angle_fwd=0.058, jaw_angle_zn=0.88, jaw_angle_lat=0.29,
    cheekbone_lat=0.165, cheekbone_zn=0.62, cheekbone_fwd=0.275,
    ear_top_zn=0.426, ear_bottom_zn=0.67, ear_front_fwd=-0.005, ear_back_fwd=-0.178,
)


# mandible lower border, side view: chin bottom (x=720,y=575) -> jaw angle (x=855,y=525)
JAW_BORDER = dict(chin_fwd=0.30, chin_zn=1.045, angle_fwd=0.058, angle_zn=0.933, curve=2.0, soft=0.022)


def fit_crown(z, w, fr, bk, z_max=0.42):
    """Least-squares superellipsoid fit to the top of the measured front + side silhouettes.

    Model: horizontal section at height zn is an ellipse with half-width a*r, front extent
    yc + b*r, back extent b*r - yc, where r = (1 - |(zn - z0)/c|^e)^(1/e).
    Returns (z0, c, a, b, yshift, e)."""
    m = z < z_max
    zz, ww, ff, bb = z[m], w[m], fr[m], bk[m]
    best = None
    for e in (2.0, 2.2, 2.4, 2.6, 2.8):
        for z0 in np.linspace(0.25, 0.6, 36):
            c = z0                                   # the top of the ellipsoid is the top of the skull (zn = 0)
            r = np.clip(1 - np.abs((zz - z0) / c) ** e, 0, 1) ** (1 / e)
            # closed-form least squares for a, and for b & shift from front/back extents
            a = float(np.sum(ww * r) / max(np.sum(r * r), 1e-9))
            bsum = ff + bb                             # = 2 b r
            b = float(np.sum(bsum * r) / max(np.sum(4 * r * r), 1e-9) * 2)
            ysh = float(np.mean((ff - bb) / 2))
            err = np.mean((a * r - ww) ** 2) + np.mean((b * r + ysh - ff) ** 2) + np.mean((b * r - ysh - bb) ** 2)
            if best is None or err < best[0]:
                best = (err, z0, c, a, b, ysh, e)
    return best[1:]
