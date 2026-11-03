"""Groom presets built on hair.py: scalp styles, beards, brows, lashes."""
from __future__ import annotations

import numpy as np

import anatomy
import hair as hr
from hair import HeadFrame, _n
from sdf import F

LM = anatomy.REF_HEAD["landmarks"]
RZ = np.array(anatomy.REF_HEAD["z"])
RW = np.array(anatomy.REF_HEAD["w"])


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def scalp_mask(hf: HeadFrame, front_zn=0.24, nape_zn=0.86, sideburn=0.6):
    ear_zn = (LM["ear_top"] + LM["ear_bottom"]) * 0.5
    def m(p):
        u, f, zn = hf.local(p)
        # hairline height as a function of how far forward the point is
        zlim = front_zn + (0.45 - front_zn) * smooth(0.32, 0.05, f) + (nape_zn - 0.45) * smooth(-0.02, -0.3, f)
        # temples recede slightly; sideburns come down in front of the ear
        zlim = zlim + 0.04 * smooth(0.12, 0.3, np.abs(u)) * smooth(0.36, 0.2, f)
        side = (np.abs(f + 0.02) < 0.08) & (np.abs(u) > 0.2)
        zlim = np.where(side, np.maximum(zlim, sideburn), zlim)
        w_here = np.interp(np.clip(zn, 0, 1), RZ, RW)
        ear = (np.abs(np.abs(u) - w_here) < 0.1) & (np.abs(f + 0.085) < 0.11) & (np.abs(zn - ear_zn) < 0.16)
        return (zn < zlim) & ~ear & (zn > -0.05)
    return m


# hairline map (female base): hairline height zn as a function of the azimuth around the head
# (0 = straight forward, 90 = over the ear, 180 = nape).  Front at a third of the face above the brow
# (Loomis: hairline -> brow -> nose base -> chin in equal thirds: 0.48 - 0.26 = 0.22), gently rounded,
# shallow temple recesses, sideburn in front of the ear, up and over the ear, down to the nape.
HL_THETA = [0, 12, 24, 36, 48, 60, 70, 78, 84, 90, 102, 114, 124, 140, 160, 180]
HL_ZN = [0.218, 0.219, 0.226, 0.245, 0.275, 0.315, 0.37, 0.47, 0.50, 0.405, 0.395, 0.42, 0.58, 0.76, 0.84, 0.86]


def hairline_zn(u, f):
    th = np.degrees(np.arctan2(np.abs(u), f + 0.02))
    return np.interp(th, HL_THETA, HL_ZN)


def hairline_density(hf, p, feather=0.028):
    """1 well inside the hairline, feathering to 0 at the hairline (no hard edge)"""
    u, f, zn = hf.local(p)
    return smooth(0.0, feather, hairline_zn(u, f) - zn)


def hair_roots(shape, J, n, feather=0.028, seed=0):
    """area-uniform roots inside the mapped hairline, with a feathered (sparser) hairline edge"""
    hf = HeadFrame(J)
    lo, hi = _clip_head(hf, J)

    def mask(p):
        u, f, zn = hf.local(p)
        return (zn < hairline_zn(u, f)) & (zn > -0.05)
    roots, nrm = hr.surface_samples(shape, J, lo, hi, int(n * 1.25), mask)
    keep = np.random.default_rng(seed).random(len(roots)) < hairline_density(hf, roots, feather)
    return roots[keep][:n], nrm[keep][:n]


# --------------------------------------------------------------------------
def _clip_head(hf, J, below=1.25):
    c = np.array([0, hf.yc, hf.top - 0.5 * hf.hh], F)
    return c - np.array([0.45, 0.45, below * 0.5 + 0.1], F) * hf.hh * 1.2, c + np.array([0.45, 0.45, 0.65], F) * hf.hh * 1.2


def male_swept_back(shape, J, field, n_guides=1200, n_children=62000, M=14):
    hf = HeadFrame(J)
    lo, hi = _clip_head(hf, J)
    roots, nrm = hr.surface_samples(shape, J, lo, hi, n_guides + n_children, scalp_mask(hf, front_zn=0.235, nape_zn=0.8))
    u, f, zn = hf.local(roots)
    k = hf.hh / 0.23
    # length: ~5.5 cm on top/front, tapering to ~1.2 cm at the sides and nape (classic short back & sides)
    top = smooth(0.42, 0.12, zn) * smooth(0.24, 0.12, np.abs(u))
    length = (0.012 + 0.045 * top + 0.008 * smooth(0.6, 0.3, zn)) * k
    lift = (0.0012 + 0.007 * top).astype(F)

    def flow(p, nn, s):
        uu, ff, z = hf.local(p)
        back = np.tile(np.array([0, 1, 0], F), (len(p), 1))
        up = np.tile(np.array([0, 0, 1], F), (len(p), 1))
        down = -up
        w_down = smooth(0.25, 0.6, z)[:, None]
        w_up = (smooth(0.0, 0.3, ff) * smooth(0.35, 0.15, z))[:, None] * (1 - s)
        d = back * (1 - 0.6 * w_down) + down * w_down + up * 0.45 * w_up
        return _n(d.astype(F))
    rest = hr.grow_along_surface(roots, nrm, flow, length, M, field, lift, rise=0.7)
    gi = slice(0, n_guides)
    guides = hr.simulate(rest[gi], field, 0.0012, iters=40, stiff=0.45)
    kids = hr.interpolate_children(roots[gi], guides, roots[n_guides:], nrm[n_guides:])
    kids = hr.clump_and_frizz(kids, roots[n_guides:], 2600, clump=0.3, frizz=0.0006, length_var=0.15)
    kids = _push(kids, field, 0.0008)
    return dict(name="scalp", strands=np.concatenate([guides, kids]), r_root=0.0001, r_tip=0.00003)


def male_beard(shape, J, field, n_guides=600, n_children=16000, M=8):
    hf = HeadFrame(J)
    lo, hi = _clip_head(hf, J, below=1.5)
    mouth = LM["mouth"]

    def mask(p):
        u, f, zn = hf.local(p)
        w = np.interp(np.clip(zn, 0, 1.05), RZ, RW)
        cheek_line = 0.66 + 0.06 * smooth(0.05, 0.25, np.abs(u))            # beard starts below the cheekbones
        ok = (zn > cheek_line) & (zn < 1.1) & (f > -0.12)
        ok &= (np.abs(u) < w + 0.02)
        lips = (np.abs(u) < 0.11) & (np.abs(zn - mouth) < 0.045) & (f > 0.2)
        mustache = (np.abs(u) < 0.12) & (zn > LM["subnasale"] + 0.012) & (zn < mouth - 0.035) & (f > 0.2)
        under_chin = (zn > 1.0) & (f < -0.05)
        return (ok & ~lips & ~under_chin) | mustache
    roots, nrm = hr.surface_samples(shape, J, lo, hi, int((n_guides + n_children) * 1.6), mask)
    u, f, zn = hf.local(roots)
    k = hf.hh / 0.23
    # soft, sparse edges: density falls off near the cheek line and the neckline
    cheek_line = 0.66 + 0.06 * smooth(0.05, 0.25, np.abs(u))
    edge = smooth(cheek_line, cheek_line + 0.06, zn) * smooth(1.1, 1.02, zn) * smooth(-0.12, -0.02, f)
    edge = np.where(zn < mouth - 0.02, 1.0, edge)
    keep = np.random.default_rng(4).random(len(roots)) < edge ** 0.8
    roots, nrm = roots[keep][: n_guides + n_children], nrm[keep][: n_guides + n_children]
    n_children = len(roots) - n_guides
    u, f, zn = hf.local(roots)
    # trimmed: short on the cheeks, fuller on the chin, short moustache
    length = (0.006 + 0.007 * smooth(0.8, 0.98, zn) * smooth(0.2, 0.05, np.abs(u))) * k
    mus = zn < mouth - 0.02
    length = np.where(mus, 0.007 * k, length) * (0.8 + 0.4 * np.random.default_rng(5).random(len(roots)))
    lift = np.full(len(roots), 0.0015, F)

    def flow(p, nn, s):
        uu, ff, z = hf.local(p)
        d = np.array([0, 0, -1], F)[None, :] + np.stack([np.sign(uu) * 0.25 * (z < mouth), np.full_like(uu, -0.25), np.zeros_like(uu)], 1)
        return _n(d.astype(F))
    rest = hr.grow_along_surface(roots, nrm, flow, length, M, field, lift, rise=0.35)
    gi = slice(0, n_guides)
    guides = hr.simulate(rest[gi], field, 0.0012, iters=25, stiff=0.6)
    kids = hr.interpolate_children(roots[gi], guides, roots[n_guides:], nrm[n_guides:])
    kids = hr.clump_and_frizz(kids, roots[n_guides:], 1500, clump=0.25, frizz=0.0008, curl=0.0009, curl_freq=900, length_var=0.35)
    return dict(name="beard", strands=np.concatenate([guides, kids]), r_root=0.0001, r_tip=0.00004)


def brows(shape, J, field, density=1.0, M=6):
    """Eyebrows rooted on the measured brow-hair arch (face.BROW_HAIR_*): full medially, tapering tail;
    medial hairs point up, the body sweeps out along the arch."""
    import face
    hf = HeadFrame(J)
    lo, hi = _clip_head(hf, J)

    def mask(p):
        u, f, zn = hf.local(p)
        au = np.abs(u)
        centre = np.interp(au, face.BROW_HAIR_LAT, face.BROW_HAIR_ZN)
        half = 0.016 * (1 - 0.65 * smooth(0.08, 0.24, au))
        return (au > 0.058) & (au < 0.242) & (np.abs(zn - centre) < half) & (f > 0.15)
    n = int(2800 * density)
    blo_ = hf.world(np.array([-0.3]), np.array([0.15]), np.array([0.53]))[0]
    bhi_ = hf.world(np.array([0.3]), np.array([0.5]), np.array([0.38]))[0]
    roots, nrm = hr.surface_samples(shape, J, np.minimum(blo_, bhi_), np.maximum(blo_, bhi_), n, mask, h=0.0012)
    k = hf.hh / 0.23
    length = (0.0065 + 0.0035 * RNG_uniform(len(roots))) * k
    lift = np.full(len(roots), 0.0006, F)

    def flow(p, nn, s):
        uu, ff, z = hf.local(p)
        au = np.abs(uu)
        medial = smooth(0.10, 0.065, au)
        # along the arch: slope of the measured arch
        dz = np.interp(au + 0.01, face.BROW_HAIR_LAT, face.BROW_HAIR_ZN) - np.interp(au, face.BROW_HAIR_LAT, face.BROW_HAIR_ZN)
        along = np.stack([np.sign(uu) * 0.01, np.zeros_like(uu), -dz], 1)
        along = along / (np.linalg.norm(along, axis=1, keepdims=True) + 1e-9)
        up = np.tile(np.array([0, 0, 1.0], F), (len(p), 1))
        d = along * (1 - medial)[:, None] + up * (0.25 + 0.75 * medial)[:, None]
        return _n(d.astype(F))
    # depth: hairs layer over each other (varied lift / rise); ~12% strays stand off and cross the grain
    rng = np.random.default_rng(21)
    stray = rng.random(len(roots)) < 0.12
    lift = np.where(stray, 0.0016, 0.0003 + 0.0009 * rng.random(len(roots))).astype(F)
    S = hr.grow_along_surface(roots, nrm, flow, length, M, field, lift, rise=0.35)
    S = hr.clump_and_frizz(S, roots, 260, clump=0.22, frizz=0.00025, length_var=0.3, seed=5)
    if stray.any():
        jit = rng.normal(0, 1, (int(stray.sum()), 1, 3)).astype(F) * 0.0012 * k
        ramp = np.linspace(0, 1, M, dtype=F)[None, :, None]
        S[stray] = S[stray] + jit * ramp + nrm[stray][:, None, :] * (0.0012 * k * ramp)
    return dict(name="brows", strands=S, r_root=0.000085, r_tip=0.00003)


def visible_lid_edge(shape, eye, c, xs, upper, zmax=0.016, dz=0.00015, depth=0.03, dy=0.0002):
    """For each x: scan from the eye centre's height up (upper) / down (lower) and find where the view
    from the front stops hitting the eyeball and hits lid skin.  Returns the skin point at that edge
    (the front of the lid margin) for each x, or None."""
    out = []
    zs = np.arange(0.0, zmax, dz) * (1 if upper else -1)
    ys = np.arange(c[1] - depth, c[1] + 0.002, dy)
    for x in xs:
        Z, Y = np.meshgrid(zs, ys, indexing="ij")
        P = np.stack([np.full(Z.size, x), Y.ravel(), (c[2] + Z).ravel()], 1).astype(F)
        ds = shape.eval(P).reshape(Z.shape)
        de = eye.eval(P).reshape(Z.shape)
        hit_s = np.where((ds < 0).any(1), np.argmax(ds < 0, 1), 10 ** 6)
        hit_e = np.where((de < 0).any(1), np.argmax(de < 0, 1), 10 ** 6)
        skin_first = hit_s < hit_e
        seen_eye = ~skin_first & (hit_e < 10 ** 6)
        if not seen_eye.any():
            out.append(None)
            continue
        i0 = int(np.argmax(seen_eye))                       # first row where the eye is seen
        rows = np.where(skin_first[i0:])[0]
        if len(rows) == 0:
            out.append(None)
            continue
        i = i0 + int(rows[0])                               # first row (moving away from centre) that is lid
        out.append(np.array([x, ys[hit_s[i]], c[2] + zs[i]], F))
    return out


def lashes(J, p, eye_centers=None, length=0.0095, n=150, M=8, shape=None):
    """Upper and lower lashes growing out of the lid margin exactly at the visible edge of the eye
    opening (found by ray casting the sculpt + eyeball from the front): two staggered rows, leaving the
    lid forward (upper: slightly down) then curling away from the eye, longest at the outer third,
    clumped in small groups; any lash point that would clip the lid is pushed back out of it."""
    import face, mesher
    hf = HeadFrame(J)
    k = hf.hh / 0.23
    eye = anatomy.eye_shape(p, J)
    out, roots_all = [], []
    rng = np.random.default_rng(11)
    for sx in face.LID_EDGES:
        up, lo = face.LID_EDGES[sx]
        c = face.LID_ARCS[sx][2]
        for arcpts, nn, L0, lower in ((up, n, length, False), (lo, n // 3, length * 0.42, True)):
            x0, x1 = float(arcpts[0][0]), float(arcpts[-1][0])
            tt = np.sort(rng.uniform(0.04, 0.96, nn))
            xs = x0 + (x1 - x0) * tt
            edge = visible_lid_edge(shape, eye, c, xs, upper=not lower)
            vert = -1.0 if lower else 1.0
            for t, x, e in zip(tt, xs, edge):
                if e is None:
                    continue
                row = int(rng.random() < 0.4)
                base = e + np.array([0, 0.0002, vert * (0.0002 + row * 0.00035)], F)   # just inside the lid rim
                lat_side = t if (x1 * sx > x0 * sx) else 1 - t
                prof = np.sin(np.clip(t, 0.02, 0.98) * np.pi) ** 0.5
                L = L0 * k * (0.4 + 0.6 * prof) * (0.75 + 0.45 * lat_side) * rng.uniform(0.7, 1.08) * (1 - 0.2 * row)
                d = _n(np.array([sx * 0.4 * (lat_side - 0.4), -1.0, -0.1 * vert], F) + rng.normal(0, 0.07, 3).astype(F))
                curl = np.array([0, 0.05, (0.36 + 0.14 * rng.random()) * vert], F)
                pts = [base]
                for j in range(1, M):
                    d = _n(d + curl * (0.6 + 0.8 * j / M))
                    pts.append(pts[-1] + d * L / (M - 1))
                out.append(pts)
                roots_all.append(base)
    S = np.array(out, F)
    R = np.array(roots_all, F)
    S = hr.clump_and_frizz(S, R, max(len(S) // 3, 1), clump=0.4, frizz=0.00004, length_var=0.05, seed=12)
    # nothing passes through the lid: push clipping points (not the root) back out along the gradient
    for _ in range(3):
        P = S[:, 1:].reshape(-1, 3)
        dd = shape.eval(P)
        bad = dd < 0.00025
        if not bad.any():
            break
        gr = mesher.gradient(shape, P[bad])
        gr /= np.linalg.norm(gr, axis=1, keepdims=True) + 1e-9
        P[bad] += gr * (0.00025 - dd[bad])[:, None]
        S[:, 1:] = P.reshape(S[:, 1:].shape)
    return dict(name="lashes", strands=S, r_root=0.000095, r_tip=0.000015)


def female_braid(shape, J, field, n_guides=1600, n_children=95000, M=40, braid_len=0.42):
    """Hair pulled back from the face and gathered into a single three-strand braid down the back,
    with a few loose face-framing tendrils."""
    hf = HeadFrame(J)
    lo, hi = _clip_head(hf, J)
    roots, nrm = hair_roots(shape, J, n_guides + n_children)
    u, f, zn = hf.local(roots)
    k = hf.hh / 0.23
    G = hf.world(np.array([0.0]), np.array([-0.44]), np.array([0.62]))[0]          # gather point at the back of the head
    # scalp part: follow the head surface toward G
    d_scalp = np.linalg.norm(roots - G, axis=1) * 1.25 + 0.01
    # volume: hair stands off the scalp a little, more on the crown, least at the feathered hairline
    lift = (0.0012 + 0.0045 * smooth(0.45, 0.08, zn) * smooth(0.0, 0.04, hairline_zn(u, f) - zn)).astype(F)

    def flow(p, nn, s):
        return _n((G[None, :] - p).astype(F))
    Ms = 14
    scalp = hr.grow_along_surface(roots, nrm, flow, d_scalp, Ms, field, lift, rise=0.2)
    # braid: three interleaved groups following a path down the back
    group = (np.floor(((np.arctan2(u, -zn + 0.3) / np.pi) + 1) * 1.5).astype(int)) % 3
    path_n = M - Ms
    s = np.linspace(0, 1, path_n)
    # braid centreline: from G down and slightly out from the back surface
    down = np.array([0, 0, -1], F)
    path = G[None, :] + down[None, :] * (s[:, None] * braid_len) + np.array([0, 1, 0], F)[None, :] * (0.012 + 0.03 * s[:, None])
    path = field.push_out(path.astype(F), 0.025)
    width = 0.016 * (1 - 0.45 * s) * k
    omega = 2 * np.pi * braid_len / (0.045 * k)
    N = len(roots)
    S = np.zeros((N, M, 3), F)
    S[:, :Ms] = scalp
    rng = np.random.default_rng(3)
    jitter = rng.normal(0, 1, (N, 2)).astype(F) * 0.0045 * k
    lat = np.array([1, 0, 0], F)
    dep = np.array([0, 1, 0], F)
    for j in range(path_n):
        ph = omega * s[j] + group * 2 * np.pi / 3
        off = lat[None] * (np.sin(ph) * width[j])[:, None] + dep[None] * (np.sin(2 * ph) * width[j] * 0.45)[:, None]
        jj = jitter * (1 - 0.5 * s[j])
        S[:, Ms + j] = path[j][None] + off + lat[None] * jj[:, :1] + dep[None] * jj[:, 1:]
    # blend the junction between scalp part and braid (pony gathering)
    for b in range(4):
        a = (b + 1) / 5
        S[:, Ms - 2 + b] = S[:, Ms - 2 + b] * (1 - a * 0.5) + S[:, Ms + 2] * (a * 0.5)
    # loose tuft after the tie: fan out
    tail = 6
    for j in range(M - tail, M):
        fan = (j - (M - tail)) / tail
        S[:, j] += (rng.normal(0, 1, (N, 3)).astype(F) * np.array([1, 0.6, 0.2], F)) * 0.012 * fan * k
    gi = slice(0, n_guides)
    S = _push(S, field, 0.001)
    kids = S[n_guides:]
    kids = hr.clump_and_frizz(kids, roots[n_guides:], 1800, clump=0.25, frizz=0.0006, length_var=0.04)
    main = dict(name="scalp", strands=np.concatenate([S[gi], kids]), r_root=0.00009, r_tip=0.00005)
    # face-framing tendrils (simulated: they actually hang under gravity)
    tm = (np.abs(u) > 0.24) & (np.abs(u) < 0.34) & (f > 0.05) & (hairline_zn(u, f) - zn < 0.03)
    tr, tn = roots[tm][:220], nrm[tm][:220]
    tl = np.full(len(tr), 0.16 * k, F)

    def tflow(p, nn, s):
        return _n(np.tile(np.array([0, 0.15, -1], F), (len(p), 1)))
    trest = hr.grow_along_surface(tr, tn, tflow, tl, 16, field, np.full(len(tr), 0.004, F), rise=0.8)
    tsim = hr.simulate(trest, field, 0.002, iters=80, stiff=0.08)
    tsim = hr.clump_and_frizz(tsim, tr, 30, clump=0.5, frizz=0.0008, curl=0.004, curl_freq=60, length_var=0.3, seed=9)
    tend = dict(name="tendrils", strands=tsim, r_root=0.00008, r_tip=0.00003)
    return [main, tend]


def _push(S, field, margin):
    N, M, _ = S.shape
    flat = S[:, 1:].reshape(-1, 3).copy()
    flat = field.push_out(flat, margin)
    S = S.copy()
    S[:, 1:] = flat.reshape(N, M - 1, 3)
    return S


def RNG_uniform(n):
    return np.random.default_rng(11).random(n).astype(F)
