"""Strand hair: groomed guide strands, verlet physics, child interpolation, clumping & frizz.

Pipeline (same idea as film/game grooming tools):
  1. roots are sampled on the real scalp/face surface (meshed from the body SDF) inside
     style-specific masks (hairline, beard, brows, lashes)
  2. guide strands are grown along a groom flow field (scalp-following, gravity, targets)
  3. guides are relaxed with a verlet solver: gravity, inextensible segments, bend/shape
     stiffness, collision with a precomputed signed-distance grid of head+body
  4. thousands of child strands are interpolated from the 3 nearest guides, then clumped,
     frizzed and length-varied, and pushed out of the body once more
Output: list of (N, M, 3) strand arrays + radii + material settings per groom.
"""
from __future__ import annotations

import math
import time

import numpy as np

import mesher
from sdf import F, Shape

RNG = np.random.default_rng(7)


def _n(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-9)


# --------------------------------------------------------------------------
# collision field
# --------------------------------------------------------------------------
class Field:
    """Signed distance of the (undisplaced) body sampled on a grid, trilinear lookups."""

    def __init__(self, shape: Shape, lo, hi, h=0.004):
        self.lo = np.asarray(lo, F)
        self.h = h
        n = np.ceil((np.asarray(hi, F) - self.lo) / h).astype(int) + 1
        self.n = n
        xs = [self.lo[i] + np.arange(n[i]) * h for i in range(3)]
        g = np.empty(tuple(n), F)
        t = time.time()
        for k in range(n[2]):
            X, Y = np.meshgrid(xs[0], xs[1], indexing="ij")
            P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, xs[2][k], F)], 1).astype(F)
            g[:, :, k] = shape.eval(P).reshape(n[0], n[1])
        self.g = g
        self.grad = np.stack(np.gradient(g, h), -1).astype(F)
        print(f"  collision field {tuple(n)} in {time.time() - t:.1f}s")

    def sample(self, p):
        q = (p - self.lo) / self.h
        i = np.clip(np.floor(q).astype(int), 0, self.n - 2)
        f = np.clip(q - i, 0, 1)
        d = np.zeros(len(p), F)
        gr = np.zeros((len(p), 3), F)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    ii, jj, kk = i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz
                    d += w * self.g[ii, jj, kk]
                    gr += w[:, None] * self.grad[ii, jj, kk]
        outside = np.any((q < 0) | (q > self.n - 1), axis=1)
        d[outside] = 1.0
        return d, gr

    def push_out(self, p, margin):
        d, g = self.sample(p)
        pen = margin - d
        m = pen > 0
        if np.any(m):
            p[m] += _n(g[m]) * pen[m, None]
        return p


# --------------------------------------------------------------------------
# head coordinate frame (matches anatomy._head)
# --------------------------------------------------------------------------
class HeadFrame:
    def __init__(self, J):
        self.hh = float(J["_head_h"])
        self.top = float(J["head_top"][2])
        self.yc = float(J["head"][1])

    def local(self, p):
        """-> (u lateral, f forward, zn down-from-top) in head heights."""
        return (p[:, 0] / self.hh, (self.yc - p[:, 1]) / self.hh, (self.top - p[:, 2]) / self.hh)

    def world(self, u, f, zn):
        return np.stack([u * self.hh, self.yc - f * self.hh, self.top - zn * self.hh], -1).astype(F)


def surface_samples(shape, J, clip_lo, clip_hi, n, mask_fn, h=0.0028):
    """Area-weighted random points (+normals) on the body surface inside mask_fn."""
    v, t = mesher.mesh_shape(shape, h=h, clip=(np.asarray(clip_lo, F), np.asarray(clip_hi, F)), verbose=False)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    cen = (a + b + c) / 3
    keep = mask_fn(cen)
    a, b, c = a[keep], b[keep], c[keep]
    nrm = np.cross(b - a, c - a)
    area = np.linalg.norm(nrm, axis=1)
    nrm = nrm / (area[:, None] + 1e-12)
    idx = RNG.choice(len(area), size=n, p=area / area.sum())
    r1, r2 = RNG.random(n), RNG.random(n)
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
    return pts.astype(F), nrm[idx].astype(F)


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
def simulate(rest, field, margin, iters=60, gravity=9.8, stiff=0.25, damping=0.9, dt=1 / 60, root_lock=2):
    """Verlet strands. rest: (N, M, 3) groomed shape (also the shape-memory target)."""
    P = rest.copy()
    prev = P.copy()
    seg = np.linalg.norm(np.diff(rest, axis=1), axis=2)
    N, M, _ = P.shape
    g = np.array([0, 0, -gravity], F) * dt * dt
    for it in range(iters):
        vel = (P - prev) * damping
        prev = P.copy()
        P[:, root_lock:] += vel[:, root_lock:] + g
        # shape memory (styling products / hair stiffness): pull toward the groom, stronger near the root
        w = stiff * (1 - np.linspace(0, 1, M)) ** 1.5
        P += (rest - P) * w[None, :, None]
        for _ in range(4):
            # inextensible segments (root-to-tip Gauss-Seidel)
            for j in range(1, M):
                d = P[:, j] - P[:, j - 1]
                L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
                corr = d * (1 - seg[:, j - 1:j] / L)
                if j <= root_lock - 1:
                    P[:, j] -= corr
                else:
                    P[:, j] -= corr * 0.5
                    if j - 1 >= root_lock:
                        P[:, j - 1] += corr * 0.5
            flat = P[:, root_lock:].reshape(-1, 3)
            flat = field.push_out(flat, margin)
            P[:, root_lock:] = flat.reshape(N, M - root_lock, 3)
        P[:, :root_lock] = rest[:, :root_lock]
    return P


def interpolate_children(guide_roots, guides, child_roots, child_normals, k=3):
    """Children follow the k nearest guides (inverse-distance weights), offset to their own roots."""
    d = np.linalg.norm(child_roots[:, None, :] - guide_roots[None, :, :], axis=2)
    nn = np.argsort(d, axis=1)[:, :k]
    dd = np.take_along_axis(d, nn, axis=1)
    w = 1 / (dd + 1e-4)
    w /= w.sum(1, keepdims=True)
    rel = guides - guides[:, :1]                       # guide shape relative to its root
    shape = np.einsum("nk,nkmc->nmc", w, rel[nn])
    return child_roots[:, None, :] + shape


def clump_and_frizz(S, roots, n_clumps, clump=0.55, frizz=0.0015, curl=0.0, curl_freq=60.0, length_var=0.15, seed=1):
    rng = np.random.default_rng(seed)
    N, M, _ = S.shape
    centers = roots[rng.choice(N, size=min(n_clumps, N), replace=False)]
    lab = np.argmin(np.linalg.norm(roots[:, None] - centers[None], axis=2), axis=1)
    mean = np.zeros((len(centers), M, 3), F)
    np.add.at(mean, lab, S)
    cnt = np.bincount(lab, minlength=len(centers)).astype(F)
    mean /= np.maximum(cnt, 1)[:, None, None]
    s = np.linspace(0, 1, M)[None, :, None]
    cs = clump * (0.6 + 0.4 * rng.random((N, 1, 1)))
    S = S + (mean[lab] - S) * (cs * s ** 1.3)
    # frizz + optional curl (helix around the strand direction)
    tang = _n(np.gradient(S, axis=1))
    a1 = _n(np.cross(tang, np.array([0.3, 0.5, 0.8], F)))
    a2 = np.cross(tang, a1)
    phase = rng.random((N, 1)) * 6.28
    arc = np.cumsum(np.concatenate([np.zeros((N, 1)), np.linalg.norm(np.diff(S, axis=1), axis=2)], 1), axis=1)
    if curl > 0:
        ang = arc * curl_freq + phase
        S = S + (a1 * np.cos(ang)[..., None] + a2 * np.sin(ang)[..., None]) * (curl * s)
    S = S + rng.normal(0, 1, S.shape).astype(F) * frizz * s ** 1.5
    # length variation: trim tips by re-sampling a shorter portion of each strand
    keep = 1 - length_var * rng.random(N)
    t_new = np.linspace(0, 1, M)[None, :] * keep[:, None] * (M - 1)
    i0 = np.clip(np.floor(t_new).astype(int), 0, M - 2)
    fr = (t_new - i0)[..., None]
    S = np.take_along_axis(S, i0[..., None].repeat(3, 2), 1) * (1 - fr) + np.take_along_axis(S, (i0 + 1)[..., None].repeat(3, 2), 1) * fr
    return S.astype(F)


# --------------------------------------------------------------------------
# groom growth helpers
# --------------------------------------------------------------------------
def grow_along_surface(roots, normals, dir_fn, length, M, field, lift, rise=0.0):
    """Grow strands hugging the surface: each step goes along the flow, then is projected
    back to (surface + lift).  `rise` lifts the first segments off the scalp (volume)."""
    N = len(roots)
    S = np.zeros((N, M, 3), F)
    S[:, 0] = roots
    p = roots.copy()
    step = (length / (M - 1))[:, None]
    nrm = normals.copy()
    for j in range(1, M):
        s = j / (M - 1)
        d = dir_fn(p, nrm, s)
        d = d - nrm * (np.sum(d * nrm, 1, keepdims=True) * (1 - rise * (1 - s)))
        p = p + _n(d) * step
        dist, gr = field.sample(p)
        nrm = _n(gr)
        target = lift[:, None] * (0.35 + 0.65 * min(1.0, s * 4))
        p = p + nrm * (target - dist[:, None])
        S[:, j] = p
    return S


def eval_strand_points(S, n):
    """Resample each strand to n points (uniform in index)."""
    M = S.shape[1]
    t = np.linspace(0, M - 1, n)
    i0 = np.clip(np.floor(t).astype(int), 0, M - 2)
    fr = (t - i0)[None, :, None]
    return S[:, i0] * (1 - fr) + S[:, i0 + 1] * fr
