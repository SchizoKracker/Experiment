"""Sparse, block-based Surface Nets mesher for `sdf.Shape`.

1. Evaluate the field at coarse block centres; keep blocks near the surface.
2. Densely evaluate only those blocks at the fine resolution.
3. Surface-nets: one vertex per sign-changing cell (mean of edge crossings),
   one quad per sign-changing lattice edge.
4. Newton-project vertices onto the true zero set for exact silhouettes.

Produces a watertight-in-practice quad mesh (as triangles) and per-vertex tags.
"""
from __future__ import annotations

import time

import numpy as np

from sdf import F, Shape

# the 12 cube edges as (corner_a, corner_b); corners indexed by (dx, dy, dz)
_CORNERS = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=np.int64)
_EDGES = []
for a in range(8):
    for b in range(a + 1, 8):
        if np.abs(_CORNERS[a] - _CORNERS[b]).sum() == 1:
            _EDGES.append((a, b))
_EDGES = np.array(_EDGES)


def mesh_shape(shape: Shape, h: float = 0.004, block: int = 8, pad: float = 0.03,
               project_steps: int = 3, verbose: bool = True, clip=None):
    t0 = time.time()
    lo, hi = shape.bounds()
    if clip is not None:
        lo = np.maximum(lo, clip[0])
        hi = np.minimum(hi, clip[1])
        if np.any(hi <= lo):
            return np.zeros((0, 3), F), np.zeros((0, 3), np.int64)
    lo = lo - pad
    hi = hi + pad
    npts = np.ceil((hi - lo) / h).astype(np.int64) + 1
    nblk = np.ceil((npts - 1) / block).astype(np.int64)
    NX, NY, NZ = (nblk * block + 1)
    H = h * block

    # ---- 1. coarse activity test -------------------------------------
    bi = np.stack(np.meshgrid(np.arange(nblk[0]), np.arange(nblk[1]), np.arange(nblk[2]), indexing="ij"), -1).reshape(-1, 3)
    centres = lo + (bi + 0.5) * H
    d_c = np.empty(len(bi), F)
    for s in range(0, len(bi), 200000):
        d_c[s:s + 200000] = shape.eval(centres[s:s + 200000])
    active = bi[np.abs(d_c) < H * 3.4]
    if len(active) == 0:
        return np.zeros((0, 3), F), np.zeros((0, 3), np.int64)

    # ---- 2. dense block evaluation ------------------------------------
    B1 = block + 1
    loc = np.stack(np.meshgrid(np.arange(B1), np.arange(B1), np.arange(B1), indexing="ij"), -1).reshape(-1, 3)
    batch = max(1, 400000 // len(loc))
    def _eval_blocks(blks):
        out = np.empty((len(blks), B1, B1, B1), F)
        for s in range(0, len(blks), batch):
            blk = blks[s:s + batch]
            gidx = (blk[:, None, :] * block + loc[None, :, :]).reshape(-1, 3)
            pts = (lo + gidx * h).astype(F)
            blo = lo + blk.min(axis=0) * H
            bhi = lo + (blk.max(axis=0) + 1) * H
            out[s:s + batch] = shape.eval(pts, blo, bhi).reshape(len(blk), B1, B1, B1)
        return out
    vals = _eval_blocks(active)

    # ---- 2b. hole repair: a block whose face is crossed by the surface needs its neighbour across that
    # face too.  The coarse test can miss such a neighbour where displacements make the field steep;
    # add missing neighbours until none are left (so the mesh is closed wherever the surface goes).
    # all 26 neighbours: an edge lying along a block edge/corner takes its quad's cells from diagonal blocks
    faces = []
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for oz in (-1, 0, 1):
                if ox == oy == oz == 0:
                    continue
                sl = tuple(slice(None) if o == 0 else (0 if o < 0 else -1) for o in (ox, oy, oz))
                faces.append((sl, (ox, oy, oz)))
    for _ in range(8):
        have = set(map(tuple, active))
        need = []
        for sl, off in faces:
            fv = vals[(slice(None),) + sl].reshape(len(active), -1)
            crossed = (fv < 0).any(1) & (fv >= 0).any(1)
            nb = active[crossed] + np.array(off)
            ok = np.all((nb >= 0) & (nb < nblk), axis=1)
            for t in map(tuple, nb[ok]):
                if t not in have:
                    have.add(t)
                    need.append(t)
        if not need:
            break
        need = np.array(need, np.int64)
        active = np.concatenate([active, need])
        vals = np.concatenate([vals, _eval_blocks(need)])

    # ---- 3a. cells with a sign change -> vertices ------------------------
    inside = vals < 0
    corner_in = np.stack([inside[:, cx:cx + block, cy:cy + block, cz:cz + block] for cx, cy, cz in _CORNERS], -1)
    n_in = corner_in.sum(-1)
    mixed = (n_in > 0) & (n_in < 8)
    bidx, ci, cj, ck = np.nonzero(mixed)
    if len(bidx) == 0:
        return np.zeros((0, 3), F), np.zeros((0, 3), np.int64)
    corner_v = np.stack([vals[bidx, ci + cx, cj + cy, ck + cz] for cx, cy, cz in _CORNERS], -1)  # (M, 8)
    acc = np.zeros((len(bidx), 3), np.float64)
    cnt = np.zeros(len(bidx), np.float64)
    for a, b in _EDGES:
        va, vb = corner_v[:, a], corner_v[:, b]
        cross = (va < 0) != (vb < 0)
        t = np.where(cross, va / np.where(cross, va - vb, 1.0), 0.0)
        pa = _CORNERS[a].astype(np.float64)
        pb = _CORNERS[b].astype(np.float64)
        pos = pa[None, :] + t[:, None] * (pb - pa)[None, :]
        acc += np.where(cross[:, None], pos, 0.0)
        cnt += cross
    local = acc / np.maximum(cnt, 1)[:, None]
    cell_g = active[bidx] * block + np.stack([ci, cj, ck], -1)
    verts = lo + (cell_g + local) * h
    keys = cell_g[:, 0] + NX * (cell_g[:, 1] + NY * cell_g[:, 2])
    keys, uniq = np.unique(keys, return_index=True)
    verts = verts[uniq]

    def lookup(g):
        k = g[:, 0] + NX * (g[:, 1] + NY * g[:, 2])
        pos = np.searchsorted(keys, k)
        pos = np.clip(pos, 0, len(keys) - 1)
        ok = keys[pos] == k
        return pos, ok

    # ---- 3b. sign-changing edges -> quads --------------------------------
    # pass 1: every cell a quad needs must have a vertex.  Adjacent blocks evaluate shared lattice points
    # with different culling bounds, so near-zero values can differ in sign between them; a cell a
    # crossing needs may then have no vertex.  Create those at the cell centre (projected below) instead
    # of dropping the quad (which left holes).
    flips = []
    need = []
    for axis in range(3):
        sl0 = [slice(0, block)] * 3
        sl1 = [slice(0, block)] * 3
        sl1[axis] = slice(1, block + 1)
        v0 = vals[(slice(None),) + tuple(sl0)]
        v1 = vals[(slice(None),) + tuple(sl1)]
        flip = (v0 < 0) != (v1 < 0)
        b2, i2, j2, k2 = np.nonzero(flip)
        g = active[b2] * block + np.stack([i2, j2, k2], -1)
        flips.append((g, v0[b2, i2, j2, k2] < 0))
        o1, o2 = [(1, 2), (2, 0), (0, 1)][axis]
        for da, db in [(-1, -1), (0, -1), (0, 0), (-1, 0)]:
            c = g.copy()
            c[:, o1] += da
            c[:, o2] += db
            need.append(c)
    need = np.concatenate(need) if need else np.zeros((0, 3), np.int64)
    nk = need[:, 0] + NX * (need[:, 1] + NY * need[:, 2])
    miss = ~np.isin(nk, keys)
    if np.any(miss):
        mk, mi = np.unique(nk[miss], return_index=True)
        mc = need[miss][mi]
        mv = lo + (mc + 0.5) * h
        allk = np.concatenate([keys, mk])
        allv = np.concatenate([verts, mv])
        order = np.argsort(allk)
        keys, verts = allk[order], allv[order]

    quads = []
    for axis in range(3):
        sl0 = [slice(0, block)] * 3
        sl1 = [slice(0, block)] * 3
        sl1[axis] = slice(1, block + 1)
        # interior lattice points excluding the far face along non-edge axes keeps ownership unique
        v0 = vals[(slice(None),) + tuple(sl0)]
        v1 = vals[(slice(None),) + tuple(sl1)]
        flip = (v0 < 0) != (v1 < 0)
        b2, i2, j2, k2 = np.nonzero(flip)
        g = active[b2] * block + np.stack([i2, j2, k2], -1)
        start_inside = v0[b2, i2, j2, k2] < 0
        o1, o2 = [(1, 2), (2, 0), (0, 1)][axis]
        offs = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
        cells = []
        good = np.ones(len(g), bool)
        for da, db in offs:
            c = g.copy()
            c[:, o1] += da
            c[:, o2] += db
            pidx, ok = lookup(c)
            cells.append(pidx)
            good &= ok
        q = np.stack(cells, -1)[good]
        si = start_inside[good]
        q[~si] = q[~si][:, ::-1]
        quads.append(q)
    quads = np.concatenate(quads)
    quads = np.unique(quads, axis=0)
    # drop degenerate quads (repeated vertex)
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]

    tri = _fill_small_holes(tri, verts, lo + h, lo + (nblk * block - 1) * h)

    # ---- 4. projection onto the zero set ---------------------------------
    verts = project(shape, verts.astype(F), project_steps, h)
    if verbose:
        print(f"  mesh: {len(active)} blocks, {len(verts)} verts, {len(tri)} tris, {time.time() - t0:.1f}s")
    return verts, tri


def _fill_small_holes(tri, verts, inner_lo, inner_hi, max_len=12):
    """Close small boundary loops (not at the meshing box edge): fan-triangulate each, oriented to match
    the surrounding faces.  Catches the rare cell where adjacent blocks disagreed about the surface."""
    if len(tri) == 0:
        return tri
    he = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.minimum(he[:, 0], he[:, 1]) * (len(verts) + 1) + np.maximum(he[:, 0], he[:, 1])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bnd = he[cnt[inv] == 1]
    if len(bnd) == 0:
        return tri
    nxt = {}
    for a, b in bnd:
        nxt.setdefault(int(b), []).append(int(a))       # walk boundary half-edges reversed -> hole orientation
    used = set()
    new = []
    for start in list(nxt):
        if start in used or len(nxt[start]) != 1:
            continue
        loop, cur = [start], nxt[start][0]
        while cur != start and len(loop) <= max_len and cur in nxt and len(nxt[cur]) == 1:
            loop.append(cur)
            cur = nxt[cur][0]
        if cur != start or len(loop) < 3 or len(loop) > max_len:
            continue
        P = verts[loop]
        if np.any(P < inner_lo) or np.any(P > inner_hi):
            continue
        used.update(loop)
        for i in range(1, len(loop) - 1):
            new.append((loop[0], loop[i], loop[i + 1]))
    if new:
        tri = np.concatenate([tri, np.array(new, tri.dtype)])
    return tri


def gradient(shape: Shape, p: np.ndarray, eps: float = 5e-4):
    out = np.empty_like(p)
    for a in range(3):
        o = np.zeros(3, F)
        o[a] = eps
        out[:, a] = (shape.eval(p + o) - shape.eval(p - o)) / (2 * eps)
    return out


def project(shape: Shape, p: np.ndarray, steps: int, h: float):
    for _ in range(steps):
        d = shape.eval(p)
        g = gradient(shape, p)
        gl2 = np.maximum((g * g).sum(1), 1e-8)
        step = (d / gl2)[:, None] * g
        # never move more than a cell per step (keeps thin features stable)
        n = np.linalg.norm(step, axis=1, keepdims=True)
        step = np.where(n > h, step * (h / np.maximum(n, 1e-9)), step)
        p = p - step
    return p


def vertex_tags(shape: Shape, verts: np.ndarray):
    _, idx = shape.eval(verts, with_tags=True)
    return shape.tag_names(idx)


def remove_small_islands(verts, tris, min_faces=200):
    """Drop disconnected fragments (noise specks) – union-find on faces."""
    parent = np.arange(len(verts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # vectorised-ish union via repeated min-label propagation
    lab = np.arange(len(verts))
    for _ in range(200):
        m = np.minimum(np.minimum(lab[tris[:, 0]], lab[tris[:, 1]]), lab[tris[:, 2]])
        new = lab.copy()
        np.minimum.at(new, tris[:, 0], m)
        np.minimum.at(new, tris[:, 1], m)
        np.minimum.at(new, tris[:, 2], m)
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    flab = lab[tris[:, 0]]
    u, c = np.unique(flab, return_counts=True)
    keep = np.isin(flab, u[c >= min_faces])
    return verts, tris[keep]
