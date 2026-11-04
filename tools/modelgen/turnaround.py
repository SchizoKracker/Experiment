"""Full-body turnaround (front, 3/4 left, 3/4 right, side profile, back) in Cycles.

usage: python turnaround.py [--presets human_f,human_m] [--samples 128] [--h 0.0025] [--workers N]

Meshing is split into horizontal slabs that are meshed in parallel on every CPU core (one process per
slab); the slabs share one lattice, so their seams line up and are welded.  Meshes are cached per preset
in MODELGEN_OUT (rebuilt automatically when the model code changes).  Rendering uses the GPU when present.
"""
import glob
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT_DIR = os.environ.get("MODELGEN_OUT", os.path.join(HERE, "out"))
VIEWS = (("front", 0), ("34L", -40), ("34R", 40), ("side", -90), ("back", 180))
EXCLUDE = ("headshot.py", "browlash_preview.py", "face_preview.py", "turnaround.py")


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def src_hash():
    hs = hashlib.sha1()
    for fn in sorted(glob.glob(os.path.join(HERE, "*.py")) + glob.glob(os.path.join(HERE, "*.json"))):
        if os.path.basename(fn) not in EXCLUDE:
            hs.update(open(fn, "rb").read())
    return hs.hexdigest()


def _build(preset):
    import races, humanoid as hu, body
    hu.ARM_DROP_DEG = 72
    p = races.preset(preset)
    J = hu.build_skeleton(p)
    return p, J, body.build_body(p, J)


def _slab(job):
    """Worker: mesh one z-slab and keep only the triangles whose centroid lies in [z0, z1)."""
    preset, h, z0, z1, clip_lo, clip_hi = job
    import mesher
    from sdf import F
    _, _, s = _build(preset)
    v, t = mesher.mesh_shape(s, h=h, clip=(np.array(clip_lo, F), np.array(clip_hi, F)), verbose=False)
    if len(t) == 0:
        return v[:0], t, v[:0]
    cz = v[t].mean(axis=1)[:, 2]
    t = t[(cz >= z0) & (cz < z1)]
    used = np.unique(t)
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(len(used))
    v = v[used]
    return v, remap[t], mesher.gradient(s, v.astype(F))


def mesh_parallel(preset, h, workers):
    import mesher, anatomy
    from concurrent.futures import ProcessPoolExecutor
    p, J, s = _build(preset)
    if workers <= 1 or "--no-split" in sys.argv:                     # plain single-process mesh (fallback)
        from sdf import F
        v, t = mesher.mesh_shape(s, h=h, verbose=False)
        ev, ef = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0008, verbose=False)
        return v, t, mesher.gradient(s, v.astype(F)), ev, ef
    lo, hi = s.bounds()
    pad, block = 0.03, 8
    step = h * block
    glo = lo[2] - pad                                  # global lattice origin in z (what an unsplit mesh would use)
    nsl = max(1, workers * 2)                          # more slabs than workers: legs are cheap, the head is not
    span = hi[2] - lo[2]
    cuts = [lo[2] + np.ceil(span * i / nsl / step) * step for i in range(nsl)] + [hi[2] + 1.0]
    jobs = []
    for i in range(nsl):
        z0, z1 = cuts[i], cuts[i + 1]
        # clip so the slab's lattice origin (clip_lo - pad) falls on the global lattice -> identical seam vertices
        clo = [-10.0, -10.0, glo + pad + np.floor((z0 - 2 * step - glo) / step) * step if i else lo[2]]
        chi = [10.0, 10.0, z1 + 2 * step]
        jobs.append((preset, h, z0 if i else -10.0, z1, clo, chi))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_slab, jobs))
    vs, ts, ns, off = [], [], [], 0
    for v, t, n in res:
        vs.append(v); ts.append(t + off); ns.append(n); off += len(v)
    v, t, n = np.concatenate(vs), np.concatenate(ts), np.concatenate(ns)
    # weld seam vertices (same lattice cell -> same position)
    key = np.round(v / 1e-6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    v, n, t = v[first], n[first], inv[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    ev, ef = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0008, verbose=False)
    print(f"[{preset}] meshed {len(t)} tris on {workers} workers in {time.time() - t0:.0f}s", flush=True)
    return v, t, n, ev, ef


def get_mesh(preset, h, workers):
    cache = os.path.join(OUT_DIR, f"body_cache_{preset}.npz")
    sh = src_hash() + f"|{h}"
    if os.path.exists(cache) and "--rebuild" not in sys.argv:
        try:
            C = np.load(cache)
            if str(C["src_hash"]) == sh:
                print(f"[{preset}] using cached mesh", flush=True)
                return C["v"], C["f"], C["nr"], C["ev"], C["ef"]
        except Exception:
            pass
    v, f, nr, ev, ef = mesh_parallel(preset, h, workers)
    np.savez(cache, v=v, f=f, nr=nr, ev=ev, ef=ef, src_hash=sh)
    return v, f, nr, ev, ef


def render(preset, mesh, samples, res):
    import bpy
    import blendutil as bu
    import character
    import races, humanoid as hu
    from PIL import Image
    v, f, nr, ev, ef = mesh
    hu.ARM_DROP_DEG = 72
    p = races.preset(preset)
    J = hu.build_skeleton(p)
    bu.reset()
    ob = bu.mesh_object("body", v, f)
    ob.data.materials.append(bu.material("skin", tuple(min(1, c * 1.06) for c in races.skin_color(p)), rough=0.5, sss=0.12))
    bu.set_true_normals(ob, nr)
    eo = bu.mesh_object("eyes", ev, ef)
    bu.set_vertex_colors(eo, character.eye_colors(p, J, ev, iris=(0.22, 0.35, 0.45)))
    eo.data.materials.append(bu.vc_material("eye", rough=0.04))
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    bpy.context.object.data.materials.append(bu.material("floor", (0.2, 0.2, 0.21), rough=0.8))
    H = p["height"]
    W, Hh = res
    bu.setup_render((W, Hh), samples)
    sc = bpy.context.scene
    sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.3, 0.31, 0.33, 1)
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.4
    c = np.array([0, 0, 0.5 * H])
    bu.add_light("AREA", tuple(c + np.array([-1.8, -2.6, 1.6])), 550, (1.0, 0.95, 0.88), 1.8, tuple(c))   # key
    bu.add_light("AREA", tuple(c + np.array([2.4, -1.2, 0.4])), 160, (0.85, 0.9, 1.0), 2.0, tuple(c))   # fill
    bu.add_light("AREA", tuple(c + np.array([0.6, 2.6, 1.4])), 350, (1, 1, 1), 1.5, tuple(c))           # rim
    cam = bu.camera(tuple(c), tuple(c), lens=85)
    ims = []
    for name, ang in VIEWS:
        a = np.radians(ang)
        d = 6.3 * H / 1.65
        cam.location = tuple(c + np.array([np.sin(a) * d, -np.cos(a) * d, 0.0]))
        bu.look_at(cam, tuple(c))
        pth = os.path.join(OUT_DIR, f"body_{preset}_{name}.png")
        bu.render(pth)
        ims.append(Image.open(pth).convert("RGB"))
        print(f"[{preset}] rendered {name}", flush=True)
    sheet = Image.new("RGB", (W * len(ims), Hh))
    for i, im in enumerate(ims):
        sheet.paste(im, (W * i, 0))
    sheet.save(os.path.join(OUT_DIR, f"body_{preset}_sheet.png"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    presets = arg("--presets", "human_f").split(",")
    samples = int(arg("--samples", "128"))
    h = float(arg("--h", "0.0025"))
    workers = int(arg("--workers", str(os.cpu_count() or 4)))
    res = tuple(int(x) for x in arg("--res", "800x1450").split("x"))
    t0 = time.time()
    meshes = {pr: get_mesh(pr, h, workers) for pr in presets}       # all CPU work first...
    for pr in presets:                                               # ...then all GPU work
        render(pr, meshes[pr], samples, res)
    print(f"turnaround done in {time.time() - t0:.0f}s -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
