"""Parallel meshing: one mesher, with every field evaluation spread over all CPU cores (or run on the GPU).

The mesher is unchanged; it is handed a proxy shape whose eval() splits each batch of points into chunks
and evaluates them in a pool of worker processes.  Every chunk is evaluated with the SAME culling box the
serial call would use, so the mesh is identical to a single-process run (no seams, no welding).
With the GPU backend active the evaluation runs in-process on the GPU instead.

    mesh(spec, h, clip=None)              in-process (the caller must be under `if __name__ == "__main__"`)
    mesh_subprocess(spec, h, clip=None)   from any script (runs this file as the pool's main module,
                                          which keeps Windows' spawn start method from re-running the caller)
spec = (preset, arm_drop_deg): the shape is rebuilt from it once in each worker.
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def build(spec):
    import races, humanoid as hu, body
    preset, arm_drop = spec
    hu.ARM_DROP_DEG = arm_drop
    p = races.preset(preset)
    J = hu.build_skeleton(p)
    return p, J, body.build_body(p, J)


def total_ram_gb():
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return 16.0


def default_workers():
    env = os.environ.get("MODELGEN_WORKERS")
    if env:
        return max(1, int(env))
    # each worker holds one shape + a chunk of points (well under 1 GB); leave headroom for the OS/Blender
    return max(1, min(os.cpu_count() or 4, int(total_ram_gb() * 0.6)))


# ------------------------------------------------------------------ worker side
_W = {}


def _init(spec):
    os.environ["MODELGEN_GPU"] = "0"
    _W["shape"] = build(spec)[2]


def _eval_chunk(job):
    pts, lo, hi = job
    return _W["shape"].eval(pts, lo, hi)


# ------------------------------------------------------------------ parent side
class PoolShape:
    """Drop-in for a Shape in mesher.mesh_shape / mesher.gradient: eval() runs on the worker pool."""

    def __init__(self, shape, pool, workers, min_chunk=4000):
        self.shape, self.pool, self.workers, self.min_chunk = shape, pool, workers, min_chunk

    def bounds(self):
        return self.shape.bounds()

    def eval(self, p, lo=None, hi=None, with_tags=False):
        p = np.asarray(p, np.float32)
        if with_tags or len(p) < 2 * self.min_chunk:
            return self.shape.eval(p, lo, hi, with_tags)
        if lo is None:
            lo, hi = p.min(axis=0), p.max(axis=0)
        n = min(self.workers * 2, len(p) // self.min_chunk)
        parts = np.array_split(p, n)
        return np.concatenate(list(self.pool.map(_eval_chunk, [(c, lo, hi) for c in parts])))


class Pool:
    """Context manager: a PoolShape on all cores (or the plain shape when the GPU backend is active)."""

    def __init__(self, spec, workers=None, shape=None):
        self.spec, self.workers, self.shape = spec, workers or default_workers(), shape

    def __enter__(self):
        import gpu
        if self.shape is None:
            self.shape = build(self.spec)[2]
        self.ex = None
        if gpu.enable() or self.workers <= 1:
            return self.shape
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # 1 BLAS thread per worker
            os.environ.setdefault(k, "1")
        self.ex = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"),
                                      initializer=_init, initargs=(self.spec,))
        return PoolShape(self.shape, self.ex, self.workers)

    def __exit__(self, *exc):
        if self.ex is not None:
            self.ex.shutdown()


def mesh(spec, h, clip=None, workers=None, normals=False, verbose=True):
    import gpu
    import mesher
    from sdf import F
    t0 = time.time()
    with Pool(spec, workers) as s:
        v, t = mesher.mesh_shape(s, h=h, clip=clip, verbose=False)
        n = mesher.gradient(s, v.astype(F)) if normals else None
    if verbose:
        where = "the GPU" if gpu.ACTIVE else f"{getattr(s, 'workers', 1)} CPU workers"
        print(f"  [mesh] {len(t)} tris on {where} in {time.time() - t0:.1f}s", flush=True)
    return (v, t, n) if normals else (v, t)


def mesh_subprocess(spec, h, clip=None, workers=None, normals=False):
    """Same as mesh(), safe to call from an unguarded script (headshot.py)."""
    fd, out = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    cmd = [sys.executable, os.path.abspath(__file__), spec[0], str(spec[1]), repr(h), out]
    if clip is not None:
        cmd += ["--clip"] + [repr(float(x)) for x in list(clip[0]) + list(clip[1])]
    if workers:
        cmd += ["--workers", str(workers)]
    if normals:
        cmd.append("--normals")
    subprocess.check_call(cmd)
    C = np.load(out)
    res = (C["v"], C["t"], C["n"]) if normals else (C["v"], C["t"])
    C.close()
    os.remove(out)
    return res


if __name__ == "__main__":
    a = sys.argv
    spec = (a[1], float(a[2]))
    clip = None
    if "--clip" in a:
        c = [float(x) for x in a[a.index("--clip") + 1:a.index("--clip") + 7]]
        clip = (np.array(c[:3], np.float32), np.array(c[3:], np.float32))
    w = int(a[a.index("--workers") + 1]) if "--workers" in a else None
    r = mesh(spec, float(a[3]), clip=clip, workers=w, normals="--normals" in a)
    np.savez(a[4], v=r[0], t=r[1], **({"n": r[2]} if len(r) > 2 else {}))
