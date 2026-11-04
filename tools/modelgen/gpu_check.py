"""Check the GPU field evaluation against the CPU and time both.

usage: python gpu_check.py [preset=human_f] [--fake] [--n 2000000]
  --fake   use a strict CPU stand-in for CuPy (raises on any host/device array mixing, like CuPy does):
           validates the device-clone machinery on a machine without a GPU.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np


# ------------------------------------------------------------------ strict fake device module
class Dev(np.ndarray):
    """ndarray that refuses to mix with plain (host) ndarrays, like cupy.ndarray."""

    def __array_ufunc__(self, uf, method, *inputs, **kw):
        _check(inputs)
        args = [x.view(np.ndarray) if isinstance(x, Dev) else x for x in inputs]
        if "out" in kw:
            _check(kw["out"])
            kw["out"] = tuple(x.view(np.ndarray) if isinstance(x, Dev) else x for x in kw["out"])
        r = getattr(uf, method)(*args, **kw)
        return _wrap(r)

    def __array_function__(self, func, types, args, kwargs):
        _check(args)
        _check(tuple(kwargs.values()))
        r = func(*_unwrap(args), **_unwrap(kwargs))
        return _wrap(r)

    def __setitem__(self, k, v):
        _check((v,))
        np.ndarray.__setitem__(self.view(np.ndarray), _unwrap(k), _unwrap(v))

    def __getitem__(self, k):
        _check((k,) if not isinstance(k, tuple) else k)
        return _wrap(np.ndarray.__getitem__(self.view(np.ndarray), _unwrap(k)))

    def __iter__(self):                                  # CuPy allows it (one small array per row)
        return (_wrap(np.asarray(r)) for r in self.view(np.ndarray))


def _host(x):
    return isinstance(x, np.ndarray) and not isinstance(x, Dev) and x.ndim > 0


def _check(seq):
    for x in seq:
        if _host(x):
            raise TypeError(f"host ndarray {x.shape} mixed into a device op")
        if isinstance(x, (list, tuple)):
            _check(x)


def _unwrap(x):
    if isinstance(x, Dev):
        return x.view(np.ndarray)
    if isinstance(x, (list, tuple)):
        return type(x)(_unwrap(y) for y in x)
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    return x


def _wrap(r):
    if isinstance(r, np.ndarray) and not isinstance(r, Dev):
        return r.view(Dev)
    if isinstance(r, tuple):
        return tuple(_wrap(x) for x in r)
    return r


class _FakeMod:
    _FAKE = True
    ndarray = Dev

    def __init__(self, base):
        self._b = base

    def __getattr__(self, name):
        v = getattr(self._b, name)
        if isinstance(v, type(np)):
            return _FakeMod(v)
        if callable(v) and not isinstance(v, type):
            def f(*a, _v=v, **k):
                _check(a)
                _check(tuple(k.values()))
                return _wrap(_v(*_unwrap(a), **_unwrap(k)))
            return f
        return v

    def asarray(self, x, dtype=None):
        if _host(x) or isinstance(x, (list, tuple, float, int)) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return np.array(x, dtype=dtype).view(Dev)
        _check((x,))
        return _wrap(np.asarray(_unwrap(x), dtype=dtype))

    def array(self, x, dtype=None):
        return self.asarray(x, dtype).copy()

    def asnumpy(self, x):
        return np.array(_unwrap(x))


# ------------------------------------------------------------------ check
def main():
    os.environ.setdefault("MODELGEN_GPU", "1")
    preset = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "human_f"
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 2_000_000
    import gpu
    import pmesh
    import mesher
    t = time.time()
    _, _, ref = pmesh.build((preset, 72))                    # reference shape, built on the CPU
    print(f"built on CPU in {time.time() - t:.1f}s")
    lo, hi = ref.bounds()
    rng = np.random.default_rng(1)
    v, _ = mesher.mesh_shape(ref, h=0.012, verbose=False)    # coarse surface: test points near the skin too
    pts = np.concatenate([rng.uniform(lo, hi, (n // 2, 3)),
                          v[rng.integers(0, len(v), n - n // 2)] + rng.normal(0, 0.004, (n - n // 2, 3))]).astype(np.float32)
    t = time.time()
    d_cpu = np.concatenate([ref.eval(pts[i:i + 400000]) for i in range(0, n, 400000)])
    t_cpu = time.time() - t
    print(f"CPU: {n} points in {t_cpu:.1f}s")

    backend = _FakeMod(np) if "--fake" in sys.argv else None
    if not gpu.enable(backend):
        sys.exit("GPU backend unavailable (is cupy installed for your CUDA version?)")
    t = time.time()
    _, _, dev = pmesh.build((preset, 72))                    # same shape, built with build-time evals on the GPU
    print(f"built with GPU evals in {time.time() - t:.1f}s")
    for name, shp in (("cpu-built", ref), ("gpu-built", dev)):
        dev_eval = shp.eval(pts[:1000])                        # warm-up (kernel compilation)
        t = time.time()
        bs = mesher.EVAL_POINTS
        d_gpu = np.concatenate([shp.eval(pts[i:i + bs]) for i in range(0, n, bs)])
        t_gpu = time.time() - t
        err = np.abs(d_gpu - d_cpu)
        near = np.abs(d_cpu) < 0.01
        print(f"GPU ({name} shape): {t_gpu:.2f}s ({t_cpu / max(t_gpu, 1e-9):.1f}x)   max |diff| {err.max():.2e} m, "
              f"near-surface max {err[near].max():.2e} m")
        if err[near].max() > 2e-5:
            sys.exit("MISMATCH near the surface")
    print("OK")


if __name__ == "__main__":
    main()
