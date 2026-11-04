"""GPU backend for Shape.eval (CuPy).

The sculpt is hundreds of primitive/brush closures written against numpy.  Instead of rewriting them, each
closure is cloned once with its captured constants moved to the GPU, and while a Shape is evaluated the
`np` name inside the model modules points at a shim that forwards to CuPy.  Build code (profiles, frames,
skeleton) keeps running on numpy; only the dense field evaluation moves to the GPU.

    gpu.enable()       -> True when CuPy and a CUDA device are usable (MODELGEN_GPU=0 turns it off)
    Shape.eval(...)    routes here automatically once enabled; takes and returns numpy arrays
"""
from __future__ import annotations

import os
import sys
import types

import numpy as _np

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE = False
xp = None                 # the device array module (cupy, or the strict fake used by gpu_check --fake)
_tried = False
_SKIP_MODULES = {"mesher", "gpu", "pmesh", "__main__"}
_KEEP_ATTRS = {"lo", "hi"}          # culling boxes stay on the host


def enable(backend=None):
    """Turn the GPU path on if possible.  backend: None (cupy), or a module object (testing)."""
    global ACTIVE, xp, _tried
    if ACTIVE:
        return True
    if os.environ.get("MODELGEN_GPU", "1") == "0":
        return False
    if backend is None:
        if _tried:
            return False
        _tried = True
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                return False
            (cupy.arange(4, dtype=cupy.float32) * 2).sum().item()   # driver/runtime actually work
            backend = cupy
            name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
            print("gpu: shape evaluation on", name.decode() if isinstance(name, bytes) else name, flush=True)
        except Exception as e:                                # no cupy / no CUDA: stay on the CPU
            if os.environ.get("MODELGEN_GPU") == "1":
                print("gpu: unavailable,", type(e).__name__, e, flush=True)
            return False
    xp = backend
    ACTIVE = True
    import mesher
    mesher.EVAL_POINTS = int(os.environ.get("MODELGEN_GPU_BATCH", 4_000_000))
    return True


def release():
    """Free the CuPy memory pool (call before Cycles renders in the same process)."""
    if ACTIVE and hasattr(xp, "get_default_memory_pool"):
        xp.get_default_memory_pool().free_all_blocks()


def disable():
    global ACTIVE
    ACTIVE = False


def is_device(a):
    return xp is not None and isinstance(a, xp.ndarray)


def to_host(a):
    if xp is not None and hasattr(xp, "asnumpy"):
        return xp.asnumpy(a)
    return _np.asarray(a)


# ------------------------------------------------------------------ numpy -> device shim
def _dev(a):
    """numpy array (or list/tuple holding some) -> device; everything else unchanged."""
    if isinstance(a, _np.ndarray) and not is_device(a) and a.dtype != object:
        return xp.asarray(a)
    if isinstance(a, (list, tuple)) and any(isinstance(x, _np.ndarray) and not is_device(x) for x in a):
        return type(a)(_dev(x) for x in a)
    return a


class _Shim(types.ModuleType):
    """Stands in for `np` during device evaluation: attribute lookups go to the device module (falling back
    to numpy for what it lacks); host numpy arrays passed to a function are moved to the device first."""

    def __init__(self, dev, host, name="np"):
        super().__init__(name)
        self._d, self._h, self._c = dev, host, {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        c = self._c
        if name in c:
            return c[name]
        val = getattr(self._d, name, None)
        hval = getattr(self._h, name, None)
        if val is None:
            val = hval
            if val is None:
                raise AttributeError(name)
        if isinstance(hval, types.ModuleType):                # np.linalg, ...
            val = _Shim(getattr(self._d, name, hval), hval, name)
        elif callable(val) and not isinstance(val, type) and val is not hval:
            fn = val
            if name == "interp":                              # cupy.interp needs device xp/fp
                def val(x, xp_, fp, *a, _f=fn, **k):
                    return _f(_dev(_np.asarray(x) if not is_device(x) else x),
                              xp.asarray(xp_), xp.asarray(fp), *a, **k)
            else:
                def val(*a, _f=fn, **k):
                    return _f(*[_dev(x) for x in a], **{kk: _dev(vv) for kk, vv in k.items()})
        c[name] = val
        return val


def _model_modules():
    out = []
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f or name in _SKIP_MODULES or os.path.dirname(os.path.abspath(f)) != HERE:
            continue
        if getattr(m, "np", None) is _np:
            out.append(m)
    return out


class _swap:
    depth = 0

    def __enter__(self):
        if _swap.depth == 0:
            self.mods = _model_modules()
            shim = _Shim(xp, _np)
            for m in self.mods:
                m.np = shim
        _swap.depth += 1

    def __exit__(self, *exc):
        _swap.depth -= 1
        if _swap.depth == 0:
            for m in self.mods:
                m.np = _np


# ------------------------------------------------------------------ device clones of closures
def _clone(obj, memo):
    oid = id(obj)
    if oid in memo:
        return memo[oid]
    if isinstance(obj, _np.ndarray):
        out = obj if obj.dtype == object or is_device(obj) else xp.asarray(obj)
    elif isinstance(obj, types.FunctionType):
        cells = None
        if obj.__closure__:
            cells = tuple(types.CellType() for _ in obj.__closure__)
        g = types.FunctionType(obj.__code__, obj.__globals__, obj.__name__, None, cells)
        memo[oid] = g
        if cells:
            for src, dst in zip(obj.__closure__, cells):
                try:
                    dst.cell_contents = _clone(src.cell_contents, memo)
                except ValueError:                            # empty cell
                    pass
        if obj.__defaults__:
            g.__defaults__ = tuple(_clone(d, memo) for d in obj.__defaults__)
        if obj.__kwdefaults__:
            g.__kwdefaults__ = {k: _clone(d, memo) for k, d in obj.__kwdefaults__.items()}
        g.__dict__.update({k: (v if k in _KEEP_ATTRS else _clone(v, memo)) for k, v in obj.__dict__.items()})
        return g
    elif isinstance(obj, types.MethodType):
        out = types.MethodType(_clone(obj.__func__, memo), _clone(obj.__self__, memo))
    elif isinstance(obj, list):
        out = []
        memo[oid] = out
        out.extend(_clone(x, memo) for x in obj)
        return out
    elif isinstance(obj, tuple):
        out = tuple(_clone(x, memo) for x in obj)
    elif isinstance(obj, dict):
        out = {}
        memo[oid] = out
        out.update({k: _clone(v, memo) for k, v in obj.items()})
        return out
    elif _is_model_instance(obj):
        import copy
        out = copy.copy(obj)
        memo[oid] = out
        for k, v in vars(obj).items():
            if k not in _KEEP_ATTRS:
                setattr(out, k, _clone(v, memo))
        return out
    else:
        out = obj
    memo[oid] = out
    return out


def _is_model_instance(obj):
    from sdf import Shape
    if isinstance(obj, (type, types.ModuleType, Shape)) or not hasattr(obj, "__dict__"):
        return False
    mod = sys.modules.get(type(obj).__module__)
    f = getattr(mod, "__file__", None)
    return bool(f) and os.path.dirname(os.path.abspath(f)) == HERE


def _device_lists(shape):
    st = shape.__dict__.setdefault("_gpu_state", {"memo": {}, "fn": {}, "keep": []})
    memo, fns = st["memo"], st["fn"]

    def dev(f):
        k = id(f)
        if k not in fns:
            fns[k] = _clone(f, memo)
            st["keep"].append(f)                              # keeps id(f) from being reused
        return fns[k]
    return [dev(pr.fn) for pr in shape.prims], [dev(d) for d in shape.displace]


def eval_shape(shape, p, lo, hi, with_tags):
    nested = is_device(p)
    if lo is None:
        lo = to_host(p.min(axis=0)) if nested else _np.asarray(p).min(axis=0)
        hi = to_host(p.max(axis=0)) if nested else _np.asarray(p).max(axis=0)
    fns, disps = _device_lists(shape)
    with _swap():
        q = p if nested else xp.asarray(_np.asarray(p, _np.float32))
        out = shape._eval_impl(q, _np.asarray(lo), _np.asarray(hi), with_tags, fns, disps)
    if nested:
        return out
    if with_tags:
        return to_host(out[0]), to_host(out[1])
    return to_host(out)
