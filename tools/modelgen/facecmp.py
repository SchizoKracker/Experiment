"""Numeric face comparison against the orthographic head reference (4.webp front + side).

Measures, per height (zn: 0 = top of skull, 1 = chin), in head heights:
  side profile: front-most point of the head (incl. nose, lips, chin) relative to the back of the skull
  front view:   half-width of the head (incl. ears)
for both the reference silhouettes and the model (computed directly from the SDF), and prints
the differences.  Same method as the body (silcmp.py).
"""
import sys
import numpy as np
import races, humanoid as hu, body
import hair as hr
from sdf import F

REF = "/tmp/claude-0/ref/head_mask.npy"


def ref_profiles():
    m = np.load(REF)
    side = m[:, 610:]
    front = m[:, :600]
    top, chin = 10, 562
    hh = chin - top
    rows = np.arange(top, 690)
    zn = (rows - top) / hh
    fr, bk, w = [], [], []
    fcx = None
    def main_run(row):
        d = np.diff(np.concatenate([[0], row.astype(int), [0]]))
        s_, e_ = np.where(d == 1)[0], np.where(d == -1)[0]
        if len(s_) == 0:
            return None
        i = np.argmax(e_ - s_)
        return s_[i], e_[i] - 1
    for y in rows:
        r = main_run(side[y])
        fr.append(r[0] if r else np.nan)
        bk.append(r[1] if r else np.nan)
        q = main_run(front[y])
        if fcx is None and q:
            fcx = (q[0] + q[1]) / 2
        w.append((q[1] - q[0]) / 2 if q else np.nan)
    fr, bk, w = np.array(fr, float), np.array(bk, float), np.array(w, float)
    back_ref = np.nanmax(bk[(zn > 0.2) & (zn < 0.5)])     # back of the cranium only
    depth = (back_ref - fr) / hh          # front-most point measured from the back of the skull
    return zn, depth, np.array(w) / hh


def model_profiles(p, J, s, zs):
    hf = hr.HeadFrame(J)
    hh = hf.hh
    ys = np.linspace(hf.yc - 0.7 * hh, hf.yc + 0.6 * hh, 700).astype(F)
    xs = np.linspace(0, 0.5 * hh, 260).astype(F)
    depth, width = [], []
    backs = []
    for zn in zs:
        z = hf.top - zn * hh
        P = np.stack([np.zeros_like(ys), ys, np.full_like(ys, z)], 1)
        d = s.eval(P)
        inside = np.where(d < 0)[0]
        if len(inside):
            depth.append(ys[inside.min()])
            backs.append(ys[inside.max()])
        else:
            depth.append(np.nan); backs.append(np.nan)
        # width: scan x at the y of max extent (min over a band of y)
        X, Y = np.meshgrid(xs, np.linspace(hf.yc - 0.5 * hh, hf.yc + 0.5 * hh, 90).astype(F), indexing="ij")
        Q = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z, F)], 1)
        dd = s.eval(Q).reshape(X.shape).min(1)
        ii = np.where(dd < 0)[0]
        width.append(xs[ii.max()] / hh if len(ii) else np.nan)
    bz = []
    for zn in np.arange(0.2, 0.5, 0.02):
        z = hf.top - zn * hh
        P = np.stack([np.zeros_like(ys), ys, np.full_like(ys, z)], 1)
        ii = np.where(s.eval(P) < 0)[0]
        if len(ii):
            bz.append(ys[ii.max()])
    back = max(bz)
    return (back - np.array(depth)) / hh, np.array(width)


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "human_f"
    p = races.preset(preset)
    J = hu.build_skeleton(p)
    s = body.build_body(p, J)
    zr, dr, wr = ref_profiles()
    zs = np.arange(0.40, 1.16, 0.02)
    dm, wm = model_profiles(p, J, s, zs)
    labels = {0.5: "brow", 0.54: "nasion", 0.56: "eye", 0.68: "nose tip", 0.74: "subnasale", 0.78: "upper lip",
              0.82: "mouth", 0.86: "lower lip", 0.9: "labiomental", 0.96: "chin", 1.02: "under chin"}
    print(" zn   | profile depth ref  model   diff  | half-width ref  model  diff")
    err = []
    for i, z in enumerate(zs):
        rd = np.interp(z, zr, dr)
        rw = np.interp(z, zr, wr)
        lab = labels.get(round(z, 2), "")
        print(f"{z:.2f} | {rd:16.3f} {dm[i]:6.3f} {dm[i] - rd:+6.3f} | {rw:13.3f} {wm[i]:6.3f} {wm[i] - rw:+6.3f}  {lab}")
        err.append(abs(dm[i] - rd))
    print("mean |profile diff| =", round(float(np.nanmean(err)), 4))


if __name__ == "__main__":
    main()
