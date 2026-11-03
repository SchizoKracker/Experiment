"""Numeric front-silhouette comparison: my render vs reference mask, per height (fractions of H)."""
import sys
import numpy as np
from PIL import Image

def runs_table(mask):
    rows = np.where(mask.any(1))[0]
    top, bot = rows[0], rows[-1]
    Hpx = bot - top
    cx = np.where(mask[top + int(0.04 * Hpx)])[0].mean()
    tab = {}
    for zf in np.arange(0.44, 0.9, 0.02):
        y = int(top + (1 - zf) * Hpx)
        r = mask[y].astype(int)
        d = np.diff(np.concatenate([[0], r, [0]]))
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        tab[round(zf, 2)] = [((a - cx) / Hpx, (b - cx) / Hpx) for a, b in zip(s, e)]
    return tab

def fmt(rs):
    return " ".join(f"[{a:+.3f},{b:+.3f}]" for a, b in rs if b > 0)

_L = np.asarray(Image.open(sys.argv[1]).convert("L")).astype(int); mine = _L > _L[:5, :5].mean() + 12
ref = np.load(sys.argv[2])
tm, tr = runs_table(mine), runs_table(ref)
print(" z    | ref (right side runs)                | mine")
for z in sorted(tm, reverse=True):
    print(f"{z:.2f} | {fmt(tr[z]):38s} | {fmt(tm[z])}")
