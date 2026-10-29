"""Exploratory: which temporal RoPE frequencies make the peaks of the attention-to-sink probe (jobs/run_sinkband.sh,
patch_sf_sinkband.py)? For each frequency band (the frequencies rotated to the grid distance, the others held at the
reference distance 20) the mean share over passes, layers, heads, chunks and openings per distance, as
analyze_sinkprobe.py; its major peaks by the rule of analyze_sinkprobe_long.py (prominence >= 0.04); its correlation
with the all-frequency curve; and the peak contrast, the mean share at the four peaks 71, 134, 203, 266 minus the median
share over distances 10..300. Temporal frequency j has period 2*pi*10000^(2j/44) latent frames.
usage: python analyze_sinkband.py BAND_DIR OUT.json
"""
import glob
import json
import sys

import numpy as np
from scipy.signal import find_peaks

src, out = sys.argv[1], sys.argv[2]
NL, PROM, PEAKS = 30, 0.04, [71, 134, 203, 266]
A = list(range(22))
GROUPS = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12)), list(range(12, 22))]
NAMES = (["all", "none"] + [f"only j{g[0]}-{g[-1]}" for g in GROUPS] + [f"all but j{g[0]}-{g[-1]}" for g in GROUPS]
         + [f"all but j{k}" for k in range(12)])
period = [2 * np.pi * 10000 ** (2 * j / 44) for j in A]
curves = []
for p in sorted(glob.glob(f"{src}/[a-z][a-z][0-9][0-9].npz")):
    d = np.load(p)
    grid = d["grid"]
    for c in d["chunks"]:
        a = d[f"c{c}"]                          # (calls, bands, grid, heads)
        npass = a.shape[0] // NL
        curves.append(a[:npass * NL].reshape(npass, NL, *a.shape[1:]).mean((0, 1, 4)))   # (bands, grid)
g = np.array(grid)
m = np.mean(curves, 0)
assert m.shape[0] == len(NAMES), m.shape
full = m[0]
med = (g >= 10)
res = {"grid": g.tolist(), "n_curves": len(curves), "period": period, "bands": {}}
for b, name in enumerate(NAMES):
    cv = m[b]
    i, pr = find_peaks(cv, prominence=PROM)
    contrast = float(np.mean([cv[g == P][0] for P in PEAKS]) - np.median(cv[med]))
    r = float(np.corrcoef(cv, full)[0, 1]) if cv.std() > 0 else float("nan")
    res["bands"][name] = {"curve": cv.tolist(), "peaks": [int(g[k]) for k in i], "contrast": contrast, "r_full": r,
                          "median": float(np.median(cv[med]))}
    print(f"{name:16s} contrast {contrast:+.4f}  r {r:+.3f}  median {np.median(cv[med]):.4f}  peaks {[int(g[k]) for k in i]}")
json.dump(res, open(out, "w"))
