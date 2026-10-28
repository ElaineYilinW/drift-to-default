"""Exploratory: share of attention that Self Forcing's queries would give to a three-frame sink at each query-sink
distance (jobs/run_sinkprobe.sh, hold_gen_sinkpos.py --probe). Each PROBE_DIR/{scene}.npz holds, per probed chunk
(start latent frame c), an array (calls, distances, heads): calls are 4 denoising passes + 1 context pass, 30 layers
each, in call order (pass-major). Prints the mean share over heads, layers, passes, chunks and scenes for every distance,
its local maxima, and the same curve per pass; writes OUT.json.
usage: python analyze_sinkprobe.py PROBE_DIR OUT.json
"""
import glob
import json
import os
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
NL = 30
curves, by_pass, by_layer, natural = [], [], [], {}
for p in sorted(glob.glob(f"{src}/[a-z][a-z][0-9][0-9].npz")):
    d = np.load(p)
    grid = d["grid"]
    for c in d["chunks"]:
        a = d[f"c{c}"]                          # (calls, grid, heads)
        npass = a.shape[0] // NL
        a = a[:npass * NL].reshape(npass, NL, len(grid), -1)
        curves.append(a.mean((0, 1, 3)))
        by_pass.append(a.mean((1, 3)))
        by_layer.append(a.mean((0, 3)))
        dn = int(c) + 2                          # natural distance of the chunk's newest frame to sink frame 0
        if dn in set(grid.tolist()):
            natural.setdefault(int(c), []).append(float(a.mean((0, 1, 3))[list(grid).index(dn)]))
grid = grid.tolist()
m = np.mean(curves, 0)
sm = np.convolve(m, np.ones(3) / 3, mode="same")
peaks = [grid[i] for i in range(2, len(grid) - 2) if sm[i] == max(sm[i - 2:i + 3]) and sm[i] > np.median(sm)]
res = {"grid": grid, "mean": m.tolist(), "per_pass": np.mean(by_pass, 0).tolist(),
       "per_layer": np.mean(by_layer, 0).tolist(), "peaks": peaks,
       "natural": {c: float(np.mean(v)) for c, v in sorted(natural.items())}, "n_curves": len(curves)}
json.dump(res, open(out, "w"))
print("curves", len(curves), "peaks (3-pt smoothed local maxima above median):", peaks)
for D in range(grid[0], grid[-1] + 1, 4):
    i = grid.index(D)
    print(f"{D:4d} {m[i]:.4f} " + "#" * int(200 * m[i] / m.max() / 4))
