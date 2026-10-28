"""Exploratory: the attention probe over sink distances 3..723 (jobs/run_sinkprobe_long.sh: openings rx02, rx03, gx02,
gx03, chunks 24, 60, ..., 204). As analyze_sinkprobe.py (mean share over heads, layers, passes, chunks and openings per
distance), plus a peak rule fixed before this output was looked at (committed with this script): the major peaks are
the local maxima of the unsmoothed mean curve with prominence >= 0.04 (scipy.signal.find_peaks). On the first probe
(sinkprobe_out.json, distances 3..300) this rule returns exactly 71, 134, 203, 266. A sink at its original positions
reaches distance P at (P - 5) / 4 s after the opening. Also writes the probe's predicted share at the natural distance
for every second (mean over D = 4t + 5 .. 4t + 8), for the probe check of study 26.
usage: python analyze_sinkprobe_long.py PROBE_DIR OUT.json [FIRST_PROBE.json]
"""
import glob
import json
import sys

import numpy as np
from scipy.signal import find_peaks

src, out = sys.argv[1], sys.argv[2]
NL, PROM = 30, 0.04
curves, by_open, by_chunk = [], {}, {}
for p in sorted(glob.glob(f"{src}/[a-z][a-z][0-9][0-9].npz")):
    d = np.load(p)
    grid = d["grid"]
    for c in d["chunks"]:
        a = d[f"c{c}"]                          # (calls, grid, heads)
        npass = a.shape[0] // NL
        cv = a[:npass * NL].reshape(npass, NL, len(grid), -1).mean((0, 1, 3))
        curves.append(cv)
        by_open.setdefault(p.split("/")[-1][:4], []).append(cv)
        by_chunk.setdefault(int(c), []).append(cv)
grid = grid.tolist()
g = np.array(grid)
m = np.mean(curves, 0)


def major(curve):
    i, pr = find_peaks(curve, prominence=PROM)
    return [int(g[k]) for k in i], [float(x) for x in pr["prominences"]]


peaks, prom = major(m)
res = {"grid": grid, "mean": m.tolist(), "n_curves": len(curves), "prominence": PROM, "peaks": peaks,
       "peak_prominence": prom, "peak_share": [float(m[g == P][0]) for P in peaks],
       "onsets_s": [(P - 5) / 4 for P in peaks],
       "peaks_by_opening": {k: major(np.mean(v, 0))[0] for k, v in by_open.items()},
       "peaks_by_chunk": {k: major(np.mean(v, 0))[0] for k, v in sorted(by_chunk.items())},
       "median_share": float(np.median(m[g >= 10])),
       "share_by_second": {t: float(m[(g >= 4 * t + 5) & (g <= 4 * t + 8)].mean()) for t in range(0, 180)
                           if (g <= 4 * t + 8).any() and (g >= 4 * t + 5).any() and 4 * t + 8 <= g[-1]}}
if len(sys.argv) > 3:
    f = json.load(open(sys.argv[3]))
    f_g, f_m = np.array(f["grid"]), np.array(f["mean"])
    i, _ = find_peaks(f_m, prominence=PROM)
    res["first_probe_peaks"] = [int(f_g[k]) for k in i]
    common = [D for D in grid if D in set(f["grid"])]
    res["first_probe_r"] = float(np.corrcoef([m[g == D][0] for D in common], [f_m[f_g == D][0] for D in common])[0, 1])
json.dump(res, open(out, "w"))
print("curves", len(curves), "major peaks (prominence >= %.2f):" % PROM, peaks)
print("onsets (s):", res["onsets_s"])
print("by opening:", res["peaks_by_opening"])
print("by chunk:", res["peaks_by_chunk"])
if "first_probe_peaks" in res:
    print("first probe, same rule:", res["first_probe_peaks"], "r over common distances:", round(res["first_probe_r"], 3))
for D in range(grid[0], grid[-1] + 1, 6):
    i = grid.index(D)
    print(f"{D:4d} {(D - 5) / 4:6.1f}s {m[i]:.4f} " + "#" * int(200 * m[i] / m.max() / 4))
