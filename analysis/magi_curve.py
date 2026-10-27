"""Descriptive persistence curves of MAGI-1 for the figures (studies 18 and 19; not a preregistered analysis).

alpha over 1-s windows [t - 1, t) after the opening, per arm, with scene-bootstrap 95% CIs (10,000 draws).
Files {scene}_{arm}__{m16|p16}_s0.npz in each RUN_DIR (24 fps, frames 0..8 held, open = mean L* of frames 1..8).
usage: python magi_curve.py OUT_JSON RUN_DIR [RUN_DIR ...]
"""
import glob
import json
import os
import re
import sys

import numpy as np

FPS, HOLD = 24, 9
rng = np.random.default_rng(0)
out, dirs = sys.argv[1], sys.argv[2:]
D = {}
for d in dirs:
    for p in glob.glob(f"{d}/*.npz"):
        m = re.match(r"^(\w+?)_(mg\w+?)__(m16|p16)_s0\.npz$", os.path.basename(p))
        if m:
            D.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]
res = {}
for arm, sc in D.items():
    curve = {}
    for t in range(1, 61):
        v = []
        for s, x in sc.items():
            if len(x) < 2 or HOLD + t * FPS > min(len(x["m16"]), len(x["p16"])):
                continue
            i0, i1 = HOLD + (t - 1) * FPS, HOLD + t * FPS
            v.append((x["p16"][i0:i1].mean() - x["m16"][i0:i1].mean()) / (x["p16"][1:HOLD].mean() - x["m16"][1:HOLD].mean()))
        if len(v) >= 3:
            v = np.array(v)
            b = [rng.choice(v, len(v)).mean() for _ in range(10000)]
            curve[t] = dict(mean=float(v.mean()), ci=np.percentile(b, [2.5, 97.5]).tolist(), n=len(v))
    res[arm] = curve
json.dump(res, open(out, "w"), indent=1)
print({a: len(c) for a, c in res.items()})
