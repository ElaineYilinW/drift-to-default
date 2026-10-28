"""Exploratory, after the frozen analysis of study 24: does the share level off between 60 and 180 s?
Per scene: alpha over the 60-s windows [60, 120) and [120, 180) and their difference; a per-scene least-squares slope of
the 10-s alpha from 70 to 180 s (per 100 s); scene bootstrap CIs (10,000 draws).
usage: python explore_s24_windows.py RUN_DIR OUT_JSON
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
FPS, HOLD = 16, 9
pat = re.compile(r"^(\w+?)_(sf180_s0|sf180_s3rr|sf180_s3|rf180|ll180_s3)__(m16|p16)_s0\.npz$")
L = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        L.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]


def alpha(x, t0, t1):
    d = float(x["p16"][1:HOLD].mean() - x["m16"][1:HOLD].mean())
    i0, i1 = HOLD + int(t0 * FPS), HOLD + int(t1 * FPS)
    return (x["p16"][i0:i1].mean() - x["m16"][i0:i1].mean()) / d if d >= 4 else np.nan


def summ(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    b = [rng.choice(v, len(v)).mean() for _ in range(10000)]
    return dict(est=float(v.mean()), ci=np.percentile(b, [2.5, 97.5]).tolist(),
                p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


R = {}
Ts = np.arange(70, 181, 10)
for arm, S in sorted(L.items()):
    a1, a2, dd, sl = [], [], [], []
    for s, x in S.items():
        if "m16" not in x or "p16" not in x:
            continue
        u, w = alpha(x, 60, 120), alpha(x, 120, 180)
        a1.append(u); a2.append(w); dd.append(w - u)
        y = np.array([alpha(x, T - 10, T) for T in Ts])
        sl.append(np.polyfit(Ts, y, 1)[0] * 100)
    R[arm] = {"a60_120": summ(a1), "a120_180": summ(a2), "diff": summ(dd), "slope_per100s_70_180": summ(sl)}
    print(arm, {k: (round(v["est"], 3), [round(c, 3) for c in v["ci"]]) for k, v in R[arm].items()})
json.dump(R, open(out, "w"), indent=1)
