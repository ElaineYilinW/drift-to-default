"""Exploratory (not preregistered; written after study 23 was frozen, before its data): is persistence independent of
the shift size in study 6? Study 6 continued every opening at five L* levels (-16, -8, 0, +8, +16) for 31 s.

Per scene and configuration, with open = mean L* of the held frames 1..8 and end = mean L* of the last 2 s (as in
analyze_relax.py):
  alpha_small = (end_p8 - end_m8) / (open_p8 - open_m8)        alpha_large = (end_p16 - end_m16) / (open_p16 - open_m16)
  alpha_up = (end_p16 - end_z0) / (open_p16 - open_z0)          alpha_down = (end_z0 - end_m16) / (open_z0 - open_m16)
and the quadratic coefficient of end on centred open over the five levels. Reports scene means with scene-bootstrap
95% CIs of large - small and up - down per configuration, and the mean R^2 of the linear fit.
usage: python explore_linearity_s6.py RELAX_DIR OUT_JSON
"""
import glob
import json
import os
import re
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
HOLD = 9
pat = re.compile(r"^((?:rx|gx)\d\d)_(\w+?)__(m16|m8|z0|p8|p16)_s0\.npz$")
D = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        L = np.load(p)["lab_mean"][:, 0]
        fps = 24 if m[2] == "sky" else 16
        D.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = (float(L[1:HOLD].mean()), float(L[-2 * fps:].mean()))


def a(x, hi, lo):
    if hi not in x or lo not in x or x[hi][0] - x[lo][0] < 4:
        return None
    return (x[hi][1] - x[lo][1]) / (x[hi][0] - x[lo][0])


def ci(v):
    v = np.asarray(v, float)
    return [round(float(v.mean()), 3)] + np.round(np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)],
                                                               [2.5, 97.5]), 3).tolist()


res = {}
for cfg, scenes in sorted(D.items()):
    r = {k: [] for k in ("small", "large", "up", "down", "large_minus_small", "up_minus_down", "quad", "r2")}
    for x in scenes.values():
        v = {k: a(x, *p) for k, p in dict(small=("p8", "m8"), large=("p16", "m16"), up=("p16", "z0"),
                                           down=("z0", "m16")).items()}
        for k in ("small", "large", "up", "down"):
            if v[k] is not None:
                r[k].append(v[k])
        if v["small"] is not None and v["large"] is not None:
            r["large_minus_small"].append(v["large"] - v["small"])
        if v["up"] is not None and v["down"] is not None:
            r["up_minus_down"].append(v["up"] - v["down"])
        if len(x) == 5:
            o = np.array([x[k][0] for k in ("m16", "m8", "z0", "p8", "p16")])
            e = np.array([x[k][1] for k in ("m16", "m8", "z0", "p8", "p16")])
            r["quad"].append(float(np.polyfit(o - o.mean(), e, 2)[0]))
            b1, b0 = np.polyfit(o, e, 1)
            r["r2"].append(float(1 - ((e - b1 * o - b0) ** 2).sum() / ((e - e.mean()) ** 2).sum()))
    res[cfg] = {k: (ci(v) if len(v) >= 3 else None) for k, v in r.items()} | {"n": len(r["large_minus_small"])}
json.dump(res, open(out, "w"), indent=1)
print(f"{'config':10s} {'small':>7s} {'large':>7s} {'large-small [95% CI]':>24s} {'up':>7s} {'down':>7s} "
      f"{'up-down [95% CI]':>24s} {'R2':>5s}  n")
for cfg, r in res.items():
    if r["large_minus_small"] is None:
        continue
    f = lambda k: f"{r[k][0]:+.2f} [{r[k][1]:+.2f}, {r[k][2]:+.2f}]" if r[k] else "-"
    print(f"{cfg:10s} {r['small'][0]:7.2f} {r['large'][0]:7.2f} {f('large_minus_small'):>24s} {r['up'][0]:7.2f} "
          f"{r['down'][0]:7.2f} {f('up_minus_down'):>24s} {r['r2'][0] if r['r2'] else float('nan'):5.2f} {r['n']:2d}")
