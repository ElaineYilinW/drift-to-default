"""Exploratory: does every low-level statistic regress toward a default, not only brightness?

Data: the held-opening runs with the generic prompt (40 openings: 20 NAVSIM, 20 generated; Self Forcing w21s3 with
4 seeds, LongLive w12s3 with 2 seeds). For each statistic x (L*, a*, b*, saturation, log sharpness, L* contrast = std of
L* within the frame), per opening (seed mean): x_open (held frames 1-8) and the change x_end - x_open (last 2 s).
Reported per model and statistic: slope of the change on x_open (-1 = complete regression, 0 = none), persistence
1 + slope, correlation, zero-drift level, and opening-cluster bootstrap 95% CIs; plus the time course of persistence.
usage: python analyze_attributes.py OUT MODEL=DIR[,DIR...] ...
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
runs = {a.split("=")[0]: a.split("=")[1].split(",") for a in sys.argv[2:]}
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
ATT = {"L*": lambda z: z["lab_mean"][:, 0], "a*": lambda z: z["lab_mean"][:, 1], "b*": lambda z: z["lab_mean"][:, 2],
       "saturation": lambda z: z["sat_mean"], "log sharpness": lambda z: np.log(z["sharpness"] + 1),
       "L* contrast": lambda z: z["lab_std"][:, 0]}
res, curves = {}, {}
for model, dirs in runs.items():
    vids = {}
    for d in dirs:
        for p in glob.glob(f"{d}/*_none_s*.npz"):
            m = re.match(r"((?:real|self)\d+)_none_s(\d+)\.npz$", os.path.basename(p))
            if m:
                vids.setdefault(m[1], []).append(np.load(p))
    res[model] = {}
    for an, f in ATT.items():
        X = np.array([np.mean([f(z) for z in zs], 0) for zs in vids.values()])        # opening x frame
        x0, x1 = X[:, 1:9].mean(1), X[:, -32:].mean(1)
        dx = x1 - x0

        def stat(i):
            b, a = np.polyfit(x0[i], dx[i], 1)
            return b, np.corrcoef(x0[i], dx[i])[0, 1], -a / b

        b, r, z = stat(np.arange(len(x0)))
        bs = np.array([stat(rng.integers(0, len(x0), len(x0))) for _ in range(4000)])
        res[model][an] = dict(n=int(len(x0)), slope=float(b), slope_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                              persistence=float(1 + b), r=float(r), r_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                              zero_drift=float(z), zero_ci=np.percentile(bs[:, 2], [2.5, 97.5]).tolist())
        curves[(model, an)] = [(t / 16 + 1, 1 + np.polyfit(x0, X[:, t:t + 16].mean(1) - x0, 1)[0]) for t in range(16, X.shape[1] - 16, 16)]
json.dump(res, open(f"{out}/attributes.json", "w"), indent=1)

fig, axs = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.6), sharey=True, squeeze=False)
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
for ax, model in zip(axs[0], runs):
    for c, an in zip(cols, ATT):
        t, p = zip(*curves[(model, an)])
        ax.plot(t, p, color=c, lw=1.8, label=an)
    ax.axhline(0, color="#898781", lw=.8)
    ax.set_title(model, fontsize=10, loc="left"), ax.set_xlabel("seconds")
axs[0][0].set_ylabel("persistence of the opening's value")
axs[0][0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{out}/attributes_persistence.png", dpi=140)
for model in runs:
    for an, v in res[model].items():
        print(f"{model:13s} {an:14s} persistence {v['persistence']:.2f} slope CI [{v['slope_ci'][0]:.2f},{v['slope_ci'][1]:.2f}] r {v['r']:.2f}")
