"""Real-opening experiment: continuations from a held real NAVSIM front-camera frame, with and without correction.

Drift is measured relative to the real opening chunk (which is also the anchor): per video, the change of mean
L* / a* / b* / saturation over time, end-of-video colour shift dE_ab = |Lab(end) - Lab(opening)| (frame means,
last 2 s), its signed L component, and flicker. Arms are paired within scene and seed (same GPU architecture).
Scene-cluster bootstrap CIs for the paired differences. usage: python analyze_real.py RUN_DIR OUT_DIR
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
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
pat = re.compile(r"((?:real|self)\d+)_(none|ttc_s1|ttc_s1_b15)_s(\d+)\.npz$")
rng = np.random.default_rng(0)
rows, curves = [], []
for p in sorted(glob.glob(f"{src}/*.npz")):
    m = pat.match(os.path.basename(p))
    if not m:
        continue
    z = np.load(p)
    lab, sat = z["lab_mean"], z["sat_mean"]
    o_lab, o_sat = lab[1:9].mean(0), sat[1:9].mean()          # the held real frame (decoded opening chunk)
    e_lab, e_sat = lab[-32:].mean(0), sat[-32:].mean()
    L = lab[:, 0]
    rows.append(dict(scene=m[1], arm=m[2], seed=int(m[3]), L_open=o_lab[0], dL_end=e_lab[0] - o_lab[0],
                     absdL_end=abs(e_lab[0] - o_lab[0]), dE_end=float(np.linalg.norm(e_lab - o_lab)),
                     dsat_end=e_sat - o_sat, jitter=float(np.std(np.diff(L, 2))),
                     boundary=float(np.mean([abs(L[1 + 12 * k] - L[12 * k]) for k in range(1, 41)]))))
    sec = np.arange(len(L)) / 16
    curves.append(dict(scene=m[1], arm=m[2], seed=int(m[3]),
                       dE=[float(np.linalg.norm(lab[(sec >= s) & (sec < s + 1)].mean(0) - o_lab)) for s in range(31)]))
df = pd.DataFrame(rows)
df.to_csv(f"{out}/real_videos.csv", index=False)
res = {"n": df.groupby("arm").size().to_dict(), "drift_direction_none": {
    "frac_brighter": float((df[df.arm == "none"].dL_end > 0).mean()),
    "mean_dL": float(df[df.arm == "none"].dL_end.mean()), "mean_absdL": float(df[df.arm == "none"].absdL_end.mean())}}
w = df.pivot_table(index=["scene", "seed"], columns="arm", values=["dE_end", "absdL_end", "dsat_end", "jitter", "boundary"])
for arm in [x for x in ["ttc_s1", "ttc_s1_b15"] if x in set(df.arm)]:
    r = {}
    for y in ["dE_end", "absdL_end", "dsat_end"]:
        diff = (w[(y, arm)] - w[(y, "none")]).groupby(level=0).mean().dropna()
        bs = [diff.sample(len(diff), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 2**31, 2000)]
        r[y] = dict(none=float(w[(y, "none")].mean()), arm=float(w[(y, arm)].mean()), diff=float(diff.mean()),
                    ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    scenes_improved=int((diff < 0).sum()), n_scenes=int(len(diff)))
    r["flicker_ratio"] = {y: float(w[(y, arm)].mean() / w[(y, "none")].mean()) for y in ["jitter", "boundary"]}
    res[arm] = r
json.dump(res, open(f"{out}/real_summary.json", "w"), indent=1)
C = pd.DataFrame(curves)
fig, ax = plt.subplots(figsize=(7, 4))
for arm, g in C.groupby("arm"):
    M = np.array(g.dE.tolist())
    m, se = M.mean(0), M.std(0) / np.sqrt(len(M))
    ax.plot(np.arange(31) + .5, m, label=arm)
    ax.fill_between(np.arange(31) + .5, m - 1.96 * se, m + 1.96 * se, alpha=.2)
ax.set_xlabel("seconds"), ax.set_ylabel("colour shift from the real opening (dE_ab)"), ax.legend()
fig.tight_layout()
fig.savefig(f"{out}/real_dE_curve.png", dpi=110)
print(json.dumps(res, indent=1))
