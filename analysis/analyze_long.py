"""Sink retention at 60 s (preregistered, docs/preregistration_long_cn.md).

24 scenes x opening L* shift -16 / 0 / +16 x {sink 3, no sink} x {Self Forcing DMD, CausVid}, 60 s (243 latent frames
-> 969 frames), seed 0, generic prompt; each scene x model in one task (arch_guard). Files {scene}_{cfg}__{ver}_s0.npz,
cfg in sf60_s3, sf60_s0, cv60_s3, cv60_s0.
alpha(T) = within-scene OLS slope of the L* mean over seconds [T-10, T) after the held opening on the opening L*;
primary endpoint T = 60 s.
Primary (Holm over 2): H1 alpha(sf60_s3) > alpha(sf60_s0); H2 alpha(cv60_s3) > alpha(cv60_s0) (paired over scenes,
two-sided one-sample t-test of the per-scene difference, scene-bootstrap 95% CI; supported if Holm p < .05 and > 0).
Secondary: alpha at 20 / 30 / 40 / 50 / 60 s per configuration; whether alpha(sf60_s3, 60 s) lies in the band
[0.25, 0.54] predicted by the chunk-recursion fit of preregistration 06 and |alpha(sf60_s0, 60 s)| < 0.15.
usage: python analyze_long.py RUN_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
pat = re.compile(r"^(\w+?)_(\w+?)__(m16|z0|p16)_s0\.npz$")
D = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        D.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]


def slopes(cfg, T):
    o = {}
    for s, v in D.get(cfg, {}).items():
        if len(v) < 3 or len(v["z0"]) < 9 + T * 16:
            continue
        x = np.array([v[k][1:9].mean() for k in ("m16", "z0", "p16")])
        y = np.array([v[k][9 + (T - 10) * 16:9 + T * 16].mean() for k in ("m16", "z0", "p16")])
        o[s] = np.polyfit(x, y, 1)[0]
    return o


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


res = {"counts": {c: len(v) for c, v in D.items()}, "alpha": {}, "primary": {}}
for c in sorted(D):
    res["alpha"][c] = {}
    for T in (20, 30, 40, 50, 60):
        s = slopes(c, T)
        if s:
            res["alpha"][c][T] = dict(mean=float(np.mean(list(s.values()))), ci=ci(list(s.values())), n=len(s))
tests = {}
for h, (a, b) in {"H1": ("sf60_s3", "sf60_s0"), "H2": ("cv60_s3", "cv60_s0")}.items():
    sa, sb = slopes(a, 60), slopes(b, 60)
    d = np.array([sa[s] - sb[s] for s in sa if s in sb])
    tests[h] = dict(est=float(d.mean()), ci=ci(d), p=float(stats.ttest_1samp(d, 0).pvalue), n=int(len(d)),
                    mean_sink=float(np.mean([sa[s] for s in sa if s in sb])), mean_nosink=float(np.mean([sb[s] for s in sa if s in sb])))
running = 0.0
for i, k in enumerate(sorted(tests, key=lambda k: tests[k]["p"])):
    running = max(running, min(1.0, (len(tests) - i) * tests[k]["p"]))
    tests[k]["p_holm"] = running
    tests[k]["supported"] = bool(running < 0.05 and tests[k]["est"] > 0)
res["primary"] = tests
a3 = res["alpha"].get("sf60_s3", {}).get(60, {}).get("mean")
a0 = res["alpha"].get("sf60_s0", {}).get(60, {}).get("mean")
res["secondary_band"] = dict(sf60_s3_in_band=None if a3 is None else bool(0.25 <= a3 <= 0.54),
                             sf60_s0_near_zero=None if a0 is None else bool(abs(a0) < 0.15))
json.dump(res, open(f"{out}/long_results.json", "w"), indent=1)
print(json.dumps(res["primary"], indent=1))
print({c: {T: round(v["mean"], 3) for T, v in r.items()} for c, r in res["alpha"].items()}, res["secondary_band"])
