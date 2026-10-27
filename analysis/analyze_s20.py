"""Study 20: new seeds for the Self Forcing sink results (preregistered, docs/preregistrations/20_seed_replication.md).

Files {scene}_{arm}__{m16|p16}_s{1|2|3}.npz (16 fps; frames 0..8 are the held opening), arm in sf60_s3, sf60_s0,
sf60_s3rr. For one scene, seed and arm, alpha(T) = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L* of
frames 1..8, end = mean L* over seconds [T - 10, T) after the opening. A scene's value is the mean over its seeds.
Primary (Holm over 3; per-scene values, two-sided one-sample t-tests, scene-bootstrap 95% CIs (10,000 draws); supported
if Holm p < .05 and the estimate is on the predicted side; T = 60 s):
  H1 a sink keeps a share:                       alpha(sf60_s3) - alpha(sf60_s0) > 0
  H2 re-rotation lowers it:                      alpha(sf60_s3rr) - alpha(sf60_s3) < 0
  H3 a re-rotated sink keeps more than no sink:  alpha(sf60_s3rr) - alpha(sf60_s0) > 0
Secondary: H1-H3 over [40, 50) s; alpha(T) at T = 10..60 per arm; each seed's H1-H3 estimate; variance of alpha at
60 s per arm split into between-scene and within-scene (between-seed) parts (one-way random-effects ICC); the jump
alpha([34, 35)) - alpha([31, 32)) (1-s windows) per arm; seed 0 of the same scenes and arms from studies 10 and 16 for
comparison (descriptive, when SEED0_DIRS are given).
usage: python analyze_s20.py RUN_DIR OUT_DIR [SEED0_DIR ...]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, out, seed0_dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD = 16, 9
ARMS = ("sf60_s3", "sf60_s0", "sf60_s3rr")
pat = re.compile(r"^(\w+?)_(sf60_s3|sf60_s0|sf60_s3rr)__(m16|p16)_s(\d+)\.npz$")


def load(dirs, seeds):
    D = {}
    for d in dirs:
        for p in glob.glob(f"{d}/*.npz"):
            m = pat.match(os.path.basename(p))
            if m and int(m[4]) in seeds:
                D.setdefault(m[2], {}).setdefault(m[1], {}).setdefault(int(m[4]), {})[m[3]] = np.load(p)["lab_mean"][:, 0]
    return D


def a_one(v, t0, t1):
    i0, i1 = HOLD + int(t0 * FPS), HOLD + int(t1 * FPS)
    if len(v) < 2 or i1 > min(len(v["m16"]), len(v["p16"])):
        return None
    return float((v["p16"][i0:i1].mean() - v["m16"][i0:i1].mean()) / (v["p16"][1:HOLD].mean() - v["m16"][1:HOLD].mean()))


def alpha(D, arm, t0, t1, seed=None):
    """{scene: alpha averaged over the scene's seeds (or of one seed)}"""
    o = {}
    for s, by in D.get(arm, {}).items():
        v = [a_one(x, t0, t1) for k, x in by.items() if seed is None or k == seed]
        v = [x for x in v if x is not None]
        if v:
            o[s] = float(np.mean(v))
    return o


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def ttest(d):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def pdiff(x, y):
    return {s: x[s] - y[s] for s in x if s in y}


def tests(D, t0, t1, seed=None):
    A = {a: alpha(D, a, t0, t1, seed) for a in ARMS}
    return {"H1": (ttest(pdiff(A["sf60_s3"], A["sf60_s0"])), +1),
            "H2": (ttest(pdiff(A["sf60_s3rr"], A["sf60_s3"])), -1),
            "H3": (ttest(pdiff(A["sf60_s3rr"], A["sf60_s0"])), +1)}


def holm(t):
    prim = {k: dict(x, side=s) for k, (x, s) in t.items() if x is not None}
    running = 0.0
    pv = {k: (prim[k]["p"] if np.isfinite(prim[k]["p"]) else 1.0) for k in prim}
    for i, k in enumerate(sorted(prim, key=pv.get)):
        running = max(running, min(1.0, (3 - i) * pv[k]))
        prim[k]["p_holm"] = running
        prim[k]["supported"] = bool(running < 0.05 and np.sign(prim[k]["est"]) == prim[k]["side"])
    return prim


def icc(D, arm, t0=50, t1=60):
    rows = [[a_one(x, t0, t1) for x in by.values()] for by in D.get(arm, {}).values()]
    rows = [[v for v in r if v is not None] for r in rows]
    rows = [r for r in rows if len(r) >= 2]
    if len(rows) < 3:
        return None
    k = min(len(r) for r in rows)
    X = np.array([r[:k] for r in rows])
    n = len(X)
    msb = k * X.mean(1).var(ddof=1)
    msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return dict(icc=float((msb - msw) / (msb + (k - 1) * msw)), sd_between_seeds=float(np.sqrt(msw)),
                sd_scene_means=float(X.mean(1).std(ddof=1)), n_scenes=n, seeds=k)


D = load([src], {1, 2, 3})
seeds = sorted({k for a in D.values() for by in a.values() for k in by})
res = {"counts": {a: {s: len(by) for s, by in D.get(a, {}).items()} for a in ARMS}, "seeds": seeds,
       "primary": holm(tests(D, 50, 60)),
       "window_40_50": {k: x for k, (x, _) in tests(D, 40, 50).items()},
       "per_seed": {sd: {k: x for k, (x, _) in tests(D, 50, 60, sd).items()} for sd in seeds},
       "alpha": {a: {T: dict(mean=float(np.mean(list(x.values()))), ci=ci(list(x.values())), n=len(x))
                     for T in (10, 20, 30, 40, 50, 60) if (x := alpha(D, a, T - 10, T))} for a in ARMS},
       "icc_60": {a: icc(D, a) for a in ARMS},
       "jump_31_34": {a: ttest(pdiff(alpha(D, a, 34, 35), alpha(D, a, 31, 32))) for a in ARMS}}
if seed0_dirs:
    D0 = load(seed0_dirs, {0})
    res["seed0"] = {a: (dict(mean=float(np.mean(list(x.values()))), n=len(x)) if (x := alpha(D0, a, 50, 60)) else None)
                    for a in ARMS}
json.dump(res, open(f"{out}/s20_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
print("alpha60:", {a: round(res["alpha"][a][60]["mean"], 3) for a in ARMS if 60 in res["alpha"].get(a, {})})
