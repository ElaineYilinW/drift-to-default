"""60-s persistence atlas (preregistered, docs/preregistration_atlas_cn.md).

Files {scene}_{cfg}__{m16|z0|p16}_s0.npz (16 fps; frames 0..8 are the held opening). Persistence alpha(T) = within-scene
OLS slope of the mean L* over seconds [T-10, T) after the opening on the opening's L* (mean of frames 1..8), over the
available versions (two or three). Zero-drift level (three-version configurations): the end level at which the fitted
line meets end = opening, b0 / (1 - b1), median over scenes.
Primary (Holm over 4; per-scene differences, two-sided one-sample t-tests, scene-bootstrap 95% CIs):
  H1 a deeper sink keeps more:          alpha(sf60_s6) - alpha(sf60_s3) > 0      at 60 s
  H2 any sink keeps some:               alpha(sf60_s1) - alpha(sf60_s0) > 0      at 60 s
  H3 the LongLive sink keeps at 60 s:   alpha(ll60_s3) - alpha(ll60_s0) > 0      at 60 s
  H4 sink position matters:             alpha(sf60_s3rr) - alpha(sf60_s3) != 0   at 60 s
H1-H3 are supported if Holm p < .05 and the estimate > 0; H4 if Holm p < .05 (either sign, reported).
Secondary: alpha(sf60_s1rr) - alpha(sf60_s1); alpha at 10..60 s for every configuration; Spearman correlation between
alpha at 60 s and alpha at 31 s of preregistration 06 (same configuration); the change alpha(34.5 s) - alpha(31.5 s)
(1-s windows) for each Self Forcing DMD arm; share of the opening lost by 4.5 s, 1 - alpha(4.5 s) (1-s window), against
the total loss by 60 s; zero-drift level at 60 s per configuration.
usage: python analyze_atlas.py RUN_DIR RELAX_RESULTS_JSON OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, relax_json, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
pat = re.compile(r"^(\w+?)_(\w+?)__(m16|z0|p16)_s0\.npz$")
D = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        D.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]
REL = {"sf60_s3": "sf_s3", "sf60_s0": "sf_s0", "sf60_s1": "sf_s1", "ll60_s3": "ll_s3", "ll60_s0": "ll_s0",
       "gan60_s3": "gan_s3", "sid60_s3": "sid_s3", "sidv260_s3": "sidv2_s3", "sf10s60_s3": "sf10s_s3", "rf60": "rf"}


def fits(cfg, t0, t1):
    """per scene: (slope, intercept) of end (mean L* over seconds [t0, t1) after the opening) on opening L*."""
    o = {}
    for s, v in D.get(cfg, {}).items():
        ks = [k for k in ("m16", "z0", "p16") if k in v]
        if len(ks) < 2 or "m16" not in ks or "p16" not in ks or any(len(v[k]) < 9 + int(t1 * 16) for k in ks):
            continue
        x = np.array([v[k][1:9].mean() for k in ks])
        y = np.array([v[k][9 + int(t0 * 16):9 + int(t1 * 16)].mean() for k in ks])
        o[s] = np.polyfit(x, y, 1)
    return o


def alpha(cfg, T, w=10):
    return {s: f[0] for s, f in fits(cfg, T - w, T).items()}


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def diff(a, b, T=60, w=10):
    x, y = alpha(a, T, w), alpha(b, T, w)
    d = np.array([x[s] - y[s] for s in x if s in y])
    if len(d) < 3:
        return None
    return dict(est=float(d.mean()), ci=ci(d), p=float(stats.ttest_1samp(d, 0).pvalue), n=int(len(d)),
                mean_a=float(np.mean([x[s] for s in x if s in y])), mean_b=float(np.mean([y[s] for s in x if s in y])))


res = {"counts": {c: len(v) for c, v in D.items()}, "alpha": {}, "zero_drift_60s": {}}
for c in sorted(D):
    res["alpha"][c] = {T: dict(mean=float(np.mean(list(a.values()))), ci=ci(list(a.values())), n=len(a))
                       for T in (10, 20, 30, 40, 50, 60) if (a := alpha(c, T))}
    f = fits(c, 50, 60)
    zd = [b0 / (1 - b1) for b1, b0 in f.values() if b1 < 0.95] if f and all(len(D[c][s]) == 3 for s in f) else []
    if zd:
        res["zero_drift_60s"][c] = dict(median=float(np.median(zd)), n=len(zd))
tests = {"H1": diff("sf60_s6", "sf60_s3"), "H2": diff("sf60_s1", "sf60_s0"), "H3": diff("ll60_s3", "ll60_s0"),
         "H4": diff("sf60_s3rr", "sf60_s3")}
tests = {k: v for k, v in tests.items() if v}
running = 0.0
for i, k in enumerate(sorted(tests, key=lambda k: tests[k]["p"])):
    running = max(running, min(1.0, (len(tests) - i) * tests[k]["p"]))
    tests[k]["p_holm"] = running
    tests[k]["supported"] = bool(running < 0.05 and (k == "H4" or tests[k]["est"] > 0))
res["primary"] = tests
res["s1_rerotation"] = diff("sf60_s1rr", "sf60_s1")
rel = json.load(open(relax_json))["per_cfg"]
pairs = [(res["alpha"][c][60]["mean"], rel[r]["alpha_L"]["mean"]) for c, r in REL.items()
         if c in res["alpha"] and 60 in res["alpha"][c] and r in rel and rel[r].get("alpha_L")]
if len(pairs) >= 4:
    rho, p = stats.spearmanr(*zip(*pairs))
    res["spearman_60s_vs_31s"] = dict(rho=float(rho), p=float(p), n=len(pairs), pairs=pairs)
res["jump_31_34"] = {}
for c in ("sf60_s3", "sf60_s6", "sf60_s3rr", "sf60_s0", "sf60_s1", "sf60_s1rr"):
    a1, a2 = alpha(c, 32, 1), alpha(c, 35, 1)
    d = np.array([a2[s] - a1[s] for s in a1 if s in a2])
    if len(d) >= 3:
        res["jump_31_34"][c] = dict(est=float(d.mean()), ci=ci(d), n=int(len(d)))
res["early_loss"] = {}
for c in sorted(D):
    e, l = alpha(c, 5, 1), alpha(c, 60)
    ss = [s for s in e if s in l]
    if ss:
        res["early_loss"][c] = dict(loss_4p5s=float(np.mean([1 - e[s] for s in ss])),
                                    loss_60s=float(np.mean([1 - l[s] for s in ss])), n=len(ss))
res["curves"] = {c: [(T, float(np.mean(list(a.values())))) for T in range(2, 61) if (a := alpha(c, T, 1))] for c in D}
json.dump(res, open(f"{out}/atlas_results.json", "w"), indent=1)
print(json.dumps(res["primary"], indent=1))
print({c: round(v[60]["mean"], 3) for c, v in res["alpha"].items() if 60 in v})
