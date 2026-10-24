"""Prompt generality (preregistered, docs/preregistration_pgen_cn.md).

24 openings (4 non-driving domain prompts x 6 Self Forcing-generated openings), each continued under its own prompt
with the opening's L* shifted by -16 / 0 / +16. Files {opening}_{cfg}__{m16|z0|p16}_s0.npz, cfg in sf60_s3, sf60_s0
(60 s), ll_s3, ll_s0 (31 s). alpha(T) = within-opening OLS slope of the mean L* over seconds [T-10, T) after the
held opening on the opening L* (frames 1..8). T = 60 s for Self Forcing, 30 s for LongLive.
Primary (Holm over 3):
  H1 alpha(sf60_s3) - alpha(sf60_s0) > 0 at 60 s   (two-sided one-sample t-test over openings; Holm p < .05, est > 0)
  H2 alpha(ll_s3) - alpha(ll_s0) > 0 at 30 s        (same)
  H3 persistence differs between prompts: one-way ANOVA of per-opening alpha(sf60_s3, 60 s) over the 4 domains
     (Holm p < .05; no direction)
Secondary: alpha per domain and configuration at 10..60 s; zero-drift level b0 / (1 - b1) per domain (median);
H3 repeated for ll_s3 at 30 s. 95% CIs: opening bootstrap within domain.
usage: python analyze_pgen.py RUN_DIR OUT_DIR"""
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
pat = re.compile(r"^pg([a-z]+)(\d)_(\w+?)__(m16|z0|p16)_s0\.npz$")
D = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        D.setdefault(m[3], {}).setdefault(f"{m[1]}{m[2]}", {})[m[4]] = np.load(p)["lab_mean"][:, 0]
DOM = sorted({o[:-1] for c in D.values() for o in c})


def fits(cfg, T):
    o = {}
    for s, v in D.get(cfg, {}).items():
        if len(v) < 3 or len(v["z0"]) < 9 + T * 16:
            continue
        x = np.array([v[k][1:9].mean() for k in ("m16", "z0", "p16")])
        y = np.array([v[k][9 + (T - 10) * 16:9 + T * 16].mean() for k in ("m16", "z0", "p16")])
        o[s] = np.polyfit(x, y, 1)
    return o


def ci(vals, keys):
    by = {d: [vals[k] for k in keys if k[:-1] == d] for d in DOM}
    bs = [np.mean([x for d in DOM if by[d] for x in rng.choice(by[d], len(by[d]))]) for _ in range(10000)]
    return np.percentile(bs, [2.5, 97.5]).tolist()


def diff(a, b, T):
    x, y = {k: f[0] for k, f in fits(a, T).items()}, {k: f[0] for k, f in fits(b, T).items()}
    ks = [k for k in x if k in y]
    d = {k: x[k] - y[k] for k in ks}
    v = np.array(list(d.values()))
    return dict(est=float(v.mean()), ci=ci(d, ks), p=float(stats.ttest_1samp(v, 0).pvalue), n=len(v),
                mean_a=float(np.mean([x[k] for k in ks])), mean_b=float(np.mean([y[k] for k in ks])))


def anova(cfg, T):
    a = {k: f[0] for k, f in fits(cfg, T).items()}
    groups = [[a[k] for k in a if k[:-1] == d] for d in DOM]
    F, p = stats.f_oneway(*[g for g in groups if len(g) > 1])
    return dict(F=float(F), p=float(p), means={d: float(np.mean(g)) for d, g in zip(DOM, groups) if g})


tests = {"H1": diff("sf60_s3", "sf60_s0", 60), "H2": diff("ll_s3", "ll_s0", 30), "H3": anova("sf60_s3", 60)}
running = 0.0
for i, k in enumerate(sorted(tests, key=lambda k: tests[k]["p"])):
    running = max(running, min(1.0, (len(tests) - i) * tests[k]["p"]))
    tests[k]["p_holm"] = running
    tests[k]["supported"] = bool(running < 0.05 and (k == "H3" or tests[k]["est"] > 0))
res = {"counts": {c: len(v) for c, v in D.items()}, "primary": tests, "H3_ll": anova("ll_s3", 30), "alpha": {}, "zero": {}}
for c in D:
    for T in ((10, 20, 30, 40, 50, 60) if c.startswith("sf60") else (10, 20, 30)):
        f = fits(c, T)
        if f:
            a = {k: v[0] for k, v in f.items()}
            res["alpha"].setdefault(c, {})[T] = dict(mean=float(np.mean(list(a.values()))), ci=ci(a, list(a)),
                                                    per_domain={d: float(np.mean([a[k] for k in a if k[:-1] == d])) for d in DOM})
    f = fits(c, 60 if c.startswith("sf60") else 30)
    res["zero"][c] = {d: float(np.median([b0 / (1 - b1) for k, (b1, b0) in f.items() if k[:-1] == d and b1 < 0.95]))
                      for d in DOM if any(k[:-1] == d and b1 < 0.95 for k, (b1, b0) in f.items())}
json.dump(res, open(f"{out}/pgen_results.json", "w"), indent=1)
print(json.dumps(res["primary"], indent=1))
