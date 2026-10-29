"""Study 32: is the smaller attractor offset of LongLive explained by its smaller inference window? Self Forcing DMD
with LongLive's window of 12 latent frames against LongLive (preregistered,
docs/preregistrations/32_window_matched_offset.md).

Files {prompt}_{arm}_s{seed}.npz (plain text-to-video, 16 fps, 60 s, seeds 10, 11, 12, the 40 prompts of
prompts/main.jsonl): sf12_s0, sf12_s3 in RUN_DIR (this study); ll_s0, ll_s3, sf_s0, sf_s3 in S30_DIR (study 30).
Metric as in study 30 (analyze_s30.py): start = mean L* of frames 1..8, end = mean L* over seconds [50, 60) after frame
9, offset = end - start; per prompt and arm the mean offset over its seeds (at least 2) and its magnitude.
Primary (Holm over 2; paired over prompts with both arms; two-sided one-sample t-tests of the difference of
magnitudes; supported if Holm p < .05 and the estimate > 0; 95% CIs by prompt bootstrap, 10,000 draws):
  W1 window 12, no sink:                 |off(sf12_s0)| - |off(ll_s0)| > 0
  W2 window 12, sink 3 (LongLive's own): |off(sf12_s3)| - |off(ll_s3)| > 0
Secondary: the window's own effect in Self Forcing, |off(sf12_s0)| - |off(sf_s0)| and |off(sf12_s3)| - |off(sf_s3)|
(two-sided, estimates and CIs); per arm the mean signed offset, mean magnitude, seed-unbiased root-mean-square offset and
mean between-seed SD; W1, W2 on the seed-unbiased squared offset, at 31 s (end over [29, 31) s) and per prompt set
(MovieGen, driving); 1-s curves of the mean offset; collapse counts (a 1-s mean L* below 3 or above 97).
usage: python analyze_s32.py RUN_DIR S30_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys
from itertools import combinations

import numpy as np
from scipy import stats

src, s30, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD, TMAX = 16, 9, 60
ARMS = ("sf12_s0", "sf12_s3", "ll_s0", "ll_s3", "sf_s0", "sf_s3")
SEEDS = (10, 11, 12)
pat = re.compile(r"^(\w+?)_(sf12_s0|sf12_s3|ll_s0|ll_s3|sf_s0|sf_s3)_s(\d+)\.npz$")

L = {}   # L[arm][prompt][seed] = per-frame mean L*
for d, arms in ((src, ("sf12_s0", "sf12_s3")), (s30, ("ll_s0", "ll_s3", "sf_s0", "sf_s3"))):
    for p in glob.glob(f"{d}/*.npz"):
        m = pat.match(os.path.basename(p))
        if m and m[2] in arms and int(m[3]) in SEEDS:
            x = np.load(p)["lab_mean"][:, 0]
            if len(x) >= HOLD + TMAX * FPS:
                L.setdefault(m[2], {}).setdefault(m[1], {})[int(m[3])] = x
DRV = {k for a in L.values() for k in a if k.startswith("drv_")}


def off(x, t0=50, t1=60):
    return float(x[HOLD + t0 * FPS:HOLD + t1 * FPS].mean() - x[1:9].mean())


def offsets(arm, t0=50, t1=60, prompts=None):
    """{prompt: [offset per seed]} for prompts with at least 2 seeds."""
    o = {p: [off(x, t0, t1) for x in S.values()] for p, S in L.get(arm, {}).items()}
    return {p: v for p, v in o.items() if len(v) >= 2 and (prompts is None or p in prompts)}


def mag(arm, **k):
    return {p: abs(float(np.mean(v))) for p, v in offsets(arm, **k).items()}


def sq(arm, **k):   # seed-unbiased squared offset: E[off_i off_j] = mu^2 for independent seeds
    return {p: float(np.mean([a * b for a, b in combinations(v, 2)])) for p, v in offsets(arm, **k).items()}


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def paired(x, y):
    k = sorted(set(x) & set(y))
    if len(k) < 3:
        return None
    d = np.array([x[p] - y[p] for p in k])
    return dict(est=float(d.mean()), ci=ci(d), p=float(stats.ttest_1samp(d, 0).pvalue), n=len(k),
                share_first_larger=float(np.mean(d > 0)))


PAIRS = {"W1": ("sf12_s0", "ll_s0"), "W2": ("sf12_s3", "ll_s3")}
WINDOW = {"sf12_s0 - sf_s0": ("sf12_s0", "sf_s0"), "sf12_s3 - sf_s3": ("sf12_s3", "sf_s3")}


def tests(f=mag, pairs=PAIRS, **k):
    return {t: paired(f(a, **k), f(b, **k)) for t, (a, b) in pairs.items()}


def holm(t):
    P = {k: dict(x) for k, x in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(t) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and P[k]["est"] > 0)
    return P


def per_arm(arm):
    o = offsets(arm)
    if not o:
        return None
    s, m, q = [float(np.mean(v)) for v in o.values()], list(mag(arm).values()), list(sq(arm).values())
    return dict(n_prompts=len(o), signed=dict(mean=float(np.mean(s)), ci=ci(s)), magnitude=dict(mean=float(np.mean(m)), ci=ci(m)),
                rms_unbiased=float(np.sqrt(max(0.0, np.mean(q)))), seed_sd=float(np.mean([np.std(v, ddof=1) for v in o.values()])),
                start=float(np.mean([x[1:9].mean() for S in L[arm].values() for x in S.values()])))


def curve(arm):
    c = {}
    for t in range(1, TMAX + 1):
        o = offsets(arm, t - 1, t)
        if o:
            c[t] = float(np.mean([np.mean(v) for v in o.values()]))
    return c


def collapse(arm):
    n = 0
    for S in L.get(arm, {}).values():
        for x in S.values():
            k = len(x[HOLD:]) // FPS
            m = x[HOLD:HOLD + k * FPS].reshape(k, FPS).mean(1)
            n += int(bool((m < 3).any() or (m > 97).any()))
    return n


allp = {k for a in L.values() for k in a}
res = {"counts": {a: {s: sum(s in S for S in L.get(a, {}).values()) for s in SEEDS} for a in ARMS},
       "primary": holm(tests()),
       "window_effect": tests(pairs=WINDOW),
       "per_arm": {a: per_arm(a) for a in ARMS},
       "unbiased_sq": tests(sq),
       "at_31s": tests(t0=29, t1=31),
       "moviegen": tests(prompts=allp - DRV),
       "driving": tests(prompts=DRV),
       "curves_1s": {a: curve(a) for a in ARMS},
       "collapse": {a: collapse(a) for a in ARMS}}
json.dump(res, open(f"{out}/s32_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, PAIRS[k], f"est={t['est']:+.2f} ci=[{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
for k, t in res["window_effect"].items():
    if t:
        print("window", k, f"est={t['est']:+.2f} ci=[{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}] p={t['p']:.3g}")
for a in ARMS:
    q = res["per_arm"][a]
    if q:
        print(a, f"signed={q['signed']['mean']:+.2f} |off|={q['magnitude']['mean']:.2f} rms={q['rms_unbiased']:.2f} "
                 f"seed_sd={q['seed_sd']:.2f} n={q['n_prompts']}")
