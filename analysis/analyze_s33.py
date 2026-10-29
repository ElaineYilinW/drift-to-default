"""Study 33: does the attractor offset grow with the key-value window, in Self Forcing, LongLive and CausVid?
(preregistered, docs/preregistrations/33_window_dose.md).

Files {prompt}_{arm}_s{seed}.npz (plain text-to-video, 16 fps, 60 s, seeds 10, 11, 12, the 40 prompts of
prompts/main.jsonl, no sink): sf06_s0, sf09_s0, sf15_s0, sf18_s0, ll06_s0, ll21_s0, cv12_s0 in RUN_DIR (this study);
sf12_s0 in S32_DIR (study 32); sf_s0 (window 21), ll_s0 (window 12), cv_s0 (window 21) in S30_DIR (study 30).
Metric as in studies 30 and 32: start = mean L* of frames 1..8, end = mean L* over seconds [50, 60) after frame 9,
offset = end - start; per prompt and arm the mean offset over its seeds (at least 2) and its magnitude |off|.
Primary (Holm over 3; per-prompt values; two-sided one-sample t-tests; supported if Holm p < .05 and the estimate > 0;
95% CIs by prompt bootstrap, 10,000 draws):
  V1 Self Forcing: the per-prompt least-squares slope of |off| on the window (6, 9, 12, 15, 18, 21; prompts with all six)
     is > 0
  V2 LongLive:     |off(ll21_s0)| - |off(ll_s0)| > 0
  V3 CausVid:      |off(cv_s0)| - |off(cv12_s0)| > 0
Secondary: per arm the mean signed offset, mean magnitude, seed-unbiased root-mean-square offset and mean between-seed
SD; Self Forcing's mean |off| and signed offset per window, the per-prompt Spearman correlation of |off| with the window
(mean, CI) and the slope of the signed offset; |off(ll06_s0)| - |off(ll_s0)| and |off(sf06_s0)| - |off(sf12_s0)|; V1-V3
on the seed-unbiased squared offset, at 31 s (end over [29, 31) s) and per prompt set (MovieGen, driving); 1-s curves of
the mean signed offset per arm and the plateau time (first T with |mean offset over [T - 5, T)| >= 0.9 |mean offset over
[50, 60)|); per arm the sharpness ratio (mean sharpness over [50, 60) s / over [0, 10) s) and the mean absolute
frame-to-frame change of mean L* over [50, 60) s; collapse counts (a 1-s mean L* below 3 or above 97) and V1 without
prompts that have a collapsed Self Forcing video.
usage: python analyze_s33.py RUN_DIR S32_DIR S30_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys
from itertools import combinations

import numpy as np
from scipy import stats

src, s32, s30, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD, TMAX = 16, 9, 60
SEEDS = (10, 11, 12)
SOURCES = ((src, ("sf06_s0", "sf09_s0", "sf15_s0", "sf18_s0", "ll06_s0", "ll21_s0", "cv12_s0")),
           (s32, ("sf12_s0",)), (s30, ("sf_s0", "ll_s0", "cv_s0")))
ARMS = [a for _, arms in SOURCES for a in arms]
SF = {6: "sf06_s0", 9: "sf09_s0", 12: "sf12_s0", 15: "sf15_s0", 18: "sf18_s0", 21: "sf_s0"}
LLW = {6: "ll06_s0", 12: "ll_s0", 21: "ll21_s0"}
CVW = {12: "cv12_s0", 21: "cv_s0"}
pat = re.compile(r"^(\w+?)_(" + "|".join(ARMS) + r")_s(\d+)\.npz$")

L, SH = {}, {}   # [arm][prompt][seed] = per-frame mean L*, sharpness
for d, arms in SOURCES:
    for p in glob.glob(f"{d}/*.npz"):
        m = pat.match(os.path.basename(p))
        if m and m[2] in arms and int(m[3]) in SEEDS:
            z = np.load(p)
            x = z["lab_mean"][:, 0]
            if len(x) >= HOLD + TMAX * FPS:
                L.setdefault(m[2], {}).setdefault(m[1], {})[int(m[3])] = x
                SH.setdefault(m[2], {}).setdefault(m[1], {})[int(m[3])] = z["sharpness"]
DRV = {k for a in L.values() for k in a if k.startswith("drv_")}
ALLP = {k for a in L.values() for k in a}


def off(x, t0=50, t1=60):
    return float(x[HOLD + t0 * FPS:HOLD + t1 * FPS].mean() - x[1:9].mean())


def offsets(arm, t0=50, t1=60, prompts=None):
    """{prompt: [offset per seed]} for prompts with at least 2 seeds."""
    o = {p: [off(x, t0, t1) for x in S.values()] for p, S in L.get(arm, {}).items()}
    return {p: v for p, v in o.items() if len(v) >= 2 and (prompts is None or p in prompts)}


def mag(arm, **k):
    return {p: abs(float(np.mean(v))) for p, v in offsets(arm, **k).items()}


def signed(arm, **k):
    return {p: float(np.mean(v)) for p, v in offsets(arm, **k).items()}


def sq(arm, **k):   # seed-unbiased squared offset: E[off_i off_j] = mu^2 for independent seeds
    return {p: float(np.mean([a * b for a, b in combinations(v, 2)])) for p, v in offsets(arm, **k).items()}


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def onesample(d):
    v = np.array(list(d.values()), float)
    if len(v) < 3:
        return None
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=len(v),
                share_positive=float(np.mean(v > 0)))


def paired(x, y):
    return onesample({p: x[p] - y[p] for p in sorted(set(x) & set(y))})


def slopes(f=mag, exclude=(), **k):
    per = {w: f(a, **k) for w, a in SF.items()}
    ps = sorted(set.intersection(*[set(v) for v in per.values()]) - set(exclude))
    W = np.array(sorted(SF))
    return {p: float(np.polyfit(W, [per[w][p] for w in W], 1)[0]) for p in ps}


def tests(f=mag, exclude=(), **k):
    return {"V1": onesample(slopes(f, exclude, **k)),
            "V2": paired(f("ll21_s0", **k), f("ll_s0", **k)),
            "V3": paired(f("cv_s0", **k), f("cv12_s0", **k))}


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
    s, m, q = list(signed(arm).values()), list(mag(arm).values()), list(sq(arm).values())
    shr = [np.mean([y[HOLD + 50 * FPS:HOLD + 60 * FPS].mean() / y[HOLD:HOLD + 10 * FPS].mean() for y in S.values()])
           for S in SH[arm].values()]
    dl = [np.mean([np.abs(np.diff(x[HOLD + 50 * FPS:HOLD + 60 * FPS])).mean() for x in S.values()]) for S in L[arm].values()]
    return dict(n_prompts=len(o), signed=dict(mean=float(np.mean(s)), ci=ci(s)), magnitude=dict(mean=float(np.mean(m)), ci=ci(m)),
                rms_unbiased=float(np.sqrt(max(0.0, np.mean(q)))), seed_sd=float(np.mean([np.std(v, ddof=1) for v in o.values()])),
                sharpness_ratio=float(np.mean(shr)), dL_frame=float(np.mean(dl)))


def curve(arm):
    c = {}
    for t in range(1, TMAX + 1):
        o = offsets(arm, t - 1, t)
        if o:
            c[t] = float(np.mean([np.mean(v) for v in o.values()]))
    return c


def plateau(c):
    if not c:
        return None
    end = abs(np.mean([c[t] for t in range(51, 61)]))
    for T in range(5, TMAX + 1):
        if abs(np.mean([c[t] for t in range(T - 4, T + 1)])) >= 0.9 * end:
            return T
    return None


def collapsed(arm):
    bad = set()
    for p, S in L.get(arm, {}).items():
        for x in S.values():
            k = len(x[HOLD:]) // FPS
            m = x[HOLD:HOLD + k * FPS].reshape(k, FPS).mean(1)
            if (m < 3).any() or (m > 97).any():
                bad.add(p)
    return bad


def spearman_sf():
    per = {w: mag(a) for w, a in SF.items()}
    ps = sorted(set.intersection(*[set(v) for v in per.values()]))
    W = sorted(SF)
    r = [stats.spearmanr(W, [per[w][p] for w in W])[0] for p in ps]
    r = [x for x in r if np.isfinite(x)]
    return dict(mean=float(np.mean(r)), ci=ci(r), n=len(r))


curves = {a: curve(a) for a in ARMS}
sf_bad = set().union(*[collapsed(a) for a in SF.values()])
res = {"counts": {a: {s: sum(s in S for S in L.get(a, {}).values()) for s in SEEDS} for a in ARMS},
       "primary": holm(tests()),
       "per_arm": {a: per_arm(a) for a in ARMS},
       "sf_by_window": {w: dict(magnitude=float(np.mean(list(mag(a).values()))), signed=float(np.mean(list(signed(a).values()))))
                        for w, a in SF.items() if a in L},
       "sf_spearman": spearman_sf(),
       "sf_signed_slope": onesample(slopes(signed)),
       "window6": {"ll06 - ll12": paired(mag("ll06_s0"), mag("ll_s0")), "sf06 - sf12": paired(mag("sf06_s0"), mag("sf12_s0"))},
       "unbiased_sq": tests(sq),
       "at_31s": tests(t0=29, t1=31),
       "moviegen": tests(prompts=ALLP - DRV),
       "driving": tests(prompts=DRV),
       "curves_1s": curves,
       "plateau_s": {a: plateau(c) for a, c in curves.items()},
       "collapse": {a: sorted(collapsed(a)) for a in ARMS},
       "V1_without_collapsed": onesample(slopes(exclude=sf_bad))}
json.dump(res, open(f"{out}/s33_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
for w, q in res["sf_by_window"].items():
    print(f"SF window {w:2d}: |off| {q['magnitude']:.2f} signed {q['signed']:+.2f}")
for a in ARMS:
    q = res["per_arm"][a]
    if q:
        print(a, f"signed={q['signed']['mean']:+.2f} |off|={q['magnitude']['mean']:.2f} rms={q['rms_unbiased']:.2f} "
                 f"seed_sd={q['seed_sd']:.2f} sharp={q['sharpness_ratio']:.2f} dL={q['dL_frame']:.3f} n={q['n_prompts']} "
                 f"plateau={res['plateau_s'][a]}")
