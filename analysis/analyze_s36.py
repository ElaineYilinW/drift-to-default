"""Study 36: does the attractor offset keep growing with the key-value window beyond the released window, and where in
18-21 does Self Forcing's step lie? (preregistered, docs/preregistrations/36_window_beyond.md).

Files {prompt}_{arm}_s{seed}.npz (plain text-to-video, 16 fps, 60 s, seeds 10, 11, 12, the 40 prompts of
prompts/main.jsonl, no sink): sf19_s0, sf20_s0, sf24_s0, sf27_s0, ll24_s0 in RUN_DIR (this study); sf06_s0, sf09_s0,
sf15_s0, sf18_s0, ll06_s0, ll21_s0 in S33_DIR (study 33); sf12_s0 in S32_DIR (study 32); sf_s0 (window 21), ll_s0
(window 12) in S30_DIR (study 30).
Metric as in studies 30, 32, 33: start = mean L* of frames 1..8, end = mean L* over seconds [50, 60) after frame 9,
offset = end - start; per prompt and arm the mean offset over its seeds (at least 2) and its magnitude |off|.
Primary (Holm over 3; per-prompt values; two-sided one-sample t-tests; supported if Holm p < .05 and the estimate > 0;
95% CIs by prompt bootstrap, 10,000 draws):
  X1 Self Forcing, beyond the released window:  |off(sf24_s0)| - |off(sf_s0)|  > 0
  X2 Self Forcing, further beyond:              |off(sf27_s0)| - |off(sf_s0)|  > 0
  X3 LongLive, beyond window 21:                |off(ll24_s0)| - |off(ll21_s0)| > 0
Secondary: per arm the mean signed offset, mean magnitude, seed-unbiased root-mean-square offset, mean between-seed SD,
sharpness ratio and mean absolute frame-to-frame change of mean L* over [50, 60) s (as study 33); Self Forcing's mean
|off| and signed offset per window (6, 9, 12, 15, 18, 19, 20, 21, 24, 27); the step: |off| differences between
neighbouring windows from 18 to 21 (19 - 18, 20 - 19, 21 - 20) and the per-prompt least-squares slope over 18-21
(18, 19, 20, 21) minus that over 6-18 (6, 9, 12, 15, 18); X1-X3 with a 90% t interval and two one-sided tests at
+-2 L*; X1-X3 on the seed-unbiased squared offset, at 31 s and per prompt set (MovieGen, driving); 1-s curves of the
mean signed offset and plateau times; collapse counts (a 1-s mean L* below 3 or above 97).
usage: python analyze_s36.py RUN_DIR S33_DIR S32_DIR S30_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys
from itertools import combinations

import numpy as np
from scipy import stats

src, s33, s32, s30, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD, TMAX = 16, 9, 60
SEEDS = (10, 11, 12)
SOURCES = ((src, ("sf19_s0", "sf20_s0", "sf24_s0", "sf27_s0", "ll24_s0")),
           (s33, ("sf06_s0", "sf09_s0", "sf15_s0", "sf18_s0", "ll06_s0", "ll21_s0")),
           (s32, ("sf12_s0",)), (s30, ("sf_s0", "ll_s0")))
ARMS = [a for _, arms in SOURCES for a in arms]
SF = {6: "sf06_s0", 9: "sf09_s0", 12: "sf12_s0", 15: "sf15_s0", 18: "sf18_s0", 19: "sf19_s0", 20: "sf20_s0",
      21: "sf_s0", 24: "sf24_s0", 27: "sf27_s0"}
LLW = {6: "ll06_s0", 12: "ll_s0", 21: "ll21_s0", 24: "ll24_s0"}
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


def tost(x, y, m=2.0):
    v = np.array([x[p] - y[p] for p in sorted(set(x) & set(y))], float)
    if len(v) < 3:
        return None
    n, mu, se = len(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.95, n - 1) * se
    p_lo, p_hi = float(stats.t.sf((mu + m) / se, n - 1)), float(stats.t.cdf((mu - m) / se, n - 1))
    return dict(est=float(mu), ci90=[float(mu - h), float(mu + h)], margin=m, p=max(p_lo, p_hi), n=int(n))


def tests(f=mag, **k):
    return {"X1": paired(f("sf24_s0", **k), f("sf_s0", **k)),
            "X2": paired(f("sf27_s0", **k), f("sf_s0", **k)),
            "X3": paired(f("ll24_s0", **k), f("ll21_s0", **k))}


def holm(t):
    P = {k: dict(x) for k, x in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(t) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and P[k]["est"] > 0)
    return P


def slopes(windows, f=mag):
    per = {w: f(SF[w]) for w in windows}
    ps = sorted(set.intersection(*[set(v) for v in per.values()]))
    W = np.array(sorted(windows))
    return {p: float(np.polyfit(W, [per[w][p] for w in W], 1)[0]) for p in ps}


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


curves = {a: curve(a) for a in ARMS}
s_hi, s_lo = slopes((18, 19, 20, 21)), slopes((6, 9, 12, 15, 18))
res = {"counts": {a: {s: sum(s in S for S in L.get(a, {}).values()) for s in SEEDS} for a in ARMS},
       "primary": holm(tests()),
       "per_arm": {a: per_arm(a) for a in ARMS},
       "sf_by_window": {w: dict(magnitude=float(np.mean(list(mag(a).values()))), signed=float(np.mean(list(signed(a).values()))))
                        for w, a in SF.items() if a in L},
       "ll_by_window": {w: dict(magnitude=float(np.mean(list(mag(a).values()))), signed=float(np.mean(list(signed(a).values()))))
                        for w, a in LLW.items() if a in L},
       "step": {"19 - 18": paired(mag("sf19_s0"), mag("sf18_s0")), "20 - 19": paired(mag("sf20_s0"), mag("sf19_s0")),
                "21 - 20": paired(mag("sf_s0"), mag("sf20_s0")),
                "slope 18-21 minus slope 6-18": onesample({p: s_hi[p] - s_lo[p] for p in set(s_hi) & set(s_lo)}),
                "slope 18-21": onesample(s_hi), "slope 6-18": onesample(s_lo)},
       "equivalence_2Lstar": {"X1": tost(mag("sf24_s0"), mag("sf_s0")), "X2": tost(mag("sf27_s0"), mag("sf_s0")),
                              "X3": tost(mag("ll24_s0"), mag("ll21_s0"))},
       "unbiased_sq": tests(sq),
       "at_31s": tests(t0=29, t1=31),
       "moviegen": tests(prompts=ALLP - DRV),
       "driving": tests(prompts=DRV),
       "curves_1s": curves,
       "plateau_s": {a: plateau(c) for a, c in curves.items()},
       "collapse": {a: sorted(collapsed(a)) for a in ARMS}}
json.dump(res, open(f"{out}/s36_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
for w, q in res["sf_by_window"].items():
    print(f"SF window {w:2d}: |off| {q['magnitude']:.2f} signed {q['signed']:+.2f}")
for w, q in res["ll_by_window"].items():
    print(f"LL window {w:2d}: |off| {q['magnitude']:.2f} signed {q['signed']:+.2f}")
print("step:", {k: (round(v["est"], 2), [round(c, 2) for c in v["ci"]]) for k, v in res["step"].items() if v})
