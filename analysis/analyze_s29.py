"""Study 29: does an opening that stays in the context keep a steady share from the second to the fourth minute?
(preregistered, docs/preregistrations/29_steady_share.md).

Files s{1,2}/{scene}_{arm}__{m16|p16}_s{seed}.npz in RUN_DIR (16 fps; frames 0..8 are the held opening), arm in
sf240_s3rr, rf240, sf240_s0. alpha[t0, t1) = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L* of frames
1..8, end = mean L* over seconds [t0, t1) after the opening; undefined if open_p16 - open_m16 < 4. No collapse
exclusion. A scene's value is the mean over the seeds for which it is defined.
Primary (Holm over 5; per-scene values; t-tests two-sided, supported if Holm p < .05 and the estimate is on the
predicted side; equivalence by two one-sided t-tests against the margin (TOST), p = the larger one-sided p, supported
if Holm p < .05; 95% CIs by scene bootstrap, 10,000 draws; 90% t intervals for TOST):
  M1 a re-rotated sink keeps a share in the fourth minute:  alpha(sf240_s3rr, [180, 240)) > 0
  M2 Rolling Forcing keeps a share in the fourth minute:    alpha(rf240, [180, 240)) > 0
  M3 the re-rotated sink's share is steady:  alpha(sf240_s3rr, [180, 240)) - alpha(sf240_s3rr, [60, 120)) within +-0.10
  M4 Rolling Forcing's share is steady:      alpha(rf240, [180, 240)) - alpha(rf240, [60, 120)) within +-0.10
  M5 without a sink the opening stays forgotten:  alpha(sf240_s0, [60, 240)) within +-0.10
Secondary: M1-M5 per seed; the 60-s windows [60, 120), [120, 180), [180, 240) per arm; per-scene least-squares slope of
the 10-s alpha from 70 to 240 s (per 100 s); exponential fit alpha(t) = a_inf + (a_0 - a_inf) exp(-t / tau) to each
arm's mean 10-s curve (T >= 10) with a scene-bootstrap CI of a_inf (2,000 draws); alpha(T) for T = 10..240 (10-s
windows) and 1-s curves; study 24's five tests on these seeds at 180 s (10-s windows: [170, 180) > 0 for sf240_s3rr and
rf240, [170, 180) - [50, 60) within +-0.10, sf240_s0 [170, 180) within +-0.10); seed agreement, the Pearson correlation
across scenes of alpha[60, 240) between seeds 1 and 2; collapse counts (a 1-s mean L* below 3 or above 97).
usage: python analyze_s29.py RUN_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import optimize, stats

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD, TMAX = 16, 9, 240
ARMS = ("sf240_s3rr", "rf240", "sf240_s0")
SEEDS = (1, 2)
pat = re.compile(r"^(\w+?)_(sf240_s3rr|rf240|sf240_s0)__(m16|p16)_s(\d+)\.npz$")

L = {}   # L[arm][seed][scene][version] = per-frame mean L*
for p in glob.glob(f"{src}/s*/*.npz"):
    m = pat.match(os.path.basename(p))
    if m and int(m[4]) in SEEDS:
        L.setdefault(m[2], {}).setdefault(int(m[4]), {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]


def alpha_seed(arm, seed, t0, t1):
    o = {}
    for s, x in L.get(arm, {}).get(seed, {}).items():
        if "m16" in x and "p16" in x:
            d = float(x["p16"][1:HOLD].mean() - x["m16"][1:HOLD].mean())
            i0, i1 = HOLD + int(t0 * FPS), HOLD + int(t1 * FPS)
            if d >= 4 and i1 <= len(x["p16"]) and i1 <= len(x["m16"]):
                o[s] = float(x["p16"][i0:i1].mean() - x["m16"][i0:i1].mean()) / d
    return o


def alpha(arm, t0, t1, seeds=SEEDS):
    per = [alpha_seed(arm, k, t0, t1) for k in seeds]
    scenes = set().union(*per)
    return {s: float(np.mean([p[s] for p in per if s in p])) for s in scenes}


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def ttest(d):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def tost(d, m):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    n, mu, se = len(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    p_lo, p_hi = float(stats.t.sf((mu + m) / se, n - 1)), float(stats.t.cdf((mu - m) / se, n - 1))
    h = stats.t.ppf(0.95, n - 1) * se
    return dict(est=float(mu), ci=ci(v), ci90=[float(mu - h), float(mu + h)], margin=m, p_lower=p_lo, p_upper=p_hi,
                p=max(p_lo, p_hi), n=int(n))


def pdiff(x, y):
    return {s: x[s] - y[s] for s in x if s in y}


def tests(seeds=SEEDS):
    A = lambda arm, a, b: alpha(arm, a, b, seeds)
    return {"M1": (ttest(A("sf240_s3rr", 180, 240)), +1),
            "M2": (ttest(A("rf240", 180, 240)), +1),
            "M3": (tost(pdiff(A("sf240_s3rr", 180, 240), A("sf240_s3rr", 60, 120)), 0.10), 0),
            "M4": (tost(pdiff(A("rf240", 180, 240), A("rf240", 60, 120)), 0.10), 0),
            "M5": (tost(A("sf240_s0", 60, 240), 0.10), 0)}


def s24_tests(seeds=SEEDS):
    A = lambda arm, a, b: alpha(arm, a, b, seeds)
    return {"L1": ttest(A("sf240_s3rr", 170, 180)), "L2": ttest(A("rf240", 170, 180)),
            "L3": tost(pdiff(A("sf240_s3rr", 170, 180), A("sf240_s3rr", 50, 60)), 0.10),
            "L4": tost(pdiff(A("rf240", 170, 180), A("rf240", 50, 60)), 0.10),
            "L5": tost(A("sf240_s0", 170, 180), 0.10)}


def holm(t):
    P = {k: dict(x, side=s) for k, (x, s) in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(t) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and (P[k]["side"] == 0 or np.sign(P[k]["est"]) == P[k]["side"]))
    return P


def slope(arm):
    Ts = np.arange(70, TMAX + 1, 10)
    per = {T: alpha(arm, T - 10, T) for T in Ts}
    scenes = sorted(set.intersection(*[set(v) for v in per.values()])) if all(per.values()) else []
    return ttest({s: float(np.polyfit(Ts, [per[T][s] for T in Ts], 1)[0] * 100) for s in scenes})


def expfit(arm, boot=2000):
    Ts = list(range(10, TMAX + 1, 10))
    per = {T: alpha(arm, T - 10, T) for T in Ts}
    scenes = sorted(set.intersection(*[set(v) for v in per.values()])) if all(per.values()) else []
    if len(scenes) < 5:
        return None
    M = np.array([[per[T][s] for T in Ts] for s in scenes])
    t = np.array(Ts, float)
    f = lambda t, ai, a0, tau: ai + (a0 - ai) * np.exp(-t / tau)

    def fit(y):
        try:
            p, _ = optimize.curve_fit(f, t, y, p0=(y[-1], y[0], 20.0), bounds=([-1, -1, 0.5], [2, 2, 2000]), maxfev=20000)
            return p
        except Exception:
            return None
    p = fit(M.mean(0))
    bs = [q for _ in range(boot) if (q := fit(M[rng.integers(0, len(scenes), len(scenes))].mean(0))) is not None]
    return dict(a_inf=float(p[0]), a_0=float(p[1]), tau_s=float(p[2]), n=len(scenes), n_boot=len(bs),
                a_inf_ci=np.percentile([q[0] for q in bs], [2.5, 97.5]).tolist() if bs else None) if p is not None else None


def seed_agreement(arm):
    a, b = alpha_seed(arm, 1, 60, 240), alpha_seed(arm, 2, 60, 240)
    s = sorted(set(a) & set(b))
    if len(s) < 5:
        return None
    r, p = stats.pearsonr([a[k] for k in s], [b[k] for k in s])
    return dict(r=float(r), p=float(p), n=len(s))


def collapse(arm):
    n = 0
    for S in L.get(arm, {}).values():
        for x in S.values():
            for v in x.values():
                k = len(v[HOLD:]) // FPS
                m = v[HOLD:HOLD + k * FPS].reshape(k, FPS).mean(1)
                n += int(bool((m < 3).any() or (m > 97).any()))
    return n


def summ(d):
    return dict(mean=float(np.mean(list(d.values()))), ci=ci(list(d.values())), n=len(d)) if d else None


res = {"counts": {a: {k: len(L.get(a, {}).get(k, {})) for k in SEEDS} for a in ARMS},
       "primary": holm(tests()),
       "per_seed": {k: {t: x for t, (x, _) in tests((k,)).items()} for k in SEEDS},
       "windows_60s": {a: {f"{t0}_{t0 + 60}": summ(alpha(a, t0, t0 + 60)) for t0 in (0, 60, 120, 180)} for a in ARMS},
       "slope_per100s_70_240": {a: slope(a) for a in ARMS},
       "expfit": {a: expfit(a) for a in ARMS},
       "alpha": {a: {T: summ(x) for T in range(10, TMAX + 1, 10) if (x := alpha(a, T - 10, T))} for a in ARMS},
       "alpha_1s": {a: {t: float(np.mean(list(x.values()))) for t in range(1, TMAX + 1) if (x := alpha(a, t - 1, t))}
                    for a in ARMS},
       "s24_tests": s24_tests(),
       "seed_agreement": {a: seed_agreement(a) for a in ARMS},
       "collapse": {a: collapse(a) for a in ARMS}}
json.dump(res, open(f"{out}/s29_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
print("60-s windows:", {a: [round(w["mean"], 2) if w else None for w in res["windows_60s"][a].values()] for a in ARMS})
print("a_inf:", {a: (round(e["a_inf"], 2), [round(x, 2) for x in e["a_inf_ci"] or []]) for a, e in res["expfit"].items() if e})
