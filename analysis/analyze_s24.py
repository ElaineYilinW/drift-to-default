"""Study 24: does an opening that stays in the context keep a steady share over 180 s, and is it forgotten without one?
(preregistered, docs/preregistrations/24_long_horizon.md).

Files {scene}_{arm}__{m16|p16}_s0.npz in RUN_DIR (16 fps; frames 0..8 are the held opening), arm in sf180_s0,
sf180_s3rr, sf180_s3, rf180, ll180_s3; fresh starts fresh/rx00_fresh_{arm}_s{0..3}.npz. alpha(T) = (end_p16 - end_m16) /
(open_p16 - open_m16), open = mean L* of frames 1..8, end = mean L* over seconds [T - 10, T) after the opening;
undefined if open_p16 - open_m16 < 4. No collapse exclusion.
Primary (Holm over 5; per-scene values, T = 180 s; t-tests two-sided, supported if Holm p < .05 and the estimate is on
the predicted side; equivalence by two one-sided t-tests against the margin (TOST), p = the larger one-sided p,
supported if Holm p < .05; 95% CIs by scene bootstrap, 10,000 draws; 90% t intervals for TOST):
  L1 a re-rotated sink keeps a share at 180 s:      alpha(sf180_s3rr, 180) > 0
  L2 Rolling Forcing keeps a share at 180 s:         alpha(rf180, 180) > 0
  L3 the re-rotated sink's share is steady:          alpha(sf180_s3rr, 180) - alpha(sf180_s3rr, 60) within +-0.10
  L4 Rolling Forcing's share is steady:              alpha(rf180, 180) - alpha(rf180, 60) within +-0.10
  L5 without a sink the opening stays forgotten:     alpha(sf180_s0, 180) within +-0.10
Secondary: L1-L5 with the window [160, 170) s; alpha(T) for T = 10..180 (10-s windows) and 1-s curves per arm;
exponential fit alpha(t) = a_inf + (a_0 - a_inf) exp(-t / tau) to each arm's mean 10-s curve (T >= 10), with a scene-
bootstrap CI of a_inf (2,000 draws); alpha(sf180_s3, 180) - alpha(sf180_s3rr, 180) and alpha(ll180_s3, 180); the times
of the 1-s jumps above 0.15 of the sinks at original positions (sf180_s3, ll180_s3); collapse counts (a 1-s mean L*
below 3 or above 97); attractor offset of the fresh starts, end(T) - mean L* of their first second, T = 60, 120, 180.
usage: python analyze_s24.py RUN_DIR OUT_DIR
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
FPS, HOLD, TMAX = 16, 9, 180
ARMS = ("sf180_s0", "sf180_s3rr", "sf180_s3", "rf180", "ll180_s3")
pat = re.compile(r"^(\w+?)_(sf180_s0|sf180_s3rr|sf180_s3|rf180|ll180_s3)__(m16|p16)_s0\.npz$")
fpat = re.compile(r"^rx00_fresh_(sf180_s0|sf180_s3rr|sf180_s3|rf180|ll180_s3)_s(\d+)\.npz$")

L = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        L.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]
F = {}
for p in glob.glob(f"{src}/fresh/*.npz"):
    m = fpat.match(os.path.basename(p))
    if m:
        F.setdefault(m[1], {})[int(m[2])] = np.load(p)["lab_mean"][:, 0]


def endL(v, t0, t1):
    i0, i1 = HOLD + int(t0 * FPS), HOLD + int(t1 * FPS)
    return float(v[i0:i1].mean()) if i1 <= len(v) else None


def alpha(arm, t0, t1):
    o = {}
    for s, x in L.get(arm, {}).items():
        if "m16" in x and "p16" in x:
            d = float(x["p16"][1:HOLD].mean() - x["m16"][1:HOLD].mean())
            eh, el = endL(x["p16"], t0, t1), endL(x["m16"], t0, t1)
            if d >= 4 and eh is not None and el is not None:
                o[s] = (eh - el) / d
    return o


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


def tests(t0, t1):
    A = lambda arm, a, b: alpha(arm, a, b)
    return {"L1": (ttest(A("sf180_s3rr", t0, t1)), +1),
            "L2": (ttest(A("rf180", t0, t1)), +1),
            "L3": (tost(pdiff(A("sf180_s3rr", t0, t1), A("sf180_s3rr", 50, 60)), 0.10), 0),
            "L4": (tost(pdiff(A("rf180", t0, t1), A("rf180", 50, 60)), 0.10), 0),
            "L5": (tost(A("sf180_s0", t0, t1), 0.10), 0)}


def holm(t):
    P = {k: dict(x, side=s) for k, (x, s) in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(t) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and (P[k]["side"] == 0 or np.sign(P[k]["est"]) == P[k]["side"]))
    return P


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


def jumps(arm, thr=0.15):
    c = {t: np.mean(list(x.values())) for t in range(1, TMAX + 1) if (x := alpha(arm, t - 1, t))}
    return [t for t in sorted(c) if t - 1 in c and c[t] - c[t - 1] > thr]


def collapse(arm):
    n = 0
    for x in L.get(arm, {}).values():
        for v in x.values():
            k = len(v[HOLD:]) // FPS
            m = v[HOLD:HOLD + k * FPS].reshape(k, FPS).mean(1)
            n += int(bool((m < 3).any() or (m > 97).any()))
    return n


res = {"counts": {a: len(L.get(a, {})) for a in ARMS}, "fresh_counts": {a: len(F.get(a, {})) for a in ARMS},
       "primary": holm(tests(TMAX - 10, TMAX)),
       "window_160_170": {k: x for k, (x, _) in tests(160, 170).items()},
       "alpha": {a: {T: dict(mean=float(np.mean(list(x.values()))), ci=ci(list(x.values())), n=len(x))
                     for T in range(10, TMAX + 1, 10) if (x := alpha(a, T - 10, T))} for a in ARMS},
       "alpha_1s": {a: {t: float(np.mean(list(x.values()))) for t in range(1, TMAX + 1) if (x := alpha(a, t - 1, t))}
                    for a in ARMS},
       "expfit": {a: expfit(a) for a in ARMS},
       "s3_minus_s3rr_180": ttest(pdiff(alpha("sf180_s3", 170, 180), alpha("sf180_s3rr", 170, 180))),
       "ll_s3_180": ttest(alpha("ll180_s3", 170, 180)),
       "jumps": {a: jumps(a) for a in ("sf180_s3", "ll180_s3", "sf180_s3rr", "rf180")},
       "collapse": {a: collapse(a) for a in ARMS},
       "fresh_offset": {a: {T: float(np.mean([endL(v, T - 10, T) - v[HOLD:HOLD + FPS].mean() for v in F[a].values()]))
                            for T in (60, 120, 180)} for a in ARMS if F.get(a)}}
json.dump(res, open(f"{out}/s24_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
print("alpha at 60/120/180:", {a: [round(res["alpha"][a][T]["mean"], 2) for T in (60, 120, 180) if T in res["alpha"][a]]
                               for a in ARMS})
print("a_inf:", {a: (round(e["a_inf"], 2), [round(x, 2) for x in e["a_inf_ci"] or []]) for a, e in res["expfit"].items() if e})
