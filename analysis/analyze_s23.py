"""Study 23: is relaxation a linear response? Dose-response of tone persistence on Self Forcing
(preregistered, docs/preregistrations/23_dose_response.md).

Files {scene}_{arm}__{v}_s{seed}.npz anywhere under RUN_DIR (16 fps; frames 0..8 are the held opening), arm in sf60_s3,
sf60_s0, sf60_s3rr, v in m24, m8, z0, p8, p24 (the opening's L* shifted by -24, -8, 0, +8, +24). open_v = mean L* of
frames 1..8, end_v = mean L* over seconds [T - 10, T) after the opening. For one scene, arm and seed:
  alpha(hi, lo) = (end_hi - end_lo) / (open_hi - open_lo), undefined if open_hi - open_lo < 4;
  alpha_small = alpha(p8, m8), alpha_large = alpha(p24, m24), alpha_up = alpha(p24, z0), alpha_down = alpha(z0, m24).
A scene's value is the mean over its seeds. Primary (Holm over 4; T = 60 s; per-scene values; equivalence by two
one-sided t-tests against the margin (TOST), p = the larger one-sided p; supported if Holm p < .05; 95% CIs by scene
bootstrap (10,000 draws), 90% t intervals reported alongside):
  D1 sink 3, persistence does not depend on the shift size:  alpha_large - alpha_small  within +-0.15
  D2 re-rotated sink 3, the same:                             alpha_large - alpha_small  within +-0.15
  D3 sink 3, darkening and brightening are kept alike:        alpha_up - alpha_down      within +-0.15
  D4 no sink, the large shift is forgotten too:               alpha_large(sf60_s0)       within +-0.10
Secondary: D1-D4 over [40, 50) s; alpha_small, alpha_large, alpha_up, alpha_down at T = 10..60 per arm; per scene, arm
and seed, least-squares fits of end(60) on open over the five openings (slope, R^2, and the quadratic coefficient of a
fit in centred open, tested against 0 per arm); each seed's D1-D4 estimates; the jump alpha([34, 35)) - alpha([31, 32))
of the small and the large pair per arm, and their difference; the attractor offset end_z0(60) - open_z0 per arm;
alpha_16 = alpha(p16, m16) of the same scenes, arms and seeds from studies 10, 16 and 20 next to alpha_small and
alpha_large (descriptive, when PRIOR_DIRs are given); the number of videos per arm and opening with a 1-s mean L* below
3 or above 97 (collapse; no exclusion).
usage: python analyze_s23.py RUN_DIR OUT_DIR [PRIOR_DIR ...]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, out, prior_dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD = 16, 9
ARMS = ("sf60_s3", "sf60_s0", "sf60_s3rr")
VERS = {"m24": -24, "m8": -8, "z0": 0, "p8": 8, "p24": 24}
PAIRS = {"small": ("p8", "m8"), "large": ("p24", "m24"), "up": ("p24", "z0"), "down": ("z0", "m24")}
pat = re.compile(r"^(\w+?)_(sf60_s3|sf60_s0|sf60_s3rr)__(m24|m8|z0|p8|p24|m16|p16)_s(\d+)\.npz$")


def load(dirs, seeds=None):
    D = {}
    for d in dirs:
        for p in glob.glob(f"{d}/**/*.npz", recursive=True):
            m = pat.match(os.path.basename(p))
            if m and (seeds is None or int(m[4]) in seeds):
                D.setdefault(m[2], {}).setdefault(m[1], {}).setdefault(int(m[4]), {})[m[3]] = np.load(p)["lab_mean"][:, 0]
    return D


def openL(v):
    return float(v[1:HOLD].mean())


def endL(v, t0, t1):
    i0, i1 = HOLD + int(t0 * FPS), HOLD + int(t1 * FPS)
    return float(v[i0:i1].mean()) if i1 <= len(v) else None


def a_pair(x, hi, lo, t0, t1):
    if hi not in x or lo not in x:
        return None
    eh, el = endL(x[hi], t0, t1), endL(x[lo], t0, t1)
    d = openL(x[hi]) - openL(x[lo])
    if eh is None or el is None or d < 4:
        return None
    return (eh - el) / d


def alpha(D, arm, pair, t0, t1, seed=None):
    """{scene: alpha of the pair averaged over the scene's seeds (or of one seed)}"""
    o = {}
    for s, by in D.get(arm, {}).items():
        v = [a_pair(x, *PAIRS[pair], t0, t1) if pair in PAIRS else a_pair(x, *pair, t0, t1)
             for k, x in by.items() if seed is None or k == seed]
        v = [x for x in v if x is not None]
        if v:
            o[s] = float(np.mean(v))
    return o


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def summ(d):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def tost(d, m):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    n, mu, se = len(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    p_lo = float(stats.t.sf((mu + m) / se, n - 1))        # H0: mean <= -m
    p_hi = float(stats.t.cdf((mu - m) / se, n - 1))       # H0: mean >= +m
    h = stats.t.ppf(0.95, n - 1) * se
    return dict(est=float(mu), ci=ci(v), ci90=[float(mu - h), float(mu + h)], margin=m, p_lower=p_lo, p_upper=p_hi,
                p=max(p_lo, p_hi), n=int(n))


def pdiff(x, y):
    return {s: x[s] - y[s] for s in x if s in y}


def tests(D, t0, t1, seed=None):
    A = lambda arm, pair: alpha(D, arm, pair, t0, t1, seed)
    return {"D1": tost(pdiff(A("sf60_s3", "large"), A("sf60_s3", "small")), 0.15),
            "D2": tost(pdiff(A("sf60_s3rr", "large"), A("sf60_s3rr", "small")), 0.15),
            "D3": tost(pdiff(A("sf60_s3", "up"), A("sf60_s3", "down")), 0.15),
            "D4": tost(A("sf60_s0", "large"), 0.10)}


def holm(t):
    prim = {k: dict(x) for k, x in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(prim, key=lambda k: prim[k]["p"])):
        running = max(running, min(1.0, (len(t) - i) * prim[k]["p"]))
        prim[k]["p_holm"] = running
        prim[k]["supported"] = bool(running < 0.05)
    return prim


def fits(D, arm, t0=50, t1=60):
    """per scene and seed: slope, R^2 and quadratic coefficient of end on open over the five openings"""
    rows = []
    for s, by in D.get(arm, {}).items():
        for k, x in by.items():
            vs = [v for v in VERS if v in x and endL(x[v], t0, t1) is not None]
            if len(vs) < 5:
                continue
            o = np.array([openL(x[v]) for v in vs])
            e = np.array([endL(x[v], t0, t1) for v in vs])
            b1, b0 = np.polyfit(o, e, 1)
            r2 = 1 - ((e - (b1 * o + b0)) ** 2).sum() / ((e - e.mean()) ** 2).sum()
            c2 = np.polyfit(o - o.mean(), e, 2)[0]
            rows.append(dict(scene=s, seed=k, slope=float(b1), r2=float(r2), quad=float(c2)))
    if not rows:
        return None
    sl = {}
    for r in rows:
        sl.setdefault(r["scene"], []).append(r)
    by_scene = lambda key: {s: float(np.mean([r[key] for r in rs])) for s, rs in sl.items()}
    return dict(slope=summ(by_scene("slope")), r2_mean=float(np.mean([r["r2"] for r in rows])),
                quad=summ(by_scene("quad")), n_fits=len(rows))


def collapse(D, arm):
    c = {}
    for by in D.get(arm, {}).values():
        for x in by.values():
            for v, L in x.items():
                n = len(L[HOLD:]) // FPS
                m = L[HOLD:HOLD + n * FPS].reshape(n, FPS).mean(1)
                c[v] = c.get(v, 0) + int(bool((m < 3).any() or (m > 97).any()))
    return c


D = load([src])
seeds = sorted({k for a in D.values() for by in a.values() for k in by})
res = {"counts": {a: {s: {k: sorted(x) for k, x in by.items()} for s, by in D.get(a, {}).items()} for a in ARMS},
       "seeds": seeds,
       "primary": holm(tests(D, 50, 60)),
       "window_40_50": tests(D, 40, 50),
       "per_seed": {sd: tests(D, 50, 60, sd) for sd in seeds},
       "alpha": {a: {p: {T: dict(mean=float(np.mean(list(x.values()))), ci=ci(list(x.values())), n=len(x))
                         for T in (10, 20, 30, 40, 50, 60) if (x := alpha(D, a, p, T - 10, T))} for p in PAIRS}
                 for a in ARMS},
       "fits_60": {a: fits(D, a) for a in ARMS},
       "jump_31_34": {a: {p: summ(pdiff(alpha(D, a, p, 34, 35), alpha(D, a, p, 31, 32))) for p in ("small", "large")}
                      for a in ARMS},
       "attractor_offset_60": {a: summ({s: float(np.mean([endL(x["z0"], 50, 60) - openL(x["z0"]) for x in by.values()
                                                          if "z0" in x and endL(x["z0"], 50, 60) is not None]))
                                        for s, by in D.get(a, {}).items()
                                        if any("z0" in x and endL(x["z0"], 50, 60) is not None for x in by.values())})
                               for a in ARMS},
       "collapse": {a: collapse(D, a) for a in ARMS}}
for a in ARMS:
    j = res["jump_31_34"][a]
    res["jump_31_34"][a]["large_minus_small"] = summ(pdiff(
        pdiff(alpha(D, a, "large", 34, 35), alpha(D, a, "large", 31, 32)),
        pdiff(alpha(D, a, "small", 34, 35), alpha(D, a, "small", 31, 32))))
if prior_dirs:
    P = load(prior_dirs, set(seeds))
    res["alpha16"] = {}
    for a in ARMS:
        x16 = alpha(P, a, ("p16", "m16"), 50, 60)
        both = [s for s in x16 if s in alpha(D, a, "small", 50, 60)]
        res["alpha16"][a] = dict(alpha16=summ(x16), n_matched=len(both),
                                 small=summ({s: alpha(D, a, "small", 50, 60)[s] for s in both}),
                                 large=summ({s: alpha(D, a, "large", 50, 60)[s] for s in both}))
json.dump(res, open(f"{out}/s23_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci90=[{t['ci90'][0]:+.3f}, {t['ci90'][1]:+.3f}] margin={t['margin']} n={t['n']} "
             f"p_holm={t['p_holm']:.4g}", "supported" if t["supported"] else "not supported")
print("alpha60:", {a: {p: round(res["alpha"][a][p][60]["mean"], 3) for p in PAIRS if 60 in res["alpha"][a].get(p, {})}
                   for a in ARMS})
