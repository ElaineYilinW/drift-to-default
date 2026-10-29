"""Study 34: do the bursts come from single temporal RoPE frequencies of the sink? Holding one band of the sink's
temporal frequencies at the re-rotated distance in generation (preregistered, docs/preregistrations/34_band_hold.md).

Files {scene}_{arm}__{m16|p16}_s13.npz in RUN_DIR (Self Forcing DMD, sink 3, window 21, 16 fps, 60 s; frames 0..8 are
the held opening; lab_mean, motion), arm in
  bh_abs   sink at its original positions
  bh_j4    temporal frequency j = 4 (period 33.5 latent frames) of the sink held at distance 20, the others original
  bh_j3    j = 3 (period 22.1) held at 20
  bh_j811  j = 8..11 (periods 179-628) held at 20 (control band)
  bh_rr    all frequencies re-rotated (distance ~20; reference)
alpha over [t0, t1) s = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L* of frames 1..8, end = mean L* over
seconds [t0, t1) after the opening; undefined if open_p16 - open_m16 < 4. No collapse exclusion.
Onsets (as study 25: a probe peak at distance P is reached (P - 5) / 4 s after the opening):
  ORIG = 16.5, 32.25, 49.5 s     (peaks 71, 134, 203 of the first probe)
  NEW  = 7.0, 24.5, 40.0, 57.25 s (peaks 33, 103, 165, 234 of the band probe's "all but j3" curve that are not peaks of
                                   the full probe; results/corr/sinkband_out, prominence rule of analyze_sinkprobe_long.py)
Burst contrast C_arm(onsets) per scene = mean over the onsets tau of alpha([tau + 0.5, tau + 2.5)) - alpha([tau - 2.5,
tau - 0.5)).
Primary (Holm over 4; per-scene values; B1, B3, B4 two-sided one-sample t-tests, supported if Holm p < .05 and the
estimate > 0; B2 two one-sided t-tests (TOST), p = the larger one-sided p, supported if Holm p < .05; 95% CIs by scene
bootstrap, 10,000 draws; 90% t interval for B2):
  B1 holding j4 removes the bursts:         C_abs(ORIG) - C_j4(ORIG)                 > 0
  B2 holding j8-11 leaves them:             C_j811(ORIG) - C_abs(ORIG)               within +-0.10
  B3 holding j3 makes bursts at new times:  C_j3(NEW) - C_abs(NEW)                   > 0
  B4 without bursts less is kept:           alpha(bh_abs, [50, 60)) - alpha(bh_j4, [50, 60)) > 0
Secondary: every arm's C(ORIG), C(NEW) and per-onset contrasts; C_j3(16.5 s) - C_abs(16.5 s) (the burst j3 should
remove); alpha over 10-s windows and 1-s curves per arm; alpha over [50, 60) of every arm against bh_abs and bh_rr;
times of 1-s jumps above 0.15; correlation over t = 6..59 s of the first differences of each arm's 1-s alpha curve with
those of the band probe's share at the sink's natural distance (BAND_JSON curves "all" for bh_abs, "all but j4", "all
but j3", "all but j8-11"); motion and share of videos with motion < 1.5 over [50, 60) s per arm.
usage: python analyze_s34.py RUN_DIR OUT_DIR [BAND_JSON]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, out = sys.argv[1], sys.argv[2]
band = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD, TMAX = 16, 9, 60
ARMS = ("bh_abs", "bh_j4", "bh_j3", "bh_j811", "bh_rr")
BAND_CURVE = {"bh_abs": "all", "bh_j4": "all but j4", "bh_j3": "all but j3", "bh_j811": "all but j8-11"}
ORIG = (16.5, 32.25, 49.5)
NEW = (7.0, 24.5, 40.0, 57.25)
pat = re.compile(r"^(\w+?)_(" + "|".join(ARMS) + r")__(m16|p16)_s13\.npz$")

L, X = {}, {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        z = np.load(p)
        L.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = z["lab_mean"][:, 0]
        if "motion" in z.files:
            X.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = z["motion"]


def endL(v, t0, t1):
    i0, i1 = HOLD + int(round(t0 * FPS)), HOLD + int(round(t1 * FPS))
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


def contrast(arm, onsets):
    per = [(alpha(arm, t + 0.5, t + 2.5), alpha(arm, t - 2.5, t - 0.5)) for t in onsets]
    sc = set.intersection(*[set(a) & set(b) for a, b in per]) if per else set()
    return {s: float(np.mean([a[s] - b[s] for a, b in per])) for s in sc}


def pdiff(x, y):
    return {s: x[s] - y[s] for s in x if s in y}


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


def holm(t):
    P = {k: dict(x, side=s) for k, (x, s) in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(t) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and (P[k]["side"] == 0 or np.sign(P[k]["est"]) == P[k]["side"]))
    return P


def summ(d):
    return dict(mean=float(np.mean(list(d.values()))), ci=ci(list(d.values())), n=len(d)) if d else None


def curve1(arm):
    return {t: float(np.mean(list(x.values()))) for t in range(1, TMAX + 1) if (x := alpha(arm, t - 1, t))}


def jumps(arm, thr=0.15):
    c = curve1(arm)
    return [t for t in sorted(c) if t - 1 in c and c[t] - c[t - 1] > thr]


def band_share(name):
    """mean band-probe share over the chunks starting in each second [t - 1, t) at the sink's natural distance c + 2"""
    g, m = band["grid"], band["bands"][name]["curve"]
    o = {}
    for t in range(1, TMAX + 1):
        v = [m[g.index(c + 2)] for c in range(3, 243, 3) if t - 1 <= (c - 3) / 4 < t and c + 2 in g]
        if v:
            o[t] = float(np.mean(v))
    return o


def probe_corr():
    res = {}
    for arm, name in BAND_CURVE.items():
        c, q = curve1(arm), band_share(name)
        ts = [t for t in range(6, TMAX) if all(u in c and u in q for u in (t, t - 1))]
        if len(ts) > 5:
            r, p = stats.pearsonr([c[t] - c[t - 1] for t in ts], [q[t] - q[t - 1] for t in ts])
            res[f"{arm}_vs_{name}"] = dict(r=float(r), p=float(p), n=len(ts))
    return res


def motion(arm, t0=50, t1=60):
    o = [float(mo[HOLD + t0 * FPS:HOLD + t1 * FPS].mean()) for x in X.get(arm, {}).values() for mo in x.values()]
    return dict(motion=float(np.mean(o)), frozen=float(np.mean(np.array(o) < 1.5)), n=len(o)) if o else None


a60 = {a: alpha(a, 50, 60) for a in ARMS}
prim = {"B1": (ttest(pdiff(contrast("bh_abs", ORIG), contrast("bh_j4", ORIG))), +1),
        "B2": (tost(pdiff(contrast("bh_j811", ORIG), contrast("bh_abs", ORIG)), 0.10), 0),
        "B3": (ttest(pdiff(contrast("bh_j3", NEW), contrast("bh_abs", NEW))), +1),
        "B4": (ttest(pdiff(a60["bh_abs"], a60["bh_j4"])), +1)}
res = {"counts": {a: sum(len(x) == 2 for x in L.get(a, {}).values()) for a in ARMS},
       "onsets": {"orig": ORIG, "new": NEW},
       "primary": holm(prim),
       "contrast": {a: {"orig": summ(contrast(a, ORIG)), "new": summ(contrast(a, NEW)),
                        "per_onset": {t: summ(contrast(a, (t,))) for t in ORIG + NEW}} for a in ARMS},
       "j3_removes_16.5": ttest(pdiff(contrast("bh_j3", (16.5,)), contrast("bh_abs", (16.5,)))),
       "alpha": {a: {T: summ(alpha(a, T - 10, T)) for T in range(10, TMAX + 1, 10)} for a in ARMS},
       "alpha60_vs_abs": {a: ttest(pdiff(a60[a], a60["bh_abs"])) for a in ARMS if a != "bh_abs"},
       "alpha60_vs_rr": {a: ttest(pdiff(a60[a], a60["bh_rr"])) for a in ARMS if a != "bh_rr"},
       "alpha_1s": {a: curve1(a) for a in ARMS},
       "jumps": {a: jumps(a) for a in ARMS},
       "probe_corr": probe_corr() if band else None,
       "motion_50_60": {a: motion(a) for a in ARMS}}
json.dump(res, open(f"{out}/s34_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
for a in ARMS:
    c = res["contrast"][a]
    if c["orig"]:
        print(a, f"C(orig) {c['orig']['mean']:+.3f} C(new) {c['new']['mean']:+.3f} alpha60 "
                 f"{res['alpha'][a][60]['mean']:.2f} jumps {res['jumps'][a]} motion {res['motion_50_60'][a]}")
print("probe corr:", res["probe_corr"])
