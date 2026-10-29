"""Study 31: does a sink held at a probe-chosen distance below the freezing threshold keep more of the opening than a
re-rotated sink for 180 s, while moving more than the sink at its original positions and without bursts?
(preregistered, docs/preregistrations/31_held_sink_lever.md).

Files {scene}_{arm}__{m16|p16}_s4.npz in RUN_DIR (16 fps; frames 0..8 are the held opening), arm in sp_d066, sp_d448,
sp_rr, sp_abs (Self Forcing, sink 3, hold_gen_sinkpos.py). alpha over [t0, t1) s = (end_p16 - end_m16) /
(open_p16 - open_m16), open = mean L* of frames 1..8, end = mean L* over seconds [t0, t1) after the opening; undefined
if open_p16 - open_m16 < 4. No collapse exclusion. motion over [t0, t1) = mean frame-to-frame grey-level change (the
generator's per-frame motion), averaged over the two versions. Burst contrast of one scene (as in study 26): mean over
the 13 predicted onsets tau after 60 s of alpha([tau + 0.5, tau + 2.5)) - alpha([tau - 2.5, tau - 0.5)).
Primary (Holm over 4; per-scene values; t-tests two-sided, supported if Holm p < .05 and the estimate > 0; equivalence
by two one-sided t-tests (TOST), p = the larger one-sided p, supported if Holm p < .05; 95% CIs by scene bootstrap,
10,000 draws; 90% t intervals for TOST):
  K1 held at 66 keeps more than re-rotated:    alpha(sp_d066, [120, 180)) - alpha(sp_rr, [120, 180))      > 0
  K2 held at 448 keeps more than re-rotated:   alpha(sp_d448, [120, 180)) - alpha(sp_rr, [120, 180))      > 0
  K3 held at 66 moves more than the original:  motion(sp_d066, [120, 180)) - motion(sp_abs, [120, 180))   > 0
  K4 held at 66 does not burst:                burst contrast(sp_d066) within +-0.05
Secondary: K1-K3 over [60, 120); per arm alpha over the 60-s windows [0, 60), [60, 120), [120, 180), alpha (10-s
windows) and 1-s curves, motion over the same 60-s windows, share of videos with motion < 1.5 over [120, 180), sharpness
over [120, 180) relative to [0, 10) (both versions), burst contrast of every arm; motion(sp_d066) - motion(sp_rr) and
motion(sp_d448) - motion(sp_abs) over [120, 180); alpha(sp_d066, [50, 60)) against study 27's 0.41 (new seed); collapse
counts (a 1-s mean L* below 3 or above 97).
usage: python analyze_s31.py RUN_DIR OUT_DIR
"""
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
FPS, HOLD, TMAX, FROZEN = 16, 9, 180, 1.5
ARMS = ("sp_d066", "sp_d448", "sp_rr", "sp_abs")
PEAKS = (71, 90, 134, 203, 266, 309, 335, 404, 441, 467, 510, 536, 574, 599, 642, 669, 711)  # study 26
PRED = [t for t in ((P - 5) / 4 for P in PEAKS) if 62.5 <= t and t + 2.5 <= TMAX]
pat = re.compile(r"^(\w+?)_(sp_d066|sp_d448|sp_rr|sp_abs)__(m16|p16)_s4\.npz$")

Z = {}   # Z[arm][scene][version] = npz contents
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        z = np.load(p)
        Z.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = {k: z[k] for k in ("lab_mean", "motion", "sharpness")}


def endL(v, t0, t1):
    i0, i1 = HOLD + int(round(t0 * FPS)), HOLD + int(round(t1 * FPS))
    return float(v[i0:i1].mean()) if i1 <= len(v) else None


def alpha(a, t0, t1):
    o = {}
    for s, x in Z.get(a, {}).items():
        if "m16" in x and "p16" in x:
            h, l = x["p16"]["lab_mean"][:, 0], x["m16"]["lab_mean"][:, 0]
            d = float(h[1:HOLD].mean() - l[1:HOLD].mean())
            eh, el = endL(h, t0, t1), endL(l, t0, t1)
            if d >= 4 and eh is not None and el is not None:
                o[s] = (eh - el) / d
    return o


def motion(a, t0, t1):
    o = {}
    i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
    for s, x in Z.get(a, {}).items():
        v = [float(q["motion"][i0:i1].mean()) for q in x.values() if len(q["motion"]) >= i1]
        if v:
            o[s] = float(np.mean(v))
    return o


def frozen(a, t0=120, t1=180):
    i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
    v = [float(q["motion"][i0:i1].mean()) < FROZEN for x in Z.get(a, {}).values() for q in x.values()
         if len(q["motion"]) >= i1]
    return float(np.mean(v)) if v else None


def sharp(a, t0=120, t1=180):
    o = {}
    for s, x in Z.get(a, {}).items():
        v = [float(q["sharpness"][HOLD + t0 * FPS:HOLD + t1 * FPS].mean() / q["sharpness"][HOLD:HOLD + 10 * FPS].mean())
             for q in x.values() if len(q["sharpness"]) >= HOLD + t1 * FPS]
        if v:
            o[s] = float(np.mean(v))
    return o


def contrast(a, onsets=PRED):
    per = [(alpha(a, t + 0.5, t + 2.5), alpha(a, t - 2.5, t - 0.5)) for t in onsets]
    sc = set.intersection(*[set(x) & set(y) for x, y in per]) if per else set()
    return {s: float(np.mean([x[s] - y[s] for x, y in per])) for s in sc}


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


def tests(t0=120, t1=180):
    return {"K1": (ttest(pdiff(alpha("sp_d066", t0, t1), alpha("sp_rr", t0, t1))), +1),
            "K2": (ttest(pdiff(alpha("sp_d448", t0, t1), alpha("sp_rr", t0, t1))), +1),
            "K3": (ttest(pdiff(motion("sp_d066", t0, t1), motion("sp_abs", t0, t1))), +1),
            "K4": (tost(contrast("sp_d066"), 0.05), 0)}


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


def collapse(a):
    n = 0
    for x in Z.get(a, {}).values():
        for q in x.values():
            v = q["lab_mean"][:, 0]
            k = len(v[HOLD:]) // FPS
            m = v[HOLD:HOLD + k * FPS].reshape(k, FPS).mean(1)
            n += int(bool((m < 3).any() or (m > 97).any()))
    return n


W = ((0, 60), (60, 120), (120, 180))
res = {"counts": {a: sum("m16" in x and "p16" in x for x in Z.get(a, {}).values()) for a in ARMS},
       "predicted_onsets": PRED,
       "primary": holm(tests()),
       "window_60_120": {k: x for k, (x, _) in tests(60, 120).items()},
       "alpha_60s": {a: {f"{t0}_{t1}": summ(alpha(a, t0, t1)) for t0, t1 in W} for a in ARMS},
       "motion_60s": {a: {f"{t0}_{t1}": summ(motion(a, t0, t1)) for t0, t1 in W} for a in ARMS},
       "frozen_share_120_180": {a: frozen(a) for a in ARMS},
       "sharpness_ratio_120_180": {a: summ(sharp(a)) for a in ARMS},
       "burst_contrast": {a: ttest(contrast(a)) for a in ARMS},
       "motion_d066_minus_rr": ttest(pdiff(motion("sp_d066", 120, 180), motion("sp_rr", 120, 180))),
       "motion_d448_minus_abs": ttest(pdiff(motion("sp_d448", 120, 180), motion("sp_abs", 120, 180))),
       "d066_alpha_50_60": summ(alpha("sp_d066", 50, 60)),
       "alpha": {a: {T: summ(x) for T in range(10, TMAX + 1, 10) if (x := alpha(a, T - 10, T))} for a in ARMS},
       "alpha_1s": {a: {t: float(np.mean(list(x.values()))) for t in range(1, TMAX + 1) if (x := alpha(a, t - 1, t))}
                    for a in ARMS},
       "collapse": {a: collapse(a) for a in ARMS}}
json.dump(res, open(f"{out}/s31_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
print("alpha 60-s:", {a: [round(w["mean"], 2) if w else None for w in res["alpha_60s"][a].values()] for a in ARMS})
print("motion 60-s:", {a: [round(w["mean"], 2) if w else None for w in res["motion_60s"][a].values()] for a in ARMS})
