"""Study 38: does the sink held at the distance chosen by study 37's rule keep more of the opening than the re-aligned
sink, without bursts and without moving less, on new openings and four models? (preregistered,
docs/preregistrations/38_held_sink_test.md).

Files {scene}_{m}_{held|rr|abs}__{m16|p16}_s5.npz in RUN_DIR, m in sf, ll, cv, gan (16 fps; frames 0..8 are the held
opening; 120 s). alpha over [t0, t1) = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L* of frames 1..8,
end = mean L* over seconds [t0, t1) after the opening; undefined if open_p16 - open_m16 < 4. No collapse exclusion.
motion over [t0, t1) = the generator's per-frame motion averaged over [t0, t1) and the two versions. Burst contrast of
one scene = mean over a model's predicted onsets tau of alpha([tau + 0.5, tau + 2.5)) - alpha([tau - 2.5, tau - 0.5));
the onsets are (P - 5) / 4 s for the model's major probe peaks P (prominence >= 0.04 on its probe's mean curve, the rule
of analyze_sinkprobe_long.py) with 62.5 <= tau <= 117.5. Probe outputs: PROBES = sf=FILE,ll=FILE,cv=FILE,gan=FILE.
Primary, for every model with a D* (Holm over all of them; per-scene values; 95% CIs by scene bootstrap, 10,000 draws):
  H1_m  held keeps more than re-aligned:  alpha(held, [60, 120)) - alpha(rr, [60, 120)) > 0      (two-sided t)
  H2_m  held does not burst:              burst contrast(held) within +-0.05                       (TOST; untestable and
                                                                                                    left out if the
                                                                                                    model has no onset)
  H3_m  held moves no less than re-aligned: motion(held, [60, 120)) - motion(rr, [60, 120)) + 0.5 > 0 (two-sided t on
                                                                                                    the shifted value)
A t-test hypothesis is supported if its Holm p < .05 and the estimate is > 0; a TOST hypothesis if its Holm p < .05.
Secondary: per model and arm alpha over [0, 60), [60, 120) and 10-s windows, 1-s curves, motion, frozen share (mean
motion over [60, 120) below 1.5), sharpness ratio [60, 120) / [0, 10), burst contrast of every arm;
alpha(held) - alpha(abs) over [60, 120); motion(held) - motion(abs); collapse counts.
usage: python analyze_s38.py RUN_DIR DSTAR.json PROBES OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

src, dstar, probes, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD, TMAX, FROZEN, PROM = 16, 9, 120, 1.5, 0.04
sel = json.load(open(dstar))
MODELS = [m for m in ("sf", "ll", "cv", "gan") if sel.get(m, {}).get("dstar") is not None]
PROBE = dict(kv.split("=", 1) for kv in probes.split(","))
pat = re.compile(r"^(\w+?)_(sf|ll|cv|gan)_(held|rr|abs)__(m16|p16)_s5\.npz$")


def onsets(m):
    p = json.load(open(PROBE[m]))
    g, c = np.array(p["grid"]), np.array(p["mean"])
    i, _ = find_peaks(c, prominence=PROM)
    return [float(t) for t in ((g[i] - 5) / 4) if 62.5 <= t <= TMAX - 2.5]


Z = {}   # Z[(m, arm)][scene][version]
for p in glob.glob(f"{src}/*.npz"):
    x = pat.match(os.path.basename(p))
    if x:
        z = np.load(p)
        Z.setdefault((x[2], x[3]), {}).setdefault(x[1], {})[x[4]] = {k: z[k] for k in ("lab_mean", "motion", "sharpness")}


def endL(v, t0, t1):
    i0, i1 = HOLD + int(round(t0 * FPS)), HOLD + int(round(t1 * FPS))
    return float(v[i0:i1].mean()) if i1 <= len(v) else None


def alpha(k, t0, t1):
    o = {}
    for s, x in Z.get(k, {}).items():
        if "m16" in x and "p16" in x:
            h, l = x["p16"]["lab_mean"][:, 0], x["m16"]["lab_mean"][:, 0]
            d = float(h[1:HOLD].mean() - l[1:HOLD].mean())
            eh, el = endL(h, t0, t1), endL(l, t0, t1)
            if d >= 4 and eh is not None and el is not None:
                o[s] = (eh - el) / d
    return o


def motion(k, t0, t1):
    o = {}
    i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
    for s, x in Z.get(k, {}).items():
        v = [float(q["motion"][i0:i1].mean()) for q in x.values() if len(q["motion"]) >= i1]
        if v:
            o[s] = float(np.mean(v))
    return o


def frozen(k, t0=60, t1=120):
    i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
    v = [float(q["motion"][i0:i1].mean()) < FROZEN for x in Z.get(k, {}).values() for q in x.values()
         if len(q["motion"]) >= i1]
    return float(np.mean(v)) if v else None


def sharp(k, t0=60, t1=120):
    o = {}
    for s, x in Z.get(k, {}).items():
        v = [float(q["sharpness"][HOLD + t0 * FPS:HOLD + t1 * FPS].mean() / q["sharpness"][HOLD:HOLD + 10 * FPS].mean())
             for q in x.values() if len(q["sharpness"]) >= HOLD + t1 * FPS]
        if v:
            o[s] = float(np.mean(v))
    return o


def contrast(k, ons):
    per = [(alpha(k, t + 0.5, t + 2.5), alpha(k, t - 2.5, t - 0.5)) for t in ons]
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


def tost(d, mg):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    n, mu, se = len(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    p_lo, p_hi = float(stats.t.sf((mu + mg) / se, n - 1)), float(stats.t.cdf((mu - mg) / se, n - 1))
    h = stats.t.ppf(0.95, n - 1) * se
    return dict(est=float(mu), ci=ci(v), ci90=[float(mu - h), float(mu + h)], margin=mg, p_lower=p_lo, p_upper=p_hi,
                p=max(p_lo, p_hi), n=int(n))


def pdiff(x, y, shift=0.0):
    return {s: x[s] - y[s] + shift for s in x if s in y}


ONS = {m: onsets(m) for m in MODELS}
tests = {}
for m in MODELS:
    H, R = (m, "held"), (m, "rr")
    tests[f"H1_{m}"] = (ttest(pdiff(alpha(H, 60, 120), alpha(R, 60, 120))), +1)
    if ONS[m]:
        tests[f"H2_{m}"] = (tost(contrast(H, ONS[m]), 0.05), 0)
    tests[f"H3_{m}"] = (ttest(pdiff(motion(H, 60, 120), motion(R, 60, 120), 0.5)), +1)


def holm(t):
    P = {k: dict(x, side=s) for k, (x, s) in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(P) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and (P[k]["side"] == 0 or np.sign(P[k]["est"]) == P[k]["side"]))
    return P


def summ(d):
    return dict(mean=float(np.mean(list(d.values()))), ci=ci(list(d.values())), n=len(d)) if d else None


def collapse(k):
    n = 0
    for x in Z.get(k, {}).values():
        for q in x.values():
            v = q["lab_mean"][:, 0]
            kk = len(v[HOLD:]) // FPS
            mm = v[HOLD:HOLD + kk * FPS].reshape(kk, FPS).mean(1)
            n += int(bool((mm < 3).any() or (mm > 97).any()))
    return n


ARMS = ("held", "rr", "abs")
res = {"dstar": {m: sel[m]["dstar"] for m in MODELS}, "realign": {m: sel[m]["realign"] for m in MODELS},
       "onsets": ONS, "untested": [m for m in MODELS if not ONS[m]],
       "counts": {f"{m}_{a}": sum("m16" in x and "p16" in x for x in Z.get((m, a), {}).values())
                  for m in MODELS for a in ARMS},
       "primary": holm(tests)}
sec = {}
for m in MODELS:
    k = {a: (m, a) for a in ARMS}
    sec[m] = {"alpha_60s": {a: {f"{w[0]}_{w[1]}": summ(alpha(k[a], *w)) for w in ((0, 60), (60, 120))} for a in ARMS},
              "motion_60s": {a: {f"{w[0]}_{w[1]}": summ(motion(k[a], *w)) for w in ((0, 60), (60, 120))} for a in ARMS},
              "frozen_60_120": {a: frozen(k[a]) for a in ARMS},
              "sharpness_ratio": {a: summ(sharp(k[a])) for a in ARMS},
              "burst_contrast": {a: ttest(contrast(k[a], ONS[m])) for a in ARMS} if ONS[m] else None,
              "held_minus_abs_alpha": ttest(pdiff(alpha(k["held"], 60, 120), alpha(k["abs"], 60, 120))),
              "held_minus_abs_motion": ttest(pdiff(motion(k["held"], 60, 120), motion(k["abs"], 60, 120))),
              "alpha": {a: {T: summ(x) for T in range(10, TMAX + 1, 10) if (x := alpha(k[a], T - 10, T))} for a in ARMS},
              "alpha_1s": {a: {t: float(np.mean(list(x.values()))) for t in range(1, TMAX + 1)
                               if (x := alpha(k[a], t - 1, t))} for a in ARMS},
              "collapse": {a: collapse(k[a]) for a in ARMS}}
res["secondary"] = sec
json.dump(res, open(f"{out}/s38_results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
for m in MODELS:
    print(m, "D*", sel[m]["dstar"], "alpha 60-120:",
          {a: round(w["mean"], 3) if (w := sec[m]["alpha_60s"][a]["60_120"]) else None for a in ARMS},
          "frozen:", sec[m]["frozen_60_120"])
