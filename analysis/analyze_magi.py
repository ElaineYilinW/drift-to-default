"""Study 18: MAGI-1, a non-Wan autoregressive model, with the released attention window and with full history
(preregistered, docs/preregistrations/18_magi.md).

Files {scene}_{mgw|mgf}__{m16|p16}_s0.npz (24 fps; frames 0..8 are the held opening) and rx02_fresh_{mgw|mgf}_s{0..3}.npz
(text-to-video, no opening). alpha(T) = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L* of frames 1..8,
end = mean L* over seconds [T - 10, T) after the opening for T >= 20, and over [T - 1, T) for T < 20.
Collapse rule: a scene is excluded from an arm if either version has a 1-s mean L* below 3 or above 97 in [0, 30) s.
Primary (Holm over 3; per-scene values, two-sided one-sample t-tests, scene-bootstrap 95% CIs; supported if Holm
p < .05 and the estimate is on the predicted side):
  H1 the released window forgets:            alpha(mgw, 30) - alpha(mgw, 5) < 0
  H2 full history keeps more than the window: alpha(mgf, 30) - alpha(mgw, 30) > 0
  H3 full history keeps a share:              alpha(mgf, 30) > 0
Secondary: alpha at 1, 2, 5, 10, 20, 30 s per arm; frozen videos: a video counts as frozen if the mean absolute
frame-to-frame change of its mean L* over [20, 30) s is below 0.01, reported per arm; H1-H3 without frozen scenes;
attractor offset per arm: mean L* of fresh starts over [20, 30) s minus over [0, 2) s; zero-drift level per arm from the
two versions (end = open crossing); mean |end - open| per arm.
usage: python analyze_magi.py RUN_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD = 24, 9
D, F = {}, {}
for p in glob.glob(f"{src}/*.npz"):
    b = os.path.basename(p)
    m = re.match(r"^(\w+?)_(mg[wf])__(m16|p16)_s0\.npz$", b)
    f = re.match(r"^rx02_fresh_(mg[wf])_s(\d)\.npz$", b)
    if m:
        D.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]
    elif f:
        F.setdefault(f[1], {})[int(f[2])] = np.load(p)["lab_mean"][:, 0]


def win(T):
    return (T - 10, T) if T >= 20 else (T - 1, T)


def collapsed(arm):
    bad = set()
    for s, v in D.get(arm, {}).items():
        for L in v.values():
            n = min(30 * FPS, len(L) - HOLD) // FPS
            sec = L[HOLD:HOLD + n * FPS].reshape(n, FPS).mean(1)
            if (sec < 3).any() or (sec > 97).any():
                bad.add(s)
    return bad


def frozen(L, hold=HOLD):
    seg = L[hold + 20 * FPS:hold + 30 * FPS]
    return bool(len(seg) > 1 and np.abs(np.diff(seg)).mean() < 0.01)


def alpha(arm, T, exclude=()):
    o, (t0, t1) = {}, win(T)
    for s, v in D.get(arm, {}).items():
        if len(v) < 2 or s in exclude:
            continue
        i0, i1 = HOLD + int(t0 * FPS), HOLD + int(t1 * FPS)
        if i1 > min(len(v["m16"]), len(v["p16"])):
            continue
        o[s] = float((v["p16"][i0:i1].mean() - v["m16"][i0:i1].mean()) / (v["p16"][1:HOLD].mean() - v["m16"][1:HOLD].mean()))
    return o


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def ttest(d):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def pdiff(x, y):
    return {s: x[s] - y[s] for s in x if s in y}


def tests_for(excl):
    return {"H1": (ttest(pdiff(alpha("mgw", 30, excl["mgw"]), alpha("mgw", 5, excl["mgw"]))), -1),
            "H2": (ttest(pdiff(alpha("mgf", 30, excl["mgf"]), alpha("mgw", 30, excl["mgw"]))), +1),
            "H3": (ttest(alpha("mgf", 30, excl["mgf"])), +1)}


def holm(tests):
    prim = {k: dict(t, side=s) for k, (t, s) in tests.items() if t is not None}
    running = 0.0
    for i, k in enumerate(sorted(prim, key=lambda k: prim[k]["p"] if np.isfinite(prim[k]["p"]) else 1.0)):
        running = max(running, min(1.0, (3 - i) * (prim[k]["p"] if np.isfinite(prim[k]["p"]) else 1.0)))
        prim[k]["p_holm"] = running
        prim[k]["supported"] = bool(running < 0.05 and np.sign(prim[k]["est"]) == prim[k]["side"])
    return prim


EXCL = {a: collapsed(a) for a in ("mgw", "mgf")}
FROZ = {a: {s for s, v in D.get(a, {}).items() if any(frozen(L) for L in v.values())} for a in ("mgw", "mgf")}
res = {"counts": {a: len(v) for a, v in D.items()}, "fresh_counts": {a: len(v) for a, v in F.items()},
       "excluded": {a: sorted(v) for a, v in EXCL.items()}, "frozen_scenes": {a: sorted(v) for a, v in FROZ.items()},
       "frozen_fraction": {a: float(np.mean([frozen(L) for v in D.get(a, {}).values() for L in v.values()]))
                           for a in D},
       "primary": holm(tests_for(EXCL)),
       "without_frozen": holm(tests_for({a: EXCL[a] | FROZ[a] for a in EXCL})),
       "alpha": {a: {T: dict(mean=float(np.mean(list(x.values()))), ci=ci(list(x.values())), n=len(x))
                     for T in (1, 2, 5, 10, 20, 30) if (x := alpha(a, T, EXCL[a]))} for a in D}}
res["fresh_offset"] = {a: dict(mean=float(np.mean([L[20 * FPS:30 * FPS].mean() - L[:2 * FPS].mean() for L in v.values()])),
                               n=len(v), frozen=int(sum(frozen(L, 0) for L in v.values()))) for a, v in F.items()}
res["zero_drift"], res["abs_end_minus_open"] = {}, {}
for a, sc in D.items():
    zd, ad = [], []
    for s, v in sc.items():
        if len(v) < 2 or s in EXCL[a]:
            continue
        x = np.array([v[k][1:HOLD].mean() for k in ("m16", "p16")])
        y = np.array([v[k][HOLD + 20 * FPS:HOLD + 30 * FPS].mean() for k in ("m16", "p16")])
        b1 = (y[1] - y[0]) / (x[1] - x[0])
        if b1 < 0.95:
            zd.append(float((y[0] - b1 * x[0]) / (1 - b1)))
        ad += list(np.abs(y - x))
    res["zero_drift"][a] = dict(median=float(np.median(zd)), n=len(zd)) if zd else None
    res["abs_end_minus_open"][a] = float(np.mean(ad)) if ad else None
json.dump(res, open(f"{out}/magi_results.json", "w"), indent=1, default=float)
for k, t in sorted(res["primary"].items()):
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4f}",
          "supported" if t["supported"] else "not supported")
print({a: {T: round(v["mean"], 3) for T, v in x.items()} for a, x in res["alpha"].items()})
print("frozen", res["frozen_fraction"], "offset", res["fresh_offset"], "excluded", res["excluded"])
