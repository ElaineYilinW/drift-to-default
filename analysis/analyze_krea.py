"""Study 22: Krea Realtime 14B (Self Forcing distilled from Wan2.1-14B; KV-cache recomputation over a 3-latent-frame
window) -- is the opening forgotten, and does the release's first-frame anchor keep a share? (preregistered,
docs/preregistrations/22_krea.md).

A  tone: files {scene}_{arm}__{m16|p16}_s0.npz (16 fps; frames 0..8 are the held opening), arm in kr60_none (released
   default), kr60_first (keep_first_frame). alpha(T) = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L* of
   frames 1..8, end = mean L* over seconds [T - 10, T) after the opening.
B  content: files {opening}_mis_{arm}_s0.clip.npz for the 24 openings ct{cls}{k} of preregistration 09 (generic city
   prompt) and fresh_{arm}_s{0..7}.clip.npz (clip_probe.py). Class score z = scale * (cos(frame, class label) -
   cos(frame, city label)); s[t0, t1) = mean z over frames in seconds [t0, t1) after the opening (frame f -> (f - 9)/16 s);
   s_open = mean z over frames 1..8; mu = mean over the fresh starts of the same window score for that class;
   kappa = (s - mu) / (s_open - mu), undefined when s_open - mu < 2.
Primary (Holm over 5; per-scene / per-opening values, two-sided one-sample t-tests; supported if Holm p < .05 and the
estimate is on the predicted side):
  A1 the released window forgets the opening:  alpha(kr60_none, 60) - alpha(kr60_none, 10) < 0
  A2 the first-frame anchor keeps more:        alpha(kr60_first, 60) - alpha(kr60_none, 60) > 0
  A3 the anchor keeps a share:                 alpha(kr60_first, 60) > 0
  B1 content relaxes under the release:        s[55, 60)(kr60_none) - s_open < 0
  B2 the anchor keeps more content:            kappa[55, 60)(kr60_first) - kappa[55, 60)(kr60_none) > 0
95% CIs: bootstrap, 10,000 draws (A: scenes; B: openings within class and fresh seeds).
Secondary: alpha(T) at T = 10..60 and 1-s curves per arm; A1-A3 in [40, 50) s; kappa curves (T = 5..60) per arm;
B on tone-normalised embeddings; top-1 accuracy of the opening's class in [55, 60); per-class kappa; the mean L* of
fresh starts per arm (the default); collapse check: number of scenes with a 1-s mean L* below 3 or above 97.
usage: python analyze_krea.py A_DIR B_DIR TEXT_NPZ OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

a_dir, b_dir, text_npz, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD = 16, 9
ARMS = ("kr60_none", "kr60_first")


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def ttest(d, ci_fn=None):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=ci_fn() if ci_fn else ci(v), p=float(stats.ttest_1samp(v, 0).pvalue),
                n=int(len(v)))


# ---- A: tone -------------------------------------------------------------------------------------------------------
pat = re.compile(r"^(\w+?)_(kr60_none|kr60_first)__(m16|p16)_s0\.npz$")
L = {}
for p in glob.glob(f"{a_dir}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        L.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]


def a_one(v, t0, t1):
    i0, i1 = HOLD + int(t0 * FPS), HOLD + int(t1 * FPS)
    if len(v) < 2 or i1 > min(len(v["m16"]), len(v["p16"])):
        return None
    return float((v["p16"][i0:i1].mean() - v["m16"][i0:i1].mean()) / (v["p16"][1:HOLD].mean() - v["m16"][1:HOLD].mean()))


def alpha(arm, t0, t1):
    o = {s: a_one(v, t0, t1) for s, v in L.get(arm, {}).items()}
    return {s: x for s, x in o.items() if x is not None}


def pdiff(x, y):
    return {s: x[s] - y[s] for s in x if s in y}


def collapsed(arm):
    n = 0
    for v in L.get(arm, {}).values():
        for x in v.values():
            sec = x[HOLD:HOLD + (len(x) - HOLD) // FPS * FPS].reshape(-1, FPS).mean(1)
            if (sec < 3).any() or (sec > 97).any():
                n += 1
                break
    return n


# ---- B: content ----------------------------------------------------------------------------------------------------
TX = np.load(text_npz)
LAB, TEXT, SCALE = [str(x) for x in TX["labels"]], TX["text"].astype(np.float32), float(TX["scale"])
cpat = re.compile(r"^ct([a-z]+)(\d)_mis_(kr60_none|kr60_first)_s0\.clip\.npz$")
fpat = re.compile(r"^fresh_(kr60_none|kr60_first)_s(\d+)\.clip\.npz$")
WINS = [(t - 5, t) for t in range(5, 61)]


def wmean(idx, z, t0, t1):
    m = (idx >= HOLD + FPS * t0) & (idx < HOLD + FPS * t1)
    return z[m].mean(0) if m.any() else None


V, F = {}, {}
for p in sorted(glob.glob(f"{b_dir}/*.clip.npz")):
    b, d = os.path.basename(p), np.load(p)
    m, fm = cpat.match(b), fpat.match(b)
    if m:
        cls, rec = m[1], {"cls": m[1]}
        for key in ("emb", "emb_tn"):
            if key in d:
                idx, z = d["idx"], SCALE * (d[key].astype(np.float32) @ TEXT.T)
                lo = z[:, LAB.index(cls)] - z[:, LAB.index("city")]
                top = (z.argmax(1) == LAB.index(cls)).astype(float)
                rec[key] = dict(open=float(lo[(idx >= 1) & (idx < HOLD)].mean()),
                                win={w: (None if (x := wmean(idx, lo[:, None], *w)) is None else float(x[0])) for w in WINS},
                                top=None if (x := wmean(idx, top[:, None], 55, 60)) is None else float(x[0]))
        V.setdefault(m[3], {})[f"{cls}{m[2]}"] = rec
    elif fm:
        idx, z = d["idx"], SCALE * (d["emb"].astype(np.float32) @ TEXT.T)
        F.setdefault(fm[1], {})[int(fm[2])] = {w: wmean(idx, z, *w) for w in WINS}


def mu(arm, cls, w, seeds):
    ci_, cc = LAB.index(cls), LAB.index("city")
    v = [F[arm][s][w][ci_] - F[arm][s][w][cc] for s in seeds if F[arm][s].get(w) is not None]
    return float(np.mean(v)) if v else None


def kap(arm, w=(55, 60), ops=None, seeds=None, key="emb"):
    seeds = sorted(F.get(arm, {})) if seeds is None else seeds
    o_ = {}
    for o, r in V.get(arm, {}).items():
        if ops is not None and o not in ops or key not in r or r[key]["win"].get(w) is None:
            continue
        m_ = mu(arm, r["cls"], w, seeds)
        if m_ is None or r[key]["open"] - m_ < 2:
            continue
        o_[o] = (r[key]["win"][w] - m_) / (r[key]["open"] - m_)
    return o_


def drop(arm, ops=None, key="emb"):
    return {o: r[key]["win"][(55, 60)] - r[key]["open"] for o, r in V.get(arm, {}).items()
            if (ops is None or o in ops) and key in r and r[key]["win"].get((55, 60)) is not None}


ALL_O = sorted({o for c in V.values() for o in c})
CLASSES = sorted({o[:-1] for o in ALL_O})


def boot_b(fn, n=10000):
    by = {c: [o for o in ALL_O if o[:-1] == c] for c in CLASSES}
    vals = []
    for _ in range(n):
        ops = [o for c in CLASSES for o in rng.choice(by[c], len(by[c]))]
        sd = {c: list(rng.choice(sorted(F[c]), len(F[c]))) for c in F}
        d = fn(set(ops), sd)
        v = [d[o] for o in ops if o in d]
        if v:
            vals.append(np.mean(v))
    return np.percentile(vals, [2.5, 97.5]).tolist() if vals else None


def b2(ops=None, sd=None, key="emb"):
    x = kap("kr60_first", ops=ops, seeds=None if sd is None else sd["kr60_first"], key=key)
    y = kap("kr60_none", ops=ops, seeds=None if sd is None else sd["kr60_none"], key=key)
    return {o: x[o] - y[o] for o in x if o in y}


def b1(ops=None, sd=None, key="emb"):
    return drop("kr60_none", ops, key)


def tests(t0=50, t1=60):
    A = {a: alpha(a, t0, t1) for a in ARMS}
    t = {"A1": (ttest(pdiff(A["kr60_none"], alpha("kr60_none", 0, 10))), -1),
         "A2": (ttest(pdiff(A["kr60_first"], A["kr60_none"])), +1),
         "A3": (ttest(A["kr60_first"]), +1)}
    return t


prim = tests()
prim["B1"] = (ttest(b1(), lambda: boot_b(b1)) if V else None, -1)
prim["B2"] = (ttest(b2(), lambda: boot_b(b2)) if V else None, +1)
P = {k: dict(x, side=s) for k, (x, s) in prim.items() if x is not None}
running = 0.0
for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
    running = max(running, min(1.0, (5 - i) * P[k]["p"]))
    P[k]["p_holm"] = running
    P[k]["supported"] = bool(running < 0.05 and np.sign(P[k]["est"]) == P[k]["side"])
res = {"counts": {a: len(L.get(a, {})) for a in ARMS}, "content_counts": {a: len(V.get(a, {})) for a in ARMS},
       "fresh_counts": {a: len(F.get(a, {})) for a in ARMS}, "primary": P,
       "primary_missing": sorted(set(prim) - set(P)),
       "window_40_50": {k: x for k, (x, _) in tests(40, 50).items()},
       "alpha": {a: {T: dict(mean=float(np.mean(list(x.values()))), ci=ci(list(x.values())), n=len(x))
                     for T in (10, 20, 30, 40, 50, 60) if (x := alpha(a, T - 10, T))} for a in ARMS},
       "alpha_1s": {a: {t: float(np.mean(list(x.values()))) for t in range(1, 61) if (x := alpha(a, t - 1, t))}
                    for a in ARMS},
       "collapsed_scenes": {a: collapsed(a) for a in ARMS},
       "kappa": {a: {t: float(np.mean(list(k.values()))) for t in range(5, 61) if (k := kap(a, (t - 5, t)))} for a in ARMS},
       "B_tone_normalised": {"B1": ttest(b1(key="emb_tn")), "B2": ttest(b2(key="emb_tn"))} if V else None,
       "top1_55_60": {a: float(np.mean([r["emb"]["top"] for r in V.get(a, {}).values() if r.get("emb", {}).get("top") is not None]))
                      for a in ARMS if V.get(a)},
       "per_class_kappa": {a: {c: float(np.mean([v for o, v in kap(a).items() if o[:-1] == c])) for c in CLASSES
                               if any(o[:-1] == c for o in kap(a))} for a in ARMS if V.get(a)}}
json.dump(res, open(f"{out}/krea_results.json", "w"), indent=1, default=float)
for k in sorted(P):
    t = P[k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
print("missing:", res["primary_missing"], "| alpha60:", {a: round(res["alpha"][a][60]["mean"], 3) for a in ARMS
                                                       if 60 in res["alpha"].get(a, {})})
