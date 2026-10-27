"""Study 21: new seeds for the Self Forcing content results (preregistered, docs/preregistrations/21_content_seeds.md).

Files {opening}_mis_{cfg}_s{1|2|3}.clip.npz (clip_probe.py) for the 24 openings ct{cls}{k} of preregistration 09, generic
city prompt, Self Forcing DMD 60 s, cfg in sf60_s3 (sink 3), sf60_s3rr (sink 3 re-rotated), sf60_s0 (no sink); fresh
starts fresh_{cfg}_s{8..15}.clip.npz give each configuration's default. Frame f -> (f - 9) / 16 s after the opening.
Class score z = scale * (cos(frame, class label) - cos(frame, city label)); s[t0, t1) = mean z over frames in seconds
[t0, t1); s_open = mean z over frames 1..8; mu[t0, t1) = mean over the fresh seeds of the same window score for that
class; kappa[t0, t1) = (s - mu) / (s_open - mu), undefined when s_open - mu < 2. An opening's value is the mean over its
seeds. Burst contrast R = kappa[50, 55) - kappa[45, 50).
Primary (Holm over 4; per-opening values, two-sided one-sample t-tests; supported if Holm p < .05 and the estimate is on
the predicted side; window [55, 60)):
  H1 a sink keeps content:                   kappa(sf60_s3) - kappa(sf60_s0) > 0
  H2 re-rotation lowers it:                  kappa(sf60_s3rr) - kappa(sf60_s3) < 0
  H3 a re-rotated sink keeps content:        kappa(sf60_s3rr) - kappa(sf60_s0) > 0
  H4 the late content return is positional:  R(sf60_s3rr) - R(sf60_s3) < 0
95% CIs: 10,000 bootstrap draws resampling openings within class and fresh seeds.
Secondary: H1-H4 on tone-normalised embeddings; each seed's H1-H4 estimate; kappa curves (windows [T-5, T), T = 5..60)
per configuration; H1-H4 pooled with seed 0 of studies 9 and 16 (when SEED0_DIR is given: its openings and fresh starts,
mu from the fresh starts of each run); variance of kappa[55, 60) per configuration split into between-opening and
between-seed parts (one-way random-effects ICC).
usage: python analyze_s21.py RUN_DIR TEXT_NPZ OUT_DIR [SEED0_DIR]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, text_npz, out = sys.argv[1:4]
seed0_dir = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
TX = np.load(text_npz)
LAB, TEXT, SCALE = [str(x) for x in TX["labels"]], TX["text"].astype(np.float32), float(TX["scale"])
CFG = ["sf60_s3", "sf60_s3rr", "sf60_s0"]
pat = re.compile(r"^ct([a-z]+)(\d)_mis_(sf60_s3|sf60_s3rr|sf60_s0)_s(\d+)\.clip\.npz$")
fpat = re.compile(r"^fresh_(sf60_s3|sf60_s3rr|sf60_s0)_s(\d+)\.clip\.npz$")
WINS = [(t - 5, t) for t in range(5, 61)] + [(45, 50), (50, 55)]


def wmean(idx, z, t0, t1):
    m = (idx >= 9 + 16 * t0) & (idx < 9 + 16 * t1)
    return z[m].mean(0) if m.any() else None


def load(d, seeds, fresh_seeds):
    """V[cfg][opening][seed] = {key: {open, win}}, F[cfg][fresh seed] = {w: z (7,)}"""
    V, F = {}, {}
    for p in sorted(glob.glob(f"{d}/*.clip.npz")):
        b, x = os.path.basename(p), np.load(p)
        m, fm = pat.match(b), fpat.match(b)
        if m and int(m[4]) in seeds:
            cls, rec = m[1], {}
            for key in ("emb", "emb_tn"):
                if key in x:
                    idx, z = x["idx"], SCALE * (x[key].astype(np.float32) @ TEXT.T)
                    lo = z[:, LAB.index(cls)] - z[:, LAB.index("city")]
                    rec[key] = dict(open=float(lo[(idx >= 1) & (idx < 9)].mean()),
                                    win={w: (None if (v := wmean(idx, lo[:, None], *w)) is None else float(v[0]))
                                         for w in WINS})
            V.setdefault(m[3], {}).setdefault(f"{cls}{m[2]}", {})[int(m[4])] = rec
        elif fm and int(fm[2]) in fresh_seeds:
            idx, z = x["idx"], SCALE * (x["emb"].astype(np.float32) @ TEXT.T)
            F.setdefault(fm[1], {})[int(fm[2])] = {w: wmean(idx, z, *w) for w in WINS}
    return V, F


RUNS = [load(src, {1, 2, 3}, set(range(8, 16)))]
if seed0_dir:
    RUNS.append(load(seed0_dir, {0}, set(range(8))))


def mu(F, cfg, cls, w, fseeds):
    ci_, cc = LAB.index(cls), LAB.index("city")
    v = [F[cfg][s][w][ci_] - F[cfg][s][w][cc] for s in fseeds if F[cfg][s].get(w) is not None]
    return float(np.mean(v)) if v else None


def kap(cfg, w, ops=None, sd=None, key="emb", runs=(0,), seed=None):
    """{opening: kappa averaged over its seeds (all runs in `runs`)}; sd = {run: {cfg: fresh seeds}}"""
    acc = {}
    for r in runs:
        V, F = RUNS[r]
        fs = sorted(F.get(cfg, {})) if sd is None else sd[r][cfg]
        for o, by in V.get(cfg, {}).items():
            if ops is not None and o not in ops:
                continue
            for s, rec in by.items():
                if seed is not None and s != seed or key not in rec or rec[key]["win"].get(w) is None:
                    continue
                m_ = mu(F, cfg, o[:-1], w, fs)
                if m_ is None or rec[key]["open"] - m_ < 2:
                    continue
                acc.setdefault(o, []).append((rec[key]["win"][w] - m_) / (rec[key]["open"] - m_))
    return {o: float(np.mean(v)) for o, v in acc.items()}


def burst(cfg, **kw):
    k1, k0 = kap(cfg, (50, 55), **kw), kap(cfg, (45, 50), **kw)
    return {o: k1[o] - k0[o] for o in k1 if o in k0}


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


ALL_O = sorted({o for V, _ in RUNS for c in V.values() for o in c})
CLASSES = sorted({o[:-1] for o in ALL_O})


def boot(fn, runs, n=10000):
    by = {c: [o for o in ALL_O if o[:-1] == c] for c in CLASSES}
    vals = []
    for _ in range(n):
        ops = [o for c in CLASSES for o in rng.choice(by[c], len(by[c]))]
        sd = {r: {c: list(rng.choice(sorted(RUNS[r][1][c]), len(RUNS[r][1][c]))) for c in RUNS[r][1]} for r in runs}
        d = fn(set(ops), sd)
        v = [d[o] for o in ops if o in d]
        if v:
            vals.append(np.mean(v))
    return np.percentile(vals, [2.5, 97.5]).tolist() if vals else None


def make(stat, x, y, key="emb", runs=(0,), seed=None):
    def f(ops=None, sd=None):
        a = stat(x, ops=ops, sd=sd, key=key, runs=runs, seed=seed)
        b = stat(y, ops=ops, sd=sd, key=key, runs=runs, seed=seed)
        return {o: a[o] - b[o] for o in a if o in b}
    return f


def K(cfg, **kw):
    return kap(cfg, (55, 60), **kw)


def R(cfg, **kw):
    return burst(cfg, **kw)


HYP = {"H1": (K, "sf60_s3", "sf60_s0", +1), "H2": (K, "sf60_s3rr", "sf60_s3", -1), "H3": (K, "sf60_s3rr", "sf60_s0", +1),
       "H4": (R, "sf60_s3rr", "sf60_s3", -1)}


def run_tests(key="emb", runs=(0,), seed=None, with_ci=True):
    o = {}
    for h, (st, x, y, side) in HYP.items():
        f = make(st, x, y, key, runs, seed)
        d = f()
        if len(d) < 3:
            o[h] = None
            continue
        v = np.array(list(d.values()), float)
        o[h] = dict(est=float(v.mean()), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)), side=side)
        if with_ci:
            o[h]["ci"] = boot(f, runs)
    return o


def holm(t):
    prim = {k: x for k, x in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(prim, key=lambda k: prim[k]["p"])):
        running = max(running, min(1.0, (len(HYP) - i) * prim[k]["p"]))
        prim[k]["p_holm"] = running
        prim[k]["supported"] = bool(running < 0.05 and np.sign(prim[k]["est"]) == prim[k]["side"])
    return prim


def icc(cfg):
    V, F = RUNS[0]
    rows = []
    for o in V.get(cfg, {}):
        v = [kap(cfg, (55, 60), ops={o}, seed=s).get(o) for s in sorted(V[cfg][o])]
        v = [x for x in v if x is not None]
        if len(v) >= 2:
            rows.append(v)
    if len(rows) < 3:
        return None
    k = min(len(r) for r in rows)
    X = np.array([r[:k] for r in rows])
    n = len(X)
    msb = k * X.mean(1).var(ddof=1)
    msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return dict(icc=float((msb - msw) / (msb + (k - 1) * msw)), sd_between_seeds=float(np.sqrt(msw)),
                sd_opening_means=float(X.mean(1).std(ddof=1)), n_openings=n, seeds=k)


V0, F0 = RUNS[0]
res = {"counts": {c: {o: len(by) for o, by in V0.get(c, {}).items()} for c in CFG},
       "fresh_counts": {c: len(v) for c, v in F0.items()},
       "primary": holm(run_tests()),
       "tone_normalised": run_tests("emb_tn"),
       "per_seed": {s: run_tests(seed=s, with_ci=False) for s in (1, 2, 3)},
       "kappa": {c: {t: float(np.mean(list(k.values()))) for t in range(5, 61) if (k := kap(c, (t - 5, t)))}
                 for c in CFG},
       "icc_60": {c: icc(c) for c in CFG}}
if seed0_dir:
    res["pooled_with_seed0"] = run_tests(runs=(0, 1))
    res["seed0_only"] = run_tests(runs=(1,), with_ci=False)
json.dump(res, open(f"{out}/s21_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
print("kappa60:", {c: round(res["kappa"][c][60], 3) for c in CFG if 60 in res["kappa"].get(c, {})})
