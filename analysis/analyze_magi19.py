"""Study 19: MAGI-1 scene content under the released window and full history, and the window at 60 s (preregistered,
docs/preregistrations/19_magi_content_horizon.md).

A  content (CLIP, as preregistration 09; 24 fps): files ct{class}{k}_{mis|mat}_{mgw|mgf}_s0.{npz,clip.npz} (frames 0..8
   are the held opening; frame f -> (f - 9) / 24 s) and fresh_{mgw|mgf}_s{0..7}.{npz,clip.npz} (text-to-video, no
   opening; frame f -> f / 24 s). Class score z = scale * (cos(frame, class label) - cos(frame, city label));
   s[t0, t1) = mean z over the frames in that window; s_open = mean z over frames 1..8 of the same video;
   mu[t0, t1) = mean over the fresh starts of the same window setting of that class's window score;
   kappa = (s - mu) / (s_open - mu), undefined when s_open - mu < 2.
B  horizon (L*, as preregistration 18): files {scene}_mgw60__{m16|p16}_s0.npz; alpha(T) = (end_p16 - end_m16) /
   (open_p16 - open_m16), open = mean L* of frames 1..8, end = mean L* over seconds [T - 10, T) after the opening.
   Collapse rule: a scene is excluded if either version has a 1-s mean L* below 3 or above 97 in [0, 60) s.
Primary (Holm over 4; per-opening or per-scene values, two-sided one-sample t-tests; supported if Holm p < .05 and the
estimate is on the predicted side; window W = [25, 30) s):
  A1 content relaxes under the released window: s(mis_mgw, W) - s_open < 0
  A2 full history keeps more content:            kappa(mis_mgf, W) - kappa(mis_mgw, W) > 0
  A3 the prompt sets the content default:        s(mat_mgw, W) - s(mis_mgw, W) > 0
  B1 the released window keeps forgetting:       alpha(mgw60, 60) - alpha(mgw60, 30) < 0
95% CIs: bootstrap (A: openings within class and fresh starts, 10,000 draws; B: scenes, 10,000 draws).
Secondary: kappa curves (windows [T - 5, T), T = 5..30) for mis_mgw and mis_mgf; kappa(mis_mgf, W) and
kappa(mis_mgw, W) against 0; A1-A3 on tone-normalised embeddings; kappa per class; top-1 accuracy of the opening's
class among the 7 labels in W per arm; alpha(mgw60, T) at T = 10, 20, 30, 40, 50, 60 and alpha(mgw60, 60) against 0;
frozen videos (mean absolute frame-to-frame change of mean L* below 0.01 over [20, 30) s, for B also [50, 60) s) per
arm, and A1-A3, B1 without them; attractor offset of the fresh starts ([20, 30) s minus [0, 2) s of mean L*).
usage: python analyze_magi19.py A_DIR TEXT_NPZ B_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

a_dir, text_npz, b_dir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD = 24, 9
W = (25, 30)
WINS = [(t - 5, t) for t in range(5, 31)]
TX = np.load(text_npz)
LAB, TEXT, SCALE = [str(x) for x in TX["labels"]], TX["text"].astype(np.float32), float(TX["scale"])
CLS = [c for c in LAB if c != "city"]
ARMS = ("mis_mgw", "mis_mgf", "mat_mgw")


def frozen(L, hold, t0=20, t1=30):
    seg = L[hold + t0 * FPS:hold + t1 * FPS]
    return bool(len(seg) > 1 and np.abs(np.diff(seg)).mean() < 0.01)


# ---- A: content ----------------------------------------------------------------------------------------------------
pat = re.compile(r"^ct([a-z]+)(\d)_(mis|mat)_(mgw|mgf)_s0\.clip\.npz$")
fpat = re.compile(r"^fresh_(mgw|mgf)_s(\d+)\.clip\.npz$")
V, F, FL, FROZ = {}, {}, {}, {}
for p in sorted(glob.glob(f"{a_dir}/*.clip.npz")):
    b = os.path.basename(p)
    m, fm = pat.match(b), fpat.match(b)
    if not (m or fm):
        continue
    d, hold = np.load(p), (HOLD if m else 0)
    idx = d["idx"]
    sec = (idx - hold) / FPS
    z = {k: SCALE * (d[k].astype(np.float32) @ TEXT.T) for k in ("emb", "emb_tn") if k in d}
    win = {k: {w: (v[(sec >= w[0]) & (sec < w[1])].mean(0) if ((sec >= w[0]) & (sec < w[1])).any() else None)
               for w in WINS} for k, v in z.items()}
    L = np.load(p.replace(".clip.npz", ".npz"))["lab_mean"][:, 0]
    if m:
        cls, o, arm = m[1], f"{m[1]}{m[2]}", f"{m[3]}_{m[4]}"
        ci_, cc = LAB.index(cls), LAB.index("city")
        rec = dict(cls=cls)
        for k in z:
            lo = z[k][:, ci_] - z[k][:, cc]
            sel = (sec >= W[0]) & (sec < W[1])
            rec[k] = dict(open=float(lo[(idx >= 1) & (idx < 9)].mean()),
                          win={w: (None if v is None else float(v[ci_] - v[cc])) for w, v in win[k].items()},
                          top1=float(np.mean(z[k][sel].argmax(1) == ci_)) if sel.any() else None)
        V.setdefault(arm, {})[o] = rec
        FROZ.setdefault(arm, {})[o] = frozen(L, HOLD)
    elif fm:
        F.setdefault(fm[1], {})[int(fm[2])] = win["emb"]
        FL.setdefault(fm[1], {})[int(fm[2])] = L
OPS = sorted({o for v in V.values() for o in v})


def mu(kv, cls, w, seeds):
    v = [F[kv][s][w][LAB.index(cls)] - F[kv][s][w][LAB.index("city")] for s in seeds
         if s in F.get(kv, {}) and F[kv][s].get(w) is not None]
    return float(np.mean(v)) if v else None


def kap(arm, w, ops=None, sd=None, key="emb", drop=()):
    kv = arm.split("_")[1]
    seeds = sorted(F.get(kv, {})) if sd is None else sd.get(kv, [])
    o_ = {}
    for o, r in V.get(arm, {}).items():
        if (ops is not None and o not in ops) or o in drop or key not in r or r[key]["win"].get(w) is None:
            continue
        m_ = mu(kv, r["cls"], w, seeds)
        if m_ is None or r[key]["open"] - m_ < 2:
            continue
        o_[o] = (r[key]["win"][w] - m_) / (r[key]["open"] - m_)
    return o_


def score(arm, ops=None, key="emb", drop=()):
    return {o: r[key]["win"][W] for o, r in V.get(arm, {}).items()
            if (ops is None or o in ops) and o not in drop and key in r and r[key]["win"].get(W) is not None}


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def boot_a(fn, n=10000):
    by = {c: [o for o in OPS if o[:-1] == c] for c in CLS if any(o[:-1] == c for o in OPS)}
    vals = []
    for _ in range(n):
        ops = [o for c in by for o in rng.choice(by[c], len(by[c]))]
        sd = {kv: list(rng.choice(sorted(F[kv]), len(F[kv]))) for kv in F}
        x = fn(set(ops), sd)
        v = [x[o] for o in ops if o in x]
        if v:
            vals.append(float(np.mean(v)))
    return np.percentile(vals, [2.5, 97.5]).tolist() if vals else None


def test_a(fn):
    """fn(ops, seeds) -> {opening: value}; t-test on the full sample, CI by bootstrap."""
    d = fn(None, None)
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=boot_a(fn), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def a1(key="emb", drop=()):
    return lambda ops, sd: {o: r[key]["win"][W] - r[key]["open"] for o, r in V.get("mis_mgw", {}).items()
                            if (ops is None or o in ops) and o not in drop and key in r and r[key]["win"].get(W) is not None}


def a2(key="emb", drop=()):
    def f(ops, sd):
        x, y = kap("mis_mgf", W, ops, sd, key, drop), kap("mis_mgw", W, ops, sd, key, drop)
        return {o: x[o] - y[o] for o in x if o in y}
    return f


def a3(key="emb", drop=()):
    def f(ops, sd):
        x, y = score("mat_mgw", ops, key, drop), score("mis_mgw", ops, key, drop)
        return {o: x[o] - y[o] for o in x if o in y}
    return f


# ---- B: horizon ----------------------------------------------------------------------------------------------------
D = {}
for p in glob.glob(f"{b_dir}/*.npz"):
    m = re.match(r"^(\w+?)_mgw60__(m16|p16)_s0\.npz$", os.path.basename(p))
    if m:
        D.setdefault(m[1], {})[m[2]] = np.load(p)["lab_mean"][:, 0]


def collapsed_b():
    bad = set()
    for s, v in D.items():
        for L in v.values():
            n = min(60 * FPS, len(L) - HOLD) // FPS
            sec = L[HOLD:HOLD + n * FPS].reshape(n, FPS).mean(1)
            if (sec < 3).any() or (sec > 97).any():
                bad.add(s)
    return bad


def alpha(T, exclude=()):
    o, (t0, t1) = {}, (T - 10, T)
    for s, v in D.items():
        if len(v) < 2 or s in exclude:
            continue
        i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
        if i1 > min(len(v["m16"]), len(v["p16"])):
            continue
        o[s] = float((v["p16"][i0:i1].mean() - v["m16"][i0:i1].mean()) / (v["p16"][1:HOLD].mean() - v["m16"][1:HOLD].mean()))
    return o


def ttest(d):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def b1(exclude):
    x, y = alpha(60, exclude), alpha(30, exclude)
    return ttest({s: x[s] - y[s] for s in x if s in y})


def holm(tests):
    prim = {k: dict(t, side=s) for k, (t, s) in tests.items() if t is not None}
    running = 0.0
    pv = {k: (prim[k]["p"] if np.isfinite(prim[k]["p"]) else 1.0) for k in prim}
    for i, k in enumerate(sorted(prim, key=pv.get)):
        running = max(running, min(1.0, (4 - i) * pv[k]))
        prim[k]["p_holm"] = running
        prim[k]["supported"] = bool(running < 0.05 and np.sign(prim[k]["est"]) == prim[k]["side"])
    return prim


EXB = collapsed_b()
FROZ_B = {s for s, v in D.items() if any(frozen(L, HOLD) or frozen(L, HOLD, 50, 60) for L in v.values())}
FROZ_A = {o for arm in ARMS for o, f in FROZ.get(arm, {}).items() if f}
tests = {"A1": (test_a(a1()), -1), "A2": (test_a(a2()), +1), "A3": (test_a(a3()), +1), "B1": (b1(EXB), -1)}
res = {"counts": {a: len(V.get(a, {})) for a in ARMS}, "fresh_counts": {k: len(v) for k, v in F.items()},
       "b_scenes": len(D), "b_excluded": sorted(EXB), "primary": holm(tests),
       "primary_missing": sorted(k for k, (t, _) in tests.items() if t is None)}
res["without_frozen"] = holm({"A1": (test_a(a1(drop=FROZ_A)), -1), "A2": (test_a(a2(drop=FROZ_A)), +1),
                              "A3": (test_a(a3(drop=FROZ_A)), +1), "B1": (b1(EXB | FROZ_B), -1)})
res["frozen"] = {"A": {a: sorted(o for o, f in FROZ.get(a, {}).items() if f) for a in ARMS},
                 "A_fraction": {a: float(np.mean(list(FROZ[a].values()))) for a in ARMS if FROZ.get(a)},
                 "fresh": {k: int(sum(frozen(L, 0) for L in v.values())) for k, v in FL.items()},
                 "B": sorted(FROZ_B)}
res["kappa"] = {a: {w[1]: float(np.mean(list(k.values()))) for w in WINS if (k := kap(a, w))} for a in ("mis_mgw", "mis_mgf")}
res["kappa_W"] = {a: test_a(lambda ops, sd, a=a: kap(a, W, ops, sd)) for a in ("mis_mgw", "mis_mgf")}
res["tone_normalised"] = {"A1": test_a(a1("emb_tn")), "A2": test_a(a2("emb_tn")), "A3": test_a(a3("emb_tn"))}
res["per_class"] = {a: {c: float(np.mean([v for o, v in kap(a, W).items() if o[:-1] == c])) for c in CLS
                        if any(o[:-1] == c for o in kap(a, W))} for a in ("mis_mgw", "mis_mgf")}
res["top1"] = {a: float(np.mean([r["emb"]["top1"] for r in V[a].values() if r["emb"]["top1"] is not None]))
               for a in ARMS if V.get(a)}
res["alpha_b"] = {T: dict(mean=float(np.mean(list(x.values()))), ci=ci(list(x.values())), n=len(x))
                  for T in (10, 20, 30, 40, 50, 60) if (x := alpha(T, EXB))}
res["alpha_b_60_vs_0"] = ttest(alpha(60, EXB))
res["fresh_offset"] = {k: dict(mean=float(np.mean([L[20 * FPS:30 * FPS].mean() - L[:2 * FPS].mean() for L in v.values()])),
                               n=len(v)) for k, v in FL.items()}
json.dump(res, open(f"{out}/magi19_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4f}",
          "supported" if t["supported"] else "not supported")
print("missing:", res["primary_missing"], "| frozen A:", res["frozen"]["A_fraction"], "B:", res["frozen"]["B"])
