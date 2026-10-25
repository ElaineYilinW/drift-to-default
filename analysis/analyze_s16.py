"""Study 16: the re-rotated sink and the 60-s horizon of other anchors (preregistered,
docs/preregistrations/16_rerotation_and_horizon.md).

A  content (CLIP, as preregistration 09): files {opening}_mis_{cfg}_s0.clip.npz and fresh_{cfg}_s{seed}.clip.npz,
   cfg in sf60_s3, sf60_s3rr, sf60_s0. Class score z = scale * (cos(frame, class label) - cos(frame, city label));
   s[t0, t1) = mean z over frames in seconds [t0, t1) after the held opening (frame f -> (f - 9) / 16 s);
   s_open = mean z over frames 1..8; mu[t0, t1) = mean over fresh seeds of the same window score for that class;
   kappa[t0, t1) = (s - mu) / (s_open - mu), undefined when s_open - mu < 2.
   Burst contrast of one video: R = kappa[50, 55) - kappa[45, 50) (the second positional burst of preregistration 10
   lies in [50, 54) s).
B  tone (as preregistration 10): files {scene}_{cfg}__{m16|p16}_s0.npz (16 fps, frames 0..8 held), cfg in sf60_s6,
   sf60_s6rr, sf60_s3rr. alpha(T) = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L* of frames 1..8,
   end = mean L* over seconds [T - 10, T) after the opening.
C  FramePack-F1 at 60 s: {scene}_{fpv_both|fpv_ctx}__{m16|p16}_s0.npz (30 fps, hold 9); alpha(T) as in B.
D  SkyReels-V2-DF at 60 s: {scene}_{sky5|sky17}__{m16|p16}_s0.npz (24 fps, hold 17, open = frames 1..16); alpha(T) as B.
   Collapse rule (C and D): a scene is excluded from an arm at endpoint T if either version has a 1-s mean L* below 3 or
   above 97 in [0, T) s after the opening. If more than 6 of the 12 scenes are excluded at T = 60, that arm's test uses
   the latest T in (50, 40, 30) at which at most 6 are excluded (for D, alpha(T) - alpha(20 s) at that T).
Primary (Holm over 9; per-opening / per-scene values, two-sided one-sample t-tests; supported if Holm p < .05 and the
estimate is on the predicted side):
  A1 the late content return is positional:   R(sf60_s3rr) - R(sf60_s3) < 0
  A2 re-rotation lowers content kept at 60 s: kappa[55, 60)(sf60_s3rr) - kappa[55, 60)(sf60_s3) < 0
  A3 a re-rotated sink still keeps content:   kappa[55, 60)(sf60_s3rr) - kappa[55, 60)(sf60_s0) > 0
  B1 re-rotation lowers a deep sink's share:  alpha(sf60_s6rr, 60) - alpha(sf60_s6, 60) < 0
  B2 depth adds persistence without bursts:   alpha(sf60_s6rr, 60) - alpha(sf60_s3rr, 60) > 0
  C1 FramePack keeps a share at 60 s:         alpha(fpv_both, 60) > 0
  C2 its context prefix alone keeps a share:  alpha(fpv_ctx, 60) > 0
  D1 a 5-frame overlap keeps forgetting:      alpha(sky5, 60) - alpha(sky5, 20) < 0
  D2 a 17-frame overlap keeps forgetting:     alpha(sky17, 60) - alpha(sky17, 20) < 0
95% CIs: bootstrap (A: openings within class and fresh seeds, 10,000 draws; B-D: scenes, 10,000 draws).
Secondary: kappa curves (windows [T-5, T), T = 5..60) per configuration; the first-burst contrast
kappa[31, 36) - kappa[26, 31); A1/A2 on tone-normalised embeddings; alpha(T) at T = 10..60 for every arm of B-D;
B1/B2 in the burst-free window [40, 50); the jump alpha([34, 35)) - alpha([31, 32)) per arm of B; FramePack
alpha(fpv_both, 60) - alpha(fpv_ctx, 60) and alpha(60) - alpha(20) per arm; SkyReels alpha(sky17, 60) - alpha(sky5, 60);
number of excluded scenes. (Reproduction of earlier runs with the same seed and opening is reported separately.)
usage: python analyze_s16.py A_DIR TEXT_NPZ B_DIR C_DIR D_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

a_dir, text_npz, b_dir, c_dir, d_dir, out = sys.argv[1:7]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)


def ci(v, n=10000):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(n)], [2.5, 97.5]).tolist()


def ttest(d):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


# ---- A: content ----------------------------------------------------------------------------------------------------
TX = np.load(text_npz)
LAB, TEXT, SCALE = [str(x) for x in TX["labels"]], TX["text"].astype(np.float32), float(TX["scale"])
ACFG = ["sf60_s3", "sf60_s3rr", "sf60_s0"]
pat = re.compile(r"^ct([a-z]+)(\d)_mis_(\w+)_s0\.clip\.npz$")
fpat = re.compile(r"^fresh_(\w+)_s(\d+)\.clip\.npz$")


def wmean(idx, z, t0, t1):
    m = (idx >= 9 + 16 * t0) & (idx < 9 + 16 * t1)
    return z[m].mean(0) if m.any() else None


WINS = [(t - 5, t) for t in range(5, 61)] + [(45, 50), (50, 55), (26, 31), (31, 36)]
V, F = {}, {}
for p in sorted(glob.glob(f"{a_dir}/*.clip.npz")):
    b, d = os.path.basename(p), np.load(p)
    m, fm = pat.match(b), fpat.match(b)
    if m and m[3] in ACFG:
        cls = m[1]
        rec = dict(cls=cls)
        for key in ("emb", "emb_tn"):
            if key not in d:
                continue
            idx, z = d["idx"], SCALE * (d[key].astype(np.float32) @ TEXT.T)
            lo = z[:, LAB.index(cls)] - z[:, LAB.index("city")]
            rec[key] = dict(open=float(lo[(idx >= 1) & (idx < 9)].mean()),
                            win={w: (None if (x := wmean(idx, lo[:, None], *w)) is None else float(x[0])) for w in WINS})
        V.setdefault(m[3], {})[f"{cls}{m[2]}"] = rec
    elif fm and fm[1] in ACFG:
        idx, z = d["idx"], SCALE * (d["emb"].astype(np.float32) @ TEXT.T)
        F.setdefault(fm[1], {})[int(fm[2])] = {w: wmean(idx, z, *w) for w in WINS}


def mu(cfg, cls, w, seeds):
    ci_, cc = LAB.index(cls), LAB.index("city")
    v = [F[cfg][s][w][ci_] - F[cfg][s][w][cc] for s in seeds if F[cfg][s].get(w) is not None]
    return float(np.mean(v)) if v else None


def kap(cfg, w, ops=None, seeds=None, key="emb"):
    seeds = sorted(F.get(cfg, {})) if seeds is None else seeds
    o_ = {}
    for o, r in V.get(cfg, {}).items():
        if ops is not None and o not in ops or key not in r or r[key]["win"].get(w) is None:
            continue
        m_ = mu(cfg, r["cls"], w, seeds)
        if m_ is None or r[key]["open"] - m_ < 2:
            continue
        o_[o] = (r[key]["win"][w] - m_) / (r[key]["open"] - m_)
    return o_


def burst(cfg, ops=None, seeds=None, key="emb", w1=(50, 55), w0=(45, 50)):
    k1, k0 = kap(cfg, w1, ops, seeds, key), kap(cfg, w0, ops, seeds, key)
    return {o: k1[o] - k0[o] for o in k1 if o in k0}


ALL_O = sorted({o for c in V.values() for o in c})
CLASSES = sorted({o[:-1] for o in ALL_O})


def boot_a(fn, n=10000):
    by = {c: [o for o in ALL_O if o[:-1] == c] for c in CLASSES}
    vals = []
    for _ in range(n):
        ops = [o for c in CLASSES for o in rng.choice(by[c], len(by[c]))]
        sd = {c: list(rng.choice(sorted(F[c]), len(F[c]))) for c in F}
        v = fn(ops, sd)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]).tolist() if vals else None


def mean_over(ops, d):
    v = [d[o] for o in ops if o in d]
    return float(np.mean(v)) if v else None


def diff_a(fx, fy, ops=None, sd=None):
    x, y = fx(ops, sd), fy(ops, sd)
    return {o: x[o] - y[o] for o in x if o in y}


def test_a(fx, fy):
    t = ttest(diff_a(fx, fy))
    if t is None:
        return None
    t["ci"] = boot_a(lambda ops, sd: mean_over(ops, diff_a(fx, fy, set(ops), sd)))
    return t


def K60(cfg, key="emb"):
    return lambda ops, sd: kap(cfg, (55, 60), None if ops is None else set(ops), None if sd is None else sd.get(cfg), key)


def RB(cfg, key="emb"):
    return lambda ops, sd: burst(cfg, None if ops is None else set(ops), None if sd is None else sd.get(cfg), key)


res = {"A": {"counts": {c: len(v) for c, v in V.items()}, "fresh_counts": {c: len(v) for c, v in F.items()},
             "kappa": {}, "burst2": {}, "burst1": {}}}
for c in ACFG:
    if c in V:
        res["A"]["kappa"][c] = {t: float(np.mean(list(k.values()))) for t in range(5, 61) if (k := kap(c, (t - 5, t)))}
        for nm, w1, w0 in (("burst2", (50, 55), (45, 50)), ("burst1", (31, 36), (26, 31))):
            b = burst(c, w1=w1, w0=w0)
            if b:
                res["A"][nm][c] = dict(mean=float(np.mean(list(b.values()))), ci=ci(list(b.values())), n=len(b))
tests = {"A1": (test_a(RB("sf60_s3rr"), RB("sf60_s3")), -1), "A2": (test_a(K60("sf60_s3rr"), K60("sf60_s3")), -1),
         "A3": (test_a(K60("sf60_s3rr"), K60("sf60_s0")), +1)}
res["A"]["tone_normalised"] = {"A1": test_a(RB("sf60_s3rr", "emb_tn"), RB("sf60_s3", "emb_tn")),
                               "A2": test_a(K60("sf60_s3rr", "emb_tn"), K60("sf60_s3", "emb_tn"))}


# ---- B-D: tone ------------------------------------------------------------------------------------------------------
def load_tone(src, tags, fps, hold):
    D = {}
    for p in glob.glob(f"{src}/*.npz"):
        m = re.match(r"^(\w+?)_(\w+?)__(m16|p16)_s0\.npz$", os.path.basename(p))
        if m and m[2] in tags:
            D.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]
    return dict(D=D, fps=fps, hold=hold)


def alpha(S, cfg, t0, t1, scenes=None):
    o, f, h = {}, S["fps"], S["hold"]
    for s, v in S["D"].get(cfg, {}).items():
        if len(v) < 2 or scenes is not None and s not in scenes:
            continue
        i0, i1 = h + int(round(t0 * f)), h + int(round(t1 * f))
        if i1 > len(v["m16"]) or i1 > len(v["p16"]):
            continue
        o[s] = float((v["p16"][i0:i1].mean() - v["m16"][i0:i1].mean()) / (v["p16"][1:h].mean() - v["m16"][1:h].mean()))
    return o


def collapsed(S, cfg, T):
    bad, f, h = set(), S["fps"], S["hold"]
    for s, v in S["D"].get(cfg, {}).items():
        for L in v.values():
            n = min(T * f, len(L) - h) // f
            sec = L[h:h + n * f].reshape(n, f).mean(1) if n else np.array([])
            if (sec < 3).any() or (sec > 97).any():
                bad.add(s)
    return bad


def endpoint(S, cfg):
    for T in (60, 50, 40, 30):
        ok = set(S["D"].get(cfg, {})) - collapsed(S, cfg, T)
        if len(S["D"].get(cfg, {})) - len(ok) <= 6:
            return T, ok
    return None, set()


def pdiff(x, y):
    return {s: x[s] - y[s] for s in x if s in y}


SB = load_tone(b_dir, ("sf60_s6", "sf60_s6rr", "sf60_s3rr"), 16, 9)
SC = load_tone(c_dir, ("fpv_both", "fpv_ctx"), 30, 9)
SD = load_tone(d_dir, ("sky5", "sky17"), 24, 17)
for k, S in (("B", SB), ("C", SC), ("D", SD)):
    res[k] = {"counts": {c: len(v) for c, v in S["D"].items()}, "alpha": {}}
    for c in S["D"]:
        res[k]["alpha"][c] = {T: dict(mean=float(np.mean(list(a.values()))), ci=ci(list(a.values())), n=len(a))
                              for T in range(10, 61, 10) if (a := alpha(S, c, T - 10, T))}
a60 = {c: alpha(SB, c, 50, 60) for c in SB["D"]}
tests["B1"] = (ttest(pdiff(a60.get("sf60_s6rr", {}), a60.get("sf60_s6", {}))), -1)
tests["B2"] = (ttest(pdiff(a60.get("sf60_s6rr", {}), a60.get("sf60_s3rr", {}))), +1)
a45 = {c: alpha(SB, c, 40, 50) for c in SB["D"]}
res["B"]["burst_free_40_50"] = {k: ttest(pdiff(a45[x], a45[y])) for k, (x, y) in
                                {"B1": ("sf60_s6rr", "sf60_s6"), "B2": ("sf60_s6rr", "sf60_s3rr")}.items()
                                if x in a45 and y in a45}
res["B"]["jump_31_34"] = {c: ttest(pdiff(alpha(SB, c, 34, 35), alpha(SB, c, 31, 32))) for c in SB["D"]}
for k, S, arms in (("C", SC, ("fpv_both", "fpv_ctx")), ("D", SD, ("sky5", "sky17"))):
    res[k]["endpoint"], res[k]["excluded_at_60"] = {}, {}
    for i, c in enumerate(arms):
        T, ok = endpoint(S, c)
        res[k]["endpoint"][c] = T
        res[k]["excluded_at_60"][c] = sorted(collapsed(S, c, 60))
        if T is None:
            tests[f"{k}{i + 1}"] = (None, -1 if k == "D" else +1)
            continue
        aT = alpha(S, c, T - 10, T, ok)
        if k == "C":
            tests[f"C{i + 1}"] = (ttest(aT), +1)
        else:
            d = pdiff(aT, alpha(S, c, 10, 20, ok))
            tests[f"D{i + 1}"] = (ttest(d), -1)
        res[k].setdefault("alpha60_minus_20", {})[c] = ttest(pdiff(aT, alpha(S, c, 10, 20, ok)))
c60 = {c: alpha(SC, c, 50, 60, endpoint(SC, c)[1]) for c in SC["D"]}
if len(c60) == 2:
    res["C"]["both_minus_ctx_60"] = ttest(pdiff(c60["fpv_both"], c60["fpv_ctx"]))
d60 = {c: alpha(SD, c, 50, 60, endpoint(SD, c)[1]) for c in SD["D"]}
if len(d60) == 2:
    res["D"]["sky17_minus_sky5_60"] = ttest(pdiff(d60["sky17"], d60["sky5"]))

# ---- Holm over the nine primary tests ---------------------------------------------------------------------------------
prim = {k: dict(t, side=s) for k, (t, s) in tests.items() if t is not None}
for t in prim.values():
    t["p"] = 1.0 if not np.isfinite(t["p"]) else t["p"]            # zero variance
running = 0.0
for i, k in enumerate(sorted(prim, key=lambda k: prim[k]["p"])):
    running = max(running, min(1.0, (9 - i) * prim[k]["p"]))
    prim[k]["p_holm"] = running
    prim[k]["supported"] = bool(running < 0.05 and np.sign(prim[k]["est"]) == prim[k]["side"])
res["primary"] = prim
res["primary_missing"] = sorted(set(tests) - set(prim))
json.dump(res, open(f"{out}/s16_results.json", "w"), indent=1, default=float)
for k in sorted(prim):
    t = prim[k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4f}",
          "supported" if t["supported"] else "not supported")
print("missing:", res["primary_missing"])
