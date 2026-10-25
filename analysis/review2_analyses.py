"""Exploratory analyses for the second internal review (not preregistered).

A  matched horizon: persistence over seconds [15, 20) after the opening for every configuration, from the per-frame L*
   of the shifted versions (cache models: study 10 atlas and study 7; FramePack: study 11; SkyReels: study 8;
   Matrix-Game: study 15, scenes without a black second before 20 s).
B  burst-aware 60-s persistence for the atlas: window [40, 50) (between bursts) and the mean of 1-s windows over
   [35, 60) (a full burst period), next to the preregistered [50, 60).
C  forgetting-only share alpha(60 s) / alpha(4.5 s).
D  transitions kept: prompt-bootstrap CIs of the preregistered statistic (median over valid control prompts of the
   per-prompt values in confirm_out / deepsink_out results.json).
usage: python review2_analyses.py ATLAS_DIR LONG60_PACK FP_DIR SKY_PACK MG_DIR RESULTS_DIR OUT_JSON"""
import glob
import json
import os
import re
import sys

import numpy as np

atlas, long60, fpdir, sky, mgdir, R, out = sys.argv[1:8]
rng = np.random.default_rng(0)


def ci(v, n=10000):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(n)], [2.5, 97.5]).tolist()


def slope(pairs):
    x, y = zip(*pairs)
    return np.polyfit(x, y, 1)[0]


def alpha_from(series, t0, t1):
    """series: {scene: {version: (open_L, L array, fps, hold)}} -> per-scene slope over [t0, t1) s."""
    o = {}
    for s, v in series.items():
        ks = [k for k in ("m16", "z0", "p16") if k in v]
        if "m16" not in ks or "p16" not in ks:
            continue
        pts = []
        for k in ks:
            op, L, fps, hold = v[k]
            i0, i1 = hold + int(t0 * fps), hold + int(t1 * fps)
            if i1 > len(L):
                break
            pts.append((op, L[i0:i1].mean()))
        if len(pts) == len(ks):
            o[s] = slope(pts)
    return o


def load_npz_dir(d, pat, fps, hold, open_from_json=False):
    S = {}
    for p in glob.glob(f"{d}/*.npz"):
        m = pat.match(os.path.basename(p))
        if not m:
            continue
        L = np.load(p)["lab_mean"][:, 0]
        if open_from_json:
            j = json.load(open(p[:-4] + ".json"))
            op, f, h = j["open_L_shifted"], j["fps"], j["hold_frames"]
        else:
            op, f, h = L[1:hold].mean(), fps, hold
        S.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = (op, L, f, h)
    return S


res = {"A_matched_20s": {}, "B_burst_aware_60s": {}, "C_forgetting_only": {}, "D_transitions_kept": {}}
AT = load_npz_dir(atlas, re.compile(r"^(\w+?)_(\w+?)__(m16|z0|p16)_s0\.npz$"), 16, 9)
z = np.load(long60)
LG = {}
for k in z.files:
    if "|" in k:
        continue
    m = re.match(r"^(\w+?)_(\w+?)__(m16|z0|p16)_s0$", k)
    if m:
        L = z[k][:, 0]
        LG.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = (L[1:9].mean(), L, 16, 9)
for c, sc in list(AT.items()) + [(c, s) for c, s in LG.items() if c.startswith("cv60")]:
    a = alpha_from(sc, 15, 20)
    res["A_matched_20s"][c] = dict(mean=float(np.mean(list(a.values()))), ci=ci(list(a.values())), n=len(a))
FP = load_npz_dir(fpdir, re.compile(r"^(\w+?)_(fp[vi]_\w+?)__(m16|p16)_s0\.npz$"), 30, 9, open_from_json=True)
for c, sc in FP.items():
    a = alpha_from(sc, 15, 20)
    res["A_matched_20s"][c] = dict(mean=float(np.mean(list(a.values()))), ci=ci(list(a.values())), n=len(a))
zs = np.load(sky)
SK = {}
for k in zs.files:
    if "|" in k:
        continue
    m = re.match(r"^(\w+?)_(skyc\d+)__(m16|p16)_s0$", k)
    if m:
        L = zs[k][:, 0]
        SK.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = (L[1:17].mean(), L, 24, 17)
for c, sc in SK.items():
    a = alpha_from(sc, 15, 20)
    res["A_matched_20s"][c] = dict(mean=float(np.mean(list(a.values()))), ci=ci(list(a.values())), n=len(a))
MG = load_npz_dir(mgdir, re.compile(r"^(\w+?)_(mg_\w+?)__(m16|p16)_s0\.npz$"), 25, 9, open_from_json=True)
black = {s for c in MG.values() for s, v in c.items() for k, (op, L, f, h) in v.items()
         if min(L[h + i * f:h + (i + 1) * f].mean() for i in range(20)) < 3}
for c, sc in MG.items():
    a = alpha_from({s: v for s, v in sc.items() if s not in black}, 15, 20)
    res["A_matched_20s"][c] = dict(mean=float(np.mean(list(a.values()))), ci=ci(list(a.values())), n=len(a))
for c, sc in AT.items():
    a50, a40 = alpha_from(sc, 50, 60), alpha_from(sc, 40, 50)
    per = [alpha_from(sc, t, t + 1) for t in range(35, 60)]
    avg = {s: float(np.mean([p[s] for p in per if s in p])) for s in a50}
    early = alpha_from(sc, 4, 5)
    res["B_burst_aware_60s"][c] = {k: dict(mean=float(np.mean(list(v.values()))), ci=ci(list(v.values())))
                                   for k, v in (("w50_60", a50), ("w40_50", a40), ("avg35_60", avg))}
    res["C_forgetting_only"][c] = float(np.mean([a50[s] / early[s] for s in a50 if s in early and early[s] > 0.2]))
for name in ("sf60_s3", "sf60_s3rr"):
    pass
d40 = alpha_from(AT["sf60_s3rr"], 40, 50)
d40b = alpha_from(AT["sf60_s3"], 40, 50)
dd = [d40[s] - d40b[s] for s in d40 if s in d40b]
res["B_rerotation_40_50"] = dict(est=float(np.mean(dd)), ci=ci(dd))
for f, arms in ((f"{R}/corr/confirm_out/results.json", ("ttc_s1", "ttc_s1_b15")), (f"{R}/corr/deepsink_out/results.json", ("s6", "s9"))):
    c = json.load(open(f))["controls"]
    for a in arms:
        k = np.array([c[a]["kept_by_prompt"][p] for p in c["valid"]])
        bs = [np.median(rng.choice(k, len(k))) for _ in range(10000)]
        res["D_transitions_kept"][a] = dict(median=float(np.median(k)), ci=np.percentile(bs, [2.5, 97.5]).tolist(),
                                            range=[float(k.min()), float(k.max())], n=int(len(k)))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: (v if k != "B_burst_aware_60s" else {c: {kk: round(x["mean"], 2) for kk, x in vv.items()} for c, vv in v.items()}) for k, v in res.items() if k != "A_matched_20s"}, indent=0, default=float)[:3000])
print({c: round(v["mean"], 2) for c, v in res["A_matched_20s"].items()})
