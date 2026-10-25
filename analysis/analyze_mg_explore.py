"""Exploratory (not preregistered) sensitivity analysis of the world-model study: the preregistered 30-s endpoint lost
17 of 24 scenes to the black-collapse exclusion, mostly after 19 s. Here the endpoint is 15 s (window [10, 15) s), and
a scene is excluded if any of its videos has a black second (1-s mean L* < 3) within the first 15 s. Same alpha as
analyze_mg.py; also the share of videos that collapse to black before 30 s, and the onset times.
usage: python analyze_mg_explore.py RUN_DIR OUT_JSON"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

src, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
D, bad, onset = {}, set(), []
for p in glob.glob(f"{src}/*.npz"):
    b = os.path.basename(p)[:-4]
    s, c, v = b.split("_")[0], "_".join(b.split("__")[0].split("_")[1:]), b.split("__")[1].split("_")[0]
    j, L = json.load(open(p[:-4] + ".json")), np.load(p)["lab_mean"][:, 0]
    sec = np.array([L[9 + k * 25:9 + (k + 1) * 25].mean() for k in range(30)])
    if (sec < 3).any():
        onset.append(int(np.argmax(sec < 3)))
    if (sec[:15] < 3).any():
        bad.add(s)
    D.setdefault(c, {}).setdefault(s, {})[v] = (j["open_L_shifted"], L)


def alphas(c, T, w):
    o = {}
    for s, v in D[c].items():
        if s in bad or len(v) < 2:
            continue
        e = [v[k][1][9 + int((T - w) * 25):9 + int(T * 25)].mean() for k in ("m16", "p16")]
        o[s] = (e[1] - e[0]) / (v["p16"][0] - v["m16"][0])
    return o


def summ(d):
    x = np.array(list(d.values()))
    return dict(est=float(x.mean()), ci=np.percentile([rng.choice(x, len(x)).mean() for _ in range(10000)], [2.5, 97.5]).tolist(),
                p=float(stats.ttest_1samp(x, 0).pvalue), n=int(len(x)))


res = {"excluded_15s": sorted(bad), "videos_black_before_30s": len(onset), "videos": sum(len(sv) for c in D.values() for sv in c.values()),
       "black_onset_seconds": sorted(onset), "alpha": {}}
for c in D:
    res["alpha"][c] = {T: summ(alphas(c, T, 1 if T < 10 else 5)) for T in (2, 5, 10, 15)}
c15, c5, e15, b15 = alphas("mg_ctx", 15, 5), alphas("mg_ctx", 5, 1), alphas("mg_emb", 15, 5), alphas("mg_both", 15, 5)
res["ctx_fades"] = summ({s: c15[s] - c5[s] for s in c15 if s in c5})
res["emb_15s"] = summ(e15)
res["both_minus_ctx_15s"] = summ({s: b15[s] - c15[s] for s in b15 if s in c15})
res["additivity_15s"] = summ({s: c15[s] + e15[s] - b15[s] for s in b15 if s in c15 and s in e15})
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("alpha", "black_onset_seconds")}, indent=1))
