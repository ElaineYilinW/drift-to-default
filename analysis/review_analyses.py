"""Exploratory analyses of existing data prompted by an internal review (not preregistered).

A  drift decomposition per configuration: attractor offset (fresh end at 31 s minus fresh start) and forgetting
   (1 - alpha) * (opening - default); zero-drift level vs fresh end (slope, mean absolute difference).
B  study 7 (60 s) re-tested on the 18 scenes not in the 120-s subset that motivated it.
C  lighting study: per-prompt slopes of the change on the opening with opening-cluster bootstrap CIs, and a
   cluster-robust Wald test of the prompt x opening interaction (Self Forcing and LongLive).
D  SkyReels: share of scenes with alpha > 1, and persistence against the number of autoregressive extensions.
E  exponential fits alpha(t) = a_inf + (1 - a_inf) exp(-t / tau) from the preregistration-06 results.
(Intervals for the share of legitimate transitions kept are in review2_analyses.py, from the preregistered statistic.)
usage: python review_analyses.py LABPACK_DIR RESULTS_DIR OUT_JSON
"""
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

pack, R, out = sys.argv[1:4]
rng = np.random.default_rng(0)
res = {}


def ci(v, n=10000):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(n)], [2.5, 97.5]).tolist()


# A ------------------------------------------------------------------------------------------------------------------
rel = json.load(open(f"{R}/corr/relax_out/relax_results.json"))["per_cfg"]
A = {}
for c, d in rel.items():
    if "fresh" in d and d.get("alpha_L"):
        A[c] = dict(start=d["fresh"]["L_start"], end=d["fresh"]["L_end"], offset=d["fresh"]["L_end"] - d["fresh"]["L_start"],
                    alpha=d["alpha_L"]["mean"], zero=d.get("zero_L", {}).get("median"))
x = np.array([v["end"] for v in A.values() if v["zero"] is not None])
y = np.array([v["zero"] for v in A.values() if v["zero"] is not None])
res["A_decomposition"] = dict(per_cfg=A, zero_vs_fresh=dict(r=float(np.corrcoef(x, y)[0, 1]), slope=float(np.polyfit(x, y, 1)[0]),
                                                             mad=float(np.abs(x - y).mean()), maxad=float(np.abs(x - y).max())))

# B ------------------------------------------------------------------------------------------------------------------
z = np.load(f"{pack}/long60.npz")
L = {k: z[k][:, 0] for k in z.files if "|" not in k}
scenes = sorted({k.split("_")[0] for k in L})
motiv = {"rx00", "rx01", "rx02", "gx00", "gx01", "gx02"}


def slope60(scene, cfg):
    ks = [f"{scene}_{cfg}__{v}_s0" for v in ("m16", "z0", "p16")]
    if not all(k in L for k in ks):
        return None
    xx = np.array([L[k][1:9].mean() for k in ks])
    yy = np.array([L[k][9 + 50 * 16:9 + 60 * 16].mean() for k in ks])
    return np.polyfit(xx, yy, 1)[0]


B = {}
for name, (a, b) in {"SF": ("sf60_s3", "sf60_s0"), "CausVid": ("cv60_s3", "cv60_s0")}.items():
    for subset, ss in (("all24", scenes), ("indep18", [s for s in scenes if s not in motiv])):
        d = np.array([slope60(s, a) - slope60(s, b) for s in ss if slope60(s, a) is not None and slope60(s, b) is not None])
        B[f"{name}_{subset}"] = dict(est=float(d.mean()), ci=ci(d), p=float(stats.ttest_1samp(d, 0).pvalue), n=int(len(d)))
res["B_study7_independent_scenes"] = B

# C ------------------------------------------------------------------------------------------------------------------
CL = {}
for model, f in (("SF", "light"), ("LL", "ll_light")):
    z = np.load(f"{pack}/{f}.npz")
    rows = []
    for k in z.files:
        if "|" in k or "_none_s" not in k:
            continue
        op, rest = k.split("--")
        light, seed = rest.split("_none_s")
        Lk = z[k][:, 0]
        rows.append(dict(open=op, light=light, seed=int(seed), L_open=Lk[1:9].mean(), L_end=Lk[-32:].mean()))
    D = pd.DataFrame(rows)
    D["dL"] = D.L_end - D.L_open
    G = D.groupby(["open", "light"], as_index=False).mean(numeric_only=True)
    per = {}
    for lt, g in G.groupby("light"):
        b = np.polyfit(g.L_open, g.dL, 1)[0]
        opens = g.open.values
        bs = [np.polyfit(*zip(*[(r.L_open, r.dL) for o in rng.choice(opens, len(opens)) for r in g[g.open == o].itertuples()]), 1)[0]
              for _ in range(2000)]
        per[lt] = dict(slope=float(b), persistence=float(1 + b), ci_persistence=(1 + np.percentile(bs, [2.5, 97.5])).tolist(),
                       n=int(len(g)))
    full = smf.ols("dL ~ L_open * C(light)", G).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(G.open)[0]})
    names = [n for n in full.params.index if ":" in n]
    W = full.wald_test(" , ".join(f"{n} = 0" for n in names), scalar=True)
    CL[model] = dict(per_prompt=per, interaction_wald_p_cluster=float(W.pvalue), n_openings=int(G.open.nunique()))
res["C_lighting_interaction"] = CL

# D ------------------------------------------------------------------------------------------------------------------
sk = json.load(open(f"{R}/corr/skyconf_out/skyov_results.json"))
ps = sk["per_scene"]
a5, a17 = np.array([v[0] for v in ps.values()]), np.array([v[1] for v in ps.values()])
res["D_skyreels"] = dict(per_scene=ps, share_gt1_overlap17=float((a17 > 1).mean()), share_gt1_overlap5=float((a5 > 1).mean()),
                         note="extensions: 20 s at 24 fps with 97-frame segments; overlap 17 adds 80 frames "
                                            "per extension, overlap 5 adds 92; KV models add 12 frames (0.75 s) per chunk")
res["D_skyreels"]["extensions_20s"] = {"overlap17": int(np.ceil((480 + 17 - 97) / 80)), "overlap5": int(np.ceil((480 + 17 - 97) / 92)),
                                       "kv_chunks_20s": int(20 / 0.75)}

# E ------------------------------------------------------------------------------------------------------------------
res["E_exponential_fits_31s"] = {c: d.get("fit") for c, d in rel.items() if d.get("fit")}
json.dump(res, open(out, "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k in ("B_study7_independent_scenes",)}, indent=1))
print(json.dumps(res["A_decomposition"]["zero_vs_fresh"]), {m: (v["interaction_wald_p_cluster"], {k: round(x["persistence"], 2) for k, x in v["per_prompt"].items()}) for m, v in CL.items()})
