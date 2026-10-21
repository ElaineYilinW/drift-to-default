"""Main-experiment analysis, frozen with docs/preregistration_cn.md.

Inputs per video in SRC: {prompt}_{strategy}_s{seed}.npz (gen.py colour stats) and .feat.npz (features.py).
Metrics: M1 colour drift D, M2 DINOv2 distribution drift K (both per 1-s segment, endpoint = seconds 29-31),
signed z-scores, prior-work scalars. Tests: mixed model y ~ strategy + (1|prompt) + (1|prompt:seed), Wald contrasts,
Holm within the primary and secondary families; prompt/seed-nested bootstrap CIs as robustness.
usage: python analyze_main.py SRC OUT
"""
import glob
import json
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, t as tdist
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
FPS, K_NN, END = 16, 5, (29, 31)
rng = np.random.default_rng(0)
pat = re.compile(r"(.+)_(ref|w\d+s\d+r?)_s(\d+)\.npz$")

# ---------- load ----------
col, feat = {}, {}
for p in glob.glob(f"{src}/*.npz"):
    b = os.path.basename(p)
    if b.endswith(".feat.npz"):
        m = pat.match(b.replace(".feat.npz", ".npz"))
        if m:
            feat[(m[1], m[2], int(m[3]))] = p
    else:
        m = pat.match(b)
        if m:
            col[(m[1], m[2], int(m[3]))] = p
prompts = sorted({k[0] for k in col})
strategies = sorted({k[1] for k in col if k[1] != "ref"})
setname = {}
if os.path.exists(f"{src}/../prompts.jsonl"):
    setname = {r["id"]: r["set"] for r in map(json.loads, open(f"{src}/../prompts.jsonl"))}
setof = lambda p: setname.get(p, "driving" if p.startswith(("drv", "drive")) else "moviegen")
counts = {"videos_colour": sum(k[1] != "ref" for k in col), "videos_feat": sum(k[1] != "ref" for k in feat),
          "ref_colour": sum(k[1] == "ref" for k in col), "ref_feat": sum(k[1] == "ref" for k in feat),
          "prompts": len(prompts), "strategies": strategies}
print(counts)


def colour_feats(path):
    z = np.load(path)
    return np.column_stack([z["lab_mean"], z["sat_mean"], np.log(z["sharpness"] + 1)]), z["lab_mean"], z["hue_hist"].astype(float)


def unit(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def knn(e, pool):
    return np.sort(1 - e @ pool.T, axis=1)[:, :K_NN].mean(1)


rows, scal = [], []
for p in prompts:
    # M1 reference
    rc = {k[2]: colour_feats(v)[0] for k, v in col.items() if k[0] == p and k[1] == "ref"}
    x = np.vstack(list(rc.values()))
    mu, cov = x.mean(0), np.cov(x.T) + 1e-6 * np.eye(x.shape[1])
    icov, sd = np.linalg.inv(cov), np.sqrt(np.diag(cov))
    # M2 reference
    rf = {k[2]: unit(np.load(v)["cls"]) for k, v in feat.items() if k[0] == p and k[1] == "ref"}
    pool = np.vstack(list(rf.values())) if rf else None
    k_med = np.median(np.concatenate([knn(r, np.vstack([v for t, v in rf.items() if t != s])) for s, r in rf.items()])) if len(rf) > 1 else np.nan
    for (pp, strat, seed), path in col.items():
        if pp != p or strat == "ref":
            continue
        f, lab, hue = colour_feats(path)
        d = f - mu
        D = np.sqrt(np.einsum("ti,ij,tj->t", d, icov, d))
        z = d / sd
        sec = np.arange(len(f)) / FPS
        Kf, ksec, mq, msec = None, None, None, None
        if (pp, strat, seed) in feat and pool is not None:
            fz = np.load(feat[(pp, strat, seed)])
            Kf, ksec = knn(unit(fz["cls"]), pool) / k_med, fz["dino_idx"] / FPS
            mq, msec = fz["musiq"], fz["musiq_idx"] / FPS
        for g in range(int(sec.max())):
            sl = (sec >= g) & (sec < g + 1)
            r = dict(prompt=p, set=setof(p), strategy=strat, seed=seed, sec=g, D=D[sl].mean(),
                     z_L=z[sl, 0].mean(), z_a=z[sl, 1].mean(), z_b=z[sl, 2].mean(), z_sat=z[sl, 3].mean(), z_sharp=z[sl, 4].mean())
            if Kf is not None:
                r["K"] = Kf[(ksec >= g) & (ksec < g + 1)].mean()
                r["musiq"] = mq[(msec >= g) & (msec < g + 1)].mean()
            rows.append(r)
        s = dict(prompt=p, set=setof(p), strategy=strat, seed=seed, dE_first_last=float(np.linalg.norm(lab[-1] - lab[0])),
                 idf_color=float(100 * (1 - 0.5 * np.abs(hue[-1] - hue[0]).sum())))
        if mq is not None:
            s["musiq_dDrift"] = float(abs(mq[msec < 5].mean() - mq[msec >= msec.max() - 5].mean()))
            s["musiq_last5"] = float(mq[msec >= msec.max() - 5].mean())
        scal.append(s)
seg, scal = pd.DataFrame(rows), pd.DataFrame(scal)
seg.to_csv(f"{out}/segments.csv", index=False)
scal.to_csv(f"{out}/scalars.csv", index=False)
metrics = ["D", "K"] if "K" in seg else ["D"]
end = seg[(seg.sec >= END[0]) & (seg.sec < END[1])].groupby(["prompt", "set", "strategy", "seed"], as_index=False).mean(numeric_only=True)
end.to_csv(f"{out}/endpoints.csv", index=False)

# ---------- contrasts ----------
CONTRASTS = {
    "C1_sink3_vs_0": {"w21s3": .5, "w21s0": -.5, "w12s3": .5, "w12s0": -.5},
    "C2_window_x_sink": {"w12s0": 1, "w21s0": -1, "w12s3": -1, "w21s3": 1},
    "C3_rerotate": {"w21s3r": .5, "w21s3": -.5, "w12s3r": .5, "w12s3": -.5},
    "C4_sink3_vs_1": {"w21s3": .5, "w21s1": -.5, "w12s3": .5, "w12s1": -.5},
}
DIRECTION = {"C1_sink3_vs_0": "less", "C2_window_x_sink": "greater"}  # pre-registered one-sided


def lmm_contrast(df, y, w):
    if not set(w) <= set(df.strategy):
        return None
    d = df[["prompt", "seed", "strategy", y]].dropna().copy()
    d["ps"] = d.prompt + "_" + d.seed.astype(str)
    m = smf.mixedlm(f"{y} ~ 0 + C(strategy)", d, groups=d["prompt"], re_formula="1", vc_formula={"ps": "0 + C(ps)"}).fit(reml=True)
    names = list(m.fe_params.index)
    L = np.array([w.get(re.search(r"\[(.+)\]", n)[1], 0) for n in names], float)
    est = float(L @ m.fe_params.values)
    se = float(np.sqrt(L @ m.cov_params().loc[names, names].values @ L))
    return dict(est=est, se=se, z=est / se, ci=[est - 1.96 * se, est + 1.96 * se],
                vc_prompt=float(m.cov_re.iloc[0, 0]), vc_seed=float(m.vcomp[0]), resid=float(m.scale))


def boot_contrast(df, y, w, B=2000):
    if not set(w) <= set(df.strategy):
        return None
    wide = df.pivot_table(index=["prompt", "seed"], columns="strategy", values=y)
    c = sum(wt * wide[s] for s, wt in w.items()).dropna().unstack("seed")
    tab, vals = c.to_numpy(), []
    for _ in range(B):
        ps = tab[rng.integers(0, len(tab), len(tab))]
        vals.append(np.nanmean([np.nanmean(r[rng.integers(0, len(r), len(r))]) for r in ps]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


res = {"counts": counts, "endpoint_seconds": END, "contrasts": {}}
fam = {"primary": [], "secondary": []}
for y in metrics:
    for cname, w in CONTRASTS.items():
        r = lmm_contrast(end, y, w)
        if r is None:
            continue
        r["boot_ci"] = boot_contrast(end, y, w)
        if cname in DIRECTION:
            r["p"] = float(norm.cdf(r["z"]) if DIRECTION[cname] == "less" else norm.sf(r["z"]))
            r["sided"] = DIRECTION[cname]
            fam["primary"].append((y, cname))
        else:
            r["p"] = float(2 * norm.sf(abs(r["z"])))
            r["sided"] = "two"
            fam["secondary"].append((y, cname))
        res["contrasts"][f"{y}:{cname}"] = r
# H4: darkening persists with sink (prompt-level one-sample t, one-sided)
if {"w21s3", "w12s3"} <= set(end.strategy):
    zl = end[end.strategy.isin(["w21s3", "w12s3"])].groupby("prompt").z_L.mean()
    tt = zl.mean() / (zl.std(ddof=1) / np.sqrt(len(zl)))
    res["contrasts"]["z_L:H4_dark_with_sink"] = dict(est=float(zl.mean()), t=float(tt), df=len(zl) - 1,
                                                    p=float(tdist.cdf(tt, len(zl) - 1)), sided="less")
    fam["secondary"].append(("z_L", "H4_dark_with_sink"))
for f, items in fam.items():
    if not items:
        continue
    keys = [f"{y}:{c}" for y, c in items]
    rej, padj, _, _ = multipletests([res["contrasts"][k]["p"] for k in keys], method="holm")
    for k, a, pa in zip(keys, rej, padj):
        r = res["contrasts"][k]
        r["family"], r["p_holm"] = f, float(pa)
        if f == "primary":
            # pre-registered: Holm-significant AND the two-sided 95% model CI excludes 0 in the predicted direction
            r["supported"] = bool(a and (r["ci"][1] < 0 if r["sided"] == "less" else r["ci"][0] > 0))

# ---------- descriptives ----------
res["endpoint_means"] = end.groupby("strategy")[metrics + ["z_L", "z_sat", "z_b"]].mean().round(3).to_dict()
res["endpoint_means_by_set"] = {st: g.groupby("strategy")[metrics].mean().round(3).to_dict()
                                for st, g in end.groupby("set")}
res["prior_metrics_mean"] = scal.groupby("strategy").mean(numeric_only=True).drop(columns="seed").round(3).to_dict()
if "musiq" in seg:
    res["musiq_vs_saturation_corr_endpoint"] = float(end[["musiq", "z_sat"]].corr().iloc[0, 1])
with open(f"{out}/results.json", "w") as fh:
    json.dump(res, fh, indent=1)

# ---------- figures ----------
groups = [("no sink", ["w21s0", "w12s0"]), ("sink 1", ["w21s1", "w12s1", "w21s1r", "w12s1r"]), ("sink 3", ["w21s3", "w12s3", "w21s3r", "w12s3r"])]
fig, axes = plt.subplots(len(metrics) + 2, 3, figsize=(16, 3.4 * (len(metrics) + 2)), sharex=True)
for i, y in enumerate(metrics + ["z_L", "z_sat"]):
    for j, (title, ss) in enumerate(groups):
        ax = axes[i, j]
        for s in ss:
            if s not in set(seg.strategy):
                continue
            g = seg[seg.strategy == s].groupby(["sec", "prompt"])[y].mean().groupby("sec")
            m, se = g.mean(), g.std() / np.sqrt(g.count())
            ax.plot(m.index + .5, m, label=s, ls="--" if s.endswith("r") else "-")
            ax.fill_between(m.index + .5, m - 1.96 * se, m + 1.96 * se, alpha=.15)
        ax.set_title(f"{y} — {title}"), ax.legend(fontsize=7)
for ax in axes[-1]:
    ax.set_xlabel("seconds")
fig.tight_layout()
fig.savefig(f"{out}/curves.png", dpi=100)
print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("est", "ci", "boot_ci", "p", "p_holm", "supported")}
                  for k, v in res["contrasts"].items()}, indent=1))
