"""Pilot analysis: candidate drift metrics, strategy curves with prompt-bootstrap CIs, variance components, power.

Self-referenced colour drift: for each prompt, the reference distribution is the per-frame colour features of
fresh-start short videos (strategy "ref", many seeds). Drift of frame t = Mahalanobis distance of its features
from that reference (and signed z-scores per feature). Calibration: the same distance for held-out reference
seeds (leave-one-seed-out) gives the "normal" band.

usage: python analyze_pilot.py RESULTS_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
FEATS = ["L", "a", "b", "sat", "logsharp"]
SEG = 16  # frames per segment (1 s at 16 fps)
rng = np.random.default_rng(0)


def load(path):
    z = np.load(path)
    f = np.column_stack([z["lab_mean"], z["sat_mean"], np.log(z["sharpness"] + 1)])
    return f, z["lab_mean"], z["hue_hist"].astype(np.float64)


runs = {}
for path in sorted(glob.glob(f"{src}/*.npz")):
    m = re.match(r"(.+)_(ref|w\d+s\d+)_s(\d+)\.npz", os.path.basename(path))
    runs[(m[1], m[2], int(m[3]))] = load(path)
prompts = sorted({k[0] for k in runs})
strategies = sorted({k[1] for k in runs if k[1] != "ref"})
print(f"{len(runs)} runs, prompts {len(prompts)}, strategies {strategies}")


def ref_stats(feats):
    x = np.vstack(feats)
    mu, cov = x.mean(0), np.cov(x.T) + 1e-6 * np.eye(x.shape[1])
    return mu, np.sqrt(np.diag(cov)), np.linalg.inv(cov)


def maha(f, mu, icov):
    d = f - mu
    return np.sqrt(np.einsum("ti,ij,tj->t", d, icov, d))


rows, calib = [], []
for p in prompts:
    ref = {s: runs[k][0] for k in runs if k[0] == p and k[1] == "ref" for s in [k[2]]}
    mu, sd, icov = ref_stats(list(ref.values()))
    for s, f in ref.items():  # leave-one-seed-out calibration of the "normal" distance
        m2, _, ic2 = ref_stats([g for t, g in ref.items() if t != s])
        calib.append(maha(f, m2, ic2))
    for (pp, strat, seed), (f, lab, hue) in runs.items():
        if pp != p or strat == "ref":
            continue
        d, z = maha(f, mu, icov), (f - mu) / sd
        T = len(f)
        for g in range(T // SEG):
            sl = slice(g * SEG, (g + 1) * SEG)
            rows.append(dict(prompt=p, set="driving" if p.startswith("drive") else "moviegen", strategy=strat,
                             seed=seed, seg=g, t_sec=(g + 0.5) * SEG / 16, D=d[sl].mean(),
                             **{f"z_{k}": z[sl, i].mean() for i, k in enumerate(FEATS)}))
        # prior-work style scalars (first vs last), for comparability
        rows.append(dict(prompt=p, set="driving" if p.startswith("drive") else "moviegen", strategy=strat, seed=seed,
                         seg=-1, dE_first_last=float(np.linalg.norm(lab[-1] - lab[0])),
                         idf_color=float(100 * (1 - 0.5 * np.abs(hue[-1] - hue[0]).sum())),
                         sat_slope=float(np.polyfit(np.arange(T) / 16, f[:, 3], 1)[0])))
df = pd.DataFrame(rows)
seg = df[df.seg >= 0].copy()
scal = df[df.seg < 0].dropna(axis=1, how="all")
calib = np.concatenate(calib)
band = float(np.quantile(calib, 0.95))
print(f"reference leave-one-seed-out distance: median {np.median(calib):.2f}, 95% {band:.2f}")


def boot_ci(table, stat=np.mean, B=2000):
    """table: prompts x seeds array (nan allowed). Resample prompts, then seeds within prompt."""
    n_p = table.shape[0]
    vals = []
    for _ in range(B):
        ps = table[rng.integers(0, n_p, n_p)]
        ss = np.array([r[rng.integers(0, len(r), len(r))] for r in ps])
        vals.append(stat(np.nanmean(ss, axis=1)))
    return np.percentile(vals, [2.5, 97.5])


# 1. curves
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, metric in zip(axes, ["D", "z_L", "z_sat", "z_b"]):
    for strat in strategies:
        s = seg[seg.strategy == strat]
        mean = s.groupby("t_sec")[metric].mean()
        lo, hi = [], []
        for t in mean.index:
            tab = s[s.t_sec == t].pivot_table(index="prompt", columns="seed", values=metric).to_numpy()
            l, h = boot_ci(tab, B=300)
            lo.append(l), hi.append(h)
        ax.plot(mean.index, mean.values, label=strat)
        ax.fill_between(mean.index, lo, hi, alpha=0.2)
    if metric == "D":
        ax.axhline(band, color="k", ls="--", lw=1, label="reference 95%")
    ax.set_title(metric), ax.set_xlabel("seconds")
axes[0].legend()
fig.tight_layout()
fig.savefig(f"{out}/curves.png", dpi=110)

# 2. final 2 s: strategy means and paired differences vs w21s0
last = seg[seg.seg >= seg.seg.max() - 1].groupby(["prompt", "set", "strategy", "seed"], as_index=False)[["D", "z_L", "z_sat", "z_b"]].mean()
summary = {"reference_band95": band, "final": {}, "paired_vs_w21s0": {}, "by_set": {}}
for strat in strategies:
    tab = last[last.strategy == strat].pivot_table(index="prompt", columns="seed", values="D").to_numpy()
    summary["final"][strat] = dict(mean=float(np.nanmean(tab)), ci=boot_ci(tab).round(3).tolist(),
                                   frac_above_band=float((tab > band).mean()))
    for st in ["moviegen", "driving"]:
        summary["by_set"].setdefault(st, {})[strat] = float(last[(last.strategy == strat) & (last.set == st)].D.mean())
wide = last.pivot_table(index=["prompt", "seed"], columns="strategy", values="D")
for strat in strategies:
    if strat == "w21s0":
        continue
    diff = (wide[strat] - wide["w21s0"]).unstack("seed").to_numpy()
    summary["paired_vs_w21s0"][strat] = dict(mean=float(np.nanmean(diff)), ci=boot_ci(diff).round(3).tolist())

# 3. variance components of final-segment D, per strategy: prompt vs seed-within-prompt (= residual here)
vc = {}
for strat in strategies:
    s = last[last.strategy == strat]
    m = smf.mixedlm("D ~ 1", s, groups=s["prompt"]).fit(reml=True)
    vc[strat] = dict(prompt_var=float(m.cov_re.iloc[0, 0]), seed_var=float(m.scale))
summary["variance_components_final_D"] = vc

# 4. power for the paired sink effect (w21s3 - w21s0) as a function of #prompts (seeds fixed at 3 and 5)
diff = (wide["w21s3"] - wide["w21s0"]).rename("d").reset_index()
m = smf.mixedlm("d ~ 1", diff, groups=diff["prompt"]).fit()
sp2, se2, eff = float(m.cov_re.iloc[0, 0]), float(m.scale), float(m.params["Intercept"])
power = {}
for delta_frac in (1.0, 0.5, 0.25):
    for n_seed in (3, 5):
        for n_p in (10, 20, 40, 60):
            se = np.sqrt(sp2 / n_p + se2 / (n_p * n_seed))
            power[f"delta={delta_frac}x,seeds={n_seed},prompts={n_p}"] = round(float(norm.cdf(abs(eff * delta_frac) / se - 1.96)), 3)
summary["sink_effect_paired"] = dict(effect=eff, prompt_var=sp2, resid_var=se2)
summary["power_two_sided_0.05"] = power

# prior-work scalars
summary["prior_metrics_mean"] = scal.groupby("strategy")[["dE_first_last", "idf_color", "sat_slope"]].mean().round(3).to_dict()
# direction consistency of the colour shift at the end (drift should be systematic, content change idiosyncratic)
summary["sign_consistency_final"] = {strat: {k: float((last[last.strategy == strat][k] > 0).mean())
                                             for k in ["z_L", "z_sat", "z_b"]} for strat in strategies}

seg.to_csv(f"{out}/segments.csv", index=False)
scal.to_csv(f"{out}/scalars.csv", index=False)
with open(f"{out}/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
