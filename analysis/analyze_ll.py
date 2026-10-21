"""Second model: is drift on LongLive (Wan2.1-1.3B + LoRA, window 12 + sink 3) also regression to a default brightness?

Same 40 held openings (20 real NAVSIM, 20 model-generated) and generic prompts as the Self Forcing real-opening runs,
no correction. Per model: dL_end = a + b * L_open on opening-level means, zero-drift level -a/b, opening-cluster
bootstrap CIs; joint model dL_end ~ L_open * model + source (real vs self).
usage: python analyze_ll.py SF_REAL SF_SELF LL_OPEN OUT_DIR
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

sf_dirs, ll_dir, out = sys.argv[1:3], sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
rows = []
for model, dirs in [("Self Forcing", sf_dirs), ("LongLive", [ll_dir])]:
    for d in dirs:
        for p in glob.glob(f"{d}/*_none_s*.npz"):
            m = re.match(r"((real|self)\d+)_none_s(\d+)\.npz", os.path.basename(p))
            if m:
                L = np.load(p)["lab_mean"][:, 0]
                rows.append(dict(model=model, open=m[1], src=m[2], seed=int(m[3]), L_open=L[1:9].mean(),
                                 dL_end=L[-32:].mean() - L[1:9].mean()))
D = pd.DataFrame(rows)
G = D.groupby(["model", "open", "src"], as_index=False).mean(numeric_only=True)
G.to_csv(f"{out}/ll_openings.csv", index=False)


def fit(g):
    b, a = np.polyfit(g.L_open, g.dL_end, 1)
    return b, -a / b


res = {"n_videos": D.groupby("model").size().to_dict(), "per_model": {}}
for model, g in G.groupby("model"):
    b, z = fit(g)
    bs = np.array([fit(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(2000)])
    a = -b * z
    res["per_model"][model] = dict(n_openings=int(len(g)), slope=float(b), slope_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                                   zero_drift_level=float(z), zero_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                                   r2=float(np.corrcoef(g.L_open, g.dL_end)[0, 1] ** 2), mean_abs_dL=float(g.dL_end.abs().mean()),
                                   resid_sd=float(np.std(g.dL_end - (a + b * g.L_open))),
                                   seed_sd=float(D[D.model == model].groupby("open").dL_end.std().mean()))
m = smf.ols("dL_end ~ L_open * C(model, Treatment('Self Forcing')) + src", G).fit()
res["joint"] = {k: [float(m.params[k]), float(m.pvalues[k])] for k in m.params.index}
json.dump(res, open(f"{out}/ll_summary.json", "w"), indent=1)

C = {"Self Forcing": "#2a78d6", "LongLive": "#eb6834"}
fig, ax = plt.subplots(figsize=(7, 4.6))
xs = np.linspace(G.L_open.min() - 2, G.L_open.max() + 2, 50)
for model, g in G.groupby("model"):
    r = res["per_model"][model]
    ax.scatter(g.L_open, g.dL_end, s=24, color=C[model], alpha=.7)
    ax.plot(xs, r["slope"] * (xs - r["zero_drift_level"]), color=C[model], lw=2,
            label=f"{model}: slope {r['slope']:.2f}, zero-drift {r['zero_drift_level']:.0f}, R² {r['r2']:.2f}")
ax.axhline(0, color="#52514e", lw=1)
ax.set_xlabel("brightness of the held opening frame (L*)"), ax.set_ylabel("brightness change after 31 s (L*)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(f"{out}/ll_regression.png", dpi=130)
print(json.dumps(res, indent=1))
