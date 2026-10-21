"""Does the text prompt set the default brightness? Same 40 held openings (20 real NAVSIM, 20 model-generated),
continued under four lighting prompts (sunny, overcast, dusk, night), no correction.

Per prompt: dL_end = a + b * L_open (opening-level means over seeds), zero-drift level -a/b with an opening-cluster
bootstrap CI; joint model dL_end ~ L_open + light (common slope) and with a slope interaction; mean end brightness.
Optionally the generic-prompt runs (real/self dirs) are added as a fifth level.
usage: python analyze_light.py LIGHT_DIR OUT_DIR [REAL_DIR SELF_DIR]
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

src, out = sys.argv[1], sys.argv[2]
extra = sys.argv[3:5]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
rows = []
for p in glob.glob(f"{src}/*--*_none_s*.npz"):
    m = re.match(r"(.+)--(\w+)_none_s(\d+)\.npz", os.path.basename(p))
    L = np.load(p)["lab_mean"][:, 0]
    rows.append(dict(open=m[1], light=m[2], seed=int(m[3]), L_open=L[1:9].mean(), L_end=L[-32:].mean()))
for d in extra:
    for p in glob.glob(f"{d}/*_none_s*.npz"):
        m = re.match(r"((?:real|self)\d+)_none_s(\d+)\.npz", os.path.basename(p))
        if m:
            L = np.load(p)["lab_mean"][:, 0]
            rows.append(dict(open=m[1], light="generic", seed=int(m[2]), L_open=L[1:9].mean(), L_end=L[-32:].mean()))
D = pd.DataFrame(rows)
D["dL_end"] = D.L_end - D.L_open
G = D.groupby(["open", "light"], as_index=False).mean(numeric_only=True)
G.to_csv(f"{out}/light_openings.csv", index=False)
order = [x for x in ["sunny", "generic", "overcast", "dusk", "night"] if x in set(G.light)]


def zero_level(g):
    b1, b0 = np.polyfit(g.L_open, g.dL_end, 1)
    return -b0 / b1, b1


res = {"n_videos": int(len(D)), "per_prompt": {}}
for lt in order:
    g = G[G.light == lt]
    z, b1 = zero_level(g)
    opens = g.open.values
    bs = []
    for _ in range(2000):
        pick = rng.choice(opens, len(opens))
        gg = pd.concat([g[g.open == o] for o in pick])
        bs.append(zero_level(gg)[0])
    res["per_prompt"][lt] = dict(n_openings=int(len(g)), zero_drift_level=float(z), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                 slope=float(b1), mean_L_end=float(g.L_end.mean()), r2=float(np.corrcoef(g.L_open, g.dL_end)[0, 1] ** 2))
m1 = smf.ols("dL_end ~ L_open + C(light, Treatment('overcast'))", G).fit()
m2 = smf.ols("dL_end ~ L_open * C(light, Treatment('overcast'))", G).fit()
light_terms = [p for p in m1.params.index if p.startswith("C(light")]
res["ancova_common_slope"] = dict(r2=float(m1.rsquared),
                                  light_F_p=float(m1.f_test(np.eye(len(m1.params))[[list(m1.params.index).index(t) for t in light_terms]]).pvalue),
                                  coefs={k: [float(m1.params[k]), *map(float, m1.conf_int().loc[k])] for k in m1.params.index})
res["slope_interaction_p"] = float(m2.compare_f_test(m1)[1])
json.dump(res, open(f"{out}/light_summary.json", "w"), indent=1)

C = {"sunny": "#eda100", "generic": "#52514e", "overcast": "#2a78d6", "dusk": "#eb6834", "night": "#6250d6"}
fig, ax = plt.subplots(figsize=(7.5, 4.8))
xs = np.linspace(G.L_open.min() - 2, G.L_open.max() + 2, 50)
for lt in order:
    g = G[G.light == lt]
    b1, b0 = np.polyfit(g.L_open, g.dL_end, 1)
    ax.scatter(g.L_open, g.dL_end, s=22, color=C[lt], alpha=.7)
    ax.plot(xs, b0 + b1 * xs, color=C[lt], lw=2, label=f"{lt}: zero-drift {res['per_prompt'][lt]['zero_drift_level']:.0f}")
ax.axhline(0, color="#52514e", lw=1)
ax.set_xlabel("brightness of the held opening frame (L*)"), ax.set_ylabel("brightness change after 31 s (L*)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(f"{out}/light_regression.png", dpi=130)
print(json.dumps(res, indent=1))
