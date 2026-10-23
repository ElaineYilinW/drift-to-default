"""Figures of the relaxation study (preregistration 06) from results/corr/relax_out/ and relax_theory_out/.
usage: python report/make_relax_figs.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
res = json.load(open(f"{R}/results/corr/relax_out/relax_results.json"))
thy = json.load(open(f"{R}/results/corr/relax_theory_out/relax_theory.json"))
NAME = {"sf_s3": "Self Forcing (DMD)", "sf_s1": "SF, sink 1", "sf_s0": "SF, no sink", "sf_ttc": "SF + correction",
        "sid_s3": "SF-SiD", "sidv2_s3": "SF-SiD v2", "gan_s3": "SF-GAN", "sf10s_s3": "SF-10s", "cv_s3": "CausVid",
        "cv_ttc": "CausVid + correction", "ll_s3": "LongLive", "ll_s0": "LongLive, no sink", "ll_ttc": "LongLive + correction",
        "rf": "Rolling Forcing", "sky": "SkyReels-V2 DF (20 s)"}
order = [c for c in NAME if c in res["per_cfg"]]

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.05, 1]})
ax = axs[0]
for i, c in enumerate(order):
    for j, (an, col) in enumerate([("alpha_L", "#2a78d6"), ("alpha_b", "#eb6834")]):
        v = res["per_cfg"][c].get(an)
        if not v:
            continue
        y = i + (j - 0.5) * 0.3
        ax.errorbar(v["mean"], y, xerr=[[v["mean"] - v["ci"][0]], [v["ci"][1] - v["mean"]]], fmt="o", ms=4, color=col, lw=1.3,
                    label={"alpha_L": "brightness L*", "alpha_b": "colour temperature b*"}[an] if i == 0 else None)
ax.set_yticks(range(len(order)), [NAME[c] for c in order], fontsize=8), ax.invert_yaxis()
ax.axvline(0, color=MUTED, lw=.8), ax.axvline(1, color=GRID, lw=.8)
ax.set_xlabel("persistence α of the opening at the end (within-scene slope)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("(a) how much of the opening survives", fontsize=9.5, loc="left")
ax = axs[1]
pal = {"sf_s3": "#2a78d6", "sf_s1": "#86b6ef", "sf_s0": "#184f95", "sf_ttc": "#1baf7a", "ll_s3": "#eb6834", "ll_s0": "#a8441d"}
for c, col in pal.items():
    tc = res["per_cfg"].get(c, {}).get("alpha_L_t")
    if not tc:
        continue
    t, a = zip(*tc)
    ax.plot(t, a, color=col, lw=1.8, label=NAME[c])
ax.axhline(0, color=MUTED, lw=.8)
ax.set_xlabel("seconds"), ax.set_ylabel("persistence α(t)")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("(b) sink and correction set the plateau", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(f"{R}/report/fig_relax.pdf")
print("ok relax")
