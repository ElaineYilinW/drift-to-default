"""Figure of the relaxation study (preregistration 06) from results/corr/relax_out, relax_supp_out.
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
supp = json.load(open(f"{R}/results/corr/relax_supp_out/relax_supp.json"))
long = json.load(open(f"{R}/results/corr/relax_out/long_curves.json"))
NAME = {"sf_s3": "Self Forcing DMD", "sf_s0": "SF, no sink", "sf_s1": "SF, sink 1 (expl.)", "sf_ttc": "SF + correction",
        "sid_s3": "SF-SiD", "sidv2_s3": "SF-SiD v2", "gan_s3": "SF-GAN", "sf10s_s3": "SF-10s", "cv_s3": "CausVid",
        "cv_ttc": "CausVid + corr. (expl.)", "ll_s3": "LongLive", "ll_s0": "LongLive, no sink",
        "ll_ttc": "LongLive + corr. (expl.)", "rf": "Rolling Forcing", "sky": "SkyReels-V2 DF (20 s)"}
order = [c for c in NAME if c in res["per_cfg"] and res["per_cfg"][c].get("alpha_L")] + ["sky"]

fig = plt.figure(figsize=(10.5, 7.2))
ax = fig.add_subplot(1, 2, 1)
for i, c in enumerate(order):
    if c == "sky":
        v = supp["sky"]["alpha_L_20s"]
        ax.errorbar(v["mean"], i, xerr=[[v["mean"] - v["ci"][0]], [v["ci"][1] - v["mean"]]], fmt="s", ms=5, mfc="white",
                    color="#2a78d6", lw=1.3)
        continue
    for j, (an, col) in enumerate([("alpha_L", "#2a78d6"), ("alpha_b", "#eb6834")]):
        v = res["per_cfg"][c].get(an)
        y = i + (j - 0.5) * 0.3
        ax.errorbar(v["mean"], y, xerr=[[v["mean"] - v["ci"][0]], [v["ci"][1] - v["mean"]]], fmt="o", ms=4, color=col, lw=1.3,
                    label={"alpha_L": "brightness L*", "alpha_b": "colour temperature b*"}[an] if i == 0 else None)
ax.set_yticks(range(len(order)), [NAME[c] for c in order], fontsize=8), ax.invert_yaxis()
ax.axvline(0, color=MUTED, lw=.8), ax.axvline(1, color=GRID, lw=.8)
ax.set_xlabel("persistence α at 31 s (within-scene slope, 24 scenes)")
ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.06))
ax.set_title("(a) how much of the opening survives", fontsize=9.5, loc="left")

ax = fig.add_subplot(2, 2, 2)
pal = {"sf_s3": "#2a78d6", "sf_s0": "#184f95", "sf_ttc": "#1baf7a", "ll_s3": "#eb6834", "ll_s0": "#a8441d"}
for c, col in pal.items():
    t, a = map(np.array, zip(*res["per_cfg"][c]["alpha_L_t"]))
    k = np.ones(3) / 3
    ax.plot(t[1:-1], np.convolve(a, k, "valid"), color=col, lw=1.8, label=NAME[c])
ax.axhline(0, color=MUTED, lw=.8)
ax.set_xlabel("seconds"), ax.set_ylabel("persistence α(t)")
ax.legend(frameon=False, fontsize=7.5, ncol=2)
ax.set_title("(b) the first 31 s (3-s moving average)", fontsize=9.5, loc="left")

ax = fig.add_subplot(2, 2, 4)
for c, col in (("sf_s3", "#2a78d6"), ("sf_s0", "#184f95"), ("ll_s3", "#eb6834")):
    v = long[c]
    ax.plot(v["t"], v["mean"], color=col, lw=1.8, label=NAME[c])
    ax.fill_between(v["t"], v["lo"], v["hi"], color=col, alpha=.15, lw=0)
ax.axhline(0, color=MUTED, lw=.8)
ax.set_xlabel("seconds"), ax.set_ylabel("persistence α(t)")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("(c) 120 s: the sink holds about half (6 scenes)", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(f"{R}/report/fig_relax.pdf")

fig, ax = plt.subplots(figsize=(4.2, 3.6))
for c, d in res["per_cfg"].items():
    if "fresh" in d and "zero_L" in d:
        ax.scatter(d["fresh"]["L_end"], d["zero_L"]["median"], s=30, color="#2a78d6")
        ax.annotate(NAME[c], (d["fresh"]["L_end"], d["zero_L"]["median"]), textcoords="offset points", xytext=(4, 2), fontsize=6.5)
lim = [12, 60]
ax.plot(lim, lim, color=GRID, lw=1)
ax.set_xlim(lim), ax.set_ylim(lim)
ax.set_xlabel("end brightness of fresh starts (L*)"), ax.set_ylabel("zero-drift level from shifted openings (L*)")
ax.set_title(f"default = where a fresh rollout ends (r = {np.corrcoef([d['fresh']['L_end'] for d in res['per_cfg'].values() if 'fresh' in d and 'zero_L' in d], [d['zero_L']['median'] for d in res['per_cfg'].values() if 'fresh' in d and 'zero_L' in d])[0, 1]:.2f})", fontsize=8.5, loc="left")
fig.tight_layout()
fig.savefig(f"{R}/report/fig_default_fresh.pdf")
print("ok relax")
