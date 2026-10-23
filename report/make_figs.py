"""Figures for the technical report, drawn from the summary CSVs in results/. usage: python report/make_figs.py"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(__file__), "..")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def panel(ax, groups, title):
    allx = np.concatenate([g.L_open.values for _, g, _ in groups])
    xs = np.linspace(allx.min() - 2, allx.max() + 2, 50)
    for name, g, col in groups:
        b, a = np.polyfit(g.L_open, g.dL_end, 1)
        ax.scatter(g.L_open, g.dL_end, s=10, color=col, alpha=.55, lw=0)
        ax.plot(xs, a + b * xs, color=col, lw=2, label=f"{name} ({-a / b:.0f})")
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_xlabel("opening brightness L*")
    ax.legend(frameon=False, fontsize=7.5, title="zero-drift level", title_fontsize=7.5, loc="lower left")


ll = pd.read_csv(f"{R}/results/corr/ll_out/ll_openings.csv")
li = pd.read_csv(f"{R}/results/corr/light_out/light_openings.csv")
fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
sf = ll[ll.model == "Self Forcing"]
panel(axs[0], [("real NAVSIM", sf[sf.src == "real"], "#2a78d6"), ("model-generated", sf[sf.src == "self"], "#eb6834")],
      "(a) Self Forcing, generic prompt")
cols = {"sunny": "#eda100", "overcast": "#2a78d6", "dusk": "#eb6834", "night": "#6250d6"}
panel(axs[1], [(k, li[li.light == k], c) for k, c in cols.items()], "(b) same openings, four lighting prompts")
panel(axs[2], [("Self Forcing", sf, "#2a78d6"), ("LongLive", ll[ll.model == "LongLive"], "#eb6834")],
      "(c) two models, generic prompt")
axs[0].set_ylabel("brightness change after 31 s (L*)")
fig.tight_layout()
fig.savefig(f"{R}/report/fig_default.pdf")
print("ok")

# matched-prompt experiment: paired effects (left) and the caption-lighting breakdown (right)
res = json.load(open(f"{R}/results/corr/match_out/match_results.json"))
V = pd.read_csv(f"{R}/results/corr/match_out/match_videos.csv")
cap = {json.loads(l)["open"]: json.loads(l)["caption"]["lighting"].lower() for l in open(f"{R}/prompts/matched.jsonl")}
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
con = ["M - G", "M+ttc - M", "G+ttc - G", "M+ttc - G"]
for ax, y, lab in [(axs[0], "absdL_end", "|ΔL| after 31 s, paired difference"), (axs[1], "retention", "content retention, paired difference")]:
    for j, (m, col) in enumerate([("SF", "#2a78d6"), ("LL", "#eb6834")]):
        for i, c in enumerate(con):
            e = res["secondary"][m][y][c]
            yy = i + (j - .5) * .28
            ax.errorbar(e["est"], yy, xerr=[[e["est"] - e["ci"][0]], [e["ci"][1] - e["est"]]], fmt="o", ms=4, color=col,
                        lw=1.4, label={"SF": "Self Forcing", "LL": "LongLive"}[m] if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=.8)
    ax.set_yticks(range(len(con)), con), ax.invert_yaxis(), ax.set_xlabel(lab)
axs[1].legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=7.5, loc="lower left")
axs[0].set_title("(a) brightness: primary H1, H2 not supported", fontsize=9.5, loc="left")
axs[1].set_title("(b) content kept: large gain from M", fontsize=9.5, loc="left")


def cls(t):
    if re.search(r"night|very dark", t):
        return "night / very dark"
    if re.search(r"dim|overcast|subdued|dusk|sunset|setting sun|evening", t):
        return "dim / overcast / dusk"
    return "bright / sunny"


O = V.groupby(["model", "open", "arm"], as_index=False).mean(numeric_only=True)
O["cap"] = O.open.map(lambda o: cls(cap[o]))
g = O[(O.model == "SF") & O.arm.isin(["G", "M"])].groupby(["cap", "arm"]).dL_end.mean().unstack()
order = ["bright / sunny", "dim / overcast / dusk", "night / very dark"]
n = O[(O.model == "SF") & (O.arm == "G")].cap.value_counts()
x = np.arange(3)
axs[2].bar(x - .18, g.loc[order, "G"], .34, color="#898781", label="G (generic)")
axs[2].bar(x + .18, g.loc[order, "M"], .34, color="#eb6834", label="M (caption)")
axs[2].axhline(0, color=MUTED, lw=.8)
axs[2].set_xticks(x, [f"{s}\n(n={n[o]})" for s, o in zip(["bright", "dim / dusk", "dark"], order)], fontsize=8)
axs[2].set_ylabel("signed ΔL after 31 s (L*)"), axs[2].legend(frameon=False, fontsize=7.5)
axs[2].set_title("(c) Self Forcing, by caption lighting", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(f"{R}/report/fig_match.pdf")
print("ok match")

# calibrated-lighting experiment: paired |dL| and retention vs the generic prompt, and signed dL by lighting class
V = pd.read_csv(f"{R}/results/corr/calib_out/calib_videos.csv")
O = V.groupby(["model", "open", "light", "arm"], as_index=False).mean(numeric_only=True)
rng = np.random.default_rng(0)
arms = ["M", "CL", "C", "C+ttc"]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for ax, y, lab in [(axs[0], "absdL_end", "|ΔL| after 31 s, minus generic prompt"),
                   (axs[1], "retention", "content retention, minus generic prompt")]:
    for j, (m, col) in enumerate([("SF", "#2a78d6"), ("LL", "#eb6834")]):
        w = O[O.model == m].pivot(index="open", columns="arm", values=y)
        for i, a in enumerate(arms):
            d = (w[a] - w["G"]).values
            bs = np.percentile([rng.choice(d, len(d)).mean() for _ in range(4000)], [2.5, 97.5])
            ax.errorbar(d.mean(), i + (j - .5) * .28, xerr=[[d.mean() - bs[0]], [bs[1] - d.mean()]], fmt="o", ms=4,
                        color=col, lw=1.4, label={"SF": "Self Forcing", "LL": "LongLive"}[m] if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=.8)
    ax.set_yticks(range(4), ["M (VLM)", "CL", "C", "C+ttc"]), ax.invert_yaxis(), ax.set_xlabel(lab, fontsize=8.5)
axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
axs[0].set_title("(a) brightness drift", fontsize=9.5, loc="left")
axs[1].set_title("(b) content kept", fontsize=9.5, loc="left")
g = O[O.model == "SF"].copy()
g["cls"] = np.where(g.light == "sunny", "sunny (n=23)", "night/dusk/\novercast (n=17)")
t = g.pivot_table(index="cls", columns="arm", values="dL_end")
x = np.arange(2)
for k, (a, c) in enumerate([("G", "#898781"), ("M", "#eb6834"), ("C", "#1baf7a"), ("C+ttc", "#6250d6")]):
    axs[2].bar(x + (k - 1.5) * .2, t[a].values, .19, color=c, label=a)
axs[2].axhline(0, color=MUTED, lw=.8)
axs[2].set_xticks(x, t.index, fontsize=8), axs[2].set_ylabel("signed ΔL after 31 s (L*)")
axs[2].legend(frameon=False, fontsize=7.5, ncol=2)
axs[2].set_title("(c) Self Forcing, by calibrated class", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(f"{R}/report/fig_calib.pdf")
print("ok calib")

# correction trade-off: drift reduction vs legitimate transitions kept
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
b = json.load(open(f"{R}/results/corr/baseline_out/results.json"))
pts = [("none", 0.0, 1.0, MUTED), ("correction β=1", b["tests"]["D:ttc_s1"]["est"], b["controls"]["ttc_s1"]["median_kept_valid"], "#2a78d6"),
       ("correction β=1.5", b["tests"]["D:ttc_s1_b15"]["est"], b["controls"]["ttc_s1_b15"]["median_kept_valid"], "#2a78d6"),
       ("post-hoc mean", b["tests"]["D:phmean"]["est"], b["controls"]["phmean"]["median_kept_valid"], "#eb6834"),
       ("post-hoc Reinhard", b["tests"]["D:phreinhard"]["est"], b["controls"]["phreinhard"]["median_kept_valid"], "#eb6834")]
for name, x, y, c in pts:
    axs[0].scatter(x, y, s=60, color=c, zorder=3)
    axs[0].annotate(name, (x, y), textcoords="offset points", xytext=(6, -3), fontsize=8)
axs[0].set_xlabel("change in colour drift D (held-out prompts)"), axs[0].set_ylabel("legitimate transitions kept")
axs[0].set_title("(a) preregistered confirmation set", fontsize=9.5, loc="left")
axs[0].set_xlim(-2.1, 0.5), axs[0].set_ylim(-0.05, 1.1)
e = json.load(open(f"{R}/results/corr/e3_out/e2_summary.json"))
lab = {"ttc_s0": "step 0", "ttc_s01": "steps 0+1", "ttc_s1": "step 1", "ttc_s1_b15": "step 1, β1.5", "ttc_s1_b2": "step 1, β2",
       "ttc_s2": "step 2", "ttc_s3": "step 3", "luma_s3_a02": "brightness controller"}
for k, name in lab.items():
    v = e[k]
    c = "#eb6834" if k.startswith("luma") else "#2a78d6"
    axs[1].scatter(v["D_end"]["diff"], v["control_kept_median"], s=50, color=c, zorder=3)
    axs[1].annotate(name, (v["D_end"]["diff"], v["control_kept_median"]), textcoords="offset points", xytext=(5, -3), fontsize=7.5)
axs[1].scatter(0, 1, s=50, color=MUTED)
axs[1].set_xlabel("change in colour drift D (development set)")
axs[1].set_title("(b) the denoising step is a knob", fontsize=9.5, loc="left")
axs[1].set_ylim(-0.15, 1.15)
for ax in axs:
    ax.axhline(1, color=GRID, lw=.8), ax.axvline(0, color=GRID, lw=.8)
fig.tight_layout()
fig.savefig(f"{R}/report/fig_tradeoff.pdf")
print("ok tradeoff")
