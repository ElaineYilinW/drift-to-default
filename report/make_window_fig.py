"""Window figure: the attractor offset of fresh rollouts against the key-value window (studies 33 and 36 with the window-12
Self Forcing arm of study 32 and the released-window arms of study 30; 40 prompts, 3 seeds, no sink). (a) mean |end - start|
over [50, 60) s with prompt-bootstrap 95% CIs; (b) the mean signed offset.
usage: python report/make_window_fig.py [RESULTS_DIR] [OUT_PDF]   (defaults: results/corr/, paper/fig_window.pdf)"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "corr")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper", "fig_window.pdf")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
MODELS = [("Self Forcing", BLUE, "o", {6: "sf06_s0", 9: "sf09_s0", 12: "sf12_s0", 15: "sf15_s0", 18: "sf18_s0",
                                         19: "sf19_s0", 20: "sf20_s0", 21: "sf_s0", 24: "sf24_s0", 27: "sf27_s0"}),
          ("LongLive", GREEN, "s", {6: "ll06_s0", 12: "ll_s0", 21: "ll21_s0", 24: "ll24_s0"}),
          ("CausVid", ORANGE, "^", {12: "cv12_s0", 21: "cv_s0"})]
RELEASED = {"Self Forcing": 21, "LongLive": 12, "CausVid": 21}

r = json.load(open(os.path.join(R, "s33_out", "s33_results.json")))
r["per_arm"].update({a: v for a, v in json.load(open(os.path.join(R, "s36_out", "s36_results.json")))["per_arm"].items() if v})
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.4))
for k, (key, lab) in enumerate((("magnitude", "fresh |end − start| (L*)"), ("signed", "fresh end − start (L*)"))):
    a = ax[k]
    for j, (name, col, mk, arms) in enumerate(MODELS):
        W = [w for w in sorted(arms) if r["per_arm"].get(arms[w])]
        x = [w + (j - 1) * 0.25 for w in W]
        m = [r["per_arm"][arms[w]][key]["mean"] for w in W]
        lo = [r["per_arm"][arms[w]][key]["ci"][0] for w in W]
        hi = [r["per_arm"][arms[w]][key]["ci"][1] for w in W]
        a.errorbar(x, m, yerr=[[u - v for u, v in zip(m, lo)], [v - u for u, v in zip(m, hi)]], fmt=mk + "-", color=col,
                   ms=4, lw=1.0, label=name)
        w0 = RELEASED[name]
        if w0 in W:
            i = W.index(w0)
            a.plot(x[i], m[i], mk, ms=8, mfc="none", mec=col, mew=0.9)
    if key == "signed":
        a.axhline(0, color=MUTED, lw=0.8)
    else:
        a.set_ylim(0, None)
    a.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
    a.set_xlabel("key–value window (latent frames)")
    a.set_ylabel(lab)
    a.text(0.01, 0.98, "(a) magnitude" if k == 0 else "(b) signed", transform=a.transAxes, va="top", fontsize=8.5)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
