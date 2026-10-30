"""Band figure: (a) the attention-to-sink probe with all temporal RoPE frequencies rotated to each distance and with one
band held at the reference distance 20 (band probe, lab note 42); (b) persistence of the sink with the same bands held
in generation (study 34, 1-s alpha, scene means), with the burst onsets the probe predicts; (c) LongLive with j4 or the
neighbouring j5 held (study 35). Panels (b) and (c) are skipped if the study 34 / 35 results are missing.
usage: python report/make_band_fig.py [RESULTS_DIR] [OUT_PDF]   (defaults: results/corr/, paper/fig_band.pdf)"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "corr")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper", "fig_band.pdf")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7a5bd6"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
ORIG, NEW = (16.5, 32.25, 49.5), (7.0, 24.5, 40.0, 57.25)
SERIES = [("all", "bh_abs", INK, "original positions"),
          ("all but j4", "bh_j4", BLUE, "$j{=}4$ held (period 34)"),
          ("all but j3", "bh_j3", ORANGE, "$j{=}3$ held (period 22)"),
          ("all but j8-11", "bh_j811", GREEN, "$j{=}8$–11 held")]

band = json.load(open(os.path.join(R, "sinkband_out", "sinkband.json")))
p34 = os.path.join(R, "s34_out", "s34_results.json")
s34 = json.load(open(p34)) if os.path.exists(p34) else None
p35 = os.path.join(R, "s35_out", "s35_results.json")
s35 = json.load(open(p35)) if os.path.exists(p35) and s34 else None
panels = 1 + bool(s34) + bool(s35)
fig, ax = plt.subplots(1, panels, figsize=({1: 3.6, 2: 6.9, 3: 6.9}[panels], 2.3), squeeze=False,
                       gridspec_kw={"width_ratios": [1.25, 1, 1][:panels]} if panels == 3 else None)
ax = ax[0]

a = ax[0]
g = np.array(band["grid"])
for name, _, col, lab in SERIES:
    c = np.array(band["bands"][name]["curve"])
    a.plot(g, c, color=col, lw=1.0 if name == "all" else 0.9, alpha=1 if name in ("all", "all but j4", "all but j3") else 0.8,
           label=lab, zorder=3 if name == "all" else 2)
for P in (71, 134, 203, 266):
    a.axvline(P, color=GRID, lw=0.6, zorder=0)
a.set_xlim(5, 300)
a.set_ylim(0.04, 0.19)
a.set_xlabel("query–sink distance (latent frames)")
a.set_ylabel("attention share to the sink")
a.text(0.01, 0.98, "(a) probe", transform=a.transAxes, va="top", fontsize=8.5, color=INK)
top = a.secondary_xaxis("top", functions=(lambda D: (D - 5) / 4, lambda t: 4 * t + 5))
top.set_xlabel("s after the opening, sink at its original positions", fontsize=7.5, color=MUTED)
top.tick_params(labelsize=7, colors=MUTED)

if s34:
    b = ax[1]
    for _, arm, col, lab in SERIES:
        c = s34["alpha_1s"].get(arm)
        if c:
            t = sorted(int(k) for k in c)
            b.plot([x - 0.5 for x in t], [c[str(x)] for x in t], color=col, lw=1.0, label=lab)
    c = s34["alpha_1s"].get("bh_rr")
    if c:
        t = sorted(int(k) for k in c)
        b.plot([x - 0.5 for x in t], [c[str(x)] for x in t], color=MUTED, lw=0.9, ls="--", label="re-rotated")
    yl = b.get_ylim()
    b.plot(ORIG, [yl[0]] * 3, "^", color=INK, ms=4.5, clip_on=False, zorder=4)
    b.plot(NEW, [yl[0]] * 4, "^", color=ORANGE, ms=4.5, clip_on=False, zorder=4)
    b.set_ylim(yl[0], 1.15)
    b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_xlim(0, 60)
    b.set_xlabel("s after the opening")
    b.set_ylabel(r"persistence $\alpha$ (1-s)")
    b.text(0.01, 0.98, "(b) generation (study 34)", transform=b.transAxes, va="top", fontsize=8.5, color=INK)
if s35:
    b = ax[2]
    for arm, col, lab in (("ll_abs", INK, None), ("ll_j4", BLUE, None), ("ll_j5", PURPLE, "$j{=}5$ held (period 51)")):
        c = s35["alpha_1s"][arm]
        t = sorted(int(k) for k in c)
        b.plot([x - 0.5 for x in t], [c[str(x)] for x in t], color=col, lw=1.0, label=lab)
    b.set_ylim(ax[1].get_ylim())
    b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    b.plot(ORIG, [b.get_ylim()[0]] * 3, "^", color=INK, ms=4.5, clip_on=False, zorder=4)
    b.set_xlim(0, 60)
    b.set_xlabel("s after the opening")
    b.text(0.01, 0.98, "(c) LongLive (study 35)", transform=b.transAxes, va="top", fontsize=8.5, color=INK)
h, l = (ax[1] if panels >= 2 else ax[0]).get_legend_handles_labels()
if s35:
    h2, l2 = ax[2].get_legend_handles_labels()
    h, l = h + h2, l + l2
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6 if s35 else 5, fontsize=7.2, handlelength=1.6,
           columnspacing=1.2)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
