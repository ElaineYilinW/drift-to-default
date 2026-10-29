"""Held-distance figure: with the sink held at one query-sink distance, (a) persistence at 60 s and (b) frame-to-frame
change over 50-60 s against the probe's attention share at that distance; Self Forcing (study 27, 17 distances; the 3
of study 25 marked) and LongLive with its own probe's shares (study 28), scene-bootstrap 95% CIs; dashed: the sink at
its original positions. Entries whose results file is missing are skipped.
usage: python report/make_held_fig.py [RESULTS_DIR] [OUT_PDF]   (defaults: results/corr/, paper/fig_held.pdf)"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "corr")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper", "fig_held.pdf")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, GREEN = "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def load(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


S27, S28, S25 = load("s27_out/s27_results.json"), load("s28_out/s28_results.json"), load("s25_out/s25_results.json")


def rows(per, share, key):
    """(D, share, mean, lo, hi) per distance for 'alpha60' or 'motion'."""
    out = []
    for D, q in per.items():
        x = q["alpha"].get("60") if key == "alpha60" else q.get("motion_50_60")
        if x:
            out.append((int(D), float(share[str(D)] if isinstance(share, dict) else share), x["mean"], *x["ci"]))
    return sorted(out, key=lambda r: r[1])


series = []
if S27:
    series.append(("Self Forcing", BLUE, "o", S27["per_distance"], {D: q["share"] for D, q in S27["per_distance"].items()},
                   S27["distances"]["study25"]))
if S28:
    per = {k[4:].lstrip("0") or "0": v for k, v in S28["per_arm"].items() if k.startswith("ll_d")}
    series.append(("LongLive", GREEN, "s", per, S28["share_ll"], []))

fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.3))
for j, (key, ylab) in enumerate((("alpha60", r"persistence $\alpha$ at 60 s"),
                                 ("motion", "frame-to-frame change, 50–60 s"))):
    a = ax[j]
    for name, col, mk, per, share, old in series:
        r = np.array(rows(per, share, key))
        if not len(r):
            continue
        a.errorbar(r[:, 1], r[:, 2], yerr=[r[:, 2] - r[:, 3], r[:, 4] - r[:, 2]], fmt=mk, ms=3.6, color=col, ecolor=col,
                   elinewidth=0.7, capsize=0, alpha=0.9, label=name)
        for D, s, m, lo, hi in r:
            if int(D) in old:
                a.plot(s, m, marker=mk, ms=7.5, mfc="none", mec=INK, mew=0.7, lw=0)
    if S25:   # the sink at its original positions (Self Forcing: study 25; LongLive: study 28)
        ref = S25["motion_50_60"]["sp_abs"]["motion"] if key == "motion" else S25["alpha"]["sp_abs"]["60"]["mean"]
        a.axhline(ref, color=BLUE, lw=0.8, ls="--", zorder=0, label="original positions")
    if S28 and S28["per_arm"].get("ll_abs"):
        x = S28["per_arm"]["ll_abs"]["motion_50_60" if key == "motion" else "alpha"]
        x = x if key == "motion" else x.get("60")
        if x:
            a.axhline(x["mean"], color=GREEN, lw=0.8, ls="--", zorder=0)
    a.set(xlabel="probe's attention share at the held distance", ylabel=ylab)
a = ax[0]
a.axhline(0, color=GRID, lw=0.8, zorder=0)
a.set_title("a  how much is kept", loc="left", fontsize=8.5)
ax[1].set_title("b  how much the video moves", loc="left", fontsize=8.5)
a.legend(fontsize=6.8, loc="lower right", handlelength=1.4)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
