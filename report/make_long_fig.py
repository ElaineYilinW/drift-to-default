"""180-s figure: (a) persistence over 10-s windows to 180 s for no sink, the re-rotated sink, the sink at its original
positions, Rolling Forcing and LongLive (study 24; scene-bootstrap 95% CIs); (b) 1-s persistence curves of the sinks at
their original positions (Self Forcing, LongLive) and the re-rotated sink, with the burst onsets the probe predicts
(study 26; triangles: before 60 s in sample, after 60 s the preregistered predictions; ticks: control onsets).
usage: python report/make_long_fig.py [RESULTS_DIR] [OUT_PDF]   (defaults: results/corr/, paper/fig_long.pdf)"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "corr")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper", "fig_long.pdf")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7a5bd6"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

S24 = json.load(open(os.path.join(R, "s24_out", "s24_results.json")))
S26 = json.load(open(os.path.join(R, "s26_out", "s26_results.json")))
ARMS = (("sf180_s3", BLUE, "sink, original positions"), ("ll180_s3", GREEN, "LongLive sink"),
        ("rf180", ORANGE, "Rolling Forcing"), ("sf180_s3rr", PURPLE, "re-rotated sink"), ("sf180_s0", MUTED, "no sink"))

fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={"width_ratios": [1, 2.1]})

# (a) 10-s windows
a = ax[0]
for arm, col, lab in ARMS:
    A = S24["alpha"][arm]
    T = np.array(sorted(int(k) for k in A))
    m = np.array([A[str(k)]["mean"] for k in T])
    lo, hi = np.array([A[str(k)]["ci"] for k in T]).T
    a.fill_between(T, lo, hi, color=col, alpha=0.13, lw=0)
    a.plot(T, m, color=col, lw=1.3, label=lab)
a.axhline(0, color=GRID, lw=0.8, zorder=0)
a.set(xlabel="seconds after the opening", ylabel=r"persistence $\alpha$ (10-s windows)", xlim=(10, 180),
      ylim=(-0.12, 0.9), xticks=(10, 60, 120, 180))
a.set_title("a  180 s, five configurations", loc="left", fontsize=8.5)
a.legend(fontsize=6.3, loc="upper right", handlelength=1.1, labelspacing=0.25)

# (b) 1-s curves and predicted onsets
b = ax[1]
for arm, col, lab in ARMS[:2] + ARMS[3:4]:
    c = S26["alpha_1s"][arm]
    t = np.array(sorted(int(k) for k in c))
    b.plot(t - 0.5, [c[str(k)] for k in t], color=col, lw=0.9, label=lab)
early = S26["early"]["onsets"]
for tau in early + S26["predicted"]:
    b.plot(tau, 1.02, marker="v", ms=3.4, color=INK if tau < 60 else ORANGE, lw=0, clip_on=False)
for tau in S26["early"]["control"] + S26["control"]:
    b.plot([tau, tau], [0.985, 1.005], color=MUTED, lw=0.7, clip_on=False)
b.axvline(60, color=MUTED, lw=0.7, ls=":")
b.text(61, 0.93, "predicted out of sample", fontsize=6.5, color=ORANGE, va="top")
b.text(59, 0.93, "in sample", fontsize=6.5, color=INK, va="top", ha="right")
b.axhline(0, color=GRID, lw=0.8, zorder=0)
b.set(xlabel="seconds after the opening", ylabel=r"$\alpha$ (1-s windows)", xlim=(0, 180), ylim=(-0.15, 1.0),
      xticks=range(0, 181, 30))
b.set_title("b  bursts at the onsets the probe predicts (triangles; ticks: controls)", loc="left", fontsize=8.5, pad=8)
b.legend(fontsize=6.5, loc="lower right", ncol=3, handlelength=1.1, columnspacing=0.9, bbox_to_anchor=(1.0, -0.02))
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
