"""Sink-position figure: (a) the exploratory attention probes, share of attention a Self Forcing sink would receive at
each query-sink distance (sinkprobe_long_out to 723 when present, else sinkprobe_out to 300; markers show the major
peaks, those after 60 s being study 26's predictions; the first probe's curve is the same, r = 0.999); (b) persistence
over time (1-s windows) with the sink at its original positions and moved 18 or 36 frames further back (study 25;
study 10's sink 3 if study 25 is missing), with the burst onsets the probe predicts; (c) persistence with the sink
held at a peak distance, a trough and next to the window (study 25, when present). Entries whose results file is
missing are skipped.
usage: python report/make_probe_fig.py [RESULTS_DIR] [OUT_PDF]   (defaults: results/corr/, paper/fig_probe.pdf)"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "corr")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper", "fig_probe.pdf")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7a5bd6"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
PEAKS = (71, 134, 203, 266)
TPAD = 24   # titles of (b), (c) level with (a), whose title sits above its top axis


def load(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


PR, S25, ATL = load("sinkprobe_out/sinkprobe_out.json"), load("s25_out/s25_results.json"), load("atlas_out/atlas_results.json")
PL = load("sinkprobe_long_out/sinkprobe_long_out.json")
onset = lambda P, d: (P - 5 - d) / 4


def smooth(t, v, k=3):
    v = np.convolve(v, np.ones(k) / k, mode="same")
    return t[k // 2:-(k // 2)], v[k // 2:-(k // 2)]


panels = 3 if S25 else 2
fig, ax = plt.subplots(1, panels, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [1.15, 1, 1][:panels]})

# (a) probe
g, m = np.array(PR["grid"]), np.array(PR["mean"])
if PL:
    gl, ml = np.array(PL["grid"]), np.array(PL["mean"])
    ax[0].plot(gl, ml, color=BLUE, lw=1.0)
    for P in PL["peaks"]:
        ax[0].plot(P, ml[gl == P][0] + 0.012, marker="v", ms=3.2, color=ORANGE if P > 245 else INK, lw=0)
    for P in PEAKS[:3]:
        ax[0].annotate(str(P), (P, ml[gl == P][0] + 0.018), xytext=(0, 2), textcoords="offset points", ha="center",
                       fontsize=6.5, color=INK)
    ax[0].axvline(245, color=MUTED, lw=0.7, ls=":")
    xmax, med = 723, np.median(ml[gl >= 10])
else:
    ax[0].plot(g, m, color=BLUE, lw=1.4)
    for P in PEAKS:
        ax[0].annotate(str(P), (P, m[g == P][0]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7,
                       color=INK)
    xmax, med = 300, np.median(m[g >= 10])
ax[0].axhline(med, color=MUTED, lw=0.8, ls=":")
ax[0].set(xlabel="query–sink distance (latent frames)", ylabel="attention share of the sink", xlim=(0, xmax),
          ylim=(0.03, 0.22) if PL else None)
ax[0].set_title("a  attention to a sink by distance", loc="left", fontsize=8.5)
top = ax[0].secondary_xaxis("top", functions=(lambda d: (d - 5) / 4, lambda t: 4 * t + 5))
top.set_xlabel("time at the original position (s)", fontsize=7.5, color=MUTED)
top.tick_params(labelsize=7)

# (b) bursts over time
b = ax[1]
if S25:
    for arm, d, col, lab in (("sp_abs", 0, BLUE, "original"), ("sp_off18", 18, PURPLE, "+18 frames"),
                             ("sp_off36", 36, ORANGE, "+36 frames")):
        c = S25["alpha_1s"][arm]
        t = np.array(sorted(int(k) for k in c))
        b.plot(*smooth(t - 0.5, np.array([c[str(k)] for k in t])), color=col, lw=1.4, label=lab)
        for P in PEAKS:
            if 5 <= onset(P, d) <= 60:
                b.axvline(onset(P, d), color=col, lw=0.7, ls=":", alpha=0.8)
else:
    c = dict(ATL["curves"]["sf60_s3"])
    t = np.array(sorted(c))
    b.plot(*smooth(t - 0.5, np.array([c[k] for k in t])), color=BLUE, lw=1.4, label="sink 3, original (study 10)")
    for P in PEAKS:
        if onset(P, 0) <= 60:
            b.axvline(onset(P, 0), color=BLUE, lw=0.7, ls=":")
b.set(xlabel="time after the opening (s)", ylabel=r"persistence $\alpha$ (1-s)", xlim=(0, 60), ylim=(0, 1))
b.set_title("b  bursts at the predicted onsets", loc="left", fontsize=8.5, pad=TPAD)
b.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, handlelength=1.0, columnspacing=0.8)

# (c) held distances
if S25:
    for arm, col, lab in (("sp_d203", BLUE, "203 (peak)"), ("sp_d187", ORANGE, "187 (trough)"),
                          ("sp_d20", GREEN, "20 (window)")):
        c = S25["alpha_1s"][arm]
        t = np.array(sorted(int(k) for k in c))
        ax[2].plot(*smooth(t - 0.5, np.array([c[str(k)] for k in t])), color=col, lw=1.4, label=lab)
    ax[2].set(xlabel="time after the opening (s)", xlim=(0, 60), ylim=(0, 1))
    ax[2].set_title("c  sink held at one distance", loc="left", fontsize=8.5, pad=TPAD)
    ax[2].legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.59), handlelength=1.4)
for a in ax:
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT)
print("wrote", OUT, "panels", panels)
