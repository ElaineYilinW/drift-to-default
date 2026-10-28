"""Summary figure: long-run persistence of the opening's tone, grouped by how each model holds the opening
(studies 10, 16, 18, 19, 20, 22). Entries whose results file is missing are skipped.
usage: python report/make_hold_fig.py [RESULTS_DIR] [OUT_PDF]   (defaults: results/corr/, paper/fig_hold.pdf)"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "corr")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper", "fig_hold.pdf")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7a5bd6"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def load(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


S20, S16, ATL = load("s20_out/s20_results.json"), load("s16_out/s16_results.json"), load("atlas_out/atlas_results.json")
M18, M19, KR = load("magi_out/magi_results.json"), load("magi19_out/magi19_results.json"), load("krea_out/krea_results.json")


def curve(src, *keys):
    """alpha(T) dict {T: {mean, ci}} from a results json, or None"""
    x = src
    for k in keys:
        x = x.get(k) if isinstance(x, dict) else None
    return {int(t): v for t, v in x.items()} if x else None


# group: (title, colour); entries: (label, curve, model marker)
GROUPS = [
    ("opening only in the window", ORANGE, [
        ("Self Forcing, no sink", curve(S20, "alpha", "sf60_s0")),
        ("LongLive, no sink", curve(ATL, "alpha", "ll60_s0")),
        ("MAGI-1, released window", curve(M19, "alpha_b") or curve(M18, "alpha", "mgw")),
        ("Krea 14B, released window", curve(KR, "alpha", "kr60_none"))]),
    ("overlapping frames re-denoised", PURPLE, [
        ("SkyReels-V2, 17 frames", curve(S16, "D", "alpha", "sky17")),
        ("SkyReels-V2, 5 frames", curve(S16, "D", "alpha", "sky5"))]),
    ("opening kept next to the window", GREEN, [
        ("Self Forcing, sink 3 re-rotated", curve(S20, "alpha", "sf60_s3rr")),
        ("Self Forcing, sink 6 re-rotated", curve(S16, "B", "alpha", "sf60_s6rr")),
        ("Krea 14B, first-frame anchor", curve(KR, "alpha", "kr60_first")),
        ("FramePack, context prefix", curve(S16, "C", "alpha", "fpv_ctx")),
        ("FramePack, prefix + embedding", curve(S16, "C", "alpha", "fpv_both")),
        ("MAGI-1, full history", curve(M18, "alpha", "mgf"))]),
    ("sink at its original positions", BLUE, [
        ("Self Forcing, sink 3", curve(S20, "alpha", "sf60_s3")),
        ("Self Forcing, sink 6", curve(ATL, "alpha", "sf60_s6")),
        ("LongLive, sink", curve(ATL, "alpha", "ll60_s3"))])]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9, 4.0), gridspec_kw=dict(width_ratios=[1.2, 1]), layout="constrained")
y, ticks, labels = 0, [], []
for title, col, items in GROUPS:
    items = [(lab, c) for lab, c in items if c]
    if not items:
        continue
    ax.text(-0.03, y, title, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.4, color=col,
            fontweight="bold")
    y -= 1
    for lab, c in items:
        T = max(c)
        v = c[T]
        ax.errorbar(v["mean"], y, xerr=[[v["mean"] - v["ci"][0]], [v["ci"][1] - v["mean"]]], fmt="o", ms=4, lw=1.2,
                    color=col)
        if 10 in c:
            ax.plot(c[10]["mean"], y, "|", ms=7, mew=1.2, color=col, alpha=.55)
        ticks.append(y)
        labels.append(lab + ("" if T == 60 else f" ({T} s)"))
        y -= 1
    y -= 0.25
ax.set_yticks(ticks, labels, fontsize=7.6)
ax.tick_params(axis="y", length=0, labelcolor=INK)
ax.axvline(0, color=MUTED, lw=.8)
ax.axvline(1, color=GRID, lw=.8)
ax.set_xlim(-0.45, 1.15)
ax.set_xlabel(r"persistence $\alpha$ at 60 s (bar: at 10 s)")
ax.set_title("(a) long-run share of a brightness shift", fontsize=8.5, loc="left", color=INK)

SHOW = [("Self Forcing, no sink", ORANGE, "-"), ("Krea 14B, released window", ORANGE, "--"),
        ("SkyReels-V2, 17 frames", PURPLE, "-"), ("Self Forcing, sink 3 re-rotated", GREEN, "-"),
        ("Krea 14B, first-frame anchor", GREEN, "--"), ("FramePack, prefix + embedding", GREEN, ":"),
        ("Self Forcing, sink 3", BLUE, "-")]
allc = {lab: c for _, _, items in GROUPS for lab, c in items if c}
for lab, col, ls in SHOW:
    c = allc.get(lab)
    if not c:
        continue
    T = sorted(t for t in c if t >= 10)
    bx.plot(T, [c[t]["mean"] for t in T], ls=ls, color=col, lw=1.5, marker="o", ms=2.8, label=lab)
bx.axhline(0, color=MUTED, lw=.8)
bx.axhline(1, color=GRID, lw=.8)
bx.set_xticks([10, 20, 30, 40, 50, 60])
bx.set_xlabel("end of the 10-s window (s)")
bx.set_ylabel(r"$\alpha$")
bx.set_title("(b) over time", fontsize=8.5, loc="left", color=INK)
fig.legend(*bx.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=6.6, handlelength=2.2,
           columnspacing=1.2)
fig.savefig(OUT)
print("wrote", OUT, {lab: round(c[max(c)]["mean"], 2) for lab, c in allc.items()})
