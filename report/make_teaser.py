"""Figure 1: frames of one scene (gx09, generated mountain road) continued for 60 s under the generic city prompt by Self
Forcing DMD with and without an attention sink, from a darkened and a brightened version of the same opening, and the
mean brightness trajectories of the 24 scenes of preregistration 07 (3-s moving average).
usage: python report/make_teaser.py TILE_DIR LABPACK_NPZ RELAX_RESULTS_JSON OUT_PDF"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

tiles, pack, relax, out = sys.argv[1:5]  # relax unused (kept for the interface)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
ROWS = [("sf60_s3", "m16", "sink\n−16 L*"), ("sf60_s3", "p16", "sink\n+16 L*"), ("sf60_s0", "p16", "no sink\n+16 L*")]
TS = [0, 10, 30, 60]
fig = plt.figure(figsize=(10.5, 3.6))
gs = fig.add_gridspec(len(ROWS), len(TS), left=0.05, right=0.60, top=0.92, bottom=0.04, wspace=0.03, hspace=0.05)
for i, (c, v, lab) in enumerate(ROWS):
    for j, t in enumerate(TS):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(Image.open(f"{tiles}/gx09_{c}__{v}_s0__t{t}.jpg"))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_title("opening" if t == 0 else f"{t} s", fontsize=8.5)
        if j == 0:
            ax.set_ylabel(lab, fontsize=7.5)
z = np.load(pack)
ax = fig.add_axes([0.68, 0.13, 0.31, 0.79])
rel = json.load(open(relax))["per_cfg"]
scenes = sorted({k.split("_")[0] for k in z.files if "|" not in k and k[:2] in ("rx", "gx")})
t = (np.arange(969 - 9)) / 16
for c, col, lab in (("sf60_s3", "#2a78d6", "sink"), ("sf60_s0", "#898781", "no sink")):
    for v, ls in (("p16", "--"), ("z0", "-"), ("m16", ":")):
        L = np.mean([z[f"{s}_{c}__{v}_s0"][9:, 0] for s in scenes if f"{s}_{c}__{v}_s0" in z.files], axis=0)
        k = np.ones(48) / 48
        ax.plot(t[23:-24], np.convolve(L, k, "valid"), color=col, ls=ls, lw=1.6,
                label=f"{lab}: opening {dict(p16='+16', z0='unshifted', m16='−16')[v]}")
ax.set_xlabel("seconds after the opening")
ax.set_ylabel("mean L* (24 scenes)")
ax.set_ylim(top=64)
ax.legend(fontsize=6.4, loc="upper right", ncol=2, handlelength=2.2, columnspacing=0.8)
ax.set_title("Shifted openings converge; the sink keeps them apart", fontsize=8.5, loc="left")
fig.savefig(out)
print("ok teaser")
