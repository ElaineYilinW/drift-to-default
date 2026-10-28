"""Appendix figure: frames of two scenes (gx00, rx00; +16 L* opening) continued for 60 s by Self Forcing DMD with a
three-frame sink at its original positions and held at query-sink distances 203 (a probe peak), 187 (the trough before
it) and 20 (next to the window), study 25. Row labels give the 24-scene persistence and mean frame-to-frame change
(motion, 0-100 grey levels) over 50-60 s from s25_results.json.
usage: python report/make_s25_frames_fig.py [TILE_DIR] [S25_RESULTS_JSON] [OUT_PDF]
       (defaults: report/tiles/s25, results/corr/s25_out/s25_results.json, paper/fig_s25frames.pdf)"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
tiles = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tiles", "s25")
res = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results", "corr", "s25_out", "s25_results.json")
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "paper", "fig_s25frames.pdf")
MUTED = "#898781"
plt.rcParams.update({"font.size": 8})
R = json.load(open(res))
ROWS = [("sp_abs", "original"), ("sp_d203", "held at 203 (peak)"), ("sp_d187", "held at 187 (trough)"),
        ("sp_d20", "held at 20")]
SCENES, TS = ("gx00", "rx00"), (0, 10, 20, 40, 60)
fig = plt.figure(figsize=(6.9, 2.05))
gs = fig.add_gridspec(len(ROWS), 2 * len(TS) + 1, left=0.155, right=0.995, top=0.88, bottom=0.01, wspace=0.03,
                      hspace=0.06, width_ratios=[1] * len(TS) + [0.12] + [1] * len(TS))
for i, (arm, lab) in enumerate(ROWS):
    for k, s in enumerate(SCENES):
        for j, t in enumerate(TS):
            ax = fig.add_subplot(gs[i, j + k * (len(TS) + 1)])
            ax.imshow(Image.open(f"{tiles}/{s}_{arm}__p16_s0__t{t}.jpg"))
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title("opening" if t == 0 else f"{t} s", fontsize=7.5)
            if j == 0 and k == 0:
                a60 = R["alpha"][arm]["60"]["mean"]
                mo = R["motion_50_60"][arm]["motion"]
                ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="bottom", labelpad=4)
                ax.text(-0.06, 0.1, rf"$\alpha$ {a60:.2f}, motion {mo:.1f}", transform=ax.transAxes, ha="right",
                        va="top", fontsize=5.8, color=MUTED)
fig.savefig(out, dpi=300)
print("wrote", out)
