"""Exploratory: fit the chunk-recursion model (ar_equilibrium.py) to the interventional persistence curves of the
relaxation study, per configuration, and compare the fitted per-chunk weights with the KV-cache replay estimates
(w_sink 0.053 [0.040, 0.066] synthetic / 0.036 dose; w_win 0.76 synthetic, 0.90-0.99 natural).

alpha(t) per chunk (0.75 s bins) = mean over scenes of the within-scene slope of the bin mean on the opening value
(5 L* levels); scene-bootstrap CIs for the fitted weights.
usage: python analyze_relax_theory.py RELAX_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_equilibrium import CHUNK_S, alpha_curve, fit_curve  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
LV = {"m16": -16, "m8": -8, "z0": 0, "p8": 8, "p16": 16}
# rolling-window length in chunks (latent frames / 3) and whether the opening is kept as a sink
WIN = {"sf_s3": (6, True), "sf_s1": (7, True), "sf_s0": (7, False), "sf_ttc": (6, True), "sid_s3": (6, True),
       "sidv2_s3": (6, True), "gan_s3": (6, True), "sf10s_s3": (6, True), "cv_s3": (6, True), "cv_ttc": (6, True),
       "ll_s3": (3, True), "ll_s0": (4, False), "ll_ttc": (3, True), "rf": (6, True)}
pat = re.compile(r"^(\w+?)_(\w+?)__(m16|m8|z0|p8|p16)_s0\.npz$")
data = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if not m or m[2] not in WIN:
        continue
    L = np.load(p)["lab_mean"][:, 0]
    nb = (len(L) - 9) // 12
    data.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = (L[1:9].mean(), np.array([L[9 + 12 * b:21 + 12 * b].mean() for b in range(nb)]))


def curve(scenes, d):
    sl = []
    for s in scenes:
        v = d[s]
        if len(v) < 5:
            continue
        x = np.array([v[k][0] for k in LV])
        Y = np.array([v[k][1] for k in LV])                     # 5 x bins
        sl.append([np.polyfit(x, Y[:, b], 1)[0] for b in range(Y.shape[1])])
    a = np.array(sl).mean(0)
    t = (np.arange(len(a)) + 1) * CHUNK_S
    return t, a


res = {}
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for cfg, d in sorted(data.items()):
    scenes = sorted(d)
    win, sink = WIN[cfg]
    t, a = curve(scenes, d)
    f = fit_curve(t, a, win, sink, w_s=None if sink else 0.0)
    bs = []
    for _ in range(300):
        tb, ab = curve(list(rng.choice(scenes, len(scenes))), d)
        g = fit_curve(tb, ab, win, sink, w_s=None if sink else 0.0)
        bs.append([g["w_w"], g["w_s"], g["alpha_inf"]])
    bs = np.array(bs)
    f["ci"] = {k: np.percentile(bs[:, i], [2.5, 97.5]).round(3).tolist() for i, k in enumerate(["w_w", "w_s", "alpha_inf"])}
    f["n_scenes"] = len(scenes)
    f["alpha_obs_at"] = {s: round(float(np.interp(s, t, a)), 3) for s in (5, 10, 20, 30)}
    res[cfg] = f
    line, = ax.plot(t, a, lw=1.6, label=cfg)
    ax.plot(t, alpha_curve(f["w_w"], f["w_s"], win, sink, len(t) + 1)[1:], ls="--", lw=1, color=line.get_color())
ax.axhline(0, color="#898781", lw=.8)
ax.set_xlabel("seconds"), ax.set_ylabel("persistence of opening brightness")
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(f"{out}/relax_theory_fits.png", dpi=140)
json.dump(res, open(f"{out}/relax_theory.json", "w"), indent=1)
for c, f in res.items():
    print(f"{c:9s} w_w {f['w_w']:.3f} {f['ci']['w_w']}  w_s {f['w_s']:.3f} {f['ci']['w_s']}  alpha_inf {f['alpha_inf']:.2f}  rmse {f['rmse']:.3f}  obs {f['alpha_obs_at']}")
