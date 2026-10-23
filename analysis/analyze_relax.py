"""Relaxation study (preregistered, docs/preregistration_relax_cn.md): how much of a held opening's brightness / colour
survives a long autoregressive rollout, per model and context strategy?

Design: 24 base scenes x 7 controlled versions of the opening (L* shift -16..+16 in 5 levels, b* shift -12/+12),
seed 0, generic prompt; configurations (cfg) = model x context strategy. File names: {scene}_{cfg}__{ver}_s0.npz with
ver in m16, m8, z0, p8, p16 (L*) and bm12, bp12 (b*); fresh starts: fresh_{cfg}__fresh_s{seed}.npz.

Per video: opening value = mean over the held frames (excluding frame 0), end value = mean over the last 2 s, and a
1-s time course. Persistence alpha = within-scene OLS slope of the end value on the opening value (5 L* levels for
alpha_L; b* levels -12/0/+12 for alpha_b), averaged over scenes. The zero-drift level of a scene is the opening value at
which end = opening, from the same per-scene line.
Primary (Holm over 4): H1 alpha_L(sf_s0) < alpha_L(sf_s3); H2 alpha_L(ll_s0) < alpha_L(ll_s3);
H3 alpha_L(sf_ttc) > alpha_L(sf_s3); H4 alpha_b(sf_s0) < alpha_b(sf_s3). Paired over scenes: two-sided one-sample
t-test of per-scene slope differences, scene-bootstrap 95% CI.
usage: python analyze_relax.py RUN_DIR OUT_DIR
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
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
pat = re.compile(r"^(\w+?)_(\w+?)__(m16|m8|z0|p8|p16|bm12|bp12|fresh)_s(\d+)\.npz$")
LV = {"m16": ("L", -16), "m8": ("L", -8), "z0": ("0", 0), "p8": ("L", 8), "p16": ("L", 16), "bm12": ("b", -12), "bp12": ("b", 12)}
ATT = {"L": lambda z: z["lab_mean"][:, 0], "a": lambda z: z["lab_mean"][:, 1], "b": lambda z: z["lab_mean"][:, 2],
       "sat": lambda z: z["sat_mean"]}

rows, curves = [], {}
for p in sorted(glob.glob(f"{src}/*.npz")):
    m = pat.match(os.path.basename(p))
    if not m or p.endswith(".feat.npz"):
        continue
    scene, cfg, ver, seed = m[1], m[2], m[3], int(m[4])
    info = json.load(open(p.replace(".npz", ".json")))
    fps = info.get("fps", 16)
    hold = info.get("hold_frames", 9) or 9
    z = np.load(p)
    T = len(z["lab_mean"])
    nsec = int(T // fps)
    r = dict(scene=scene, cfg=cfg, ver=ver, seed=seed, fps=fps, gpu=info.get("gpu"), seconds=T / fps)
    for an, f in ATT.items():
        x = f(z)
        r[f"{an}_open"] = float(x[1:hold].mean()) if ver != "fresh" else float(x[1:9].mean())
        r[f"{an}_end"] = float(x[-2 * fps:].mean())
        curves[(scene, cfg, ver, seed, an)] = np.array([x[int(s * fps):int((s + 1) * fps)].mean() for s in range(nsec)])
    rows.append(r)
V = pd.DataFrame(rows)
V.to_csv(f"{out}/relax_videos.csv", index=False)
P = V[V.ver != "fresh"]


def slopes(cfg, an, end_col=None, t=None):
    """Per-scene OLS slope of end (or the 1-s mean at second t) on the opening value."""
    vers = ["m16", "m8", "z0", "p8", "p16"] if an == "L" else ["bm12", "z0", "bp12"]
    res = {}
    for s, g in P[(P.cfg == cfg) & P.ver.isin(vers)].groupby("scene"):
        if len(g) < len(vers):
            continue
        x = g[f"{an}_open"].values
        if t is None:
            y = g[f"{an}_end"].values
        else:
            y = np.array([curves[(s, cfg, v, sd, an)][t] if len(curves[(s, cfg, v, sd, an)]) > t else np.nan
                          for v, sd in zip(g.ver, g.seed)])
            if np.isnan(y).any():
                continue
        b1, b0 = np.polyfit(x, y, 1)
        res[s] = dict(slope=b1, zero=(b0 / (1 - b1)) if abs(1 - b1) > 1e-6 else np.nan)
    return pd.DataFrame(res).T


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    bs = [rng.choice(v, len(v)).mean() for _ in range(10000)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


cfgs = sorted(P.cfg.unique())
res = {"counts": P.groupby("cfg").size().to_dict(), "per_cfg": {}, "primary": {}}
for c in cfgs:
    d = {}
    for an in ["L", "b", "a", "sat"]:
        sl = slopes(c, "L" if an in ("L",) else ("b" if an == "b" else an)) if an in ("L", "b") else None
        if an in ("a", "sat"):      # cross-attribute persistence: slope of a* / saturation end on their own opening
            sl = None              # values under the L* series (their openings move with the L* shift)
            vers = ["m16", "m8", "z0", "p8", "p16"]
            rr = []
            for s, g in P[(P.cfg == c) & P.ver.isin(vers)].groupby("scene"):
                if len(g) == 5 and np.ptp(g[f"{an}_open"]) > 1e-6:
                    rr.append(np.polyfit(g[f"{an}_open"], g[f"{an}_end"], 1)[0])
            if rr:
                d[f"alpha_{an}"] = dict(mean=float(np.mean(rr)), ci=ci(rr), n=len(rr))
            continue
        if len(sl):
            d[f"alpha_{an}"] = dict(mean=float(sl.slope.mean()), ci=ci(sl.slope), n=int(len(sl)))
            d[f"zero_{an}"] = dict(median=float(np.nanmedian(sl.zero.astype(float))), ci=ci(sl.zero.astype(float)))
    # time course of alpha_L and a saturating-exponential fit alpha(t) = a_inf + (1 - a_inf) exp(-t / tau)
    tc = []
    nsec = int(P[P.cfg == c].seconds.min())
    for t in range(1, nsec):
        sl = slopes(c, "L", t=t)
        if len(sl):
            tc.append((t + 0.5, float(sl.slope.mean())))
    if len(tc) > 4:
        tt, aa = map(np.array, zip(*tc))
        try:
            (ainf, tau), _ = curve_fit(lambda t, ai, ta: ai + (1 - ai) * np.exp(-t / ta), tt, aa, p0=[0.3, 5],
                                       bounds=([-1, 0.1], [1.5, 500]))
            d["fit"] = dict(alpha_inf=float(ainf), tau_s=float(tau), half_life_s=float(tau * np.log(2)))
        except RuntimeError:
            pass
        d["alpha_L_t"] = tc
    F = V[(V.ver == "fresh") & (V.cfg == c)]
    if len(F):
        d["fresh"] = dict(n=int(len(F)), L_start=float(F.L_open.mean()), L_end=float(F.L_end.mean()))
    res["per_cfg"][c] = d


def contrast(c1, c0, an, sign):
    s1, s0 = slopes(c1, an), slopes(c0, an)
    common = s1.index.intersection(s0.index)
    dl = (s1.loc[common].slope - s0.loc[common].slope).astype(float).values
    return dict(est=float(dl.mean()), ci=ci(dl), p=float(stats.ttest_1samp(dl, 0).pvalue), n=int(len(dl)),
                expected_sign=sign, mean_1=float(s1.loc[common].slope.mean()), mean_0=float(s0.loc[common].slope.mean()))


H = {"H1": ("sf_s0", "sf_s3", "L", "-"), "H2": ("ll_s0", "ll_s3", "L", "-"), "H3": ("sf_ttc", "sf_s3", "L", "+"),
     "H4": ("sf_s0", "sf_s3", "b", "-")}
tests = {k: contrast(*v) for k, v in H.items() if v[0] in cfgs and v[1] in cfgs}
running = 0.0
for i, k in enumerate(sorted(tests, key=lambda k: tests[k]["p"])):
    running = max(running, min(1.0, (len(tests) - i) * tests[k]["p"]))
    tests[k]["p_holm"] = running
    sgn = tests[k]["est"] < 0 if tests[k]["expected_sign"] == "-" else tests[k]["est"] > 0
    tests[k]["supported"] = bool(running < 0.05 and sgn)
res["primary"] = tests
json.dump(res, open(f"{out}/relax_results.json", "w"), indent=1, default=float)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for c in cfgs:
    tc = res["per_cfg"][c].get("alpha_L_t")
    if tc:
        t, a_ = zip(*tc)
        ax.plot(t, a_, lw=1.8, label=c)
ax.axhline(0, color="#898781", lw=.8)
ax.set_xlabel("seconds"), ax.set_ylabel("persistence of opening brightness (within-scene slope)")
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(f"{out}/relax_persistence.png", dpi=130)
print(json.dumps({k: {kk: (vv if not isinstance(vv, float) else round(vv, 3)) for kk, vv in v.items()} for k, v in tests.items()}, indent=1))
for c in cfgs:
    d = res["per_cfg"][c]
    print(c, {k: (round(v["mean"], 3) if "mean" in v else v) for k, v in d.items() if k.startswith("alpha_") and k != "alpha_L_t"},
          d.get("fit"), d.get("zero_L"))
