"""Matched-prompt experiment (preregistered, docs/preregistration_match_cn.md): does a prompt that describes the
opening frame reduce drift, and does the correction add to it? Two models, 40 new held openings, 2 seeds,
5 arms: G (generic prompt), G+ttc, M (VLM caption: scene + lighting), M+ttc, ML (generic scene + caption lighting).

Per video, relative to the held opening (frames 1-8) and the last 2 s (32 frames): |dL_end| (primary), dE_end,
signed dL_end, flicker (jitter, chunk-boundary jump), content retention (DINOv2 CLS cosine between the last-2-s frames
and the opening frames, if .feat.npz exists).
Primary tests (per model, Holm over 2 models x 2 hypotheses): H1 M - G < 0 and H2 M+ttc - M < 0 on |dL_end|;
seed-averaged paired differences per opening, two-sided one-sample t-test; opening-cluster bootstrap 95% CI.
usage: python analyze_match.py OUT_DIR MODEL=RUN_DIR [MODEL=RUN_DIR ...]
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

out = sys.argv[1]
runs = dict(a.split("=", 1) for a in sys.argv[2:])
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
pat = re.compile(r"(n[rg]\d+)--(G|M|ML)_(none|ttc_s1)_s(\d+)\.npz$")
ARM = {("G", "none"): "G", ("G", "ttc_s1"): "G+ttc", ("M", "none"): "M", ("M", "ttc_s1"): "M+ttc", ("ML", "none"): "ML"}
END = 32

rows = []
for model, d in runs.items():
    for p in sorted(glob.glob(f"{d}/*.npz")):
        m = pat.search(os.path.basename(p))
        if not m or (m[2], m[3]) not in ARM:
            continue
        z = np.load(p)
        lab = z["lab_mean"]
        L = lab[:, 0]
        o, e = lab[1:9].mean(0), lab[-END:].mean(0)
        r = dict(model=model, open=m[1], set="real" if m[1].startswith("nr") else "generated", arm=ARM[(m[2], m[3])],
                 seed=int(m[4]), gpu=json.load(open(p.replace(".npz", ".json"))).get("gpu"), L_open=float(o[0]),
                 dL_end=float(e[0] - o[0]), absdL_end=float(abs(e[0] - o[0])), dE_end=float(np.linalg.norm(e - o)),
                 jitter=float(np.std(np.diff(L, 2))),
                 boundary=float(np.mean([abs(L[1 + 12 * k] - L[12 * k]) for k in range(1, 41)])))
        fp = p.replace(".npz", ".feat.npz")
        if os.path.exists(fp):
            f = np.load(fp)
            c = f["cls"].astype(np.float64)
            c /= np.linalg.norm(c, axis=1, keepdims=True)
            idx = f["dino_idx"]
            op = c[(idx >= 1) & (idx <= 8)].mean(0)
            op /= np.linalg.norm(op)
            r["retention"] = float((c[idx >= len(L) - END] @ op).mean())
        rows.append(r)
V = pd.DataFrame(rows)
V.to_csv(f"{out}/match_videos.csv", index=False)
O = V.groupby(["model", "open", "set", "arm"], as_index=False).mean(numeric_only=True)   # seed-averaged per opening


def paired(model, a, b, y):
    """mean over openings of (a - b) on y, with t-test and opening-cluster bootstrap CI."""
    w = O[O.model == model].pivot(index="open", columns="arm", values=y)
    dlt = (w[a] - w[b]).dropna().values
    bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(10000)]
    return dict(est=float(dlt.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                p=float(stats.ttest_1samp(dlt, 0).pvalue), n=int(len(dlt)), frac_improved=float((dlt < 0).mean()),
                mean_a=float(w[a].mean()), mean_b=float(w[b].mean()))


res = {"counts": V.groupby(["model", "arm"]).size().unstack().to_dict(), "gpus": {m: g.gpu.value_counts().to_dict() for m, g in V.groupby("model")},
       "primary": {}, "secondary": {}}
prim = [(m, h, a, b) for m in runs for h, a, b in [("H1", "M", "G"), ("H2", "M+ttc", "M")]]
tests = {f"{m}:{h}": paired(m, a, b, "absdL_end") for m, h, a, b in prim}
order = sorted(tests, key=lambda k: tests[k]["p"])
running = 0.0
for i, k in enumerate(order):                       # Holm step-down
    running = max(running, min(1.0, (len(order) - i) * tests[k]["p"]))
    tests[k]["p_holm"] = running
    tests[k]["supported"] = bool(running < 0.05 and tests[k]["est"] < 0)
res["primary"] = tests

sec = [("M", "G"), ("M+ttc", "M"), ("ML", "G"), ("G+ttc", "G"), ("M", "G+ttc"), ("M+ttc", "G")]
for m in runs:
    s = {}
    for y in ["absdL_end", "dE_end", "retention"]:
        if y in O and O[O.model == m][y].notna().any():
            s[y] = {f"{a} - {b}": paired(m, a, b, y) for a, b in sec}
    w = O[O.model == m].pivot(index="open", columns="arm", values="absdL_end")
    s["interaction_ttc_x_prompt"] = float(((w["M+ttc"] - w["M"]) - (w["G+ttc"] - w["G"])).mean())
    for st in ["real", "generated"]:
        ww = O[(O.model == m) & (O.set == st)].pivot(index="open", columns="arm", values="absdL_end")
        d = (ww["M"] - ww["G"]).values
        s[f"H1_{st}"] = dict(est=float(d.mean()), n=int(len(d)), p=float(stats.ttest_1samp(d, 0).pvalue))
    reg = {}
    for arm in ["G", "M", "ML"]:
        g = O[(O.model == m) & (O.arm == arm)]
        b1, b0 = np.polyfit(g.L_open, g.dL_end, 1)
        reg[arm] = dict(slope=float(b1), zero_drift=float(-b0 / b1), r2=float(np.corrcoef(g.L_open, g.dL_end)[0, 1] ** 2))
    s["regression_to_default"] = reg
    g = O[(O.model == m) & (O.arm == "G")].set_index("open")
    gain = (O[(O.model == m) & (O.arm == "M")].set_index("open").absdL_end - g.absdL_end)
    mis = (g.L_open - reg["G"]["zero_drift"]).abs()
    s["gain_vs_mismatch_slope"] = float(np.polyfit(mis.loc[gain.index], gain, 1)[0])
    base = V[(V.model == m) & (V.arm == "G")][["jitter", "boundary"]].mean()
    s["flicker_ratio"] = {a: (V[(V.model == m) & (V.arm == a)][["jitter", "boundary"]].mean() / base).round(3).to_dict()
                          for a in ARM.values()}
    res["secondary"][m] = s
json.dump(res, open(f"{out}/match_results.json", "w"), indent=1)

C = {"G": "#898781", "G+ttc": "#2a78d6", "M": "#eb6834", "M+ttc": "#1baf7a", "ML": "#eda100"}
arms = ["G", "G+ttc", "ML", "M", "M+ttc"]
fig, axs = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.8), squeeze=False)
for ax, m in zip(axs[0], runs):
    g = O[O.model == m]
    mu = [g[g.arm == a].absdL_end.mean() for a in arms]
    ci = [np.percentile([rng.choice(g[g.arm == a].absdL_end.values, (g.arm == a).sum()).mean() for _ in range(2000)], [2.5, 97.5])
          for a in arms]
    ax.bar(range(5), mu, color=[C[a] for a in arms], width=.65)
    ax.errorbar(range(5), mu, yerr=np.array([[x - c[0], c[1] - x] for x, c in zip(mu, ci)]).T, fmt="none", ecolor="#0b0b0b", lw=1)
    ax.set_xticks(range(5), arms), ax.set_title(m), ax.set_ylabel("|brightness change| after 31 s (L*)")
fig.tight_layout()
fig.savefig(f"{out}/match_bars.png", dpi=130)
print(json.dumps(res["primary"], indent=1))
