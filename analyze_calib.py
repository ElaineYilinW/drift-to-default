"""Calibrated-lighting experiment (preregistered, docs/preregistration_calib_cn.md). Same 40 openings, models, seeds
and GPU architecture as the matched-prompt experiment; new arms CL (generic scene + brightness-calibrated lighting),
C (VLM scene + calibrated lighting), C+ttc, plus a seed-0 rerun of G to verify that pairing with the earlier arms
(G, M, ...) is exact.

Outcome as before: |dL_end| (primary), dE_end, signed dL_end, retention, flicker.
Primary (Holm over 2 models x 2): H1 C - M < 0, H2 C - G < 0 on |dL_end|; seed-averaged paired differences per
opening, two-sided one-sample t-test, opening-cluster bootstrap 95% CI.
usage: python analyze_calib.py OUT LIGHT_SUMMARY CALIB_PROMPTS MODEL=MATCH_DIR,CALIB_DIR [...]
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

out, summ, cprompts = sys.argv[1:4]
runs = {a.split("=", 1)[0]: a.split("=", 1)[1].split(",") for a in sys.argv[4:]}
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
pat = re.compile(r"(n[rg]\d+)--(G|M|ML|C|CL)_(none|ttc_s1)_s(\d+)\.npz$")
END = 32
light = {d["open"]: d for d in map(json.loads, open(cprompts)) if "light" in d}
P4 = json.load(open(summ))["per_prompt"]


def video(p):
    z = np.load(p)
    lab = z["lab_mean"]
    L = lab[:, 0]
    o, e = lab[1:9].mean(0), lab[-END:].mean(0)
    r = dict(L_open=float(o[0]), dL_end=float(e[0] - o[0]), absdL_end=float(abs(e[0] - o[0])),
             dE_end=float(np.linalg.norm(e - o)), jitter=float(np.std(np.diff(L, 2))),
             boundary=float(np.mean([abs(L[1 + 12 * k] - L[12 * k]) for k in range(1, 41)])),
             gpu=json.load(open(p.replace(".npz", ".json"))).get("gpu"))
    fp = p.replace(".npz", ".feat.npz")
    if os.path.exists(fp):
        f = np.load(fp)
        c = f["cls"].astype(np.float64)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        idx = f["dino_idx"]
        op = c[(idx >= 1) & (idx <= 8)].mean(0)
        op /= np.linalg.norm(op)
        r["retention"] = float((c[idx >= len(L) - END] @ op).mean())
    return r, lab


rows, check = [], []
for model, (mdir, cdir) in runs.items():
    for d, src in [(mdir, "match"), (cdir, "calib")]:
        for p in sorted(glob.glob(f"{d}/*.npz")):
            m = pat.search(os.path.basename(p))
            if not m:
                continue
            arm = m[2] + ("+ttc" if m[3] == "ttc_s1" else "")
            if src == "calib" and arm == "G":                        # pairing check against the earlier G run
                ref = f"{mdir}/{os.path.basename(p)}"
                a, b = np.load(p)["lab_mean"], np.load(ref)["lab_mean"]
                check.append(dict(model=model, open=m[1], max_abs_diff=float(np.abs(a - b).max()),
                                  identical=bool(np.array_equal(a, b))))
                continue
            if src == "calib" and arm not in ("C", "CL", "C+ttc"):
                continue
            r, _ = video(p)
            rows.append(dict(r, model=model, open=m[1], arm=arm, seed=int(m[4]), light=light[m[1]]["light"]))
V = pd.DataFrame(rows)
V.to_csv(f"{out}/calib_videos.csv", index=False)
C = pd.DataFrame(check)
C.to_csv(f"{out}/pairing_check.csv", index=False)
O = V.groupby(["model", "open", "light", "arm"], as_index=False).mean(numeric_only=True)


def paired(model, a, b, y, sub=None):
    g = O[O.model == model] if sub is None else O[(O.model == model) & sub(O)]
    w = g.pivot(index="open", columns="arm", values=y)
    dlt = (w[a] - w[b]).dropna().values
    bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(10000)]
    return dict(est=float(dlt.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                p=float(stats.ttest_1samp(dlt, 0).pvalue), n=int(len(dlt)), frac_lower=float((dlt < 0).mean()),
                mean_a=float(w[a].mean()), mean_b=float(w[b].mean()))


res = {"counts": {m: g.arm.value_counts().to_dict() for m, g in V.groupby("model")},
       "pairing_check": {m: dict(n=int(len(g)), identical=int(g.identical.sum()), max_abs_diff=float(g.max_abs_diff.max()))
                         for m, g in C.groupby("model")} if len(C) else {},
       "light_classes": pd.Series({o: v["light"] for o, v in light.items()}).value_counts().to_dict(),
       "primary": {}, "secondary": {}}
tests = {f"{m}:{h}": paired(m, a, b, "absdL_end") for m in runs for h, a, b in [("H1", "C", "M"), ("H2", "C", "G")]}
running = 0.0
for i, k in enumerate(sorted(tests, key=lambda k: tests[k]["p"])):
    running = max(running, min(1.0, (len(tests) - i) * tests[k]["p"]))
    tests[k]["p_holm"] = running
    tests[k]["supported"] = bool(running < 0.05 and tests[k]["est"] < 0)
res["primary"] = tests

reach = lambda g: g.light != "sunny"                          # openings whose brightness a lighting prompt can match
for m in runs:
    s = {}
    for y in ["absdL_end", "dE_end", "retention"]:
        if O[O.model == m][y].notna().any():
            s[y] = {f"{a} - {b}": paired(m, a, b, y) for a, b in
                    [("C", "M"), ("C", "G"), ("CL", "G"), ("C", "CL"), ("C+ttc", "C"), ("C+ttc", "G"), ("C+ttc", "M+ttc")]}
    s["subgroup_C_minus_G"] = {"night/dusk/overcast": paired(m, "C", "G", "absdL_end", reach),
                               "sunny": paired(m, "C", "G", "absdL_end", lambda g: ~reach(g))}
    s["subgroup_CL_minus_G"] = {"night/dusk/overcast": paired(m, "CL", "G", "absdL_end", reach),
                                "sunny": paired(m, "CL", "G", "absdL_end", lambda g: ~reach(g))}
    g = O[(O.model == m) & (O.arm == "CL")].copy()             # out-of-sample check of the Part 4 law (fitted on SF)
    g["pred"] = [P4[lt]["slope"] * (L - P4[lt]["zero_drift_level"]) for lt, L in zip(g.light, g.L_open)]
    s["part4_prediction_CL"] = dict(r2=float(1 - ((g.dL_end - g.pred) ** 2).sum() / ((g.dL_end - g.dL_end.mean()) ** 2).sum()),
                                    corr=float(np.corrcoef(g.pred, g.dL_end)[0, 1]), mean_error=float((g.dL_end - g.pred).mean()))
    base = V[(V.model == m) & (V.arm == "G")][["jitter", "boundary"]].mean()
    s["flicker_ratio"] = {a: (V[(V.model == m) & (V.arm == a)][["jitter", "boundary"]].mean() / base).round(3).to_dict()
                          for a in ["M", "CL", "C", "C+ttc"]}
    res["secondary"][m] = s
json.dump(res, open(f"{out}/calib_results.json", "w"), indent=1)

col = {"G": "#898781", "M": "#eb6834", "CL": "#2a78d6", "C": "#1baf7a", "C+ttc": "#6250d6"}
fig, axs = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.8), squeeze=False)
for ax, m in zip(axs[0], runs):
    g = O[O.model == m]
    for a in ["G", "M", "CL", "C"]:
        h = g[g.arm == a]
        ax.scatter(h.L_open, h.dL_end, s=12, color=col[a], alpha=.6, label=a)
        b1, b0 = np.polyfit(h.L_open, h.dL_end, 1)
        xs = np.linspace(h.L_open.min(), h.L_open.max(), 20)
        ax.plot(xs, b0 + b1 * xs, color=col[a], lw=1.6)
    ax.axhline(0, color="#898781", lw=.8)
    ax.set_title(m), ax.set_xlabel("opening brightness L*"), ax.set_ylabel("brightness change after 31 s (L*)")
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{out}/calib_regression.png", dpi=130)
print(json.dumps({"pairing": res["pairing_check"], "primary": res["primary"]}, indent=1))
