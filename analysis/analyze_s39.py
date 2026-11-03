"""Study 39: on prompt-only 60-s videos, does the sink held at D* drift less in colour and lose less imaging quality
than the re-aligned sink? (preregistered, docs/preregistrations/39_held_sink_quality.md).

Files in RUN_DIR: {prompt}_{m}_{s0|rr|held|abs}_s13.npz (generator statistics: per-frame CIELAB mean, motion) and
.vbl.npz (vbench_lite.py: per 10-s segment k = frames [9 + 160k, 9 + 160(k + 1)) imaging, aesthetic, subject,
background, flow), m in sf, ll, cv, gan; {prompt}_rf_s13.* for Rolling Forcing.
  colour offset = |mean L* over seconds [50, 60) - mean L* over seconds [0, 5)| (frames 800..959 and 0..79)
  imaging change = imaging(segment 5) - imaging(segment 0)
Primary, for every model with a D* (Holm over all of them; per-prompt values; two-sided one-sample t-tests, supported
if Holm p < .05 and the estimate > 0; 95% CIs by prompt bootstrap, 10,000 draws):
  B1_m  held drifts less in colour:          offset(rr) - offset(held) > 0
  B2_m  held loses less imaging quality:     imaging change(held) - imaging change(rr) > 0
Secondary: per model and arm (and Rolling Forcing) the mean of every dimension in segments 0 and 5 and its change, the
colour offset, motion and flow over [50, 60) s, the share of videos with motion < 1.5 over [50, 60) s; held against s0
and abs on the two primary measures and on flow; the same with the driving and the MovieGen prompts separately.
usage: python analyze_s39.py RUN_DIR DSTAR.json OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, dstar, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, FROZEN = 16, 1.5
DIMS = ("imaging", "aesthetic", "subject", "background", "flow")
sel = json.load(open(dstar))
MODELS = [m for m in ("sf", "ll", "cv", "gan") if sel.get(m, {}).get("dstar") is not None]
ARMS = ("s0", "rr", "held", "abs")
pat = re.compile(r"^(\w+?)_(sf|ll|cv|gan|rf)(?:_(s0|rr|held|abs))?_s13\.npz$")

V = {}   # V[(m, arm)][prompt] = dict of measures
for p in glob.glob(f"{src}/*_s13.npz"):
    b = os.path.basename(p)
    if b.endswith(".vbl.npz") or b.endswith(".feat.npz"):
        continue
    x = pat.match(b)
    if not x:
        continue
    key = (x[2], x[3] or "ref")
    z = np.load(p)
    L = z["lab_mean"][:, 0]
    if len(L) < 60 * FPS:
        continue
    r = {"offset": abs(float(L[50 * FPS:60 * FPS].mean()) - float(L[:5 * FPS].mean()))}
    if "motion" in z:   # deviation 1: rf_gen_long.py (Rolling Forcing) records no per-frame motion
        r["motion"] = float(z["motion"][50 * FPS:60 * FPS].mean())
    q = p[:-4] + ".vbl.npz"
    if os.path.exists(q):
        w = np.load(q)
        for d in DIMS:
            r[f"{d}_0"], r[f"{d}_5"] = float(w[d][0]), float(w[d][5])
            r[f"{d}_change"] = r[f"{d}_5"] - r[f"{d}_0"]
    V.setdefault(key, {})[x[1]] = r


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def ttest(d):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def meas(key, name, subset=None):
    return {p: r[name] for p, r in V.get(key, {}).items() if name in r and (subset is None or p in subset)}


def pdiff(x, y):
    return {p: x[p] - y[p] for p in x if p in y}


def holm(t):
    P = {k: dict(x, side=s) for k, (x, s) in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(P) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and np.sign(P[k]["est"]) == P[k]["side"])
    return P


def summ(d):
    return dict(mean=float(np.mean(list(d.values()))), ci=ci(list(d.values())), n=len(d)) if d else None


tests = {}
for m in MODELS:
    tests[f"B1_{m}"] = (ttest(pdiff(meas((m, "rr"), "offset"), meas((m, "held"), "offset"))), +1)
    tests[f"B2_{m}"] = (ttest(pdiff(meas((m, "held"), "imaging_change"), meas((m, "rr"), "imaging_change"))), +1)

drv = {p for k in V for p in V[k] if p.startswith("drv_")}
mvg = {p for k in V for p in V[k] if not p.startswith("drv_")}
NAMES = ["offset", "motion"] + [f"{d}_{s}" for d in DIMS for s in ("0", "5", "change")]
table = {f"{m}_{a}": {n: summ(meas((m, a), n)) for n in NAMES} for m in MODELS for a in ARMS}
table["rf"] = {n: summ(meas(("rf", "ref"), n)) for n in NAMES}
frz = {f"{m}_{a}": (float(np.mean([r["motion"] < FROZEN for r in V.get((m, a), {}).values()]))
                    if V.get((m, a)) else None) for m in MODELS for a in ARMS}
comp = {}
for m in MODELS:
    for other in ("s0", "abs"):
        comp[f"{m}_held_vs_{other}"] = {
            "offset_other_minus_held": ttest(pdiff(meas((m, other), "offset"), meas((m, "held"), "offset"))),
            "imaging_change_held_minus_other": ttest(pdiff(meas((m, "held"), "imaging_change"),
                                                           meas((m, other), "imaging_change"))),
            "flow5_held_minus_other": ttest(pdiff(meas((m, "held"), "flow_5"), meas((m, other), "flow_5")))}
    comp[f"{m}_held_vs_rr_flow5"] = ttest(pdiff(meas((m, "held"), "flow_5"), meas((m, "rr"), "flow_5")))
    for nm, sub in (("driving", drv), ("moviegen", mvg)):
        comp[f"{m}_{nm}"] = {
            "B1": ttest(pdiff(meas((m, "rr"), "offset", sub), meas((m, "held"), "offset", sub))),
            "B2": ttest(pdiff(meas((m, "held"), "imaging_change", sub), meas((m, "rr"), "imaging_change", sub)))}
res = {"dstar": {m: sel[m]["dstar"] for m in MODELS},
       "counts": {f"{k[0]}_{k[1]}": [len(v), sum("imaging_0" in r for r in v.values())] for k, v in V.items()},
       "primary": holm(tests), "table": table, "frozen_50_60": frz, "comparisons": comp}
json.dump(res, open(f"{out}/s39_results.json", "w"), indent=1)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
for k, v in table.items():
    print(k, {n: round(v[n]["mean"], 3) for n in ("offset", "imaging_change", "aesthetic_change", "subject_5",
                                                  "flow_5") if v.get(n)})
