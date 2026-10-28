"""Study 26: does the attention probe predict when a sink at its original positions bursts between 60 and 180 s?
(preregistered, docs/preregistrations/26_burst_prediction.md; data of study 24, analysed with this frozen script).

Files {scene}_{arm}__{m16|p16}_s0.npz in RUN_DIR (study 24; 16 fps; frames 0..8 are the held opening), arm in sf180_s3,
ll180_s3, sf180_s3rr. alpha over [t0, t1) s = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L* of frames
1..8, end = mean L* over seconds [t0, t1) after the opening; undefined if open_p16 - open_m16 < 4.
Predicted onsets: the second probe (jobs/run_sinkprobe_long.sh, distances 3..723; peaks by the rule of
analyze_sinkprobe_long.py, prominence >= 0.04, fixed before its output was read) put the major peaks of attention to the
sink at the distances PEAKS (below; fixed before any study-24 output was looked at); a sink at its original positions
reaches distance P at tau = (P - 5) / 4 s after the opening. Only onsets with 62.5 <= tau and tau + 2.5 <= 180 are used
(the new predictions beyond 60 s; 13 onsets). Burst contrast of one scene: mean over onsets of
alpha([tau + 0.5, tau + 2.5)) - alpha([tau - 2.5, tau - 0.5)). Control onsets: the midpoints between consecutive peaks'
onsets, with the same range rule (12 onsets).
Primary (Holm over 3; per-scene values; two-sided one-sample t-tests; scene-bootstrap 95% CIs (10,000 draws); supported
if Holm p < .05 and the estimate is > 0):
  B1 Self Forcing bursts at the predicted onsets:       contrast(sf180_s3, predicted) > 0
  B2 LongLive bursts at the same onsets:                contrast(ll180_s3, predicted) > 0
  B3 the timing is specific (Self Forcing):             contrast(sf180_s3, predicted) - contrast(sf180_s3, control) > 0
Secondary: the contrast of sf180_s3rr at the predicted onsets; B1-B3 over the onsets before 60 s (in-sample); times of
1-s jumps above 0.15 per arm, and how many fall within 3 s after a predicted onset against the share of seconds that
lie there; per-onset contrasts; correlation over t = 62..179 s (and 6..59 s) between the first differences of an arm's
mean 1-s alpha curve at t and of the probe's sink share at the natural distances at t - lag, lag = 0..3 s (PROBE_JSON;
share per second as in analyze_s25.py; expected lag 1-2 s).
Secondary replication: the same analysis with TMAX = 119 on the 120-s subset of study 06 (6 scenes; its sf_s3 and ll_s3
files linked under the names sf180_s3 and ll180_s3); onsets and controls then follow the same range rule up to 119 s.
usage: python analyze_s26.py RUN_DIR OUT_DIR [PROBE_JSON [TMAX]]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
probe = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
FPS, HOLD = 16, 9
TMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 180  # 119 for the 120-s subset of study 06
PEAKS = (71, 90, 134, 203, 266, 309, 335, 404, 441, 467, 510, 536, 574, 599, 642, 669, 711)  # sinkprobe_long_out.json
ARMS = ("sf180_s3", "ll180_s3", "sf180_s3rr")
pat = re.compile(r"^(\w+?)_(sf180_s3|ll180_s3|sf180_s3rr)__(m16|p16)_s0\.npz$")

L = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        L.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = np.load(p)["lab_mean"][:, 0]


def endL(v, t0, t1):
    i0, i1 = HOLD + int(round(t0 * FPS)), HOLD + int(round(t1 * FPS))
    return float(v[i0:i1].mean()) if i1 <= len(v) else None


def alpha(arm, t0, t1):
    o = {}
    for s, x in L.get(arm, {}).items():
        if "m16" in x and "p16" in x:
            d = float(x["p16"][1:HOLD].mean() - x["m16"][1:HOLD].mean())
            eh, el = endL(x["p16"], t0, t1), endL(x["m16"], t0, t1)
            if d >= 4 and eh is not None and el is not None:
                o[s] = (eh - el) / d
    return o


ALL = [(P - 5) / 4 for P in PEAKS]
PRED = [t for t in ALL if 62.5 <= t and t + 2.5 <= TMAX]
CTRL = [(a + b) / 2 for a, b in zip(ALL, ALL[1:]) if 62.5 <= (a + b) / 2 and (a + b) / 2 + 2.5 <= TMAX]
EARLY = [t for t in ALL if 5 <= t - 2.5 and t + 2.5 <= 60]
EARLY_CTRL = [(a + b) / 2 for a, b in zip(ALL, ALL[1:]) if 5 <= (a + b) / 2 - 2.5 and (a + b) / 2 + 2.5 <= 60]


def contrast(arm, onsets):
    per = [(alpha(arm, t + 0.5, t + 2.5), alpha(arm, t - 2.5, t - 0.5)) for t in onsets]
    sc = set.intersection(*[set(a) & set(b) for a, b in per]) if per else set()
    return {s: float(np.mean([a[s] - b[s] for a, b in per])) for s in sc}


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def ttest(d):
    if len(d) < 3:
        return None
    v = np.array(list(d.values()), float)
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def pdiff(x, y):
    return {s: x[s] - y[s] for s in x if s in y}


def holm(t):
    P = {k: dict(x) for k, x in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(t) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and P[k]["est"] > 0)
    return P


def tests(pred, ctrl):
    return {"B1": ttest(contrast("sf180_s3", pred)), "B2": ttest(contrast("ll180_s3", pred)),
            "B3": ttest(pdiff(contrast("sf180_s3", pred), contrast("sf180_s3", ctrl)))}


def curve1(arm):
    return {t: float(np.mean(list(x.values()))) for t in range(1, TMAX + 1) if (x := alpha(arm, t - 1, t))}


def jumps(arm, thr=0.15):
    c = curve1(arm)
    return [t for t in sorted(c) if t - 1 in c and c[t] - c[t - 1] > thr]


def hits(arm, onsets, lo=62, hi=TMAX):
    """1-s jumps (key t: window [t - 1, t)) in (lo, hi] within 3 s after a predicted onset (tau < t <= tau + 3)"""
    near = lambda t: any(tau < t <= tau + 3 for tau in onsets)
    j = [t for t in jumps(arm) if lo < t <= hi]
    return dict(jumps=len(j), near=sum(map(near, j)), chance=float(np.mean([near(t) for t in range(lo + 1, hi + 1)])))


def probe_share():
    """mean probe share over the chunks starting in each second [t - 1, t): chunk c starts (c - 3) / 4 s after the opening"""
    g, m = probe["grid"], probe["mean"]
    o = {}
    for t in range(1, TMAX + 1):
        v = [m[g.index(c + 2)] for c in range(3, 723, 3) if t - 1 <= (c - 3) / 4 < t and c + 2 in g]
        if v:
            o[t] = float(np.mean(v))
    return o


def probe_corr():
    """first differences of the 1-s alpha curve at t against those of the probe share at t - lag, lag = 0..3 s (a burst
    in [tau + 0.5, tau + 2.5) reaches the 1-s curve 1-2 s after the share peaks)"""
    q, res = probe_share(), {}
    for arm in ARMS:
        c = curve1(arm)
        for lab, lo, hi in (("late", 62, TMAX), ("early", 6, 60)):
            for lag in range(4):
                ts = [t for t in range(lo, hi) if all(u in c for u in (t, t - 1)) and all(u - lag in q for u in (t, t - 1))]
                if len(ts) > 5:
                    r, p = stats.pearsonr([c[t] - c[t - 1] for t in ts], [q[t - lag] - q[t - lag - 1] for t in ts])
                    res[f"{arm}_{lab}_lag{lag}"] = dict(r=float(r), p=float(p), n=len(ts))
    return res


res = {"counts": {a: len(L.get(a, {})) for a in ARMS}, "peaks": PEAKS, "predicted": PRED, "control": CTRL,
       "primary": holm(tests(PRED, CTRL)),
       "rr_contrast": ttest(contrast("sf180_s3rr", PRED)),
       "early": {"onsets": EARLY, "control": EARLY_CTRL, "tests": tests(EARLY, EARLY_CTRL)},
       "jumps": {a: jumps(a) for a in ARMS},
       "jump_hits": {a: hits(a, PRED) for a in ARMS},
       "probe_corr": probe_corr() if probe else None,
       "per_onset": {a: {t: ttest(contrast(a, [t])) for t in PRED} for a in ARMS},
       "alpha_1s": {a: curve1(a) for a in ARMS}}
json.dump(res, open(f"{out}/s26_results.json", "w"), indent=1, default=float)
print("predicted onsets:", PRED, "control:", CTRL)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
print("jumps:", res["jumps"])
print("jump hits:", res["jump_hits"])
print("probe corr:", res["probe_corr"])
