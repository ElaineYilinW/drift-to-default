"""Study 25: do the bursts of a sink at its absolute positions follow the sink's rotary distance, and does holding the
sink at a probe peak keep more of the opening than holding it at a trough? (preregistered,
docs/preregistrations/25_sink_position.md).

Files {scene}_{arm}__{m16|p16}_s0.npz in RUN_DIR (16 fps; frames 0..8 are the held opening), arm in sp_abs, sp_off18,
sp_off36, sp_d203, sp_d187, sp_d20. alpha over [t0, t1) s = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L*
of frames 1..8, end = mean L* over seconds [t0, t1) after the opening; undefined if open_p16 - open_m16 < 4. No
collapse exclusion.
Burst onsets. The exploratory probe (jobs/run_sinkprobe.sh) put the peaks of attention to the sink at query-sink
distances P = 71, 134, 203, 266 latent frames. A chunk starting at latent frame c sits at distance c + 2 + delta from
sink frame 0, where delta is the extra offset of the sink (0, 18, 36), and starts (c - 3) / 4 s after the opening, so
the bursts start at tau_k(delta) = (P_k - 5 - delta) / 4 s. Burst window W_k = [tau_k + 0.5, tau_k + 2.5) s, kept if
5 <= tau_k + 0.5 and tau_k + 2.5 <= 60. A_X(delta) = mean over k of alpha_X(W_k(delta)) (one value per scene).
Primary (Holm over 4; per-scene values; two-sided one-sample t-tests; scene-bootstrap 95% CIs (10,000 draws);
supported if Holm p < .05 and the estimate is > 0):
  H1 bursts follow an 18-frame shift:  [A_off18(18) - A_off18(0)] - [A_abs(18) - A_abs(0)] > 0
  H2 bursts follow a 36-frame shift:   [A_off36(36) - A_off36(0)] - [A_abs(36) - A_abs(0)] > 0
  H3 a peak distance keeps more than a trough:   alpha(sp_d203, [50, 60)) - alpha(sp_d187, [50, 60)) > 0
  H4 a peak distance keeps more than distance 20: alpha(sp_d203, [50, 60)) - alpha(sp_d20, [50, 60)) > 0
Secondary: H3, H4 over [40, 50); each arm's own burst contrast mean_k [alpha(W_k) - alpha([tau_k - 2.5, tau_k - 0.5))]
at its own onsets (abs, off18, off36) and at the abs onsets (d203, d187, d20); times of 1-s jumps above 0.15 per arm;
alpha(T) for T = 10..60 (10-s windows) and 1-s curves per arm; correlation over t = 6..59 s between the first
differences of an arm's mean 1-s alpha curve and of the probe's sink share at the arm's distances (PROBE_JSON, abs,
off18, off36, each against delta = 0, 18, 36); mean motion and open_diff over [50, 60) s per arm (both versions).
usage: python analyze_s25.py RUN_DIR OUT_DIR [PROBE_JSON]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, out = sys.argv[1], sys.argv[2]
probe = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD, TMAX = 16, 9, 60
ARMS = ("sp_abs", "sp_off18", "sp_off36", "sp_d203", "sp_d187", "sp_d20")
DELTA = {"sp_abs": 0, "sp_off18": 18, "sp_off36": 36}
PEAKS = (71, 134, 203, 266)
pat = re.compile(r"^(\w+?)_(sp_abs|sp_off18|sp_off36|sp_d203|sp_d187|sp_d20)__(m16|p16)_s0\.npz$")

L, X = {}, {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        z = np.load(p)
        L.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = z["lab_mean"][:, 0]
        if "motion" in z.files:
            X.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = (z["motion"], z["open_diff"])


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


def onsets(delta):
    return [t for P in PEAKS if 5 <= (t := (P - 5 - delta) / 4) + 0.5 and t + 2.5 <= TMAX]


def A(arm, delta):
    per = [alpha(arm, t + 0.5, t + 2.5) for t in onsets(delta)]
    sc = set.intersection(*[set(v) for v in per]) if per else set()
    return {s: float(np.mean([v[s] for v in per])) for s in sc}


def own_contrast(arm, delta):
    per = [(alpha(arm, t + 0.5, t + 2.5), alpha(arm, t - 2.5, t - 0.5)) for t in onsets(delta)]
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


def shift_test(arm, delta):
    a1, a0, b1, b0 = A(arm, delta), A(arm, 0), A("sp_abs", delta), A("sp_abs", 0)
    return pdiff(pdiff(a1, a0), pdiff(b1, b0))


def holm(t):
    P = {k: dict(x, side=s) for k, (x, s) in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(t) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and np.sign(P[k]["est"]) == P[k]["side"])
    return P


def level_tests(t0, t1):
    return {"H3": ttest(pdiff(alpha("sp_d203", t0, t1), alpha("sp_d187", t0, t1))),
            "H4": ttest(pdiff(alpha("sp_d203", t0, t1), alpha("sp_d20", t0, t1)))}


def curve1(arm):
    return {t: float(np.mean(list(x.values()))) for t in range(1, TMAX + 1) if (x := alpha(arm, t - 1, t))}


def jumps(arm, thr=0.15):
    c = curve1(arm)
    return [t for t in sorted(c) if t - 1 in c and c[t] - c[t - 1] > thr]


def probe_share(delta):
    """mean probe share over the chunks starting in each second [t - 1, t): chunk c starts (c - 3) / 4 s after the opening"""
    g, m = probe["grid"], probe["mean"]
    o = {}
    for t in range(1, TMAX + 1):
        v = [m[g.index(c + 2 + delta)] for c in range(3, 243, 3) if t - 1 <= (c - 3) / 4 < t and c + 2 + delta in g]
        if v:
            o[t] = float(np.mean(v))
    return o


def probe_corr():
    res = {}
    for arm, d0 in DELTA.items():
        c = curve1(arm)
        for d in DELTA.values():
            q = probe_share(d)
            ts = [t for t in range(6, TMAX) if all(u in c and u in q for u in (t, t - 1))]
            if len(ts) > 5:
                r, p = stats.pearsonr([c[t] - c[t - 1] for t in ts], [q[t] - q[t - 1] for t in ts])
                res[f"{arm}_vs_delta{d}"] = dict(r=float(r), p=float(p), n=len(ts))
    return res


def motion(arm, t0=50, t1=60):
    o, d = [], []
    for x in X.get(arm, {}).values():
        for mo, od in x.values():
            i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
            o.append(float(mo[i0:i1].mean()))
            d.append(float(od[i0:i1].mean()))
    return dict(motion=float(np.mean(o)), open_diff=float(np.mean(d)), n=len(o)) if o else None


prim = {"H1": (ttest(shift_test("sp_off18", 18)), +1), "H2": (ttest(shift_test("sp_off36", 36)), +1)}
prim.update({k: (v, +1) for k, v in level_tests(50, 60).items()})
res = {"counts": {a: len(L.get(a, {})) for a in ARMS},
       "onsets": {d: onsets(d) for d in (0, 18, 36)},
       "primary": holm(prim),
       "window_40_50": level_tests(40, 50),
       "own_contrast": {a: ttest(own_contrast(a, DELTA.get(a, 0))) for a in ARMS},
       "jumps": {a: jumps(a) for a in ARMS},
       "alpha": {a: {T: dict(mean=float(np.mean(list(x.values()))), ci=ci(list(x.values())), n=len(x))
                     for T in range(10, TMAX + 1, 10) if (x := alpha(a, T - 10, T))} for a in ARMS},
       "alpha_1s": {a: curve1(a) for a in ARMS},
       "probe_corr": probe_corr() if probe else None,
       "motion_50_60": {a: motion(a) for a in ARMS}}
json.dump(res, open(f"{out}/s25_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
print("onsets:", res["onsets"])
print("alpha at 60:", {a: round(res["alpha"][a][60]["mean"], 2) for a in ARMS if 60 in res["alpha"][a]})
print("jumps:", res["jumps"])
