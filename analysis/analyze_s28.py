"""Study 28: the sink's rotary position in LongLive: do its bursts move when the sink is moved, and does a sink held
at one distance keep as much, and move as little, as LongLive's attention probe predicts? (preregistered,
docs/preregistrations/28_longlive_sink_position.md).

Files {scene}_{arm}__{m16|p16}_s0.npz in RUN_DIR (16 fps; frames 0..8 are the held opening): LongLive, sink 3,
window 12, ll_hold_gen_sinkpos.py in LongLive-c; arm ll_abs (original positions), ll_off18 / ll_off36 (--sink_mode
off:18 / off:36), ll_dDDD (--sink_mode dist:DDD) for the 17 distances of study 27 (s27_distances.py: 14 chosen from the
Self Forcing probe, plus 20, 187, 203). alpha over [t0, t1) s = (end_p16 - end_m16) / (open_p16 - open_m16), open =
mean L* of frames 1..8, end = mean L* over seconds [t0, t1) after the opening; undefined if open_p16 - open_m16 < 4. No
collapse exclusion. motion over [t0, t1) = mean frame-to-frame grey-level change, averaged over the two versions.
share(D) = the mean share at D of LongLive's own exploratory probe (LL_PROBE_JSON, llprobe_out.json).
Burst onsets as in study 25: a chunk starting at latent frame c sits at distance c + 2 + delta from sink frame 0 and
starts (c - 3) / 4 s after the opening, so the bursts start at tau_k(delta) = (P_k - 5 - delta) / 4 s, P = 71, 134, 203,
266 (peaks of both probes); W_k = [tau_k + 0.5, tau_k + 2.5) s, kept if 5 <= tau_k + 0.5 and tau_k + 2.5 <= 60;
A_X(delta) = mean over k of alpha_X(W_k(delta)) (one value per scene).
Primary (Holm over 4; per-scene values; two-sided one-sample t-tests; scene-bootstrap 95% CIs (10,000 draws);
supported if Holm p < .05 and the estimate has the stated sign):
  L1 bursts follow an 18-frame shift:  [A_off18(18) - A_off18(0)] - [A_abs(18) - A_abs(0)]                        > 0
  L2 bursts follow a 36-frame shift:   [A_off36(36) - A_off36(0)] - [A_abs(36) - A_abs(0)]                        > 0
  L3 persistence follows the share:    mean_s rho_s(share, alpha [50, 60)) over the 17 held distances (>= 12 per scene) > 0
  L4 motion falls with the share:      mean_s rho_s(share, motion [50, 60)) over the 17 held distances             < 0
Secondary: L3, L4 over [40, 50); each arm's own burst contrast mean_k [alpha(W_k) - alpha([tau_k - 2.5, tau_k - 0.5))]
(abs, off18, off36 at their own delta; held arms at delta 0); 1-s jumps above 0.15; alpha (10-s windows, 1-s curves)
and motion, open_diff and the share of videos with motion < 1.5 over [50, 60) per arm; across distances, Pearson and
Spearman r of the scene-mean alpha and motion with LongLive's share and with the Self Forcing probe's share; the probe
check of study 26 (first differences of the 1-s alpha curve against the probe's share at the natural distances at
t - lag, lag 0..3 s, t = 6..59) for abs, off18 and off36 with the matching delta; with S27_RESULTS, the correlation
across the 17 distances between LongLive's and Self Forcing's mean alpha and motion over [50, 60).
usage: python analyze_s28.py RUN_DIR OUT_DIR LL_PROBE_JSON SF_PROBE_JSON [S27_RESULTS_JSON]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

from s27_distances import OLD, select

src, out = sys.argv[1], sys.argv[2]
llp, sfp = json.load(open(sys.argv[3])), json.load(open(sys.argv[4]))
s27 = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD, TMAX, FROZEN = 16, 9, 60, 1.5
PEAKS = (71, 134, 203, 266)
DIST = sorted([D for D, _ in select(sfp)] + list(OLD))
share = lambda p, D: float(p["mean"][p["grid"].index(D)])
SH = {D: share(llp, D) for D in DIST}
SH_SF = {D: share(sfp, D) for D in DIST}
DELTA = {"ll_abs": 0, "ll_off18": 18, "ll_off36": 36}
HELD = [f"ll_d{D:03d}" for D in DIST]
ARMS = list(DELTA) + HELD
pat = re.compile(r"^(\w+?)_(ll_abs|ll_off18|ll_off36|ll_d\d{3})__(m16|p16)_s0\.npz$")

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


def motion(arm, t0, t1, k=0):
    o = {}
    i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
    for s, x in X.get(arm, {}).items():
        v = [float(mv[k][i0:i1].mean()) for mv in x.values() if len(mv[k]) >= i1]
        if v:
            o[s] = float(np.mean(v))
    return o


def frozen(arm, t0=50, t1=60):
    i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
    v = [float(mv[0][i0:i1].mean()) < FROZEN for x in X.get(arm, {}).values() for mv in x.values() if len(mv[0]) >= i1]
    return float(np.mean(v)) if v else None


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
    v = np.array([x for x in d.values() if np.isfinite(x)], float)
    if len(v) < 3:
        return None
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def pdiff(x, y):
    return {s: x[s] - y[s] for s in x if s in y}


def shift_test(arm, delta):
    return pdiff(pdiff(A(arm, delta), A(arm, 0)), pdiff(A("ll_abs", delta), A("ll_abs", 0)))


def rho(measure, minn):
    per = {D: measure(f"ll_d{D:03d}") for D in DIST}
    o = {}
    for s in set().union(*[set(v) for v in per.values()]):
        ds = [D for D in DIST if s in per[D]]
        if len(ds) >= minn:
            r = stats.spearmanr([SH[D] for D in ds], [per[D][s] for D in ds]).statistic
            if np.isfinite(r):
                o[s] = float(r)
    return o


def holm(t):
    P = {k: dict(x, side=s) for k, (x, s) in t.items() if x is not None}
    running = 0.0
    for i, k in enumerate(sorted(P, key=lambda k: P[k]["p"])):
        running = max(running, min(1.0, (len(t) - i) * P[k]["p"]))
        P[k]["p_holm"] = running
        P[k]["supported"] = bool(running < 0.05 and np.sign(P[k]["est"]) == P[k]["side"])
    return P


def curve1(arm):
    return {t: float(np.mean(list(x.values()))) for t in range(1, TMAX + 1) if (x := alpha(arm, t - 1, t))}


def jumps(arm, thr=0.15):
    c = curve1(arm)
    return [t for t in sorted(c) if t - 1 in c and c[t] - c[t - 1] > thr]


def probe_share(p, delta):
    """mean probe share over the chunks starting in each second [t - 1, t): chunk c starts (c - 3) / 4 s after the opening"""
    g, m = p["grid"], p["mean"]
    o = {}
    for t in range(1, TMAX + 1):
        v = [m[g.index(c + 2 + delta)] for c in range(3, 243, 3) if t - 1 <= (c - 3) / 4 < t and c + 2 + delta in g]
        if v:
            o[t] = float(np.mean(v))
    return o


def probe_corr():
    res = {}
    for arm, d in DELTA.items():
        c, q = curve1(arm), probe_share(llp, d)
        for lag in range(4):
            ts = [t for t in range(6, TMAX) if all(u in c for u in (t, t - 1)) and all(u in q for u in (t - lag, t - lag - 1))]
            if len(ts) > 5:
                r, p = stats.pearsonr([c[t] - c[t - 1] for t in ts], [q[t - lag] - q[t - lag - 1] for t in ts])
                res[f"{arm}_lag{lag}"] = dict(r=float(r), p=float(p), n=len(ts))
    return res


def across(measure, sh):
    per = {D: measure(f"ll_d{D:03d}") for D in DIST}
    scenes = sorted(set.intersection(*[set(v) for v in per.values()]))
    if len(scenes) < 3:
        return None
    M = np.array([[per[D][s] for D in DIST] for s in scenes])
    x = np.array([sh[D] for D in DIST])
    stat = lambda M: (np.corrcoef(x, M.mean(0))[0, 1], stats.spearmanr(x, M.mean(0)).statistic)
    est = stat(M)
    boot = np.array([stat(M[rng.integers(0, len(scenes), len(scenes))]) for _ in range(2000)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {k: dict(r=float(e), ci=[float(a), float(b)]) for k, e, a, b in zip(("pearson", "spearman"), est, lo, hi)}


def summary(d):
    v = list(d.values())
    return dict(mean=float(np.mean(v)), ci=ci(v), n=len(v)) if v else None


def tests(t0, t1):
    return {"L3": (ttest(rho(lambda a: alpha(a, t0, t1), 12)), +1), "L4": (ttest(rho(lambda a: motion(a, t0, t1), 12)), -1)}


prim = {"L1": (ttest(shift_test("ll_off18", 18)), +1), "L2": (ttest(shift_test("ll_off36", 36)), +1)}
prim.update(tests(50, 60))
per_arm = {a: {"alpha": {T: summary(alpha(a, T - 10, T)) for T in range(10, TMAX + 1, 10)}, "alpha_1s": curve1(a),
               "motion_50_60": summary(motion(a, 50, 60)), "open_diff_50_60": summary(motion(a, 50, 60, k=1)),
               "frozen_share_50_60": frozen(a)} for a in ARMS}
res = {"distances": DIST, "share_ll": SH, "share_sf": SH_SF, "onsets": {d: onsets(d) for d in (0, 18, 36)},
       "counts": {a: len(L.get(a, {})) for a in ARMS},
       "primary": holm(prim),
       "window_40_50": {k: v for k, (v, _) in tests(40, 50).items()},
       "own_contrast": {a: ttest(own_contrast(a, DELTA.get(a, 0))) for a in ARMS},
       "jumps": {a: jumps(a) for a in ARMS},
       "across": {"alpha_ll_share": across(lambda a: alpha(a, 50, 60), SH),
                  "motion_ll_share": across(lambda a: motion(a, 50, 60), SH),
                  "alpha_sf_share": across(lambda a: alpha(a, 50, 60), SH_SF)},
       "probe_corr": probe_corr(),
       "per_arm": per_arm}
if s27:
    pd = s27["per_distance"]
    D2 = [D for D in DIST if per_arm[f"ll_d{D:03d}"]["alpha"].get(60) and pd.get(str(D), {}).get("alpha", {}).get("60")]
    la = [per_arm[f"ll_d{D:03d}"]["alpha"][60]["mean"] for D in D2]
    sa = [pd[str(D)]["alpha"]["60"]["mean"] for D in D2]
    lm = [per_arm[f"ll_d{D:03d}"]["motion_50_60"]["mean"] for D in D2]
    sm = [pd[str(D)]["motion_50_60"]["mean"] for D in D2]
    res["vs_self_forcing"] = {"n": len(D2), "alpha_pearson": float(np.corrcoef(la, sa)[0, 1]) if len(D2) > 2 else None,
                              "motion_pearson": float(np.corrcoef(lm, sm)[0, 1]) if len(D2) > 2 else None}
json.dump(res, open(f"{out}/s28_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
print("jumps:", {a: res["jumps"][a] for a in DELTA})
for a in ARMS:
    q = per_arm[a]
    print(f"{a:9s} share={SH.get(int(a[4:]), float('nan')) if a.startswith('ll_d') else float('nan'):.3f}"
          f" alpha60={q['alpha'][60]['mean'] if q['alpha'].get(60) else float('nan'):+.2f}"
          f" motion={q['motion_50_60']['mean'] if q['motion_50_60'] else float('nan'):.2f} frozen={q['frozen_share_50_60']}")
if s27:
    print("vs Self Forcing:", res["vs_self_forcing"])
