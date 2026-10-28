"""Study 27: does a sink held at one query-sink distance keep as much of the opening, and move as little, as the
attention probe's share at that distance predicts? (preregistered, docs/preregistrations/27_held_distance.md).

Files {scene}_sp_dDDD__{m16|p16}_s0.npz in RUN_DIR (16 fps; frames 0..8 are the held opening): Self Forcing, sink 3
held at distance DDD (hold_gen_sinkpos.py --sink_mode dist:DDD). alpha over [t0, t1) s = (end_p16 - end_m16) /
(open_p16 - open_m16), open = mean L* of frames 1..8, end = mean L* over seconds [t0, t1) after the opening; undefined
if open_p16 - open_m16 < 4. No collapse exclusion. motion over [t0, t1) = mean frame-to-frame grey-level change (the
generator's per-frame motion), averaged over the two versions.
Distances: the 14 of s27_distances.select(PROBE_JSON) (the exploratory 180-s probe, sinkprobe_long_out.json) are the
primary set ("new"); 20, 187, 203 (study 25) are secondary. share(D) = the probe's mean share at D.
Primary (Holm over 3; per-scene Spearman correlations across distances; two-sided one-sample t-tests of their mean
over scenes; scene-bootstrap 95% CIs (10,000 draws); supported if Holm p < .05 and the estimate has the stated sign):
  Q1 persistence follows the share:  mean_s rho_s(share, alpha [50, 60)), 14 new distances          > 0
  Q2 motion falls with the share:    mean_s rho_s(share, motion [50, 60)), 14 new distances         < 0
  Q3 graded below the peaks:         mean_s rho_s(share, alpha [50, 60)), the 7 new distances with
                                     the lowest share                                               > 0
Secondary: Q1-Q3 over [40, 50); Q1, Q2 over all 17 distances; per distance: alpha (10-s windows and 1-s curve), motion,
open_diff, share of videos with motion < 1.5 over [50, 60); across distances: Pearson and Spearman r of the scene-mean
alpha and motion with the share (scene-bootstrap CIs), and the Spearman partial r given the distance; a linear fit of
the scene-mean alpha on the share; with S25_DIR, alpha of the 3 study-25 distances in both studies (per-scene
difference) and the distances whose mean alpha exceeds that of distance 20 while their mean motion is at least that of
study 25's sink at its original positions.
usage: python analyze_s27.py RUN_DIR OUT_DIR PROBE_JSON [S25_DIR]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

from s27_distances import OLD, select, share

src, out, probe = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
s25 = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
FPS, HOLD, TMAX, FROZEN = 16, 9, 60, 1.5
NEW = [D for D, _ in select(probe)]
ALL = sorted(NEW + list(OLD))
SH = {D: share(probe, D) for D in ALL}
LOW = sorted(NEW, key=lambda D: (SH[D], D))[:7]
pat = re.compile(r"^(\w+?)_(sp_d\d+|sp_abs)__(m16|p16)_s0\.npz$")   # study 25 names distance 20 sp_d20


def load(d):
    L, X = {}, {}
    for p in glob.glob(f"{d}/*.npz"):
        m = pat.match(os.path.basename(p))
        if m:
            z, a = np.load(p), m[2] if m[2] == "sp_abs" else f"sp_d{int(m[2][4:]):03d}"
            L.setdefault(a, {}).setdefault(m[1], {})[m[3]] = z["lab_mean"][:, 0]
            if "motion" in z.files:
                X.setdefault(a, {}).setdefault(m[1], {})[m[3]] = (z["motion"], z["open_diff"])
    return L, X


L, X = load(src)
arm = lambda D: f"sp_d{D:03d}"


def endL(v, t0, t1):
    i0, i1 = HOLD + int(round(t0 * FPS)), HOLD + int(round(t1 * FPS))
    return float(v[i0:i1].mean()) if i1 <= len(v) else None


def alpha(a, t0, t1, LL=None):
    o = {}
    for s, x in (L if LL is None else LL).get(a, {}).items():
        if "m16" in x and "p16" in x:
            d = float(x["p16"][1:HOLD].mean() - x["m16"][1:HOLD].mean())
            eh, el = endL(x["p16"], t0, t1), endL(x["m16"], t0, t1)
            if d >= 4 and eh is not None and el is not None:
                o[s] = (eh - el) / d
    return o


def motion(a, t0, t1, k=0, XX=None):
    o = {}
    i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
    for s, x in (X if XX is None else XX).get(a, {}).items():
        v = [float(mv[k][i0:i1].mean()) for mv in x.values() if len(mv[k]) >= i1]
        if v:
            o[s] = float(np.mean(v))
    return o


def frozen(a, t0=50, t1=60):
    i0, i1 = HOLD + t0 * FPS, HOLD + t1 * FPS
    v = [float(mv[0][i0:i1].mean()) < FROZEN for x in X.get(a, {}).values() for mv in x.values() if len(mv[0]) >= i1]
    return float(np.mean(v)) if v else None


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def ttest(d):
    v = np.array([x for x in d.values() if np.isfinite(x)], float)
    if len(v) < 3:
        return None
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


def rho(dists, measure, minn):
    """per-scene Spearman correlation between the share and a per-scene measure across the given distances"""
    per = {D: measure(arm(D)) for D in dists}
    scenes = set().union(*[set(v) for v in per.values()])
    o = {}
    for s in scenes:
        ds = [D for D in dists if s in per[D]]
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


def tests(t0, t1):
    return {"Q1": (ttest(rho(NEW, lambda a: alpha(a, t0, t1), 10)), +1),
            "Q2": (ttest(rho(NEW, lambda a: motion(a, t0, t1), 10)), -1),
            "Q3": (ttest(rho(LOW, lambda a: alpha(a, t0, t1), 5)), +1)}


def across(dists, measure):
    """correlation of the scene-mean measure with the share across distances, with a scene-bootstrap CI"""
    per = {D: measure(arm(D)) for D in dists}
    scenes = sorted(set.intersection(*[set(v) for v in per.values()]))
    if len(scenes) < 3:
        return None
    M = np.array([[per[D][s] for D in dists] for s in scenes])
    sh = np.array([SH[D] for D in dists])
    dr = stats.rankdata(dists)

    def stat(M):
        m = M.mean(0)
        rs, rd, sd = stats.spearmanr(sh, m).statistic, stats.spearmanr(dr, m).statistic, stats.spearmanr(sh, dr).statistic
        return np.corrcoef(sh, m)[0, 1], rs, (rs - rd * sd) / np.sqrt((1 - rd ** 2) * (1 - sd ** 2))
    est = stat(M)
    boot = np.array([stat(M[rng.integers(0, len(scenes), len(scenes))]) for _ in range(2000)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    names = ("pearson", "spearman", "spearman_partial_distance")
    return {k: dict(r=float(e), ci=[float(a), float(b)]) for k, e, a, b in zip(names, est, lo, hi)} | {"n_scenes": len(scenes)}


def summary(d):
    v = list(d.values())
    return dict(mean=float(np.mean(v)), ci=ci(v), n=len(v)) if v else None


per_d = {}
for D in ALL:
    a = arm(D)
    per_d[D] = {"share": SH[D], "set": "new" if D in NEW else "study25",
                "alpha": {T: summary(alpha(a, T - 10, T)) for T in range(10, TMAX + 1, 10)},
                "alpha_1s": {t: float(np.mean(list(x.values()))) for t in range(1, TMAX + 1) if (x := alpha(a, t - 1, t))},
                "motion_50_60": summary(motion(a, 50, 60)), "open_diff_50_60": summary(motion(a, 50, 60, k=1)),
                "frozen_share_50_60": frozen(a)}
m60 = {D: per_d[D]["alpha"][60]["mean"] for D in NEW if per_d[D]["alpha"].get(60)}
fit = np.polyfit([SH[D] for D in m60], list(m60.values()), 1).tolist() if len(m60) > 2 else None
res = {"distances": {"new": NEW, "low": LOW, "study25": list(OLD)}, "share": SH,
       "counts": {D: len(L.get(arm(D), {})) for D in ALL},
       "primary": holm(tests(50, 60)),
       "window_40_50": {k: v for k, (v, _) in tests(40, 50).items()},
       "all17": {"Q1": ttest(rho(ALL, lambda a: alpha(a, 50, 60), 12)),
                 "Q2": ttest(rho(ALL, lambda a: motion(a, 50, 60), 12))},
       "across": {"alpha_new": across(NEW, lambda a: alpha(a, 50, 60)),
                  "motion_new": across(NEW, lambda a: motion(a, 50, 60)),
                  "alpha_all17": across(ALL, lambda a: alpha(a, 50, 60))},
       "fit_alpha_on_share_new": dict(slope=fit[0], intercept=fit[1], share_at_half=(0.5 - fit[1]) / fit[0]) if fit else None,
       "per_distance": per_d}
if s25:
    L25, X25 = load(s25)
    rep = {D: (alpha(arm(D), 50, 60), alpha(arm(D), 50, 60, L25)) for D in OLD}
    res["study25_replication"] = {D: ttest({s: a[s] - b[s] for s in a if s in b}) for D, (a, b) in rep.items()}
    mo_abs = motion("sp_abs", 50, 60, XX=X25)
    ref = float(np.mean(list(mo_abs.values()))) if mo_abs else None
    a20 = per_d[20]["alpha"][60]["mean"] if per_d[20]["alpha"].get(60) else None
    res["frontier"] = {"motion_original_s25": ref, "alpha_d20": a20,
                       "distances": [D for D in ALL if D != 20 and per_d[D]["alpha"].get(60) and per_d[D]["motion_50_60"]
                                     and a20 is not None and ref is not None
                                     and per_d[D]["alpha"][60]["mean"] > a20 and per_d[D]["motion_50_60"]["mean"] >= ref]}
json.dump(res, open(f"{out}/s27_results.json", "w"), indent=1, default=float)
for k in sorted(res["primary"]):
    t = res["primary"][k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4g}",
          "supported" if t["supported"] else "not supported")
for D in ALL:
    q = per_d[D]
    print(f"D={D:3d} share={q['share']:.3f} {q['set']:7s} alpha60={q['alpha'][60]['mean'] if q['alpha'].get(60) else float('nan'):+.2f}"
          f" motion={q['motion_50_60']['mean'] if q['motion_50_60'] else float('nan'):.2f} frozen={q['frozen_share_50_60']}")
print("across:", {k: (v["spearman"]["r"] if v else None) for k, v in res["across"].items()})
if s25:
    print("frontier:", res["frontier"])
