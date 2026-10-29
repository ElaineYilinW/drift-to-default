"""Exploratory: persistence against motion per configuration (third internal review, W1). Not preregistered.

Each argument RUN_DIR:AUDIT_DIR pairs a run directory of held, L*-shifted openings ({scene}_{arm}__{m16|p16}_s{seed}.npz
with lab_mean) with the motion_audit.py output for its videos. Per arm (prefixed with the run directory's name) and
10-s window [T-10, T) s after the 9 held frames (frame rate from the video): alpha = (end_p16 - end_m16) / (open_p16 -
open_m16) per scene and seed (undefined if the openings differ by < 4 L*), the mean motion and open_diff of the two
versions, and the share of videos with motion < 1.5 (frozen), and alpha over the scenes whose two versions both move (motion >= 1.5;
alpha_moving). Scene-mean values with scene-bootstrap 95% CIs; across
arms, the Spearman correlation of alpha with motion at each T.
usage: python motion_summary.py OUT.json RUN_DIR:AUDIT_DIR [...]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

out = sys.argv[1]
HOLD, FROZEN = 9, 1.5
rng = np.random.default_rng(0)
pat = re.compile(r"^(?P<scene>[a-z]{2}\d\d)_(?P<arm>.+?)__(?P<ver>m16|p16)_s(?P<seed>\d+)\.npz$")


def ci(v):
    v = np.asarray(v, float)
    return [float(x) for x in np.percentile([rng.choice(v, len(v)).mean() for _ in range(5000)], [2.5, 97.5])]


res = {}
for spec in sys.argv[2:]:
    run, aud = spec.split(":")
    name = os.path.basename(os.path.normpath(run))
    V = {}
    for p in glob.glob(f"{aud}/*.npz"):
        m = pat.match(os.path.basename(p))
        if not m or not os.path.exists(f"{run}/{os.path.basename(p)}"):
            continue
        a = np.load(p)
        L = np.load(f"{run}/{os.path.basename(p)}")["lab_mean"][:, 0]
        V.setdefault((m["arm"], m["scene"], m["seed"]), {})[m["ver"]] = (L, a["motion"], a["open_diff"], float(a["fps"]))
    for arm in sorted({k[0] for k in V}):
        rows = {}
        for (a_, sc, sd), vs in V.items():
            if a_ != arm or set(vs) != {"m16", "p16"}:
                continue
            fps = vs["m16"][3]
            fps = 16.0 if not np.isfinite(fps) else fps
            n = min(len(v[0]) for v in vs.values())
            od = vs["p16"][0][1:9].mean() - vs["m16"][0][1:9].mean()
            for T in range(10, int((n - HOLD) / fps) + 1, 10):
                i0, i1 = HOLD + int(round((T - 10) * fps)), HOLD + int(round(T * fps))
                if i1 > n:
                    break
                al = ((vs["p16"][0][i0:i1].mean() - vs["m16"][0][i0:i1].mean()) / od) if od >= 4 else np.nan
                mo = [v[1][i0:i1].mean() for v in vs.values()]
                rows.setdefault(T, []).append((sc, al, np.mean(mo), np.mean([v[2][i0:i1].mean() for v in vs.values()]),
                                               np.mean([x < FROZEN for x in mo]), al if min(mo) >= FROZEN else np.nan))
        r = {}
        for T, lst in sorted(rows.items()):
            by = {}
            for sc, al, mo, od_, fz, am in lst:
                by.setdefault(sc, []).append((al, mo, od_, fz, am))
            sm = {sc: np.nanmean(np.array(v), 0) for sc, v in by.items()}
            A = np.array([v[0] for v in sm.values()])
            Mo = np.array([v[1] for v in sm.values()])
            r[T] = dict(n=len(sm), alpha=float(np.nanmean(A)), alpha_ci=ci(A[np.isfinite(A)]), motion=float(Mo.mean()),
                        motion_ci=ci(Mo), open_diff=float(np.mean([v[2] for v in sm.values()])),
                        frozen=float(np.mean([v[3] for v in sm.values()])),
                        alpha_moving=float(np.nanmean([v[4] for v in sm.values()])) if np.isfinite([v[4] for v in sm.values()]).any() else None,
                        n_moving=int(np.isfinite([v[4] for v in sm.values()]).sum()))
        res[f"{name}/{arm}"] = r
        if r:
            T = max(r)
            q = r[T]
            print(f"{name}/{arm:14s} T={T:3d} n={q['n']:2d} alpha {q['alpha']:+.2f} motion {q['motion']:.2f} "
                  f"[{q['motion_ci'][0]:.2f},{q['motion_ci'][1]:.2f}] open_diff {q['open_diff']:.1f} frozen {q['frozen']:.2f} "
                  f"alpha_moving {q['alpha_moving'] if q['alpha_moving'] is None else round(q['alpha_moving'], 2)} (n {q['n_moving']})",
                  flush=True)
across = {}
for T in (30, 60):
    pts = [(v[T]["alpha"], v[T]["motion"]) for v in res.values() if T in v and np.isfinite(v[T]["alpha"])]
    if len(pts) >= 5:
        rho, p = stats.spearmanr(*zip(*pts))
        across[T] = dict(n=len(pts), spearman=float(rho), p=float(p))
        print(f"across arms T={T}: n={len(pts)} Spearman(alpha, motion) {rho:+.2f} p={p:.3g}")
json.dump(dict(arms=res, across=across), open(out, "w"), indent=1)
