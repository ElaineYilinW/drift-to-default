"""Analysis of the context-intervention experiment (ctx_swap.py).

1. Interventional weights: how much the next chunk's L follows a shift of the rolling-window context (w_win) and of
   the anchor/sink chunk (w_sink): slopes of dL_k on the realised context shifts, pooled through the origin,
   prompt-cluster bootstrap CIs; per config (w21s3, w21s0) and per intervention time k.
2. Attractor: with L_k = w_win*L_win + w_sink*L_sink + (1 - w_win - w_sink)*M, the per-video level M implied by the
   unshifted arms; compared with the opening level (chunk 0) and the prompt's fresh-start reference mean.
3. Dynamics check: predicted stationary level from (w, M) vs the observed late plateau of the base trajectory, and
   the observational AR(1) slope of L_{k+1} on the trailing-window mean (prediction) vs the interventional w_win.
usage: python analyze_swap.py SWAP_DIR REF_DIR OUT_DIR
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

src, refdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)

refL = {}
for p in glob.glob(f"{refdir}/*_ref_s*.npz"):
    if p.endswith(".feat.npz"):
        continue
    refL.setdefault(os.path.basename(p).split("_ref_")[0], []).append(np.load(p)["lab_mean"][:, 0].mean())
refL = {k: float(np.mean(v)) for k, v in refL.items()}

rows, traj = [], []
for f in sorted(glob.glob(f"{src}/*.swap.json")):
    d = json.load(open(f))
    for r in d["swaps"]:
        rows.append(dict(prompt=d["prompt_id"], tag=d["tag"], seed=d["seed"], **r))
    traj.append(dict(prompt=d["prompt_id"], tag=d["tag"], seed=d["seed"], L=d["base_chunk_L"], sink=d["sink"]))
df = pd.DataFrame(rows)
base = df[(df.ds == 0) & (df.dw == 0)].set_index(["prompt", "tag", "seed", "k"])
j = df.join(base[["L_k", "L_sink", "L_win"]], on=["prompt", "tag", "seed", "k"], rsuffix="0")
j["dLk"], j["dLs"], j["dLw"] = j.L_k - j.L_k0, j.L_sink - j.L_sink0, j.L_win - j.L_win0
res = {"n_videos": len(traj), "replay_exact": bool((base["L_k"] - base["L_k_uninterrupted"]).abs().max() < 1e-6)}


def slope_through_origin(g, x):
    g = g[g[x].abs() > 0]
    return float((g[x] * g.dLk).sum() / (g[x] ** 2).sum()) if len(g) else np.nan


def boot(g, x, B=2000):
    ps = g.prompt.unique()
    vals = []
    for _ in range(B):
        pick = rng.choice(ps, len(ps), replace=True)
        vals.append(slope_through_origin(pd.concat([g[g.prompt == p] for p in pick]), x))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


W = {}
for tag, g in j.groupby("tag"):
    gw, gs = g[g.dw != 0], g[g.ds != 0]
    W[tag] = dict(w_win=slope_through_origin(gw, "dLw"), w_win_ci=boot(gw, "dLw"),
                  by_k={int(k): round(slope_through_origin(gk, "dLw"), 3) for k, gk in gw.groupby("k")})
    if len(gs):
        W[tag].update(w_sink=slope_through_origin(gs, "dLs"), w_sink_ci=boot(gs, "dLs"),
                      by_k_sink={int(k): round(slope_through_origin(gk, "dLs"), 3) for k, gk in gs.groupby("k")})
    else:
        W[tag].update(w_sink=0.0)
res["weights"] = W

# attractor M per video and dynamics checks
att = []
for t in traj:
    w = W[t["tag"]]
    ww, wsk = w["w_win"], w["w_sink"]
    b = base.loc[(t["prompt"], t["tag"], t["seed"])]
    M = ((b.L_k - ww * b.L_win - wsk * (b.L_sink.fillna(0) if t["sink"] else 0)) / (1 - ww - wsk)).mean()
    L = np.array(t["L"])
    L_sink = float(b.L_sink.mean()) if t["sink"] else 0.0
    pred_plateau = (wsk * L_sink + (1 - ww - wsk) * M) / (1 - ww)
    # observational AR(1): L_{k+1} on trailing window mean (prediction, confounded by content)
    nwin = 6 if t["sink"] else 7
    xs = np.array([L[max(1 if t["sink"] else 0, k - nwin):k].mean() for k in range(8, len(L))])
    ys = L[8:]
    obs_slope = float(np.polyfit(xs, ys, 1)[0])
    att.append(dict(prompt=t["prompt"], tag=t["tag"], seed=t["seed"], L_open=float(L[0]), M=float(M),
                    ref_mean=refL.get(t["prompt"]), pred_plateau=float(pred_plateau),
                    obs_plateau=float(L[-14:].mean()), obs_ar1_slope=obs_slope))
A = pd.DataFrame(att)
A.to_csv(f"{out}/attractor.csv", index=False)
res["attractor"] = {tag: dict(L_open=float(g.L_open.mean()), M=float(g.M.mean()), ref_mean=float(g.ref_mean.mean()),
                              M_minus_open=float((g.M - g.L_open).mean()),
                              pred_plateau=float(g.pred_plateau.mean()), obs_plateau=float(g.obs_plateau.mean()),
                              corr_pred_obs_plateau=float(np.corrcoef(g.pred_plateau, g.obs_plateau)[0, 1]),
                              obs_ar1_slope=float(g.obs_ar1_slope.mean()))
                    for tag, g in A.groupby("tag")}
json.dump(res, open(f"{out}/swap_summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))

# 4. Out-of-sample forecast: estimate each video's attractor M from the k = 12 intervention only (about 9 s), then
#    roll the chunk-level dynamics L_{j} = w_win * mean(window) + w_sink * L_sink + (1 - w_win - w_sink) * M forward
#    from the real trajectory up to chunk 12 and compare with chunks 13..41 (never used for estimation).
fc = []
for t in traj:
    w = W[t["tag"]]
    ww, wsk = w["w_win"], w["w_sink"]
    b = base.loc[(t["prompt"], t["tag"], t["seed"])]
    if 12 not in b.index:
        continue
    r12 = b.loc[12]
    L_s = float(r12.L_sink) if t["sink"] else 0.0
    M12 = (r12.L_k - ww * r12.L_win - wsk * L_s) / (1 - ww - wsk)
    L = list(np.array(t["L"])[:12])
    nwin = 6 if t["sink"] else 7
    for jj in range(12, len(t["L"])):
        win = L[max(1 if t["sink"] else 0, jj - nwin):jj]
        L.append(ww * np.mean(win) + wsk * L_s + (1 - ww - wsk) * M12)
    obs = np.array(t["L"])
    fc.append(dict(tag=t["tag"], prompt=t["prompt"], seed=t["seed"], pred_late=float(np.mean(L[-14:])),
                   obs_late=float(obs[-14:].mean()), naive_late=float(obs[:12].mean()),
                   rmse_traj=float(np.sqrt(np.mean((np.array(L[12:]) - obs[12:]) ** 2)))))
F = pd.DataFrame(fc)
F.to_csv(f"{out}/forecast.csv", index=False)
res["forecast_oos"] = {tag: dict(n=int(len(g)), pred_late=float(g.pred_late.mean()), obs_late=float(g.obs_late.mean()),
                                 corr=float(np.corrcoef(g.pred_late, g.obs_late)[0, 1]),
                                 mae=float((g.pred_late - g.obs_late).abs().mean()),
                                 mae_naive_persist=float((g.naive_late - g.obs_late).abs().mean()),
                                 rmse_traj=float(g.rmse_traj.mean()))
                       for tag, g in F.groupby("tag")}
json.dump(res, open(f"{out}/swap_summary.json", "w"), indent=1)
print(json.dumps(res["forecast_oos"], indent=1))
