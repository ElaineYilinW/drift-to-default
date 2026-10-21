"""E2: compare closed-loop brightness corrections against no correction (paired: same prompt, seed, GPU type).

Drift set (pilot prompts): end-of-video brightness change vs the first chunk, signed z_L / D vs the pilot fresh-start
reference, structure drift K (DINOv2, if .feat.npz present), flicker (chunk-boundary |dL| and frame-to-frame jitter).
Control set (scripted brightness transitions): share of the uncorrected run's start-to-end brightness change kept.
usage: python analyze_e2.py E2_DIR PILOT_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

e2, pilot, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
pat = re.compile(r"(.+?)_(none|luma_[a-z0-9_]+|ttc_[a-z0-9_]+)_s(\d+)\.npz$")
rng = np.random.default_rng(0)


def colour_feats(z):
    return np.column_stack([z["lab_mean"], z["sat_mean"], np.log(z["sharpness"] + 1)])


refs, kref = {}, {}
for p in glob.glob(f"{pilot}/*_ref_s*.npz"):
    if p.endswith(".feat.npz"):
        continue
    pid = os.path.basename(p).split("_ref_")[0]
    refs.setdefault(pid, []).append(colour_feats(np.load(p)))
    fp = p.replace(".npz", ".feat.npz")
    if os.path.exists(fp):
        c = np.load(fp)["cls"].astype(np.float64)
        kref.setdefault(pid, []).append(c / np.linalg.norm(c, axis=1, keepdims=True))

rows = []
for p in sorted(glob.glob(f"{e2}/*.npz")):
    b = os.path.basename(p)
    if b.endswith(".feat.npz"):
        continue
    m = pat.match(b)
    if not m:
        continue
    pid, arm, seed = m[1], m[2], int(m[3])
    z = np.load(p)
    L = z["lab_mean"][:, 0]
    info = json.load(open(p.replace(".npz", ".json")))
    first = L[1:13].mean()
    r = dict(prompt=pid, arm=arm, seed=seed, set="control" if pid.startswith("ctl_") else "drift",
             L_first=first, dL_end=L[-32:].mean() - first, n_fired=info["n_fired"], gpu=info.get("gpu"),
             boundary_jump=np.mean([abs(L[1 + 12 * k] - L[12 * k]) for k in range(1, 41)]),
             jitter=np.std(np.diff(L, 2)))
    if pid in refs:
        x = np.vstack(refs[pid])
        mu, cov = x.mean(0), np.cov(x.T) + 1e-6 * np.eye(5)
        f = colour_feats(z)[-32:]
        d = f - mu
        r["D_end"] = float(np.sqrt(np.einsum("ti,ij,tj->t", d, np.linalg.inv(cov), d)).mean())
        r["zL_end"] = float((d[:, 0] / np.sqrt(cov[0, 0])).mean())
        r["zsat_end"] = float((d[:, 3] / np.sqrt(cov[3, 3])).mean())
    fp = p.replace(".npz", ".feat.npz")
    if pid in kref and os.path.exists(fp):
        pool = np.vstack(kref[pid])
        med = np.median(np.concatenate([np.sort(1 - r_ @ np.vstack([v for j, v in enumerate(kref[pid]) if j != i]).T, axis=1)[:, :5].mean(1)
                                        for i, r_ in enumerate(kref[pid])]))
        fz = np.load(fp)
        e = fz["cls"].astype(np.float64)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        sec = fz["dino_idx"] / 16
        kk = np.sort(1 - e @ pool.T, axis=1)[:, :5].mean(1) / med
        r["K_end"] = float(kk[sec >= 29].mean())
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(f"{out}/e2_videos.csv", index=False)

base = df[df.arm == "none"].set_index(["prompt", "seed"])
res = {}
for arm, g in df[df.arm != "none"].groupby("arm"):
    g = g.set_index(["prompt", "seed"])
    j = g.join(base, rsuffix="_none", how="inner")
    dr, ct = j[j.set == "drift"], j[j.set == "control"]

    def paired(col):
        if col not in dr or col + "_none" not in dr:
            return None
        diff = (dr[col] - dr[col + "_none"]).groupby(level=0).mean()
        bs = [diff.sample(len(diff), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 2000)]
        return dict(arm=float(dr[col].mean()), none=float(dr[col + "_none"].mean()), diff=float(diff.mean()),
                    ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])

    ct = ct[[i[0] != "ctl_dawn" for i in ct.index]]  # dawn never brightens in this model: not a valid control
    kept = (ct.dL_end / ct.dL_end_none).groupby(level=0).median()
    res[arm] = dict(n=int(len(j)), dL_end=paired("dL_end"), zL_end=paired("zL_end"), D_end=paired("D_end"),
                    zsat_end=paired("zsat_end"),
                    K_end=paired("K_end"), boundary_jump=paired("boundary_jump"), jitter=paired("jitter"),
                    fired_frac=float(g.n_fired.mean() / 41),
                    control_kept_share={k: round(float(v), 2) for k, v in kept.items()},
                    control_kept_median=float(kept.median()) if len(kept) else None)
json.dump(res, open(f"{out}/e2_summary.json", "w"), indent=1)
tab = pd.DataFrame({a: {"dL_end": r["dL_end"]["arm"] if r["dL_end"] else None,
                        "zL_end": r["zL_end"]["arm"] if r["zL_end"] else None,
                        "D_end": r["D_end"]["arm"] if r["D_end"] else None,
                        "dD [95% CI]": "%.2f [%.2f, %.2f]" % (r["D_end"]["diff"], *r["D_end"]["ci"]) if r["D_end"] else None,
                        "zsat_end": r["zsat_end"]["arm"] if r["zsat_end"] else None,
                        "dK vs none": r["K_end"]["diff"] if r["K_end"] else None,
                        "d boundary": r["boundary_jump"]["diff"], "d jitter": r["jitter"]["diff"],
                        "fired": r["fired_frac"], "control kept": r["control_kept_median"]} for a, r in res.items()}).T
nb = df[(df.arm == "none") & (df.set == "drift")]
print("none (drift set): dL_end %.2f zL_end %.2f zsat_end %.2f D_end %.2f" % (nb.dL_end.mean(), nb.zL_end.mean(), nb.zsat_end.mean(),
                                                                             nb.D_end.mean()))
print(tab.round(2).to_string())
