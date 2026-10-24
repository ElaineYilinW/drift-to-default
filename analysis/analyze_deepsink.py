"""Deep-sink baseline (preregistered, docs/preregistration_deepsink_cn.md): deeper attention sinks (6 and 9 latent
frames, window 21; tags s6, s9) on the confirmation set of study 2, compared with no correction (sink 3) and with the
content-aware correction, using the analysis of the frozen analyze_confirm.py unchanged except for the arm list and:
  Primary (Holm over 2): D lower for s6 and for s9 (one-sided; supported = Holm p < .05 and CI upper bound < 0).
  Secondary: kept share of legitimate transitions for s6 / s9 (study 2 criterion >= 0.80); D(s9) - D(ttc_s1) and
  D(s6) - D(ttc_s1) (two-sided, mixed model on the two arms); zL; flicker ratios.
Original description:
Confirmatory analysis of the content-aware darkening correction (frozen with docs/preregistration_correction_cn.md).

Arms: none, ttc_s1 (beta 1), ttc_s1_b15 (beta 1.5); all runs of a prompt on one GPU architecture (A100/A800).
Drift set (40 main prompts, 4 seeds): endpoints = mean over seconds 29-31 of
  D  colour drift (Mahalanobis of [L, a, b, sat, log sharpness] vs the prompt's fresh-start reference),
  K  DINOv2 distribution drift (5-NN cosine distance to reference frames / reference leave-one-seed-out median),
  zL signed brightness z-score; plus flicker (frame jitter, chunk-boundary jump) as ratios to none.
Control set (8 held-out transition prompts, 4 seeds): kept share = arm's start-to-end dL / none's, median over controls;
  a control whose uncorrected transition is < 5 L in mean |dL| over seeds is excluded (no transition to keep).
Tests: per arm vs none, mixed model y ~ arm + (1|prompt) + (1|prompt:seed) (REML), Wald contrast.
  Primary (Holm over the 2 arms): D lower (one-sided).  Supported = Holm p < .05 and 95% CI upper bound < 0.
  Secondary (Holm over 4): zL higher (one-sided) for both arms; K non-inferiority: 95% CI upper bound < +0.30 (no p).
usage: python analyze_confirm.py RUN_DIR REF_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
run, refdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
ARMS = ["ttc_s1", "ttc_s1_b15", "s6", "s9"]
PRIM = ["s6", "s9"]
FPS, END, K_NN, K_MARGIN, MIN_TRANSITION = 16, (29, 31), 5, 0.30, 5.0
pat = re.compile(r"(.+?)_(none|ttc_s1|ttc_s1_b15|s6|s9)_s(\d+)\.npz$")
rng = np.random.default_rng(0)


def cfeat(z):
    return np.column_stack([z["lab_mean"], z["sat_mean"], np.log(z["sharpness"] + 1)])


def unit(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def knn(e, pool):
    return np.sort(1 - e @ pool.T, axis=1)[:, :K_NN].mean(1)


refs, kref = {}, {}
for p in glob.glob(f"{refdir}/*_ref_s*.npz"):
    if p.endswith(".feat.npz"):
        continue
    pid = os.path.basename(p).split("_ref_")[0]
    refs.setdefault(pid, []).append(cfeat(np.load(p)))
    fp = p.replace(".npz", ".feat.npz")
    if os.path.exists(fp):
        kref.setdefault(pid, []).append(unit(np.load(fp)["cls"]))

rows = []
for p in sorted(glob.glob(f"{run}/*.npz")):
    if p.endswith(".feat.npz"):
        continue
    m = pat.match(os.path.basename(p))
    if not m:
        continue
    pid, arm, seed = m[1], m[2], int(m[3])
    z = np.load(p)
    L = z["lab_mean"][:, 0]
    sec = np.arange(len(L)) / FPS
    endm = (sec >= END[0]) & (sec < END[1])
    r = dict(prompt=pid, arm=arm, seed=seed, set="control" if pid.startswith("cfm_") else "drift",
             gpu=json.load(open(p.replace(".npz", ".json"))).get("gpu") if os.path.exists(p.replace(".npz", ".json")) else None,
             dL_end=float(L[-32:].mean() - L[1:13].mean()),
             jitter=float(np.std(np.diff(L, 2))),
             boundary=float(np.mean([abs(L[1 + 12 * k] - L[12 * k]) for k in range(1, 41)])))
    if pid in refs:
        x = np.vstack(refs[pid])
        mu, cov = x.mean(0), np.cov(x.T) + 1e-6 * np.eye(5)
        d = cfeat(z) - mu
        D = np.sqrt(np.einsum("ti,ij,tj->t", d, np.linalg.inv(cov), d))
        r["D"] = float(D[endm].mean())
        r["zL"] = float((d[endm, 0] / np.sqrt(cov[0, 0])).mean())
    fp = p.replace(".npz", ".feat.npz")
    if pid in kref and os.path.exists(fp):
        pool = np.vstack(kref[pid])
        med = np.median(np.concatenate([knn(e, np.vstack([v for j, v in enumerate(kref[pid]) if j != i]))
                                        for i, e in enumerate(kref[pid])]))
        fz = np.load(fp)
        ks = fz["dino_idx"] / FPS
        r["K"] = float((knn(unit(fz["cls"]), pool) / med)[(ks >= END[0]) & (ks < END[1])].mean())
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(f"{out}/videos.csv", index=False)
drift = df[df.set == "drift"]
res = {"counts": {a: int((drift.arm == a).sum()) for a in ["none"] + ARMS},
       "gpus": df.gpu.value_counts().to_dict(), "tests": {}}


def contrast(y, arm):
    d = drift[drift.arm.isin(["none", arm])][["prompt", "seed", "arm", y]].dropna().copy()
    d["trt"] = (d.arm == arm).astype(float)
    d["ps"] = d.prompt + "_" + d.seed.astype(str)
    m = smf.mixedlm(f"{y} ~ trt", d, groups=d["prompt"], re_formula="1", vc_formula={"ps": "0 + C(ps)"}).fit(reml=True)
    est, se = float(m.params["trt"]), float(m.bse["trt"])
    w = d.pivot_table(index=["prompt", "seed"], columns="arm", values=y)
    diff = (w[arm] - w["none"]).groupby(level=0).mean().dropna()
    bs = [diff.sample(len(diff), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 2**31, 2000)]
    return dict(est=est, se=se, ci=[est - 1.96 * se, est + 1.96 * se],
                boot_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                mean_none=float(d[d.trt == 0][y].mean()), mean_arm=float(d[d.trt == 1][y].mean()), n=len(d))


prim = {}
for arm in PRIM:
    c = contrast("D", arm)
    c["p"] = float(norm.cdf(c["est"] / c["se"]))  # H: D lower
    prim[f"D:{arm}"] = c
rej, padj, _, _ = multipletests([v["p"] for v in prim.values()], method="holm")
for (k, v), a, pa in zip(prim.items(), rej, padj):
    v.update(family="primary", p_holm=float(pa), supported=bool(a and v["ci"][1] < 0))
res["tests"].update(prim)
sec = {}
for arm in ARMS:
    c = contrast("zL", arm)
    c["p"] = float(norm.sf(c["est"] / c["se"]))  # H: zL higher (less dark)
    sec[f"zL:{arm}"] = c
rej, padj, _, _ = multipletests([v["p"] for v in sec.values()], method="holm")
for (k, v), a, pa in zip(sec.items(), rej, padj):
    v.update(family="secondary", p_holm=float(pa), supported=bool(a and v["ci"][0] > 0))
res["tests"].update(sec)
if "K" in drift:
    for arm in [a_ for a_ in ARMS if drift[drift.arm == a_].K.notna().any()]:
        c = contrast("K", arm)
        c.update(family="secondary-noninferiority", margin=K_MARGIN, noninferior=bool(c["ci"][1] < K_MARGIN))
        res["tests"][f"K:{arm}"] = c



def contrast2(y, a, b):
    """y(a) - y(b), mixed model on the two arms (prompt and prompt:seed random effects)."""
    d = drift[drift.arm.isin([a, b])][["prompt", "seed", "arm", y]].dropna().copy()
    d["trt"] = (d.arm == a).astype(float)
    d["ps"] = d.prompt + "_" + d.seed.astype(str)
    m = smf.mixedlm(f"{y} ~ trt", d, groups=d["prompt"], re_formula="1", vc_formula={"ps": "0 + C(ps)"}).fit(reml=True)
    est, se = float(m.params["trt"]), float(m.bse["trt"])
    return dict(est=est, se=se, ci=[est - 1.96 * se, est + 1.96 * se], p=float(2 * norm.sf(abs(est / se))), n=len(d))


for a in PRIM:
    try:
        res["tests"][f"D:{a}-ttc_s1"] = dict(contrast2("D", a, "ttc_s1"), family="secondary")
    except np.linalg.LinAlgError:
        res["tests"][f"D:{a}-ttc_s1"] = dict(family="secondary", error="singular (identical arms)")

# flicker ratios (descriptive)
fl = drift.groupby("arm")[["jitter", "boundary"]].mean()
res["flicker_ratio_vs_none"] = (fl / fl.loc["none"]).round(3).to_dict()

# held-out controls
ct = df[df.set == "control"]
if len(ct):
    w = ct.pivot_table(index=["prompt", "seed"], columns="arm", values="dL_end")
    trans = w["none"].abs().groupby(level=0).mean()
    valid = trans[trans >= MIN_TRANSITION].index.tolist()
    res["controls"] = {"transition_none_meanabs_dL": trans.round(2).to_dict(), "valid": valid}
    for arm in ARMS:
        if arm in w:
            kept = (w[arm] / w["none"]).groupby(level=0).median()
            res["controls"][arm] = {"kept_by_prompt": kept.round(2).to_dict(),
                                    "median_kept_valid": float(kept[valid].median()) if valid else None}
json.dump(res, open(f"{out}/results.json", "w"), indent=1)
for k, v in res["tests"].items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else ([round(x, 3) for x in vv] if isinstance(vv, list) else vv))
              for kk, vv in v.items() if kk in ("est", "ci", "boot_ci", "p_holm", "supported", "noninferior", "mean_none", "mean_arm")})
print("flicker", res["flicker_ratio_vs_none"])
print("controls", json.dumps(res.get("controls"), indent=0)[:800])
