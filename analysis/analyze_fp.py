"""FramePack: persistence and the two routes of the opening (preregistered, docs/preregistration_fp_cn.md).

Files {scene}_{cond}__{m16|p16}_s0.{npz,json} from fp_gen.py, cond in fpv_both, fpv_ctx, fpv_emb (FramePack-F1,
forward sampling; the L* shift of the opening reaches the model through both routes, only the context latent, or only
the SigLIP embedding) and fpi_both (FramePackI2V, inverted anti-drifting sampling).
alpha(T) = (end_p16 - end_m16) / (open_p16 - open_m16), with open the L* of the shifted opening image (json
open_L_shifted) and end the mean L* over seconds [T-5, T) after the opening (30 fps; frames 0..8 are the opening).
Primary (Holm over 3; two-sided one-sample t-tests over scenes, scene-bootstrap 95% CIs):
  H1 FramePack keeps most of the opening:      alpha(fpv_both, 20 s) > 0.5
  H2 the conditioning sets the default:        alpha(fpv_emb, 20 s) > 0
  H3 conditioning adds to persistence:         alpha(fpv_both, 20 s) - alpha(fpv_ctx, 20 s) > 0
Each is supported if Holm p < .05 and the estimate is on the predicted side.
Secondary: alpha of every condition at 5 / 10 / 15 / 20 s; additivity alpha(ctx) + alpha(emb) - alpha(both) at 20 s;
alpha(fpi_both, 20 s) - alpha(fpv_both, 20 s).
usage: python analyze_fp.py RUN_DIR OUT_DIR
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
pat = re.compile(r"^(\w+?)_(fp[vi]_\w+?)__(m16|p16)_s0\.npz$")
D = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        j = json.load(open(p[:-4] + ".json"))
        D.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = (j["open_L_shifted"], np.load(p)["lab_mean"][:, 0],
                                                            j["fps"], j["hold_frames"])


def alphas(cond, T):
    o = {}
    for s, v in D.get(cond, {}).items():
        if len(v) < 2:
            continue
        ends = []
        for k in ("m16", "p16"):
            op, L, fps, hold = v[k]
            i0, i1 = hold + int((T - 5) * fps), hold + int(T * fps)
            if i1 > len(L):
                break
            ends.append(L[i0:i1].mean())
        if len(ends) == 2:
            o[s] = (ends[1] - ends[0]) / (v["p16"][0] - v["m16"][0])
    return o


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def test(d, null=0.0):
    v = np.array(list(d.values())) - null
    return dict(est=float(v.mean() + null), ci=(np.array(ci(v)) + null).tolist(), p=float(stats.ttest_1samp(v, 0).pvalue),
                n=int(len(v)))


res = {"counts": {c: len(v) for c, v in D.items()}, "alpha": {}}
for c in sorted(D):
    res["alpha"][c] = {T: dict(mean=float(np.mean(list(a.values()))), ci=ci(list(a.values())), n=len(a))
                       for T in (5, 10, 15, 20) if (a := alphas(c, T))}
b, x, e = alphas("fpv_both", 20), alphas("fpv_ctx", 20), alphas("fpv_emb", 20)
tests = {"H1": test(b, 0.5), "H2": test(e), "H3": test({s: b[s] - x[s] for s in b if s in x})}
running = 0.0
for i, k in enumerate(sorted(tests, key=lambda k: tests[k]["p"])):
    running = max(running, min(1.0, (len(tests) - i) * tests[k]["p"]))
    tests[k]["p_holm"] = running
    tests[k]["supported"] = bool(running < 0.05 and tests[k]["est"] > (0.5 if k == "H1" else 0.0))
res["primary"] = tests
res["additivity_20s"] = test({s: x[s] + e[s] - b[s] for s in b if s in x and s in e})
iv = alphas("fpi_both", 20)
res["inverted_minus_vanilla_20s"] = test({s: iv[s] - b[s] for s in iv if s in b}) if iv else None
json.dump(res, open(f"{out}/fp_results.json", "w"), indent=1)
print(json.dumps(res["primary"], indent=1))
print({c: {T: round(v["mean"], 3) for T, v in r.items()} for c, r in res["alpha"].items()})
