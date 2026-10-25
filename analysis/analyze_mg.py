"""An action-conditioned world model: what keeps the opening (preregistered, docs/preregistration_mg_cn.md).

Matrix-Game-2.0 (universal), constant forward action, 30 s (the model degenerates to black after about 40 s in a
smoke test, beyond its training length). Files {scene}_{cond}__{m16|p16}_s0.{npz,json} from
mg_gen.py, cond in mg_both, mg_ctx, mg_emb: the L* shift of the opening reaches the model through both routes, only the
concatenated latent (which leaves the 6-frame key-value window after about one second; no sink), or only the CLIP image
embedding (a global condition of every chunk).
alpha(T) = (end_p16 - end_m16) / (open_p16 - open_m16): open = L* of the shifted opening image (json open_L_shifted),
end = mean L* over seconds [T-w, T) after the opening (25 fps; frames 0..8 are the opening); w = 10 s for T >= 20,
w = 1 s for T < 20. Exclusion (fixed before generation): a scene is dropped from every analysis if any 1-s mean L* of
any of its videos within the first 30 s is below 3 (collapse to black).
Primary (Holm over 3; two-sided one-sample t-tests over scenes, scene-bootstrap 95% CIs):
  H1 the context alone is forgotten:          alpha(mg_ctx, 30 s) - alpha(mg_ctx, 5 s) < 0
  H2 the conditioning moves the default:      alpha(mg_emb, 30 s) > 0
  H3 conditioning adds to persistence:        alpha(mg_both, 30 s) - alpha(mg_ctx, 30 s) > 0
Each is supported if Holm p < .05 and the estimate is on the predicted side.
Secondary: alpha of each condition at 2, 5, 10, 20, 30 s; additivity alpha(ctx) + alpha(emb) - alpha(both) at 30 s;
per-second curves; number of excluded scenes.
usage: python analyze_mg.py RUN_DIR OUT_DIR
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
pat = re.compile(r"^(\w+?)_(mg_\w+?)__(m16|p16)_s0\.npz$")
D, BLACK = {}, set()
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        j = json.load(open(p[:-4] + ".json"))
        L = np.load(p)["lab_mean"][:, 0]
        D.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = (j["open_L_shifted"], L, j["fps"], j["hold_frames"])
        sec = [L[j["hold_frames"] + k * j["fps"]:j["hold_frames"] + (k + 1) * j["fps"]].mean() for k in range(30)]
        if min(sec) < 3:
            BLACK.add(m[1])
D = {c: {s: v for s, v in sc.items() if s not in BLACK} for c, sc in D.items()}


def alphas(cond, T, w=None):
    w = w or (10 if T >= 20 else 1)
    o = {}
    for s, v in D.get(cond, {}).items():
        if len(v) < 2:
            continue
        ends = []
        for k in ("m16", "p16"):
            op, L, fps, hold = v[k]
            i0, i1 = hold + int((T - w) * fps), hold + int(T * fps)
            if i1 > len(L):
                break
            ends.append(L[i0:i1].mean())
        if len(ends) == 2:
            o[s] = (ends[1] - ends[0]) / (v["p16"][0] - v["m16"][0])
    return o


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def test(d):
    v = np.array(list(d.values()))
    return dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


res = {"counts": {c: len(v) for c, v in D.items()}, "alpha": {}}
for c in sorted(D):
    res["alpha"][c] = {T: dict(mean=float(np.mean(list(a.values()))), ci=ci(list(a.values())), n=len(a))
                       for T in (2, 5, 10, 20, 30) if (a := alphas(c, T))}
c30, c5 = alphas("mg_ctx", 30), alphas("mg_ctx", 5)
e30, b30 = alphas("mg_emb", 30), alphas("mg_both", 30)
tests = {"H1": test({s: c30[s] - c5[s] for s in c30 if s in c5}), "H2": test(e30),
         "H3": test({s: b30[s] - c30[s] for s in b30 if s in c30})}
running = 0.0
for i, k in enumerate(sorted(tests, key=lambda k: tests[k]["p"])):
    running = max(running, min(1.0, (len(tests) - i) * tests[k]["p"]))
    tests[k]["p_holm"] = running
    tests[k]["supported"] = bool(running < 0.05 and (tests[k]["est"] < 0 if k == "H1" else tests[k]["est"] > 0))
res["primary"] = tests
res["excluded_black"] = sorted(BLACK)
res["additivity_30s"] = test({s: c30[s] + e30[s] - b30[s] for s in b30 if s in c30 and s in e30})
res["curves"] = {c: [(T, float(np.mean(list(a.values())))) for T in range(1, 31) if (a := alphas(c, T, 1))] for c in D}
json.dump(res, open(f"{out}/mg_results.json", "w"), indent=1)
print(json.dumps(res["primary"], indent=1))
print({c: {T: round(v["mean"], 3) for T, v in r.items()} for c, r in res["alpha"].items()})
