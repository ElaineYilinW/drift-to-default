"""Prompt-swap replay: the prompt's per-chunk pull w_p (preregistered, docs/preregistration_pswap_cn.md).

From pswap.py outputs (*.pswap.json): for each seed, chunk k and swap prompt B, dL = L_k(B) - L_k(generic). The
prompts' default levels mu are the zero-drift levels of the lighting study on Self Forcing (lab note 06): generic 33.1,
sunny 42.9, overcast 33.6, dusk 19.1, night 16.0 L*. Per seed and k, w_p is the least-squares slope through the
origin of dL on dmu = mu_B - mu_generic over the four swap prompts.
Primary: H1 w_p(sink 3, k = 20) > 0 (two-sided one-sample t-test over seeds, supported if p < .05 and estimate > 0).
Secondary: whether w_p(sink 3, k = 20) lies in [0.04, 0.14], the value implied by the chunk-recursion fit to the 120-s
curve (1 - w_w - w_s = 1 - 0.86 - 0.06 = 0.08); w_p at k = 8, 20, 40 for sink 3 and no sink; w_p(k = 40) - w_p(k = 8);
the closure sums w_s + w_w + w_p with replay-measured w_s = 0.053 and w_w = 0.76 (synthetic) or 0.86 (fit); per-prompt
ratios dL / dmu; the replay check (B = generic must reproduce the base chunk, max |difference| reported).
95% CIs: seed bootstrap (10,000 draws).
usage: python analyze_pswap.py RUN_DIR OUT_DIR
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
MU = {"generic": 33.1, "sunny": 42.9, "overcast": 33.6, "dusk": 19.1, "night": 16.0}
R = {}
chk = []
for p in glob.glob(f"{src}/*.pswap.json"):
    j = json.load(open(p))
    for r in j["rows"]:
        R.setdefault(j["sink"], {}).setdefault((r["seed"], r["k"]), {})[r["prompt"]] = r["L_k"]
        if r["prompt"] == j["base_id"]:
            chk.append(abs(r["L_k"] - r["L_k_uninterrupted"]))


def ci(v):
    v = np.asarray(v, float)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5]).tolist()


def wp(sink, k):
    o = {}
    for (s, kk), d in R.get(sink, {}).items():
        if kk != k or "generic" not in d:
            continue
        x = np.array([MU[b] - MU["generic"] for b in d if b != "generic"])
        y = np.array([d[b] - d["generic"] for b in d if b != "generic"])
        o[s] = float((x * y).sum() / (x * x).sum())
    return o


res = {"replay_check_max_abs": max(chk) if chk else None, "w_p": {}, "ratio": {}}
for sink in sorted(R):
    res["w_p"][sink] = {}
    for k in sorted({kk for _, kk in R[sink]}):
        w = wp(sink, k)
        res["w_p"][sink][k] = dict(mean=float(np.mean(list(w.values()))), ci=ci(list(w.values())), n=len(w))
        for b in MU:
            if b == "generic" or abs(MU[b] - MU["generic"]) < 1:
                continue
            v = [d[b] - d["generic"] for (s, kk), d in R[sink].items() if kk == k and b in d]
            res["ratio"].setdefault(sink, {}).setdefault(k, {})[b] = float(np.mean(v) / (MU[b] - MU["generic"]))
        v = [d["overcast"] - d["generic"] for (s, kk), d in R[sink].items() if kk == k and "overcast" in d]
        res.setdefault("placebo_overcast_dL", {}).setdefault(sink, {})[k] = dict(mean=float(np.mean(v)), ci=ci(v))
w20 = wp(3, 20)
v = np.array(list(w20.values()))
res["H1"] = dict(est=float(v.mean()), ci=ci(v), p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)),
                 supported=bool(stats.ttest_1samp(v, 0).pvalue < 0.05 and v.mean() > 0))
res["band_0.04_0.14"] = bool(0.04 <= v.mean() <= 0.14)
w8, w40 = wp(3, 8), wp(3, 40)
d = np.array([w40[s] - w8[s] for s in w8 if s in w40])
res["time_k40_minus_k8"] = dict(est=float(d.mean()), ci=ci(d), p=float(stats.ttest_1samp(d, 0).pvalue), n=int(len(d)))
res["closure"] = {f"w_w={ww}": float(0.053 + ww + v.mean()) for ww in (0.76, 0.86)}
json.dump(res, open(f"{out}/pswap_results.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("H1", "band_0.04_0.14", "time_k40_minus_k8", "closure", "replay_check_max_abs")}, indent=1))
print({s: {k: round(x["mean"], 3) for k, x in v.items()} for s, v in res["w_p"].items()})
