"""SkyReels overlap confirmation (preregistered, docs/preregistration_skyov_cn.md).

12 scenes not used in the exploratory overlap check x opening L* -16 / +16 x overlap_history {5, 17}, 20 s, seed 0;
both overlaps of a scene run in one task (arch_guard). Files {scene}_skyc{5|17}__{m16|p16}_s0.npz.
alpha = (end_p16 - end_m16) / (open_p16 - open_m16), end = mean L* over seconds [15, 20) after the held opening.
Primary: alpha(overlap 5) < alpha(overlap 17); two-sided paired t-test over scenes, scene-bootstrap 95% CI; supported if
p < .05 and the difference < 0. Secondary: alpha at 5 / 10 / 15 / 20 s (1-s means) per overlap.
usage: python analyze_skyov.py RUN_DIR OUT_DIR
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
D = {}
for p in glob.glob(f"{src}/*_skyc*__*_s0.npz"):
    m = re.match(r"(\w+?)_skyc(5|17)__(m16|p16)_s0\.npz$", os.path.basename(p))
    if m:
        j = json.load(open(p.replace(".npz", ".json")))
        D.setdefault(int(m[2]), {}).setdefault(m[1], {})[m[3]] = (np.load(p)["lab_mean"][:, 0], j["hold_frames"], j["fps"])


def alpha(ov, a, b):
    o = {}
    for s, v in D.get(ov, {}).items():
        if len(v) < 2:
            continue
        (L0, h, fps), (L1, _, _) = v["m16"], v["p16"]
        i0, i1 = h + int(a * fps), h + int(b * fps)
        if i1 <= len(L0):
            o[s] = float((L1[i0:i1].mean() - L0[i0:i1].mean()) / (L1[1:h].mean() - L0[1:h].mean()))
    return o


a5, a17 = alpha(5, 15, 20), alpha(17, 15, 20)
d = np.array([a5[s] - a17[s] for s in a5 if s in a17])
res = dict(n=int(len(d)), alpha5=float(np.mean([a5[s] for s in a5 if s in a17])), alpha17=float(np.mean([a17[s] for s in a5 if s in a17])),
           est=float(d.mean()), ci=np.percentile([rng.choice(d, len(d)).mean() for _ in range(10000)], [2.5, 97.5]).tolist(),
           p=float(stats.ttest_1samp(d, 0).pvalue))
res["supported"] = bool(res["p"] < 0.05 and res["est"] < 0)
res["per_scene"] = {s: [round(a5[s], 3), round(a17[s], 3)] for s in sorted(a5) if s in a17}
res["time_course"] = {ov: {T: float(np.mean(list(alpha(ov, T - 1, T).values()))) for T in (5, 10, 15, 20)} for ov in (5, 17)}
json.dump(res, open(f"{out}/skyov_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_scene"}, indent=1))
