"""LongLive with its own lighting calibration (preregistered, docs/preregistration_calib_ll_cn.md): secondary
comparison of the LongLive-calibrated arms (CALIB2 dir) with the Self-Forcing-calibrated arms (CALIB dir) on the same
openings, seeds and GPU architecture. Primary tests are produced by the frozen analyze_calib.py.
usage: python analyze_calib_ll.py OUT CALIB_DIR CALIB2_DIR PROMPTS_SF PROMPTS_LL
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

out, d1, d2, p1, p2 = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
pat = re.compile(r"(n[rg]\d+)--(C|CL)_(none|ttc_s1)_s(\d+)\.npz$")
cls1 = {d["open"]: d["light"] for d in map(json.loads, open(p1)) if "light" in d}
cls2 = {d["open"]: d["light"] for d in map(json.loads, open(p2)) if "light" in d}
rows = []
for d, cal in [(d1, "SFcal"), (d2, "LLcal")]:
    for p in glob.glob(f"{d}/*.npz"):
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        L = np.load(p)["lab_mean"][:, 0]
        rows.append(dict(cal=cal, open=m[1], arm=m[2] + ("+ttc" if m[3] == "ttc_s1" else ""), seed=int(m[4]),
                         absdL=float(abs(L[-32:].mean() - L[1:9].mean())), dL=float(L[-32:].mean() - L[1:9].mean())))
V = pd.DataFrame(rows)
O = V.groupby(["cal", "open", "arm"], as_index=False).mean(numeric_only=True)
res = {}
changed = sorted(o for o in cls2 if cls2[o] != cls1[o])
for arm in ["CL", "C", "C+ttc"]:
    w = O[O.arm == arm].pivot(index="open", columns="cal", values="absdL")
    for name, idx in [("all", w.index), ("changed", changed)]:
        dl = (w.loc[idx, "LLcal"] - w.loc[idx, "SFcal"]).values
        bs = [rng.choice(dl, len(dl)).mean() for _ in range(10000)]
        res[f"{arm}:{name}"] = dict(est=float(dl.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                    p=float(stats.ttest_1samp(dl, 0).pvalue) if np.any(dl != 0) else 1.0, n=int(len(dl)))
res["changed_openings"] = changed
json.dump(res, open(f"{out}/calib_ll_vs_sf.json", "w"), indent=1)
print(json.dumps(res, indent=1))
