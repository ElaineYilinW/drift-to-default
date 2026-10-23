"""Supplementary analysis of the relaxation study (not part of the frozen analyze_relax.py).

The frozen script estimates alpha from the five L* versions; SkyReels was run (as preregistered) with three
(-16, 0, +16) and 20 s, so the frozen script skips it. Here alpha_L at a common horizon is estimated for every
configuration from the same three versions (m16, z0, p16) and the same within-scene OLS slope, at 10 s, 20 s and (if
available) 30 s, using 1-s means; time is measured from the end of the held opening (16-fps models: frame 9; SkyReels:
frame 17 at 24 fps). Scene-bootstrap 95% CIs.
usage: python analyze_relax_supp.py RELAX_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
pat = re.compile(r"^(\w+?)_(\w+?)__(m16|z0|p16)_s0\.npz$")
D = {}
for p in glob.glob(f"{src}/*.npz"):
    m = pat.match(os.path.basename(p))
    if not m:
        continue
    j = json.load(open(p.replace(".npz", ".json")))
    fps, hold = j.get("fps", 16), (j.get("hold_frames") or 9)
    L = np.load(p)["lab_mean"][:, 0]
    D.setdefault(m[2], {}).setdefault(m[1], {})[m[3]] = (L[1:hold].mean(), L, fps, hold)
res = {}
for cfg, sc in sorted(D.items()):
    r = {}
    for T in (10, 20, 30):
        sl = []
        for s, v in sc.items():
            if len(v) < 3:
                continue
            fps, hold = v["z0"][2], v["z0"][3]
            i0 = hold + int((T - 1) * fps)
            i1 = hold + int(T * fps)
            if i1 > len(v["z0"][1]):
                continue
            x = np.array([v[k][0] for k in ("m16", "z0", "p16")])
            y = np.array([v[k][1][i0:i1].mean() for k in ("m16", "z0", "p16")])
            sl.append(np.polyfit(x, y, 1)[0])
        if len(sl) >= 5:
            sl = np.array(sl)
            bs = [rng.choice(sl, len(sl)).mean() for _ in range(5000)]
            r[f"alpha_L_{T}s"] = dict(mean=float(sl.mean()), ci=np.percentile(bs, [2.5, 97.5]).round(3).tolist(), n=int(len(sl)))
    res[cfg] = r
json.dump(res, open(f"{out}/relax_supp.json", "w"), indent=1)
for c, r in res.items():
    print(f"{c:9s}", {k: (round(v['mean'], 3), v['ci'], v['n']) for k, v in r.items()})
