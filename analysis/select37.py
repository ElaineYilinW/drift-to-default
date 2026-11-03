"""Study 37: the held-sink distance chosen by one fixed rule for each model (docs/preregistrations/37_held_sink_rule.md).

Rule: among the calibrated distances D > 20, take the one with the largest mean alpha over [50, 60) s whose frozen share
is at most 0.07 (frozen: a video whose mean motion over [50, 60) s is below 1.5 grey levels; both versions counted);
ties go to the smaller D. No distance qualifies -> the model has no held distance and leaves studies 38 and 39.
  sf   Self Forcing DMD: study 27's 17 distances, 24 scenes, seed 0  (results/corr/s27_out/s27_results.json)
  ll   LongLive:         study 28's 17 distances, 24 scenes, seed 0  (results/corr/s28_out/s28_results.json)
  cv   CausVid, gan  Self Forcing GAN: study 37's calibration, 8 scenes, seed 0 (npz files in CALIB_DIR)
alpha over [t0, t1) = (end_p16 - end_m16) / (open_p16 - open_m16), open = mean L* of frames 1..8, end = mean L* over
seconds [t0, t1) after the opening (frames 9 + 16 t), undefined if open_p16 - open_m16 < 4; averaged over scenes.
The re-aligned distance (sink directly before the window) is window - 1: 20 for sf, cv, gan (window 21), 11 for ll
(window 12).
usage: python select37.py CALIB_DIR S27_RESULTS S28_RESULTS OUT.json
"""
import glob
import json
import os
import re
import sys

import numpy as np

calib, s27, s28, out = sys.argv[1:5]
FPS, HOLD, FROZEN, MAXF = 16, 9, 1.5, 0.07
REALIGN = {"sf": 20, "ll": 11, "cv": 20, "gan": 20}


def rule(table):
    ok = [(d, v) for d, v in table.items() if d > 20 and v["alpha"] is not None and v["frozen"] <= MAXF]
    if not ok:
        return None
    return min(ok, key=lambda x: (-x[1]["alpha"], x[0]))[0]


res = {}
d27 = json.load(open(s27))["per_distance"]
res["sf"] = {int(d): {"alpha": v["alpha"]["60"]["mean"], "frozen": v["frozen_share_50_60"]} for d, v in d27.items()}
d28 = json.load(open(s28))["per_arm"]
res["ll"] = {int(k[4:]): {"alpha": v["alpha"]["60"]["mean"], "frozen": v["frozen_share_50_60"]}
             for k, v in d28.items() if re.fullmatch(r"ll_d\d{3}", k)}

pat = re.compile(r"^(\w+?)_(cv|gan)_d(\d{3})__(m16|p16)_s0\.npz$")
Z = {}
for p in glob.glob(f"{calib}/*.npz"):
    m = pat.match(os.path.basename(p))
    if m:
        z = np.load(p)
        Z.setdefault(m[2], {}).setdefault(int(m[3]), {}).setdefault(m[1], {})[m[4]] = (z["lab_mean"][:, 0], z["motion"])
i0, i1 = HOLD + 50 * FPS, HOLD + 60 * FPS
for m, byd in Z.items():
    res[m] = {}
    for d, sc in byd.items():
        a, fz = [], []
        for x in sc.values():
            for L, mo in x.values():
                if len(mo) >= i1:
                    fz.append(float(mo[i0:i1].mean()) < FROZEN)
            if "m16" in x and "p16" in x and len(x["p16"][0]) >= i1 and len(x["m16"][0]) >= i1:
                h, l = x["p16"][0], x["m16"][0]
                dd = float(h[1:HOLD].mean() - l[1:HOLD].mean())
                if dd >= 4:
                    a.append((float(h[i0:i1].mean()) - float(l[i0:i1].mean())) / dd)
        res[m][d] = {"alpha": float(np.mean(a)) if a else None, "frozen": float(np.mean(fz)) if fz else 1.0,
                     "n_scenes": len(a), "n_videos": len(fz)}

sel = {m: {"dstar": rule(t), "realign": REALIGN[m], "table": {str(d): v for d, v in sorted(t.items())}}
       for m, t in res.items()}
json.dump(sel, open(out, "w"), indent=1)
for m, v in sel.items():
    print(m, "D* =", v["dstar"], "re-aligned =", v["realign"],
          {d: (round(x["alpha"], 3) if x["alpha"] is not None else None, round(x["frozen"], 2))
           for d, x in v["table"].items()})
