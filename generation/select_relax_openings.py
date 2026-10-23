"""Relaxation study: pick 24 base scenes (rule fixed before any generation) and write the perturbed-opening prompts.

real (rx00-11): NAVSIM navtest CAM_F0 frame 20 of logs used in no earlier experiment, opening brightness between 35 and
  50 L* (so that +/-16 L* shifts stay inside the gamut), 12 targets evenly spaced in that range, nearest distinct log.
generated (gx00-11): frame 0 of the Self Forcing confirmatory runs of the 20 driving prompts, seed 2 (earlier experiments
  used seeds 0 and 1), the 12 whose brightness is closest to 42 L*.
Each scene gets 7 versions of its opening: L* shift -16, -8, 0, +8, +16 and b* shift -12, +12 (applied by the generator).
Prompt for all: the generic driving prompt of the earlier real-opening runs.
usage: python select_relax_openings.py   (on Argon)
"""
import glob
import json
import os
import shutil

import av
import numpy as np
from PIL import Image

H = "/Users/ywang906/vdrift"
O = f"{H}/relax_starts"
os.makedirs(O, exist_ok=True)
GENERIC = json.loads(open(f"{H}/prompts/real.jsonl").readline())["prompt"]
used = {json.loads(l)["log"] for l in open(f"{H}/prompts/real.jsonl")}
used |= {json.loads(l).get("log") for l in open(f"{H}/prompts/match_openings.jsonl")}


def lstar(img):
    x = np.asarray(img.convert("RGB").resize((320, 180)), dtype=np.float64) / 255
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    Y = x @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(Y > 216 / 24389, np.cbrt(Y), (24389 / 27 * Y + 16) / 116)
    return float((116 * f - 16).mean())


scan = [s for s in json.load(open(f"{H}/corr_files/navtest_scan.json")) if s["log"] not in used and 35 <= s["L"] <= 50]
left, pick = set(range(len(scan))), []
for t in np.linspace(35, 50, 12):
    i = min(left, key=lambda j: (abs(scan[j]["L"] - t), scan[j]["log"]))
    left.remove(i)
    pick.append(i)
rows = []
for k, i in enumerate(sorted(pick, key=lambda j: scan[j]["L"])):
    s = scan[i]
    shutil.copy(s["img"], f"{O}/rx{k:02d}.jpg")
    rows.append(dict(id=f"rx{k:02d}", set="real", log=s["log"], L_scan=s["L"], image=f"{O}/rx{k:02d}.jpg"))
gen = []
for p in sorted(glob.glob("/nfsscratch/Users/ywang906/vdrift/corr/confirm/drv_*_none_s2.mp4")):
    with av.open(p) as c:
        fr = next(c.decode(video=0)).to_image()
    gen.append((abs(lstar(fr) - 42), lstar(fr), os.path.basename(p).split("_none_")[0], fr))
gen.sort(key=lambda g: (g[0], g[2]))
for k, (_, L, src, fr) in enumerate(sorted(gen[:12], key=lambda g: g[1])):
    fr.save(f"{O}/gx{k:02d}.jpg", quality=95)
    rows.append(dict(id=f"gx{k:02d}", set="generated", source=src, L_scan=round(L, 1), image=f"{O}/gx{k:02d}.jpg"))
with open(f"{H}/prompts/relax.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(dict(r, prompt=GENERIC)) + "\n")
print(len(rows), [r["L_scan"] for r in rows])
