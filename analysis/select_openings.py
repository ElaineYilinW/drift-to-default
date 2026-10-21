"""Pick the 40 new openings for the matched-prompt experiment (rule fixed before any generation).
real (nr00-19): navtest logs not used before; targets evenly spaced in opening brightness between the darkest and the
brightest unused log, nearest distinct log to each target. generated (ng00-19): frame 0 of the Self Forcing
confirmatory runs of the 20 driving prompts, seed 1 (the earlier self openings used other videos, seed 0)."""
import json, glob, os, shutil, numpy as np, av
from PIL import Image
H = "/Users/ywang906/vdrift"; O = f"{H}/match_starts"; os.makedirs(O, exist_ok=True)
scan = [s for s in json.load(open(f"{H}/corr_files/navtest_scan.json")) if not s["used"]]
Ls = np.array([s["L"] for s in scan]); left = set(range(len(scan))); pick = []
for t in np.linspace(Ls.min(), Ls.max(), 20):
    i = min(left, key=lambda j: (abs(Ls[j] - t), scan[j]["log"])); left.remove(i); pick.append(i)
rows = []
for k, i in enumerate(sorted(pick, key=lambda j: Ls[j])):
    s = scan[i]; shutil.copy(s["img"], f"{O}/nr{k:02d}.jpg")
    rows.append(dict(id=f"nr{k:02d}", set="real", log=s["log"], L_scan=s["L"], image=f"{O}/nr{k:02d}.jpg"))
drv = sorted(p for p in glob.glob("/nfsscratch/Users/ywang906/vdrift/corr/confirm/drv_*_none_s1.mp4"))
assert len(drv) == 20, len(drv)
for k, p in enumerate(drv):
    with av.open(p) as c:
        fr = next(c.decode(video=0)).to_image()
    fr.save(f"{O}/ng{k:02d}.jpg", quality=95)
    rows.append(dict(id=f"ng{k:02d}", set="generated", source=os.path.basename(p).split("_none_")[0], image=f"{O}/ng{k:02d}.jpg"))
with open(f"{H}/prompts/match_openings.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print([r.get("L_scan") for r in rows[:20]])
