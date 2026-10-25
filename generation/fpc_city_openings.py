"""Select the city images of the FramePack content-route study (study 17, docs/preregistrations/17_framepack_content.md).

Candidates, in this order: the fresh generic-prompt videos of preregistration 09, fresh_sf60_s3_s{0..7} then
fresh_sf60_s0_s{0..7} (Self Forcing DMD). Frame 64 (4 s) of each is classified by CLIP ViT-L/14 among the seven labels
of clip_probe.py (softmax with CLIP's logit scale), as in content_openings.py. The first 8 frames whose top-1 label is
"city" with probability >= 0.5 become ctcity{k} (k = 0..7), saved as JPEG (quality 95).
Prompt rows (generic city prompt) written to PROMPTS_OUT:
  {opening}_mis  for each of the 24 openings ct{cls}{k} of preregistration 09: image = the opening, alt_image = ctcity{k}
  ctcity{k}_mis  for k = 0..7: image = alt_image = ctcity{k} (the city starts that give the default)
usage: python fpc_city_openings.py CONTENT_DIR START_DIR CONTENT_RUNS_JSONL PROMPTS_OUT
"""
import json
import os
import sys

import cv2
import numpy as np

import clip_probe as cp

src, start_dir, runs_in, p_out = sys.argv[1:5]
rows = [r for r in map(json.loads, open(runs_in)) if r["set"] == "content" and r["arm"] == "mis"]
assert len(rows) == 24
generic = rows[0]["prompt"]
T, scale = cp.embed_text()
labels = list(cp.LABELS)
log, picked = [], []
for cfg in ("sf60_s3", "sf60_s0"):
    for seed in range(8):
        if len(picked) == 8:
            break
        v = f"{src}/fresh_{cfg}_s{seed}.mp4"
        fr = cp.frames_of(v) if os.path.exists(v) else []
        if len(fr) <= 64:
            log.append(dict(video=v, missing=True))
            continue
        e = cp.embed_images([fr[64]]).astype(np.float32)[0]
        z = scale * T @ e
        p = np.exp(z - z.max())
        p /= p.sum()
        ok = labels[int(p.argmax())] == "city" and p[labels.index("city")] >= 0.5
        log.append(dict(video=v, p_city=float(p[labels.index("city")]), top=labels[int(p.argmax())], ok=bool(ok)))
        if ok:
            path = os.path.abspath(f"{start_dir}/ctcity{len(picked)}.jpg")
            cv2.imwrite(path, cv2.cvtColor(fr[64], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            picked.append(path)
print("picked", len(picked), [(os.path.basename(d["video"]), round(d.get("p_city", -1), 2)) for d in log], flush=True)
assert len(picked) == 8, "fewer than 8 city images"
with open(p_out, "w") as f:
    for r in rows:
        k = int(r["id"].split("_")[0][-1])
        f.write(json.dumps({"id": r["id"], "set": "fpc", "cls": r["cls"], "image": r["image"], "alt_image": picked[k],
                            "prompt": generic}) + "\n")
    for k, path in enumerate(picked):
        f.write(json.dumps({"id": f"ctcity{k}_mis", "set": "fpc_city", "cls": "city", "image": path, "alt_image": path,
                            "prompt": generic}) + "\n")
json.dump(log, open(f"{start_dir}/city_selection_log.json", "w"), indent=1)
