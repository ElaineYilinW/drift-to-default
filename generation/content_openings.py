"""Select the openings of the content-relaxation study (docs/preregistration_content_cn.md).

For each class, Self Forcing DMD generated 5-s fresh videos from the class prompt (seeds 0..15, cg_{cls}_sfgen_s{seed}).
Frame 64 (4 s) of each is classified by CLIP ViT-L/14 among the seven labels of clip_probe.py (softmax with CLIP's
logit scale). In seed order, the first 4 frames whose top-1 label is their own class with probability >= 0.5 become the
openings ct{cls}{k} (k = 0..3), saved as JPEG (quality 95). Each opening gets two prompt rows: _mis (the generic city
prompt of the relaxation study) and _mat (its own class prompt).
usage: python content_openings.py GEN_DIR START_DIR PROMPTS_IN PROMPTS_OUT
"""
import json
import os
import sys

import cv2
import numpy as np

import clip_probe as cp

gen_dir, start_dir, p_in, p_out = sys.argv[1:5]
os.makedirs(start_dir, exist_ok=True)
rows = [json.loads(l) for l in open(p_in)]
generic = next(r for r in rows if r["id"] == "fresh")["prompt"]
T, scale = cp.embed_text()
labels = list(cp.LABELS)
log, out = {}, [r for r in rows]
for r in rows:
    if r["set"] != "opening_gen":
        continue
    cls, picked, log[r["cls"]] = r["cls"], 0, []
    for seed in range(16):
        v = f"{gen_dir}/{r['id']}_sfgen_s{seed}.mp4"
        fr = cp.frames_of(v) if os.path.exists(v) else []
        if len(fr) <= 64:
            log[cls].append(dict(seed=seed, missing=True))
            continue
        e = cp.embed_images([fr[64]]).astype(np.float32)[0]
        z = scale * T @ e
        p = np.exp(z - z.max())
        p /= p.sum()
        ok = labels[int(p.argmax())] == cls and p[labels.index(cls)] >= 0.5
        log[cls].append(dict(seed=seed, p_cls=float(p[labels.index(cls)]), top=labels[int(p.argmax())], ok=bool(ok),
                             picked=bool(ok and picked < 4)))
        if ok and picked < 4:
            name = f"ct{cls}{picked}"
            path = os.path.abspath(f"{start_dir}/{name}.jpg")
            cv2.imwrite(path, cv2.cvtColor(fr[64], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            for arm, prompt in (("mis", generic), ("mat", r["prompt"])):
                out.append({"id": f"{name}_{arm}", "set": "content", "cls": cls, "arm": arm, "seed_gen": seed,
                            "image": path, "prompt": prompt})
            picked += 1
    print(cls, "picked", picked, [round(d.get("p_cls", -1), 2) for d in log[cls]], flush=True)
with open(p_out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
json.dump(log, open(f"{start_dir}/selection_log.json", "w"), indent=1)
