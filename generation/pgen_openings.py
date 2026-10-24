"""Openings of the prompt-generality study (docs/preregistration_pgen_cn.md): frame 64 (4 s) of each 5-s Self Forcing
DMD fresh video pg_{domain}_sfgen_s{seed} (seeds 0..5) becomes opening pg{domain}{seed}; every opening is continued
under its own domain prompt. No selection.
usage: python pgen_openings.py GEN_DIR START_DIR PROMPTS_IN PROMPTS_OUT"""
import json
import os
import sys

import cv2

gen, start, p_in, p_out = sys.argv[1:5]
os.makedirs(start, exist_ok=True)
rows = []
for r in map(json.loads, open(p_in)):
    for seed in range(6):
        cap, fr = cv2.VideoCapture(f"{gen}/{r['id']}_sfgen_s{seed}.mp4"), None
        for i in range(65):
            ok, fr = cap.read()
            assert ok, (r["id"], seed, i)
        path = os.path.abspath(f"{start}/pg{r['domain']}{seed}.jpg")
        cv2.imwrite(path, fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"id": f"pg{r['domain']}{seed}", "set": "pgen", "domain": r["domain"], "image": path, "prompt": r["prompt"]})
with open(p_out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "openings")
