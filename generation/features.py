"""Per-frame DINOv2 embeddings and MUSIQ (SPAQ, as in VBench imaging quality) for every video of one prompt.

DINOv2-base on every 4th frame (CLS token and mean patch token), MUSIQ on every 8th frame with the longer side
resized to 512 (VBench's protocol). One .feat.npz per video next to it; existing ones are skipped.
usage: python features.py VIDEO_DIR PROMPT_ID DONE_FILE
"""
import glob
import os
import sys

import cv2
import numpy as np
import pyiqa
import torch
from transformers import AutoImageProcessor, AutoModel

vdir, pid, done = sys.argv[1], sys.argv[2], sys.argv[3]
dev = torch.device("cuda")
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval().half()
musiq = pyiqa.create_metric("musiq-spaq", device=dev)


def frames_of(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


videos = sorted(glob.glob(f"{vdir}/{pid}_*.mp4"))
assert videos, f"no videos for {pid}"
for v in videos:
    out = v[:-4] + ".feat.npz"
    if os.path.exists(out):
        continue
    fr = frames_of(v)
    di = np.arange(0, len(fr), 4)
    cls, pm = [], []
    with torch.no_grad():
        for i in range(0, len(di), 32):
            x = proc(images=[fr[j] for j in di[i:i + 32]], return_tensors="pt")["pixel_values"].to(dev).half()
            h = dino(pixel_values=x).last_hidden_state
            cls.append(h[:, 0].float().cpu().numpy())
            pm.append(h[:, 1:].mean(1).float().cpu().numpy())
        mi = np.arange(0, len(fr), 8)
        ms = []
        for j in mi:
            im = cv2.resize(fr[j], (512, round(512 * fr[j].shape[0] / fr[j].shape[1])), interpolation=cv2.INTER_AREA)
            ms.append(float(musiq(torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(dev))))
    tmp = f"{out}.{os.getpid()}.tmp.npz"
    np.savez(tmp, dino_idx=di, cls=np.concatenate(cls).astype(np.float16), patchmean=np.concatenate(pm).astype(np.float16),
             musiq_idx=mi, musiq=np.array(ms, dtype=np.float32))
    os.replace(tmp, out)
    print("done", os.path.basename(v), len(fr), flush=True)
open(done, "w").write(f"{len(videos)}\n")
