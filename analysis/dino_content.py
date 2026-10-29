"""Exploratory (third internal review, W4): DINOv2 features of the study 9 content videos, for an instance-level and a
second-encoder view of content persistence. Not preregistered.

For every video {opening}_{cond}_s{seed}.mp4 in VIDEO_DIR matching PATTERN: DINOv2-base CLS token and mean patch token
(as features.py) of frame 8 (the last held frame) and of one frame per second after it (frames 9 + 16k + 8, k = 0, 1,
...), on CPU. One .dino.npz per video in OUT_DIR (skipped if present).
usage: python dino_content.py VIDEO_DIR OUT_DIR PATTERN
"""
import glob
import os
import sys

import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

src, dst, pat = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst, exist_ok=True)
torch.set_num_threads(int(os.environ.get("NSLOTS", "4")))
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()
HOLD, FPS = 9, 16
for p in sorted(glob.glob(f"{src}/{pat}.mp4")):
    out = os.path.join(dst, os.path.basename(p)[:-4] + ".dino.npz")
    if os.path.exists(out):
        continue
    fr = []
    with av.open(p) as c:
        for fm in c.decode(video=0):
            fr.append(fm.to_ndarray(format="rgb24"))
    idx = [8] + list(range(HOLD + FPS // 2, len(fr), FPS))
    cls, pm = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 32):
            x = proc(images=[fr[j] for j in idx[i:i + 32]], return_tensors="pt")["pixel_values"]
            h = dino(pixel_values=x).last_hidden_state
            cls.append(h[:, 0].numpy())
            pm.append(h[:, 1:].mean(1).numpy())
    tmp = f"{out}.{os.getpid()}.tmp.npz"
    np.savez(tmp, idx=np.array(idx), cls=np.concatenate(cls).astype(np.float16),
             patchmean=np.concatenate(pm).astype(np.float16), n_frames=len(fr))
    os.replace(tmp, out)
    print(os.path.basename(out), len(fr), len(idx), flush=True)
