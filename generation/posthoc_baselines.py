"""Post-hoc colour-correction baselines for the content-aware correction (exploratory comparison).

Applied to already generated, uncorrected videos ({prompt}_none_s{seed}.mp4): every frame is colour-matched to the
anchor (the first chunk, frames 1-12) in CIELAB, without regenerating anything.
  phmean     : shift each frame's L*, a*, b* means to the anchor's (global colour constancy)
  phreinhard : Reinhard transfer, match each frame's per-channel mean and standard deviation to the anchor's
Writes {prompt}_{phmean|phreinhard}_s{seed}.npz with the same frame statistics as the generators, next to the inputs
(or into OUT_DIR). usage: python posthoc_baselines.py RUN_DIR [OUT_DIR] [PROMPT]
"""
import glob
import os
import sys

import av
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_util import frame_stats  # noqa: E402

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src
only = sys.argv[3] if len(sys.argv) > 3 else ""
os.makedirs(dst, exist_ok=True)


def to_lab(f):
    lab = cv2.cvtColor(f.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)     # L 0..100, a/b ~ -127..127
    return lab


def from_lab(lab):
    return (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255.0).round().astype(np.uint8)


for p in sorted(glob.glob(f"{src}/{only}*_none_s*.mp4")):
    if only and not os.path.basename(p).startswith(only + "_none_"):
        continue
    base = os.path.basename(p)[:-4]
    outs = {k: os.path.join(dst, base.replace("_none_", f"_{k}_") + ".npz") for k in ("phmean", "phreinhard")}
    if all(os.path.exists(o) for o in outs.values()):
        continue
    with av.open(p) as c:                                    # pass 1: anchor statistics (frames 1-12)
        anc = []
        for i, fr in enumerate(c.decode(video=0)):
            if 1 <= i <= 12:
                anc.append(to_lab(fr.to_ndarray(format="rgb24")).reshape(-1, 3))
            if i >= 12:
                break
    anc = np.concatenate(anc)
    am, asd = anc.mean(0), anc.std(0)
    acc = {k: [] for k in ("phmean", "phreinhard")}
    with av.open(p) as c:                                    # pass 2: stream, correct, measure frame by frame
        for fr in c.decode(video=0):
            l = to_lab(fr.to_ndarray(format="rgb24"))
            x = l.reshape(-1, 3)
            m, sd = x.mean(0), x.std(0)
            for k, y in (("phmean", x - m + am), ("phreinhard", (x - m) / np.maximum(sd, 1e-3) * asd + am)):
                acc[k].append(frame_stats(from_lab(y.reshape(l.shape).astype(np.float32))[None]))
    for k, lst in acc.items():
        st = {f: np.concatenate([d[f] for d in lst]) for f in lst[0]}
        tmp = outs[k] + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, **st)
        os.replace(tmp, outs[k])
    print(base, flush=True)
