"""Exploratory motion audit for the third internal review (W1: is a high persistence a frozen video?). Not preregistered.

For every video {scene}_{arm}__{ver}_s{seed}.mp4 in RUN_DIR matching PATTERN (default *), decoded from the mp4 (crf 12):
  motion[i]    mean absolute grey change from frame i-1 to frame i on 4x-subsampled frames, 0-100 (motion[0] = 0)
  open_diff[i] mean absolute grey difference of frame i to frame 8 (the last held frame), 0-100
as hold_gen_sinkpos.py records them from the raw frames, plus the stream's frame rate. One npz per video in OUT_DIR
(skipped if present). Summaries are made by motion_summary.py.
usage: python motion_audit.py RUN_DIR OUT_DIR [PATTERN]
"""
import glob
import os
import sys

import av
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
pat = sys.argv[3] if len(sys.argv) > 3 else "*"
os.makedirs(dst, exist_ok=True)
for p in sorted(glob.glob(f"{src}/{pat}.mp4")):
    out = os.path.join(dst, os.path.basename(p)[:-4] + ".npz")
    if os.path.exists(out):
        continue
    g = []
    try:
        with av.open(p) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate) if s.average_rate else float("nan")
            for fr in c.decode(video=0):
                g.append(fr.to_ndarray(format="rgb24")[::4, ::4].astype(np.float32).mean(-1))
    except Exception as e:            # truncated or unreadable file: report and go on
        print("skip", p, e, flush=True)
        continue
    g = np.stack(g)
    motion = np.concatenate([[0.0], np.abs(g[1:] - g[:-1]).mean((1, 2)) * 100 / 255])
    open_diff = np.abs(g - g[min(8, len(g) - 1)]).mean((1, 2)) * 100 / 255
    tmp = f"{out}.{os.getpid()}.tmp.npz"
    np.savez(tmp, motion=motion.astype(np.float32), open_diff=open_diff.astype(np.float32), fps=fps)
    os.replace(tmp, out)
    print(os.path.basename(out), len(g), f"{fps:.1f}", flush=True)
