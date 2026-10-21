"""Side-by-side GIF of two videos (e.g. no correction vs correction). usage: make_gif.py A.mp4 B.mp4 LABEL_A LABEL_B OUT.gif [step] [width]"""
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

a, b, la, lb, out = sys.argv[1:6]
step = int(sys.argv[6]) if len(sys.argv) > 6 else 3
w = int(sys.argv[7]) if len(sys.argv) > 7 else 320


def frames(p):
    cap, fs = cv2.VideoCapture(p), []
    while True:
        ok, f = cap.read()
        if not ok:
            return fs
        fs.append(f)


A, B = frames(a), frames(b)
h = round(w * A[0].shape[0] / A[0].shape[1])
out_frames = []
for i in range(0, min(len(A), len(B)), step):
    tiles = []
    for f, lab in ((A[i], la), (B[i], lb)):
        t = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{lab}  {i / 16:4.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        tiles.append(t)
    out_frames.append(cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB))
imageio.mimsave(out, out_frames, duration=1000 * step / 16 / 1.5, loop=0)
print(out, len(out_frames))
