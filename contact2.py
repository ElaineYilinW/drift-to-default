"""Contact sheet for one prompt/seed: one row per tag, columns at 0/10/20/31 s. usage: contact2.py DIR PROMPT SEED OUT TAG..."""
import sys
import cv2
import numpy as np
d, p, seed, out, tags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
rows = []
for tag in tags:
    cap, frames = cv2.VideoCapture(f"{d}/{p}_{tag}_s{seed}.mp4"), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    tiles = []
    for t in (0, 160, 320, 500):
        im = cv2.resize(frames[min(t, len(frames) - 1)], (320, 185))
        cv2.putText(im, f"{tag} {t / 16:.0f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(im)
    rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
