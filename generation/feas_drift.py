"""Quick look at the feasibility videos: per-frame mean Lab colour (drift vs frame 0) and a contact sheet."""
import json
import sys

import cv2
import numpy as np

d = sys.argv[1]
names = ["f81_global", "f249_win21", "f249_win21_sink3"]
rows, out = [], {}
for n in names:
    cap, frames = cv2.VideoCapture(f"{d}/{n}.mp4"), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    lab = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float).mean(0) for f in frames])
    lab[:, 0] *= 100 / 255  # OpenCV 8-bit Lab -> L in 0..100, a/b offset by 128
    lab[:, 1:] -= 128
    de = np.linalg.norm(lab - lab[0], axis=1)  # ΔE of the frame-mean colour vs frame 0
    out[n] = {"frames": len(frames), "dE_at": {int(t): round(float(de[t]), 1) for t in (20, 40, 80, 160, 248) if t < len(de)},
              "L_first_last": [round(lab[0, 0], 1), round(lab[-1, 0], 1)],
              "b_first_last": [round(lab[0, 2], 1), round(lab[-1, 2], 1)]}
    picks = [t for t in (0, 80, 160, 248) if t < len(frames)]
    strip = [cv2.resize(frames[t], (416, 240)) for t in picks] + [np.zeros((240, 416, 3), np.uint8)] * (4 - len(picks))
    for i, t in enumerate(picks):
        cv2.putText(strip[i], f"{n}  t={t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    rows.append(np.hstack(strip))
cv2.imwrite(f"{d}/contact_sheet.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(json.dumps(out, indent=1))
