"""Per-frame colour statistics, identical to gen.py's frame_stats (gen.py is frozen, so this is a copy)."""
import cv2
import numpy as np


def frame_stats(video):
    """video: (T, H, W, 3) uint8 RGB -> dict of per-frame arrays."""
    lab_mean, lab_std, sat, val, sharp, hue = [], [], [], [], [], []
    for f in video:
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        lab[:, 0] *= 100 / 255  # 8-bit OpenCV Lab -> L in 0..100, a/b centred at 0
        lab[:, 1:] -= 128
        lab_mean.append(lab.mean(0))
        lab_std.append(lab.std(0))
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        sat.append(hsv[..., 1].mean() / 255)
        val.append(hsv[..., 2].mean() / 255)
        hue.append(np.bincount(hsv[..., 0].ravel(), minlength=180)[:180] / hsv[..., 0].size)
        sharp.append(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())
    return dict(lab_mean=np.array(lab_mean), lab_std=np.array(lab_std), sat_mean=np.array(sat),
                val_mean=np.array(val), sharpness=np.array(sharp), hue_hist=np.array(hue, dtype=np.float16))
