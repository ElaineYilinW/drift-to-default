"""Per-latent-frame channel means aligned with decoded frame colour (for the latent -> brightness regression).

Latent frame k covers decoded frames {0} for k = 0 and 4k-3 .. 4k for k >= 1 (Wan VAE: 4x temporal compression,
first frame alone). Writes one csv row per latent frame: prompt, seed, k, m0..m15 (channel means), L, a, b.
usage: python latent_summary.py DIR OUT.csv
"""
import glob
import os
import re
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(f"{src}/*.latents.npy")):
    m = re.match(r"(.+)_(.+)_s(\d+)\.latents\.npy", os.path.basename(p))
    lat = np.load(p).astype(np.float32)                      # (K, 16, 60, 104)
    lab = np.load(p.replace(".latents.npy", ".npz"))["lab_mean"]
    cm = lat.mean(axis=(2, 3))
    for k in range(len(cm)):
        fr = [0] if k == 0 else list(range(4 * k - 3, 4 * k + 1))
        L, A, B = lab[fr].mean(0)
        rows.append([m[1], m[2], int(m[3]), k] + cm[k].tolist() + [L, A, B])
with open(out, "w") as f:
    f.write("prompt,tag,seed,k," + ",".join(f"m{i}" for i in range(16)) + ",L,a,b\n")
    for r in rows:
        f.write(",".join(str(v) for v in r) + "\n")
print(len(rows), "rows")
