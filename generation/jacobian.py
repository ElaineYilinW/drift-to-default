"""Interventional estimate of how shifting each latent channel's spatial mean changes decoded colour.

Stage "probe": for 3-latent-frame clips of saved latents, shift channel c's mean by +/-DELTA (central difference),
decode with the Wan VAE, record d(L, a, b, log sharpness). One csv per video.
Stage "validate": given a direction file (npz with `u`: 16-d shift per +1 L), shift held-out clips by intended dL
and record realised d(L, a, b). Run from the Self-Forcing repo root.
usage: jacobian.py probe LATENTS.npy OUT.csv | jacobian.py validate LATENT_DIR DIRS(comma) OUT.csv SEEDS(comma)
"""
import glob
import os
import re
import sys

import numpy as np
import torch

from utils.wan_wrapper import WanVAEWrapper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_util import frame_stats  # noqa: E402

DELTA = 0.25
dev = torch.device("cuda")
vae = WanVAEWrapper().to(dev).eval()


def decode(z):
    v = vae.decode_to_pixel(z[None].float(), use_cache=False)[0]
    return ((v.float() * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def summ(z):
    s = frame_stats(decode(z))
    return np.r_[s["lab_mean"].mean(0), np.log(s["sharpness"] + 1).mean()]


with torch.no_grad():
    if sys.argv[1] == "probe":
        path, out = sys.argv[2], sys.argv[3]
        lat = torch.from_numpy(np.load(path).astype(np.float32)).to(dev)
        rows = []
        for k in (10, 80):
            z = lat[k:k + 3]
            for c in range(16):
                e = torch.zeros(16, device=dev)
                e[c] = DELTA
                d = (summ(z + e[None, :, None, None]) - summ(z - e[None, :, None, None])) / (2 * DELTA)
                rows.append([k, c] + d.tolist())
        tmp = out + f".{os.getpid()}.tmp"
        np.savetxt(tmp, np.array(rows), delimiter=",", header="k,c,dL,da,db,dlogsharp", comments="")
        os.replace(tmp, out)
    else:
        src, dirs, out, seeds = sys.argv[2], sys.argv[3].split(","), sys.argv[4], sys.argv[5].split(",")
        U = {os.path.basename(f).replace(".npz", ""): torch.tensor(np.load(f)["u"], dtype=torch.float32, device=dev)
             for f in dirs}
        rows = []
        for p in sorted(glob.glob(f"{src}/*.latents.npy")):
            m = re.match(r"(.+)_(.+)_s(\d+)\.latents\.npy", os.path.basename(p))
            if m[3] not in seeds:
                continue
            lat = torch.from_numpy(np.load(p).astype(np.float32)).to(dev)
            for k in (40, 110):
                z = lat[k:k + 3]
                b = summ(z)
                for name, u in U.items():
                    for dl in (-6.0, -3.0, 3.0, 6.0):
                        r = summ(z + dl * u[None, :, None, None]) - b
                        rows.append([name, m[1], m[3], k, dl] + r.tolist())
            print("done", os.path.basename(p), flush=True)
        with open(out, "w") as f:
            f.write("direction,prompt,seed,k,intended,dL,da,db,dlogsharp\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")
