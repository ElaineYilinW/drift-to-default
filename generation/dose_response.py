"""Dose-response of latent brightness interventions: shift latents by an intended dL, decode, measure realised dL.

For saved per-video latents, take 3-latent-frame clips at several positions, add delta * w / (w.w) (luma direction
from fit_luma) or a gradient direction, decode with the Wan VAE, and compare the decoded mean L (and sharpness)
with the unshifted clip. Run from the Self-Forcing repo root.
usage: python dose_response.py LATENT_DIR LUMA_NPZ OUT.csv [--grad]
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

src, lumas, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]   # several direction files, comma-separated
use_grad = "--grad" in sys.argv
dev = torch.device("cuda")
vae = WanVAEWrapper().to(dev).eval()
dirs = {os.path.basename(f).replace(".npz", ""): torch.tensor(np.load(f)["w"], dtype=torch.float32, device=dev) for f in lumas}
DELTAS = [-6.0, -3.0, 0.0, 3.0, 6.0]
POS = [10, 40, 80, 110]


def decode(z):  # z: (3, 16, H, W) float -> uint8 frames (T, H, W, 3)
    v = vae.decode_to_pixel(z[None].float(), use_cache=False)[0]  # VAE weights stay fp32 here
    return ((v.float() * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def mean_L_grad(z):
    """Direction = d(mean decoded luminance proxy)/dz, normalised to a per-L-unit step via a finite check."""
    z = z.clone().float().requires_grad_(True)
    with torch.enable_grad():
        v = vae.model.decode((z[None].permute(0, 2, 1, 3, 4)).to(torch.float32),
                             [vae.mean.to(dev), 1.0 / vae.std.to(dev)])
        y = (v.clamp(-1, 1) * 0.5 + 0.5)
        lum = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).mean()
        lum.backward()
    return z.grad.detach()


rows = []
with torch.no_grad():
    for p in sorted(glob.glob(f"{src}/*.latents.npy")):
        m = re.match(r"(.+)_(.+)_s(\d+)\.latents\.npy", os.path.basename(p))
        lat = torch.from_numpy(np.load(p).astype(np.float32)).to(dev)
        for k in POS:
            z = lat[k:k + 3]
            base = frame_stats(decode(z))
            L0, sh0 = base["lab_mean"][:, 0].mean(), np.log(base["sharpness"] + 1).mean()
            units = {name: (w / (w @ w))[None, :, None, None] for name, w in dirs.items()}
            if use_grad:
                try:
                    g = mean_L_grad(z)
                except torch.cuda.OutOfMemoryError:
                    print("grad OOM, skipped", flush=True)
                    torch.cuda.empty_cache()
                    use_grad = False
            if use_grad:
                # calibrate: a small step along g, measure dL per unit step, then scale to intended deltas
                eps = 0.05 / g.abs().max()
                Ls = frame_stats(decode(z + eps * g))["lab_mean"][:, 0].mean()
                units["grad"] = eps * g / max(Ls - L0, 1e-3)
            for name, unit in units.items():
              for dl in DELTAS:
                st = frame_stats(decode(z + dl * unit)) if dl else base
                rows.append([name, m[1], int(m[3]), k, dl, st["lab_mean"][:, 0].mean() - L0,
                             np.log(st["sharpness"] + 1).mean() - sh0,
                             st["lab_mean"][:, 1].mean() - base["lab_mean"][:, 1].mean(),
                             st["lab_mean"][:, 2].mean() - base["lab_mean"][:, 2].mean(), L0])
        print("done", os.path.basename(p), flush=True)
with open(out, "w") as f:
    f.write("direction,prompt,seed,k,intended,realised_dL,dlogsharp,da,db,L0\n")
    for r in rows:
        f.write(",".join(str(v) for v in r) + "\n")
