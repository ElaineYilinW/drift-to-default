"""Exploratory (third internal review, W9): which layers, heads and denoising passes carry the peaks of the
attention-to-sink probe? Not preregistered.

Probe files of jobs/run_sinkprobe.sh (openings rx00, rx01, gx00, gx01; chunks 12..240; distances 3..300): mass[chunk] of
shape (calls, grid, heads), calls = passes x 30 layers (4 denoising passes and the context pass). Per layer, head and
pass: the mean share over the other axes, chunks and openings, and the peak contrast (mean share at 71, 134, 203, 266
minus the median share over distances 10..300). Also the share of the summed positive (layer, head) contrast carried by
the top 10% of heads, and the correlation across layers of the mean share with the contrast.
usage: python analyze_sinkprobe_heads.py PROBE_DIR OUT.json
"""
import glob
import json
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
NL, PEAKS = 30, [71, 134, 203, 266]
acc = []
for p in sorted(glob.glob(f"{src}/[a-z][a-z][0-9][0-9].npz")):
    d = np.load(p)
    g = d["grid"]
    for c in d["chunks"]:
        a = d[f"c{c}"]
        npass = a.shape[0] // NL
        acc.append(a[:npass * NL].reshape(npass, NL, len(g), -1))     # (pass, layer, grid, head)
A = np.mean(acc, 0)
pk = np.isin(g, PEAKS)
med = g >= 10


def contrast(x):   # x (..., grid)
    return x[..., pk].mean(-1) - np.median(x[..., med], -1)


LH = A.mean(0).transpose(0, 2, 1)           # (layer, head, grid)
c_lh = contrast(LH)
c_l = contrast(A.mean((0, 3)))
c_p = contrast(A.mean((1, 3)))
share_l = A.mean((0, 2, 3))
pos = np.clip(c_lh, 0, None).ravel()
top = np.sort(pos)[::-1]
k = max(1, len(top) // 10)
res = {"n_curves": len(acc), "contrast_all": float(contrast(A.mean((0, 1, 3)))),
       "contrast_by_layer": c_l.tolist(), "share_by_layer": share_l.tolist(), "contrast_by_pass": c_p.tolist(),
       "contrast_layer_head": c_lh.tolist(), "top10pct_heads_share_of_positive_contrast": float(top[:k].sum() / top.sum()),
       "n_heads_positive": int((c_lh > 0.01).sum()), "n_heads": int(c_lh.size),
       "corr_share_contrast_layers": float(np.corrcoef(share_l, c_l)[0, 1]),
       "top_heads": [(int(i // c_lh.shape[1]), int(i % c_lh.shape[1]), float(c_lh.ravel()[i])) for i in np.argsort(-c_lh.ravel())[:10]]}
json.dump(res, open(out, "w"))
print("curves", len(acc), "contrast all", round(res["contrast_all"], 4))
print("by layer:", " ".join(f"{i}:{x:+.3f}" for i, x in enumerate(c_l)))
print("share by layer:", " ".join(f"{i}:{x:.3f}" for i, x in enumerate(share_l)))
print("by pass:", [round(x, 4) for x in c_p])
print("top 10% heads carry", round(res["top10pct_heads_share_of_positive_contrast"], 2), "of positive contrast;",
      res["n_heads_positive"], "of", res["n_heads"], "heads > 0.01")
print("top heads (layer, head, contrast):", res["top_heads"])
