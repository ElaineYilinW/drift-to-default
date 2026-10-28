"""Distances of study 27 (a sink held at one query-sink distance), chosen by a fixed rule from the exploratory 180-s
attention probe (sinkprobe_long_out.json; 4 openings x 6 chunks, distances 3..723), before any video of study 27:
  candidates: the probe's distances D with 24 <= D <= 720, except those within 2 of 187 or 203 (held in study 25);
  targets: K = 14 shares equally spaced from the smallest to the largest candidate share;
  for each target in turn, the unused candidate whose share is closest (ties: smaller D).
The three distances of study 25 (20, 187, 203) are generated again as a secondary set.
usage: python s27_distances.py sinkprobe_long_out.json
"""
import json
import sys

import numpy as np

K, DMIN, DMAX, AVOID, OLD = 14, 24, 720, (187, 203), (20, 187, 203)


def select(probe):
    g, m = probe["grid"], probe["mean"]
    cand = [(int(D), float(s)) for D, s in zip(g, m) if DMIN <= D <= DMAX and min(abs(D - a) for a in AVOID) > 2]
    sh = np.array([s for _, s in cand])
    used = []
    for t in np.linspace(sh.min(), sh.max(), K):
        D, s = min((c for c in cand if c[0] not in {u[0] for u in used}), key=lambda c: (abs(c[1] - t), c[0]))
        used.append((D, s))
    return sorted(used)


def share(probe, D):
    return float(probe["mean"][probe["grid"].index(D)])


if __name__ == "__main__":
    p = json.load(open(sys.argv[1]))
    for D, s in select(p):
        print(D, round(s, 4))
    print("old:", [(D, round(share(p, D), 4)) for D in OLD])
