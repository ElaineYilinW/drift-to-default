"""Exploratory (third internal review, W4): class- and instance-level content persistence of the study 9 videos from
DINOv2 features (dino_content.py). Not preregistered.

Openings ct{class}{i} (6 classes x 4); each continued under the city prompt (mis) and its own class prompt (mat) by 8
configurations; fresh city-prompt rollouts fresh_{cfg}_s{seed}. f = L2-normalised DINOv2 feature (CLS, and mean patch
token as a check); f_o = the opening's (frame 8, averaged over its videos).
  class score   c(f, o)   = mean cos(f, f_o') over the other 3 openings of o's class - mean cos(f, f_o') over the 20 openings
                            of other classes;
  kappa_D(t)    = (c(f_t, o) - c_fresh(o)) / (c(f_o, o) - c_fresh(o)), c_fresh(o) = mean of c over the frames of that
                  configuration's fresh rollouts at the same second (0 = fresh city rollout, 1 = the opening), the
                  DINOv2 analogue of the CLIP kappa;
  instance m(t) = cos(f_t, f_o) - mean cos(f_t, f_o') over the other 3 openings of o's class (0 in expectation for a
                  video unrelated to o within its class);
  iota(t)       = m(t) / m(f_o) (1 = the opening itself); iota_1s(t) = mean m(t) / mean m(first second after the
                  hold), a ratio of opening means (bootstrap over openings).
Per configuration x prompt x second: scene means with opening-bootstrap 95% CIs (24 openings). Contrasts at 30 and 60 s
(5-s windows ending there): mat - mis for kappa_D and iota; sf60_s3 - sf60_s0; and, with CLIP_JSON (content_out
results), the correlation of kappa_D with CLIP's kappa across configurations x prompts at 30 s.
usage: python analyze_dino_content.py DINO_DIR OUT.json [CLIP_JSON]
"""
import glob
import json
import os
import re
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
clip = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
rng = np.random.default_rng(0)
pat = re.compile(r"^ct(?P<cls>[a-z]+)(?P<i>\d)_(?P<pr>mis|mat)_(?P<cfg>.+)_s(?P<seed>\d+)\.dino\.npz$")
fpat = re.compile(r"^fresh_(?P<cfg>.+)_s(?P<seed>\d+)\.dino\.npz$")
WIN = 5


def unit(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def boot(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile([rng.choice(v, len(v)).mean() for _ in range(5000)], [2.5, 97.5])]


res = {}
for feat in ("cls", "patchmean"):
    V, F, O = {}, {}, {}
    for p in glob.glob(f"{src}/*.dino.npz"):
        b = os.path.basename(p)
        d = np.load(p)
        f = unit(d[feat])
        m = pat.match(b)
        if m:
            o = f"{m['cls']}{m['i']}"
            V[(m["cfg"], m["pr"], o, m["seed"])] = f[1:]
            O.setdefault(o, []).append(f[0])
            continue
        m = fpat.match(b)
        if m:
            F.setdefault(m["cfg"], []).append(f[1:])
    ops = sorted(O)
    cls_of = {o: re.sub(r"\d", "", o) for o in ops}
    fo = unit(np.stack([np.mean(O[o], 0) for o in ops]))           # (24, d)
    idx = {o: k for k, o in enumerate(ops)}
    same = {o: [idx[x] for x in ops if cls_of[x] == cls_of[o] and x != o] for o in ops}
    diff = {o: [idx[x] for x in ops if cls_of[x] != cls_of[o]] for o in ops}

    def cscore(f, o):          # f (..., d)
        s = f @ fo.T
        return s[..., same[o]].mean(-1) - s[..., diff[o]].mean(-1)

    def inst(f, o):
        s = f @ fo.T
        return s[..., idx[o]] - s[..., same[o]].mean(-1)

    c_open = {o: float(cscore(fo[idx[o]], o)) for o in ops}
    m_open = {o: float(inst(fo[idx[o]], o)) for o in ops}
    R = {}
    for cfg in sorted({k[0] for k in V}):
        fr = F.get(cfg)
        nT = min(len(x) for k, x in V.items() if k[0] == cfg)
        if fr:
            nT = min(nT, min(len(x) for x in fr))
        for pr in ("mis", "mat"):
            per = {}
            for (c_, p_, o, sd), f in V.items():
                if c_ != cfg or p_ != pr:
                    continue
                f = f[:nT]
                cf = np.mean([cscore(x[:nT], o) for x in fr], 0) if fr else np.zeros(nT)
                kap = (cscore(f, o) - cf) / (c_open[o] - cf)
                m = inst(f, o)
                per.setdefault(o, []).append((kap, m / m_open[o], m))
            if not per:
                continue
            arr = {o: np.mean(np.array(v), 0) for o, v in per.items()}  # (3, nT) per opening, seed mean
            st = np.stack(list(arr.values()))                            # (n_open, 3, nT)
            row = {"n": len(arr), "curves": {k: st[:, j].mean(0).tolist() for j, k in enumerate(("kappa_D", "iota", "m"))}}
            row["curves"]["iota_1s"] = (st[:, 2].mean(0) / st[:, 2, 0].mean()).tolist()
            for T in (30, 60):
                if T <= nT:
                    w = st[:, :, T - WIN:T].mean(-1)
                    row[T] = {k: dict(mean=float(w[:, j].mean()), ci=boot(w[:, j]))
                              for j, k in enumerate(("kappa_D", "iota", "m"))}
                    m0 = st[:, 2, 0]
                    bs = []
                    for _ in range(5000):
                        i = rng.integers(0, len(m0), len(m0))
                        bs.append(w[i, 2].mean() / m0[i].mean())
                    row[T]["iota_1s"] = dict(mean=float(w[:, 2].mean() / m0.mean()),
                                             ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])])
                    row[T]["_per_open"] = {o: w[k].tolist() for k, o in enumerate(arr)}
            R[f"{cfg}/{pr}"] = row
    # contrasts
    C = {}
    for T in (30, 60):
        for cfg in sorted({k.split("/")[0] for k in R}):
            a, b = R.get(f"{cfg}/mat", {}).get(T), R.get(f"{cfg}/mis", {}).get(T)
            if a and b:
                oo = sorted(set(a["_per_open"]) & set(b["_per_open"]))
                for j, k in enumerate(("kappa_D", "iota")):
                    d = [a["_per_open"][o][j] - b["_per_open"][o][j] for o in oo]
                    C[f"{cfg} mat-mis {k} T{T}"] = dict(est=float(np.mean(d)), ci=boot(d))
        for pr in ("mis", "mat"):
            a, b = R.get(f"sf60_s3/{pr}", {}).get(T), R.get(f"sf60_s0/{pr}", {}).get(T)
            if a and b:
                oo = sorted(set(a["_per_open"]) & set(b["_per_open"]))
                for j, k in enumerate(("kappa_D", "iota")):
                    d = [a["_per_open"][o][j] - b["_per_open"][o][j] for o in oo]
                    C[f"sink-none {pr} {k} T{T}"] = dict(est=float(np.mean(d)), ci=boot(d))
    if clip:
        pts = []
        for key, row in R.items():
            cfg, pr = key.split("/")
            ck = clip.get("kappa30", {}).get(f"{cfg}/{pr}")
            if ck is not None and 30 in row:
                pts.append((row[30]["kappa_D"]["mean"], ck))
        if len(pts) >= 5:
            C["corr_clip_kappa30"] = dict(n=len(pts), r=float(np.corrcoef(*zip(*pts))[0, 1]))
    for key, row in R.items():
        for T in (30, 60):
            if T in row:
                row[T].pop("_per_open")
    res[feat] = dict(configs=R, contrasts=C)
    print(f"== {feat}")
    for key, row in R.items():
        s = " ".join(f"T{T} kD {row[T]['kappa_D']['mean']:+.2f} iota {row[T]['iota']['mean']:+.2f} "
                     f"iota1s {row[T]['iota_1s']['mean']:+.2f}" for T in (30, 60) if T in row)
        print(f"{key:16s} n={row['n']:2d} {s}")
    for k, v in C.items():
        print(k, json.dumps(v))
json.dump(res, open(out, "w"))
