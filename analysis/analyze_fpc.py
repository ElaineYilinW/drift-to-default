"""Study 17: which route of FramePack's opening carries scene content (preregistered,
docs/preregistrations/17_framepack_content.md).

Files {opening}_mis_fpc_{arm}_s0.clip.npz (clip_probe.py), arm in both / ctx / emb for the 24 openings ct{cls}{k} of
preregistration 09, and ctcity{k}_mis_fpc_alt_s0.clip.npz for the 8 city starts. FramePack-F1, generic city prompt,
30 fps, frames 0..8 show the context image; frame f -> (f - 9) / 30 s after the opening.
Class score of a frame: z = scale * (cos(frame, class label) - cos(frame, city label)), the opening's class against city
(as preregistration 09). s[t0, t1) = mean z over frames in seconds [t0, t1); s_open = mean z over frames 1..8 of the
opening's both-arm video (the opening image itself); mu[t0, t1) = mean over the 8 city starts of the same window score
for that class. kappa = (s - mu) / (s_open - mu), undefined when s_open - mu < 2.
Primary (Holm over 4; per-opening values, two-sided one-sample t-tests; supported if Holm p < .05 and the estimate is on
the predicted side; window [15, 20)):
  H1 content relaxes in FramePack:             s(both) - s_open < 0
  H2 the context prefix carries content:       kappa(ctx) > 0
  H3 the SigLIP embedding carries content:     kappa(emb) > 0
  H4 the embedding adds content persistence:   kappa(both) - kappa(ctx) > 0
95% CIs: 10,000 bootstrap draws resampling openings within class and the city starts.
Secondary: kappa curves (windows [T-5, T), T = 5..20) per arm; additivity kappa(ctx) + kappa(emb) - kappa(both); H2-H4 on
tone-normalised embeddings; per-class kappa; top-1 accuracy of the opening's class among the 7 labels in [15, 20);
kappa(both, 20 s) next to Self Forcing DMD's kappa at 20 s from preregistration 09 (descriptive).
usage: python analyze_fpc.py CLIP_DIR TEXT_NPZ OUT_DIR
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

src, text_npz, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
TX = np.load(text_npz)
LAB, TEXT, SCALE = [str(x) for x in TX["labels"]], TX["text"].astype(np.float32), float(TX["scale"])
CLS = [c for c in LAB if c != "city"]
pat = re.compile(r"^ct([a-z]+)(\d)_mis_fpc_(both|ctx|emb|alt)_s0\.clip\.npz$")
WINS = [(t - 5, t) for t in range(5, 21)]
FPS, HOLD = 30, 9

V, C = {}, {}       # V[arm][opening][key] = dict(open, win, top1); C[city k][key] = {cls: {w: score}}
for p in sorted(glob.glob(f"{src}/*.clip.npz")):
    m = pat.match(os.path.basename(p))
    if not m:
        continue
    d = np.load(p)
    for key in ("emb", "emb_tn"):
        if key not in d:
            continue
        idx, z = d["idx"], SCALE * (d[key].astype(np.float32) @ TEXT.T)
        lo = z - z[:, [LAB.index("city")]]                          # (n, 7) log-odds of every label against city
        win = {w: (lo[(idx >= HOLD + FPS * w[0]) & (idx < HOLD + FPS * w[1])].mean(0)
                   if ((idx >= HOLD + FPS * w[0]) & (idx < HOLD + FPS * w[1])).any() else None) for w in WINS}
        if m[1] == "city":
            C.setdefault(int(m[2]), {})[key] = win
        else:
            cls, o = m[1], f"{m[1]}{m[2]}"
            ci_ = LAB.index(cls)
            sel = (idx >= HOLD + FPS * 15) & (idx < HOLD + FPS * 20)
            V.setdefault(m[3], {}).setdefault(o, {})[key] = dict(
                cls=cls, open=float(lo[(idx >= 1) & (idx < 9), ci_].mean()),
                win={w: (None if v is None else float(v[ci_])) for w, v in win.items()},
                top1=float(np.mean(z[sel].argmax(1) == ci_)) if sel.any() else None)
OPS = sorted(V.get("both", {}))
CITY = sorted(C)


def mu(cls, w, cities, key="emb"):
    v = [C[k][key][w][LAB.index(cls)] for k in cities if key in C[k] and C[k][key].get(w) is not None]
    return float(np.mean(v)) if v else None


def s_open(o, key="emb"):
    return V["both"][o][key]["open"] if o in V.get("both", {}) and key in V["both"][o] else None


def kap(arm, w, ops=None, cities=None, key="emb"):
    cities = CITY if cities is None else cities
    o_ = {}
    for o, r in V.get(arm, {}).items():
        if ops is not None and o not in ops or key not in r or r[key]["win"].get(w) is None or s_open(o, key) is None:
            continue
        m_ = mu(r[key]["cls"], w, cities, key)
        if m_ is None or s_open(o, key) - m_ < 2:
            continue
        o_[o] = (r[key]["win"][w] - m_) / (s_open(o, key) - m_)
    return o_


def boot(fn, n=10000):
    by = {c: [o for o in OPS if o[:-1] == c] for c in CLS if any(o[:-1] == c for o in OPS)}
    vals = []
    for _ in range(n):
        ops = [o for c in by for o in rng.choice(by[c], len(by[c]))]
        v = fn(ops, list(rng.choice(CITY, len(CITY))))
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]).tolist() if vals else None


def mean_over(ops, d):
    v = [d[o] for o in ops if o in d]
    return float(np.mean(v)) if v else None


def test(fn):
    """fn(ops, cities) -> {opening: value}; t-test on the full sample, CI by bootstrap."""
    d = fn(None, None)
    if len(d) < 3:
        return None
    v = np.array(list(d.values()))
    return dict(est=float(v.mean()), ci=boot(lambda ops, cs: mean_over(ops, fn(set(ops), cs))),
                p=float(stats.ttest_1samp(v, 0).pvalue), n=int(len(v)))


W = (15, 20)


def raw_drop(ops, cs, key="emb"):
    return {o: r[key]["win"][W] - s_open(o, key) for o, r in V.get("both", {}).items()
            if (ops is None or o in ops) and key in r and r[key]["win"].get(W) is not None}


def kdiff(a, b, key="emb"):
    def f(ops, cs):
        x, y = kap(a, W, ops, cs, key), kap(b, W, ops, cs, key)
        return {o: x[o] - y[o] for o in x if o in y}
    return f


tests = {"H1": (test(raw_drop), -1),
         "H2": (test(lambda ops, cs: kap("ctx", W, ops, cs)), +1),
         "H3": (test(lambda ops, cs: kap("emb", W, ops, cs)), +1),
         "H4": (test(kdiff("both", "ctx")), +1)}
prim = {k: dict(t, side=s) for k, (t, s) in tests.items() if t is not None}
running = 0.0
for i, k in enumerate(sorted(prim, key=lambda k: prim[k]["p"])):
    running = max(running, min(1.0, (4 - i) * prim[k]["p"]))
    prim[k]["p_holm"] = running
    prim[k]["supported"] = bool(running < 0.05 and np.sign(prim[k]["est"]) == prim[k]["side"])
res = {"counts": {a: len(v) for a, v in V.items()}, "city_starts": len(CITY), "primary": prim,
       "primary_missing": sorted(set(tests) - set(prim)), "kappa": {}, "per_class": {}, "top1": {}}
for arm in ("both", "ctx", "emb"):
    res["kappa"][arm] = {w[1]: float(np.mean(list(k.values()))) for w in WINS if (k := kap(arm, w))}
    k20 = kap(arm, W)
    res["per_class"][arm] = {c: float(np.mean([v for o, v in k20.items() if o[:-1] == c])) for c in CLS
                             if any(o[:-1] == c for o in k20)}
    t1 = [r["emb"]["top1"] for r in V.get(arm, {}).values() if r.get("emb", {}).get("top1") is not None]
    res["top1"][arm] = float(np.mean(t1)) if t1 else None
kb, kc, ke = kap("both", W), kap("ctx", W), kap("emb", W)
add = {o: kc[o] + ke[o] - kb[o] for o in kb if o in kc and o in ke}
res["additivity"] = dict(mean=float(np.mean(list(add.values()))), n=len(add)) if add else None
res["tone_normalised"] = {"H2": test(lambda ops, cs: kap("ctx", W, ops, cs, "emb_tn")),
                          "H3": test(lambda ops, cs: kap("emb", W, ops, cs, "emb_tn")),
                          "H4": test(kdiff("both", "ctx", "emb_tn"))}
json.dump(res, open(f"{out}/fpc_results.json", "w"), indent=1, default=float)
for k in sorted(prim):
    t = prim[k]
    print(k, f"est={t['est']:+.3f} ci=[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] n={t['n']} p_holm={t['p_holm']:.4f}",
          "supported" if t["supported"] else "not supported")
print("missing:", res["primary_missing"], "| kappa@20:", {a: round(v.get(20, float("nan")), 3) for a, v in res["kappa"].items()})
