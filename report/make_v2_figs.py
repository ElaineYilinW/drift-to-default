"""Figures of the second round of studies (preregistrations 09-11) and the drift decomposition.
usage: python report/make_v2_figs.py [RESULTS_DIR] [OUT_DIR]   (defaults: results/, report/)
Each figure is drawn only if its results file exists."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
O = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(__file__)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, NAVY, ORANGE, RUST, GREEN, TEAL, PURPLE, GREY = ("#2a78d6", "#184f95", "#eb6834", "#a8441d", "#1baf7a", "#0d6b49",
                                                     "#7a5bd6", "#898781")
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
NAME = {"sf60_s3": "Self Forcing DMD", "sf60_s0": "SF, no sink", "sf60_s1": "SF, sink 1", "sf60_s6": "SF, sink 6",
        "sf60_s3rr": "SF, sink 3 re-rotated", "sf60_s1rr": "SF, sink 1 re-rotated", "ll60_s3": "LongLive",
        "ll60_s0": "LongLive, no sink", "gan60_s3": "SF-GAN", "sid60_s3": "SF-SiD", "sidv260_s3": "SF-SiD v2",
        "sf10s60_s3": "SF-10s", "rf60": "Rolling Forcing", "cv60_s3": "CausVid", "cv60_s0": "CausVid, no sink",
        "ll_s3": "LongLive", "ll_s0": "LongLive, no sink", "cv_s3": "CausVid", "gan_s3": "SF-GAN", "sid_s3": "SF-SiD",
        "rf": "Rolling Forcing", "sf_s3": "Self Forcing DMD", "sf_s0": "SF, no sink", "sf_s1": "SF, sink 1",
        "sidv2_s3": "SF-SiD v2", "sf10s_s3": "SF-10s"}
REL31 = {"sf60_s3": "sf_s3", "sf60_s0": "sf_s0", "sf60_s1": "sf_s1", "ll60_s3": "ll_s3", "ll60_s0": "ll_s0",
         "gan60_s3": "gan_s3", "sid60_s3": "sid_s3", "sidv260_s3": "sidv2_s3", "sf10s60_s3": "sf10s_s3", "rf60": "rf",
         "cv60_s3": "cv_s3", "cv60_s0": None}


def load(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


def errbar(ax, x, y, ci, **kw):
    ax.errorbar(x, y, xerr=[[x - ci[0]], [ci[1] - x]], fmt="o", ms=4, lw=1.2, **kw)


# ---- atlas (preregistration 10) ------------------------------------------------------------------------------------
at, rel, lg = load("corr/atlas_out/atlas_results.json"), load("corr/relax_out/relax_results.json"), \
    load("corr/long60_out/long_results.json")
if at:
    A = {c: v[str(60)] if str(60) in v else v.get(60) for c, v in at["alpha"].items()}
    if lg:                                        # CausVid at 60 s comes from preregistration 07
        for c in ("cv60_s3", "cv60_s0"):
            v = lg["alpha"].get(c, {})
            if "60" in v:
                A[c] = v["60"]
    A = {c: v for c, v in A.items() if v}
    order = sorted(A, key=lambda c: A[c]["mean"])
    fig = plt.figure(figsize=(10.5, 4.9))
    ax = fig.add_subplot(1, 3, 1)
    for i, c in enumerate(order):
        errbar(ax, A[c]["mean"], i, A[c]["ci"], color=BLUE, label="60 s" if i == 0 else None)
        r = REL31.get(c)
        if rel and r and rel["per_cfg"].get(r, {}).get("alpha_L"):
            ax.plot(rel["per_cfg"][r]["alpha_L"]["mean"], i, "o", mfc="white", mec=GREY, ms=4,
                    label="31 s (study 06)" if i == 0 else None)
    ax.set_yticks(range(len(order)), [NAME.get(c, c) for c in order], fontsize=7.5)
    ax.axvline(0, color=MUTED, lw=.8)
    ax.set_xlabel("persistence α of an L* shift")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(a) persistence at 60 s", fontsize=9, loc="left")
    for j, (title, cfgs) in enumerate((("(b) Self Forcing: sink size and position",
                                        [("sf60_s0", GREY, "-"), ("sf60_s1", ORANGE, "-"), ("sf60_s1rr", ORANGE, "--"),
                                         ("sf60_s3", BLUE, "-"), ("sf60_s3rr", BLUE, "--"), ("sf60_s6", NAVY, "-")]),
                                       ("(c) other models", [("ll60_s3", ORANGE, "-"), ("ll60_s0", RUST, "--"),
                                                             ("gan60_s3", GREEN, "-"), ("sid60_s3", TEAL, "-"),
                                                             ("rf60", PURPLE, "-"), ("sf10s60_s3", NAVY, "-")]))):
        ax = fig.add_subplot(1, 3, 2 + j)
        for c, col, ls in cfgs:
            if c in at["curves"] and at["curves"][c]:
                t, a = map(np.array, zip(*at["curves"][c]))
                k = np.ones(3) / 3
                ax.plot(t[1:-1], np.convolve(a, k, "valid"), color=col, ls=ls, lw=1.6, label=NAME[c])
        ax.axhline(0, color=MUTED, lw=.8)
        ax.axvspan(31, 35, color=GRID, alpha=.5, lw=0)
        ax.axvspan(50, 54, color=GRID, alpha=.5, lw=0)
        ax.set_xlabel("seconds after the opening")
        ax.set_ylabel("persistence α(t)")
        ax.legend(fontsize=6.6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{O}/fig_atlas.pdf")
    print("ok atlas")

# ---- content (preregistration 09) ----------------------------------------------------------------------------------
ct = load("corr/content_out/content_results.json")
TILES = os.path.join(os.path.dirname(__file__), "tiles")
if ct:
    from PIL import Image
    fig = plt.figure(figsize=(10.5, 5.2))
    rows = [("ctsnow0_mis_sf60_s3", "snow\ncity prompt"), ("ctsnow0_mat_sf60_s3", "snow\nsnow prompt"),
            ("ctdesert0_mis_sf60_s3", "desert\ncity prompt"), ("ctsnow0_mis_sf60_s0", "snow, no sink\ncity prompt")]
    gs = fig.add_gridspec(len(rows), 4, left=0.07, right=0.56, top=0.96, bottom=0.47, wspace=0.03, hspace=0.05)
    for i, (r, lab) in enumerate(rows):
        for j, tt in enumerate((0, 10, 30, 60)):
            a_ = fig.add_subplot(gs[i, j])
            a_.imshow(Image.open(f"{TILES}/content/{r}_s0__t{tt}.jpg"))
            a_.set_xticks([]), a_.set_yticks([])
            for s in a_.spines.values():
                s.set_visible(False)
            if i == 0:
                a_.set_title("opening" if tt == 0 else f"{tt} s", fontsize=8)
            if j == 0:
                a_.set_ylabel(lab, fontsize=6.8)
    ax = fig.add_axes([0.64, 0.53, 0.34, 0.42])
    for c, col in (("sf60_s3", BLUE), ("sf60_s0", GREY), ("ll_s3", ORANGE), ("ll_s0", RUST)):
        for arm, ls in (("mis", "-"), ("mat", ":")):
            v = ct["kappa"].get(c, {}).get(arm, {})
            if v:
                t = sorted(int(k) for k in v)
                ax.plot(t, [v[str(k)]["mean"] for k in t], color=col, ls=ls, lw=1.6,
                        label=f"{NAME[c]}" if arm == "mis" else None)
    ax.plot([], [], color=MUTED, ls="-", label="generic city prompt")
    ax.plot([], [], color=MUTED, ls=":", label="class prompt")
    ax.axhline(0, color=MUTED, lw=.8)
    ax.axhline(1, color=GRID, lw=.8)
    ax.set_xlabel("seconds after the opening")
    ax.set_ylabel("content persistence κ(t)")
    ax.set_ylim(-0.42, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=6.3, ncol=3, loc="lower center", columnspacing=0.9, handlelength=1.8)
    ax.set_title("(a) scene class over time", fontsize=9, loc="left")
    ax = fig.add_axes([0.13, 0.105, 0.33, 0.265])
    cfgs = [c for c in ("sf60_s3", "sf60_s0", "ll_s3", "ll_s0", "cv_s3", "gan_s3", "sid_s3", "rf") if c in ct["kappa"]]
    for i, c in enumerate(cfgs):
        for arm, col, dy in (("mis", BLUE, -0.15), ("mat", GREEN, 0.15)):
            v = ct["kappa"][c].get(arm, {}).get("30")
            if v and v.get("ci"):
                errbar(ax, v["mean"], i + dy, v["ci"], color=col,
                       label={"mis": "generic city prompt", "mat": "class prompt"}[arm] if i == 0 else None)
    ax.set_yticks(range(len(cfgs)), [NAME[c] for c in cfgs], fontsize=7.5)
    ax.axvline(0, color=MUTED, lw=.8)
    ax.axvline(1, color=GRID, lw=.8)
    ax.set_xlabel("content persistence κ at 30 s")
    v0 = ct["kappa"][cfgs[0]]
    ax.annotate("generic\ncity prompt", (v0["mis"]["30"]["mean"], -0.15), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=6.6, color=BLUE)
    ax.annotate("class\nprompt", (v0["mat"]["30"]["mean"], 0.15), xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=6.6, color=GREEN)
    ax.set_ylim(len(cfgs) - 0.5, -1.0)
    ax.set_title("(b) the prompt sets the content default", fontsize=9, loc="left", pad=16)
    ax = fig.add_axes([0.64, 0.105, 0.34, 0.265])
    pairs = ct.get("spearman_kappa_alphaL", {}).get("pairs") if "spearman_kappa_alphaL" in ct else None
    if rel:
        xs, ys, ls_ = [], [], []
        for c in cfgs:
            r = {"sf60_s3": "sf_s3", "sf60_s0": "sf_s0"}.get(c, c)
            v = ct["kappa"][c].get("mis", {}).get("30")
            if v and rel["per_cfg"].get(r, {}).get("alpha_L"):
                xs.append(rel["per_cfg"][r]["alpha_L"]["mean"]), ys.append(v["mean"]), ls_.append(NAME[c])
        ax.scatter(xs, ys, s=22, color=BLUE)
        nudge = {"CausVid": (-4, -9), "SF, no sink": (4, -6), "SF-GAN": (-4, -11)}
        for x, y, l in zip(xs, ys, ls_):
            dx, dy = nudge.get(l, (3, 2))
            ax.annotate(l, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=6.6,
                        ha="right" if dx < 0 else "left")
        sp = ct.get("spearman_kappa_alphaL")
        ax.set_title(f"(c) content vs tone persistence" + (f" (ρ = {sp['rho']:.2f})" if sp else ""), fontsize=9, loc="left")
    ax.set_xlabel(r"brightness persistence $\alpha_L$ at 31 s (study 06)")
    ax.set_ylabel("content persistence κ at 30 s")
    fig.savefig(f"{O}/fig_content.pdf")
    print("ok content")

# ---- Other architectures: FramePack (11, 17), MAGI-1 (18, 19) and the world model (15) ---------------------------------
fp, sk, mg = load("corr/fp_out/fp_results.json"), load("corr/skyconf_out/skyov_results.json"), \
    load("corr/mg_out/mg_explore_15s.json")
fpc, mcv = load("corr/fpc_out/fpc_results.json"), load("corr/magi_out/magi_curve.json")
if fp:
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.2))
    ax = axs.ravel()
    for c, col, lab in (("fpv_both", BLUE, "both routes"), ("fpv_ctx", ORANGE, "context only"),
                        ("fpv_emb", GREEN, "image embedding only"), ("fpi_both", NAVY, "both, anti-drifting sampler")):
        v = fp["alpha"].get(c, {})
        if v:
            tt = sorted(int(k) for k in v)
            ax[0].plot(tt, [v[str(k)]["mean"] for k in tt], "o-", color=col, lw=1.6, ms=3.5, label=lab)
            ax[0].fill_between(tt, [v[str(k)]["ci"][0] for k in tt], [v[str(k)]["ci"][1] for k in tt], color=col, alpha=.12, lw=0)
    if sk:
        for key, ls, lab in (("17", "--", "SkyReels, 17 overlap frames"), ("5", ":", "SkyReels, 5 overlap frames")):
            tc = sk["time_course"].get(key)
            if tc:
                tt = sorted(float(k) for k in tc)
                ax[0].plot(tt, [tc[f"{x:g}"] for x in tt], color=GREY, ls=ls, lw=1.3, label=lab)
    ax[0].set_title("(a) FramePack (13B, HunyuanVideo-based): tone", fontsize=9, loc="left")
    ax[0].set_ylabel("persistence α(t)")
    if fpc:
        for c, col, lab in (("both", BLUE, "both routes"), ("ctx", ORANGE, "context only"),
                            ("emb", GREEN, "image embedding only")):
            v = fpc["kappa"].get(c, {})
            tt = sorted(int(k) for k in v)
            ax[1].plot(tt, [v[str(k)] for k in tt], "o-", color=col, lw=1.6, ms=3.5, label=lab)
        ax[1].set_title("(b) FramePack: scene content", fontsize=9, loc="left")
        ax[1].set_ylabel("content persistence κ(t)")
        ax[1].legend(fontsize=6.5, loc="upper right")
    if mcv:
        for c, col, lab in (("mgw", ORANGE, "released window (4 chunks)"), ("mgf", BLUE, "full history"),
                            ("mgw60", ORANGE, None)):
            v = mcv.get(c, {})
            if not v:
                continue
            tt = sorted(int(k) for k in v)
            y = np.array([v[str(k)]["mean"] for k in tt])
            m = np.convolve(y, np.ones(3), mode="same") / np.convolve(np.ones_like(y), np.ones(3), mode="same")
            ax[2].plot(tt, m, color=col, lw=1.6, ls=":" if c == "mgw60" else "-", label=lab)
            ax[2].fill_between(tt, [v[str(k)]["ci"][0] for k in tt], [v[str(k)]["ci"][1] for k in tt], color=col,
                               alpha=.10, lw=0)
        ax[2].set_title("(c) MAGI-1 (4.5B, own architecture): tone", fontsize=9, loc="left")
        ax[2].set_ylabel("persistence α(t)")
        ax[2].legend(fontsize=6.5, loc="lower left")
    if mg:
        for c, col, lab in (("mg_both", BLUE, "both routes"), ("mg_ctx", ORANGE, "context only"),
                            ("mg_emb", GREEN, "image embedding only")):
            v = mg["alpha"].get(c, {})
            tt = sorted(int(k) for k in v)
            ax[3].plot(tt, [v[str(k)]["est"] for k in tt], "o-", color=col, lw=1.6, ms=3.5, label=lab)
            ax[3].fill_between(tt, [v[str(k)]["ci"][0] for k in tt], [v[str(k)]["ci"][1] for k in tt], color=col, alpha=.12, lw=0)
        ax[3].set_title("(d) Matrix-Game-2.0 world model (exploratory 15 s)", fontsize=9, loc="left")
        ax[3].set_ylabel("persistence α(t)")
        ax[3].legend(fontsize=6.5, loc="center right")
    for a_ in ax:
        a_.axhline(0, color=MUTED, lw=.8)
        a_.axhline(1, color=GRID, lw=.8)
    for a_ in ax[2:]:
        a_.set_xlabel("seconds after the opening")
    ax[0].legend(fontsize=6.3, loc="center right", bbox_to_anchor=(1.0, 0.30), ncol=2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(f"{O}/fig_framepack.pdf")
    print("ok framepack")

# ---- inputs of the combined mechanism figure ------------------------------------------------------------------
dc = load("corr/review_out/review_analyses.json")
ps = load("corr/pswap_out/pswap_results.json")

# ---- combined mechanism figure: drift decomposition + prompt pull ----------------------------------------------------
if dc and ps:
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
    P = dc["A_decomposition"]["per_cfg"]
    LONG = {"ll_s3", "ll_s0", "rf"}
    for c, v in P.items():
        ax[0].scatter(v["alpha"], v["offset"], s=22, color=ORANGE if c in LONG else BLUE)
        off = {"sf_s3": (-30, 6), "sidv2_s3": (4, 4), "sf10s_s3": (4, -9), "ll_s0": (4, 3), "ll_s3": (4, -8)}.get(c, (3, 2))
        ax[0].annotate({"sf_s3": "SF-DMD"}.get(c, NAME.get(c, c)), (v["alpha"], v["offset"]), textcoords="offset points",
                       xytext=off, fontsize=7)
    ax[0].scatter([], [], color=ORANGE, s=22, label="trained for long rollouts")
    ax[0].scatter([], [], color=BLUE, s=22, label="trained on 5–10-s clips")
    ax[0].axhline(0, color=MUTED, lw=.8)
    ax[0].set_xlabel("persistence α of the opening (31 s)")
    ax[0].set_ylabel("fresh end − fresh start (L*)")
    ax[0].legend(fontsize=7.2, loc="upper right")
    ax[0].set_title("(a) forgetting vs attractor offset", fontsize=9, loc="left")
    ax[1].axhspan(0.04, 0.14, color=GRID, alpha=.6, lw=0, label="predicted (120-s fit)")
    for s, col, lab, dx in (("3", BLUE, "sink 3", -0.25), ("0", GREY, "no sink", 0.25)):
        v = ps["w_p"][s]
        ks = sorted(int(k) for k in v)
        x = np.array([k * 0.75 for k in ks]) + dx
        m = np.array([v[str(k)]["mean"] for k in ks])
        lo, hi = np.array([v[str(k)]["ci"][0] for k in ks]), np.array([v[str(k)]["ci"][1] for k in ks])
        ax[1].errorbar(x, m, yerr=[m - lo, hi - m], fmt="o-", color=col, ms=4, lw=1.3, label=lab)
    ax[1].set_xlabel("time of the swapped chunk (s)")
    ax[1].set_ylabel("prompt pull $w_p$ per chunk")
    ax[1].set_xticks([6, 15, 30])
    ax[1].legend(fontsize=6.5, loc="upper left")
    ax[1].set_title("(b) the prompt's pull, by replay", fontsize=9, loc="left")
    MU = {"sunny": 42.9, "overcast": 33.6, "dusk": 19.1, "night": 16.0}
    for k, mk in (("8", "o"), ("20", "s"), ("40", "^")):
        r = ps["ratio"]["3"][k]
        ax[2].scatter([MU[b] - 33.1 for b in r], [r[b] * (MU[b] - 33.1) for b in r], marker=mk, color=BLUE, s=20,
                      label={"8": "6 s", "20": "15 s", "40": "30 s"}[k])
    pl = ps["placebo_overcast_dL"]["3"]
    ax[2].scatter([0.5] * 3, [pl[k]["mean"] for k in ("8", "20", "40")], marker="x", color=ORANGE, s=20, label="same default")
    ax[2].axhline(0, color=MUTED, lw=.8), ax[2].axvline(0, color=MUTED, lw=.8)
    ax[2].set_xlabel("default of swapped prompt − generic (L*)")
    ax[2].set_ylabel("change of the chunk (L*)")
    ax[2].legend(fontsize=6.5, loc="upper left")
    ax[2].set_title("(c) one swapped chunk, sink 3", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{O}/fig_mech.pdf")
    print("ok mech")
