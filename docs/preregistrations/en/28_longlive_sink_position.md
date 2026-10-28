> English translation of the frozen Chinese preregistration [`../28_longlive_sink_position.md`](../28_longlive_sink_position.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 28: do LongLive's sink bursts also follow the rotary distance, and with the sink held at one distance, does LongLive's own probe share decide how much is kept and how much the video moves? (LongLive, 60 s) (2026-10-28, frozen before generation)

## Background
- Results on Self Forcing:
  - Preregistration 25 (lab note 31): with the sink 18 or 36 latent frames further back, the bursts come 4.5 or 9 s
    earlier. Held at the peak distance 203 the sink keeps 0.86, but the video freezes; at the trough 187 it keeps 0.01.
  - Preregistration 27: the sink held at 17 distances, testing whether persistence and motion follow the probe's share
    (frozen, running).
- LongLive is a long-video fine-tune of Self Forcing (window 12, sink 3, the sink at its original positions).
  - Its bursts come at the same times as Self Forcing's (the atlas of preregistration 10).
  - **Design check (on existing data of preregistration 10):** with the burst windows defined below, on the
    scene-mean 1-s curves, the burst contrast is about +0.24 for LongLive's sink 3 and +0.34 for Self Forcing's.
- **LongLive's probe** (exploratory, lab note 30, `llprobe_out.json`):
  - set up as Self Forcing's second probe (openings rx02, rx03, gx02, gx03; distances 3–723);
  - all 17 major peaks of Self Forcing recur; the curves correlate at r = 0.964;
  - LongLive's shares are higher by about 0.02 throughout (median 0.095 vs 0.074).
- This study applies the interventions of preregistrations 25 and 27 to LongLive, to test whether the positional
  account holds in a second model.

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), generic prompt (`prompts/relax.jsonl`).
- **Model:** LongLive (base weights + LoRA), sink 3, window 12, 60 s (243 latent frames).
  - Generator `ll_hold_gen_sinkpos.py`: `ll_hold_gen_long.py` plus `--probe` and `--sink_mode`, and it records motion
    and open_diff in the npz (defined as in `hold_gen_sinkpos.py`).
  - Run in LongLive-c, a symlink tree of LongLive 1.0 in which only `wan/modules/causal_model.py` is a real file,
    patched by `jobs/patch_ll_probe.py` and then `jobs/patch_ll_sinkpos.py`.
  - The patch rotates only a copy of the sink keys (the cache keeps the original keys) and leaves the opening's own
    context pass unchanged, as Self Forcing's `patch_sf_sinkpos.py` does.
- **Consistency check** (done before freezing, see below): on one GPU, LongLive-c without the switch and with `off:0`
  is bit-identical to the original LongLive tree; with `dist:30` it differs.
- **Openings:** L\* shifts −16 / +16 (versions m16, p16).
- **Seed:** 0.
- **Arms (20):**
  - ll_abs: original positions (no switch);
  - ll_off18: sink 18 latent frames further back (`--sink_mode off:18`);
  - ll_off36: 36 latent frames further back (`off:36`);
  - ll_dDDD: sink held at distance DDD (`dist:DDD`), the 17 distances of preregistration 27: 20, 66, 90, 132, 135, 147,
    187, 203, 335, 345, 448, 466, 468, 513, 589, 643, 650.
- **LongLive's probe shares at these distances:**

| D | share | D | share | D | share |
|---|---|---|---|---|---|
| 20 | 0.124 | 147 | 0.105 | 466 | 0.134 |
| 66 | 0.121 | 187 | 0.077 | 468 | 0.144 |
| 90 | 0.131 | 203 | 0.182 | 513 | 0.104 |
| 132 | 0.130 | 335 | 0.167 | 589 | 0.063 |
| 135 | 0.167 | 345 | 0.080 | 643 | 0.150 |
| | | 448 | 0.088 | 650 | 0.068 |

- **Execution:**
  - one task per scene with all 20 arms and both versions (40 videos); arch_guard keeps all outputs of a scene on one
    GPU architecture;
  - an arm whose p16 output already exists is skipped, so an interrupted task can resume;
  - any GPU model (≥ 40 GB, bf16);
  - video files only for rx00 and gx00;
  - task list made by `make_s28_tasks.py` (`jobs/s28.jsonl`, 24 tasks).
  - In total 24 × 20 × 2 = 960 videos.

## Measures
- α[t0, t1) = (end_p16 − end_m16) / (open_p16 − open_m16):
  - open is the mean L\* of frames 1–8;
  - end is the mean L\* over seconds [t0, t1) after the opening;
  - undefined if open_p16 − open_m16 < 4 L\*.
- No collapse exclusion.
- **motion[t0, t1):** the per-frame motion (mean absolute grey-level difference from the previous frame, 0–100)
  averaged over seconds [t0, t1) and over the two versions.
- **share(D):** LongLive's probe's mean share at distance D.
- **Burst onsets** (as in preregistration 25):
  - τ_k(δ) = (P_k − 5 − δ)/4 s, P = 71, 134, 203, 266 (peaks of both models' probes);
  - burst window W_k = [τ_k + 0.5, τ_k + 2.5) s, kept only within [5, 60) s;
  - A_X(δ): arm X's mean α over all windows for δ (one value per scene).

## Hypotheses (`analyze_s28.py`, frozen; Holm correction over 4 tests)
- **L1 the bursts follow an 18-frame shift:** [A_off18(18) − A_off18(0)] − [A_abs(18) − A_abs(0)] > 0.
- **L2 the bursts follow a 36-frame shift:** [A_off36(36) − A_off36(0)] − [A_abs(36) − A_abs(0)] > 0.
- **L3 persistence rises with the share:** per scene, the Spearman ρ_s(share, α[50, 60)) across the 17 held distances
  (at least 12 defined); scene mean > 0.
- **L4 motion falls with the share:** per scene, ρ_s(share, motion[50, 60)); scene mean < 0.
- **Tests:**
  - two-sided one-sample t-tests over scenes;
  - supported if Holm p < 0.05 and the estimate has the hypothesised sign;
  - 95% CIs by scene bootstrap (10,000 draws).

## Secondary analyses
- L3 and L4 over [40, 50) s;
- each arm's burst contrast at its own onsets (ll_abs, ll_off18, ll_off36 at their own δ; held arms at δ = 0);
- times of 1-s jumps above 0.15;
- per arm:
  - α (10-s windows and the 1-s curve);
  - motion and open_diff over [50, 60) s;
  - the share of videos with motion < 1.5;
- across distances: Pearson and Spearman correlations of the scene-mean α and motion with LongLive's share and with
  Self Forcing's share;
- probe check (as in preregistration 26): the correlation between the first differences of the 1-s α curve and those
  of LongLive's probe share at t − lag, lag = 0–3 s, t = 6–59 s, for ll_abs, ll_off18 and ll_off36 at their own δ;
- when the results of preregistration 27 exist: the correlation across the 17 distances between LongLive's and Self
  Forcing's mean α and motion.

## Predictions
- **L1 and L2 supported.** LongLive's bursts are weaker than Self Forcing's (design check +0.24 vs +0.34); expected
  L1 about +0.3 and L2 about +0.15.
- **L3 and L4 supported:** mean ρ about +0.5 to +0.8 and −0.3 to −0.7.
  - Less certain than for Self Forcing: LongLive's shares are higher throughout, yet its sink at the original positions
    keeps less at 60 s than Self Forcing's (0.45 vs 0.61), so LongLive's persistence may be less sensitive to the share.
- **Rules for interpretation:**
  - if L1–L4 are all supported: the positional account (the bursts are set by the sink's rotary distance, and how
    much is kept by the attention share at that distance) holds in a second model, including the probe peaks that its
    fine-tuning left unchanged;
  - if L1 and L2 are supported and L3 is not: in LongLive too the timing of the bursts is set by the distance, but
    with the sink held at one distance, how much is kept is not set by the share (for example, LongLive's training may
    have made its reliance on the sink independent of position);
  - if L1 and L2 are not supported: LongLive's bursts come at the same times but cannot be moved by moving the sink;
    the paper's positional account is limited to Self Forcing.

## Consistency check (before freezing, 2026-10-28)
- `jobs/run_llsp_check.sh`: on one GPU (A40), rx00, L\* −16, sink 3, seed 0, 10 s, comparing per-frame mean L\*:
  - LongLive-c without the switch: bit-identical to the original LongLive tree (maximum difference 0.0);
  - with `off:0` (no rotation): bit-identical;
  - with `dist:30`: different (maximum difference 15.4 L\*), so the switch takes effect. Distance 30 is not among the
    distances of this study; these 10 s are used only for the check.

## Frozen files (SHA-256)
```
0938e2f5d25a3c345179ffacb61021932e4db626093e7c07f8a76f9ca0ffec58  analyze_s28.py
62eb515368cf774a98124831ced96ad10cc617e5a4ca81f7f9601e9d605f5360  make_s28_tasks.py
aa9b610bbb50c2b20e2103353baa66528e536de9f8fef1736730cba8465483f9  jobs/s28.jsonl
ea6a93de994e02811cfdc2cd406db6856fd84b6bc255ff2369b008837402f9ce  s27_distances.py
073f21f1476aea62a192de6aa71b00159c2af78a44bfa2304f025b7ed65821d7  ll_hold_gen_sinkpos.py
683048690f76f5845dfbf32b87354e89c6d7d9a579797b20f9f9aa30bda2be7d  jobs/patch_ll_sinkpos.py
f4e2146c9b4443fe68b187fca2c2ce058cab3ab99e45ca36e449d8dfd4b2b2f9  jobs/patch_ll_probe.py
4745cce792fdc31a0442245b9ff6596d06288bac08bf67a807e142d702d660a5  LongLive-c/wan/modules/causal_model.py (patched)
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
a84554132486503056181f94cad78b3e04c86108d920b5c4c51cee92b3c790eb  jobs/run_llsp_check.sh
89d1007ae1d6165b5027d0d681bdf54cf6a9fb7decd32ecbcdd37ee8db2c81f9  llprobe_out.json (exploratory probe result, LongLive)
64385a87e26b2110501e89abee7f07196cc093372da13b57f11f5df26cdde923  sinkprobe_long_out.json (exploratory probe result, Self Forcing)
```
