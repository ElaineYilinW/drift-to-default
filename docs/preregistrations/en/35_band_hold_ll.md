# Preregistration 35: does holding only the sink's j4 also remove the bursts in LongLive? What happens with the neighbouring single frequency j5 instead? (LongLive and Self Forcing, new seed, 60 s) (2026-10-29, frozen before generation)

*English translation; the Chinese file `../35_band_hold_ll.md` is the frozen, authoritative version.*

## Background
- Study 34 (lab note 44): in Self Forcing, holding only the sink's 4th temporal frequency (period 33.5 latent frames) at distance 20 during generation, with the other 21 frequencies at their original positions:
  - the bursts disappear (+0.33 [+0.26, +0.39]);
  - the share kept at 60 s falls from 0.58 to 0.16.
- Study 34 leaves two questions:
  - **One model only:** LongLive's bursts also follow the sink's position (study 28), and the probe shapes of the two models are almost the same (r = 0.96). If the bursts come from j4, holding j4 in LongLive should also remove the bursts.
  - **Specificity:** study 34's control band was j8–11 (4 long-period frequencies), which made the bursts slightly smaller (−0.06). The claim "only j4" needs a comparison with a control that also holds a single frequency. On the band probe, removing the neighbouring j5 (period 51.0) hardly changes the curve (contrast 0.055 against 0.058 for all frequencies, correlation ≥ 0.99, same main peaks).
- **Data status:**
  - No video of this study had been generated when it was frozen; seed 21 has not been used on these scenes before.
  - A smoke test was run only on scene rx00, seed 99, 15 s, m16 only (`jobs/s35_smoke.jsonl`). It checked that in LongLive-d "all 22 frequencies held at 20" matches `dist:20` frame by frame and that holding only j4 departs from it starting after the opening; it did not look at bursts or persistence.

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), generic prompt (`prompts/relax.jsonl`).
- **Models:** sink 3, 60 s (243 latent frames, 16 fps).
  - LongLive: window 12.
  - Self Forcing DMD: window 21.
- **Openings:** L\* shifted by −16 / +16 (versions m16, p16).
- **Seed:** 21.
- **Generators:**
  - LongLive: `ll_hold_gen_bandhold.py` (a copy of `ll_hold_gen_sinkpos.py` with only the code-tree check added), run in LongLive-d. LongLive-d is set up by `jobs/setup_ll_d.sh`; the patch is `jobs/patch_ll_bandhold.py`, the same as Self Forcing's `patch_sf_bandhold.py`.
  - Self Forcing: `hold_gen_bandhold.py`, run in Self-Forcing-d (as in study 34).
- **Arms (3 per model):**
  - abs: sink at its original positions;
  - j4: j4 held at distance 20 (`--sink_mode band:20:4`);
  - j5: j5 held at distance 20 (`--sink_mode band:20:5`).
  - Arm names: ll_abs, ll_j4, ll_j5, sf_abs, sf_j4, sf_j5.
- **Execution:**
  - One task per scene × arm (both versions); arch_guard keeps a pair of versions on one GPU type.
  - Any GPU with ≥ 40 GB and bf16.
  - Videos are written only for rx00 and gx00.
  - Task list `make_s35_tasks.py` (`jobs/s35.jsonl`, 144 tasks); 24 × 6 × 2 = 288 videos.

## Metric
- **α[t0, t1)**: as in study 34.
  - α = (end_p16 − end_m16) / (open_p16 − open_m16);
  - open is the mean L\* of frames 1–8, end is the mean L\* over seconds [t0, t1) after the opening;
  - undefined if open_p16 − open_m16 < 4 L\*;
  - no collapse exclusion.
- **Onsets:** ORIG = 16.5, 32.25, 49.5 s, as in studies 25, 28 and 34.
- **Burst contrast C_arm:** per scene, the mean over onsets τ of α([τ + 0.5, τ + 2.5)) − α([τ − 2.5, τ − 0.5)).

## Hypotheses (`analyze_s35.py`, frozen; Holm over 4 tests)
- **C1, holding j4 removes the bursts in LongLive:** C_ll_abs − C_ll_j4 > 0.
- **C2, LongLive keeps less without bursts:** α(ll_abs, [50, 60)) − α(ll_j4, [50, 60)) > 0.
- **C3, in Self Forcing not just any frequency will do:** C_sf_j5 − C_sf_j4 > 0.
- **C4, the same in LongLive:** C_ll_j5 − C_ll_j4 > 0.
- **Tests:**
  - two-sided one-sample t-test over scenes; supported if Holm p < 0.05 and the estimate > 0;
  - 95% CIs by scene bootstrap (10,000 draws).
- **Power:** the expected effects are of the same size as study 34's +0.33.
  - In study 28, LongLive's burst contrast at the original positions was +0.27 [+0.23, +0.31], and that of sinks held at one distance lay between −0.02 and +0.02.
  - With a difference of about 0.25 and a between-scene SD of about 0.1, the power with 24 scenes is close to 1.

## Secondary analyses
- **Study 34 repeated in Self Forcing:** C_sf_abs − C_sf_j4; α(sf_abs) − α(sf_j4) (over [50, 60) s).
- **j5 against abs:** C_j5 − C_abs per model, with the 90% t interval and an equivalence test at ±0.10.
- **α over [50, 60) s:** each arm against its own model's abs.
- **Burst details:** C and the contrast at each onset per arm; α in 10-s windows and 1-s curves; times of 1-s jumps above 0.15.
- **Correlation with the band probe:** per arm, the correlation (t = 7–59 s) between the first differences of the 1-s α curve and the first differences of the band probe's share at the sink's natural distance 1 s earlier.
  - Curves matched: "all" for abs, "j4 removed" for j4, "j5 removed" for j5.
  - Lag of 1 s: in study 28 and in study 34 (exploratory) the correlation was strongest at a 1-s lag.
- **Motion:** motion over [50, 60) s and the share of videos with motion < 1.5.

## Predictions
- **C1 holds:** C_ll_abs about +0.25 (study 28: +0.27), C_ll_j4 about 0, difference about +0.25.
- **C2 holds:** α(ll_abs, [50, 60)) about 0.45 (study 28's ll_abs), α(ll_j4) about 0.1–0.2, difference about +0.25 to +0.35.
- **C3 holds:** C_sf_j5 about +0.25 to +0.3 (close to abs), C_sf_j4 about 0, difference about +0.3.
- **C4 holds:** difference about +0.25.
- **Predictions for the secondary analyses:**
  - the difference between j5 and abs lies within ±0.10;
  - study 34's results are repeated in Self Forcing.
- **Interpretation rules:**
  - **C1, C2, C3 and C4 all hold:** in both models, the bursts and the extra share kept by the sink come from one temporal frequency of the sink, and the neighbouring frequency has no such effect. The paper will say "in two models, holding only one rotary frequency of the sink removes the bursts and most of what is kept; holding the neighbouring frequency does not".
  - **C1 and C2 hold but C3 or C4 does not:** j4 is necessary, but holding another single frequency also weakens the bursts. The paper says only that j4 is necessary, not "only j4".
  - **C1 does not hold:** the effect of j4 is shown only in Self Forcing; this is reported as found.
  - **C2:** the estimate is reported whatever the result.

## Frozen files (SHA-256)
```
c459eabab1849a97b031a8fa62bdc1d629e1460517e6b67e3b460c63e6bada20  analyze_s35.py
ed7afb93c56238d1c40a3f88444722cb0ab584ea75c7c0db48a77bade4ed2065  make_s35_tasks.py
f2d5cb4fdfe5bcd2fdae855cbc1ce6623e8763c8f16f7e8f13821fe8628c7523  jobs/s35.jsonl
9d722b0050f27b9ac382dda7e03266855822f9b8b6665b4c1075cdc164d60eb8  ll_hold_gen_bandhold.py
4eab5a269a3bdd683433e072c63aca147162ada5967cea2fc59e71b1ec5ae33f  jobs/patch_ll_bandhold.py
0f8f120879457b563e1779964d793bc4f8e166af4077c5a0fff14e4bae004007  jobs/setup_ll_d.sh
8bfac6050ca2c19c767851a254a00b0cb4233b4d4b3a02deed03a2abc960f965  LongLive-d/wan/modules/causal_model.py (patched)
6e9829a8b145d2dedcb81e8fdccb44404a7ed23de9608d1433633d1fa7d9076b  hold_gen_bandhold.py
dfd6acdf83bed9c0398afde6fe44f80595814eef5f1be8f9db5c0f67feacb707  jobs/patch_sf_bandhold.py
e6bb3c7f0059eec21d692a591e7fc594c986c70549743e43ef31f41c4861b858  Self-Forcing-d/wan/modules/causal_model.py (patched)
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
fe8e39f6265453d29f81c619a5b760607fc7a22c1cc40a3c677dacda846cdcbb  results/corr/sinkband_out/sinkband.json
94efb91ff9bc4f4e3a7724338cb54fc3676dbd20ae65f9079dd273cff02608e0  jobs/s35_smoke.jsonl (smoke test only)
```
- The analysis script was checked on synthetic data: known bursts (abs and j5 +0.30 at two onsets, no burst for j4) and persistence differences are recovered (C1 +0.20, C2 +0.35, C3 and C4 recover the set values).
- Smoke test (LongLive-d, rx00, seed 99, 15 s, m16 only): `band:20:` with all 22 frequencies gives per-frame L\* identical to `dist:20` (maximum difference 0.0); `band:20:4` departs from both `dist:20` and the original positions from frame 9 (the first chunk after the opening) on, which shows that the change acts only after the opening.
