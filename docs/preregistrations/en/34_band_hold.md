# Preregistration 34: are the bursts set by one or two temporal frequencies of the sink? (only some of the sink's RoPE frequencies held during generation, Self Forcing, new seed, 60 s) (2026-10-29, frozen before generation)

*English translation; the Chinese file `../34_band_hold.md` is the frozen, authoritative version.*

## Background
- A sink at its original positions bursts at 16.5, 32.25 and 49.5 s (studies 25–28); these times correspond to the attention probe's peaks at distances 71, 134 and 203.
- A band probe (lab note 42, exploratory) split the peaks over the 22 temporal RoPE frequencies:
  - **j4 (period 33.5 latent frames) is necessary:** with only j4 left at the reference distance 20 and the other frequencies rotated to the grid distance, all main peaks disappear;
  - **j3 (period 22.1) removes every other peak:** with only j3 removed, all peaks with a period of about 34 frames appear (33, 103, 134, 165, 203, 234);
  - **the long-period frequencies (j8–11 and others) do not affect the peaks:** the curve is almost unchanged (r = 0.99).
- But the probe is only a readout of "how much attention the queries would give the sink if it were at that distance"; it does not change generation. This study makes the same change during generation and tests whether the probe's band decomposition causally predicts the bursts.
- **Data status:**
  - No video of this study had been generated when it was frozen.
  - A smoke test was run only on scene rx00, seed 99, 15 s (`jobs/s34_smoke.jsonl`). It checked that "all 22 frequencies held at 20" matches `dist:20` and that the new code runs; it did not look at bursts or persistence.

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), generic prompt (`prompts/relax.jsonl`).
- **Model:** Self Forcing DMD, sink 3, window 21, 60 s (243 latent frames, 16 fps).
- **Openings:** L\* shifted by −16 / +16 (versions m16, p16).
- **Seed:** 13 (not used on these scenes before).
- **Generator:**
  - `hold_gen_bandhold.py` (a copy of `hold_gen_sinkpos.py` with only the checked code tree changed), run in Self-Forcing-d (`jobs/setup_sf_d.sh`, `jobs/patch_sf_bandhold.py`);
  - new `--sink_mode band:D:j1,j2,...`: for each chunk after the opening, only on frequencies j1, j2, ... is the copy of the sink keys rotated to query–sink distance D; the other frequencies stay at the original positions. With all 22 frequencies this is `dist:D`.
- **Arms:**
  - bh_abs: sink at its original positions (no switch);
  - bh_j4: j4 held at distance 20 (`--sink_mode band:20:4`);
  - bh_j3: j3 held at distance 20 (`--sink_mode band:20:3`);
  - bh_j811: j8–j11 held at distance 20 (`--sink_mode band:20:8,9,10,11`), control band;
  - bh_rr: re-rotated sink (`--rerotate`, all frequencies at about distance 20), as a reference.
- **Execution:**
  - One task per scene × arm (both versions); arch_guard keeps a pair of versions on one GPU type.
  - Any GPU with ≥ 40 GB and bf16.
  - Videos are written only for rx00 and gx00.
  - Task list `make_s34_tasks.py` (`jobs/s34.jsonl`, 120 tasks); 24 × 5 × 2 = 240 videos.

## Metric
- **α[t0, t1)** = (end_p16 − end_m16) / (open_p16 − open_m16):
  - open is the mean L\* of frames 1–8;
  - end is the mean L\* over seconds [t0, t1) after the opening;
  - undefined if open_p16 − open_m16 < 4 L\*;
  - no collapse exclusion.
- **Onsets** (as in study 25: the probe's peak at distance P is reached (P − 5)/4 s after the opening):
  - ORIG = 16.5, 32.25, 49.5 s (the first probe's peaks 71, 134, 203);
  - NEW = 7.0, 24.5, 40.0, 57.25 s (peaks 33, 103, 165, 234 of the band probe's "j3 removed" curve that are not peaks of the full probe; detection rule as in `analyze_sinkprobe_long.py`, prominence ≥ 0.04).
- **Burst contrast C_arm(onsets):** per scene, the mean over onsets τ of α([τ + 0.5, τ + 2.5)) − α([τ − 2.5, τ − 0.5)).

## Hypotheses (`analyze_s34.py`, frozen; Holm over 4 tests)
- **B1, holding j4 removes the bursts:** C_abs(ORIG) − C_j4(ORIG) > 0.
- **B2, holding j8–11 does not affect the bursts:** C_j811(ORIG) − C_abs(ORIG) within ±0.10.
- **B3, holding j3 produces bursts at the new times:** C_j3(NEW) − C_abs(NEW) > 0.
- **B4, less is kept without bursts:** α(bh_abs, [50, 60)) − α(bh_j4, [50, 60)) > 0.
- **Tests:**
  - B1, B3, B4: two-sided one-sample t-test over scenes; supported if Holm p < 0.05 and the estimate > 0;
  - B2: equivalence test (TOST), p is the larger of the two one-sided p-values; supported if Holm p < 0.05; the 90% t interval is also reported;
  - 95% CIs by scene bootstrap (10,000 draws).
- **Power:** in study 25 the burst contrast of the sink at its original positions was +0.33 [0.29, 0.38], and that of sinks held at one distance lay between −0.05 and −0.02. With a difference of about 0.3 and a between-scene SD of about 0.1, the power with 24 scenes is close to 1. B2's bound of 0.10 is about a third of the burst size.

## Secondary analyses
- Per arm: C(ORIG), C(NEW) and the contrast at each onset.
- C_j3(16.5 s) − C_abs(16.5 s): the burst that j3 should remove.
- Per arm: α in 10-s windows and 1-s curves; α over [50, 60) s of each arm against bh_abs and bh_rr.
- Times of 1-s jumps above 0.15.
- Per arm, the correlation (t = 6–59 s) between the first differences of the 1-s α curve and the first differences of the band probe's share at the sink's natural distance ("all" for bh_abs, "j4 removed", "j3 removed", "j8–11 removed").
- Motion over [50, 60) s and the share of videos with motion < 1.5.

## Predictions
- **B1 holds:** C_j4(ORIG) about 0 (−0.05 to +0.05), C_abs(ORIG) about +0.3, difference about +0.3.
- **B2 holds:** difference about 0 (±0.05).
- **B3 holds:** C_j3(NEW) about +0.15 to +0.25, C_abs(NEW) about 0 or slightly negative, difference about +0.2. The probe share at the new peaks (0.115–0.124) is close to that of the peak at 71 (0.121) and below those at 134 and 203 (0.15–0.17), so the new bursts may be smaller than the original ones.
- **B4 holds:** α(bh_abs, [50, 60)) about 0.6, α(bh_j4) about 0.3–0.45, difference about +0.15 to +0.3. The median share of the "j4 removed" curve (0.067) is slightly below that of all frequencies (0.079), and in study 27 a sink held at a distance with a share of about 0.07 kept about 0.35.
- **Interpretation rules:**
  - If B1, B2 and B3 all hold: the bursts are set by the sink's phase on two temporal frequencies (periods 22 and 34 latent frames). The paper will say "removing the rotation of one frequency removes the bursts, removing that of another moves the bursts to the times the probe predicts, and removing that of four other frequencies has no effect", as a frequency-level causal mechanism.
  - If B1 holds and B3 does not: j4 is necessary, but the probe's finer prediction (the new times) does not appear in generation. The paper reports only B1.
  - If B1 does not hold: the probe's band decomposition does not transfer to generation; the paper reports the band probe in the appendix and states that it was not confirmed in generation.
  - If B2 does not hold while B1 does: other bands also play a role; this is reported as found.
  - B4 shows how much of the persistence of a sink at its original positions comes from the bursts; the estimate is reported whatever the result.

## Frozen files (SHA-256)
```
e87cd803efcfe84a5bc2b526ecb7e545828dc83cf42d2e838393928c4f0cafa3  analyze_s34.py
7bfe75c7451a168d3d743773746b8244290bce8c0d2b2e81a402a2808a7c3768  make_s34_tasks.py
6fdf8ed91e26c66ff1afe5870eade3bb9432e5b1c6828a1c44582cf9e3c9b267  jobs/s34.jsonl
6e9829a8b145d2dedcb81e8fdccb44404a7ed23de9608d1433633d1fa7d9076b  hold_gen_bandhold.py
dfd6acdf83bed9c0398afde6fe44f80595814eef5f1be8f9db5c0f67feacb707  jobs/patch_sf_bandhold.py
fae9f1026b14c71825e138a33b65259e586f03922726aa461b6b07d886944665  jobs/setup_sf_d.sh
e6bb3c7f0059eec21d692a591e7fc594c986c70549743e43ef31f41c4861b858  Self-Forcing-d/wan/modules/causal_model.py (patched)
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
fe8e39f6265453d29f81c619a5b760607fc7a22c1cc40a3c677dacda846cdcbb  results/corr/sinkband_out/sinkband.json (onsets NEW)
55bb04d85f33fa87aa3ec048d14c6a29cbed79016619c05b37f29640c9952343  jobs/s34_smoke.jsonl (smoke test only)
```
- The analysis script was checked on synthetic data: known bursts (+0.30 at the original times, +0.20 at the new times, no burst) and a persistence difference of 0.18 are recovered, and the equivalence test is correct.
- Smoke test (rx00, seed 99, 15 s, m16 only): `band:20:` with all 22 frequencies gives per-frame L\* identical to `dist:20` (maximum difference 0.0); `band:20:4` departs from `dist:20` from frame 9 (the first chunk after the opening) on, which shows that the change acts only after the opening.
