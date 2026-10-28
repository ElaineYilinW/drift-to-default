> English translation of the frozen Chinese preregistration [`../29_steady_share.md`](../29_steady_share.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 29: does an opening that stays in the context keep the same share from the second to the fourth minute? (Self Forcing, Rolling Forcing, 240 s, new seeds) (2026-10-28, frozen before generation)

## Background
- Preregistration 24 (lab note 33, seed 0, 180 s):
  - at 180 s the re-rotated sink keeps 0.12 [0.04, 0.20] and Rolling Forcing 0.29 [0.21, 0.37] (L1, L2 supported);
  - all three equivalence tests failed: the changes from 60 to 180 s were −0.09 [−0.22, +0.03] and −0.07
    [−0.16, +0.01], and no sink was +0.04 [−0.06, +0.14] at 180 s, all over 10-s windows.
- Exploration after the frozen analysis (60-s windows): the shares in the second and third minutes are almost the same:
  - re-rotated sink 0.17 and 0.16 (difference −0.01 [−0.09, +0.07]);
  - Rolling Forcing 0.30 and 0.30 (+0.00 [−0.06, +0.06]);
  - no sink 0.01 and 0.04.
- So the shares fall during the first minute and seem to stop falling afterwards. But these windows were chosen after
  seeing the data.
- This study tests that with new seeds and a longer horizon (240 s):
  - the 60-s windows are the prespecified measure;
  - the fourth minute tested, [180, 240) s, has never been generated before.

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), generic prompt (`prompts/relax.jsonl`).
- **Length:** 240 s (963 latent frames, 3849 frames, 16 fps).
  - Wan's rotary table has only 1024 temporal positions, so about 255 s is the limit.
- **Openings:** L\* shifts −16 / +16 (versions m16, p16).
- **Seeds:** 1, 2.
  - Preregistration 24 used seed 0.
  - Preregistration 20 generated 60-s videos with seeds 1–3. The noise tensors here have a different length and are
    not expected to coincide with those; in any case every tested window starts at 60 s.
- **Arms (the generators of preregistration 24, frozen, unchanged):**
  - sf240_s3rr: Self Forcing DMD, sink 3, re-rotated (`hold_gen_long_rr.py --sink 3 --rerotate`);
  - rf240: Rolling Forcing (`rf_gen_long.py`);
  - sf240_s0: Self Forcing DMD, no sink (`hold_gen_long_rr.py --sink 0`).
- **Execution:**
  - one task per scene × arm × seed, with both versions; arch_guard keeps a pair on one GPU architecture;
  - one output folder per seed (`s1/`, `s2/`), because arch_guard matches outputs by scene and arm only;
  - any GPU model (≥ 40 GB, bf16);
  - video files only for rx00 and gx00, seed 1;
  - task list made by `make_s29_tasks.py` (`jobs/s29.jsonl`, 144 tasks);
  - in total 24 × 3 × 2 × 2 = 288 videos.

## Measures
- α[t0, t1) = (end_p16 − end_m16) / (open_p16 − open_m16):
  - open is the mean L\* of frames 1–8;
  - end is the mean L\* over seconds [t0, t1) after the opening;
  - undefined if open_p16 − open_m16 < 4 L\*.
- A scene's value: the mean α over the seeds for which it is defined.
- No collapse exclusion (as in preregistration 24).

## Hypotheses (`analyze_s29.py`, frozen; Holm correction over 5 tests)
- **M1 the re-rotated sink still keeps a share in the fourth minute:** α(sf240_s3rr, [180, 240)) > 0.
- **M2 Rolling Forcing still keeps a share in the fourth minute:** α(rf240, [180, 240)) > 0.
- **M3 the re-rotated sink's share does not change:** α(sf240_s3rr, [180, 240)) − α(sf240_s3rr, [60, 120)) within ±0.10.
- **M4 Rolling Forcing's share does not change:** α(rf240, [180, 240)) − α(rf240, [60, 120)) within ±0.10.
- **M5 without a sink the opening stays forgotten:** α(sf240_s0, [60, 240)) within ±0.10.
- **Tests:**
  - M1, M2: two-sided one-sample t-tests over scenes; supported if Holm p < 0.05 and the estimate > 0;
  - M3–M5: equivalence tests (TOST), p = the larger of the two one-sided p values; supported if Holm p < 0.05;
  - 95% CIs by scene bootstrap (10,000 draws); 90% t intervals also reported for the equivalence tests.
- **Margin:** ±0.10, as in preregistrations 23 and 24.
- **Power** (estimated from the 60-s windows of preregistration 24):
  - the per-scene SD of the difference at seed 0 is about 0.19 (re-rotated sink) and 0.15 (Rolling Forcing);
  - if averaging two seeds brings these to about 0.14 and 0.10, then at an uncorrected 0.05 level M3 and M4 are
    supported when the observed mean change is within ±0.05 and ±0.065;
  - if the true change is −0.01 (the exploratory estimate of preregistration 24): the p values of M3 and M4 are
    expected to rank last of the five (those of M1, M2, M5 should be very small), which puts their Holm levels at
    0.025–0.05; at 0.025 the power is about 0.85 for M3 and 0.99 for M4.

## Secondary analyses
- M1–M5 per seed;
- per arm, α over the 60-s windows [0, 60), [60, 120), [120, 180), [180, 240) s;
- per scene, the least-squares slope of the 10-s α from 70 to 240 s (per 100 s);
- an exponential fit α(t) = α∞ + (α0 − α∞)·exp(−t/τ) to each arm's mean 10-s curve, with a scene-bootstrap interval of
  α∞ (2,000 draws);
- α at T = 10, 20, …, 240 s (10-s windows) and 1-s curves;
- the five tests of preregistration 24 redone on these two seeds (180 s, 10-s windows);
- seed agreement: the Pearson correlation across scenes of α[60, 240) between seeds 1 and 2;
- collapse counts (videos with a 1-s mean L\* below 3 or above 97).

## Predictions
- **M1, M2 supported:** about 0.16 and 0.30.
- **M3, M4 supported:** changes expected within ±0.05.
  - Uncertainty: the fourth minute has never been looked at; if the shares are still falling slowly (for example
    0.02–0.03 per minute), M3 and M4 may fail.
- **M5 supported:** about 0.02.
- **Rules for interpretation:**
  - if M1–M4 are all supported: an opening that stays in the context keeps a share that no longer changes after the
    first minute (at least to 240 s). The paper can say "keeps a steady share", stating the horizon.
  - if M1 and M2 are supported but M3 or M4 is not:
    - estimate negative and beyond the margin: the share is still falling, and the paper says "forgotten more slowly";
    - estimate near 0 but the interval too wide: inconclusive, and the paper says only "still keeps a share at 240 s".
  - if M5 is not supported with an estimate > 0.10: without a sink the opening is not fully forgotten, and "forgotten
    once it leaves the context" must be rewritten.

## Frozen files (SHA-256)
```
2d5c8b6638e1e846c81957bf0b964daa8cc86a73290145a49b554481027a67fb  analyze_s29.py
3744fd18aed51e0feea6329aa82af6fb171c807f7ea9a3c6a53a04e2b0a0cf31  make_s29_tasks.py
a64acb8bd680516d83bbbb8019a756ca5ae7aa2bed675dfe923295adcb1625b9  jobs/s29.jsonl
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
9b58613c2ed1f06862b9f9b5290e5bfd9a52a9c6cc147fc58409d26628bb2599  rf_gen_long.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
- The analysis script was checked on synthetic data (known shares 0.16, 0.30 and 0 are recovered; with missing files
  the number of scenes drops correctly).
