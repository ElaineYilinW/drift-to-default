> English translation of the frozen Chinese preregistration [`../24_long_horizon.md`](../24_long_horizon.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 24: does an opening that stays in the context keep a steady share over 180 s? (Self Forcing, Rolling Forcing, LongLive, 180 s) (2026-10-28, frozen before generation)

## Background
- The paper sorts models into two classes:
  - once the opening leaves the context (only in a sliding window, or passed forward by overlapping frames), it is
    forgotten;
  - an opening that stays in the context (a re-rotated sink, Rolling Forcing's anchor, FramePack's prefix, MAGI-1's full
    history) keeps a share.
- For the second class this "keeping" has only been seen up to 60 s, and the curves are still falling:
  - Self Forcing's re-rotated sink 3 gives 0.30, 0.26, 0.24, 0.23 at 30, 40, 50, 60 s (preregistration 16);
  - Rolling Forcing falls from 0.70 to 0.42 between 10 and 60 s (preregistration 10).
- A review pointed out that "keeps a steady share" is a claim about an asymptote, and that 60 s of data cannot tell a
  steady share from slower forgetting.
- This study runs the same 24 scenes to 180 s and tests three things:
  - whether these shares are still above 0 at 180 s;
  - whether they stay unchanged from 60 to 180 s;
  - whether the opening stays forgotten without a sink.

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), generic prompt (`prompts/relax.jsonl`).
- **Length:** 180 s (723 latent frames, 2889 frames, 16 fps).
- **Openings:** L\* shifted by −16 / +16 (versions m16, p16), as in preregistrations 06 and 10.
- **Seed:** 0.
- **Arms:**
  - sf180_s0: Self Forcing DMD, no sink (`hold_gen_long_rr.py --sink 0`);
  - sf180_s3rr: Self Forcing DMD, sink 3, re-rotated (`--sink 3 --rerotate`);
  - sf180_s3: Self Forcing DMD, sink 3 at its original positions (`--sink 3`);
  - rf180: Rolling Forcing (anchor at re-aligned positions; `rf_gen_long.py`);
  - ll180_s3: LongLive, sink 3 (`ll_hold_gen_long.py --sink 3`).
- **Fresh starts:** 4 seeds (0–3) per arm without an opening (`--fresh`), for the attractor offset (secondary).
- **Execution:**
  - one task per scene × arm, holding both versions; arch_guard keeps a scene's pair on one GPU architecture;
  - any GPU with ≥ 40 GB and bf16;
  - video files are written only for rx00 and gx00; all statistics are computed from the frames;
  - the task list is built by `make_s24_tasks.py` (`jobs/s24.jsonl`, 125 tasks);
  - 24 × 5 × 2 + 20 = 260 videos in total.

## Measures
- open: mean L\* of frames 1–8.
- end(T): mean L\* over seconds [T − 10, T) after the opening.
- α(T) = (end_p16 − end_m16) / (open_p16 − open_m16); undefined if open_p16 − open_m16 < 4 L\*.
- No collapse exclusion (as in preregistrations 10, 20, 23).

## Hypotheses (`analyze_s24.py`, frozen; Holm correction over 5 tests; T = 180 s)
- **L1 a re-rotated sink still keeps a share at 180 s:** α(sf180_s3rr, 180) > 0.
- **L2 Rolling Forcing still keeps a share at 180 s:** α(rf180, 180) > 0.
- **L3 the re-rotated sink's share is steady:** α(sf180_s3rr, 180) − α(sf180_s3rr, 60) within ±0.10.
- **L4 Rolling Forcing's share is steady:** α(rf180, 180) − α(rf180, 60) within ±0.10.
- **L5 without a sink the opening stays forgotten:** α(sf180_s0, 180) within ±0.10.
- **Tests:**
  - L1, L2: two-sided one-sample t-tests on per-scene values; supported if Holm p < 0.05 and the estimate is on the
    predicted side;
  - L3–L5: equivalence by two one-sided t-tests against the margins (TOST), p = the larger one; supported if Holm
    p < 0.05;
  - 95% CIs by scene bootstrap (10,000 draws); 90% t intervals reported for the equivalence tests.
- **Margin:** 0.10, the same "fully forgotten" definition as in preregistration 23, about half the share a re-rotated
  sink keeps at 60 s.

## Secondary analyses
- L1–L5 with the window [160, 170) s;
- α per arm at T = 10, 20, …, 180 s (10-s windows) and 1-s curves;
- a fit of α(t) = α∞ + (α0 − α∞)·exp(−t/τ) to each arm's mean 10-s curve (T ≥ 10), with a scene-bootstrap interval
  for α∞ (2,000 draws);
- α(sf180_s3, 180) − α(sf180_s3rr, 180), and α(ll180_s3, 180);
- the times of 1-s jumps above 0.15 for the sinks at their original positions (sf180_s3, ll180_s3): do the bursts go
  on after 60 s, compared with the positional prediction of preregistration 25;
- collapse counts (videos with a 1-s mean L\* below 3 or above 97);
- attractor offset of the fresh starts: end(T) − mean L\* of the first second, T = 60, 120, 180.

## Predictions
- **L1 supported:** at 60 s the re-rotated sink keeps about 0.2 and its curve is flattening.
- **L2 supported:** Rolling Forcing gives 0.42 at 60 s; at its current rate of decline it is still above 0 at 180 s.
- **L3:** leaning supported (an exponential fit extrapolates a fall of about 0.03 from 60 to 180 s), but uncertain.
- **L4:** leaning not supported: before 60 s Rolling Forcing falls by about 0.04 every 10 s.
- **L5 supported.**
- **Rules for interpretation:**
  - if L3 is not supported and the share keeps falling, the paper's "an opening that stays in the context keeps a
    steady share" becomes "is forgotten more slowly", and the two classes differ in rate;
  - if L1 is not supported, a re-rotated sink is no different from no sink over long horizons.

## Frozen files (SHA-256)
```
0912cdde0d78423c1a32b945b44dcb926fe7b5643a8f1bcfa946402e501e3011  analyze_s24.py
6098e5df140d097a588ec3e7c20ebf879566b4ad784cfe0e2c6e59f1f79dc43c  make_s24_tasks.py
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
9b58613c2ed1f06862b9f9b5290e5bfd9a52a9c6cc147fc58409d26628bb2599  rf_gen_long.py
8723343defd6974f6d6f8a7c7e3d971272451882bc15952a4c70427d959a6a7e  ll_hold_gen_long.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
6a6333293db9ffd54934bb6d07e88e7a7f9e7e90520b04cc191820cb3dbff155  jobs/s24.jsonl
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
