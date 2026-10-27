> English translation of the frozen Chinese preregistration [`../20_seed_replication.md`](../20_seed_replication.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 20: do the Self Forcing sink results hold with new seeds? (2026-10-27, frozen before generation)

## Background
- Every Self Forcing sink result in the paper uses one seed per scene (seed 0):
  - preregistrations 07 and 10: sink 3 keeps about 0.6 at 60 s, no sink about 0.05;
  - preregistration 10: re-rotating the sink lowers this to 0.16;
  - preregistration 16: re-rotated sinks of 3 and 6 frames both keep about 0.22.
- Two points are untested:
  - **Seeds:** are the conclusions specific to seed 0? How much of the variation is between scenes and how much
    between seeds?
  - **Does a re-rotated sink still keep tone?** The paper now states that a sink at in-distribution positions keeps
    about 0.2 of tone, but neither preregistration 10 nor 16 compared a re-rotated sink with no sink directly.
- This study repeats the three arms with three new seeds (1, 2, 3).

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), openings L\* −16 / +16, generic prompt.
- **Model:** Self Forcing DMD, 60 s (243 latent frames), `hold_gen_long_rr.py` (as in preregistrations 10 and 16).
- **Arms:**
  - sf60_s3: sink 3 at its original positions;
  - sf60_s0: no sink;
  - sf60_s3rr: sink 3 re-rotated.
- **Seeds:** 1, 2, 3.
- **Execution:**
  - one task per scene with all arms, seeds and versions (arch_guard checks that all of the scene's outputs come from
    one architecture);
  - any GPU type (≥ 40 GB, bf16);
  - task list from `make_s20_tasks.py` (`jobs/s20.jsonl`).
  - 24 × 3 × 3 × 2 = 432 videos.

## Measures
- For each scene, seed and arm:
  - α(T) = (end_p16 − end_m16) / (open_p16 − open_m16);
  - open is the mean L\* of frames 1–8, end the mean L\* over [T − 10, T) s after the opening (16 fps).
- A scene's value: the mean of α over its seeds.
- No collapse exclusion (as in preregistration 10).

## Hypotheses (`analyze_s20.py`, frozen; Holm over 3 tests; T = 60 s)
- **H1 a sink keeps a share:** α(sf60_s3) − α(sf60_s0) > 0.
- **H2 re-rotation lowers it:** α(sf60_s3rr) − α(sf60_s3) < 0.
- **H3 a re-rotated sink keeps more than no sink:** α(sf60_s3rr) − α(sf60_s0) > 0.
- **Tests:**
  - two-sided one-sample t-tests over scenes;
  - supported if Holm p < 0.05 and the estimate is on the predicted side;
  - 95% CIs by scene bootstrap (10,000 draws).

## Secondary analyses
- H1–H3 in the window [40, 50) s;
- α at 10, 20, 30, 40, 50, 60 s per arm;
- each seed's H1–H3 estimate;
- variance of α at 60 s split into between-scene and between-seed parts (one-way random-effects ICC);
- the jump α([34, 35)) − α([31, 32)) per arm;
- seed 0 of the same scenes and arms from preregistrations 10 and 16, side by side (descriptive).

## Predictions
- **H1 supported:** about +0.5.
- **H2 supported:** about −0.4.
- **H3 supported:** about +0.15 (re-rotated sink about 0.2, no sink about 0.05).
- **Jump:** the sink at its original positions jumps between 31 and 34 s with all three seeds; the re-rotated sink and
  no sink do not.

## Frozen files (SHA-256)
See the Chinese file.
