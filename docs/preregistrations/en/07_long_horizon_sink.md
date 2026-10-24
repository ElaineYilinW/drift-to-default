> English translation of the frozen Chinese preregistration [`../07_long_horizon_sink.md`](../07_long_horizon_sink.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 07: Does the sink keep the opening at 60 s? (2026-10-23, frozen before generation)

## Background
- H1 of study 06 (on Self Forcing, brightness persistence at 31 s is lower without a sink than with one) was not
  supported: the estimated difference was −0.056 [−0.17, +0.05].
- The 120-s subset suggested that the problem is the time scale. Without a sink, Self Forcing forgets slowly: at 60 s
  the sink keeps about 0.5 and no-sink about 0 (difference +0.64 [0.33, 0.87]). That subset has only 6 scenes and was a
  secondary analysis.
- The chunk-recursion model fitted to the 120-s curves gives α∞ = 0.42 [0.25, 0.54].
- This study tests the sink effect confirmatorily at 60 s on all 24 scenes and adds a third model, CausVid (α = 0.15
  with a sink in study 06).

## Design
- **Scenes and prompt:** as in study 06: 24 scenes, generic prompt, seed 0.
- **Opening versions:** L\* −16, 0, +16.
- **Configurations:** Self Forcing DMD and CausVid, each with sink 3 and without a sink, window 21.
- **Length:** 60 s (243 latent frames).
- **Execution:** each scene × model in one task, with arch_guard at the start so that paired arms share a GPU
  architecture. Generator `hold_gen_long.py` (segment-wise decoding), task list `jobs/long60.jsonl`. 288 videos.

## Measure and hypotheses (`analyze_long.py`, frozen)
- α(T): within-scene OLS slope of the mean L\* over seconds [T−10, T) after the opening on the opening's L\*.
  Primary endpoint T = 60 s.
- **Primary hypotheses (Holm over 2 tests):**
  - H1: α(sf60_s3) > α(sf60_s0)
  - H2: α(cv60_s3) > α(cv60_s0)
- **Tests:** per-scene difference between the arms, two-sided one-sample t-test, scene-bootstrap 95% CI. Supported if
  Holm p < 0.05 and the estimate > 0.
- **Secondary:**
  - α of each configuration at 20, 30, 40, 50 and 60 s.
  - Quantitative prediction: α(sf60_s3, 60 s) lies in [0.25, 0.54], and |α(sf60_s0, 60 s)| < 0.15.

## Frozen files (SHA-256)
```
3ea2210feda4d9fc20db1931273a7d93bd7affe78c923c7a3e76629763b5c61d  analyze_long.py
d468456274cfc794150a3f8252b28b5c261568f95d0451c14307bcaa44a11d64  hold_gen_long.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
1c97911c6858cee4973e47ffd6b3047a8d0e15f3fe1b499534aa1979c68e1034  jobs/long60.jsonl
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
