> English translation of the frozen Chinese preregistration [`../21_content_seeds.md`](../21_content_seeds.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 21: do the Self Forcing content results hold with new seeds? (2026-10-27, frozen before generation)

## Background
- Every sink result on scene content (snow, desert, forest, ...) in the paper uses one seed per opening (seed 0):
  - preregistration 09: sink 3 keeps about 0.6 of the content at 60 s, no sink about 0.14;
  - preregistration 16 A:
    - a re-rotated sink drops to about 0.25;
    - the content "return" at 50–55 s appears only with the sink at its original positions;
    - re-rotated sink minus no sink is +0.10 (95% CI [−0.00, 0.21]), not significant.
- Preregistration 20 replicated the sink results for tone and found that per-video values vary mostly between seeds,
  so the content results need a multi-seed test too.
- This study repeats the three arms of preregistration 16 A with three new seeds (1, 2, 3). Each opening's value is
  the mean of three seeds, which lowers the noise and gives a stronger test of whether a re-rotated sink keeps content.

## Design
- **Openings:** the 24 openings of preregistration 09 (6 classes × 4), generic city prompt (arm mis).
- **Model:** Self Forcing DMD, 60 s (243 latent frames), `hold_gen_long_rr.py` (as in preregistration 16).
- **Arms:**
  - sf60_s3: sink 3 at its original positions;
  - sf60_s3rr: sink 3 re-rotated;
  - sf60_s0: no sink.
- **Seeds:** 1, 2, 3.
- **Default:** 8 new fresh starts of the generic prompt per arm (seeds 8–15) to estimate μ (earlier fresh videos are
  not reused).
- **Execution:**
  - one task per opening with all three arms and seeds (arch_guard checks that all of the opening's outputs come from
    one architecture);
  - one task per arm for the fresh starts;
  - any GPU type (≥ 40 GB, bf16);
  - task lists from `make_s21_tasks.py` (`jobs/s21_a.jsonl`; embeddings `jobs/s21_e.jsonl`).
  - 24 × 3 × 3 + 3 × 8 = 240 videos.
- **Embeddings:** the primary analysis uses CLIP ViT-L/14 (as in preregistrations 09 and 16, `clip_probe.py`). SigLIP
  so400m embeddings (`siglip_probe.py`) are computed too, for a secondary analysis only.

## Measures (as in preregistration 16 A)
- Class score: z = scale × (cos(frame, class label) − cos(frame, city label)).
- Window score: s[t0, t1) is the mean z over frames in seconds [t0, t1) after the opening (frame f ↦ (f − 9)/16 s);
  s_open is the mean over frames 1–8.
- Default: μ[t0, t1) is the mean of the same window's class score over the fresh starts.
- Content persistence: κ = (s − μ)/(s_open − μ); undefined when s_open − μ < 2.
- An opening's value: the mean of κ over its seeds.
- Burst contrast: R = κ[50, 55) − κ[45, 50).

## Hypotheses (`analyze_s21.py`, frozen; Holm over 4 tests; window [55, 60))
- **H1 a sink keeps content:** κ(sf60_s3) − κ(sf60_s0) > 0.
- **H2 re-rotation lowers it:** κ(sf60_s3rr) − κ(sf60_s3) < 0.
- **H3 a re-rotated sink keeps content:** κ(sf60_s3rr) − κ(sf60_s0) > 0.
- **H4 the late content return is positional:** R(sf60_s3rr) − R(sf60_s3) < 0.
- **Tests:**
  - two-sided one-sample t-tests over openings;
  - supported if Holm p < 0.05 and the estimate is on the predicted side;
  - 95% CIs by bootstrap (10,000 draws, resampling openings within class and the fresh seeds).

## Secondary analyses
- H1–H4 on tone-normalised embeddings (emb_tn);
- each seed's H1–H4 estimate;
- κ curves per arm (windows [T − 5, T), T = 5…60);
- H1–H4 pooled with seed 0 of preregistration 16 (each run's μ from its own fresh starts);
- variance of κ[55, 60) split into between-opening and between-seed parts (one-way random-effects ICC);
- H1–H4 with SigLIP embeddings (encoder robustness).

## Predictions
- **H1 supported:** about +0.4.
- **H2 supported:** about −0.3.
- **H3:** about +0.1. Seed 0 was near the significance boundary, so whether it holds is uncertain; if it does not, the
  paper keeps saying "no detectable content".
- **H4 supported:** about −0.3.

## Frozen files (SHA-256)
See the Chinese file.
