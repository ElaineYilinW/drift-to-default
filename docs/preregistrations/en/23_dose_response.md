> English translation of the frozen Chinese preregistration [`../23_dose_response.md`](../23_dose_response.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 23: is relaxation a linear response? Dose-response of tone persistence (Self Forcing, 60 s) (2026-10-28, frozen before generation)

## Background
- The paper's central relation x∞ ≈ α·x_open + (1 − α)·μ is linear in the opening: α is treated as a property of the
  model and of how the opening is held, independent of how far the opening was shifted.
- But every tone α in the paper is estimated from one pair of shifts (L\* −16 / +16).
- If α changed with the shift size (for example because a very dark opening is read as night, or one side
  saturates), α would only be a local linearisation, and the paper's statements "no sink forgets completely" and "a
  re-rotated sink keeps about 0.2" would hold only at ±16.
- The only dose check so far (lab note 05) shifted a single chunk inside the KV window by ±2 / ±5 / ±10 over a short
  time (α 0.67 / 0.75 / 0.79, slightly rising with dose). It did not test the opening's persistence at 60 s.
- This study applies five shifts of different size to the same scenes and tests three things:
  - whether α is independent of the shift size;
  - whether darkening and brightening are kept alike;
  - whether a large shift is also forgotten completely without a sink.

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), generic prompt (`prompts/relax.jsonl`).
  - The openings' mean L\* is about 35–50; at −24 about 11–26 (very dark), at +24 about 59–74.
- **Model:** Self Forcing DMD, 60 s (243 latent frames), `hold_gen_long_rr.py` (as in preregistrations 10, 16, 20).
- **Arms:**
  - sf60_s3: sink 3 at its original positions;
  - sf60_s0: no sink;
  - sf60_s3rr: sink 3 re-rotated.
- **Openings:** the opening image's L\* is shifted in CIELAB by −24, −8, 0, +8, +24 (versions m24, m8, z0, p8, p24)
  before VAE encoding (as in the earlier studies).
  - The unshifted version uses the original image without a Lab round trip.
- **Seeds:** 0, 1.
- **Execution:**
  - one task per scene × seed with all arms and versions (15 videos);
  - each seed writes to its own folder, and arch_guard keeps all of a scene's outputs in that folder on one GPU
    architecture;
  - any GPU type (≥ 40 GB, bf16);
  - task list from `make_s23_tasks.py` (`jobs/s23.jsonl`).
  - 24 × 3 × 5 × 2 = 720 videos.

## Measures
- For each scene, arm, seed and version v:
  - open_v: mean L\* of frames 1–8;
  - end_v: mean L\* over seconds [T − 10, T) after the opening (16 fps).
- For a pair of versions (hi, lo): α(hi, lo) = (end_hi − end_lo) / (open_hi − open_lo); undefined if
  open_hi − open_lo < 4 L\*.
  - α_small = α(p8, m8);
  - α_large = α(p24, m24);
  - α_up = α(p24, z0);
  - α_down = α(z0, m24).
- A scene's value: the mean over its seeds.
- No collapse exclusion (as in preregistrations 10 and 20).

## Hypotheses (`analyze_s23.py`, frozen; Holm over 4 tests; T = 60 s)
- **D1 sink 3's persistence does not depend on the shift size:** α_large − α_small within ±0.15.
- **D2 the same for the re-rotated sink 3:** α_large − α_small within ±0.15.
- **D3 sink 3 keeps darkening and brightening alike:** α_up − α_down within ±0.15.
- **D4 without a sink the large shift is forgotten too:** α_large(sf60_s0) within ±0.10.
- **Tests:**
  - equivalence: over scenes, two one-sided t-tests against the margins (TOST); p is the larger of the two;
  - supported if Holm p < 0.05;
  - 95% CIs by scene bootstrap (10,000 draws); 90% t intervals are reported alongside.
- **Choice of margins:**
  - 0.15 is smaller than the smallest arm difference the paper interprets (a re-rotated sink keeps about 0.2 more
    than no sink);
  - 0.10 is the operational definition of "forgets completely".

## Secondary analyses
- D1–D4 in the [40, 50) s window;
- α_small, α_large, α_up, α_down at 10, 20, 30, 40, 50, 60 s per arm;
- per scene, arm and seed, least-squares fits of end(60) on open over the five openings:
  - slope and R²;
  - a quadratic fit in centred open, its quadratic coefficient tested against 0 per arm;
- each seed's D1–D4 estimates;
- the jump α([34, 35)) − α([31, 32)) of the small and the large pair per arm, and their difference (is the jump
  proportional to the shift?);
- attractor offset: end(60) − open of the z0 version per arm;
- α_16 = α(p16, m16) of the same scenes, arms and seeds from preregistrations 10, 16 and 20, next to α_small and
  α_large (descriptive);
- the number of videos per arm and opening with a 1-s mean L\* below 3 or above 97 (collapse; no exclusion).

## Predictions
- **D1, D2 supported:** the paper's linear relation predicts that α does not depend on the shift size. In the KV-window
  dose check α rose slightly with the dose (0.67 → 0.79), so α_large may be a little larger than α_small, but within
  ±0.15.
- **D3:** uncertain. A very dark opening (L\* about 11–26) may be read as night and pulled toward the default
  differently from a brighter one. If D3 fails, the linear relation holds only within a range around the opening.
- **D4 supported:** without a sink α is about 0 at 60 s (preregistration 20: −0.01).
- If D1 or D2 fails, the paper's α should be stated as the persistence of a ±16 shift, not as a constant of the model.

## Frozen files (SHA-256)
See the Chinese file.
