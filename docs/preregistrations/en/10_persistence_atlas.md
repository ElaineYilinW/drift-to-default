> English translation of the frozen Chinese preregistration [`../10_persistence_atlas.md`](../10_persistence_atlas.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 10: a 60-s persistence atlas, and the role of sink size and sink position (2026-10-24, frozen before generation)

## Background
- Study 06 compared the persistence α of 14 configurations at 31 s. Study 07 showed that Self Forcing is not at
  equilibrium at 31 s (with a sink: 0.28 at 31 s, 0.58 at 60 s; without: 0.23 → 0.03), so the 31-s ranking may not be the
  equilibrium ranking.
- Two anomalies are unexplained: sink 1 keeps less than no sink (31 s, exploratory), and with a sink α jumps between
  31.5 s and 34.5 s (study 07 curves).
- A candidate explanation: Self Forcing's sink keeps its original absolute RoPE positions, which drift ever further from
  the current frame and out of the training range. Re-rotating the sink keys to sit just before the window (as Rolling
  Forcing and Deep Forcing do) tests this.
- This study re-measures α for all models at 60 s and tests sink size and sink position encoding.

## Design
- **Scenes and prompt:** the 24 scenes of study 06, generic prompt, seed 0; 60 s (243 latent frames).
- **Configurations:**
  - Self Forcing DMD: sink 3, sink 6, sink 3 re-rotated (one task); no sink, sink 1, sink 1 re-rotated (another task).
    Openings L\* −16 and +16. Window 21 throughout. Sink 6 = the opening chunk + the first generated chunk.
  - LongLive sink 3 and no sink; Self Forcing GAN, SiD, SiD v2, 10-s; Rolling Forcing. Openings L\* −16, 0, +16.
- **Execution:** arms that are compared share a task, which starts with arch_guard so they share a GPU architecture
  (`make_atlas_tasks.py`, task list `jobs/atlas.jsonl`). Generators: `hold_gen_long_rr.py` (hold_gen_long.py plus
  `--rerotate`), `ll_hold_gen_long.py`, `rf_gen_long.py` (rf_gen.py plus segment-wise decoding).
  24 × (12 + 6 + 6 + 6 + 3) = 792 videos.
- **Measure:** α(T): within-scene OLS slope of the mean L\* over seconds [T−10, T) after the opening on the opening's L\*
  (frames 1–8), over two or three versions. Primary endpoint T = 60 s.

## Hypotheses (`analyze_atlas.py`, frozen; Holm over 4 tests)
- **H1, a deeper sink keeps more:** α(sf60_s6) − α(sf60_s3) > 0.
- **H2, any sink keeps some:** α(sf60_s1) − α(sf60_s0) > 0 (the chunk recursion's prediction; the exploratory 31-s result
  went the other way).
- **H3, LongLive's sink still keeps at 60 s:** α(ll60_s3) − α(ll60_s0) > 0.
- **H4, the sink's position encoding matters:** α(sf60_s3rr) − α(sf60_s3) ≠ 0 (two-sided, no direction predicted).
- **Tests:** per-scene differences, two-sided one-sample t-tests, scene-bootstrap 95% CIs. H1–H3: Holm p < 0.05 and
  estimate > 0; H4: Holm p < 0.05 (direction reported).
- **Secondary:** α(sf60_s1rr) − α(sf60_s1); α at 10–60 s for every configuration; Spearman correlation of 60-s α with
  study 06's 31-s α; the jump α(34.5 s) − α(31.5 s) (1-s windows) for each Self Forcing DMD arm; the share of the opening
  lost by 4.5 s, 1 − α(4.5 s), against the total loss by 60 s (separating early rejection of an artificial shift from
  slow forgetting); the zero-drift level at 60 s for three-version configurations.
- **Predictions:** H1–H3 hold; no direction for H4.

## Frozen files
See the hash block in the Chinese original.
