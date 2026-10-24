> English translation of the frozen Chinese preregistration [`../12_prompt_generality.md`](../12_prompt_generality.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 12: do relaxation and the sink effect hold under non-driving prompts? (2026-10-24, frozen before generation)

## Background
- The interventional studies 06, 07 and 10 used a single generic driving prompt. Whether relaxation and the sink effect
  are specific to driving scenes is an obvious question.
- An exploratory re-analysis of the lighting experiment (lab note 06) found that Self Forcing's persistence varies with
  the prompt (sunny 0.34, overcast 0.24, dusk 0.17, night 0.05; interaction cluster-robust p = 0.001). This study
  preregisters a test of "persistence varies with the prompt" in new scene types.

## Design
- **Prompts:** four non-driving scenes: a walk in a park, a drone over a coastline, a kitchen, an outdoor market
  (`prompts/pgen.jsonl`), in the template of the generic driving prompt.
- **Openings:** Self Forcing DMD generates 5-s fresh videos from each prompt (seeds 0–5); frame 64 (4 s) is the opening.
  No selection (`pgen_openings.py`). 24 openings.
- **Continuations:** each opening is continued under its own prompt with the opening's L\* shifted by −16, 0, +16, seed 0:
  Self Forcing DMD with sink 3 and no sink, 60 s (`hold_gen_long.py`), in one task; LongLive with sink 3 and no sink,
  31 s (`ll_hold_gen.py`), in another. Every task starts with arch_guard (`make_pgen_tasks.py`). 288 videos.
- **Measure:** α(T): within-opening OLS slope of the mean L\* over seconds [T−10, T) after the opening on the opening's
  L\* (frames 1–8). T = 60 s for Self Forcing, 30 s for LongLive.

## Hypotheses (`analyze_pgen.py`, frozen; Holm over 3 tests)
- **H1:** α(sf60_s3) − α(sf60_s0) > 0 at 60 s.
- **H2:** α(ll_s3) − α(ll_s0) > 0 at 30 s.
- **H3, persistence differs between prompts:** one-way ANOVA of per-opening α(sf60_s3, 60 s) over the four scenes
  (no direction).
- **Tests:** H1, H2: two-sided one-sample t-tests over openings, supported if Holm p < 0.05 and estimate > 0; H3:
  supported if Holm p < 0.05. 95% CIs resample openings within scene.
- **Secondary:** α per scene and configuration at 10–60 s; zero-drift level per scene; H3 repeated for LongLive (30 s).
- **Predictions:** H1–H3 hold.

## Frozen files
See the hash block in the Chinese original. `jobs/pgen_b.jsonl` (phase 2) is produced by `make_pgen_tasks.py b` after
the openings are generated and is fully determined by the frozen scripts and the openings.
