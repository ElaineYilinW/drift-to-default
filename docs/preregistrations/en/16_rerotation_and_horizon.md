# Preregistration 16: the re-rotated sink, and other anchors at 60 s (2026-10-25, frozen before generation)

*English translation; the Chinese file `../16_rerotation_and_horizon.md` is the frozen, authoritative version.*

## Background
- **The late return of content.** In preregistration 09, the content persistence κ of Self Forcing DMD (sink 3, generic city prompt) is about 0.3 at 30 s, but rises again to 0.74 at 50–55 s. In time this coincides with the positional bursts in tone persistence in preregistration 10 (around 32 s and 52 s). Preregistration 10 showed that re-rotating the sink's keys to just in front of the window removes the tone bursts. The paper states that this study tests **whether the late return of content also disappears with re-rotation**.
- **What happens to a deeper sink when it is re-rotated.** Preregistration 10 re-rotated only sink 1 and sink 3 (α at 60 s: 0.07, 0.16); sink 6 was not tested (0.76 at its original positions). If a deeper re-rotated sink still keeps more, deepening the sink is a way of keeping the opening without bursts.
- **Other anchors were only measured to 20 s.** FramePack (the only non-Wan base) and SkyReels were both run for only 20 s. Under our recursion model:
  - FramePack carries a clean prefix of the opening in every section (a permanent anchor), so α should approach a positive plateau;
  - SkyReels' overlap frames are the end of the previous segment, i.e. a sliding window rather than a permanent anchor, so it should keep forgetting, only slowly.
  The paper's discussion lists "overlap frames" together with the sink and the prefix as permanent anchors; this needs to be tested at 60 s.

## Design
- **A Content (Self Forcing DMD, 60 s):**
  - the 24 openings of preregistration 09, generic city prompt (arm mis), seed 0;
  - three configurations: sink 3 (sf60_s3), sink 3 re-rotated (sf60_s3rr), no sink (sf60_s0);
  - the three configurations of one opening run in the same task (arch_guard);
  - for each configuration, 8 videos are generated from scratch with the generic prompt (seeds 0–7) as the default μ;
  - CLIP embeddings with `clip_probe.py` (the same label texts as preregistration 09). 72 + 24 videos in total.
- **B Tone (Self Forcing DMD, 60 s):**
  - the 24 scenes of preregistration 06, openings L\* −16 / +16;
  - three configurations: sink 6 (sf60_s6), sink 6 re-rotated (sf60_s6rr), sink 3 re-rotated (sf60_s3rr);
  - the three configurations of one scene are in the same task (arch_guard). 144 videos in total.
- **C FramePack-F1 (60 s):**
  - the 12 scenes of preregistration 08, openings L\* −16 / +16;
  - two conditions: the shift enters both routes (fpv_both), or only the context latent (fpv_ctx). 48 videos in total.
- **D SkyReels-V2-DF (60 s):**
  - the same 12 scenes and openings, overlap_history 5 (sky5) and 17 (sky17). 48 videos in total.
- **Execution of C and D:** one video per task, all tasks pinned to A40 (`gpu_only`), so that the two versions of a scene are on the same GPU architecture; each task still runs arch_guard at its start.
- **Task lists:** `make_s16_tasks.py` → `jobs/s16_ab.jsonl`, `jobs/s16_cd.jsonl`, `jobs/s16_e.jsonl` (CLIP, run after A has finished).
- **No separate smoke test.** The first few C/D tasks serve as the smoke test. If a code error requires a change, the change and its reason are recorded in the lab notes as a deviation.

## Metrics
- **A:**
  - the frame-level class score z, s[t0, t1), s_open and μ[t0, t1) are defined as in preregistration 09;
  - κ[t0, t1) = (s − μ) / (s_open − μ), undefined when s_open − μ < 2.
  - Burst contrast R = κ[50, 55) − κ[45, 50). The windows come from the tone-burst location [50, 54) s in preregistration 10, not from the content data.
- **B–D:**
  - α(T) = (p16 end − m16 end) / (p16 opening − m16 opening);
  - the opening is the mean L\* of the held frames (frame 1 to frame hold−1);
  - the end is the mean L\* over [T−10, T) s after the opening.
- **Collapse rule (C, D):**
  - if either version has any 1-s mean L\* below 3 or above 97 within [0, T) s after the opening, the scene is excluded from that condition at endpoint T;
  - if more than 6 of the 12 scenes are excluded at 60 s, the test of that condition instead uses the latest of 50, 40 and 30 s at which no more than 6 are excluded;
  - D compares α(T) − α(20 s) at that endpoint.

## Hypotheses (`analyze_s16.py`, frozen; Holm correction, 9 tests in total)
- **A1 the late return of content is positional:** R(sf60_s3rr) − R(sf60_s3) < 0.
- **A2 re-rotation lowers the content kept at 60 s:** κ[55, 60)(sf60_s3rr) − κ[55, 60)(sf60_s3) < 0.
- **A3 the re-rotated sink still keeps content:** κ[55, 60)(sf60_s3rr) − κ[55, 60)(sf60_s0) > 0.
- **B1 re-rotation also lowers what a deep sink keeps:** α(sf60_s6rr, 60) − α(sf60_s6, 60) < 0.
- **B2 without bursts, a deeper sink still keeps more:** α(sf60_s6rr, 60) − α(sf60_s3rr, 60) > 0.
- **C1 FramePack still keeps a share at 60 s:** α(fpv_both, 60) > 0.
- **C2 the context prefix alone also keeps a share:** α(fpv_ctx, 60) > 0.
- **D1 a 5-frame overlap keeps forgetting:** α(sky5, 60) − α(sky5, 20) < 0.
- **D2 a 17-frame overlap keeps forgetting:** α(sky17, 60) − α(sky17, 20) < 0.
- **Test method:**
  - two-sided one-sample t-tests over openings (A) or scenes (B–D);
  - supported if Holm p < 0.05 and the estimate is on the predicted side;
  - 95% CIs by bootstrap (A: openings resampled within class, seeds of the from-scratch videos resampled as well; B–D: scenes resampled; 10,000 draws each).

## Secondary analyses
- κ curves of each configuration (windows [T−5, T), T = 5..60);
- contrast of the first burst, κ[31, 36) − κ[26, 31);
- A1, A2 on tone-normalised embeddings;
- α of each condition of B–D at 10–60 s;
- B1, B2 in the burst-free window [40, 50);
- the jump α([34, 35)) − α([31, 32)) of each condition of B;
- for FramePack, α(fpv_both, 60) − α(fpv_ctx, 60), and α(60) − α(20) for each condition;
- for SkyReels, α(sky17, 60) − α(sky5, 60);
- number of excluded scenes;
- reproduction of earlier runs with the same seed and opening (reported separately).

## Predictions
- **A1 holds:** R(sf60_s3) about +0.35 (preregistration 09), R(sf60_s3rr) about 0.
- **A2 holds:** κ[55, 60) falls from about 0.6 to 0.2–0.35.
- **A3 uncertain, leaning towards holding:** on tone, re-rotated sink 3 gives 0.16 and no sink 0.05.
- **B1 holds:** sink 6 falls from about 0.76 to about 0.3.
- **B2 holds:** sf60_s6rr is above sf60_s3rr (about 0.16).
- **C1, C2 hold:** fpv_both about 0.6, fpv_ctx about 0.4, with only a small decline from 20 s to 60 s.
- **D1 holds:** sky5 falls from 0.77 to about 0.5.
- **D2 uncertain:** the decline is small and may not be significant.

## Frozen files (SHA-256)
```
788f3ddbbb03ebb534a523d56c75557bcfc4a1ff5825b41e1dd505d7c5befe44  analyze_s16.py
6113a0ad9c54d255c3aaaee9174da1320bbc66c806130b53341ece31f61d5aac  make_s16_tasks.py
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
e19888da3167a5190f28e1577474435b8d5bb5add5ea12c606a66194f2d03e3e  fp_gen.py
4eb020bf429c1a6526d87e1b1496c3cb2a61dd1ffc36e08c583c31a3f54a7036  sky_gen_x.py
002fe72cfc2c768d72e6f17f53228fb66664278ba12f6a6a1a0008dfc700d08f  clip_probe.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
13883002dc681afe246e668c5d27e365f0058782d02476d5caa393aded3b4ab4  jobs/s16_ab.jsonl
6ce25cbe0eb8a15f0aebd1cddef1e846098307f03592d02c254e0d16dcb8c37a  jobs/s16_cd.jsonl
13dc6c4ca54878c25ad79c807838d2e32df00ef6d278240082e29e0547adc796  jobs/s16_e.jsonl
c41b6d71c33c9428cd8c8e1a559dc44cb72a9daae5b075a637e668ca4c5074d1  prompts/content_runs.jsonl
cfb7068527b22670c611c8b9155b2570875e49c5625a84fe1f62a4ee3a7c38dc  prompts/content.jsonl
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
