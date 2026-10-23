> English translation of the frozen Chinese preregistration [`../03_matched_prompts.md`](../03_matched_prompts.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration: does making the prompt describe the opening frame reduce drift? (2026-10-19, frozen before generation started)

## Background and question
Part 4, an exploratory analysis, found that after 31 seconds the frame brightness returns to the default level set by the prompt: only about 22% of the opening frame's brightness deviation remains. Both models behave this way, with slopes of about −0.7 to −0.8.

This gives a testable prediction: a large part of the drift comes from "mismatch between the prompt and the opening frame". If the prompt itself describes the opening frame (scene and lighting), drift should be clearly smaller. Content-aware correction (ttc_s1) should reduce drift further on top of this.

This experiment is the confirmatory test of Part 4. All opening frames are new and do not overlap with the 40 used before.

## Design
- **Opening frames (40, `prompts/match_openings.jsonl`; selection rules in `jobs/select_openings.py`)**
  - Real frames nr00–19: from 127 NAVSIM navtest logs not used before, take the CAM_F0 image at frame 20 of each log. Take 20 equally spaced target brightness values between the darkest (15.6) and the brightest (60.7). For each target, take the log whose brightness is closest (`jobs/scan_logs.py`).
  - Generated frames ng00–19: frame 0 of the 20 driving prompts with seed 1 from the Self Forcing confirmatory experiment. The earlier self opening frames used other videos and seed 0.
  - Each opening frame is held fixed for 0.5 seconds (9 frames) as the first chunk. The video is then continued to 31 seconds (501 frames).
- **Prompts (`prompts/matched.jsonl`, frozen before generation)**
  - G: the generic prompt used in the earlier real-opening experiment.
  - M: a caption of the opening frame is generated with Qwen2.5-VL-3B-Instruct using greedy decoding (script `caption_vlm.py`). It outputs two items, scene and lighting, which are combined as: "…driving {scene}. {lighting}. The camera moves forward steadily with realistic lighting and colors."
  - ML: a generic urban-road scene + the lighting sentence from the caption. Used to look separately at the effect of "matching lighting only".
  - Formatting rules: remove the period from scene and lowercase its first letter; if it does not start with on/through, prepend "through ". Capitalize the first letter of lighting. These rules were applied to all 40 entries before any video was generated. The raw captions are saved in the `caption` field.
- **Branches (5):** G, G+ttc, M, M+ttc, ML. ttc is always at_step 1, β = 1.
- **Models (2):**
  - Self Forcing: window 21 + sink 3, `correct_gen2.py`.
  - LongLive 1.0: window 12 + sink 3, `ll_gen.py`.
- **Seeds:** 0, 1. In total 40 × 5 × 2 × 2 = 800 videos.
- **Pairing:** for the same model and the same opening frame, the 5 branches and 2 seeds run sequentially in the same task, and therefore on the same GPU (`jobs/match_tasks.jsonl`).

## Outcome measures (relative to the opening frame)
- **Primary:** |dL_end|, the absolute difference between the mean L\* of the last 2 seconds (32 frames) and the mean L\* of the opening frame (frames 1–8).
- **Secondary:**
  - dE_end: Euclidean distance between the Lab means.
  - Signed dL_end.
  - Flicker: jitter, and chunk-boundary jumps.
  - Content retention: cosine similarity between the DINOv2 CLS of the last 2 seconds and the CLS of the opening frame. Higher is better.

## Hypotheses and tests (`analyze_match.py`, frozen)
- **Primary hypotheses (each tested once per model, 4 tests in total, Holm correction)**
  - H1: |dL_end| of M is smaller than that of G.
  - H2: |dL_end| of M+ttc is smaller than that of M.
- **Test method:** first average over the two seeds within each opening frame, then compute paired differences and run a two-sided one-sample t-test. Also give an opening-frame-clustered bootstrap 95% CI (10000 resamples).
- **Decision rule:** a hypothesis holds if the Holm-corrected p < 0.05 and the estimate < 0.
- **Overall conclusion:**
  - H1 holds on both models → "making the prompt consistent with the opening frame reduces drift" is confirmed.
  - H1 holds on only one model → the conclusion is limited to that model.
  - Same for H2.
- **Secondary analyses (estimates and CIs only, no multiplicity correction, not confirmatory conclusions)**
  - ML vs G: the effect of matching lighting only.
  - G+ttc vs G.
  - M vs G+ttc: which is more effective, changing the prompt or correction.
  - M+ttc vs G.
  - Interaction: prompt × correction.
  - H1 reported separately for real frames and generated frames.
  - Regression slope and zero-drift level of each branch (M's slope is expected to be closer to 0).
  - Relationship between the H1 gain and the distance from the opening frame to G's default brightness.
  - dE, content retention, flicker ratios.

## Expectations and risks
- **Expectation:** opening frames that are dark or bright, far from the default brightness, gain the most from M.
- **Risk 1:** M also changes the scene description, which may introduce other changes (for example, more motion). ML is therefore added to isolate the effect of lighting.
- **Risk 2:** captions that say "dusk / night" may instead ask the frame to keep getting darker.
- **Risk 3:** the NAVSIM test set has no true night scenes. The darkest real frames are shadows and parking lots.

## Frozen files (SHA-256)
```
315fbf7d6e18c9b6002f8dcddd8c02e59be77087143e6d7a4e793cfd1aec22de  analyze_match.py
c4ea682a419fc3e4903e4f4539820813094ca6d0edabfd71480f01b84136cc10  caption_vlm.py
cee2d465d38345a2f487debc5983b8161dac02870653680ac212ec6ea28b41a2  jobs/select_openings.py
16403f81203011a5dd53bd7bcc06df30bf5404eef2c52cbbd409ebe3e22e2d17  prompts/match_openings.jsonl
7aac5cce3728a0f9dc4f6bce4bb965c8a0e568a202fa4de6f438bdebf30436e0  prompts/matched.jsonl
5f6e35ca194e4999d99271ab3ac9a531bdd364119da5703e1ec1f212c2bd6ba9  jobs/match_tasks.jsonl
7418c2f266fba61f4c2261c2f8a1f50245d4d8136dc8d83d67db58252ee1b81d  correct_gen2.py
b351d4b017fa51226488ae2d288b18f3b9cbbb54c362ef6dd1f006d376dd31f2  ll_gen.py
f6c15dba0bc97f0ba39b4a75c4c07c5f7f489eb9172d065f1f83d50434a95454  features.py
```
How to run: `python analyze_match.py OUT SF=RUN_SF LL=RUN_LL`.
