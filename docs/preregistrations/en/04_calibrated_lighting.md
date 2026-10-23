> English translation of the frozen Chinese preregistration [`../04_calibrated_lighting.md`](../04_calibrated_lighting.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration: does choosing the lighting description by measured brightness reduce drift? (2026-10-20, frozen before generation started)

## Background
The result of the matched-prompt experiment (`docs/match_results_cn.md`) was negative: writing the prompt from a free VLM description of the opening frame did not reliably reduce brightness drift. The cause found by exploratory analysis is that the VLM's lighting words do not match the actual brightness. It often describes normally lit frames as "dim, early morning or dusk", so the prompt pulls the frame toward darkness.

This gives a testable hypothesis: if the lighting description is chosen by the **measured brightness** of the opening frame, instead of being written freely by the VLM, drift should decrease.

In addition, the default brightness of the sunny prompt (≈43) is roughly the upper limit that text can reach. Opening frames brighter than that are therefore expected to gain little.

## Design
- **Opening frames, models, seeds:** exactly the same as in the matched-prompt experiment. 40 opening frames, Self Forcing and LongLive, seeds 0 and 1.
- **GPU architecture:** each opening frame is fixed to the GPU architecture used by its earlier G branch, specified with `gpu_only` (A40; A100/A800; L40S/RTX 6000 Ada). This keeps the new branches strictly paired with the old branches (G, M, M+ttc, etc.).
  - Each task also reruns G seed 0 as a pairing check. It is expected to be bitwise identical to the old file.
- **Lighting classification rule (`make_calib_prompts.py`, applied before generation):**
  - The candidate classes are the four lighting prompts of Part 4. Their zero-drift levels on Self Forcing come from an earlier, independent batch of 40 opening frames: night 16.0, dusk 19.1, overcast 33.6, sunny 42.9.
  - Each opening frame is assigned the class whose zero-drift level is closest to its opening brightness. The opening brightness is the L_open measured in the Self Forcing G branch (mean of the two seeds). Both models use the same value.
  - Classification result: sunny 23, overcast 10, dusk 5, night 2.
- **New branches:**
  - CL: generic scene + calibrated lighting, i.e. the original text of the Part 4 prompt for that class.
  - C: the VLM scene phrase (same as M) + calibrated lighting.
  - C+ttc: C plus correction (step 1, β = 1).
  - In total 40 × 2 models × (3 branches × 2 seeds + 1 check video) = 560 videos.
- **Known limitation:** in C, the VLM scene phrase sometimes contains weather or time words, for example "on a wet, rainy night" classified as dusk, or "under an overcast sky" classified as sunny. These conflict with the calibrated lighting. CL does not have this problem and serves as a cleaner control.

## Fix to the matched-prompt experiment data (carried out together with this experiment)
- **Problem:** the generation script skips output files that already exist. When a preempted task was rerun, it continued from the earlier progress instead of starting over. As a result, on LongLive, for the 4 opening frames ng16–ng19, the first three branches (G, G+ttc, M) were generated on A100 and the last two branches (M+ttc, ML) on A40. This broke the pairing.
- **Other experiments checked:** all other paired experiments ran within a single GPU architecture and do not have this problem.
- **Fix:** move ML and M+ttc of these 4 opening frames to `corr/match_ll_a40_split/` (kept), then regenerate them on A100/A800.
- **Subsequent handling:**
  - Reanalyze with the frozen `analyze_match.py` as a **sensitivity analysis**. The original results are kept as the "as-executed" results, and both are reported.
  - This fix only affects LongLive ML and M+ttc. It does not affect the data for H1 of the matched experiment (M vs G).

## Outcome measures
Same as the matched experiment. The primary outcome is |dL_end|; the secondary outcomes are dE_end, signed dL_end, content retention and flicker.

## Hypotheses and tests (`analyze_calib.py`, frozen)
- **Primary hypotheses (each tested once per model, 4 tests in total, Holm correction):**
  - H1: |dL_end| of C is smaller than that of M. The scene is the same; only the VLM's lighting description is replaced by the brightness-calibrated description.
  - H2: |dL_end| of C is smaller than that of G.
- **Test method:** for each opening frame, first average over the two seeds, then compute paired differences and run a two-sided one-sample t-test. Also give an opening-frame-clustered bootstrap 95% CI.
- **Decision rule:** a hypothesis holds if the Holm-corrected p < 0.05 and the estimate < 0.
- **Secondary analyses (no multiplicity correction):**
  - Paired differences: CL − G, C − CL, C+ttc − C, C+ttc − G, C+ttc − M+ttc.
  - Content retention: C − M (expected to be close to 0, because the scene is the same), C − G.
  - C − G and CL − G grouped by lighting class: night/dusk/overcast (17 frames) and sunny (23 frames) separately. The gain is expected to be concentrated in the former.
  - Out-of-sample test of the Part 4 pattern: use the per-prompt slopes and zero-drift levels fitted on Self Forcing in Part 4 to predict dL_end of the CL branch. Report R², correlation and mean error.
  - Flicker ratios.
  - Pairing check: the proportion of G seed 0 reruns that are bitwise identical to the old files.

## Frozen files (SHA-256)
```
63adfed7ff451539ffec91cb7d03b829d290b86e3a0eb8391a57de0352bfbcaf  analyze_calib.py
b3a631746c25fda1a51558a9c89f6b0bf44dd8a40fed0bcc35bb45d77a21183c  make_calib_prompts.py
b08602d83481323f3b6d1bbee5e0c2fdd93f09987b7b2fe2d308a9c98a1ac704  prompts/calibrated.jsonl
860c8be4ad14e7cef8f003c75ed50310bf8056e3e8ce29ad202f078c98756d88  jobs/calib_tasks_a40.jsonl
887ce7db03ea44b9adefcde90a727e693d22c5739721d1079c960c5782681952  jobs/calib_tasks_a100.jsonl
6b512116ce7b54ba3e623428e9b66a97b0c798998fc0842ff0fed7474c967727  jobs/calib_tasks_l40s.jsonl
7418c2f266fba61f4c2261c2f8a1f50245d4d8136dc8d83d67db58252ee1b81d  correct_gen2.py
b351d4b017fa51226488ae2d288b18f3b9cbbb54c362ef6dd1f006d376dd31f2  ll_gen.py
f6c15dba0bc97f0ba39b4a75c4c07c5f7f489eb9172d065f1f83d50434a95454  features.py
```
How to run: `python analyze_calib.py OUT results/corr/light_out/light_summary.json prompts/calibrated.jsonl SF=MATCH_SF,CALIB_SF LL=MATCH_LL,CALIB_LL`
