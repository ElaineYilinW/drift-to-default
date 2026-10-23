> English translation of the frozen Chinese preregistration [`../05_calibrated_lighting_longlive.md`](../05_calibrated_lighting_longlive.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration: calibrating lighting descriptions for LongLive with its own brightness scale (2026-10-20, frozen before generation)

## Background
The previous experiment (`docs/calib_results_cn.md`) chose the lighting description by the measured brightness of the opening frame, using the brightness scale measured on Self Forcing. The result held on Self Forcing (|dL| −26%, Holm p = 0.013) and was not significant on LongLive. Predictions of the LongLive results made with this scale were systematically too high, by +4.3 L\*. From this we infer that the scale must be measured separately for each model.

## Step 1 (completed, exploratory): measuring the LongLive scale
- **Procedure:** using the 40 old opening frames of Part 4 (real00–19, self00–19, independent of the new opening frames in this experiment), pair each one on LongLive with each of the 4 lighting prompts, 2 seeds per prompt, 320 videos in total. The analysis script is `analyze_light.py`; the results are in `results/corr/ll_light_out/`.
- **Zero-drift level of each prompt on LongLive:**

  | Prompt | Zero-drift level | 95% CI |
  |---|---|---|
  | night | 21.2 | [16.7, 23.8] |
  | dusk | 32.3 | [30.4, 34.4] |
  | overcast | 39.8 | [37.7, 41.7] |
  | sunny | 44.8 | [43.2, 46.6] |
  | Generic prompt (control) | 39.9 | — |

  The corresponding values on Self Forcing are: night 16.0, dusk 19.1, overcast 33.6, sunny 42.9.
- **Test of the lighting term:** p ≈ 4×10⁻⁵¹. The finding that the prompt determines the default brightness is replicated on LongLive. LongLive's range is narrower, and dusk in particular is brighter.

## Step 2 (this preregistration)
- **Classification rule:** reuse `make_calib_prompts.py` with the code unchanged; only the brightness scale is replaced with LongLive's. The resulting prompts are in `prompts/calibrated_ll.jsonl`.
- **Classification result:** sunny 19, overcast 7, dusk 7, night 7. Compared with the classification by the Self Forcing scale, 16 opening frames changed class:
  - overcast → dusk: 7
  - dusk → night: 5
  - sunny → overcast: 4
- **Generation and reuse:**
  - For these 16, CL, C and C+ttc are regenerated on LongLive (2 seeds each), plus G seed 0 as a pairing check. The GPU architecture is fixed to the one used by their original G branch (`jobs/calib2_tasks_*.jsonl`).
  - For the other 24, the prompts are exactly the same as in the previous experiment, and the model, seeds and architecture are also the same, so the outputs are necessarily bitwise identical. The files from the previous experiment are therefore copied directly (`jobs/calib2_reuse.txt`).
  - All files are placed in `corr/calib2_ll`.
- **Primary hypotheses** (computed with the frozen `analyze_calib.py`; only one model, LongLive; 2 tests, Holm correction):
  - H2 (most important): |dL_end| of C is smaller than that of G.
  - H1: |dL_end| of C is smaller than that of M.
- **Decision rule:** Holm-corrected p < 0.05 and estimate < 0.
- **Secondary analyses:**
  - From `analyze_calib.py`: CL − G, C+ttc − G, content retention; grouping by lighting class; in-sample prediction with LongLive's own scale (R², mean error).
  - From `analyze_calib_ll.py`: branches chosen by the LongLive scale vs branches chosen by the Self Forcing scale, compared on all 40 opening frames and on the 16 that changed class. The improvement is expected to be concentrated in these 16.
- **Note:** the Step 1 scale was determined after looking at the LongLive Step 1 data. However, those data come from a different batch of opening frames, so they are independent of this experiment.

## Frozen files (SHA-256)
```
63adfed7ff451539ffec91cb7d03b829d290b86e3a0eb8391a57de0352bfbcaf  analyze_calib.py
0d514f802890c6da95cb76c6f1c518ad9061be20c432c98bc9092da7c01c7eae  analyze_calib_ll.py
b3a631746c25fda1a51558a9c89f6b0bf44dd8a40fed0bcc35bb45d77a21183c  make_calib_prompts.py
51c31a69978fdc4e416fbbbb7c81489b992d5c9f7340cd0073e4a8b14248d19a  prompts/calibrated_ll.jsonl
c3f70f7516362bd7391615096286e7617aabe01c7c1d7a85b170a376067a7880  jobs/calib2_tasks_a100.jsonl
efffdffe7005ef7eaea14960bcb557523baf7449b4097fc0b9608c4c48114a96  jobs/calib2_tasks_a40.jsonl
ac7d3247fb467ce06087c74285d392005606374e39c53d00df1a67d0ee68d899  jobs/calib2_tasks_l40s.jsonl
30b2b6b0cb79aef4c026f96bff1764baecfd5d0e12ae5ecfe9c5f65476f3e182  jobs/calib2_reuse.txt
475eecb8cc125fcd792c24da60221b13bf8b735084d8443ad3a39e496ffc34c7  results/corr/ll_light_out/light_summary.json
```
How to run:
- `python analyze_calib.py OUT results/corr/ll_light_out/light_summary.json prompts/calibrated_ll.jsonl LL=MATCH_LL,CALIB2_LL`
- `python analyze_calib_ll.py OUT CALIB_LL CALIB2_LL prompts/calibrated.jsonl prompts/calibrated_ll.jsonl`
