> English translation of the frozen Chinese preregistration [`../08_skyreels_overlap.md`](../08_skyreels_overlap.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 08: Does reducing the overlap make SkyReels relax? (2026-10-23, frozen before generation)

## Background
- SkyReels-V2-DF hardly relaxes (study 06: α = 0.96 at 20 s), and this does not change with CFG, history-frame noise
  or the number of denoising steps (lab note 12).
- In an exploratory check (4 scenes), reducing the overlap from 17 to 5 frames lowered α at 20 s from 0.89 to 0.52.
  This supports the conjecture that the hard constraint formed by the overlapping frames is what keeps the opening.
- This study confirms it on 12 scenes **not used in the exploratory check**.

## Design
- **Scenes:** rx02, rx06, rx08, rx10, gx02, gx06, gx08, gx10, rx01, rx03, gx01, gx03.
- **Conditions:** opening L\* −16 and +16; `overlap_history` 5 or 17; 20 s; seed 0; generic prompt.
- **Execution:** both overlaps of a scene run in one task (arch_guard). Generator `sky_gen_x.py`, task list
  `jobs/skyconf.jsonl`. 48 videos.
- **Measure:** α = (p16 end − m16 end) / (p16 opening − m16 opening), with the end taken as the mean L\* over seconds
  [15, 20) after the opening.

## Hypothesis (`analyze_skyov.py`, frozen)
- **Primary:** α(overlap 5) < α(overlap 17).
- **Test:** per-scene difference of α between the two overlaps, two-sided paired t-test, scene-bootstrap 95% CI.
  Supported if p < 0.05 and the difference < 0.
- **Secondary:** α of both overlaps at 5, 10, 15 and 20 s.

## Frozen files (SHA-256)
```
75eaa574085cb55839e95c019c640b254d6e07f10d52dd1e9da88feddcf0b90a  analyze_skyov.py
4eb020bf429c1a6526d87e1b1496c3cb2a61dd1ffc36e08c583c31a3f54a7036  sky_gen_x.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
f3fe8b8a2e20278c297d9f1f1c27e332e4614c6b3278bd937525a742107c7276  jobs/skyconf.jsonl
```
