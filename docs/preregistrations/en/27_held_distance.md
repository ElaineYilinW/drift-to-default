> English translation of the frozen Chinese preregistration [`../27_held_distance.md`](../27_held_distance.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 27: with the sink held at one distance, does the probe's attention share there decide how much is kept and how much the video moves? (Self Forcing, 17 distances, 60 s) (2026-10-28, frozen before generation)

## Background
- Results of preregistration 25 (lab note 31):
  - The bursts follow the sink's rotary distance. With the sink 18 or 36 latent frames further back, the bursts come
    4.5 or 9 s earlier, as the probe predicts.
  - A sink held at one distance no longer bursts. At 60 s it keeps 0.86 at distance 203 (a peak), 0.22 at distance 20
    (next to the window) and 0.01 at distance 187 (a trough). These are in the same order as the probe's shares at the
    three distances (0.165, 0.086, 0.054 in the second probe).
  - The cost: held at the peak distance, the video nearly freezes on its opening. Motion over [50, 60) s is 1.10,
    against 2.38 at the original positions and 5.11 at the trough.
- That the share decides how much is kept rests on 3 points, and the 3 distances were chosen after looking at the
  probe. This study tests the relation at 14 new distances chosen by a rule, and tests whether motion falls with the
  share.
- **Probe:** the second probe (exploratory, `sinkprobe_long_out.json`, distances 3–723; 4 openings × 6 chunks; see
  preregistration 26 and lab note 30).
- **State of the data:** at freezing, this study has generated no video. The videos of preregistration 25 (the same 24
  scenes, seed 0) have been analysed; its 3 distances (20, 187, 203) are generated again here for secondary analyses
  only.

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), generic prompt (`prompts/relax.jsonl`).
- **Model:** Self Forcing DMD, sink 3, window 21, 60 s (243 latent frames).
  - Generator `hold_gen_sinkpos.py --sink_mode dist:D`, run in Self-Forcing-b, as in preregistration 25.
  - From the first chunk after the opening, a copy of the sink keys is rotated to distance D.
- **Openings:** L\* shifts −16 / +16 (versions m16, p16).
- **Seed:** 0.
- **Rule for choosing the distances** (`s27_distances.py`, fixed before any video):
  - candidates: the probe's grid distances with 24 ≤ D ≤ 720, except those within 2 of 187 or 203 (used in
    preregistration 25);
  - targets: 14 equally spaced share values from the smallest to the largest candidate share;
  - for each target in turn, the unused candidate whose share is closest (ties: smaller D).
- **Result (14 new distances; shares from the second probe):**

| D | share | D | share |
|---|---|---|---|
| 66 | 0.0932 | 448 | 0.0682 |
| 90 | 0.1017 | 466 | 0.1187 |
| 132 | 0.1096 | 468 | 0.1268 |
| 135 | 0.1461 | 513 | 0.0850 |
| 147 | 0.0766 | 589 | 0.0432 |
| 335 | 0.1516 | 643 | 0.1348 |
| 345 | 0.0601 | 650 | 0.0516 |

  - The 7 with the lowest share (used by Q3): 589, 650, 345, 448, 147, 513, 66.
  - The rank correlation between distance and share is −0.27 (not significant), so the two are largely unconfounded.
  - 132 and 135, and 466 and 468, are only 2–3 frames apart but have different shares, which tests the specificity
    of the share.
  - 9 distances exceed 245, which a sink at its original positions never reaches within 60 s.
- **Also generated again:** the 3 distances of preregistration 25, 20 (0.0859), 187 (0.0543) and 203 (0.1648), for
  secondary analyses only.
- **Execution:**
  - one task per scene with all 17 distances and both versions (34 videos); arch_guard keeps all outputs of a scene on
    one GPU architecture;
  - a distance whose p16 output already exists is skipped, so an interrupted task can resume;
  - any GPU model (≥ 40 GB, bf16);
  - video files only for rx00 and gx00;
  - task list made by `make_s27_tasks.py` (`jobs/s27.jsonl`, 24 tasks).
  - In total 24 × 17 × 2 = 816 videos.

## Measures
- α[t0, t1) = (end_p16 − end_m16) / (open_p16 − open_m16):
  - open is the mean L\* of frames 1–8;
  - end is the mean L\* over seconds [t0, t1) after the opening;
  - undefined if open_p16 − open_m16 < 4 L\*.
- No collapse exclusion.
- **motion[t0, t1):** the generator's per-frame motion (mean absolute grey-level difference from the previous frame,
  0–100) averaged over seconds [t0, t1) and over the two versions.
- **share(D):** the second probe's mean share at distance D.

## Hypotheses (`analyze_s27.py`, frozen; Holm correction over 3 tests)
- One Spearman correlation per scene (across distances), averaged over scenes:
  - **Q1 persistence rises with the share:** ρ_s(share, α[50, 60)) over the 14 new distances (at least 10 defined per
    scene); mean > 0.
  - **Q2 motion falls with the share:** ρ_s(share, motion[50, 60)) over the 14 new distances; mean < 0.
  - **Q3 graded below the peaks:** ρ_s(share, α[50, 60)) over the 7 new distances with the lowest share (at least 5
    per scene); mean > 0.
- **Tests:**
  - two-sided one-sample t-tests over scenes;
  - supported if Holm p < 0.05 and the estimate has the hypothesised sign;
  - 95% CIs by scene bootstrap (10,000 draws).

## Secondary analyses
- Q1–Q3 over [40, 50) s;
- Q1 and Q2 over all 17 distances;
- per distance:
  - α (10-s windows and the 1-s curve);
  - motion and open_diff;
  - the share of videos with motion < 1.5 over [50, 60) s ("frozen");
- across distances:
  - Pearson and Spearman correlations of the scene-mean α and motion with the share (scene-bootstrap CIs);
  - the Spearman partial correlation given the distance;
  - a linear fit of the scene-mean α on the share, and the share at which α = 0.5;
- against preregistration 25:
  - per-scene differences in α between the two studies at the 3 old distances;
  - the distances whose mean α exceeds that of distance 20 while their mean motion is at least that of preregistration
    25's sink at its original positions (2.38) (exploratory: is there a distance that keeps more without freezing?).

## Predictions
- **Q1 supported:** mean ρ about +0.6 to +0.85.
- **Q2 supported:** mean ρ about −0.4 to −0.8.
- **Q3 supported, less certainly than Q1:** mean ρ about +0.2 to +0.5. At the lowest shares α may be near 0 at every
  distance (a floor), which would weaken the correlation.
- **Exploratory question:** is there a distance that keeps more without freezing? Distances with shares of 0.10–0.12
  (90, 132, 466) may keep 0.3–0.6 with motion near that of the original positions; whether one meets both conditions
  is about even.
- **Replication:** at the 3 old distances, α within ±0.1 of preregistration 25 (numerical differences between GPU
  architectures may add small deviations).
- **Rules for interpretation:**
  - if Q1 and Q2 are supported: the probe's share quantitatively predicts how much a sink held at a distance keeps
    and how much the video moves, at 14 distances never tested, 9 of them beyond any distance a 60-s video reaches;
    keeping and freezing trade along one axis;
  - if Q1 is supported and Q3 is not: the relation is concentrated at the peaks (threshold-like), with no gradation
    at lower shares;
  - if Q1 is not supported: the 3 distances of preregistration 25 were special and the probe only coarsely orders
    distances; the paper's statement that the share decides how much is kept is withdrawn;
  - if Q2 is not supported while Q1 is: keeping more does not have to cost motion; the per-distance motion is then
    examined.

## Frozen files (SHA-256)
```
a47e3d57688a05989eecf126ba11e01ea28cb223c9dbeacbd07e9de82fc184e0  analyze_s27.py
ea6a93de994e02811cfdc2cd406db6856fd84b6bc255ff2369b008837402f9ce  s27_distances.py
76268591edb98990ffb85aa59aedff89b8522051761b14da54cd04f0e5c47efe  make_s27_tasks.py
ac8d80f617ec532d176d82e33accf34dc147922ddb1522cbeb732df3e48acd77  jobs/s27.jsonl
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
1b14c939ebd86f8efb4f9a78bad8a8bd09ccaca7953aaf42457c02006f9ab7dc  jobs/patch_sf_sinkpos.py
a57bcf3d47b7952b81c80eca9ee045abba1d28c07844fc51a885ae1543a7a824  Self-Forcing-b/wan/modules/causal_model.py (patched)
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
64385a87e26b2110501e89abee7f07196cc093372da13b57f11f5df26cdde923  sinkprobe_long_out.json (exploratory probe result)
```
