> English translation of the frozen Chinese preregistration [`../37_held_sink_rule.md`](../37_held_sink_rule.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 37: choosing each model's held sink distance with one fixed rule (Self Forcing and LongLive from existing data; new calibration for CausVid and Self Forcing GAN) (2026-11-02, frozen before generation)

## Background
- Preregistrations 27 and 31: with Self Forcing DMD's sink held at distance 66, 0.45 is kept in the third minute,
  against only 0.22 for the re-rotated sink, with no bursts and no freezing.
- Preregistration 28: the same distance 66 keeps only 0.17 on LongLive; LongLive keeps 0.36 at distance 448, with 4%
  frozen. So the distance does not transfer directly and has to be chosen per model.
- This study writes "choosing the distance" as a rule fixed in advance and applies it to 4 models. Study 38 tests the
  distances the rule chooses on new scenes and a new seed; study 39 evaluates them with VBench-style measures on 60-s
  videos generated without a given opening.
- **Data status:** no calibration video of CausVid or GAN had been generated when this study was frozen. Attention
  probes of the two models (exploratory, same settings as `jobs/run_sinkprobe_long.sh`: rx02, rx03, gx02, gx03, 6
  chunks, distances 3–723) are already running to produce the candidate distances; they had no results at freezing.
  The calibration data of Self Forcing and LongLive are the results of preregistrations 27 and 28, which are already
  public.

## Rule (`select37.py`, frozen)
- Among the calibrated distances D > 20, take the one with the largest mean α over [50, 60) s and a frozen share
  ≤ 0.07; call it D\*.
  - α as before: (end_p16 − end_m16) / (open_p16 − open_m16), undefined if open_p16 − open_m16 < 4 L\*; averaged over
    scenes.
  - Frozen: a video's mean motion over [50, 60) s is < 1.5 grey levels; both versions are counted.
  - Ties go to the smaller D. If no distance meets the conditions, the model has no D\* and does not enter studies 38
    and 39.
- **Re-aligned distance** (the sink placed directly before the window) = window − 1: 20 for Self Forcing, CausVid and
  GAN, 11 for LongLive.

## Calibration data
- **Self Forcing DMD:** the 17 distances of preregistration 27, 24 scenes, seed 0
  (`results/corr/s27_out/s27_results.json`).
- **LongLive:** the 17 distances of preregistration 28, 24 scenes, seed 0 (`results/corr/s28_out/s28_results.json`).
- **CausVid (autoregressive_checkpoint, 3 steps 1000,757,522, shift 8, no warp) and Self Forcing GAN:** new.
  - Candidate distances: the rule of preregistration 27 (`s27_distances.select`, K = 14) applied to the mean curve of
    each model's own probe, plus distance 20 as a reference.
  - Calibration scenes: rx00–rx03, gx00–gx03 (8 of the 24 scenes of preregistration 06), L\* −16 / +16, generic prompt,
    seed 0, 60 s (243 latent frames).
  - Sink 3, window 21, `hold_gen_sinkpos.py` (Self-Forcing-b, frozen).
  - One task per model × scene, running all distances (arch_guard), no video files. The task list is made by
    `make_s37_tasks.py` once the probe results are in (`jobs/s37.jsonl`, 16 tasks).

## Known results (the rule applied to existing data)
- Self Forcing: D\* = 66 (α 0.405, frozen 0). Distance 90 (0.364) is 25% frozen and 448 (0.37) 12% frozen; both are
  excluded.
- LongLive: D\* = 448 (α 0.356, frozen 4%). Distance 90 (0.315) is 12% frozen and is excluded; next is 147 (0.287).
- These two results could be seen before the rule was written, so they only show what the rule chooses and are not a
  test. The test is study 38.

## Predictions (CausVid, GAN)
- Both models have the same architecture as Self Forcing, so their probe curves should correlate with Self Forcing's
  (0.94 for LongLive), with peaks at the same or nearby positions.
- GAN keeps the most at 60 s (0.79 with sink 3 at its original positions); D\* is expected at a distance with a medium
  share (about 60–100), with α about 0.5.
- CausVid's sink effect is weaker (0.23 at 60 s against 0.04 without a sink); its D\* is expected to have α of about
  0.2–0.3. It is also possible that no distance both keeps much and does not freeze; in that case this is reported as
  it is.

## Frozen files (SHA-256)
```
5aded6f546a31a3d7c26ef49fbce8e9b2e0dfe78e4b65ee5dee1c4eb1d25580c  select37.py
73e081bcd6ba89d3dd17848aaf34b053bfaf72cfb79202151b2139dc9e91366f  make_s37_tasks.py
ea6a93de994e02811cfdc2cd406db6856fd84b6bc255ff2369b008837402f9ce  s27_distances.py
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
1d2d8dbece97de8b2a54af51d229db8aec41269707946387d14bda324cee8957  jobs/p37_probe.jsonl
```
- The rule has been applied to the results of preregistrations 27 and 28 (`jobs/dstar_sfll.json`): Self Forcing 66,
  LongLive 448.
