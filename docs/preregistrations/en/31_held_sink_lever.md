> English translation of the frozen Chinese preregistration [`../31_held_sink_lever.md`](../31_held_sink_lever.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 31: does a sink held at a probe-chosen distance below the freezing threshold keep more than a re-rotated sink for 180 s, while moving more than the sink at its original positions and without bursts? (Self Forcing, new seed) (2026-10-28, frozen before generation)

## Background
- Preregistration 27 (lab note 34, seed 0, 60 s):
  - with the sink held at one distance, persistence rises with the probe's share (ρ = +0.83) and motion falls with it
    (ρ = −0.81);
  - every distance with a share ≥ 0.110 freezes on the opening; every distance with a share ≤ 0.102 moves;
  - exploratory: distance 66 (share 0.093) keeps 0.41 [0.33, 0.47] over [50, 60) s with motion 4.16; distance 448
    (share 0.068) keeps 0.37 [0.32, 0.42] with motion 2.90.
- References (earlier results):
  - the re-rotated sink (in effect held next to the window, distance about 20): 0.21 at 60 s, 0.12 at 180 s; 0.17 and
    0.16 over the second and third minute in 60-s windows (preregistration 24, seed 0); distance 20 keeps 0.25 with
    motion 4.86 in preregistration 27.
  - the sink at its original positions: 0.64 at 60 s with motion 2.38 (preregistration 25), 0.41 and 0.37 in the second
    and third minute (preregistration 24), but much of what it keeps comes from bursts, and it freezes near the peaks.
- If the distance-66 result is real, it gives a training-free control: pick a distance below the freezing threshold
  with the probe and hold the sink there, to keep about twice the opening that the usual re-rotation keeps, without
  bursts and without freezing.
- But distance 66 is the one that kept most of the 4 distances that "keep more without freezing", chosen after seeing
  the data, so a winner's curse is possible; and the result covers 60 s and one seed. This study tests it with a new
  seed to 180 s, and adds distance 448 as a second distance.
- **Data status:** no video of this study had been generated when it was frozen.

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), generic prompt (`prompts/relax.jsonl`).
- **Model:** Self Forcing DMD, sink 3, window 21, 180 s (723 latent frames, 2889 frames, 16 fps).
- **Generator:** `hold_gen_sinkpos.py` (Self-Forcing-b, as in preregistrations 25 and 27, frozen), which also records
  per-frame motion.
- **Openings:** L\* shifts −16 / +16 (versions m16, p16).
- **Seed:** 4 (not used before on these scenes with Self Forcing).
- **Arms:**
  - sp_d066: sink held at distance 66 (`--sink_mode dist:66`);
  - sp_d448: sink held at distance 448 (`--sink_mode dist:448`);
  - sp_rr: re-rotated sink (`--rerotate`);
  - sp_abs: sink at its original positions (no flag).
- **Execution:**
  - one task per scene × arm (both versions); arch_guard keeps a pair on one GPU architecture;
  - any GPU model (≥ 40 GB, bf16);
  - video files only for rx00 and gx00;
  - task list made by `make_s31_tasks.py` (`jobs/s31.jsonl`, 96 tasks);
  - in total 24 × 4 × 2 = 192 videos.

## Measures
- α[t0, t1) = (end_p16 − end_m16) / (open_p16 − open_m16):
  - open is the mean L\* of frames 1–8;
  - end is the mean L\* over seconds [t0, t1) after the opening;
  - undefined if open_p16 − open_m16 < 4 L\*.
- No collapse exclusion.
- **motion[t0, t1):** the generator's per-frame motion (mean absolute grey difference from the previous frame, 0–100)
  averaged over [t0, t1) s and over the two versions.
- **Burst contrast** (as in preregistration 26): the mean over the 13 predicted onsets τ after 60 s of
  α([τ + 0.5, τ + 2.5)) − α([τ − 2.5, τ − 0.5)).

## Hypotheses (`analyze_s31.py`, frozen; Holm correction over 4 tests)
- **K1 held at 66 keeps more than re-rotated:** α(sp_d066, [120, 180)) − α(sp_rr, [120, 180)) > 0.
- **K2 held at 448 keeps more than re-rotated:** α(sp_d448, [120, 180)) − α(sp_rr, [120, 180)) > 0.
- **K3 held at 66 moves more than the original positions:** motion(sp_d066, [120, 180)) − motion(sp_abs, [120, 180)) > 0.
- **K4 held at 66 does not burst:** the burst contrast of sp_d066 within ±0.05.
- **Tests:**
  - K1–K3: two-sided one-sample t-tests over scenes; supported if Holm p < 0.05 and the estimate > 0;
  - K4: equivalence test (TOST), p = the larger of the two one-sided p values; supported if Holm p < 0.05;
  - 95% CIs by scene bootstrap (10,000 draws); 90% t intervals also reported for the equivalence test.
- **Power:** the between-scene SD of the differences is estimated at 0.15–0.20 (preregistrations 24, 27). If the true
  difference in K1 is 0.15, d ≈ 0.75–1.0, and 24 scenes give power of about 0.85–0.98 at the 0.0125 level.

## Secondary analyses
- K1–K3 over [60, 120) s;
- per arm, α and motion over [0, 60), [60, 120) and [120, 180) s;
- the share of videos with motion < 1.5 over [120, 180) s (frozen);
- sharpness over [120, 180) s relative to [0, 10) s (both versions), to see whether holding the sink blurs the image;
- the burst contrast of every arm;
- motion(sp_d066) − motion(sp_rr) and motion(sp_d448) − motion(sp_abs) over [120, 180) s;
- α of sp_d066 over [50, 60) s, compared with preregistration 27's 0.41 (new seed);
- α (10-s windows) and 1-s curves; collapse counts.

## Predictions
- **K1 supported:** a difference of about +0.15 to +0.25 (distance 66 about 0.3–0.4, re-rotated about 0.15).
  - Uncertainty: distance 66 was selected and may fall back to about 0.3 on a new seed; if the held share keeps
    falling after the first minute, the difference will be smaller.
- **K2 supported:** a difference of about +0.1 to +0.2.
- **K3 supported:** a difference of about +1 to +2 grey levels (4.16 against 2.38 at 60 s).
- **K4 supported:** an estimate of about 0 (the held sink did not burst within 60 s).
- **Rules for interpretation:**
  - if K1, K3 and K4 are all supported: moving the re-rotated sink from next to the window to the probe-chosen distance
    66 keeps about twice the opening for 180 s, without bursts, while the video moves more than with the sink at its
    original positions. The paper reports it as a training-free control ("the probe tells where to put the sink"),
    stating that the distance was chosen on one seed and confirmed on another.
  - if K2 is also supported: the approach is not limited to one distance.
  - if K1 is not supported: the 60-s result for distance 66 did not replicate on a new seed and a longer horizon, and
    the paper does not present it as a control.
  - if K1 is supported but K3 is not: keeping more costs motion, and the paper reports that trade-off as it is.
  - if K4 is not supported: a held sink also jumps at the predicted onsets within 180 s, and the bursts need a new
    explanation.

## Frozen files (SHA-256)
```
d05714f514d9afaafa145e00a580a67ed5335e173722840a272022313cbafcdb  analyze_s31.py
685ed614d1471b003814f87b880d4fb95604f36ddfb8ba45844205b81f07973e  make_s31_tasks.py
8c69b1e75957d6a17391811d92567fd5f5abb15306a429e45f6a84c19556bded  jobs/s31.jsonl
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
1b14c939ebd86f8efb4f9a78bad8a8bd09ccaca7953aaf42457c02006f9ab7dc  jobs/patch_sf_sinkpos.py
a57bcf3d47b7952b81c80eca9ee045abba1d28c07844fc51a885ae1543a7a824  Self-Forcing-b/wan/modules/causal_model.py (patched)
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
- The analysis script was checked on synthetic data (known shares of 0.40, 0.35, 0.15 and 0.40, the motion levels and
  the bursts of the original positions are recovered; missing scenes are handled correctly).
