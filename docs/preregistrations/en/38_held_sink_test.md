> English translation of the frozen Chinese preregistration [`../38_held_sink_test.md`](../38_held_sink_test.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 38: does the held distance chosen by the rule keep more than the re-aligned sink, without bursts and without moving less, on new openings, a new seed and 4 models? (2026-11-02, frozen before generation)

## Background
- Study 37 chooses each model's held sink distance D\* with one fixed rule: Self Forcing 66, LongLive 448, CausVid and
  GAN set by the new calibration.
- The only earlier confirmation covers Self Forcing, distance 66 and the original 24 scenes (preregistration 31). This
  study switches to 16 openings never used in a sink study and a new seed, and applies the same approach to 4 models.
  The comparison is the re-aligned sink (directly before the window) used by Rolling Forcing and Deep Forcing.
- **Data status:** no video of this study had been generated when it was frozen.

## Design
- **Openings:** the 16 of `prompts/heldtest.jsonl`: real NAVSIM frames nr01, nr04, nr06, nr09, nr11, nr14, nr16, nr18
  (one taken about every 2.5 in brightness order, L\* about 22–58) and generated frames ng00–ng07. They were used in
  preregistrations 03–05 but never in a sink study.
- **Prompt:** generic prompt (as in preregistration 06).
- **Opening shifts:** L\* −16 / +16 (m16, p16). **Seed:** 5. **Duration:** 120 s (483 latent frames).
- **Models:** Self Forcing DMD, CausVid, Self Forcing GAN (`hold_gen_sinkpos.py`, Self-Forcing-b, window 21), LongLive
  (`ll_hold_gen_sinkpos.py`, LongLive-c, window 12); sink 3. Models without a D\* do not take part.
- **Arms (per model):**
  - held: sink held at D\* (`--sink_mode dist:D*`);
  - rr: sink held at the re-aligned distance (window − 1: 20, LongLive 11);
  - abs: sink at its original positions.
- **Execution:** one task per model × opening, running the 3 arms and both versions (arch_guard keeps them on one GPU
  architecture). Video files only for nr01 and ng00. Task lists are made by `make_s38_tasks.py` from study 37's
  `dstar.json`: for Self Forcing and LongLive now (`jobs/s38_sf_ll.jsonl`, 32 tasks), for CausVid and GAN after study
  37's calibration is complete (`jobs/s38_cv_gan.jsonl`).

## Measures
- α, motion and burst contrast are defined as in preregistration 31. The primary window is [60, 120) s.
- Burst onsets: the peaks P with prominence ≥ 0.04 on each model's own probe curve (the rule of
  `analyze_sinkprobe_long.py`), onset (P − 5) / 4 s, keeping those between 62.5 and 117.5 s. Self Forcing uses
  `sinkprobe_long_out.json`, LongLive uses `llprobe_out.json`, CausVid and GAN use the probes of study 37.

## Hypotheses (`analyze_s38.py`, frozen; Holm correction over all participating models together, at most 12 tests)
- **H1_m held keeps more than rr:** α(held, [60, 120)) − α(rr, [60, 120)) > 0.
- **H2_m held does not burst:** the burst contrast of held at the model's burst onsets is within ±0.05 (TOST). If the
  model has no onset in 62.5–117.5 s, H2_m is not tested.
- **H3_m held moves no less than rr (non-inferiority, margin 0.5 grey levels):** motion(held) − motion(rr) + 0.5 > 0,
  over [60, 120) s.
- **Tests:** openings are the unit; t-tests are two-sided, supported if Holm p < 0.05 and the estimate > 0; TOST takes
  the larger of the two one-sided p values, supported if Holm p < 0.05; 95% CIs by opening bootstrap (10,000 draws).
- **Power:** in preregistration 31 the Self Forcing difference was +0.23 with a between-scene SD of about 0.18; with 16
  openings d ≈ 1.3, giving power of about 0.95 at the 0.004 level.

## Secondary analyses
- per model and arm: α over [0, 60) and [60, 120) s and in 10-s windows, 1-s curves, motion, frozen share, sharpness
  ratio, and the burst contrast of every arm;
- held − abs for α and motion; collapse counts.

## Predictions
- **Self Forcing:** H1 supported, a difference of about +0.2 (held about 0.4, rr about 0.2); H2 and H3 supported.
- **LongLive:** H1 supported, a difference of about +0.3 (448 about 0.35, distance 11 about 0.05); H2 and H3 supported.
- **CausVid:** H1 may be supported but with a small difference (about +0.1); the burst onsets may be weak.
- **GAN:** H1 supported, a difference of about +0.2–0.3.
- **Rules for interpretation:**
  - if H1 is supported for most of the 4 models, and H2 and H3 are also supported: the probe plus a little calibration
    finds, for each model, a sink position that keeps more than re-alignment, without bursts and without freezing, and
    without training. The paper reports it as a method.
  - if it holds only on Self Forcing: the rule does not generalize, and the paper states plainly that it is
    "model-specific".
  - if H3 is not supported: keeping more costs motion, and that trade-off is reported.

## Frozen files (SHA-256)
```
1008f82648f96079bf8b9d87024b2b481d5defff34149b41e49d4a6372d9ed82  analyze_s38.py
63eed4085473f8e02e14d19904504c6a9bd99ef16ca08f1ede529b541cd2f839  make_s38_tasks.py
5aded6f546a31a3d7c26ef49fbce8e9b2e0dfe78e4b65ee5dee1c4eb1d25580c  select37.py
cbc24d57e382b4cf1a7586024aa04d2eba0c133a095a853c9a01bd955c61df5d  prompts/heldtest.jsonl
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
073f21f1476aea62a192de6aa71b00159c2af78a44bfa2304f025b7ed65821d7  ll_hold_gen_sinkpos.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
fee24ef4bccf0bd7bc1ab277fcf15ed74e22c13344627ad1b77abfe068729873  jobs/dstar_sfll.json
0cfe007d7c956b621aeeafbd122ce89bc15c1d36a6507bbc3196d290e357b7c3  jobs/s38_sf_ll.jsonl
```
- The analysis script was checked on synthetic data: with held 0.45, rr 0.22 and abs 0.43 set, H1 recovers +0.23, H2
  recovers about 0, and H3 recovers about +0.5.
