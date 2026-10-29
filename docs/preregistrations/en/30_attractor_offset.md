> English translation of the frozen Chinese preregistration [`../30_attractor_offset.md`](../30_attractor_offset.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 30: do fresh rollouts of models trained only on short clips end further from where they start than those of models trained on their own long rollouts? (40 prompts, 60 s) (2026-10-28, frozen before generation)

## Background
- The abstract and the contributions state an exploratory finding: drift separates into forgetting the opening and an
  "attractor offset" (how far the default itself lies from where the same prompt starts). Models trained or tuned on
  their own long rollouts (LongLive, Rolling Forcing) have almost no offset; models trained only on 5–10-s clips have
  4–11 L\*.
- The evidence is thin (lab note 18, exploratory):
  - a single prompt (the generic city-driving prompt), 4 seeds, 31 s;
  - Self Forcing DMD (sink 3) −6.0, no sink −10.1; CausVid (sink 3) +11.2; LongLive (sink 3) +0.7, no sink +2.3;
    Rolling Forcing +1.0.
  - The fresh rollouts of preregistration 24 (same prompt, 4 seeds, 60 s) point the same way: Self Forcing without a
    sink −12.7, re-rotated sink −8.6, Rolling Forcing −0.5, LongLive (sink 3) −3.1.
- This study tests the finding on 40 prompts and 3 new seeds.
- **Data status:**
  - no video of this study had been generated when it was frozen.
  - The 40 prompts were used before: the correction experiments (studies 1–3) generated 31-s Self Forcing (sink 3)
    and LongLive (sink 3) videos from them (other generators and seeds) to measure drift with and without the
    correction. Those data were never compared between models as fresh-rollout offsets; this study does not use them,
    and they were not inspected for this question before freezing.

## Design
- **Prompts:** the 40 prompts of `prompts/main.jsonl` (20 MovieGen, 20 driving).
- **Generation:** plain text-to-video (`--fresh`, no opening image), 60 s (243 latent frames, 969 frames, 16 fps).
- **Seeds:** 10, 11, 12 (never used before).
- **Arms (frozen generators, as in preregistrations 24 and 29):**
  - sf_s0: Self Forcing DMD, no sink (`hold_gen_long_rr.py --sink 0`);
  - sf_rr: Self Forcing DMD, sink 3 re-rotated (`--sink 3 --rerotate`);
  - cv_s0: CausVid, no sink (same generator, the CausVid checkpoint and sampler settings of the earlier CausVid
    experiments);
  - ll_s0: LongLive, no sink (`ll_hold_gen_long.py --sink 0`);
  - rf: Rolling Forcing (`rf_gen_long.py`);
  - secondary: sf_s3 (Self Forcing, sink 3 at its original positions) and ll_s3 (LongLive, sink 3 at its original
    positions).
- **Why the primary comparisons avoid sinks at their original positions:** such a sink periodically bursts and copies
  the opening (lab notes 31, 32), which at some times pulls the "end" back to the start and mixes with the attractor
  offset. The primary comparisons therefore use only configurations without bursts.
- **Execution:**
  - one task per prompt × arm (3 seeds), 280 tasks (`make_s30_tasks.py`, `jobs/s30.jsonl`);
  - any GPU model (≥ 40 GB, bf16);
  - video files only for drv_dawn_coast and mg0084, seed 10;
  - in total 40 × 7 × 3 = 840 videos.

## Measures
- Per video:
  - start = mean L\* of frames 1–8 (the first chunk);
  - end = mean L\* over [50, 60) s after frame 9 (frames 809–968);
  - offset = end − start.
- Per prompt × arm: the mean offset over seeds (at least 2 seeds) and its absolute value |offset|.

## Hypotheses (`analyze_s30.py`, frozen; Holm correction over 3 tests)
- **A1 Self Forcing drifts further than LongLive (both without a sink):** |offset(sf_s0)| − |offset(ll_s0)| > 0.
- **A2 CausVid drifts further than LongLive (both without a sink):** |offset(cv_s0)| − |offset(ll_s0)| > 0.
- **A3 Self Forcing (re-rotated sink) drifts further than Rolling Forcing:** |offset(sf_rr)| − |offset(rf)| > 0.
- **Tests:** paired by prompt (prompts with values in both arms), two-sided one-sample t-tests; supported if Holm
  p < 0.05 and the estimate > 0; 95% CIs by prompt bootstrap (10,000 draws).
- **Notes:**
  - A1 and A2 are the cleanest comparisons: none of the three arms has a sink, and all forget the opening (LongLive
    without a sink has α = 0.02 at 31 s).
  - In A3 both arms keep the opening in the context, but different shares of it (Rolling Forcing about 0.3, the
    re-rotated sink about 0.16), so Rolling Forcing's offset is mechanically smaller for that reason (by about 15%).
    A3 supports the finding but is less clean than A1 and A2.
- **Power:** the SD of the differences across prompts is unknown. With a mean difference of 5 L\* and an SD of 5 L\*
  (d = 1), 40 prompts give power > 0.99 at the 0.017 level; with a mean difference of only 2 L\* (d = 0.4), about 0.6.

## Secondary analyses
- Per arm:
  - the mean signed offset and the mean |offset|;
  - the root-mean-square offset without seed noise: per prompt the mean over seed pairs of the product of their
    offsets (whose expectation for independent seeds is the squared true offset), averaged over prompts, square root
    of the positive part;
  - the mean between-seed SD.
- A1–A3 on the seed-unbiased squared offset;
- A1–A3 at 31 s (end over [29, 31) s, the definition of lab note 18);
- A1–A3 separately on the MovieGen and the driving prompts;
- sf_s3 against ll_s3 (sinks at their original positions, the two models' default settings);
- for each pair, the share of prompts where the first arm's |offset| is larger;
- 1-s curves of the offset;
- collapse counts (videos with a 1-s mean L\* below 3 or above 97).

## Predictions
- **A1, A2 supported:** differences of about 4–8 L\*.
- **A3 likely supported,** a difference of about 3–7 L\*.
- Uncertainties:
  - the MovieGen prompts have much camera motion and scene change (push-ins, pull-outs), which changes brightness by
    itself and adds noise to every arm;
  - with one prompt Self Forcing darkens and CausVid brightens; over 40 prompts the direction may vary with the
    prompt, so the tests use absolute values.
- **Rules for interpretation:**
  - if A1 and A2 are both supported: over 40 prompts, fresh rollouts of the models trained only on short clips end
    further from where they start than LongLive's; the abstract's "the offset is small in the models trained on their
    own long rollouts" stays (stated as LongLive's result, with the number of prompts). If A3 is also supported,
    Rolling Forcing stays as well.
  - if A1 or A2 is not supported: the two-part decomposition of drift still holds (it is a definition), but "training
    on long rollouts makes the offset small" is removed from the abstract or restricted to one class of prompts.
  - if A3 is not supported while A1 and A2 are: the finding holds for LongLive only, and the Rolling Forcing claim is
    removed.
  - whatever the result, the paper states that these models differ in more than training length (data, distillation,
    window), so the finding is correlational.

## Frozen files (SHA-256)
```
8c48f2f67f2db223db9f6e14c742bc6cc627f24367adc478926da9af8b259317  analyze_s30.py
e7cd9901615527e5855502b0c9e3d38c719d22f1acffe3c36da5b2d8b6bb08df  make_s30_tasks.py
6d586aeb19f779f0d431a55915dee13bc7675a83d00ecf010992a4a8548dc94c  jobs/s30.jsonl
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
8723343defd6974f6d6f8a7c7e3d971272451882bc15952a4c70427d959a6a7e  ll_hold_gen_long.py
9b58613c2ed1f06862b9f9b5290e5bfd9a52a9c6cc147fc58409d26628bb2599  rf_gen_long.py
dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124  prompts/main.jsonl
```
- The analysis script was checked on synthetic data (known offsets of −10, −8, +10, +2 and +1 are recovered; missing
  seeds and prompts are handled correctly).
