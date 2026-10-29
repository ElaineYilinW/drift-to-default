# Preregistration 32: is LongLive's small attractor offset due to its smaller inference window? (Self Forcing with a 12-frame window, 40 prompts, 60 s) (2026-10-29, frozen before generation)

*English translation; the Chinese file `../32_window_matched_offset.md` is the frozen, authoritative version.*

## Background
- Preregistration 30 tests whether fresh rollouts of models trained only on short clips end further from where they start than those of models trained on their own long rollouts (A1: Self Forcing without a sink vs LongLive without a sink).
- The two models differ in their inference settings, though:
  - Self Forcing's key–value window is 21 latent frames;
  - LongLive's is 12.
- So even if A1 holds, the smaller offset could come from the smaller window rather than from LongLive's training on its own long rollouts.
- This study runs Self Forcing DMD with LongLive's window (12 latent frames) on the same 40 prompts and seeds as study 30 and compares it with study 30's LongLive videos.
  - The two models then share architecture (Wan2.1-1.3B), number of denoising steps (4) and window.
  - LongLive was trained further from Self Forcing on its own long rollouts, so the main remaining difference is that training (and the data and pipeline differences that come with it).
- **Data status:**
  - When this was frozen, about half of study 30's videos had been generated, but its results had not been read or analysed. This study is frozen before study 30's results are read.
  - Study 1 (lab note 02) generated 31-s Self Forcing videos with windows 12 and 21 on the same 40 prompts (seeds 0–3, gen.py). Note 02 reported the standardized brightness of the last 2 s relative to fresh 5-s videos, z_L:
    - without a sink: −1.81 with window 12 and −3.55 with window 21;
    - with sink 3: −1.97 and −2.01.
  - These numbers were known at freezing and show that Self Forcing darkens with both windows. They are not start-to-end offsets and were never compared with LongLive; this study does not use them.

## Design
- **Prompts, length, seeds:** as in study 30.
  - The 40 prompts of `prompts/main.jsonl` (20 MovieGen, 20 driving).
  - Plain text-to-video (`--fresh`), 60 s (243 latent frames).
  - Seeds 10, 11, 12.
- **New arms** (generator `hold_gen_long_rr.py`, as study 30's sf_s0 with only the window changed):
  - sf12_s0: Self Forcing DMD, window 12, no sink (`--window 12 --sink 0`);
  - sf12_s3: Self Forcing DMD, window 12, sink 3 at its original positions (`--window 12 --sink 3`), LongLive's default setting.
- **Comparison arms** (study 30's videos, not regenerated):
  - ll_s0: LongLive, no sink;
  - ll_s3: LongLive, sink 3;
  - sf_s0, sf_s3: Self Forcing, window 21.
- **Execution:**
  - One task per prompt × arm (three seeds), 80 tasks (`make_s32_tasks.py`, `jobs/s32.jsonl`).
  - Any GPU with ≥ 40 GB and bf16. Comparisons between models are not video-paired in any case; they are paired by prompt.
  - Videos are written only for drv_dawn_coast and mg0084, seed 10.
  - 40 × 2 × 3 = 240 videos.

## Metric
As in study 30:
- start = mean L\* of frames 1–8;
- end = mean L\* over [50, 60) s after frame 9;
- offset = end − start.
- Per prompt × arm: the mean offset over seeds (at least 2) and its magnitude.

## Hypotheses (`analyze_s32.py`, frozen; Holm over 2 tests)
- **W1, same window, no sink in either:** Self Forcing drifts further than LongLive, |offset(sf12_s0)| − |offset(ll_s0)| > 0.
- **W2, same window, sink 3 at its original positions in both (LongLive's default):** the same, |offset(sf12_s3)| − |offset(ll_s3)| > 0.
- **Test:** paired by prompt, two-sided one-sample t-test. Supported if Holm p < 0.05 and the estimate > 0. 95% CI by prompt bootstrap (10,000 draws).

## Secondary analyses
- The window's own effect in Self Forcing (two-sided, estimates and CIs):
  - |offset(sf12_s0)| − |offset(sf_s0)|;
  - |offset(sf12_s3)| − |offset(sf_s3)|.
- Per arm: mean signed offset, mean magnitude, seed-unbiased root-mean-square offset, mean between-seed SD.
- W1 and W2 in three variants:
  - on the seed-unbiased squared offset;
  - at 31 s ([29, 31) s);
  - per prompt set (MovieGen, driving).
- 1-s curves of the offset.
- Collapse counts.

## Predictions
- **W1 holds:** a difference of about 4–8 L\*. In study 1, Self Forcing with window 12 and no sink also darkened clearly.
- **W2 tends to hold,** with a smaller difference (about 2–5 L\*). Sinks at their original positions periodically pull the picture back to the opening, which shrinks both offsets.
- **The window's own effect:** expected within ±3 L\*.
- **Interpretation rules:**
  - If W1 holds, LongLive's small offset is not explained by its smaller window. The paper will say that "with the same architecture, steps and window, LongLive, trained further on its own long rollouts, has a smaller offset", still noting that this is a comparison of one pair of models.
  - If W1 does not hold:
    - The window may be one cause. The paper's "long-rollout training makes the offset small" becomes "LongLive and Rolling Forcing have small offsets, because of their training or their inference settings", and the training explanation is removed from the abstract.
    - If study 30's A1 holds and W1 does not, both are reported.
  - W2 is supplementary: it compares the two models' default settings.

## Frozen files (SHA-256)
```
65bb7450f54916d1417e94dff521908cba259a59586ac868a8684473ddfb679f  analyze_s32.py
937449383b3bb9f9ddd14ff7063c933e7e44d1e2c2bce01ef1fdfd27e521c8ff  make_s32_tasks.py
30e3cfe47d78c89f024b59838f2ce115e9c144de0db55a7ac84e5a05306462fa  jobs/s32.jsonl
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124  prompts/main.jsonl
```
- The analysis script was checked on synthetic data: known offsets of −10, −6, +2, +1, −8 and −5 are recovered, and prompts with missing seeds are dropped correctly.
