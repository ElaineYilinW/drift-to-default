# Preregistration 33: does the attractor offset grow with the key–value window? (Self Forcing with 6 windows, LongLive and CausVid with 2–3 windows each, 40 prompts, 60 s) (2026-10-29, frozen before generation)

*English translation; the Chinese file `../33_window_dose.md` is the frozen, authoritative version.*

## Background
- Preregistration 30 (lab note 37): on 40 prompts, in fresh 60-s rollouts, Self Forcing (window 21, no sink) ends 16.4 L\* from where it starts, CausVid (window 21) 11.7, LongLive (window 12) 6.8.
- Preregistration 32 (lab note 41): with LongLive's 12-frame window, Self Forcing's offset falls to 8.0, not significantly different from LongLive; the window's own effect is −8.3 [−10.1, −6.5]. This is the opposite of the prediction (within ±3).
- So the paper now says "the offset depends on the inference settings at least as much as on training". But this is one model and two windows:
  - it is not known whether the offset changes monotonically with the window (dose-response) or jumps somewhere between 12 and 21;
  - it is not known whether the effect is specific to Self Forcing. LongLive and CausVid have the same architecture (Wan2.1-1.3B); if the window is the cause, their offsets should also change with the window.
- **Data status:**
  - No video of this study had been generated when it was frozen.
  - The results of studies 30 and 32 are known (the numbers above). This study reuses their sf12_s0, sf_s0, ll_s0 and cv_s0 videos (same prompts and seeds), so Self Forcing's values at windows 12 and 21 are known. The new information is the intermediate windows (6, 9, 15, 18) and the other two models.

## Design
- **Prompts, length, seeds:** as in studies 30 and 32.
  - The 40 prompts of `prompts/main.jsonl` (20 MovieGen, 20 driving).
  - Plain text-to-video (`--fresh`), 60 s (243 latent frames).
  - Seeds 10, 11, 12; no sink in any arm.
- **New arms (only the window changed, same generators as study 30):**
  - sf06_s0, sf09_s0, sf15_s0, sf18_s0: Self Forcing DMD, windows 6, 9, 15, 18 (`hold_gen_long_rr.py --window W --sink 0`);
  - ll06_s0, ll21_s0: LongLive, windows 6, 21 (`ll_hold_gen_long.py --window W --sink 0`);
  - cv12_s0: CausVid, window 12 (`hold_gen_long_rr.py --window 12 --sink 0`, with CausVid's checkpoint and sampler settings, as study 30's cv_s0).
- **Reused arms (not regenerated):**
  - sf12_s0 (study 32);
  - sf_s0 (window 21), ll_s0 (window 12), cv_s0 (window 21) (study 30).
- **Execution:**
  - One task per prompt × arm (three seeds), 280 tasks (`make_s33_tasks.py`, `jobs/s33.jsonl`).
  - Any GPU with ≥ 40 GB and bf16.
  - Videos are written only for drv_dawn_coast and mg0084, seed 10.
  - 40 × 7 × 3 = 840 new videos.

## Metric
As in studies 30 and 32:
- start = mean L\* of frames 1–8;
- end = mean L\* over [50, 60) s after frame 9;
- offset = end − start;
- per prompt × arm: the mean offset over seeds (at least 2) and its magnitude |off|.

## Hypotheses (`analyze_s33.py`, frozen; Holm over 3 tests)
- **V1, Self Forcing's offset grows with the window:** the per-prompt least-squares slope of |off| on the window (6, 9, 12, 15, 18, 21) > 0 (only prompts with values at all 6 windows).
- **V2, LongLive likewise:** |off(ll21_s0)| − |off(ll_s0)| > 0 (window 21 vs 12).
- **V3, CausVid likewise:** |off(cv_s0)| − |off(cv12_s0)| > 0 (window 21 vs 12).
- **Test:** two-sided one-sample t-test over prompts. Supported if Holm p < 0.05 and the estimate > 0. 95% CI by prompt bootstrap (10,000 draws).

## Secondary analyses
- Per arm: mean signed offset, mean |off|, seed-unbiased root-mean-square offset, mean between-seed SD.
- Self Forcing per window: mean |off| and signed offset; per-prompt Spearman correlation of |off| with the window (mean and CI); slope of the signed offset.
- Window 6: |off(ll06_s0)| − |off(ll_s0)|, |off(sf06_s0)| − |off(sf12_s0)|.
- V1–V3 in three variants: on the seed-unbiased squared offset; at 31 s ([29, 31) s); per prompt set (MovieGen, driving).
- Per arm: 1-s curves of the signed offset, and the plateau time (the first T at which the magnitude of the mean offset over [T − 5, T) s is ≥ 0.9 times that over [50, 60) s).
- Rough image-quality checks per arm: the sharpness ratio of [50, 60) s to [0, 10) s; the absolute frame-to-frame change of mean L\* over [50, 60) s.
- Collapse counts (1-s mean L\* < 3 or > 97), and V1 after dropping the prompts with a collapsed Self Forcing video.

## Predictions
- **V1 holds:** window 12 is known to give 8.0 and window 21 16.4. The intermediate windows are expected to be roughly monotone: 6 about 5, 9 about 6–7, 15 about 10–12, 18 about 13–14; a slope of about +0.5 to +0.8 L\*/frame.
  - Uncertainties: window 6 leaves only 3 frames of context, which may degrade image quality or change behaviour; the change between 12 and 21 may also be a jump rather than smooth.
- **V2 tends to hold:** a difference of about +2 to +6 L\*. LongLive has the same architecture as Self Forcing; if the window is the cause, it should also drift further with a larger window. But LongLive was trained with window 12, and window 21 lies outside its training setting, which by itself may also enlarge the offset. So V2 holding alone does not show that the window's effect is independent of training.
- **V3 holds:** a difference of about +3 to +6 L\* (CausVid's |off| about 6–8 at window 12).
- **Interpretation rules:**
  - If V1, V2 and V3 all hold: the offset grows with the window in all three Wan models. The paper will say "the offset grows with the key–value window (three models, preregistered)" and report a shorter window as an inference setting that reduces the offset, together with its cost in sharpness and change.
  - If only V1 holds: the dose-response is shown only for Self Forcing, and the paper does not call it general.
  - If V1 does not hold: the difference between windows 12 and 21 is not a smooth dose-response, and the paper reports only those two windows.
  - If V2 does not hold and V3 does (or the reverse): the paper reports as found which model changes with the window.

## Frozen files (SHA-256)
```
90e77188f82cc4911ffa80fd343fb5487e86faacba9aec16be48b02f84f902ca  analyze_s33.py
41bb8e3ac178a41adee272223e0602ab770f1b312c42f1a93487ad63f5989504  make_s33_tasks.py
ddcc642631fffd59e44d167ba9c8435ca2da8b41cf9ad4bd27b366c68b659221  jobs/s33.jsonl
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
8723343defd6974f6d6f8a7c7e3d971272451882bc15952a4c70427d959a6a7e  ll_hold_gen_long.py
dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124  prompts/main.jsonl
```
- The analysis script was checked on synthetic data: a known slope of 0.67, a LongLive difference of 4 and a CausVid difference of 5 are recovered; prompts with missing seeds are dropped correctly; the plateau time is correct.
