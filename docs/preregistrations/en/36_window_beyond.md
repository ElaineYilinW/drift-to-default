# Preregistration 36: does the attractor offset keep growing with windows beyond the released setting? Where between windows 18 and 21 is Self Forcing's jump? (Self Forcing windows 19, 20, 24, 27, LongLive window 24, 40 prompts, 60 s) (2026-10-30, frozen before generation)

*English translation; the Chinese file `../36_window_beyond.md` is the frozen, authoritative version.*

## Background
- Study 33 (lab note 45): in fresh 60-s rollouts, Self Forcing's offset grows with the key–value window.
  - Per-prompt slope +0.63 L\*/frame [+0.49, +0.77].
  - The mean |off| at windows 6, 9, 12, 15, 18, 21 is 5.2, 6.4, 8.0, 8.4, 9.7, 16.4.
  - Most of the increase lies between 18 and 21: the slope over 6–18 is about 0.37/frame, over 18–21 about 2.2/frame.
  - LongLive increases by +2.1 from window 12 to 21; CausVid does not change.
- Two questions remain:
  - **Where the jump is:** 21 is Self Forcing's released window and also its attention length in training. In the Self Forcing code, without a window setting the attention limit is 32760 tokens, exactly 21 frames. The jump may be concentrated at 20 → 21 (linked to the training length), or it may be a steeper continuous rise between 18 and 21.
  - **Beyond 21:** if the offset is simply "the longer the context, the larger the offset", windows 24 and 27 should give larger offsets; if 21 is a special point (the training length), the offset may stop growing beyond it, or even shrink.
- **Data status:**
  - No video of this study had been generated when it was frozen. The results of studies 30, 32 and 33 are known (the numbers above); this study reuses their videos.
  - A smoke test was run only on prompt mg0030, seed 99, 15 s (60 latent frames) (`jobs/s36_smoke.jsonl`): it checked that Self Forcing with window 27 and LongLive with window 24 run, that the recorded window is correct, and that L\* lies in the normal range (one number for each of the two videos). No 60-s offset was computed.

## Design
- **Prompts, length, seeds:** as in studies 30, 32 and 33.
  - The 40 prompts of `prompts/main.jsonl` (20 MovieGen, 20 driving).
  - Plain text-to-video (`--fresh`), 60 s (243 latent frames).
  - Seeds 10, 11, 12; no sink in any arm.
- **New arms (only the window changed, same generators as study 33):**
  - sf19_s0, sf20_s0, sf24_s0, sf27_s0: Self Forcing DMD, windows 19, 20, 24, 27 (`hold_gen_long_rr.py --window W --sink 0`);
  - ll24_s0: LongLive, window 24 (`ll_hold_gen_long.py --window 24 --sink 0`).
- **Reused arms:**
  - study 33: sf06_s0, sf09_s0, sf15_s0, sf18_s0, ll06_s0, ll21_s0;
  - study 32: sf12_s0;
  - study 30: sf_s0 (window 21), ll_s0 (window 12).
- **Execution:**
  - One task per prompt × arm (three seeds), 200 tasks (`make_s36_tasks.py`, `jobs/s36.jsonl`).
  - Any GPU with ≥ 40 GB and bf16.
  - Videos are written only for drv_dawn_coast and mg0084, seed 10.
  - 40 × 5 × 3 = 600 new videos.

## Metric
As in studies 30, 32 and 33:
- start = mean L\* of frames 1–8;
- end = mean L\* over [50, 60) s after frame 9;
- offset = end − start;
- per prompt × arm: the mean offset over seeds (at least 2) and its magnitude |off|.

## Hypotheses (`analyze_s36.py`, frozen; Holm over 3 tests)
- **X1, Self Forcing, the offset keeps growing beyond the released window:** |off(sf24_s0)| − |off(sf_s0)| > 0.
- **X2, larger still:** |off(sf27_s0)| − |off(sf_s0)| > 0.
- **X3, LongLive, beyond window 21:** |off(ll24_s0)| − |off(ll21_s0)| > 0.
- **Tests:**
  - two-sided one-sample t-test over prompts; supported if Holm p < 0.05 and the estimate > 0;
  - 95% CIs by prompt bootstrap (10,000 draws).
- **Power:** in study 33 the between-prompt SD of window differences was about 4.5 L\*. With 40 prompts, the power is about 0.9 for a difference of +2.5 and about 0.5 for +1.5. X3's expected difference is smaller, and its power may be insufficient.

## Secondary analyses
- **Per arm:** mean signed offset, mean |off|, seed-unbiased root-mean-square offset, between-seed SD, sharpness ratio, absolute frame-to-frame change of mean L\* over [50, 60) s (as in study 33).
- **Self Forcing per window:** mean |off| and signed offset at windows 6, 9, 12, 15, 18, 19, 20, 21, 24, 27.
- **The jump:**
  - |off| differences between neighbouring windows: 19 − 18, 20 − 19, 21 − 20;
  - per prompt, the slope over 18–21 (18, 19, 20, 21) minus the slope over 6–18 (6, 9, 12, 15, 18).
- **Equivalence tests:** 90% t intervals for X1–X3 and equivalence tests at ±2 L\*.
- **Variants of the main tests:** on the seed-unbiased squared offset; at 31 s ([29, 31) s); per prompt set (MovieGen, driving).
- **Other:** per arm, 1-s curves of the signed offset and the plateau time; collapse counts (1-s mean L\* < 3 or > 97).

## Predictions
- **Overall:** uncertain. We have no prior results for windows beyond 21.
- **X1, X2:** tend to hold. If the offset is a monotone function of the context length, window 24 gives about 18–20 and window 27 about 19–23 (differences of +2 to +6).
  - Another possibility: 21 is the training length, and the offset stops growing or falls beyond 21. Then X1 and X2 do not hold, and the equivalence tests may hold.
  - Windows beyond the training length may also degrade image quality or cause collapse; the collapse counts will report this.
- **X3:** a difference of about +0.5 to +2; power is limited, and it may not hold.
- **Shape of the jump:** no prediction.
  - If it is concentrated at 20 → 21: 19 and 20 close to 18 (about 10), 21 − 20 about +6.
  - If it is a continuous rise between 18 and 21: 19 about 12, 20 about 14.
- **Interpretation rules:**
  - **X1 and X2 both hold:** Self Forcing's offset keeps growing with the window beyond the released window. The paper will say "the offset grows with the window, including beyond the released window".
  - **Neither X1 nor X2 holds:** the offset stops growing beyond the released window. The paper says only that the offset grows with the window over 6–21, and reports the equivalence tests.
  - **Only one holds:** both estimates are reported as found, and the offset is not called monotone.
  - **X3:** the estimate is reported whatever the result. If it does not hold, nothing is said about LongLive beyond 21.
  - **Location of the jump:** a secondary analysis, described only, not used as a conclusion.

## Frozen files (SHA-256)
```
8e969e7fd2d29aff741942ddf38efd9d1d4f5684b0fa8d3b0179a3bde72d0e5c  analyze_s36.py
b6c1fbe4ac1078d4b5ed4b38c6f60d7a6a84c44a4b22645ec525651a7746830d  make_s36_tasks.py
c5280a2bb7fc93a654c43c482a4a8f7dc2828a3c89b5146d8e17c1a47e532df6  jobs/s36.jsonl
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
8723343defd6974f6d6f8a7c7e3d971272451882bc15952a4c70427d959a6a7e  ll_hold_gen_long.py
dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124  prompts/main.jsonl
43e64d479ee9d95b92510309ac510b6274214a902eb3f4245ee2a80ed69f3b33  jobs/s36_smoke.jsonl (smoke test only)
```
- The analysis script was checked on synthetic data: with files of existing arms renamed as the new arms (sf24, sf20 = window 21; sf27, sf19 = window 18; ll24 = LongLive window 21), X1 and X3 recover 0, X2 recovers −6.69 (= 9.67 − 16.36), and the neighbouring-window differences and slopes are recovered as set.
