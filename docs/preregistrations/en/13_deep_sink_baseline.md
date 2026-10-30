# Preregistration 13: deeper sinks as a baseline for the correction (2026-10-24, frozen before generation)

*English translation; the Chinese file `../13_deep_sink_baseline.md` is the frozen, authoritative version.*

## Background
- The content-aware correction (preregistration 02) has only been compared with post-hoc colour matching. Post-hoc matching by construction erases every global change and is a weak baseline.
- The training-free method proposed by Deep Forcing is a "deeper sink" (keeping more early frames as the sink). By our recursion model, a deeper sink keeps more of the opening (H1 of preregistration 10 tests persistence), so it should also reduce drift. But it acts on all content through attention, and may also suppress legitimate brightness changes.
- This study compares deeper sinks with the correction on the confirmation set of preregistration 02.

## Design
- **Data set:** the confirmation set of preregistration 02: 40 held-out drift prompts × 4 seeds, 8 control prompts with legitimate brightness changes × 4 seeds; Self Forcing DMD, window 21, 31 s.
- **New arms:** sink 6 (s6) and sink 9 (s9), no correction (`correct_gen.py --sink 6|9 --method none`). Compared with the existing no-correction (sink 3) and correction arms.
- **Execution:** as for the original confirmation set, pinned to A100/A800 (these two GPUs give bit-identical outputs), strictly paired with the existing arms (`jobs/deepsink.jsonl`). 48 × 4 × 2 = 384 videos in total.
- **Measures and analysis:** identical to the frozen `analyze_confirm.py` (D, z_L, flicker, share of legitimate changes kept on the control set), with only the arm list extended (`analyze_deepsink.py`).

## Hypotheses (`analyze_deepsink.py`, frozen)
- **Primary hypotheses (Holm correction, 2 tests in total, one-sided):** D(s6) < D(no correction); D(s9) < D(no correction). Decision: Holm p < 0.05 and 95% CI upper bound < 0.
- **Secondary analyses:**
  - share of legitimate brightness changes kept on the control set (the criterion of preregistration 02 is ≥ 0.80);
  - D(s6) − D(correction β=1), D(s9) − D(correction β=1) (two-sided, mixed model on the two arms);
  - z_L; flicker share.
- **Predictions:** both primary hypotheses hold; deeper sinks keep fewer legitimate brightness changes than the correction.

## Frozen files (SHA-256)
```
5a80004d1b767fdc62462d6447575917c43dd5289615bc3063453f642af5be80  analyze_deepsink.py
d079fc471d43c3988012d73cd2cb507281f6e9f4093acf767fe5bb2ae976617b  correct_gen.py
550605c5f50b1f9cc2b2721085ef248d1d7403fa9d3e6565c796d194b5a71bc0  jobs/deepsink.jsonl
dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124  prompts/main.jsonl
31eeca8883bb20785123a9b4bd7affa13a94d5cc8bd0c285641379015b3e3f8d  prompts/controls_confirm.jsonl
```
