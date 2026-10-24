> English translation of the frozen Chinese preregistration [`../13_deep_sink_baseline.md`](../13_deep_sink_baseline.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 13: deeper sinks as a baseline for the correction (2026-10-24, frozen before generation)

## Background
- The content-aware correction (study 02) was compared only with post-hoc colour matching, which by construction erases
  every global change and is a weak baseline.
- Deep Forcing proposes a training-free "deeper sink" (keeping more early frames as the sink). By our recursion a
  deeper sink keeps more of the opening (study 10, H1, tests persistence), so it should also reduce drift. But it acts on
  all content through attention and may also suppress legitimate brightness changes.
- This study compares deeper sinks with the correction on the confirmation set of study 02.

## Design
- **Data:** the confirmation set of study 02: 40 held-out drift prompts × 4 seeds and 8 control prompts with legitimate
  brightness changes × 4 seeds; Self Forcing DMD, window 21, 31 s.
- **New arms:** sink 6 (s6) and sink 9 (s9), no correction (`correct_gen.py --sink 6|9 --method none`), compared with
  the existing no-correction (sink 3) and correction arms.
- **Execution:** as in the original confirmation set, pinned to A100/A800 (bit-identical outputs), strictly paired with
  the existing arms (`jobs/deepsink.jsonl`). 48 × 4 × 2 = 384 videos.
- **Measures and analysis:** identical to the frozen `analyze_confirm.py` (D, z_L, flicker, share of legitimate
  transitions kept on the control set), with the arm list extended (`analyze_deepsink.py`).

## Hypotheses (`analyze_deepsink.py`, frozen)
- **Primary (Holm over 2 tests, one-sided):** D(s6) < D(no correction); D(s9) < D(no correction). Supported if Holm
  p < 0.05 and the 95% CI upper bound < 0.
- **Secondary:** share of legitimate transitions kept on the control set (study 02 criterion ≥ 0.80);
  D(s6) − D(correction β = 1) and D(s9) − D(correction β = 1) (two-sided, mixed model on the two arms); z_L; flicker.
- **Predictions:** both primary hypotheses hold; deeper sinks keep fewer legitimate transitions than the correction.

## Frozen files
See the hash block in the Chinese original.
