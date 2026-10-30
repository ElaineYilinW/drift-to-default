# Preregistration 14: measuring the prompt's per-chunk pull w_p directly (2026-10-24, frozen before generation)

*English translation; the Chinese file `../14_prompt_pull_replay.md` is the frozen, authoritative version.*

## Background
- Chunk recursion model: x_{k+1} = w_w·window mean + w_s·sink + w_p·μ, with w_w + w_s + w_p = 1. The share kept at equilibrium is α∞ = w_s/(w_s + w_p).
- w_s and w_w have been measured directly with KV-cache replay (w_s ≈ 0.053; w_w 0.76 for synthetic shifts, 0.90–0.99 for natural differences). But w_p has never been measured directly; it is only the residual 1 − w_w − w_s.
- The parameters fitted to the 120-s curve (w_s = 0.060, w_w = 0.86) imply w_p ≈ 0.08. This study measures w_p directly and tests this prediction.

## Design
- **Model:** Self Forcing DMD, window 21, sink 3 and no sink; text-to-video rollouts of 31 s under the generic prompt (seeds 0–15, 16 videos per sink setting).
- **Intervention:** before chunk k (k = 8, 20, 40, i.e. about 6, 15, 30 s), keep the KV cache, random-number state and noise, swap only the text conditioning for another prompt B (sunny, overcast, dusk, night; each with its own cross-attention cache), and regenerate chunk k; then restore the cache and continue the original video. With B = generic prompt the original chunk must be reproduced bit for bit (confirmed in a smoke test, maximum difference 0).
- **Measure:** mean L\* of the decoded chunk k; ΔL = L_k(B) − L_k(generic).
- **Prompt defaults μ:** the zero-drift brightness on Self Forcing from the lighting experiment (lab note 06): generic 33.1, sunny 42.9, overcast 33.6, dusk 19.1, night 16.0.
- **Estimate:** for each seed and k, least squares through the origin of ΔL on Δμ = μ_B − μ_generic (4 prompts); the slope is w_p.
- **Execution:** `pswap.py`, task list `jobs/pswap.jsonl` (4 seeds per task; all swaps of one video are in the same run, so they are naturally paired).

## Hypotheses (`analyze_pswap.py`, frozen)
- **Primary hypothesis:** w_p(sink 3, k = 20) > 0 (two-sided one-sample t-test over seeds, p < 0.05 and estimate > 0).
- **Quantitative prediction (secondary):** w_p(sink 3, k = 20) lies within [0.04, 0.14] (0.08 as implied by the 120-s fit).
- **Secondary analyses:** w_p for sink 3 and no sink at k = 8, 20, 40; w_p(k = 40) − w_p(k = 8) (time invariance); closure check w_s + w_w + w_p (w_s = 0.053, w_w = 0.76 or 0.86); ΔL/Δμ per prompt; ΔL for overcast (Δμ ≈ 0) as a placebo; the replay check. 95% CI: seed bootstrap.
- **Predictions:** the primary hypothesis holds; w_p lies within the predicted interval.

## Frozen files (SHA-256)
```
58d27e95539b7551a2ff39db4d4cf7a0af47d6b18535ff40449d2ad7b2b33b9c  analyze_pswap.py
510c704afbedd102116fd36bb69168781b0933b10396f6255f3001fafd611f85  pswap.py
a160cb096a2ba03b72331a0409764b12b9c61956811fc35bf291ab4f723af4a2  prompts/pswap.jsonl
ebd90936d448f024a0129fc0427bbd363e16b22c4dc51ad057a097d3b5cce6c6  jobs/pswap.jsonl
```
