> English translation of the frozen Chinese preregistration [`../14_prompt_pull_replay.md`](../14_prompt_pull_replay.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 14: measuring the prompt's per-chunk pull w_p directly (2026-10-24, frozen before generation)

## Background
- Chunk recursion: x_{k+1} = w_w·window mean + w_s·sink + w_p·μ, with w_w + w_s + w_p = 1. The equilibrium share kept
  is α∞ = w_s/(w_s + w_p).
- w_s and w_w were measured directly by KV-cache replay (w_s ≈ 0.053; w_w 0.76 for synthetic shifts, 0.90–0.99 for
  natural differences). w_p was never measured; it was the residual 1 − w_w − w_s.
- The fit to the 120-s curve (w_s = 0.060, w_w = 0.86) implies w_p ≈ 0.08. This study measures w_p directly and tests
  that prediction.

## Design
- **Model:** Self Forcing DMD, window 21, sink 3 and no sink; text-to-video rollouts of 31 s under the generic prompt
  (seeds 0–15, 16 videos per sink setting).
- **Intervention:** before chunk k (k = 8, 20, 40, about 6, 15, 30 s), keep the KV cache, RNG state and noise, swap only
  the text conditioning for another prompt B (sunny, overcast, dusk, night; each with its own cross-attention cache), and
  regenerate chunk k; then restore the cache and continue the base video. B = generic must reproduce the base chunk bit
  for bit (confirmed in a smoke test, maximum difference 0).
- **Measure:** mean L\* of the decoded chunk k; ΔL = L_k(B) − L_k(generic).
- **Prompt defaults μ:** the zero-drift brightness levels on Self Forcing from the lighting experiment (lab note 06):
  generic 33.1, sunny 42.9, overcast 33.6, dusk 19.1, night 16.0.
- **Estimate:** per seed and k, the least-squares slope through the origin of ΔL on Δμ = μ_B − μ_generic over the four
  prompts is w_p.
- **Execution:** `pswap.py`, task list `jobs/pswap.jsonl` (4 seeds per task; all swaps of one video in one run, so
  naturally paired).

## Hypotheses (`analyze_pswap.py`, frozen)
- **Primary:** w_p(sink 3, k = 20) > 0 (two-sided one-sample t-test over seeds; p < 0.05 and estimate > 0).
- **Quantitative prediction (secondary):** w_p(sink 3, k = 20) lies in [0.04, 0.14] (0.08 implied by the 120-s fit).
- **Secondary:** w_p for sink 3 and no sink at k = 8, 20, 40; w_p(k = 40) − w_p(k = 8) (time invariance); closure
  w_s + w_w + w_p (w_s = 0.053, w_w = 0.76 or 0.86); ΔL/Δμ per prompt; ΔL for overcast (Δμ ≈ 0) as a placebo; the
  replay check. 95% CIs: seed bootstrap.
- **Predictions:** the primary hypothesis holds; w_p lies in the predicted band.

## Frozen files
See the hash block in the Chinese original.
