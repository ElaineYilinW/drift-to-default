> English translation of the frozen Chinese preregistration [`../02_correction.md`](../02_correction.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration: content-referenced darkening correction (confirmatory experiment, 2026-10-19)

Written before generation of the confirmatory experiment. Method, parameters and decision criteria **all come from the development set** (the 10 pilot prompts + 6 development controls, see `docs/correction_dev_cn.md`). The confirmatory experiment uses the 40 main prompts, **which were not used in development**, and 8 newly written controls. The analysis script `analyze_confirm.py` is frozen together with this file; the SHA-256 hashes are recorded at the end.

## 1. Method
- Base setup: Self Forcing DMD (Wan2.1-T2V-1.3B), window of 21 latent frames + 3 sink frames (official sink implementation, positional encoding not re-aligned), 501 frames.
- Correction (`correct_gen.py --method ttc --at_step 1`): starting from the 2nd chunk, in each chunk at denoising step 1 (input timestep 937.5), run one more forward pass on the same noisy input, using a temporary KV cache that contains only the sink frames. This gives x0ᶜ. For each latent frame, take the difference in the spatial mean of each of the 16 channels between x0ᶜ and the actual prediction x0, multiply it by β and add it to x0. Then add noise as usual and go to the next step. No training. The main KV cache and the random number sequence are unchanged.
- Three comparison groups: `none` (no correction), `ttc_s1` (β = 1), `ttc_s1_b15` (β = 1.5).
- Pairing: all groups and all seeds of the same prompt are generated on the same GPU architecture (A100/A800, verified to be bitwise identical). So before the first correction, the frames of all groups are exactly the same.

## 2. Data
- **Drift set**: the 40 prompts in `prompts/main.jsonl` (20 MovieGen + 20 driving), seeds 0–3. The reference distribution reuses the reference videos of the main experiment (seeds 100–109).
- **Control set**: the 8 newly written brightness-transition prompts in `prompts/controls_confirm.jsonl` (5 darkening, 3 brightening), seeds 0–3.

## 3. Metrics (endpoint is seconds 29–31)
- **D**: color drift, same definition as in the main experiment.
- **K**: DINOv2 distribution drift, same definition as in the main experiment.
- **z_L**: signed brightness z-score.
- **Flicker**: the SD of the frame-by-frame second difference of brightness (jitter), and the mean brightness jump at chunk boundaries. Reported as a ratio relative to the uncorrected group; descriptive only.
- **Control retention ratio**: (first-to-last brightness difference in the corrected group) ÷ (first-to-last brightness difference in the uncorrected group). First take the median over seeds within each prompt, then the median over controls. **If a control's mean |first-to-last brightness difference| in the uncorrected group is < 5 L, the model did not produce the transition at all, and this control is excluded** (rule taken from the "dawn" control in the development phase).

## 4. Tests
Model: y ~ group + (1 | prompt) + (1 | prompt:seed), REML. Each correction group is compared with `none` separately.

- **Primary hypothesis family** (Holm correction, 2 tests):
  - H1a: `ttc_s1` lowers D;
  - H1b: `ttc_s1_b15` lowers D.
  - One-sided tests. A "supported" verdict requires Holm-corrected p < 0.05 **and** a 95% CI upper bound < 0.
- **Secondary hypothesis family** (Holm correction, 2 tests):
  - H2a, H2b: z_L increases (darkening is reduced), one-sided, same decision rule as above.
- **Structural non-inferiority** (no hypothesis test): if the upper bound of the 95% CI of ΔK < +0.30, the correction is judged to "not harm structure". +0.30 is about 12% of the K value of the uncorrected group in the development set.
- **Retention of normal change** (descriptive verdict): on the valid controls, if the median retention ratio ≥ 0.80, the correction is judged to "largely preserve normal change".
- **Flicker**: descriptive only, no verdict. In the development set, with β = 1.5 the chunk-boundary jumps were about 3.3×. This is a known cost.
- **Robustness**: CIs from a prompt-clustered bootstrap; driving and MovieGen reported separately.

## 5. Deviations
Any changes are listed in the final report with the reasons.

## Freeze record
- Freeze time: 2026-10-19 00:28 (local time). The confirmatory experiment had not yet been generated.
- `docs/preregistration_correction_cn.md` sha256 c98d2dd50f1b543471992af7898e028da028dc247a02e95ec6ed3155d815fa47 (content before this line was appended)
- `analyze_confirm.py` sha256 a5363d5c9b3cd484670f27023a7daf5e21c153e95edaf36db5a4500ba3a9202e
- `correct_gen.py` sha256 d079fc471d43c3988012d73cd2cb507281f6e9f4093acf767fe5bb2ae976617b
- `features.py` sha256 f6c15dba0bc97f0ba39b4a75c4c07c5f7f489eb9172d065f1f83d50434a95454
- `stats_util.py` sha256 042bf0062da74964db9447fb21ab1151f0437dbf47fb9afebbce7daaf45c77ff
- `jobs/patch_sf.py` sha256 e27d487e2fe816baa78961a33a600f15341bfdc5f27505ecafacf6228dffd897
- `prompts/main.jsonl` sha256 dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124
- `prompts/controls_confirm.jsonl` sha256 31eeca8883bb20785123a9b4bd7affa13a94d5cc8bd0c285641379015b3e3f8d
