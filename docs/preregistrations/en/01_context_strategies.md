> English translation of the frozen Chinese preregistration [`../01_context_strategies.md`](../01_context_strategies.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration: comparing conditioning-frame strategies for long-video drift in Self Forcing (2026-10-18)

Written before analysis of the main experiment data. Generation of the main videos had already started, but before this file was frozen, only the number of completed videos was checked. No content or metrics of the main data had been looked at.
All design decisions come from the pilot experiment (`docs/pilot_report_cn.md`). The pilot prompts are not reused in the main experiment.
The analysis code `analyze_main.py` is frozen together with this file. The SHA-256 of both at freezing time is recorded at the end of this file.

## 1. Design
- Model: Self Forcing DMD checkpoint (Wan2.1-T2V-1.3B), 4-step sampling, bf16, A40 GPUs. The implementation of sink re-rotation is in `jobs/patch_sf.py`.
- Prompts: 40 (`prompts/main.jsonl`).
  - 20 MovieGen: from the 1003 prompts, excluding the 5 used in the pilot, randomly drawn with `numpy.random.default_rng(2026)`. The extended versions are used.
  - 20 driving scenes: written by us, covering different times of day, weather and road types.
- Seeds: 0, 1, 2, 3. Under the same seed, all strategies use the same initial noise and the same sampling random numbers (paired design).
- Video length: 126 latent frames, i.e. 501 frames, 16 fps, about 31 seconds.
- 10 strategies:
  - window {21, 12} × sink {0, 1, 3}, 6 in total;
  - plus re-rotated versions: window {21, 12} × sink {1, 3}, denoted w*s*r, 4 in total.
- Reference distribution: for each prompt, one 81-frame short video is generated for each of seeds 100–109, with global attention and no rolling.

## 2. Metrics (all implemented in `analyze_main.py`)

**M1 Color drift D (primary metric)**
- Take 5 features per frame: the means of Lab L, a, b; the mean HSV saturation; log(Laplacian variance + 1).
- Using the mean and covariance of the reference frames of the same prompt, compute the Mahalanobis distance of each frame, then average over 1-second segments.

**M2 Distribution drift K (primary metric)**
- Take one frame out of every 4, compute its DINOv2-base CLS feature and L2-normalize it.
- Compute its mean cosine distance to its 5 nearest neighbours among the reference frames of the same prompt.
- Divide by the median of the same distance computed on the reference frames themselves, leaving one seed out. Average over 1-second segments.
- Interpretation: K measures the degree of "departure from the normal output distribution for this prompt". It includes both image degradation and content gradually departing from the prompt, and **does not distinguish between the two**.

**Secondary metrics and metrics comparable with prior work**
- Signed z-scores: z_L, z_sat, z_b.
- MUSIQ-SPAQ ΔDrift = |mean of the first 5 seconds − mean of the last 5 seconds|, as defined by Rolling Forcing.
- ID-Forcing color metric = 100·(1 − ½‖h_first − h_last‖₁).
- Color difference ΔE between the first and last frames.

**Endpoint**: the segment mean over seconds 29–31 (the last 2 seconds).

## 3. Hypotheses and tests

Statistical model: for the endpoint of each metric, fit the linear mixed model y ~ strategy + (1 | prompt) + (1 | prompt:seed), estimated by REML. Test the linear contrasts below with Wald tests.

**Primary hypothesis family** (2 tests each for M1 and M2, 4 tests in total, Holm correction, α = 0.05):
- **H1 Attention sink reduces drift**: C1 = ½[(w21s3 − w21s0) + (w12s3 − w12s0)] < 0.
- **H2 Window × sink interaction**: C2 = (w12s0 − w21s0) − (w12s3 − w21s3) > 0. That is, without a sink the small window is worse, and with a sink this gap shrinks.

Directional hypotheses use one-sided p-values. A hypothesis is judged "supported" only if the Holm-corrected p < 0.05 and the direction of the 95% CI agrees with the hypothesis.

**Secondary analysis family** (Holm correction, two-sided tests):
- C3 effect of re-rotation: ½[(w21s3r − w21s3) + (w12s3r − w12s3)], tested separately on M1 and M2.
- C4 number of sink frames: ½[(w21s3 − w21s1) + (w12s3 − w12s1)], tested separately on M1 and M2.
- **H4 Darkening persists with a sink**: in the w21s3 and w12s3 videos, the mean z_L over the last 2 seconds < 0. One-sample t-test with prompt as the unit, one-sided.

**Robustness and exploratory analyses** (no verdicts):
- Recompute the CIs of C1–C4 by bootstrap: first resample prompts, then resample seeds within each prompt, B = 2000.
- Analyze the driving and MovieGen subsets separately.
- Curves of drift over time.
- Variance decomposition: how much of the variation comes from prompt and how much from seed.
- Agreement of prior-work metrics with M1 and M2, including whether MUSIQ favours oversaturated images.
- Observe whether sink-collapse (the image being pulled back to the content of the sink frames) occurs.

## 4. Data handling
- No videos are excluded. Videos that fail to generate are retried. Any still missing at the end are listed in the report, and the analysis uses the available data, because the mixed model can handle unbalanced data.
- Features of the reference frames and of the main videos are extracted with the same code (`gen.py` and `features.py`).

## 5. Deviations from the preregistration
If any change is made in the actual analysis, it is written in the "Deviations from preregistration" section of the final report, with the reason.

## Freeze record
(see below; appended by script)
- Freeze time: 2026-10-18 15:14 (local time). At freezing time, the analysis results of the main experiment had not been looked at.
- `docs/preregistration_cn.md` sha256 7938201aa540f3f88f55460410484430748eaad969c9b1ec3f518b9a3761a16e (file content before this line was appended)
- `analyze_main.py` sha256 f1548b77d31a0fdf9175cba3d4498f8f0c3f6633997964e909d13003f05957b2
- `gen.py` sha256 3109bf0afe76caf21593386ef57acb9000c7c3f2bf2d86f4a1e1601c83c02dd5
- `features.py` sha256 f6c15dba0bc97f0ba39b4a75c4c07c5f7f489eb9172d065f1f83d50434a95454
- `jobs/patch_sf.py` sha256 e27d487e2fe816baa78961a33a600f15341bfdc5f27505ecafacf6228dffd897
- `prompts/main.jsonl` sha256 dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124
