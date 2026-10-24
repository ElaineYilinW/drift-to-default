# Drift to Default

**Long autoregressive video rollouts relax to a default set by the prompt. This repository measures how much of an opening survives across 14 configurations of four KV-cache model families (plus SkyReels-V2), explains it with an equilibrium between the attention sink and the prompt, and tests two training-free levers. It reports eight preregistered studies, including the hypotheses that failed.**

[![Paper draft](https://img.shields.io/badge/paper-PDF-b31b1b)](paper/main.pdf)
[![Technical report](https://img.shields.io/badge/report-PDF-6e6e6e)](report/drift_to_default.pdf)
[![Preregistered](https://img.shields.io/badge/preregistered-8%20studies-2a78d6)](docs/README.md#preregistrations)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

<p align="center">
  <img src="assets/gif_farm.gif" width="49%" alt="the same seed without and with the correction, rural road">
  <img src="assets/gif_tunnel.gif" width="49%" alt="naive brightness controller versus content-aware correction, tunnel">
</p>
<p align="center"><sub>Left: one seed without and with the content-aware correction (31 s, Self Forcing). Right: a naive brightness controller lifts a tunnel to overexposure; the content-aware correction keeps it dark.</sub></p>

## Overview

Autoregressive ("causal") video diffusion models such as [Self Forcing](https://github.com/guandeh17/Self-Forcing) and [LongLive](https://github.com/NVlabs/LongLive) generate a video a few frames at a time. Each new chunk is conditioned on the model's own earlier output through a KV cache. This makes long, streaming generation possible, which world models need. Over tens of seconds the picture drifts: it darkens, saturates or loses detail.

This study treats drift as a measurement problem. It uses the design tools of clinical research, which are rare in this field:
- paired seeds;
- prompts held out from development;
- mixed-effects models and cluster-bootstrap confidence intervals;
- analysis plans and code frozen before the data existed.

About **9,600 videos**, most of them 31 seconds long, were generated on a university GPU cluster. No model was trained; everything is inference on public weights followed by measurement.

## Key findings

The core result is an equilibrium law. After a long rollout, a statistic such as brightness or colour settles at

&nbsp;&nbsp;&nbsp;&nbsp;**x<sub>∞</sub> ≈ α · x<sub>opening</sub> + (1 − α) · μ<sub>prompt</sub>**

- **α** is the share of the opening that is kept. A sink and the correction raise it.
- **μ** is the default the prompt pulls toward. Calibrated prompts move it toward the opening.

| | Finding | Evidence | Status |
|---|---|---|---|
| 1 | **Every attention-based model relaxes toward a default.** The share of a shifted opening that survives 31 s is 0.02–0.67. It depends strongly on the training objective even with identical architecture. | Within-scene shifts of 24 openings × 14 configurations (2,352 videos): Self Forcing SiD 0.12, CausVid 0.15, LongLive 0.19, DMD 0.28, SiD-v2 0.35, Rolling Forcing 0.53, GAN 0.67. Colour temperature behaves alike. | Preregistered study |
| 2 | **The default is set by the prompt, and it is where a fresh rollout ends.** | Zero-drift level vs fresh-start end, 10 configurations: r = 0.93. Lighting prompts: sunny 43, overcast 34, dusk 19, night 16 L\* (Self Forcing); 45 / 40 / 32 / 21 on LongLive. | Exploratory |
| 3 | **An attention sink holds a fixed share of the opening; without it the opening is forgotten.** The fitted per-chunk sink weight matches a direct KV-cache measurement. | At 60 s: Self Forcing 0.58 vs 0.03 (+0.55 [0.45, 0.65]); CausVid 0.23 vs 0.04 (+0.19 [0.14, 0.24]); LongLive at 31 s 0.19 vs 0.02. Fitted w<sub>s</sub> 0.060 [0.021, 0.097] vs replay 0.053 [0.040, 0.066]. Self Forcing brightness was not significant at the earlier 31-s endpoint (slow forgetting). | Preregistered (06, 07); 1 of 4 failed in 06 |
| 4 | **Diffusion forcing is the exception that confirms the mechanism.** SkyReels-V2, which continues by re-denoising 17 overlapping frames, keeps its opening. With 5 overlapping frames it relaxes. | α ≈ 1 at 10–20 s; unchanged without CFG, with 3× history noise, or with 10 steps. Overlap 5 vs 17: 0.77 vs 1.01 at 15–20 s, −0.24 [−0.40, −0.09], p = 0.014 (12 new scenes). | Preregistered (08) |
| 5 | **A training-free, content-aware correction raises α, reduces drift and keeps legitimate changes.** Post-hoc colour matching does not. | Colour drift D −1.48 [−1.76, −1.19] on 40 held-out prompts; α +0.12 [0.05, 0.20]; legitimate transitions kept 83–93% vs 7–12% for post-hoc matching. | Preregistered (02, 06) |
| 6 | **Moving μ: a lighting description calibrated to the opening's measured brightness reduces drift; a free-form VLM caption does not.** The law predicts these outcomes out of sample. | Calibrated vs generic −26% (Holm p = 0.013, Self Forcing); caption n.s. on both models; LongLive with its own calibration n.s. (lighting-only −34%, secondary). Predicted vs observed r = 0.82 / 0.77. | Preregistered (03–05): 1 supported, 2 failed |
| 7 | **Context strategies:** a sink cuts colour drift, and window size matters only without one. | 2,000 videos: D −7.56 [−8.32, −6.81]. | Preregistered (01) |

![Brightness change after 31 seconds against the brightness of the opening frame](assets/fig_default_brightness.png)
<sub>Brightness change after 31 s against the brightness of a held opening frame. (a) Real NAVSIM and model-generated openings fall on one line. (b) Four lighting prompts set four default levels. (c) Two models: same slope, different default.</sub>

## Study design

Every experiment compares arms on the **same openings or prompts, with the same seeds, on the same GPU architecture**. In bf16 the GPU type changes the output; see [Lessons](#lessons-for-evaluating-video-models).

| Part | Question | Design | Videos | Preregistration | Notes |
|---|---|---|---|---|---|
| 1 | How do rolling windows and attention sinks affect drift? | 10 strategies × 40 prompts × 4 seeds, 31 s | 2,000 | [01](docs/preregistrations/01_context_strategies.md) | [02](docs/lab-notes/02_context_strategies_results.md) |
| 2 | Can drift be corrected at inference time? | Developed on 16 prompts, then confirmed on 40 held-out prompts and 8 held-out control prompts with legitimate changes | ~1,200 | [02](docs/preregistrations/02_correction.md) | [03](docs/lab-notes/03_correction_development.md), [04](docs/lab-notes/04_correction_results.md) |
| 3 | What is drift? | Held real (NAVSIM) and generated openings; KV-cache interventions | ~400 | — | [05](docs/lab-notes/05_mechanism_and_real_openings.md) |
| 4 | Does the prompt set the default? Does it replicate? | 40 openings × 4 lighting prompts; LongLive replication | ~1,250 | — | [06](docs/lab-notes/06_prompt_default_and_longlive.md) |
| 5 | Does describing the opening help? | 40 new openings × 5 arms × 2 models | 800 | [03](docs/preregistrations/03_matched_prompts.md) | [07](docs/lab-notes/07_matched_prompts_results.md) |
| 6 | Does a brightness-calibrated description help? | Same openings, 3 new arms, calibrated per model | ~670 | [04](docs/preregistrations/04_calibrated_lighting.md), [05](docs/preregistrations/05_calibrated_lighting_longlive.md) | [08](docs/lab-notes/08_calibrated_lighting_results.md), [09](docs/lab-notes/09_calibrated_lighting_longlive_results.md) |
| 7 | How much of an opening survives, across models? | 24 scenes × 7 shifted openings × 14 configurations, + fresh starts, + 120-s subset; SkyReels on 12 scenes | ~2,500 | [06](docs/preregistrations/06_relaxation.md) | [11](docs/lab-notes/11_relaxation_exploratory_addendum.md), [12](docs/lab-notes/12_skyreels_mechanism.md), [13](docs/lab-notes/13_relaxation_results.md) |
| 8 | Does the sink hold the opening at 60 s? | 24 scenes × 3 openings × sink on/off × Self Forcing, CausVid | 288 | [07](docs/preregistrations/07_long_horizon_sink.md) | [14](docs/lab-notes/14_long_horizon_sink_results.md) |
| 9 | Does a weaker overlap make diffusion forcing relax? | 12 new scenes × overlap 5 / 17 × ±16 L\*, 20 s, SkyReels | 48 | [08](docs/preregistrations/08_skyreels_overlap.md) | [12](docs/lab-notes/12_skyreels_mechanism.md), [15](docs/lab-notes/15_skyreels_overlap_results.md) |

**Metrics.** For each prompt, 10 fresh 5-second generations define the model's own normal output. Drift is measured against that reference.
- **D:** Mahalanobis distance of per-frame colour statistics (Lab means, saturation, sharpness).
- **K:** DINOv2 nearest-neighbour distance, scaled so that 1 means "as typical as a fresh start".
- **Held-opening experiments:** the brightness change \|ΔL\| and colour shift ΔE between the opening and the last 2 seconds.

## Results

### Relaxation across models (Parts 7–8)

![Persistence of a shifted opening across 14 configurations, over time, and at 60 s](assets/fig_relax.png)

<p align="center"><img src="assets/fig_default_fresh.png" width="45%" alt="zero-drift level vs fresh-start end"></p>

- **Shifted openings.** Each of 24 scenes was shifted in brightness and colour and continued by every model. The persistence α is the slope of the end value on the opening value within a scene.
- **Sink.** With a sink, a fixed share is held for at least 60–120 s; without it the opening is forgotten (panel c).
- **Default.** The default each model relaxes to equals where its fresh rollouts end (bottom figure).
- **The exception.** SkyReels' diffusion-forcing continuation does not relax with its default 17-frame overlap. It does relax with a 5-frame overlap (preregistration 08), so the hard pixel-level overlap is what protects it.
- **Mechanism.** A linear chunk-recursion model (`analysis/ar_equilibrium.py`) reproduces the curves. Its fitted sink weight matches the bit-exact KV-cache replay measurement. Details: [lab note 13](docs/lab-notes/13_relaxation_results.md).

### Context strategies (Part 1)

![Colour drift and brightness over 31 seconds for four context strategies](assets/fig_context_strategies.png)

- Anchoring cuts colour drift, with a dose effect: 3 frames > 1 > 0.
- Window size matters only without anchoring.
- Re-aligning the sink's RoPE positions, as Rolling and Deep Forcing do, did not help on Self Forcing, which was not trained for it.
- Anchoring slows darkening but does not stop it.

### The correction (Part 2)

At denoising step 1 of 4, the noisy chunk is run once more with a temporary KV cache that holds only the anchor frames. The difference in per-channel spatial means (the "DC" tone) between that prediction and the real one is added back, scaled by β. Layout, detail, the main KV cache and the random stream are unchanged. The cost is one extra forward pass per chunk.

The denoising step works as a knob between drift removal and content fidelity:

| Correction step | Legitimate transitions kept |
|---|---|
| 0 | 56% (also erases tunnels and dusk) |
| 1 | 92% |
| 2–3 | no effect |

A statistical closed-loop controller was tried first and failed; [lab note 03](docs/lab-notes/03_correction_development.md) documents why.

![Paired effects of the correction on 40 held-out prompts](assets/fig_correction_confirmatory.png)

The cost is flicker: jitter rises ×1.13 at β = 1 and ×1.67 at β = 1.5.

Compared with post-hoc colour matching to the anchor, only the content-aware correction both reduces drift and keeps legitimate brightness changes ([lab note 10](docs/lab-notes/10_baselines_attributes_theory.md)). In the relaxation study it raises α from 0.28 to 0.40.

![Drift reduction vs legitimate transitions kept](assets/fig_tradeoff.png)

### Matching the prompt to the opening (Parts 5–6)

![Matched-prompt experiment: paired effects and breakdown by caption lighting](assets/fig_matched_prompts.png)

Part 5: a prompt written from a Qwen2.5-VL caption of the opening keeps the scene but not its brightness. The captioner describes about half of the normally lit frames as "dimly lit, early morning or late evening". Under such a prompt those videos darken more (panel c).

![Calibrated lighting: paired effects and breakdown by class](assets/fig_calibrated_lighting.png)

Part 6: choosing the lighting description from measured brightness fixes this on Self Forcing. The calibration is model-specific: "dusk" settles at 19 L\* on Self Forcing and 32 on LongLive. On LongLive, the lighting-only calibrated prompt helps (−34%). Adding the VLM scene phrase cancels the gain, probably because that phrase brings its own weather words.

## Measuring α and μ for your own model

The two numbers in the law are cheap to measure for any autoregressive video model that can continue a given first chunk.

1. **Pick openings.** Choose 10–24 opening images. For each one, write versions shifted in CIELAB L\* by −16, 0 and +16 (`--open_dL` in `generation/hold_gen.py`; for other code bases, copy the 6-line shift in its `init_latent_from_image`).
2. **Continue them.** Continue every version for 30–60 s with one prompt and one seed. Keep all versions of an opening in one job on one GPU architecture (`generation/arch_guard.py`).
3. **Estimate α.** α is the within-opening slope of end brightness (last 2 s) on opening brightness: `analysis/analyze_relax.py` or `analysis/analyze_long.py`.
4. **Estimate μ.** μ is where a few fresh rollouts of the same prompt end. In our 10 configurations it agreed with the zero-drift level from step 3 at r = 0.93.
5. **Predict a prompt change.** The expected long-horizon change is (1 − α)(μ − opening), with μ measured for each candidate prompt.

## Claims that did not hold

We report claims made during the study that were later revised or withdrawn after further checks. Details are in [lab note 05](docs/lab-notes/05_mechanism_and_real_openings.md).
- **"The sink works by stabilising the default level."** Withdrawn: trend difference +0.5 [−2.4, 3.5].
- **"The model follows its recent context about 75%."** True only for an artificial tone shift; for natural differences it is 90–100%.
- **Forecasting the final brightness of a video from its early dynamics.** Out-of-sample forecasts were weak.
- **"MUSIQ rewards collapsed videos."** Seen in the pilot, not replicated; MUSIQ is simply insensitive to this drift.
- **"Describing the opening is enough."** Not supported (Part 5).

## Lessons for evaluating video models

- **The GPU architecture changes the video.** With bf16, the same seed gives bit-identical output within an architecture but different videos on A40, A100/A800 and L40S/RTX 6000 Ada. Paired comparisons must pin the architecture. In Part 1, 62% of the nominal pairs ran on different architectures; later parts pin it and verify by bit-exact reruns.
- **A resumed job can silently break pairing.** A preempted multi-arm job resumed on another GPU type and split 4 openings across architectures. The data were repaired and a sensitivity analysis showed the conclusions unchanged ([lab note 07](docs/lab-notes/07_matched_prompts_results.md)).
- **Common quality metrics miss this drift.** MUSIQ, the model behind VBench's imaging-quality score, is nearly flat across strategies that differ greatly in drift.
- **Prompts, not seeds, carry the variance.** The seed component of end-of-video drift was 0.2 against 13.7 for prompts. More prompts buy more precision than more seeds.

## Reproducing

**Re-checking the published analysis** needs only the summary files in this repository:
```bash
pip install -r requirements.txt
python analysis/verify_preregistration.py   # frozen analysis code matches the preregistered hashes
python report/make_figs.py                  # redraws the report figures from results/
```

**Re-running the generation** needs a GPU:
1. Clone [Self Forcing](https://github.com/guandeh17/Self-Forcing) (tested at commit 33593df). Download `Wan-AI/Wan2.1-T2V-1.3B` and the `self_forcing_dmd.pt` checkpoint, then run `cluster/patch_sf.py` in that repository. The patch adds an SDPA fallback when flash-attention is unavailable, bf16/mmap loading, and optional sink RoPE re-alignment.
2. For LongLive, clone [LongLive](https://github.com/NVlabs/LongLive), download `Efficient-Large-Model/LongLive-1.3B`, and run `cluster/setup_ll.sh` / `cluster/patch_ll.py`.
3. Environment: `cluster/setup_sf_env.sh` (PyTorch 2.5.1 + cu121, transformers 4.49, pyiqa). It was written for an old-glibc SGE cluster; paths in `cluster/env.sh` are site-specific.
4. Run the scripts in `generation/` from the model's repository root. Each script's docstring gives its arguments. One 501-frame video takes about 3 minutes on an A40, A100 or L40S.

NAVSIM / nuPlan camera frames used as real openings are not redistributed. `prompts/` lists the log and frame for each.

The paper draft is [paper/main.pdf](paper/main.pdf); an earlier technical report is in [report/](report/drift_to_default.pdf).

## Repository layout

```
generation/   video generation with context strategies, the correction hook, held openings, KV-cache
              interventions, VLM captions, DINOv2/MUSIQ features, latent brightness probes
analysis/     one script per experiment (preregistered ones frozen), opening selection, verify_preregistration.py
prompts/      all prompt sets, including the caption and calibrated-lighting prompts
results/      summary tables, JSON and figures for every experiment (index: results/README.md)
paper/        paper draft (LaTeX + PDF)
report/       earlier technical report and the scripts that draw all figures
docs/         preregistrations and lab notes (index: docs/README.md)
cluster/      environment setup and runtime patches for the SGE cluster
tools/        GIFs and contact sheets
```

## Limitations

- Both models share the Wan2.1-1.3B base, and videos are 31 seconds long.
- Real openings are single held frames, not real video.
- The correction acts on colour and brightness only. Structural degradation late in a video, such as sky textures, remains.
- The lighting calibration has four discrete levels. Openings brighter than the sunny level (about 43–45 L\*) have no matching description.

## Related work

Self Forcing ([arXiv:2506.08009](https://arxiv.org/abs/2506.08009)) and its long-video successors use rolling windows, sinks and re-aligned positions: Self-Forcing++, LongLive, Rolling Forcing and Deep Forcing ([arXiv:2512.05081](https://arxiv.org/abs/2512.05081)). This study measures those choices with uncertainty. The correction is closest to Pathwise Test-Time Correction ([arXiv:2602.05871](https://arxiv.org/abs/2602.05871)) and FreqForcing ([arXiv:2607.27110](https://arxiv.org/abs/2607.27110)); it is a simplification that transfers only the DC tone at one early step. Also related are ID-Forcing ([arXiv:2610.03120](https://arxiv.org/abs/2610.03120)), Stable Video Infinity, MotionStream and Meta-ARVDM ([arXiv:2503.10704](https://arxiv.org/abs/2503.10704)). A literature review up to October 2026 is in [lab note 00](docs/lab-notes/00_literature_review.md).

## Citation

```bibtex
@misc{wang2026drifttodefault,
  title  = {Drift to Default: Brightness Drift in Autoregressive Video Diffusion Is Regression to a Prompt-Set Level},
  author = {Wang, Yilin},
  year   = {2026},
  url    = {https://github.com/ElaineYilinW/drift-to-default}
}
```

## License and acknowledgements

The code in this repository is released under the [MIT License](LICENSE). It drives and patches [Self Forcing](https://github.com/guandeh17/Self-Forcing) and [LongLive](https://github.com/NVlabs/LongLive). It also uses [Wan2.1](https://github.com/Wan-Video/Wan2.1), [DINOv2](https://github.com/facebookresearch/dinov2), [pyiqa](https://github.com/chaofengc/IQA-PyTorch) and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), each under its own licence. NAVSIM / nuPlan images are not redistributed. Computation ran on the University of Iowa Argon cluster.

This is an independent study and is not affiliated with any company.
