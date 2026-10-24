# Documentation

The [paper draft](../paper/main.pdf) is the English write-up of the whole study (an earlier [technical report](../report/drift_to_default.pdf) covers Parts 1–6). This folder holds the working documents behind it, written in Chinese as the study went along.

## Preregistrations

Each preregistration was written and frozen before any of its videos were generated. It records the hypotheses, the decision rules and the SHA-256 hashes of the analysis code. Preregistrations 03–11 were also committed to this repository before generation. English translations are in [`preregistrations/en/`](preregistrations/en/README.md); the Chinese files are the frozen, authoritative versions. To check that the published code matches the frozen versions, run `python analysis/verify_preregistration.py`.

| # | Study | Primary hypotheses | Outcome |
|---|---|---|---|
| [01](preregistrations/01_context_strategies.md) | Context strategies (window × sink × RoPE re-alignment) | Sink reduces drift; window size matters only without a sink | Supported (all 4 primary tests) |
| [02](preregistrations/02_correction.md) | Content-aware correction, held-out prompts | Correction reduces colour drift without structural harm; legitimate transitions kept | Supported (all criteria) |
| [03](preregistrations/03_matched_prompts.md) | Prompts written from a VLM caption of the opening | Caption prompt reduces brightness drift; correction adds to it | Not supported |
| [04](preregistrations/04_calibrated_lighting.md) | Lighting description chosen from measured brightness | Calibrated prompt beats the caption prompt and the generic prompt | Supported vs generic on Self Forcing; not supported otherwise |
| [05](preregistrations/05_calibrated_lighting_longlive.md) | Same, with LongLive's own calibration | Calibrated prompt beats the generic prompt and the caption prompt on LongLive | Not supported (the lighting-only prompt helps as a secondary result) |
| [06](preregistrations/06_relaxation.md) | Relaxation: shifted openings × 11 model configurations | A sink raises the opening's persistence (two models); the correction raises it; same for colour | 3 of 4 supported (Self Forcing brightness not at 31 s) |
| [07](preregistrations/07_long_horizon_sink.md) | Sink retention at 60 s, Self Forcing and CausVid | A sink raises the opening's persistence at 60 s on both | Supported (both) |
| [08](preregistrations/08_skyreels_overlap.md) | SkyReels overlap: does a weaker overlap constraint make diffusion forcing relax? | Persistence lower with 5 than with 17 overlapping frames | Supported |
| [09](preregistrations/09_content_relaxation.md) | Content: does scene content (snow, desert, forest, ...) relax to the prompt's default? | Class score lower under the generic prompt; the sink keeps content | Running |
| [10](preregistrations/10_persistence_atlas.md) | Atlas: persistence of every model at 60 s; does sink size or sink position matter? | Deeper sink keeps more; any sink keeps some; LongLive's sink at 60 s; re-rotation (two-sided) | Running |
| [11](preregistrations/11_framepack_channels.md) | FramePack (non-Wan): does it relax, and which route of the opening (context or conditioning) sets the default? | Keeps most; the conditioning route moves the default and adds persistence | Running |

## Lab notes

| # | Note | Content |
|---|---|---|
| [00](lab-notes/00_literature_review.md) | Literature review | Self Forcing and its long-video successors, test-time correction, evaluation practice |
| [01](lab-notes/01_pilot.md) | Pilot | 10 prompts; choice of metrics and sample size |
| [02](lab-notes/02_context_strategies_results.md) | Context strategies: results | Preregistered analysis of 2,000 videos |
| [03](lab-notes/03_correction_development.md) | Correction: development | Failed statistical controller; latent brightness directions; content-aware correction and its step/β knobs |
| [04](lab-notes/04_correction_results.md) | Correction: confirmatory results | Held-out prompts, control prompts, flicker cost |
| [05](lab-notes/05_mechanism_and_real_openings.md) | Mechanism checks and real openings | KV-cache interventions, real NAVSIM openings, validity checks, retracted claims |
| [06](lab-notes/06_prompt_default_and_longlive.md) | The prompt sets the default; second model | Lighting prompts; LongLive replication |
| [07](lab-notes/07_matched_prompts_results.md) | Matched prompts: results | Negative primary result; why the caption fails; pairing repair |
| [08](lab-notes/08_calibrated_lighting_results.md) | Calibrated lighting: results | Confirmed on Self Forcing; calibration is model-specific |
| [09](lab-notes/09_calibrated_lighting_longlive_results.md) | Calibrated lighting on LongLive: results | LongLive's own calibration; primary not significant |
| [10](lab-notes/10_baselines_attributes_theory.md) | Baselines, attributes, theory | Post-hoc colour matching vs correction; every statistic relaxes; chunk-recursion model |
| [11](lab-notes/11_relaxation_exploratory_addendum.md) | Relaxation: addenda | Exploratory configurations and scheduling deviations, recorded before data |
| [12](lab-notes/12_skyreels_mechanism.md) | SkyReels mechanism | Diffusion forcing does not relax; not explained by CFG, history noise or steps |
| [13](lab-notes/13_relaxation_results.md) | Relaxation: results | Preregistration 06 results across 13 configurations |
| [14](lab-notes/14_long_horizon_sink_results.md) | Sink at 60 s: results | Preregistration 07 results |
| [15](lab-notes/15_skyreels_overlap_results.md) | SkyReels overlap: results | Preregistration 08: a 5-frame overlap makes diffusion forcing relax |

The lab notes use the folder and file names from before the repository was reorganised (for example `analyze_match.py` without its `analysis/` folder). [`results/README.md`](../results/README.md) maps every experiment to its script, its output folder and its note.
