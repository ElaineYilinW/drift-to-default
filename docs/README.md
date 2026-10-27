# Documentation

The [paper draft](../paper/main.pdf) is the English write-up of the whole study (an earlier [technical report](../report/drift_to_default.pdf) covers Parts 1–6). This folder holds the working documents behind it, written in Chinese as the study went along.

## Preregistrations

Each preregistration was written and frozen before any of its videos were generated. It records the hypotheses, the decision rules and the SHA-256 hashes of the analysis code. Preregistrations 03–15 were also committed to this repository before generation. English translations are in [`preregistrations/en/`](preregistrations/en/README.md); the Chinese files are the frozen, authoritative versions. To check that the published code matches the frozen versions, run `python analysis/verify_preregistration.py`.

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
| [09](preregistrations/09_content_relaxation.md) | Content: does scene content (snow, desert, forest, ...) relax to the prompt's default? | Class score lower under the generic prompt; the sink keeps content | 2 of 3 supported: content relaxes to the prompt's default; Self Forcing's sink keeps content at 60 s, LongLive's does not |
| [10](preregistrations/10_persistence_atlas.md) | Atlas: persistence of every model at 60 s; does sink size or sink position matter? | Deeper sink keeps more; any sink keeps some; LongLive's sink at 60 s; re-rotation (two-sided) | Supported (4 of 4); re-rotating the sink cuts persistence from 0.61 to 0.16 and removes the 32-s jump |
| [11](preregistrations/11_framepack_channels.md) | FramePack (non-Wan): does it relax, and which route of the opening (context or conditioning) sets the default? | Keeps most; the conditioning route moves the default and adds persistence | 1 of 3 supported: keeps 0.68 at 20 s; the image-embedding route has no effect on tone |
| [12](preregistrations/12_prompt_generality.md) | Prompt generality: sink effects under four non-driving prompts; does persistence depend on the prompt? | Sinks keep more (Self Forcing, LongLive); persistence differs between prompts | Supported (3 of 3): sink effects replicate outside driving; persistence differs by domain |
| [13](preregistrations/13_deep_sink_baseline.md) | Deep-sink baseline: do deeper sinks reduce drift, and at what cost to legitimate changes? | Lower colour drift with sink 6 and sink 9 | Supported (2 of 2): deeper sinks cut drift as much as the correction with less flicker, but keep fewer legitimate changes (0.59–0.70) |
| [14](preregistrations/14_prompt_pull_replay.md) | Prompt pull: measure w_p by swapping only the text conditioning for one chunk | w_p > 0, in [0.04, 0.14] as the recursion predicts | Supported: 0.088 [0.045, 0.149], in the band; not time-invariant (0.26 at 30 s) |
| [15](preregistrations/15_world_model.md) | World model (Matrix-Game-2.0): is an opening held only in the context forgotten, and does the image conditioning set the default? | Context-only shift fades; conditioning-only shift persists; conditioning adds persistence | 1 of 3 supported (only 7 of 24 scenes survive the model's collapse to black before 30 s); all three hold at 15 s (exploratory, 20 scenes) |
| [16](preregistrations/16_rerotation_and_horizon.md) | Re-rotated sink and the 60-s horizon: does the late return of content vanish without the sink's original position; does a deep re-rotated sink keep more; do FramePack and SkyReels keep a share at 60 s? | Re-rotation removes the content return and lowers persistence (sink 3, sink 6); depth still helps; FramePack keeps a share; SkyReels overlaps keep forgetting | Running |
| [17](preregistrations/17_framepack_content.md) | FramePack content routes: does scene content relax on a non-Wan model, and which route (context prefix or SigLIP embedding) carries it? | Content relaxes; the context prefix and the SigLIP embedding each carry content; the embedding adds persistence | Supported (4 of 4); content kept at 20 s 0.55 (both), 0.24 (context only), 0.23 (embedding only, rising) |
| [18](preregistrations/18_magi.md) | MAGI-1 (non-Wan, chunk-wise KV cache): does the released attention window forget the opening, and does full history keep a share? | The window forgets; full history keeps more; full history keeps a share | Supported (3 of 3); persistence at 30 s 0.34 (window) vs 0.64 (full history); no freezing |

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
| [16](lab-notes/16_prompt_pull_results.md) | Prompt pull measured by replay (study 14) | w_p = 0.088, as predicted; triples by 30 s |
| [17](lab-notes/17_content_relaxation_results.md) | Scene content relaxes (study 9) | Snow, desert, forest openings become city under a city prompt; the sink brings content back late |
| [18](lab-notes/18_review_analyses.md) | Exploratory re-analyses for an internal review | Drift = forgetting + attractor offset; persistence depends on the prompt; study 7 holds on independent scenes |
| [19](lab-notes/19_world_model_results.md) | Action-conditioned world model (study 15) | Context forgets, image conditioning sets the default; the model collapses to black near its training length |
| [20](lab-notes/20_prompt_generality_results.md) | Non-driving prompts (study 12) | Sink effects replicate in park, drone, kitchen, market scenes; persistence differs by domain |
| [21](lab-notes/21_deep_sink_results.md) | Deeper sinks as a baseline (study 13) | Strong baseline: less drift and flicker, but legitimate brightness changes are suppressed |
| [22](lab-notes/22_framepack_results.md) | FramePack (study 11) | Relaxes but keeps 0.68; context prefix acts as a permanent sink; SigLIP route inert for tone |
| [23](lab-notes/23_atlas_results.md) | 60-s atlas (study 10) | Persistence monotone in sink depth; the sink's absolute position causes the late jump and much of its retention |
| [25](lab-notes/25_magi_results.md) | MAGI-1, a non-Wan architecture (study 18) | The released window forgets (0.87 at 5 s to 0.34 at 30 s); full history plateaus near 0.65; keeping the opening raises persistence but not the shared drift toward a brighter default |
| [26](lab-notes/26_framepack_content_results.md) | FramePack content routes (study 17) | Content relaxes on FramePack too (0.55 at 20 s); the SigLIP embedding, inert for tone, pulls content toward its class like a prompt; the two routes add up |

The lab notes use the folder and file names from before the repository was reorganised (for example `analyze_match.py` without its `analysis/` folder). [`results/README.md`](../results/README.md) maps every experiment to its script, its output folder and its note.
