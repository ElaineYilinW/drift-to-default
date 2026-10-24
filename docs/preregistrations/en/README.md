# Preregistrations: English translations

These files are English translations of the ten preregistrations in the parent folder. The originals are in Chinese and were frozen, with SHA-256 hashes recorded, before data collection. The Chinese files are the authoritative versions. If a translation and its original disagree, the original wins.

Hashes, file paths, commands and formulas are copied verbatim from the originals. File names and paths inside the documents (for example `docs/preregistration_cn.md`) are the names used at freezing time.

| Translation | Original | Content |
|---|---|---|
| [`01_context_strategies.md`](01_context_strategies.md) | [`../01_context_strategies.md`](../01_context_strategies.md) | Self Forcing: compares 10 window × attention sink × re-rotation strategies for long-video drift (2026-10-18). |
| [`02_correction.md`](02_correction.md) | [`../02_correction.md`](../02_correction.md) | Confirmatory test of the content-referenced darkening correction (ttc, β = 1 and 1.5) on Self Forcing (2026-10-19). |
| [`03_matched_prompts.md`](03_matched_prompts.md) | [`../03_matched_prompts.md`](../03_matched_prompts.md) | Whether a VLM prompt that describes the opening frame reduces drift, on Self Forcing and LongLive (2026-10-19). |
| [`04_calibrated_lighting.md`](04_calibrated_lighting.md) | [`../04_calibrated_lighting.md`](../04_calibrated_lighting.md) | Whether choosing the lighting description by measured opening brightness reduces drift; also a fix to the matched-prompt data (2026-10-20). |
| [`05_calibrated_lighting_longlive.md`](05_calibrated_lighting_longlive.md) | [`../05_calibrated_lighting_longlive.md`](../05_calibrated_lighting_longlive.md) | The calibrated-lighting test repeated on LongLive with its own brightness scale (2026-10-20). |
| [`06_relaxation.md`](06_relaxation.md) | [`../06_relaxation.md`](../06_relaxation.md) | Interventional "relaxation" study: persistence α of shifted openings across models, sinks and correction (2026-10-23). |
| [`07_long_horizon_sink.md`](07_long_horizon_sink.md) | [`../07_long_horizon_sink.md`](../07_long_horizon_sink.md) | Confirmatory test at 60 s that the attention sink keeps a share of the opening, on Self Forcing and CausVid (2026-10-23). |
| [`08_skyreels_overlap.md`](08_skyreels_overlap.md) | [`../08_skyreels_overlap.md`](../08_skyreels_overlap.md) | Whether fewer overlapping history frames make SkyReels-V2 diffusion forcing relax (2026-10-23). |
| [`09_content_relaxation.md`](09_content_relaxation.md) | [`../09_content_relaxation.md`](../09_content_relaxation.md) | Whether scene content (snow, desert, forest, ...) relaxes to the prompt's default, measured with CLIP across 8 configurations (2026-10-24). |
| [`10_persistence_atlas.md`](10_persistence_atlas.md) | [`../10_persistence_atlas.md`](../10_persistence_atlas.md) | Persistence of all models at 60 s; sink size and sink position encoding (re-rotation) on Self Forcing (2026-10-24). |
