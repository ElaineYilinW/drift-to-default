# Results

This folder holds the summary outputs of every experiment: tables, JSON summaries and figures. Per-video raw data (frame statistics, DINOv2 features) and the videos are not included because of their size. Each folder below was written by the analysis script named next to it, and every number in the README and the report can be traced to one of these files.

| Experiment | Folder | Written by | Lab note |
|---|---|---|---|
| Feasibility | `feas/` | `generation/feas_drift.py` | — |
| Pilot | `pilot/` | `analysis/analyze_pilot.py` | `docs/lab-notes/01_pilot.md` |
| Context strategies (preregistered) | `main/` | `analysis/analyze_main.py` | `docs/lab-notes/02_context_strategies_results.md` |
| Correction development | `corr/e2_out/`, `corr/e3_out/` | `analysis/analyze_e2.py` | `docs/lab-notes/03_correction_development.md` |
| Latent brightness direction | `corr/latent_summary.csv`, `corr/dose.csv`, `corr/jac/`, `corr/jac_*.npz`, `corr/luma_*.npz` | `generation/latent_summary.py`, `dose_response.py`, `jacobian.py` | `docs/lab-notes/03_correction_development.md` |
| Correction, confirmatory (preregistered) | `corr/confirm_out/` | `analysis/analyze_confirm.py` | `docs/lab-notes/04_correction_results.md` |
| Context interventions | `corr/swap_out/`, `corr/checks_out/` | `analysis/analyze_swap.py`, `analyze_checks.py` | `docs/lab-notes/05_mechanism_and_real_openings.md` |
| Real and generated openings | `corr/real_out/`, `corr/self_out/` | `analysis/analyze_real.py` | `docs/lab-notes/05_mechanism_and_real_openings.md` |
| Lighting prompts, Self Forcing | `corr/light_out/` | `analysis/analyze_light.py` | `docs/lab-notes/06_prompt_default_and_longlive.md` |
| LongLive: openings, correction, two-model comparison | `corr/ll_open_out/`, `corr/ll_confirm_out/`, `corr/ll_out/` | `analysis/analyze_real.py`, `analyze_confirm.py`, `analyze_ll.py` | `docs/lab-notes/06_prompt_default_and_longlive.md` |
| Matched prompts (preregistered) | `corr/match_out/` (as run), `corr/match_out_repaired/` (sensitivity) | `analysis/analyze_match.py` | `docs/lab-notes/07_matched_prompts_results.md` |
| Calibrated lighting (preregistered) | `corr/calib_out/` | `analysis/analyze_calib.py` | `docs/lab-notes/08_calibrated_lighting_results.md` |
| Lighting prompts, LongLive | `corr/ll_light_out/` | `analysis/analyze_light.py` | `docs/lab-notes/09_calibrated_lighting_longlive_results.md` |
| Calibrated lighting, LongLive (preregistered) | `corr/calib2_out/` | `analysis/analyze_calib.py`, `analyze_calib_ll.py` | `docs/lab-notes/09_calibrated_lighting_longlive_results.md` |

`corr/` stands for "correction". The folder grew out of the correction work and later held all the opening-based experiments.
