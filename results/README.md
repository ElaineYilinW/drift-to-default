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
| Baselines, global attributes, recursion fits | `corr/baseline_out/`, `corr/attr_out/`, `corr/relax_theory_out/` | `analysis/analyze_baselines.py`, `analyze_attributes.py`, `analyze_relax_theory.py` | `docs/lab-notes/10_baselines_attributes_theory.md` |
| Relaxation, 13 configurations (study 6, preregistered) | `corr/relax_out/`, `corr/relax_supp_out/` | `analysis/analyze_relax.py`, `analyze_relax_supp.py` | `docs/lab-notes/13_relaxation_results.md` |
| SkyReels history-noise exploration | `corr/relax_skyx_out/` | ad-hoc script (not published); figure by `report/make_relax_figs.py` | `docs/lab-notes/11_relaxation_exploratory_addendum.md` |
| Linearity in the shift, study 6 data (exploratory) | `corr/lin_out/` | `analysis/explore_linearity_s6.py` | paper, Appendix (robustness) |
| Sink at 60 s (study 7, preregistered) | `corr/long60_out/` | `analysis/analyze_long.py` | `docs/lab-notes/14_long_horizon_sink_results.md` |
| SkyReels overlap (study 8, preregistered) | `corr/skyconf_out/` | `analysis/analyze_skyov.py` | `docs/lab-notes/15_skyreels_overlap_results.md` |
| Prompt pull by replay (study 14, preregistered) | `corr/pswap_out/` | `analysis/analyze_pswap.py` | `docs/lab-notes/16_prompt_pull_results.md` |
| Scene content (study 9, preregistered) | `corr/content_out/` | `analysis/analyze_content.py` | `docs/lab-notes/17_content_relaxation_results.md` |
| Exploratory re-analyses for an internal review | `corr/review_out/` | `analysis/review_analyses.py`, `review2_analyses.py` | `docs/lab-notes/18_review_analyses.md` |
| Action-conditioned world model (study 15, preregistered) | `corr/mg_out/` | `analysis/analyze_mg.py`, `analyze_mg_explore.py` | `docs/lab-notes/19_world_model_results.md` |
| Non-driving prompts (study 12, preregistered) | `corr/pgen_out/` | `analysis/analyze_pgen.py` | `docs/lab-notes/20_prompt_generality_results.md` |
| Deeper sinks (study 13, preregistered) | `corr/deepsink_out/` | `analysis/analyze_deepsink.py` | `docs/lab-notes/21_deep_sink_results.md` |
| FramePack tone (study 11, preregistered) | `corr/fp_out/` | `analysis/analyze_fp.py` | `docs/lab-notes/22_framepack_results.md` |
| 60-s atlas (study 10, preregistered) | `corr/atlas_out/` | `analysis/analyze_atlas.py` | `docs/lab-notes/23_atlas_results.md` |
| Re-rotation and the 60-s horizon (study 16, preregistered) | `corr/s16_out/` | `analysis/analyze_s16.py` | `docs/lab-notes/24_rerotation_horizon_results.md` |
| MAGI-1 (study 18, preregistered) | `corr/magi_out/` | `analysis/analyze_magi.py`, `magi_curve.py` | `docs/lab-notes/25_magi_results.md` |
| FramePack content (study 17, preregistered) | `corr/fpc_out/` | `analysis/analyze_fpc.py` | `docs/lab-notes/26_framepack_content_results.md` |
| New seeds, tone (study 20, preregistered) | `corr/s20_out/` | `analysis/analyze_s20.py` | `docs/lab-notes/27_seed_replication_results.md` |
| New seeds, content (study 21, preregistered); second encoder | `corr/s21_out/`, `corr/enc2_out/` | `analysis/analyze_s21.py`, `encoder_robustness.py` | `docs/lab-notes/28_content_seeds_results.md` |
| Krea Realtime 14B (study 22, preregistered) | `corr/krea_out/`, `corr/krea_out_siglip/` | `analysis/analyze_krea.py` | `docs/lab-notes/29_krea_results.md` |
| Attention-to-sink probes, Self Forcing and LongLive (exploratory) | `corr/sinkprobe_out/`, `corr/sinkprobe_long_out/`, `corr/llprobe_out/` | `analysis/analyze_sinkprobe.py`, `analyze_sinkprobe_long.py` | `docs/lab-notes/30_sink_probe.md` |
| The sink's rotary position (study 25, preregistered) | `corr/s25_out/` | `analysis/analyze_s25.py`; figures `report/make_probe_fig.py`, `make_s25_frames_fig.py` | `docs/lab-notes/31_sink_position_results.md` |
| The 180-s horizon (study 24, preregistered; 60-s windows exploratory) | `corr/s24_out/` | `analysis/analyze_s24.py`, `explore_s24_windows.py` | `docs/lab-notes/33_long_horizon_results.md` |
| Out-of-sample burst times (study 26, preregistered; 120-s replication on study 6) | `corr/s26_out/`, `corr/s26_rep120_out/` | `analysis/analyze_s26.py` | `docs/lab-notes/32_burst_prediction_results.md` |

`corr/` stands for "correction". The folder grew out of the correction work and later held all the opening-based experiments.
