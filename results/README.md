# Results

This folder holds the summary outputs of every experiment: tables, JSON summaries and figures. Per-video raw data (frame statistics, DINOv2 features) and the videos are not included because of their size. Each folder below was written by the analysis script named next to it, and every number in the README and the report can be traced to one of these files.

| Experiment | Folder | Written by |
|---|---|---|
| Context strategies (preregistered) | `main/` | `analysis/analyze_main.py` |
| Correction development | `corr/e2_out/`, `corr/e3_out/` | `analysis/analyze_e2.py` |
| Latent brightness direction | `corr/latent_summary.csv`, `corr/dose.csv`, `corr/jac/`, `corr/jac_*.npz`, `corr/luma_*.npz` | `generation/latent_summary.py`, `dose_response.py`, `jacobian.py` |
| Correction, confirmatory (preregistered) | `corr/confirm_out/` | `analysis/analyze_confirm.py` |
| Context interventions | `corr/swap_out/`, `corr/checks_out/` | `analysis/analyze_swap.py`, `analyze_checks.py` |
| Real and generated openings | `corr/real_out/`, `corr/self_out/` | `analysis/analyze_real.py` |
| Lighting prompts, Self Forcing | `corr/light_out/` | `analysis/analyze_light.py` |
| LongLive: openings, correction, two-model comparison | `corr/ll_open_out/`, `corr/ll_confirm_out/`, `corr/ll_out/` | `analysis/analyze_real.py`, `analyze_confirm.py`, `analyze_ll.py` |
| Matched prompts (preregistered) | `corr/match_out/` (as run), `corr/match_out_repaired/` (sensitivity) | `analysis/analyze_match.py` |
| Calibrated lighting (preregistered) | `corr/calib_out/` | `analysis/analyze_calib.py` |
| Lighting prompts, LongLive | `corr/ll_light_out/` | `analysis/analyze_light.py` |
| Calibrated lighting, LongLive (preregistered) | `corr/calib2_out/` | `analysis/analyze_calib.py`, `analyze_calib_ll.py` |
| Baselines, global attributes, recursion fits | `corr/baseline_out/`, `corr/attr_out/`, `corr/relax_theory_out/` | `analysis/analyze_baselines.py`, `analyze_attributes.py`, `analyze_relax_theory.py` |
| Relaxation, 13 configurations (study 6, preregistered) | `corr/relax_out/`, `corr/relax_supp_out/` | `analysis/analyze_relax.py`, `analyze_relax_supp.py` |
| Linearity in the shift, study 6 data (exploratory) | `corr/lin_out/` | `analysis/robustness_linearity_s6.py` |
| Sink at 60 s (study 7, preregistered) | `corr/long60_out/` | `analysis/analyze_long.py` |
| SkyReels overlap (study 8, preregistered) | `corr/skyconf_out/` | `analysis/analyze_skyov.py` |
| Prompt pull by replay (study 14, preregistered) | `corr/pswap_out/` | `analysis/analyze_pswap.py` |
| Scene content (study 9, preregistered) | `corr/content_out/` | `analysis/analyze_content.py` |
| Supplementary analyses | `corr/review_out/` | `analysis/supplementary_analyses.py`, `supplementary_analyses2.py` |
| Action-conditioned world model (study 15, preregistered) | `corr/mg_out/` | `analysis/analyze_mg.py`, `analyze_mg_supp.py` |
| Non-driving prompts (study 12, preregistered) | `corr/pgen_out/` | `analysis/analyze_pgen.py` |
| Deeper sinks (study 13, preregistered) | `corr/deepsink_out/` | `analysis/analyze_deepsink.py` |
| FramePack tone (study 11, preregistered) | `corr/fp_out/` | `analysis/analyze_fp.py` |
| 60-s atlas (study 10, preregistered) | `corr/atlas_out/` | `analysis/analyze_atlas.py` |
| Re-rotation and the 60-s horizon (study 16, preregistered) | `corr/s16_out/` | `analysis/analyze_s16.py` |
| MAGI-1 (study 18, preregistered) | `corr/magi_out/` | `analysis/analyze_magi.py`, `magi_curve.py` |
| FramePack content (study 17, preregistered) | `corr/fpc_out/` | `analysis/analyze_fpc.py` |
| New seeds, tone (study 20, preregistered) | `corr/s20_out/` | `analysis/analyze_s20.py` |
| New seeds, content (study 21, preregistered); second encoder | `corr/s21_out/`, `corr/enc2_out/` | `analysis/analyze_s21.py`, `encoder_robustness.py` |
| Krea Realtime 14B (study 22, preregistered) | `corr/krea_out/`, `corr/krea_out_siglip/` | `analysis/analyze_krea.py` |
| Attention-to-sink probes, Self Forcing and LongLive (exploratory) | `corr/sinkprobe_out/`, `corr/sinkprobe_long_out/`, `corr/llprobe_out/` | `analysis/analyze_sinkprobe.py`, `analyze_sinkprobe_long.py` |
| The sink's rotary position (study 25, preregistered) | `corr/s25_out/` | `analysis/analyze_s25.py`; figures `report/make_probe_fig.py`, `make_s25_frames_fig.py` |
| The 180-s horizon (study 24, preregistered; 60-s windows exploratory) | `corr/s24_out/` | `analysis/analyze_s24.py`, `robustness_s24_windows.py` |
| Out-of-sample burst times (study 26, preregistered; 120-s replication on study 6) | `corr/s26_out/`, `corr/s26_rep120_out/` | `analysis/analyze_s26.py` |
| A sink held at one distance (study 27, preregistered) | `corr/s27_out/` | `analysis/analyze_s27.py`, `s27_distances.py` |
| The sink's position in LongLive (study 28, preregistered) | `corr/s28_out/` | `analysis/analyze_s28.py`, `s27_distances.py` |
| Attractor offset on 40 prompts (study 30, preregistered) | `corr/s30_out/` | `analysis/analyze_s30.py` |
| Steady share to 240 s (study 29, preregistered) | `corr/s29_out/` | `analysis/analyze_s29.py` |
| A held sink as a lever (study 31, preregistered) | `corr/s31_out/` | `analysis/analyze_s31.py` |
| Window-matched attractor offset (study 32, preregistered) | `corr/s32_out/` | `analysis/analyze_s32.py` |
| Attractor offset against the window (study 33, preregistered) | `corr/s33_out/` | `analysis/analyze_s33.py` |
| One sink frequency held in two models (study 35, preregistered) | `corr/s35_out/` | `analysis/analyze_s35.py` |
| The offset beyond the released window (study 36, preregistered) | `corr/s36_out/` | `analysis/analyze_s36.py` |
| Probes of CausVid and Self Forcing GAN (exploratory) | `corr/probe37_out/` | `analysis/analyze_sinkprobe_long.py` |
| The held-sink distance chosen by one rule (study 37, preregistered) | `corr/s37_out/dstar.json` | `analysis/select37.py` |
| The held sink on new openings, four models (study 38, preregistered) | `corr/s38_out/` | `analysis/analyze_s38.py` |
| The held sink on prompt-only videos, VBench-style scores (study 39, preregistered) | `corr/s39_out/` | `analysis/analyze_s39.py`, `generation/vbench_lite.py` |
| Supplementary analyses: motion audit, opening source, μ over time, DINOv2 content (exploratory) | `corr/review3_out/` | `analysis/supplementary_analyses3.py`, `motion_audit.py`, `motion_summary.py`, `dino_content.py`, `analyze_dino_content.py` |
| Band probe: which temporal RoPE frequencies make the probe's peaks; probe peaks by layer, head and pass (exploratory) | `corr/sinkband_out/`, `corr/sinkprobe_out/sinkprobe_heads.json` | `analysis/analyze_sinkband.py`, `analyze_sinkprobe_heads.py` |
| MAGI-1 content and 60-s horizon (study 19, preregistered) | `corr/magi19_out/`, `corr/magi19_out_siglip/` (SigLIP rescoring) | `analysis/analyze_magi19.py` |
| One rotary frequency of the sink held in generation (study 34, preregistered) | `corr/s34_out/` | `analysis/analyze_s34.py` |
| Dose–response of tone persistence (study 23, preregistered) | `corr/s23_out/` | `analysis/analyze_s23.py` |

`corr/` holds all the opening-based experiments.

**Study 39, deviation 1.** The frozen `analyze_s39.py` read a per-frame motion series from every generator output, but `rf_gen_long.py` (Rolling Forcing, the reference) records none, so the script stopped on the first Rolling Forcing file. The fix reads motion only where it exists; the primary tests use the four sink models, which all record it, and are unaffected. The frozen version is in the git history (commit 9f2a27a).
