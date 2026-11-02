# Generation

Scripts that generate videos and extract the features the analyses read. The `*_long`, `*_sinkpos`, `*_bandhold` and `*_probe` scripts are separate copies because the earlier versions are frozen by the hashes in their preregistrations.

| Model or task | Scripts |
|---|---|
| Self Forcing | `gen.py`, `correct_gen.py`, `correct_gen2.py`, `hold_gen*.py` |
| LongLive | `ll_gen.py`, `ll_hold_gen*.py` |
| Rolling Forcing | `rf_gen.py`, `rf_gen_long.py` |
| SkyReels-V2 | `sky_gen.py`, `sky_gen_x.py`, `sky_gen_x_cpupost.py` |
| FramePack | `fp_gen.py`, `fp_gen_c.py` |
| MAGI-1 | `magi_gen.py` |
| Krea Realtime | `krea_gen.py` |
| Matrix-Game-2.0 | `mg_gen.py` |
| Context and prompt interventions | `ctx_swap.py`, `ctx_swap2.py`, `pswap.py` |
| Latent brightness | `latent_summary.py`, `dose_response.py`, `jacobian.py` |
| Features | `features.py`, `clip_probe.py`, `siglip_probe.py`, `stats_util.py` |
| Openings and prompts | `content_openings.py`, `fpc_city_openings.py`, `pgen_openings.py`, `select_relax_openings.py`, `caption_vlm.py` |
| Correction baselines | `posthoc_baselines.py` |
| Pairing | `arch_guard.py` keeps the paired arms of a task on one GPU architecture |
