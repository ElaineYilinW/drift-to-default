> English translation of the frozen Chinese preregistration [`../25_sink_position.md`](../25_sink_position.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 25: do the bursts of a sink follow its rotary distance, and does a sink held at a probe peak keep more than one held at a trough? (Self Forcing, 60 s) (2026-10-28, frozen before generation)

## Background
- With the sink at its original (absolute) rotary positions, persistence jumps near 17, 33 and 51 s ("bursts"), and
  the opening's scene content returns at these times:
  - in five Self Forcing checkpoints and in LongLive (preregistrations 10, 16, 21);
  - never with a re-rotated sink or without a sink.
- Proposed mechanism: as generation proceeds, the sink's keys drift ever further from the current chunk; at some
  query–sink distances the model gives the sink much of its attention, which re-injects the opening.
- **Exploratory attention probe** (`jobs/run_sinkprobe.sh`, `analyze_sinkprobe.py`; not part of any
  preregistration):
  - setup: 4 unshifted openings (rx00, rx01, gx00, gx01), sink 3 at its original positions; 20 chunks per opening
    (starts 12, 24, …, 240), 128 random queries per chunk, 30 layers, 5 attention calls (4 denoising steps + 1
    context pass);
  - measure: for each distance D in a grid (3–300 latent frames), the share of attention the queries would give to
    the sink if it sat at distance D (sink vs window logsumexp);
  - result (`sinkprobe_out.json`): the share peaks sharply at D = 71, 134, 203, 266 (0.109, 0.136, 0.150, 0.098;
    median over 3 ≤ D ≤ 300: 0.073), each peak about 4 latent frames wide; a trough precedes each peak (e.g. 0.052
    at D = 187, 0.055 at D = 123);
  - the peaks sit at exactly the same distances in all 4 openings and all chunks, so they are a property of the
    model's positional code, not of content;
  - at the original positions, the chunks' natural distance reaches 71, 134 and 203 at 16.5, 32.25 and 49.5 s after
    the opening: the observed burst times.
- **Consistency check:** the patched Self-Forcing-b without switches reproduces the output of preregistration 10 bit
  for bit (rx00, sink 3, m16, A40).
- **Design check (existing data of preregistration 10):** with the burst windows defined below, the burst contrast is
  +0.33 [0.29, 0.38] for sink 3 at its original positions and −0.04 for the re-rotated sink.
- This study intervenes on the sink's position to test two things:
  - that the bursts are set by the distance, not by the time since the opening or by content;
  - whether a sink held at a peak distance keeps more than one held at a trough or next to the window.

## Design
- **Scenes:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), generic prompt (`prompts/relax.jsonl`).
- **Model:** Self Forcing DMD, sink 3, window 21, 60 s (243 latent frames).
  - Generator `hold_gen_sinkpos.py`, run in Self-Forcing-b, a copy in which only `wan/modules/causal_model.py` is a
    real file, patched by `jobs/patch_sf_sinkpos.py`.
  - The patch rotates only a copy of the sink keys (the cache keeps the original keys), as in the re-rotation of
    `patch_sf.py`.
- **Openings:** L\* shifted by −16 / +16 (versions m16, p16), as in preregistrations 06 and 10.
- **Seed:** 0.
- **Arms:**
  - sp_abs: original positions (no switch);
  - sp_off18: the sink 18 latent frames further back (`--sink_mode off:18`); distance = natural distance + 18;
  - sp_off36: 36 latent frames further back (`off:36`);
  - sp_d203: the sink held at distance 203 (a probe peak, `dist:203`);
  - sp_d187: held at distance 187 (the trough before that peak, `dist:187`);
  - sp_d20: held at distance 20 (next to the window, as re-rotation, `dist:20`).
- **Execution:**
  - one task per scene with all arms and versions (12 videos); arch_guard keeps all of a scene's outputs on one GPU
    architecture;
  - any GPU with ≥ 40 GB and bf16;
  - video files are written only for rx00 and gx00;
  - the task list is built by `make_s25_tasks.py` (`jobs/s25.jsonl`, 24 tasks);
  - 24 × 6 × 2 = 288 videos in total.
- The generator also records two per-frame quantities in the npz: motion (mean absolute grey change from the previous
  frame, 0–100) and open_diff (mean absolute grey difference from frame 8, the last opening frame).

## Measures
- α[t0, t1) = (end_p16 − end_m16) / (open_p16 − open_m16):
  - open is the mean L\* of frames 1–8;
  - end is the mean L\* over seconds [t0, t1) after the opening;
  - undefined if open_p16 − open_m16 < 4 L\*.
- No collapse exclusion.
- **Burst onsets:**
  - a chunk starting at latent frame c is at distance c + 2 + δ from sink frame 0 (δ is the sink's extra offset: 0,
    18, 36) and starts (c − 3)/4 s after the opening;
  - so the bursts start at τ_k(δ) = (P_k − 5 − δ)/4 s, P = 71, 134, 203, 266;
  - burst window W_k = [τ_k + 0.5, τ_k + 2.5) s, kept if it lies within [5, 60) s:
    - δ = 0: 16.5, 32.25, 49.5 s;
    - δ = 18: 12, 27.75, 45 s;
    - δ = 36: 7.5, 23.25, 40.5, 56.25 s;
  - A_X(δ): the mean α of arm X over the windows of δ (one value per scene).

## Hypotheses (`analyze_s25.py`, frozen; Holm over 4 tests)
- **H1 the bursts follow an 18-frame shift:** [A_off18(18) − A_off18(0)] − [A_abs(18) − A_abs(0)] > 0.
- **H2 the bursts follow a 36-frame shift:** [A_off36(36) − A_off36(0)] − [A_abs(36) − A_abs(0)] > 0.
- **H3 a peak distance keeps more than a trough:** α(sp_d203, [50, 60)) − α(sp_d187, [50, 60)) > 0.
- **H4 a peak distance keeps more than distance 20:** α(sp_d203, [50, 60)) − α(sp_d20, [50, 60)) > 0.
- **Tests:**
  - two-sided one-sample t-tests over scenes;
  - supported if Holm p < 0.05 and the estimate is > 0;
  - 95% CIs by scene bootstrap (10,000 draws).
- If the bursts were set by time (or by how the content changes after the opening), the two brackets of H1 and H2
  would be equal and the estimates 0.

## Secondary analyses
- H3 and H4 over [40, 50) s;
- each arm's contrast at its own burst onsets, mean_k [α(W_k) − α([τ_k − 2.5, τ_k − 0.5))]:
  - sp_abs, sp_off18 and sp_off36 at their own δ;
  - sp_d203, sp_d187 and sp_d20 at the onsets of δ = 0;
- times of 1-s jumps above 0.15 per arm, against the predicted onsets;
- α per arm at T = 10, 20, …, 60 s (10-s windows) and 1-s curves;
- probe check: over t = 6…59 s, the correlation between the first differences of the mean 1-s α curve of sp_abs,
  sp_off18 and sp_off36 and the first differences of the probe's predicted sink share for δ = 0, 18, 36;
- mean motion and open_diff over [50, 60) s per arm (both versions): does holding the sink at a peak freeze the video
  or copy the opening?

## Predictions
- **H1, H2 supported:** the bursts come about 4.5 and 9 s earlier.
- **H3 supported:** at distance 203 the sink receives about three times the attention it gets at 187.
  - The size is uncertain: constant attention to the sink could also push the video toward copying the opening (see
    motion and open_diff).
- **H4 supported.**
- **Rules for interpretation:**
  - if H1 and H2 are not supported (the bursts stay at their times), the bursts are not set by the distance, and the
    paper's positional account is withdrawn;
  - if H1 and H2 are supported but H3 is not, the burst times are set by the distance, but parking the sink at a
    peak does not keep more steadily: a burst is a transient response to passing a peak;
  - if H3 and H4 are supported, placing the sink at a peak distance is a training-free way to keep the opening
    steadily; motion and open_diff show whether it has a cost.

## Frozen files (SHA-256)
```
043d21dd2ac517c09385941448abc5c195709e3bd4a5a000b99f9fc267a65db3  analyze_s25.py
deee7745c9fc72d69c678df31d1780306166f0ad8f1bab8d6be147ff163832f9  make_s25_tasks.py
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
1b14c939ebd86f8efb4f9a78bad8a8bd09ccaca7953aaf42457c02006f9ab7dc  jobs/patch_sf_sinkpos.py
a57bcf3d47b7952b81c80eca9ee045abba1d28c07844fc51a885ae1543a7a824  Self-Forcing-b/wan/modules/causal_model.py (patched)
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
562e2d56d068e708c126797d5be970f47bde2d20323dd05af12abda78fe91f4d  jobs/s25.jsonl
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
c1a8dcf07bbe2eef278624448bef869f7b354ce57b9d69540980b5d749af4a15  analyze_sinkprobe.py
612d81c0ae326b8e545a3cc07e044553fcf982f404154ff9e51796286f5b6571  jobs/run_sinkprobe.sh
07f5bb2c43ae1992086bbf08ecf2cd4a600d50519fecbcaa798e3a66632c2e7e  sinkprobe_out.json (exploratory probe result)
```
