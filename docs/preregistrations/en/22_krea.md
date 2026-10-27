> English translation of the frozen Chinese preregistration [`../22_krea.md`](../22_krea.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 22: does the 14B model Krea Realtime forget the opening, and does a first-frame anchor keep a share? (2026-10-27, frozen before generation)

## Background
- Six of the eight models in the paper are built on Wan2.1-1.3B. Whether scale changes the conclusions has not been
  tested.
- Krea Realtime 14B is an autoregressive model distilled from Wan2.1-T2V-14B with Self Forcing; its weights and
  inference code are public. Its long-video design differs from the models in the paper:
  - before every block (3 latent frames) it **recomputes** the KV cache from the clean latents of the last 3 latent
    frames, with positions restarting at 0, so frames that have left the window cannot leak through the cache;
  - the window's first context frame is the oldest decoded frame of the window re-encoded (the released default), so
    the opening leaves the context after two blocks;
  - the released code has a `keep_first_frame` option: the opening's first latent stays the first context frame
    (at the position just before the window, i.e. a re-rotated sink, but one third of the context). Krea's technical
    write-up says this anchoring "limits how far the video drifts from the first frame", at the cost of artifacts.
- The paper's conclusions make clear predictions:
  - a model that sees the opening only through its window forgets it (MAGI-1's released window, Self Forcing without
    a sink);
  - an opening that stays in the context keeps a share (re-rotated sink about 0.2, MAGI-1 full history 0.64).
- This study tests both on the 14B model, for tone and for scene content.

## Design
- **Model:** Krea Realtime 14B (`krea-realtime-video-14b.safetensors`), the released inference code
  (GenerationSession of `release_server.py`), config `self_forcing_server_14b.yaml`:
  - 3 latent frames per block, kv_cache_num_frames = 3, KV recomputation;
  - denoising steps and timesteps at the GenerateParams defaults;
  - 832 × 480, 16 fps.
  - Argon can only install torch 2.6, so `jobs/patch_krea.py` makes two compatibility changes (a dynamo config name;
    dropping one flex_attention speed hint) that do not change the computation.
  - To fit 48 GB of GPU memory: the text encoder runs once on the GPU and is freed; the 14B transformer is built on
    the CPU, loaded, cast to bf16 and moved to the GPU. This computes the same as the released code.
  - The released code keeps the unfused q/k/v projections after fusing them (about 6 GB, never used again); they are
    deleted before the move to the GPU.
  - With a start frame, the start frame's 3 latent frames take the first block's slots of the released code's latent
    buffer, so the last block reads an empty noise slice and fails. The buffer gets one extra block; the number of
    generated blocks is unchanged.
- **Arms:**
  - kr60_none: the released default (the first context frame re-encoded);
  - kr60_first: keep_first_frame (the opening's first latent stays in the context).
- **Opening:** the opening image enters through the released start_frame path (9 copies VAE-encoded as the first 3
  latent frames). 8 copies of the opening image are prepended to the output, so frames 0–8 are the opening (as for
  MAGI-1).
- **A tone:** the 24 scenes of preregistration 06 (rx00–rx11, gx00–gx11), openings at L\* −16 / +16, generic prompt,
  60 s, seed 0. 24 × 2 × 2 = 96 videos.
- **B content:** the 24 openings of preregistration 09, generic city prompt (arm mis), 60 s, seed 0; 8 fresh starts of
  the generic prompt per arm (seeds 0–7). 24 × 2 + 2 × 8 = 64 videos. The primary analysis uses CLIP ViT-L/14
  embeddings (`clip_probe.py`); SigLIP so400m embeddings (`siglip_probe.py`) are computed too, for a secondary analysis
  only.
- **Execution:**
  - A: one task per scene and arm (both versions in one task); B: one task per opening (both arms); fresh starts one
    task per arm;
  - GPU: A40 only (one architecture, checked by arch_guard);
  - task lists from `make_krea_tasks.py` (`jobs/krea_a.jsonl`; embeddings `jobs/krea_e.jsonl`).
- **Pilot:** before generation only one scene (rx00) was run for 20 s, to check memory, speed and that the code runs
  (not part of the analysis):
  - peak GPU memory 40.2 GiB on an A40, about 12 s per block (about 16 min per 60-s video);
  - in both arms α fell to about 0.1 within about 6 s: in [10, 20) s it was −0.11 without the anchor and −0.02 with
    the first-frame anchor; in both arms the scene became a generic city street within about 6 s.

## Measures
- **Tone:** α(T) = (p16 end − m16 end)/(p16 opening − m16 opening); the opening is the mean L\* of frames 1–8, the end
  the mean over seconds [T − 10, T) after the opening.
- **Content (as in preregistration 16 A):**
  - class score z = scale × (cos(frame, class label) − cos(frame, city label));
  - s[t0, t1) is the mean z over seconds [t0, t1) after the opening (frame f ↦ (f − 9)/16 s), s_open the mean over
    frames 1–8;
  - μ is the mean of the same window over the fresh starts of the same arm;
  - κ = (s − μ)/(s_open − μ), undefined when s_open − μ < 2.
- No collapse exclusion rule; the number of collapsed scenes is reported as a secondary result.

## Hypotheses (`analyze_krea.py`, frozen; Holm over 5 tests)
- **A1 the released window forgets the opening:** α(kr60_none, 60) − α(kr60_none, 10) < 0.
- **A2 the first-frame anchor keeps more:** α(kr60_first, 60) − α(kr60_none, 60) > 0.
- **A3 the anchor keeps a share:** α(kr60_first, 60) > 0.
- **B1 content relaxes under the release:** s[55, 60)(kr60_none) − s_open < 0.
- **B2 the anchor keeps more content:** κ[55, 60)(kr60_first) − κ[55, 60)(kr60_none) > 0.
- **Tests:**
  - two-sided one-sample t-tests over scenes (A) or openings (B);
  - supported if Holm p < 0.05 and the estimate is on the predicted side;
  - 95% CIs by bootstrap (10,000 draws; A resamples scenes, B resamples openings within class and the fresh seeds).

## Secondary analyses
- α at 10, 20, …, 60 s and 1-s curves per arm;
- A1–A3 in the [40, 50) s window;
- κ curves per arm (T = 5…60);
- B1 and B2 on tone-normalised embeddings;
- top-1 accuracy of the opening's class in [55, 60) s; per-class κ;
- B1 and B2 with SigLIP embeddings (encoder robustness);
- collapse check: the number of scenes in which a 1-s mean L\* of either version falls below 3 or above 97.

## Predictions
- **A1 supported:** α near 0 at 60 s under the released setting.
- **A2, A3:** the paper's conclusions predict support (an opening that stays in the context keeps a share). But in the
  one-scene pilot the anchor kept no tone after 10 s, so this is uncertain. If they fail, an anchor of one latent frame
  placed right before the window is not enough to keep the opening in this model.
- **B1 supported.**
- **B2:** as A2, uncertain.

## Frozen files (SHA-256)
See the Chinese file.
