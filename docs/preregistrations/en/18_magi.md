> English translation of the frozen Chinese preregistration [`../18_magi.md`](../18_magi.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 18: MAGI-1 — a non-Wan autoregressive model, attention window versus full history (2026-10-25, frozen before generation)

## Background
- The paper's main limitation is that 6 of its 7 models are built on Wan2.1. FramePack is another base, but it is not a
  chunk-wise KV-cache model.
- MAGI-1 (Sand AI, 4.5B distilled) is an autoregressive video model with its own architecture:
  - 6 latent frames per chunk (24 frames, 1 s at 24 fps);
  - each chunk attends to earlier chunks (KV cache).
- The attention span is set by the runtime config:
  - **Released setting (mgw):** `noise2clean_kvrange [5, 4, 3, 2]` and `clean_chunk_kvrange 1`, i.e. at most the 4
    previous chunks (about 4 s), with no sink. The opening leaves the context after a few seconds.
  - **Full history (mgf):** `noise2clean_kvrange []` (the code's default range); every chunk attends to all earlier
    chunks, so the opening stays in the context throughout.
- Our theory: a model that relies only on a KV-cache window forgets the opening; when the opening stays in the
  context, a share is kept.
- **Smoke test** (rx00, unshifted, 30 s, A40, not a study scene):
  - mgw freezes after about 12 s (L\* stops changing);
  - mgf keeps moving, and by 20 s its content has become the generic prompt's city street;
  - runtime: 1.5 h for mgw, 2.3 h for mgf.
  - Freezing is the released model's behaviour over long generation. It is recorded with a predefined metric
    (secondary analysis) and is not an exclusion criterion.

## Design
- **Scenes:** the 12 scenes of preregistration 08, openings L\* −16 / +16, generic prompt, seed 0, 30 s, resolution
  432×768.
- **Generation:** `magi_gen.py` (MAGI-1 repository commit 0fcefde, unmodified).
  - flash-attn is the official prebuilt package 2.7.4.post1;
  - flashinfer is used only for `silu_and_mul`, which `shims_magi/flashinfer` implements in PyTorch with the same
    mathematics.
- **Execution:**
  - one task per scene × condition (both versions, arch_guard);
  - plus text-to-video starts from the generic prompt, seeds 0–3 per condition;
  - every task is restricted to the A100 group (A100, A800);
  - task list `jobs/magi.jsonl` (`make_magi_tasks.py`).
  - In total 48 + 8 = 56 videos.
- **Metrics:**
  - α(T) = (end_p16 − end_m16) / (open_p16 − open_m16);
  - open is the mean L\* of frames 1–8;
  - end is the mean L\* over [T−10, T) s for T ≥ 20, and over [T−1, T) for T < 20.
- **Collapse rule:** if either version has any 1-s mean L\* below 3 or above 97 within [0, 30) s, the scene is excluded
  from that condition.

## Hypotheses (`analyze_magi.py`, frozen; Holm correction over 3 tests)
- **H1 The released window forgets:** α(mgw, 30) − α(mgw, 5) < 0.
- **H2 Full history keeps more than the window:** α(mgf, 30) − α(mgw, 30) > 0.
- **H3 Full history keeps a share:** α(mgf, 30) > 0.
- **Tests:**
  - two-sided one-sample t-tests with the scene as the unit, scene-bootstrap 95% CIs;
  - a hypothesis holds if Holm p < 0.05 and the estimate is on the predicted side.

## Secondary analyses
- α of each condition at 1, 2, 5, 10, 20 and 30 s.
- Freezing: a video counts as frozen if the mean absolute frame-to-frame change of its mean L\* over [20, 30) s is
  below 0.01. The frozen fraction is reported per condition, and H1–H3 are repeated without frozen scenes.
- Attractor offset per condition: mean L\* of the fresh videos over [20, 30) s minus over [0, 2) s.
- Zero-drift level per condition (from the two versions), and the mean of |end − open|.

## Predictions
- **H1 holds.**
- **H2 holds.**
- **H3 is uncertain:** in the smoke test, content returned to the city even with full history; how much tone is kept is
  unknown.
- **Freezing:** mgw freezes more often than mgf.

## Frozen files (SHA-256)
```
cb7842463b06265d77a85de8c463943c9f8439f02fb270d0c2ea53f8a34f936d  analyze_magi.py
24f47bf25724ddb8166f08e8fe9c167e33a1a57a76cad185ade223ac01c6ba55  make_magi_tasks.py
bd5b465ccfd8a7492637702f20bb40ddd90f9f494a0fe00d5a8baa990d61313e  magi_gen.py
ff2a3e559b4bd5e5ab074a4ea30e2392b3bced2e85cc6db6fb106ced29418a5b  shims_magi/flashinfer/__init__.py
7e9625548bdc661f0c6f4b1a0cc8c45f833de5a9044da31091c1510fd72d8ef1  shims_magi/flashinfer/activation.py
23f471e6be9a8985e46c349c252b798e1b10228fe25f009962eb470c740a5242  shims_magi/flashinfer/gemm.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
a2ffe0864c446290dbfb5876a7407b9c32eceb0c050333ffa4893c184f72be8e  jobs/magi.jsonl
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
- MAGI-1 repository: https://github.com/SandAI-org/MAGI-1 commit 0fcefdef8ce2df37a3b8890979433c06eb003328; weights
  sand-ai/MAGI-1 ckpt/magi/4.5B_distill, ckpt/t5, ckpt/vae.
