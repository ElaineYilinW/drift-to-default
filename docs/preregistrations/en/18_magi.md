# Preregistration 18: MAGI-1 — a non-Wan autoregressive model, attention window versus full history (2026-10-25, frozen before generation)

*English translation; the Chinese file `../18_magi.md` is the frozen, authoritative version.*

## Background
- The paper's main limitation is that 6 of its 7 models are based on Wan2.1. FramePack is another base, but it is not a chunk-wise KV-cache model.
- MAGI-1 (Sand AI, 4.5B distilled version) is an autoregressive video model with its own architecture:
  - 6 latent frames per chunk (24 frames, 1 s at 24 fps);
  - each chunk attends to the earlier chunks (KV cache).
- The attention range is set by the run configuration:
  - **Released setting (mgw):** `noise2clean_kvrange [5, 4, 3, 2]`, `clean_chunk_kvrange 1`, i.e. at most the 4 previous chunks are seen (about 4 s), with no sink. The opening leaves the context after a few seconds.
  - **Full history (mgf):** `noise2clean_kvrange []` (the code's default range); each chunk sees all previous chunks, and the opening is always in the context.
- Our theory: a model that relies only on a KV-cache window forgets the opening; when the opening stays in the context, a share is kept.
- **Smoke test** (rx00, unshifted, 30 s, A40, not a scene of this study):
  - mgw freezes after about 12 s (L\* no longer changes);
  - mgf keeps moving, and by 20 s the content has become the generic prompt's city street;
  - runtime: mgw 1.5 h, mgf 2.3 h.
  - Freezing is behaviour of the released model under long generation; it is recorded with a predefined metric (secondary analysis) and is not used for exclusion.

## Design
- **Scenes:** the 12 scenes of preregistration 08, openings L\* −16 / +16, generic prompt, seed 0, 30 s, resolution 432×768.
- **Generation:** `magi_gen.py` (MAGI-1 repository commit 0fcefde, unmodified).
  - flash-attn uses the official prebuilt package 2.7.4.post1;
  - flashinfer is only used for `silu_and_mul`, which `shims_magi/flashinfer` implements in PyTorch, mathematically identical.
- **Execution:**
  - one task per scene × condition (both versions, arch_guard);
  - in addition, generic-prompt generations from scratch (text-to-video), seeds 0–3 per condition;
  - all tasks are pinned to the A100 group (A100, A800);
  - task list `jobs/magi.jsonl` (`make_magi_tasks.py`).
  - 48 + 8 = 56 videos in total.
- **Metrics:**
  - α(T) = (p16 end − m16 end) / (p16 opening − m16 opening);
  - the opening is the mean L\* of frames 1–8;
  - the end is the mean L\* over [T−10, T) s for T ≥ 20, and over [T−1, T) for T < 20.
- **Collapse rule:** if either version has any 1-s mean L\* below 3 or above 97 within [0, 30) s, the scene is excluded from that condition.

## Hypotheses (`analyze_magi.py`, frozen; Holm correction, 3 tests in total)
- **H1 the released window setting forgets:** α(mgw, 30) − α(mgw, 5) < 0.
- **H2 full history keeps more than the window:** α(mgf, 30) − α(mgw, 30) > 0.
- **H3 full history keeps a share:** α(mgf, 30) > 0.
- **Test method:**
  - two-sided one-sample t-tests over scenes, scene-bootstrap 95% CIs;
  - supported if Holm p < 0.05 and the estimate is on the predicted side.

## Secondary analyses
- α of each condition at 1, 2, 5, 10, 20, 30 s.
- Freezing: a video is recorded as frozen if the mean absolute frame-to-frame change of its mean L\* within [20, 30) s is below 0.01. The frozen share of each condition is reported, and H1–H3 are redone after excluding frozen scenes.
- Attractor offset of each condition: the difference between the mean L\* of the from-scratch videos over [20, 30) s and over [0, 2) s.
- Zero-drift level of each condition (obtained from the two versions), and the mean of |end − opening|.

## Predictions
- **H1 holds.**
- **H2 holds.**
- **H3 uncertain:** in the smoke test the content still returned to the city under full history; how much tone is kept is unknown.
- **Freezing:** the frozen share of mgw is higher than that of mgf.

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
- MAGI-1 repository: https://github.com/SandAI-org/MAGI-1 commit 0fcefdef8ce2df37a3b8890979433c06eb003328; weights sand-ai/MAGI-1 ckpt/magi/4.5B_distill, ckpt/t5, ckpt/vae.
