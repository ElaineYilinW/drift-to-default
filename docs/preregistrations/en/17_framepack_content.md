> English translation of the frozen Chinese preregistration [`../17_framepack_content.md`](../17_framepack_content.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 17: which of FramePack's routes carries scene content? (2026-10-25, frozen before generation)

## Background
- Preregistration 11 showed that FramePack (HunyuanVideo architecture, the only non-Wan base) keeps 0.68 of the
  opening's brightness at 20 s. The opening reaches the model through two routes:
  - **context route:** the opening's VAE latent is a clean prefix in every section;
  - **embedding route:** the opening's SigLIP image embedding is a global condition of every section.
  A brightness shift that enters only through the embedding route has almost no effect (α = 0.01).
- But SigLIP is a semantic encoder: it may carry the scene class without carrying tone. If so, "the conditioning sets
  the default" splits by attribute on FramePack: the embedding route sets the content default but not the tone default.
- The content study of preregistration 09 used only Wan-architecture models. This study extends it to FramePack and
  separates the two routes.

## Design
- **Openings:** the 24 openings of preregistration 09 (6 classes × 4).
- **City images:**
  - candidates, in order: the generic-prompt videos that Self Forcing DMD generated from scratch in preregistration 09,
    fresh_sf60_s3 seeds 0–7, then fresh_sf60_s0 seeds 0–7;
  - frame 64 (4 s) of each is classified by CLIP ViT-L/14 among the 7 labels;
  - in order, the first 8 frames whose top-1 label is city with probability ≥ 0.5 become ctcity0–7
    (`fpc_city_openings.py`).
  - Opening ct{class}{k} is paired with ctcity{k} (k = 0–3).
- **Conditions** (FramePack-F1, generic city prompt, 20 s, seed 0, generator `fp_gen_c.py`):
  - both: context = opening, embedding = opening;
  - ctx: context = opening, embedding = the paired city image;
  - emb: context = the paired city image, embedding = opening;
  - alt: context = embedding = city image (8 city starts, giving the default).
- **Execution:**
  - both, ctx and emb of one opening run in one task (arch_guard);
  - every task is restricted to the Ada group (L40S, RTX 6000 Ada);
  - task lists come from `make_fpc_tasks.py`: a selects the city images, b generates (after a; its content is fully
    determined by the frozen scripts and the selected images), c runs CLIP (`clip_probe.py`).
  - In total 24 × 3 + 8 = 80 videos.
- **Dry run of the selection script:** before freezing, `fpc_city_openings.py` was run once into a temporary folder,
  only to check the code. It reads only existing videos of preregistration 09 and involves none of this study's results.

## Metrics (as preregistration 09, at 30 fps)
- Frame-level class score z = CLIP logit scale × (cos(frame, opening's class label) − cos(frame, city label)); frame f
  is (f − 9)/30 s after the opening.
- s[t0, t1): mean z over that window.
- s_open: mean z over frames 1–8 of the opening's both video (i.e. the opening image). The emb condition's video starts
  with the city image, so all three conditions use this value.
- μ[t0, t1): mean score of the 8 city starts in the same window.
- κ = (s − μ) / (s_open − μ); undefined when s_open − μ < 2.

## Hypotheses (`analyze_fpc.py`, frozen; Holm correction over 4 tests; window [15, 20) s)
- **H1 FramePack's content relaxes:** s(both) − s_open < 0.
- **H2 The context prefix carries content:** κ(ctx) > 0.
- **H3 The SigLIP embedding carries content:** κ(emb) > 0.
- **H4 The embedding adds content persistence:** κ(both) − κ(ctx) > 0.
- **Tests:**
  - two-sided one-sample t-tests with the opening as the unit;
  - a hypothesis holds if Holm p < 0.05 and the estimate is on the predicted side;
  - 95% CIs by bootstrap (openings resampled within class, city starts resampled too, 10,000 draws).

## Secondary analyses
- κ curves of every condition (windows [T−5, T), T = 5..20);
- additivity κ(ctx) + κ(emb) − κ(both);
- H2–H4 on tone-normalised embeddings;
- κ per class;
- top-1 accuracy of the opening's class among the 7 labels in [15, 20) s;
- κ(both, 20 s) next to Self Forcing DMD's κ at 20 s from preregistration 09 (descriptive).

## Predictions
- **H1 holds:** content relaxes somewhat, but κ(both) is high (about 0.6–0.8).
- **H2 holds:** κ(ctx) about 0.4–0.6.
- **H3 holds:** κ(emb) about 0.2–0.4. This is the opposite of tone: in preregistration 11, the embedding route had no
  effect on tone.
- **H4 holds.**

## Frozen files (SHA-256)
```
6211728708452dc91a6031469131d942f25c03955d82425297c7a34bc6eeee45  analyze_fpc.py
26173b900deb6ad3bdf3d91c51ef8e3607b8b45d46644d331c323261f2b53e42  fp_gen_c.py
f327b173a94cc4094f32559d679a6185d78a654b78253384e6608ffec490b61e  fpc_city_openings.py
dd9827583110472ba61562d5d3f563a1c2b958c2d8c2b6f719ffd8e104e88cdd  make_fpc_tasks.py
002fe72cfc2c768d72e6f17f53228fb66664278ba12f6a6a1a0008dfc700d08f  clip_probe.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
7da851fd2c67c00e66c05ab4745e269fd619f844b0da3834342632460cc2e53e  jobs/fpc_a.jsonl
6a523cfce79e7d7d5afa36b085428939d499342ddf65ceac8a95c4b4eebaee68  jobs/fpc_c.jsonl
c41b6d71c33c9428cd8c8e1a559dc44cb72a9daae5b075a637e668ca4c5074d1  prompts/content_runs.jsonl
```
- The generation task list `jobs/fpc_b.jsonl` is written by `make_fpc_tasks.py b` after the city images are selected;
  its content is fully determined by the frozen scripts and the selected images.
