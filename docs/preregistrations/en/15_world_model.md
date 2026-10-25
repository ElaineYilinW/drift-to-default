> English translation of the frozen Chinese preregistration [`../15_world_model.md`](../15_world_model.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 15: an action-conditioned world model — what keeps the opening? (2026-10-24, frozen before generation)

## Background
- A review pointed out that the paper's implications for "world models" were not tested on any action-conditioned
  world model.
- Matrix-Game-2.0 (Skywork, open source) is a real-time interactive world model: Wan architecture with the text branch
  removed, generating frame by frame from keyboard and mouse actions, few-step autoregressive (3 latent frames per chunk,
  25 fps, 352×640). It has **no text prompt**.
- The opening image reaches the model through two routes:
  - **context:** the opening's VAE latent (with a mask) is concatenated to the first latent frame and afterwards lives
    only in the KV cache, whose local window is 6 latent frames with no sink, so the opening leaves the context after
    about one second;
  - **conditioning:** the opening's CLIP image embedding is a global condition of every chunk.
- Our account: the conditioning sets the default and the context sets how much is kept. Predictions: a shift sent only
  through the context is forgotten; a shift sent only through the conditioning moves the default; the conditioning
  route adds persistence. This is the test of study 11 (FramePack) on a world model.
- Smoke test (rx00, unshifted, 60 s): black frames after about 37 s, blow-out after 46 s, black after 52 s, beyond the
  training length. The endpoint is therefore 30 s, with an exclusion rule fixed in advance.

## Design
- **Scenes:** the 24 scenes of study 06; constant "forward" action, no camera motion; seed 0; 30 s (189 latent frames).
  Openings L\* −16 and +16.
- **Conditions:** mg_both (the shift reaches both routes), mg_ctx (only the context latent; CLIP embedding of the
  unshifted image), mg_emb (only the CLIP embedding).
- **Execution:** the three conditions of a scene in one task, starting with arch_guard (`make_mg_tasks.py`,
  `jobs/mg.jsonl`). Generator `mg_gen.py` (Matrix-Game-2 code; flash_attn replaced by a PyTorch SDPA stand-in in
  `shims/flash_attn`, mathematically identical). 24 × 3 × 2 = 144 videos.
- **Measure:** α(T) = (p16 end − m16 end) / (p16 opening − m16 opening); opening = L\* of the shifted image; end = mean L\*
  over seconds [T−w, T) after the opening, w = 10 s for T ≥ 20 and 1 s otherwise.
- **Exclusion:** a scene is dropped from every analysis if any of its videos has a 1-s mean L\* below 3 within the first
  30 s (collapse to black).

## Hypotheses (`analyze_mg.py`, frozen; Holm over 3 tests)
- **H1, the opening held only in the context is forgotten:** α(mg_ctx, 30 s) − α(mg_ctx, 5 s) < 0.
- **H2, the conditioning sets the default:** α(mg_emb, 30 s) > 0.
- **H3, the conditioning route adds persistence:** α(mg_both, 30 s) − α(mg_ctx, 30 s) > 0.
- **Tests:** two-sided one-sample t-tests over scenes, scene-bootstrap 95% CIs; supported if Holm p < 0.05 and the
  estimate is on the predicted side.
- **Secondary:** α of each condition at 2, 5, 10, 20, 30 s; additivity α(ctx) + α(emb) − α(both); per-second curves;
  number of excluded scenes.
- **Predictions:** H1–H3 hold.

## Frozen files
See the hash block in the Chinese original.
