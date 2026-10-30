# Preregistration 15: an action-conditioned world model — what keeps the opening? (2026-10-24, frozen before generation)

*English translation; the Chinese file `../15_world_model.md` is the frozen, authoritative version.*

## Background
- A review pointed out that the paper's implications for "world models" have not been tested on any action-conditioned world model.
- Matrix-Game-2.0 (Skywork, open source) is a real-time, interactive world model: Wan architecture with the text branch removed, generating frame by frame from keyboard and mouse actions, few-step autoregressive (3 latent frames per chunk, 25 fps, 352×640). It has **no text prompt**.
- The opening image enters the model through two routes:
  - **context route:** the opening's VAE latent (with a mask) is concatenated to the first latent frame and afterwards exists only in the KV cache; the cache's local window is only 6 latent frames, with no sink, so the opening leaves the context after about 1 second;
  - **conditioning route:** the opening's CLIP image embedding, a global condition of every chunk.
- Our theory: the conditioning sets the default, the context sets how much is kept. Hence the predictions: a shift sent only through the context is forgotten; a shift sent only through the conditioning moves the default; the conditioning route adds persistence. This is the same test as preregistration 11 (FramePack), on a world model instead.
- Smoke test (rx00, unshifted, 60 s): black frames appear after about 37 s, over-exposure after 46 s, all black after 52 s — beyond the training length. The endpoint is therefore set to 30 s, and an exclusion rule is fixed in advance.

## Design
- **Scenes:** the 24 scenes of preregistration 06; the action is constantly "forward", with no camera turning; seed 0; 30 s (189 latent frames). Openings L\* −16, +16.
- **Conditions:** mg_both (the shift enters both routes), mg_ctx (it enters only the context latent; the CLIP embedding uses the unshifted image), mg_emb (it enters only the CLIP embedding).
- **Execution:** the three conditions of a scene are in the same task, which starts by running arch_guard (`make_mg_tasks.py`, `jobs/mg.jsonl`). Generation script `mg_gen.py` (Matrix-Game-2 code; flash_attn is replaced by the PyTorch SDPA implementation in `shims/flash_attn`, mathematically identical). 24 × 3 × 2 = 144 videos in total.
- **Measure:** α(T) = (p16 end − m16 end) / (p16 opening − m16 opening); the opening is the L\* of the shifted image; the end is the mean L\* over [T−w, T) s after the opening, with w = 10 s for T ≥ 20 and w = 1 s otherwise.
- **Exclusion rule:** if any video of a scene has any 1-s mean L\* below 3 within the first 30 s (collapse to black), the scene is excluded from all analyses.

## Hypotheses (`analyze_mg.py`, frozen; Holm correction, 3 tests in total)
- **H1 an opening held only in the context is forgotten:** α(mg_ctx, 30 s) − α(mg_ctx, 5 s) < 0.
- **H2 the conditioning sets the default:** α(mg_emb, 30 s) > 0.
- **H3 the conditioning route adds persistence:** α(mg_both, 30 s) − α(mg_ctx, 30 s) > 0.
- **Test method:** two-sided one-sample t-test over scenes, scene-bootstrap 95% CI; supported if Holm p < 0.05 and the estimate is on the predicted side.
- **Secondary analyses:** α of each condition at 2, 5, 10, 20, 30 s; additivity α(ctx) + α(emb) − α(both); per-second curves; number of excluded scenes.
- **Predictions:** H1–H3 hold.

## Frozen files (SHA-256)
```
1a7c0f7eec567eed67815f5608e09b07a609855dd2a3b6689a9bb20bffd5263a  analyze_mg.py
9ca9aa7c46da83b167f78662c53f449e7a692d1d0575a478dd4111d8a8574eed  mg_gen.py
ca38114459e6afa6bc8a9599eab495df44296ef603f8a5f257ef1367f77787f2  make_mg_tasks.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
70e291f5489fe5e34e691dfbd69b34554425df7a44d33cfea2fed22c9e297402  shims/flash_attn/__init__.py
47ec9326c8522d9ae63ef2a7b1ab536c0c4eb501a6fd052bdd9b5c6776c4c991  jobs/mg.jsonl
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
