> English translation of the frozen Chinese preregistration [`../11_framepack_channels.md`](../11_framepack_channels.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 11: a non-Wan model, FramePack, and the two routes of the opening (2026-10-24, frozen before generation)

## Background
- Every model so far, SkyReels-V2-DF-1.3B included, derives from Wan2.1-1.3B. Whether the findings carry over to other
  base models is the most obvious open question.
- FramePack (HunyuanVideo-based, 13B) is designed for long videos. Every 33-frame section is conditioned on the opening
  image through **two routes**:
  - **context:** the opening's VAE latent is a clean prefix frame in the transformer's context;
  - **conditioning:** the opening's SigLIP embedding is a global condition, of the same kind as the text prompt.
- Our account is that the conditioning (the prompt) sets the default μ and the context sets how much is kept (α).
  FramePack lets the two routes be manipulated separately: shift the opening only in the context, or only in the
  conditioning.
- Prediction: an opening in the conditioning moves the default, so shifting only the embedding changes the video
  (α_emb > 0); shifting both keeps the most.

## Design
- **Scenes and prompt:** the 12 scenes of study 08 (rx02, rx06, rx08, rx10, gx02, gx06, gx08, gx10, rx01, rx03, gx01,
  gx03), generic prompt, seed 0, 20 s, 30 fps, 832×480. Openings L\* −16 and +16.
- **Conditions:**
  - fpv_both: FramePack-F1 (forward sampling, in time order like the KV-cache models), shift through both routes;
  - fpv_ctx: shift only the context latent (SigLIP embedding of the unshifted image);
  - fpv_emb: shift only the SigLIP embedding (context latent of the unshifted image);
  - fpi_both: FramePackI2V (inverted anti-drifting sampling, generated from the end of the video back toward the
    opening), both routes.
- **Settings:** the FramePack demo defaults: 25 steps, distilled guidance 10, no CFG (`fp_gen.py`).
- **Execution:** fpv_both and fpv_ctx share a task (arch_guard keeps them on one GPU architecture); fpv_emb and
  fpi_both each have their own task (`make_fp_tasks.py`, `jobs/fp.jsonl`). 12 × 4 × 2 = 96 videos.
- **Measure:** α(T) = (p16 end − m16 end) / (p16 opening − m16 opening), with the opening the L\* of the shifted image
  (json open_L_shifted) and the end the mean L\* over seconds [T−5, T) after the opening.

## Hypotheses (`analyze_fp.py`, frozen; Holm over 3 tests)
- **H1, FramePack keeps most of the opening:** α(fpv_both, 20 s) > 0.5.
- **H2, the conditioning sets the default:** α(fpv_emb, 20 s) > 0.
- **H3, the conditioning route adds to persistence:** α(fpv_both, 20 s) − α(fpv_ctx, 20 s) > 0.
- **Tests:** two-sided one-sample t-tests over scenes, scene-bootstrap 95% CIs; supported if Holm p < 0.05 and the
  estimate is on the predicted side.
- **Secondary:** α of each condition at 5, 10, 15 and 20 s; additivity α(ctx) + α(emb) − α(both) at 20 s;
  α(fpi_both) − α(fpv_both) at 20 s. The last two span tasks and may cross GPU architectures (this adds variance, not
  bias).
- **Predictions:** H1–H3 hold.

## Frozen files
See the hash block in the Chinese original.
