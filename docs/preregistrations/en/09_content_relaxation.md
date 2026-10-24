# Preregistration 09: Does scene content also relax to the prompt's default? (2026-10-24, frozen before generation)

*English translation of [../09_content_relaxation.md](../09_content_relaxation.md); the Chinese original and its hashes are binding.*

## Background
- Studies 06/07 showed that brightness and colour temperature relax toward a default set by the prompt, and that the
  attention sink keeps a share of the opening (α = w_s/(1−w_w)).
- These are global tone statistics. If **scene content** (snow, desert, forest, ...) also relaxes, "relaxation to a
  default" is a general property of these models rather than a colour artefact.
- This study measures with CLIP how much of the opening's scene class survives a rollout, with "generic city prompt vs
  the opening's own class prompt" as a paired manipulation.

## Design
- **Classes (6):** snow, desert, forest, coast, mountain, farmland. Prompts use the template of study 06's generic
  prompt with only the place and the visible objects replaced (`prompts/content.jsonl`).
- **Openings:** Self Forcing DMD generates 5-s fresh videos from each class prompt (seeds 0–15); frame 64 (4 s) is
  classified by CLIP ViT-L/14 among 7 labels (6 classes + city). In seed order, the first 4 frames whose top-1 label is
  their own class with probability ≥ 0.5 become openings (`content_openings.py`). 24 openings.
- **Continuations:** each opening is held still for 9 frames (as before) and continued under two prompts:
  mis, study 06's generic city prompt; mat, the opening's own class prompt.
- **Configurations (8):** Self Forcing DMD with sink 3 / no sink (60 s); LongLive sink 3 / no sink; CausVid;
  Self Forcing GAN; Self Forcing SiD; Rolling Forcing (31 s each). Seed 0.
- **Defaults:** 8 fresh generations of the generic prompt per configuration (seeds 0–7), same length.
- **Execution:** all arms of one opening × model family run in one task; arch_guard keeps paired arms on one GPU
  architecture (`make_content_tasks.py`). 24 × 2 × 8 = 384 continuations + 64 fresh videos.
- **Measures:**
  - frame score z = CLIP logit scale × (cos(frame, class label) − cos(frame, city label)), the log-odds of the class
    against city; frames 1–8 and every 4th frame from 9 on (`clip_probe.py`);
  - s(T): mean z over seconds [T−5, T) after the held opening; s_open: mean over frames 1–8; μ(T): the same window
    score of the configuration's fresh generic videos;
  - content persistence κ = (s(T) − μ(T)) / (s_open − μ(T)); openings with s_open − μ < 2 are excluded from κ analyses.

## Hypotheses (`analyze_content.py`, frozen; Holm over 3 tests)
- **H1, the prompt sets the content default:** Self Forcing DMD sink 3, T = 30 s, s_mat − s_mis > 0.
- **H2, the sink keeps content (LongLive):** T = 30 s, κ(ll_s3, mis) − κ(ll_s0, mis) > 0.
- **H3, the sink keeps content (Self Forcing):** T = 60 s, κ(sf60_s3, mis) − κ(sf60_s0, mis) > 0.
- **Tests:** per-opening differences, two-sided one-sample t-test; supported if Holm p < 0.05 and the estimate > 0.
  95% CIs from 10,000 bootstrap draws resampling openings within class and fresh seeds.
- **Secondary:** κ per configuration × prompt at 10/20/30 s (plus 40/50/60 s for Self Forcing); Spearman correlation
  across the 8 configurations between κ(mis, 30 s) and study 06's brightness persistence α_L (31 s), predicted positive;
  H1 after tone normalisation (each frame's CIELAB mean and sd matched to the opening); s_mat − s_mis per configuration;
  κ per class; top-1 accuracy of the opening's class among the 7 labels.
- **Predictions:** H1–H3 hold; κ(mis) < κ(mat) for every configuration; κ(mis, 30 s) between 0 and 0.7.

## Frozen files
See the hash block in the Chinese original. `jobs/content_b.jsonl` (phase 2) is produced by `make_content_tasks.py b`
after the openings are selected and is fully determined by the frozen scripts and the selected openings.
