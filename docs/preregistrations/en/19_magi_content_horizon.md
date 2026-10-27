> English translation of the frozen Chinese preregistration [`../19_magi_content_horizon.md`](../19_magi_content_horizon.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 19: scene content on MAGI-1, and the released window at 60 s (2026-10-27, frozen before generation)

## Background
- Preregistration 18 showed, on the non-Wan MAGI-1:
  - the released attention window (mgw, at most the 4 previous chunks) forgets the opening's brightness: α falls from
    0.87 at 5 s to 0.34 at 30 s and has not reached a plateau by 30 s;
  - full history (mgf) stays near 0.65 from 10 to 30 s.
- Two questions remain open:
  - **Content:** relaxation of scene content (preregistration 09) has been measured on Wan-based models and on
    FramePack (preregistration 17), not on MAGI-1.
    - Does content relax on MAGI-1?
    - Does full history also keep more content?
    - Does the prompt also set the content default?
  - **Horizon:** does the released window keep forgetting after 30 s? The paper states that models relying on a
    KV cache alone forget the opening completely.
- **Pilot** (submitted before freezing; rx00, not a study scene, unshifted, 60 s, A40): one mgw and one mgf video, only
  to check that 60 s runs, and its memory and runtime.
  - They enter no analysis;
  - the mgf 60-s result only informs whether a separate later study is run.

## Design
- **Generation:** `magi_gen.py` (as in preregistration 18; MAGI-1 4.5B distilled, 432×768, seed 0).
- **GPU:** every task is pinned to A40 (`gpu_only`); each task runs arch_guard at its start to check that all outputs
  of the same opening or scene come from one architecture.
- **A content (30 s):**
  - Openings: the 24 openings of preregistration 09 (6 classes × 4).
  - Three arms per opening:
    - mis_mgw: generic city prompt, released window;
    - mis_mgf: generic city prompt, full history;
    - mat_mgw: the prompt matched to the opening's class (the mat prompts of preregistration 09), released window.
  - Fresh starts: text-to-video of the generic city prompt, seeds 0–7, for mgw and for mgf.
  - One task per opening × arm; one task per fresh video.
  - 72 + 16 = 88 videos.
- **B horizon (60 s):**
  - the 12 scenes of preregistration 08, openings L\* −16 / +16, generic prompt, released window (mgw60);
  - one task per scene × version, 24 videos.
- **CLIP:** embeddings of every part-A video with the frozen `clip_probe.py` (the 7 labels of preregistration 09).
- Task lists from `make_magi19_tasks.py` (`jobs/magi19_a.jsonl`, `jobs/magi19_c.jsonl`).

## Measures
- **Content** (as preregistration 09, 24 fps):
  - in videos with an opening, frame f is (f − 9)/24 s after the opening; in fresh videos, frame f is at f/24 s.
  - Frame class score z = CLIP logit scale × (cos(frame, opening's class label) − cos(frame, city label)).
  - s[t0, t1): mean z in the window; s_open: mean z over frames 1–8 of the same video.
  - μ[t0, t1): mean of that class's window score over the fresh starts of the same window setting (mgw or mgf).
  - κ = (s − μ) / (s_open − μ); undefined when s_open − μ < 2.
- **Brightness** (as preregistration 18):
  - α(T) = (end_p16 − end_m16) / (open_p16 − open_m16);
  - open is the mean L\* of frames 1–8, end the mean L\* over [T − 10, T) s.
- **Collapse rule (B):** a scene is excluded if either version has a 1-s mean L\* below 3 or above 97 in [0, 60) s.

## Hypotheses (`analyze_magi19.py`, frozen; Holm over 4 tests; part-A window W = [25, 30) s)
- **A1 content relaxes under the released window:** s(mis_mgw, W) − s_open < 0.
- **A2 full history keeps more content:** κ(mis_mgf, W) − κ(mis_mgw, W) > 0.
- **A3 the prompt sets the content default:** s(mat_mgw, W) − s(mis_mgw, W) > 0.
- **B1 the released window keeps forgetting:** α(mgw60, 60) − α(mgw60, 30) < 0.
- **Tests:**
  - two-sided one-sample t-tests over openings (A) or scenes (B);
  - supported if Holm p < 0.05 and the estimate is on the predicted side;
  - 95% CIs by bootstrap: A resamples openings within class and the fresh starts; B resamples scenes; 10,000 draws each.

## Secondary analyses
- κ curves of mis_mgw and mis_mgf (windows [T−5, T), T = 5..30);
- κ(mis_mgf, W) and κ(mis_mgw, W) against 0;
- A1–A3 on tone-normalised embeddings;
- κ per class;
- top-1 accuracy of the opening's class among the 7 labels in W, per arm;
- α(mgw60, T) at T = 10, 20, 30, 40, 50, 60, and α(mgw60, 60) against 0;
- freezing (as preregistration 18: mean absolute frame-to-frame change of mean L\* over [20, 30) s below 0.01; for B
  also [50, 60) s):
  - frozen videos per arm;
  - A1–A3 and B1 again without them;
- attractor offset of the fresh starts: mean L\* over [20, 30) s minus over [0, 2) s.

## Predictions
- **A1 supported.**
- **A2 supported:** as for brightness, full history keeps more.
- **A3 supported.**
- **B1 supported:** α(mgw60, 60) about 0.1, possibly not detectably different from 0.

## Frozen files (SHA-256)
See the Chinese file.
