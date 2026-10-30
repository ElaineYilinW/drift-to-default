# Preregistration 26: can the attention probe predict the burst times between 60 and 180 s? (Self Forcing, LongLive, data of preregistration 24) (2026-10-28, frozen before preregistration 24 produced any output)

*English translation; the Chinese file `../26_burst_prediction.md` is the frozen, authoritative version.*

## Background
- With the sink at its original rotary positions, persistence jumps suddenly near 17, 33 and 51 s ("bursts"), and the opening's scene content also returns at these times (preregistrations 10, 16, 21).
- First attention probe (lab note 30, distances 3–300, openings rx00, rx01, gx00, gx01): the attention share the sink receives has sharp peaks at distances 71, 134, 203, 266; at the original positions, the chunks' natural distance reaches the first three peaks at 16.5, 32.25 and 49.5 s after the opening, exactly the observed burst times.
  - These three times were matched only after the bursts were known (in-sample).
  - Preregistration 25 tests the causal direction by moving the sink's position (running).
- This study tests an out-of-sample prediction: can the probe predict burst times between 60 and 180 s, which have never been observed?
- **Second probe** (exploratory, `jobs/run_sinkprobe_long.sh`; not part of any preregistration):
  - Setup: 4 new unshifted openings (rx02, rx03, gx02, gx03), sink 3, original positions; 6 chunks per opening (starts 24, 60, 96, 132, 168, 204), 128 random queries per chunk, 30 layers, 5 attention calls; distances 3–723 (180 s).
  - The peak rule was written and committed before the output was read (`analyze_sinkprobe_long.py`, commit 0f6123f): local maxima of the mean share curve with prominence ≥ 0.04 (scipy.signal.find_peaks). On the first probe this rule gives exactly 71, 134, 203, 266.
  - Result (`sinkprobe_long_out.json`):
    - correlation with the first probe over the common distances r = 0.999; all 4 main peaks are reproduced.
    - 17 main peaks in total: 71, 90, 134, 203, 266, 309, 335, 404, 441, 467, 510, 536, 574, 599, 642, 669, 711.
    - The peak at distance 90 (share 0.102) had prominence 0.0375 in the first probe, below the threshold; in the second probe it is 0.043 and is counted by the rule.
    - The peak shares after 60 s are 0.098–0.152, of the same order as the peaks in the first 60 s (0.102–0.165) (median over 3 ≤ D ≤ 723: 0.074).
    - The main peak positions are essentially the same in every opening and every chunk (individual openings or chunks have extra small peaks such as 177, 222, 378).
- **Data status:**
  - When this was frozen, all 125 tasks of preregistration 24 were queued, and none had produced any output.
  - The 120-s subset of preregistration 06 (6 scenes; sf_s3, ll_s3, sf_s0) had previously been analysed only in 10-s windows (lab note 13); its 1-s curves after 60 s had never been looked at.

## Design
- **Data:** the videos of preregistration 24 (24 scenes, L\* −16 / +16, seed 0, 180 s), generated from preregistration 24's frozen files; this study generates nothing of its own.
  - sf180_s3: Self Forcing, sink 3, original positions (primary);
  - ll180_s3: LongLive, sink 3, original positions (primary);
  - sf180_s3rr: Self Forcing, sink 3, re-rotated (secondary, control: the sink's distance is fixed, so there should be no bursts).
- The probe was measured only on Self Forcing. LongLive is a fine-tuned version of Self Forcing, and in the first 60 s its burst times are the same as Self Forcing's (the atlas of preregistration 10); this tests whether the prediction transfers.

## Metric
- α[t0, t1) = (end_p16 − end_m16) / (open_p16 − open_m16):
  - open is the mean L\* of frames 1–8;
  - end is the mean L\* over seconds [t0, t1) after the opening;
  - undefined if open_p16 − open_m16 < 4 L\*.
- No collapse exclusion (as in preregistration 24).
- **Predicted burst onsets:** τ = (P − 5)/4 s, with P the 17 main peaks above; only onsets with 62.5 ≤ τ and τ + 2.5 ≤ 180 are kept (new predictions after 60 s, 13 in total):
  - 65.25, 76.0, 82.5, 99.75, 109.0, 115.5, 126.25, 132.75, 142.25, 148.5, 159.25, 166.0, 176.5 s.
- **Control onsets:** the midpoints between neighbouring main-peak onsets, with the same range rule (12 in total):
  - 70.625, 79.25, 91.125, 104.375, 112.25, 120.875, 129.5, 137.5, 145.375, 153.875, 162.625, 171.25 s.
- **Burst contrast** (one value per scene): the mean over a set of onsets, mean_k [α([τ_k + 0.5, τ_k + 2.5)) − α([τ_k − 2.5, τ_k − 0.5))], with the same windows as preregistration 25.

## Hypotheses (`analyze_s26.py`, frozen; Holm over 3 tests)
- **B1, Self Forcing bursts at the predicted times:** burst contrast(sf180_s3, predicted onsets) > 0.
- **B2, LongLive bursts at the same times:** burst contrast(ll180_s3, predicted onsets) > 0.
- **B3, the times are specific (Self Forcing):** burst contrast(sf180_s3, predicted onsets) − burst contrast(sf180_s3, control onsets) > 0.
- **Tests:**
  - two-sided one-sample t-test over scenes;
  - supported if Holm p < 0.05 and the estimate > 0;
  - 95% CIs by scene bootstrap (10,000 draws).

## Secondary analyses
- The burst contrast of sf180_s3rr at the predicted onsets (expected to be about 0).
- B1–B3 within the first 60 s (in-sample): onsets 16.5, 21.25, 32.25, 49.5 s, controls 18.875, 26.75, 40.875, 57.375 s.
- Per arm, the times of jumps above 0.15 in the 1-s curve; how many of these jumps within 62–180 s fall within 3 s after some predicted onset, against the fraction of seconds that these intervals cover (the chance hit rate).
- The burst contrast at each predicted onset separately.
- Probe check: over 62–179 s (and 6–59 s), the correlation between the first difference at time t of each arm's mean 1-s α curve and the first difference at time t − lag of the sink share the probe predicts at the natural distance, lag = 0, 1, 2, 3 s (the per-second share is defined as in `analyze_s25.py`; the burst window starts at τ + 0.5 s, so a lag of 1–2 s is expected);
  - note: the analogous secondary analysis of preregistration 25 uses only lag 0; with the window definition above, the lag-0 correlation is close to 0 or negative even when the bursts appear exactly as predicted (checked on simulated data), and this will be stated when preregistration 25 is interpreted.
- **Replication:** the same script (TMAX = 119) is applied to the 120-s subset of preregistration 06 (6 scenes; the sf_s3 and ll_s3 files linked under the names sf180_s3 and ll180_s3): 6 predicted onsets in 65.25–115.5 s, 5 controls.

## Predictions
- **B1 holds.** The peak shares after 60 s are of the same order as in the first 60 s, so the bursts should continue.
  - Size: in the first 60 s the burst contrast was +0.33 (sink 3 of preregistration 10, three onsets); after 60 s the predicted onsets are denser and some peaks are weaker, so +0.1 to +0.25 is expected.
- **B2 holds**, but less certainly than B1 (the probe was not measured on LongLive).
- **B3 holds.**
- **Interpretation rules:**
  - If B1 and B3 hold: the probe has predicted, out of sample, 13 burst times that had never been observed. This is direct evidence that the bursts are set by the sink's rotary distance (complementing the intervention of preregistration 25).
  - If B1 holds and B3 does not: after 60 s α fluctuates in general, but not specifically at the predicted times; the probe explains only the first 60 s.
  - If B1 does not hold: after 60 s the bursts disappear or do not come at the predicted times; the paper can only say that the probe explains the bursts in the first 60 s, not that it predicts them.
  - If B2 does not hold while B1 does: LongLive's fine-tuning changed the query–key structure, and the probe's peaks do not transfer directly.

## Frozen files (SHA-256)
```
c186f782fd53563248d73ae8ac39c7fc0c12caa50b4d6f93e47480ad9329650d  analyze_s26.py
0ae85419e61b029ff6bf07e57c208d8e7f901a59a8ceb3e6656abdeb9d9ff372  analyze_sinkprobe_long.py
b27dced6e9842a3a86c6cd843b4ba52da196786679e6ef35ea40b7fecb73b167  jobs/run_sinkprobe_long.sh
d177b087cb912589ec6925612efcb5b80c447c0037db8d2fa0481429b0888e98  jobs/sinkprobe_long.jsonl
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
64385a87e26b2110501e89abee7f07196cc093372da13b57f11f5df26cdde923  sinkprobe_long_out.json (exploratory probe result)
```
- The data are generated from preregistration 24's frozen files: analyze_s24.py 0912cdde…, hold_gen_long_rr.py f7b6fe48…, ll_hold_gen_long.py 8723343d…, jobs/s24.jsonl 6a633329… (checked to match at freezing).
