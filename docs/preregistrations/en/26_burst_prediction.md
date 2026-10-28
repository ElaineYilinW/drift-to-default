> English translation of the frozen Chinese preregistration [`../26_burst_prediction.md`](../26_burst_prediction.md). The Chinese file is the authoritative version; its hashes and wording were fixed before data collection.

# Preregistration 26: does the attention probe predict when the bursts come between 60 and 180 s? (Self Forcing, LongLive, data of preregistration 24) (2026-10-28, frozen before preregistration 24 produced any output)

## Background
- With the sink at its original rotary positions, persistence jumps near 17, 33 and 51 s ("bursts"), and the
  opening's scene content returns at these times (preregistrations 10, 16, 21).
- First attention probe (lab note 30; distances 3–300; openings rx00, rx01, gx00, gx01): the share of attention the
  sink receives peaks at distances 71, 134, 203 and 266. At the original positions, the chunks' natural distance
  reaches the first three peaks 16.5, 32.25 and 49.5 s after the opening, the observed burst times.
  - These three times were matched after the bursts were known (in sample).
  - Preregistration 25 tests the causal direction by moving the sink (running).
- This study tests an out-of-sample prediction: can the probe predict burst times between 60 and 180 s, which have
  never been observed?
- **Second probe** (exploratory, `jobs/run_sinkprobe_long.sh`; not part of any preregistration):
  - setup: 4 new unshifted openings (rx02, rx03, gx02, gx03), sink 3 at its original positions; 6 chunks per opening
    (starts 24, 60, 96, 132, 168, 204), 128 random queries per chunk, 30 layers, 5 attention calls; distances 3–723
    (180 s);
  - the peak rule was written and committed before the output was read (`analyze_sinkprobe_long.py`, commit 0f6123f):
    the local maxima of the mean share curve with prominence ≥ 0.04 (scipy.signal.find_peaks). On the first probe this
    rule returns exactly 71, 134, 203, 266;
  - result (`sinkprobe_long_out.json`):
    - correlation with the first probe over the common distances r = 0.999; all 4 major peaks recur;
    - 17 major peaks: 71, 90, 134, 203, 266, 309, 335, 404, 441, 467, 510, 536, 574, 599, 642, 669, 711;
    - the peak at distance 90 (share 0.102) had prominence 0.0375 in the first probe, below the threshold; in the
      second probe it has 0.043 and counts under the rule;
    - the peak shares after 60 s are 0.098–0.152, the same order as the peaks in the first 60 s (0.102–0.165; median
      over 3 ≤ D ≤ 723: 0.074);
    - the major peaks sit at nearly the same distances in every opening and chunk (some openings or chunks add minor
      peaks such as 177, 222, 378).
- **State of the data:**
  - at freezing, all 125 tasks of preregistration 24 were queued and none had produced output;
  - the 120-s subset of preregistration 06 (6 scenes; sf_s3, ll_s3, sf_s0) has only been analysed in 10-s windows
    (lab note 13); its 1-s curves after 60 s have never been looked at.

## Design
- **Data:** the videos of preregistration 24 (24 scenes, L\* −16 / +16, seed 0, 180 s), generated with the frozen
  files of preregistration 24; this study generates nothing new.
  - sf180_s3: Self Forcing, sink 3, original positions (primary);
  - ll180_s3: LongLive, sink 3, original positions (primary);
  - sf180_s3rr: Self Forcing, sink 3, re-rotated (secondary; control: the sink's distance is fixed, so it should not
    burst).
- The probe was measured on Self Forcing only. LongLive is a fine-tuned Self Forcing whose bursts in the first 60 s
  come at the same times (the atlas of preregistration 10); here we test whether the prediction transfers.

## Measures
- α[t0, t1) = (end_p16 − end_m16) / (open_p16 − open_m16):
  - open is the mean L\* of frames 1–8;
  - end is the mean L\* over seconds [t0, t1) after the opening;
  - undefined if open_p16 − open_m16 < 4 L\*.
- No collapse exclusion (as in preregistration 24).
- **Predicted burst onsets:** τ = (P − 5)/4 s for the 17 major peaks P above; kept if 62.5 ≤ τ and τ + 2.5 ≤ 180
  (the new predictions after 60 s; 13 onsets):
  - 65.25, 76.0, 82.5, 99.75, 109.0, 115.5, 126.25, 132.75, 142.25, 148.5, 159.25, 166.0, 176.5 s.
- **Control onsets:** the midpoints between the onsets of consecutive major peaks, with the same range rule (12
  onsets):
  - 70.625, 79.25, 91.125, 104.375, 112.25, 120.875, 129.5, 137.5, 145.375, 153.875, 162.625, 171.25 s.
- **Burst contrast** (one value per scene): over a set of onsets, mean_k [α([τ_k + 0.5, τ_k + 2.5)) −
  α([τ_k − 2.5, τ_k − 0.5))], the windows of preregistration 25.

## Hypotheses (`analyze_s26.py`, frozen; Holm correction over 3 tests)
- **B1 Self Forcing bursts at the predicted times:** burst contrast(sf180_s3, predicted onsets) > 0.
- **B2 LongLive bursts at the same times:** burst contrast(ll180_s3, predicted onsets) > 0.
- **B3 the timing is specific (Self Forcing):** burst contrast(sf180_s3, predicted) − burst contrast(sf180_s3,
  control) > 0.
- **Tests:**
  - two-sided one-sample t-tests over scenes;
  - supported if Holm p < 0.05 and the estimate is > 0;
  - 95% CIs by scene bootstrap (10,000 draws).

## Secondary analyses
- the burst contrast of sf180_s3rr at the predicted onsets (expected about 0);
- B1–B3 within the first 60 s (in sample): onsets 16.5, 21.25, 32.25, 49.5 s; controls 18.875, 26.75, 40.875,
  57.375 s;
- times of 1-s jumps above 0.15 per arm; how many of those in 62–180 s fall within 3 s after a predicted onset,
  against the share of seconds these intervals cover (the chance hit rate);
- the burst contrast at each predicted onset separately;
- probe check: over 62–179 s (and 6–59 s), the correlation between the first differences of an arm's mean 1-s α curve
  at t and the first differences of the probe's predicted sink share at the natural distances at t − lag, lag = 0, 1,
  2, 3 s (share per second defined as in `analyze_s25.py`; since the burst window starts at τ + 0.5 s, the expected lag
  is 1–2 s);
  - note: the same secondary analysis of preregistration 25 uses lag 0 only. Under the window definition above, the
    lag-0 correlation is near 0 or negative even when the bursts come exactly as predicted (checked on simulated
    data); this will be stated when preregistration 25 is interpreted;
- **Replication:** the same script (TMAX = 119) on the 120-s subset of preregistration 06 (6 scenes; the sf_s3 and
  ll_s3 files linked under the names sf180_s3 and ll180_s3): 6 predicted onsets (65.25–115.5 s) and 5 controls.

## Predictions
- **B1 supported.** The peak shares after 60 s are of the same order as before 60 s, so the bursts should go on.
  - Size: in the first 60 s the burst contrast is +0.33 (sink 3 of preregistration 10, three onsets). After 60 s the
    predicted onsets are denser and some peaks are weaker; expected +0.1 to +0.25.
- **B2 supported**, less certainly than B1 (the probe was not measured on LongLive).
- **B3 supported.**
- **Rules for interpretation:**
  - if B1 and B3 are supported, the probe predicted 13 never-observed burst times out of sample: direct evidence
    that the bursts are set by the sink's rotary distance (complementing the intervention of preregistration 25);
  - if B1 is supported and B3 is not, α fluctuates after 60 s but not specifically at the predicted times; the probe
    accounts only for the first 60 s;
  - if B1 is not supported, the bursts after 60 s disappear or come at other times; the paper can say that the probe
    accounts for the bursts of the first 60 s, not that it predicts them;
  - if B2 is not supported while B1 is, LongLive's fine-tuning changed the query–key structure and the probe's peaks
    do not transfer directly.

## Frozen files (SHA-256)
```
c186f782fd53563248d73ae8ac39c7fc0c12caa50b4d6f93e47480ad9329650d  analyze_s26.py
0ae85419e61b029ff6bf07e57c208d8e7f901a59a8ceb3e6656abdeb9d9ff372  analyze_sinkprobe_long.py
b27dced6e9842a3a86c6cd843b4ba52da196786679e6ef35ea40b7fecb73b167  jobs/run_sinkprobe_long.sh
d177b087cb912589ec6925612efcb5b80c447c0037db8d2fa0481429b0888e98  jobs/sinkprobe_long.jsonl
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
64385a87e26b2110501e89abee7f07196cc093372da13b57f11f5df26cdde923  sinkprobe_long_out.json (exploratory probe result)
```
- The data are generated by the frozen files of preregistration 24: analyze_s24.py 0912cdde…, hold_gen_long_rr.py
  f7b6fe48…, ll_hold_gen_long.py 8723343d…, jobs/s24.jsonl 6a633329… (checked at freezing).
