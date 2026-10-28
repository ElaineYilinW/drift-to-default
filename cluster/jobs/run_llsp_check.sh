#!/bin/bash
# consistency check of LongLive-c (jobs/patch_ll_sinkpos.py): on one GPU, 10 s of rx00 (L* -16, sink 3, seed 0) from
# the original LongLive tree and from LongLive-c without the switch, with off:0 (no rotation) and with dist:30 (must
# differ); compares the per-frame L* means.
source /Users/ywang906/vdrift/jobs/env.sh
D=/nfsscratch/Users/ywang906/vdrift/corr/llsp_check
mkdir -p $D
A="--prompts $H/prompts/relax.jsonl --prompt_id rx00 --sink 3 --seeds 0 --latent_frames 42 --versions m16:-16:0 --outdir $D --no_video"
(cd $H/LongLive/LongLive1.0 && PYTHONPATH=$H/LongLive/LongLive1.0 $PY $H/ll_hold_gen_long.py $A --tag orig) || exit 1
cd $H/LongLive-c || exit 1
export PYTHONPATH=$H/LongLive-c
$PY $H/ll_hold_gen_sinkpos.py $A --tag c || exit 1
$PY $H/ll_hold_gen_sinkpos.py $A --tag off0 --sink_mode off:0 || exit 1
$PY $H/ll_hold_gen_sinkpos.py $A --tag d30 --sink_mode dist:30 || exit 1
$PY - <<PY
import numpy as np
z = {t: np.load(f"$D/rx00_{t}__m16_s0.npz")["lab_mean"] for t in ("orig", "c", "off0", "d30")}
for t in ("c", "off0", "d30"):
    print(t, "max |diff| vs orig:", float(np.abs(z[t] - z["orig"]).max()), "identical:", bool((z[t] == z["orig"]).all()))
PY
