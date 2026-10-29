#!/bin/bash
# exploratory: which temporal RoPE frequencies make the peaks of the attention-to-sink probe (patch_sf_sinkband.py)?
# One unshifted opening ($1), original sink 3, chunks 60 and 132, distances 3..300, 22 frequency bands, reference
# distance 20 (the re-rotated position). Runs against Self-Forcing-c (set up by jobs/setup_sf_c.sh).
source /Users/ywang906/vdrift/jobs/env.sh
export PYTHONPATH=$H/Self-Forcing-c
cd $H/Self-Forcing-c || exit 1
grep -q "patched: band probe" wan/modules/causal_model.py || exit 1
D=/nfsscratch/Users/ywang906/vdrift/corr
mkdir -p $D/sinkband
id=$1
[ -f $D/sinkband/$id.npz ] && exit 0
G=$(python3 -c "print(list(range(3, 301)))")
B=$(python3 -c "
import json
A = list(range(22))
groups = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12)), list(range(12, 22))]
bands = [A, []] + groups + [[j for j in A if j not in g] for g in groups] + [[j for j in A if j != k] for k in range(12)]
print(json.dumps(bands))")
$PY $H/hold_gen_sinkband.py --prompts $H/prompts/relax.jsonl --prompt_id $id --tag band_s3 --sink 3 --seeds 0 \
  --latent_frames 135 --versions z0:0:0 --outdir $D/sinkband --no_video \
  --probe "{\"out\": \"$D/sinkband/$id.npz\", \"chunks\": [60, 132], \"grid\": $G, \"nq\": 128, \"ref\": 20, \"bands\": $B}" || exit 1
ls -la $D/sinkband
