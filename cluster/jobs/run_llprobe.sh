#!/bin/bash
# exploratory: the attention-to-sink probe on LongLive (sink 3, original positions, window 12), as
# jobs/run_sinkprobe_long.sh for Self Forcing: distances 3..723, four unshifted openings, six chunks each.
source /Users/ywang906/vdrift/jobs/env.sh
export PYTHONPATH=$H/LongLive-b
cd $H/LongLive-b || exit 1
D=/nfsscratch/Users/ywang906/vdrift/corr
mkdir -p $D/llprobe
G=$(python3 -c "print(list(range(3, 724)))")
C="[24, 60, 96, 132, 168, 204]"
for id in rx02 rx03 gx02 gx03; do
  [ -f $D/llprobe/$id.npz ] && continue
  $PY $H/ll_hold_gen_probe.py --prompts $H/prompts/relax.jsonl --prompt_id $id --tag probe_s3 --sink 3 --seeds 0 \
    --latent_frames 243 --versions z0:0:0 --outdir $D/llprobe --no_video \
    --probe "{\"out\": \"$D/llprobe/$id.npz\", \"chunks\": $C, \"grid\": $G, \"nq\": 128}" || exit 1
done
ls -la $D/llprobe
