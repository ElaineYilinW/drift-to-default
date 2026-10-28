#!/bin/bash
# exploratory: attention-to-sink probe over sink distances 3..723 (the 180-s horizon) on four unshifted openings with the
# original sink 3, six chunks each; for the burst times predicted at 60-180 s.
source /Users/ywang906/vdrift/jobs/env.sh
export PYTHONPATH=$H/Self-Forcing-b
cd $H/Self-Forcing-b || exit 1
D=/nfsscratch/Users/ywang906/vdrift/corr
mkdir -p $D/sinkprobe_long
G=$(python3 -c "print(list(range(3, 724)))")
C="[24, 60, 96, 132, 168, 204]"
for id in rx02 rx03 gx02 gx03; do
  [ -f $D/sinkprobe_long/$id.npz ] && continue
  $PY $H/hold_gen_sinkpos.py --prompts $H/prompts/relax.jsonl --prompt_id $id --tag probe_s3 --sink 3 --seeds 0 \
    --latent_frames 243 --versions z0:0:0 --outdir $D/sinkprobe_long --no_video \
    --probe "{\"out\": \"$D/sinkprobe_long/$id.npz\", \"chunks\": $C, \"grid\": $G, \"nq\": 128}" || exit 1
done
ls -la $D/sinkprobe_long
