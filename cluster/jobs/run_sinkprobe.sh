#!/bin/bash
# exploratory: (1) Self-Forcing-b without switches reproduces the atlas run bit for bit (rx00 sink 3, m16, A40);
# (2) attention-to-sink probe over sink distances 3..300 on four unshifted openings with the original sink 3.
source /Users/ywang906/vdrift/jobs/env.sh
export PYTHONPATH=$H/Self-Forcing-b
cd $H/Self-Forcing-b || exit 1
D=/nfsscratch/Users/ywang906/vdrift/corr
mkdir -p $D/sinkpos_check $D/sinkprobe
$PY $H/hold_gen_sinkpos.py --prompts $H/prompts/relax.jsonl --prompt_id rx00 --tag sf60_s3 --sink 3 --seeds 0 \
  --latent_frames 243 --versions m16:-16:0 --outdir $D/sinkpos_check --no_video || exit 1
$PY - <<'P'
import numpy as np
a = np.load("/nfsscratch/Users/ywang906/vdrift/corr/atlas/rx00_sf60_s3__m16_s0.npz")
b = np.load("/nfsscratch/Users/ywang906/vdrift/corr/sinkpos_check/rx00_sf60_s3__m16_s0.npz")
print("IDENTITY", all(np.array_equal(a[k], b[k]) for k in a.files), float(np.abs(a["lab_mean"] - b["lab_mean"]).max()))
P
G=$(python3 -c "print(list(range(3, 301)))")
C=$(python3 -c "print(list(range(12, 241, 12)))")
for id in rx00 rx01 gx00 gx01; do
  [ -f $D/sinkprobe/$id.npz ] && continue
  $PY $H/hold_gen_sinkpos.py --prompts $H/prompts/relax.jsonl --prompt_id $id --tag probe_s3 --sink 3 --seeds 0 \
    --latent_frames 243 --versions z0:0:0 --outdir $D/sinkprobe --no_video \
    --probe "{\"out\": \"$D/sinkprobe/$id.npz\", \"chunks\": $C, \"grid\": $G, \"nq\": 128}" || exit 1
done
ls -la $D/sinkprobe
