#!/bin/bash
# LongLive 1.0 on Argon: reuse the Self Forcing env (+ peft), its Wan2.1 weights, download LongLive base + LoRA.
set -ex
H=/Users/ywang906/vdrift
P=$H/envs/sf/bin/python
printf "torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\ntransformers==4.49.0\nnumpy==1.24.4\n" > /tmp/ll_constraints.txt
$P -m pip install --no-cache-dir --prefer-binary -c /tmp/ll_constraints.txt peft
cd $H/LongLive/LongLive1.0
[ -e wan_models ] || ln -s $H/Self-Forcing/wan_models wan_models
[ -f longlive_models/models/lora.pt ] || $H/envs/sf/bin/hf download Efficient-Large-Model/LongLive-1.3B --local-dir longlive_models
ls -la longlive_models longlive_models/models
$P $H/jobs/patch_ll.py
export LD_LIBRARY_PATH=$H/envs/sf/lib
$P -c "import peft, torch; print('OK', peft.__version__, torch.__version__)"
touch $H/env_ll.done
