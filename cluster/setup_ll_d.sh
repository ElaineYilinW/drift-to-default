#!/bin/bash
# LongLive-d: a symlink tree of LongLive/LongLive1.0 whose causal_model.py is a real copy of LongLive-c's (patched by
# patch_ll.py, patch_ll_probe.py and patch_ll_sinkpos.py), then patched by patch_ll_bandhold.py. Leaves LongLive,
# LongLive-b and LongLive-c untouched.
source /Users/ywang906/vdrift/jobs/env.sh
cd $H || exit 1
if [ ! -d LongLive-d ]; then
  mkdir LongLive-d && cp -as $H/LongLive/LongLive1.0/. LongLive-d/ || exit 1
fi
f=LongLive-d/wan/modules/causal_model.py
if [ -L $f ]; then rm $f && cp LongLive-c/wan/modules/causal_model.py $f && python3 jobs/patch_ll_bandhold.py $f; fi
find LongLive-d -name __pycache__ -type d -prune -exec rm -rf {} +
grep -c "patched:" $f
