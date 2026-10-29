#!/bin/bash
# Self-Forcing-c: a symlink tree of Self-Forcing whose causal_model.py is a real copy of Self-Forcing-b's (patched by
# patch_sf.py and patch_sf_sinkpos.py), then patched by patch_sf_sinkband.py. Leaves Self-Forcing and -b untouched.
source /Users/ywang906/vdrift/jobs/env.sh
cd $H || exit 1
if [ ! -d Self-Forcing-c ]; then
  mkdir Self-Forcing-c && cp -as $H/Self-Forcing/. Self-Forcing-c/ || exit 1
fi
f=Self-Forcing-c/wan/modules/causal_model.py
if [ -L $f ]; then rm $f && cp Self-Forcing-b/wan/modules/causal_model.py $f; fi
find Self-Forcing-c -name __pycache__ -type d -prune -exec rm -rf {} +
python3 jobs/patch_sf_sinkband.py $f
grep -c "patched:" $f
