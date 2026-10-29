#!/bin/bash
# Self-Forcing-d: a symlink tree of Self-Forcing whose causal_model.py is a real copy of Self-Forcing-c's (patched by
# patch_sf.py, patch_sf_sinkpos.py and patch_sf_sinkband.py), then patched by patch_sf_bandhold.py. Leaves Self-Forcing,
# -b and -c untouched.
source /Users/ywang906/vdrift/jobs/env.sh
cd $H || exit 1
if [ ! -d Self-Forcing-d ]; then
  mkdir Self-Forcing-d && cp -as $H/Self-Forcing/. Self-Forcing-d/ || exit 1
fi
f=Self-Forcing-d/wan/modules/causal_model.py
if [ -L $f ]; then rm $f && cp Self-Forcing-c/wan/modules/causal_model.py $f && python3 jobs/patch_sf_bandhold.py $f; fi
find Self-Forcing-d -name __pycache__ -type d -prune -exec rm -rf {} +
grep -c "patched:" $f
