# Sourced by every job on the cluster. Set H to the project directory.
export H=${H:-$HOME/vdrift}
export PY=$H/envs/sf/bin/python
export LD_LIBRARY_PATH=$H/envs/sf/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MALLOC_ARENA_MAX=2 MKL_THREADING_LAYER=SEQUENTIAL OPENBLAS_CORETYPE=Haswell
export HF_HOME=$H/hf_cache
export PYTHONPATH=$H/Self-Forcing  # sf_test.py lives outside the repo
