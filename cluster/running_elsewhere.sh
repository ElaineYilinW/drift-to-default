#!/usr/bin/env bash
# running_elsewhere.sh RUN ID: exit 0 if task ID of the argon.py run directory RUN is running right now as an SGE job
# (its last attempt started and that job/task is in state r), else exit 1. Lets a UI lane skip tasks that all.q is
# already running, so the two do not write the same outputs at once.
l=$(tail -n 1 "$1/logs/$2.attempts" 2>/dev/null)
case "$l" in start*) ;; *) exit 1 ;; esac
j=$(sed -n 's/.* job=\([0-9]*\) task=\([0-9]*\).*/\1 \2/p' <<< "$l")
[ -n "$j" ] || exit 1
qstat -u "$USER" | awk -v j="${j% *}" -v t="${j#* }" '$1 == j && $5 ~ /r/ && $NF == t { f = 1 } END { exit !f }'
