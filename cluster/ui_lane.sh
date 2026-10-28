#!/usr/bin/env bash
# One non-preemptible UI-GPU job with several GPUs: one lane per GPU, each lane runs its share of a task list
# (lines "id<TAB>done<TAB>cmd", argon.py tasks flattened), skipping tasks whose done file exists. Lane i waits
# i * STAGGER seconds before its first task so that several large models do not load at the same moment.
#   qsub -q UI-GPU -pe smp 64 -l ngpus=4,... -v NGPUS=4,STAGGER=240 jobs/ui_lane.sh CMDS.tsv LOGDIR
export PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2
CMDS="$1"; LOG="$2"; mkdir -p "$LOG"
N="${NGPUS:-4}"; ST="${STAGGER:-240}"
echo "host $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} $(date -u +%FT%TZ)" >> "$LOG/job.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv >> "$LOG/job.txt" 2>&1
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N - 1)))}"
for ((i = 0; i < N; i++)); do
  (
    sleep $((i * ST))
    export CUDA_VISIBLE_DEVICES="${G[$i]}"
    awk -F'\t' -v i=$i -v n=$N 'NR % n == i' "$CMDS" | while IFS=$'\t' read -r id done cmd; do
      [ -e "$done" ] && continue
      echo "start $id $(date -u +%FT%TZ)"
      bash -c "$cmd" > "$LOG/$id.log" 2>&1; rc=$?
      echo "end $id rc=$rc $(date -u +%FT%TZ)"
    done
  ) >> "$LOG/lane$i.txt" 2>&1 &
done
wait
echo "UI_LANE_DONE $(date -u +%FT%TZ)" >> "$LOG/job.txt"
