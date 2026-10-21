#!/bin/bash
# Copy all vdrift outputs from /nfsscratch (40-day purge) to home, then verify file counts and sizes match.
set -e
S=/nfsscratch/Users/ywang906/vdrift
D=/Users/ywang906/vdrift/data
mkdir -p $D
rsync -a $S/feas $S/pilot $S/main $S/smoke $S/corr $D/
for d in feas pilot main smoke corr; do
  a=$(find $S/$d -type f | wc -l); b=$(find $D/$d -type f | wc -l)
  sa=$(du -sb $S/$d | cut -f1); sb=$(du -sb $D/$d | cut -f1)
  echo "$d files $a -> $b bytes $sa -> $sb"
  [ "$a" = "$b" ] && [ "$sa" = "$sb" ] || { echo "MISMATCH in $d"; exit 1; }
done
date > $D/backup.done
