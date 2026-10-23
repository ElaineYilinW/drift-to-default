#!/bin/bash
# Rename fresh-start outputs fresh_{cfg}_s{seed}.* -> fresh_{cfg}__fresh_s{seed}.* (the names the relaxation analysis
# and the task done-files expect; generators without --versions omit the version suffix). usage: fix_fresh_names.sh DIR
cd "$1" || exit 1
for f in fresh_*_s[0-9].*; do
  [ -e "$f" ] || continue
  case "$f" in *__fresh_*) continue;; esac
  n=$(echo "$f" | sed -E 's/^(fresh_.+)_s([0-9])\./\1__fresh_s\2./')
  mv "$f" "$n"
done
