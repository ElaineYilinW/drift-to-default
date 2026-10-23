"""Keep the paired arms of a task on one GPU architecture when tasks may land on different GPU types.

Run at the start of a task: for each output prefix (e.g. rx00_sf_s3__), if any existing output of that prefix was
generated on a different architecture group than the current GPU, delete all outputs of the task's prefixes so that the
whole task is regenerated here. Architecture groups: A40 | RTX A6000 | A100/A800 | L40S/RTX 6000 Ada.
usage: python arch_guard.py OUTDIR PREFIX [PREFIX ...]
"""
import glob
import json
import os
import subprocess
import sys

outdir, prefixes = sys.argv[1], sys.argv[2:]


def group(name):
    n = name.upper()
    if "A100" in n or "A800" in n:
        return "ampere-dc"
    if "L40S" in n or "6000 ADA" in n:
        return "ada"
    if "A40" in n:
        return "a40"
    if "A6000" in n:
        return "a6000"
    return n


cur = group(subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.strip().splitlines()[0])
found = set()
for pre in prefixes:
    for j in glob.glob(os.path.join(outdir, pre + "*.json")):
        try:
            found.add(group(json.load(open(j)).get("gpu", "")))
        except (json.JSONDecodeError, OSError):
            pass
if found - {cur}:
    n = 0
    for pre in prefixes:
        for f in glob.glob(os.path.join(outdir, pre + "*")):
            os.remove(f)
            n += 1
    print(f"arch_guard: current {cur}, existing {sorted(found)} -> removed {n} files, task regenerates", flush=True)
else:
    print(f"arch_guard: current {cur}, existing {sorted(found)} -> keep", flush=True)
