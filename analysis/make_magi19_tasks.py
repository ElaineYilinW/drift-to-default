"""Build the Argon task lists of study 19, MAGI-1 content and horizon (docs/preregistrations/19_magi_content_horizon.md).

A  content, 30 s: the 24 openings of preregistration 09 (6 classes x 4), MAGI-1 4.5B-distill at 432x768, seed 0, three
   arms per opening: mis_mgw (generic city prompt, released window), mis_mgf (generic city prompt, full history),
   mat_mgw (the opening's matched class prompt, released window); one task per opening x arm. Fresh text-to-video
   starts of the generic city prompt, seeds 0..7, per window setting (mgw, mgf), one task per video.
                                                                                     -> jobs/magi19_a.jsonl
B  horizon, 60 s: the 12 scenes of preregistration 08, L* -16 / +16, generic prompt, seed 0, released window (mgw60);
   one task per scene x version.                                                    -> jobs/magi19_a.jsonl
C  CLIP embeddings of the part-A videos (clip_probe.py, after A has finished)       -> jobs/magi19_c.jsonl
Every generation task is pinned to A40 (gpu_only) and runs arch_guard over all outputs of its opening or scene.
usage: python make_magi19_tasks.py a|c
"""
import json
import sys

H = "/Users/ywang906/vdrift"
R = "/nfsscratch/Users/ywang906/vdrift/corr"
DA, DB = f"{R}/magi19c", f"{R}/magi19h"
PRE = f"source {H}/jobs/env.sh && cd $H/MAGI-1 && export PYTHONPATH=$H/shims_magi:$H/MAGI-1"
GEN = f"$H/envs/magi/bin/python $H/magi_gen.py"
GUARD = f"$H/envs/sf/bin/python $H/arch_guard.py"
KV = {"mgw": "window", "mgf": "full"}
ARMS_A = {"mis_mgw": ("mis", "mgw"), "mis_mgf": ("mis", "mgf"), "mat_mgw": ("mat", "mgw")}
SCENES = ["rx02", "rx06", "rx08", "rx10", "gx02", "gx06", "gx08", "gx10", "rx01", "rx03", "gx01", "gx03"]
VERS = {"m16": "m16:-16:0", "p16": "p16:16:0"}


def task(tid, cmds, done, est):
    return {"id": tid, "cmd": PRE + " && " + " && ".join(cmds), "done": done, "est_min": est, "mem_gb": 80, "gpus": 1,
            "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16, "gpu_only": "a40"}


phase = sys.argv[1]
T = []
if phase == "a":
    ops = sorted({r["id"].rsplit("_", 1)[0] for r in map(json.loads, open("prompts/content_runs.jsonl"))
                  if r["set"] == "content"})
    assert len(ops) == 24, ops
    for o in ops:
        guard = f"{GUARD} {DA} " + " ".join(f"{o}_{p}_{k}_" for p, k in ARMS_A.values())
        for arm, (p, k) in ARMS_A.items():
            gen = (f"{GEN} --prompts $H/prompts/content_runs.jsonl --prompt_id {o}_{p} --tag {k} --kv {KV[k]} "
                   f"--seconds 30 --seeds 0 --outdir {DA}")
            T.append(task(f"m19a_{o}_{arm}", [guard, gen], f"{DA}/{o}_{p}_{k}_s0.npz", 150 if k == "mgf" else 100))
    for k in KV:
        for s in range(8):
            gen = (f"{GEN} --prompts $H/prompts/content.jsonl --prompt_id fresh --tag {k} --kv {KV[k]} --fresh "
                   f"--seconds 30 --seeds {s} --outdir {DA}")
            T.append(task(f"m19f_{k}_{s}", [gen], f"{DA}/fresh_{k}_s{s}.npz", 150 if k == "mgf" else 100))
    for sc in SCENES:
        for v, spec in VERS.items():
            gen = (f"{GEN} --prompts $H/prompts/relax.jsonl --prompt_id {sc} --tag mgw60 --kv window --seconds 60 "
                   f"--seeds 0 --versions {spec} --outdir {DB}")
            T.append(task(f"m19b_{sc}_{v}", [f"{GUARD} {DB} {sc}_mgw60__", gen], f"{DB}/{sc}_mgw60__{v}_s0.npz", 200))
elif phase == "c":
    T.append(dict(task("m19c_clip", [f"cd $H && $PY $H/clip_probe.py '{DA}/*.mp4' {R}/content/clip_text.npz "
                                     f"{DA}/clip.done"], f"{DA}/clip.done", 120),
                  cmd=f"source {H}/jobs/env.sh && cd $H && $PY $H/clip_probe.py '{DA}/*.mp4' {R}/content/clip_text.npz "
                      f"{DA}/clip.done", gpu_only=None, gpu_mem_gb=20, mem_gb=40, cores=8))
with open(f"jobs/magi19_{phase}.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps({k: v for k, v in t.items() if v is not None}) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
