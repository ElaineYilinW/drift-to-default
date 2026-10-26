"""Build the Argon task list of study 18, MAGI-1 (docs/preregistrations/18_magi.md).

12 scenes of preregistration 08, generic prompt, seed 0, 30 s, openings L* -16 / +16, MAGI-1 4.5B-distill at 432x768:
  mgw  the released attention window (noise2clean_kvrange [5, 4, 3, 2], clean_chunk_kvrange 1)
  mgf  full history (noise2clean_kvrange [])
One task per scene x arm (both versions; arch_guard). Fresh text-to-video starts of the generic prompt: seeds 0..3 per
arm, two seeds per task. Every task is pinned to the A100 group (A100, A800).          -> jobs/magi.jsonl
usage: python make_magi_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/magi"
SCENES = ["rx02", "rx06", "rx08", "rx10", "gx02", "gx06", "gx08", "gx10", "rx01", "rx03", "gx01", "gx03"]
PRE = f"source {H}/jobs/env.sh && cd $H/MAGI-1 && export PYTHONPATH=$H/shims_magi:$H/MAGI-1"
GEN = f"$H/envs/magi/bin/python $H/magi_gen.py --prompts $H/prompts/relax.jsonl --seconds 30 --outdir {D}"
ARMS = {"mgw": "window", "mgf": "full"}
T = []


def task(tid, cmds, done, est):
    return {"id": tid, "cmd": PRE + " && " + " && ".join(cmds), "done": done, "est_min": est, "mem_gb": 80, "gpus": 1,
            "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16, "gpu_only": "a100,a800"}


for s in SCENES:
    for tag, kv in ARMS.items():
        cmds = [f"$H/envs/sf/bin/python $H/arch_guard.py {D} {s}_{tag}__",
                f"{GEN} --prompt_id {s} --tag {tag} --kv {kv} --seeds 0 --versions m16:-16:0,p16:16:0"]
        T.append(task(f"magi_{s}_{tag}", cmds, f"{D}/{s}_{tag}__p16_s0.npz", 300))
for tag, kv in ARMS.items():
    for seeds in ("0,1", "2,3"):
        T.append(task(f"magi_fresh_{tag}_{seeds[0]}", [f"{GEN} --prompt_id rx02 --tag fresh_{tag} --kv {kv} --fresh "
                                                       f"--seeds {seeds}"], f"{D}/rx02_fresh_{tag}_s{seeds[-1]}.npz", 300))
with open("jobs/magi.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h")
