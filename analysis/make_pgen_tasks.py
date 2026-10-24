"""Argon task lists of the prompt-generality study (docs/preregistration_pgen_cn.md).
phase a: opening candidates (Self Forcing DMD, 5-s fresh videos of each domain prompt, seeds 0..5) -> jobs/pgen_a.jsonl
phase b: 24 openings x L* -16 / 0 / +16; Self Forcing DMD sink 3 / no sink (60 s) in one task, LongLive sink 3 / no
         sink (31 s) in another; arch_guard at the start of each task                     -> jobs/pgen_b.jsonl
usage: python make_pgen_tasks.py a|b"""
import json
import sys

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/pgen"
G = "/nfsscratch/Users/ywang906/vdrift/corr/pgen_gen"
SF = f"source {H}/jobs/env.sh && cd $H/Self-Forcing"
LL = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/LongLive/LongLive1.0 && cd {H}/LongLive/LongLive1.0"
V = "m16:-16:0,z0:0:0,p16:16:0"


def task(tid, pre, cmds, done, est):
    return {"id": tid, "cmd": pre + " && " + " && ".join(cmds), "done": done, "est_min": est, "mem_gb": 30, "gpus": 1,
            "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16}


T = []
if sys.argv[1] == "a":
    for r in map(json.loads, open("prompts/pgen.jsonl")):
        T.append(task(f"pga_{r['domain']}", SF, [f"$PY $H/hold_gen.py --prompts $H/prompts/pgen.jsonl --prompt_id {r['id']} "
                                                f"--tag sfgen --fresh --seeds 0,1,2,3,4,5 --latent_frames 21 --outdir {G}"],
                      f"{G}/{r['id']}_sfgen_s5.npz", 10))
else:
    P = "--prompts $H/prompts/pgen_runs.jsonl"
    for r in map(json.loads, open("prompts/pgen_runs.jsonl")):
        o = r["id"]
        T.append(task(f"pgb_sf_{o}", SF, [f"$PY $H/arch_guard.py {D} {o}_sf60_s3__ {o}_sf60_s0__"] +
                      [f"$PY $H/hold_gen_long.py {P} --prompt_id {o} --tag {c} --sink {s} --seeds 0 --latent_frames 243 "
                       f"--versions {V} --outdir {D}" for c, s in (("sf60_s3", 3), ("sf60_s0", 0))],
                      f"{D}/{o}_sf60_s0__p16_s0.npz", 48))
        T.append(task(f"pgb_ll_{o}", LL, [f"$PY $H/arch_guard.py {D} {o}_ll_s3__ {o}_ll_s0__"] +
                      [f"$PY $H/ll_hold_gen.py {P} --prompt_id {o} --tag {c} --sink {s} --seeds 0 --latent_frames 126 "
                       f"--versions {V} --outdir {D}" for c, s in (("ll_s3", 3), ("ll_s0", 0))],
                      f"{D}/{o}_ll_s0__p16_s0.npz", 24))
with open(f"jobs/pgen_{sys.argv[1]}.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h")
