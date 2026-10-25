"""Build the Argon task lists of study 17, FramePack content routes (docs/preregistrations/17_framepack_content.md).

phase a: select the 8 city images and write prompts/fpc_runs.jsonl (fpc_city_openings.py)        -> jobs/fpc_a.jsonl
phase b: FramePack-F1, generic city prompt, 20 s, seed 0. For each of the 24 openings of preregistration 09 the arms
         both / ctx / emb run in one task (arch_guard); the 8 city starts (arm alt) run 4 per task. Every task is
         pinned to the Ada group (L40S, RTX 6000 Ada).                                              -> jobs/fpc_b.jsonl
phase c: CLIP embeddings of every video (clip_probe.py)                                             -> jobs/fpc_c.jsonl
usage: python make_fpc_tasks.py a|b|c
"""
import json
import sys

H = "/Users/ywang906/vdrift"
R = "/nfsscratch/Users/ywang906/vdrift/corr"
D = f"{R}/fpc"
PRE = f"source {H}/jobs/env.sh && cd $H"
FP = (f"$H/envs/sky/bin/python $H/fp_gen_c.py --prompts $H/prompts/fpc_runs.jsonl --mode vanilla --seconds 20 --seeds 0 "
      f"--outdir {D}")
ADA = "l40s,rtx6000ada"


def task(tid, cmds, done, est, **kw):
    t = {"id": tid, "cmd": PRE + " && " + " && ".join(cmds), "done": done, "est_min": est, "mem_gb": 60, "gpus": 1,
         "gpu_mem_gb": 45, "gpu_bf16": True, "cores": 24}
    t.update(kw)
    return t


phase = sys.argv[1]
T = []
if phase == "a":
    T.append(task("fpc_select", [f"$PY $H/fpc_city_openings.py {R}/content $H/content_starts $H/prompts/content_runs.jsonl "
                                 f"$H/prompts/fpc_runs.jsonl"], "prompts/fpc_runs.jsonl", 10, mem_gb=30, gpu_mem_gb=16,
                  cores=8))
elif phase == "b":
    rows = [json.loads(l) for l in open("prompts/fpc_runs.jsonl")]
    ops = [r["id"] for r in rows if r["set"] == "fpc"]
    city = [r["id"] for r in rows if r["set"] == "fpc_city"]
    assert len(ops) == 24 and len(city) == 8
    for o in ops:
        cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{o}_fpc_{c}_" for c in ("both", "ctx", "emb"))]
        cmds += [f"{FP} --prompt_id {o} --tag fpc_{c} --channel {c}" for c in ("both", "ctx", "emb")]
        T.append(task(f"fpc_{o}", cmds, f"{D}/{o}_fpc_emb_s0.npz", 150, gpu_only=ADA))
    for i in range(0, 8, 4):
        cmds = [f"{FP} --prompt_id {c} --tag fpc_alt --channel alt" for c in city[i:i + 4]]
        T.append(task(f"fpc_city{i // 4}", cmds, f"{D}/{city[i + 3]}_fpc_alt_s0.npz", 200, gpu_only=ADA))
elif phase == "c":
    T.append(task("fpc_clip", [f"$PY $H/clip_probe.py '{D}/*.mp4' {R}/content/clip_text.npz {D}/clip.done"],
                  f"{D}/clip.done", 60, mem_gb=30, gpu_mem_gb=16, cores=8))
with open(f"jobs/fpc_{phase}.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(phase, len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h")
