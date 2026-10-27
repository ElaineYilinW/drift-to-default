"""Build the Argon task lists of study 22, Krea Realtime 14B (docs/preregistrations/22_krea.md).

a  A tone: the 24 scenes of preregistration 06, L* -16 / +16, generic prompt, 60 s, seed 0, arms kr60_none (released
   default) and kr60_first (keep_first_frame); one task per scene and arm (both versions).
   B content: the 24 openings of preregistration 09, generic city prompt, 60 s, seed 0, both arms in one task; fresh
   starts of the generic prompt (seeds 0..7) per arm.
   Every task pinned to A40 (gpu_only) so that all videos share an architecture; arch_guard checks each output group.
                                                                                          -> jobs/krea_a.jsonl
e  CLIP (primary) and SigLIP (secondary) embeddings of the part-B videos (after a)       -> jobs/krea_e.jsonl
usage: python make_krea_tasks.py a|e
"""
import json
import sys

H = "/Users/ywang906/vdrift"
R = "/nfsscratch/Users/ywang906/vdrift/corr"
DA, DB = f"{R}/krea_tone", f"{R}/krea_content"
PRE = f"source {H}/jobs/env.sh && cd $H"
GEN = "$H/envs/krea/bin/python $H/krea_gen.py --seconds 60 --seeds 0"
ARMS = {"kr60_none": "--anchor none", "kr60_first": "--anchor first"}
SCENES24 = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]


def task(tid, cmds, done, est):
    return {"id": tid, "cmd": PRE + " && " + " && ".join(cmds), "done": done, "est_min": est, "mem_gb": 110, "gpus": 1,
            "gpu_mem_gb": 45, "gpu_bf16": True, "cores": 32, "gpu_only": "a40"}


T = []
if sys.argv[1] == "a":
    for s in SCENES24:
        for c, x in ARMS.items():
            T.append(task(f"kra_{s}_{c}", [f"$PY $H/arch_guard.py {DA} {s}_{c}__",
                                           f"{GEN} --prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {c} {x} "
                                           f"--versions m16:-16:0,p16:16:0 --outdir {DA}"],
                          f"{DA}/{s}_{c}__p16_s0.npz", 45))
    ops = sorted({r["id"].rsplit("_", 1)[0] for r in map(json.loads, open("prompts/content_runs.jsonl"))
                  if r["set"] == "content"})
    assert len(ops) == 24, ops
    for o in ops:
        cmds = [f"$PY $H/arch_guard.py {DB} " + " ".join(f"{o}_mis_{c}_" for c in ARMS)]
        cmds += [f"{GEN} --prompts $H/prompts/content_runs.jsonl --prompt_id {o}_mis --tag {c} {x} --outdir {DB}"
                 for c, x in ARMS.items()]
        T.append(task(f"krb_{o}", cmds, f"{DB}/{o}_mis_kr60_first_s0.npz", 50))
    for c, x in ARMS.items():
        T.append(task(f"krf_{c}", [f"$H/envs/krea/bin/python $H/krea_gen.py --seconds 60 --seeds 0,1,2,3,4,5,6,7 "
                                   f"--prompts $H/prompts/content.jsonl --prompt_id fresh --tag {c} {x} --fresh "
                                   f"--outdir {DB}"], f"{DB}/fresh_{c}_s7.npz", 150))
elif sys.argv[1] == "e":
    T.append(task("kre_clip", [f"$PY $H/clip_probe.py '{DB}/*.mp4' {DB}/clip_text.npz {DB}/clip.done",
                               f"$PY $H/siglip_probe.py '{DB}/*.mp4' {DB}/siglip_text.npz {DB}/siglip.done"],
                  f"{DB}/siglip.done", 90))
    T[-1].update(mem_gb=30, cores=16)
    del T[-1]["gpu_only"]
with open(f"jobs/krea_{sys.argv[1]}.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
