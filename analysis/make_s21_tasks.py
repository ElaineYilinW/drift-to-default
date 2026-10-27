"""Build the Argon task lists of study 21, new seeds for the Self Forcing content results
(docs/preregistrations/21_content_seeds.md).

a  the 24 openings of preregistration 09, generic city prompt (arm mis), Self Forcing DMD 60 s, sink 3 / sink 3
   re-rotated / no sink, seeds 1, 2, 3; one task per opening runs all arms and seeds (arch_guard over its outputs);
   fresh starts of the generic prompt, seeds 8..15, one task per configuration; any GPU with >= 40 GB and bf16
                                                                                          -> jobs/s21_a.jsonl
e  CLIP and SigLIP embeddings of all videos (after a has finished; text embeddings written beside them)
                                                                                          -> jobs/s21_e.jsonl
usage: python make_s21_tasks.py a|e
"""
import json
import sys

H = "/Users/ywang906/vdrift"
R = "/nfsscratch/Users/ywang906/vdrift/corr"
D = f"{R}/s21_content"
SF = f"source {H}/jobs/env.sh && cd $H/Self-Forcing"
PRE = f"source {H}/jobs/env.sh && cd $H"
GEN = "$PY $H/hold_gen_long_rr.py"
ARMS = {"sf60_s3": "--sink 3", "sf60_s3rr": "--sink 3 --rerotate", "sf60_s0": "--sink 0"}


def task(tid, pre, cmds, done, est):
    return {"id": tid, "cmd": pre + " && " + " && ".join(cmds), "done": done, "est_min": est, "mem_gb": 30, "gpus": 1,
            "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16}


T = []
if sys.argv[1] == "a":
    ops = sorted({r["id"].rsplit("_", 1)[0] for r in map(json.loads, open("prompts/content_runs.jsonl"))
                  if r["set"] == "content"})
    assert len(ops) == 24, ops
    P = "$H/prompts/content_runs.jsonl"
    for o in ops:
        cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{o}_mis_{c}_" for c in ARMS)]
        cmds += [f"{GEN} --prompts {P} --prompt_id {o}_mis --tag {c} {x} --seeds 1,2,3 --latent_frames 243 --outdir {D}"
                 for c, x in ARMS.items()]
        T.append(task(f"s21a_{o}", SF, cmds, f"{D}/{o}_mis_sf60_s0_s3.npz", 8 * 3 * len(ARMS)))
    for c, x in ARMS.items():
        T.append(task(f"s21f_{c}", SF, [f"{GEN} --prompts $H/prompts/content.jsonl --prompt_id fresh --tag {c} {x} "
                                        f"--fresh --seeds {','.join(map(str, range(8, 16)))} --latent_frames 243 "
                                        f"--outdir {D}"], f"{D}/fresh_{c}_s15.npz", 64))
elif sys.argv[1] == "e":
    T.append(task("s21e_clip", PRE, [f"$PY $H/clip_probe.py '{D}/*.mp4' {D}/clip_text.npz {D}/clip.done",
                                     f"$PY $H/siglip_probe.py '{D}/*.mp4' {D}/siglip_text.npz {D}/siglip.done"],
                  f"{D}/siglip.done", 180))
with open(f"jobs/s21_{sys.argv[1]}.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
