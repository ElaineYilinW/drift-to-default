"""Build the Argon task lists of study 16 (docs/preregistrations/16_rerotation_and_horizon.md).

A  content under a re-rotated sink: the 24 openings of preregistration 09, generic city prompt (arm mis), Self Forcing
   DMD sink 3 / sink 3 re-rotated / no sink, 60 s; all three arms of an opening in one task (arch_guard); fresh starts
   of the generic prompt (8 seeds) per configuration                                      -> jobs/s16_a.jsonl
B  tone under a deeper re-rotated sink: the 24 scenes of preregistration 06, L* -16 / +16, Self Forcing DMD sink 6 /
   sink 6 re-rotated / sink 3 re-rotated, 60 s, one task per scene (arch_guard)           -> jobs/s16_a.jsonl
C  FramePack-F1 at 60 s: the 12 scenes of preregistration 08, both routes / context only, L* -16 / +16
D  SkyReels-V2-DF at 60 s: the same 12 scenes, overlap_history 5 / 17, L* -16 / +16       -> jobs/s16_cd.jsonl
   C and D: one video per task, every task pinned to A40 (gpu_only) so that the two versions of a scene share an
   architecture; arch_guard checks it again at the start of each task.
E  CLIP embeddings of the part-A videos (after A has finished)                            -> jobs/s16_e.jsonl
usage: python make_s16_tasks.py ab|cd|e
"""
import json
import sys

H = "/Users/ywang906/vdrift"
R = "/nfsscratch/Users/ywang906/vdrift/corr"
DA, DB, DC, DD = f"{R}/s16_content", f"{R}/s16_sink", f"{R}/s16_fp", f"{R}/s16_sky"
SF = f"source {H}/jobs/env.sh && cd $H/Self-Forcing"
PRE = f"source {H}/jobs/env.sh && cd $H"
GEN = "$PY $H/hold_gen_long_rr.py"
ARM_A = {"sf60_s3": "--sink 3", "sf60_s3rr": "--sink 3 --rerotate", "sf60_s0": "--sink 0"}
ARM_B = {"sf60_s6": "--sink 6", "sf60_s6rr": "--sink 6 --rerotate", "sf60_s3rr": "--sink 3 --rerotate"}
SCENES24 = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
SCENES12 = ["rx02", "rx06", "rx08", "rx10", "gx02", "gx06", "gx08", "gx10", "rx01", "rx03", "gx01", "gx03"]
VERS = {"m16": "m16:-16:0", "p16": "p16:16:0"}
FP = f"$H/envs/sky/bin/python $H/fp_gen.py --prompts $H/prompts/relax.jsonl --seconds 60 --seeds 0 --outdir {DC}"
SKY = f"$H/envs/sky/bin/python $H/sky_gen_x.py --prompts $H/prompts/relax.jsonl --seconds 60 --seeds 0 --outdir {DD}"
ARM_C = {"fpv_both": "--mode vanilla --channel both", "fpv_ctx": "--mode vanilla --channel ctx"}
ARM_D = {"sky5": "--overlap 5", "sky17": "--overlap 17"}


def task(tid, pre, cmds, done, est, **kw):
    t = {"id": tid, "cmd": pre + " && " + " && ".join(cmds), "done": done, "est_min": est, "mem_gb": 30, "gpus": 1,
         "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16}
    t.update(kw)
    return t


phase = sys.argv[1]
T = []
if phase == "ab":
    ops = sorted({r["id"].rsplit("_", 1)[0] for r in map(json.loads, open("prompts/content_runs.jsonl"))
                  if r["set"] == "content"})
    assert len(ops) == 24, ops
    P = "$H/prompts/content_runs.jsonl"
    for o in ops:
        cmds = [f"$PY $H/arch_guard.py {DA} " + " ".join(f"{o}_mis_{c}_" for c in ARM_A)]
        cmds += [f"{GEN} --prompts {P} --prompt_id {o}_mis --tag {c} {x} --seeds 0 --latent_frames 243 --outdir {DA}"
                 for c, x in ARM_A.items()]
        T.append(task(f"s16a_{o}", SF, cmds, f"{DA}/{o}_mis_sf60_s0_s0.npz", 8 * len(ARM_A)))
    seeds = ",".join(map(str, range(8)))
    for c, x in ARM_A.items():
        T.append(task(f"s16f_{c}", SF, [f"{GEN} --prompts $H/prompts/content.jsonl --prompt_id fresh --tag {c} {x} --fresh "
                                        f"--seeds {seeds} --latent_frames 243 --outdir {DA}"], f"{DA}/fresh_{c}_s7.npz", 64))
    for s in SCENES24:
        cmds = [f"$PY $H/arch_guard.py {DB} " + " ".join(f"{s}_{c}__" for c in ARM_B)]
        cmds += [f"{GEN} --prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {c} {x} --seeds 0 --latent_frames 243 "
                 f"--versions {VERS['m16']},{VERS['p16']} --outdir {DB}" for c, x in ARM_B.items()]
        T.append(task(f"s16b_{s}", SF, cmds, f"{DB}/{s}_sf60_s3rr__p16_s0.npz", 8 * 2 * len(ARM_B)))
elif phase == "cd":
    for s in SCENES12:
        for v in VERS:
            for c, x in ARM_C.items():
                cmds = [f"$PY $H/arch_guard.py {DC} {s}_{c}__",
                        f"{FP} --prompt_id {s} --tag {c} {x} --versions {VERS[v]}"]
                T.append(task(f"s16c_{s}_{c}_{v}", PRE, cmds, f"{DC}/{s}_{c}__{v}_s0.npz", 190, mem_gb=60, gpu_mem_gb=45,
                              cores=24, gpu_only="a40"))
            for c, x in ARM_D.items():
                cmds = [f"$PY $H/arch_guard.py {DD} {s}_{c}__",
                        f"{SKY} --prompt_id {s} --tag {c} {x} --versions {VERS[v]}"]
                T.append(task(f"s16d_{s}_{c}_{v}", PRE, cmds, f"{DD}/{s}_{c}__{v}_s0.npz", 150, mem_gb=60, gpu_mem_gb=45,
                              cores=24, gpu_only="a40"))
elif phase == "e":
    T.append(task("s16e_clip", PRE, [f"$PY $H/clip_probe.py '{DA}/*.mp4' {R}/content/clip_text.npz {DA}/clip.done"],
                  f"{DA}/clip.done", 90))
with open(f"jobs/s16_{phase}.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(phase, len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h")
