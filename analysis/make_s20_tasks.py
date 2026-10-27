"""Build the Argon task list of study 20, new seeds for the Self Forcing sink results
(docs/preregistrations/20_seed_replication.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), L* -16 / +16, generic prompt, Self Forcing DMD, 60 s, seeds
1, 2, 3 (seed 0 ran in studies 10 and 16), three arms: sf60_s3 (sink 3 at its original positions), sf60_s0 (no sink),
sf60_s3rr (sink 3 re-rotated). One task per scene runs all arms, seeds and versions (arch_guard over the scene's
outputs); any GPU with >= 40 GB and bf16.                                                -> jobs/s20.jsonl
usage: python make_s20_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s20_seeds"
SF = f"source {H}/jobs/env.sh && cd $H/Self-Forcing"
GEN = "$PY $H/hold_gen_long_rr.py"
ARMS = {"sf60_s3": "--sink 3", "sf60_s0": "--sink 0", "sf60_s3rr": "--sink 3 --rerotate"}
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
T = []
for s in SCENES:
    cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{s}_{c}__" for c in ARMS)]
    cmds += [f"{GEN} --prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {c} {x} --seeds 1,2,3 --latent_frames 243 "
             f"--versions m16:-16:0,p16:16:0 --outdir {D}" for c, x in ARMS.items()]
    T.append({"id": f"s20_{s}", "cmd": SF + " && " + " && ".join(cmds), "done": f"{D}/{s}_sf60_s3rr__p16_s3.npz",
              "est_min": 110, "mem_gb": 30, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s20.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
