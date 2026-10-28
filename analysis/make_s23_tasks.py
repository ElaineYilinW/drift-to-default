"""Build the Argon task list of study 23, dose-response of tone persistence (docs/preregistrations/23_dose_response.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), generic prompt, Self Forcing DMD, 60 s, seeds 0 and 1,
three arms: sf60_s3 (sink 3 at its original positions), sf60_s0 (no sink), sf60_s3rr (sink 3 re-rotated); five openings
per arm: L* -24, -8, 0, +8, +24 (m24, m8, z0, p8, p24). One task per scene and seed runs all arms and openings
(15 videos). Each seed writes to its own folder, and arch_guard keeps a scene's outputs in that folder on one GPU
architecture. Any GPU with >= 40 GB and bf16.                                                  -> jobs/s23.jsonl
usage: python make_s23_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s23_dose"
SF = f"source {H}/jobs/env.sh && cd $H/Self-Forcing"
GEN = "$PY $H/hold_gen_long_rr.py"
ARMS = {"sf60_s3": "--sink 3", "sf60_s0": "--sink 0", "sf60_s3rr": "--sink 3 --rerotate"}
VERSIONS = "m24:-24:0,m8:-8:0,z0:0:0,p8:8:0,p24:24:0"
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
T = []
for seed in (0, 1):
    out = f"{D}/seed{seed}"
    for s in SCENES:
        cmds = [f"$PY $H/arch_guard.py {out} " + " ".join(f"{s}_{c}__" for c in ARMS)]
        cmds += [f"{GEN} --prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {c} {x} --seeds {seed} "
                 f"--latent_frames 243 --versions {VERSIONS} --outdir {out}" for c, x in ARMS.items()]
        T.append({"id": f"s23_{s}_s{seed}", "cmd": SF + " && " + " && ".join(cmds),
                  "done": f"{out}/{s}_sf60_s3rr__p24_s{seed}.npz",
                  "est_min": 95, "mem_gb": 30, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s23.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
