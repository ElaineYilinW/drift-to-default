"""Build the Argon task list of study 29, the 240-s horizon on new seeds (docs/preregistrations/29_steady_share.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), generic prompt, seeds 1 and 2 (study 24 used seed 0),
240 s (963 latent frames; Wan's rotary table has 1024 temporal positions), L* -16 / +16 (m16, p16), three arms with the
frozen generators of study 24:
  sf240_s3rr  Self Forcing DMD, sink 3 re-rotated             (hold_gen_long_rr.py --sink 3 --rerotate)
  rf240       Rolling Forcing (anchor, re-aligned positions)  (rf_gen_long.py)
  sf240_s0    Self Forcing DMD, no sink                       (hold_gen_long_rr.py --sink 0)
One task per scene, arm and seed (both versions, so a pair shares a GPU architecture via arch_guard); each seed has its
own output folder (s1/, s2/), because arch_guard matches outputs by scene and arm only; videos are written only for
rx00 and gx00, seed 1. Any GPU with >= 40 GB and bf16.          -> jobs/s29.jsonl
usage: python make_s29_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s29_long"
ENV = f"source {H}/jobs/env.sh"
SF = f"{ENV} && cd $H/Self-Forcing"
RF = f"{ENV} && export PYTHONPATH={H}/RollingForcing && cd {H}/RollingForcing"
ARMS = {"sf240_s3rr": (SF, "hold_gen_long_rr.py", "--sink 3 --rerotate"),
        "rf240": (RF, "rf_gen_long.py", ""),
        "sf240_s0": (SF, "hold_gen_long_rr.py", "--sink 0")}
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
SEEDS = (1, 2)
LAT = 963
T = []
for seed in SEEDS:
    for s in SCENES:
        for arm, (env, prog, x) in ARMS.items():
            nv = "" if s in ("rx00", "gx00") and seed == 1 else " --no_video"
            o = f"{D}/s{seed}"
            cmd = (f"{env} && mkdir -p {o} && $PY $H/arch_guard.py {o} {s}_{arm}__ && $PY $H/{prog} "
                   f"--prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {arm} {x} --seeds {seed} "
                   f"--latent_frames {LAT} --versions m16:-16:0,p16:16:0 --outdir {o}{nv}")
            T.append({"id": f"s29_{arm}_{s}_s{seed}", "cmd": cmd, "done": f"{o}/{s}_{arm}__p16_s{seed}.npz",
                      "est_min": 60, "mem_gb": 80, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s29.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
