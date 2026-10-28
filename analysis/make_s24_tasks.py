"""Build the Argon task list of study 24, the 180-s horizon (docs/preregistrations/24_long_horizon.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), generic prompt, seed 0, 180 s (723 latent frames), L*
-16 / +16 (m16, p16), five arms:
  sf180_s0    Self Forcing DMD, no sink                       (hold_gen_long_rr.py --sink 0)
  sf180_s3rr  Self Forcing DMD, sink 3 re-rotated             (hold_gen_long_rr.py --sink 3 --rerotate)
  sf180_s3    Self Forcing DMD, sink 3 at original positions  (hold_gen_long_rr.py --sink 3)
  rf180       Rolling Forcing (anchor, re-aligned positions)  (rf_gen_long.py)
  ll180_s3    LongLive, sink 3                                (ll_hold_gen_long.py --sink 3)
One task per scene and arm (both versions, so a scene's pair shares a GPU architecture via arch_guard); videos are
written only for rx00 and gx00 (statistics are computed from the frames either way). Fresh starts (no opening, seeds
0-3) per arm for the attractor offset (secondary). Any GPU with >= 40 GB and bf16.          -> jobs/s24.jsonl
usage: python make_s24_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s24_long"
ENV = f"source {H}/jobs/env.sh"
SF = f"{ENV} && cd $H/Self-Forcing"
LL = f"{ENV} && export PYTHONPATH={H}/LongLive/LongLive1.0 && cd {H}/LongLive/LongLive1.0"
RF = f"{ENV} && export PYTHONPATH={H}/RollingForcing && cd {H}/RollingForcing"
ARMS = {"sf180_s0": (SF, "hold_gen_long_rr.py", "--sink 0"),
        "sf180_s3rr": (SF, "hold_gen_long_rr.py", "--sink 3 --rerotate"),
        "sf180_s3": (SF, "hold_gen_long_rr.py", "--sink 3"),
        "rf180": (RF, "rf_gen_long.py", ""),
        "ll180_s3": (LL, "ll_hold_gen_long.py", "--sink 3")}
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
LAT = 723
T = []
for s in SCENES:
    for arm, (env, prog, x) in ARMS.items():
        nv = "" if s in ("rx00", "gx00") else " --no_video"
        cmd = (f"{env} && $PY $H/arch_guard.py {D} {s}_{arm}__ && $PY $H/{prog} --prompts $H/prompts/relax.jsonl "
               f"--prompt_id {s} --tag {arm} {x} --seeds 0 --latent_frames {LAT} --versions m16:-16:0,p16:16:0 "
               f"--outdir {D}{nv}")
        T.append({"id": f"s24_{arm}_{s}", "cmd": cmd, "done": f"{D}/{s}_{arm}__p16_s0.npz",
                  "est_min": 50, "mem_gb": 64, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
for arm, (env, prog, x) in ARMS.items():
    cmd = (f"{env} && $PY $H/{prog} --prompts $H/prompts/relax.jsonl --prompt_id rx00 --tag fresh_{arm} {x} --fresh "
           f"--seeds 0,1,2,3 --latent_frames {LAT} --outdir {D}/fresh --no_video")
    T.append({"id": f"s24_fresh_{arm}", "cmd": cmd, "done": f"{D}/fresh/rx00_fresh_{arm}_s3.npz",
              "est_min": 95, "mem_gb": 64, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s24.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
