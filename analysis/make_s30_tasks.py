"""Build the Argon task list of study 30, the attractor offset of fresh rollouts on 40 prompts
(docs/preregistrations/30_attractor_offset.md).

The 40 prompts of prompts/main.jsonl (20 MovieGen, 20 driving), plain text-to-video (no opening), 60 s (243 latent
frames), seeds 10, 11, 12, seven arms with the frozen 60-s+ generators:
  sf_s0   Self Forcing DMD, no sink              (hold_gen_long_rr.py --sink 0)
  sf_rr   Self Forcing DMD, sink 3 re-rotated    (hold_gen_long_rr.py --sink 3 --rerotate)
  cv_s0   CausVid, no sink                       (hold_gen_long_rr.py --sink 0, CausVid checkpoint and sampler)
  ll_s0   LongLive, no sink                      (ll_hold_gen_long.py --sink 0)
  rf      Rolling Forcing                        (rf_gen_long.py)
  sf_s3   Self Forcing DMD, sink 3 at its original positions (secondary)
  ll_s3   LongLive, sink 3 at its original positions (secondary)
One task per prompt and arm (all three seeds); videos only for drv_dawn_coast and mg0084, seed 10. Any GPU with
>= 40 GB and bf16.                                                     -> jobs/s30.jsonl
usage: python make_s30_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s30_fresh"
ENV = f"source {H}/jobs/env.sh"
SF = f"{ENV} && cd $H/Self-Forcing"
LL = f"{ENV} && export PYTHONPATH={H}/LongLive/LongLive1.0 && cd {H}/LongLive/LongLive1.0"
RF = f"{ENV} && export PYTHONPATH={H}/RollingForcing && cd {H}/RollingForcing"
CV = f"--ckpt {H}/models/causvid/autoregressive_checkpoint/model.pt --steps 1000,757,522 --warp 0 --shift 8"
ARMS = {"sf_s0": (SF, "hold_gen_long_rr.py", "--sink 0"),
        "sf_rr": (SF, "hold_gen_long_rr.py", "--sink 3 --rerotate"),
        "cv_s0": (SF, "hold_gen_long_rr.py", f"--sink 0 {CV}"),
        "ll_s0": (LL, "ll_hold_gen_long.py", "--sink 0"),
        "rf": (RF, "rf_gen_long.py", ""),
        "sf_s3": (SF, "hold_gen_long_rr.py", "--sink 3"),
        "ll_s3": (LL, "ll_hold_gen_long.py", "--sink 3")}
PROMPTS = [json.loads(l)["id"] for l in open("prompts/main.jsonl")]
SEEDS = (10, 11, 12)
LAT = 243
VIDEO = ("drv_dawn_coast", "mg0084")
T = []
for p in PROMPTS:
    for arm, (env, prog, x) in ARMS.items():
        base = (f"$PY $H/{prog} --prompts $H/prompts/main.jsonl --prompt_id {p} --tag {arm} {x} --fresh "
                f"--latent_frames {LAT} --outdir {D}")
        if p in VIDEO:
            cmd = f"{base} --seeds {SEEDS[0]} && {base} --seeds {','.join(map(str, SEEDS[1:]))} --no_video"
        else:
            cmd = f"{base} --seeds {','.join(map(str, SEEDS))} --no_video"
        T.append({"id": f"s30_{arm}_{p}", "cmd": f"{env} && mkdir -p {D} && {cmd}",
                  "done": f"{D}/{p}_{arm}_s{SEEDS[-1]}.npz", "est_min": 40, "mem_gb": 40, "gpus": 1, "gpu_mem_gb": 40,
                  "gpu_bf16": True, "cores": 16})
with open("jobs/s30.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
