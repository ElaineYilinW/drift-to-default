"""Build the Argon task list of study 33, the attractor offset against the inference window
(docs/preregistrations/33_window_dose.md).

As studies 30 and 32 (the 40 prompts of prompts/main.jsonl, plain text-to-video, 60 s = 243 latent frames, seeds 10,
11, 12, no sink), seven new arms that change only the key-value window (latent frames):
  sf06_s0, sf09_s0, sf15_s0, sf18_s0  Self Forcing DMD, window 6, 9, 15, 18  (hold_gen_long_rr.py --window W --sink 0)
  ll06_s0, ll21_s0                    LongLive, window 6, 21                  (ll_hold_gen_long.py --window W --sink 0)
  cv12_s0                             CausVid, window 12                      (hold_gen_long_rr.py --window 12 --sink 0,
                                                                               CausVid checkpoint and sampler)
to be read with study 32's sf12_s0 and study 30's sf_s0 (window 21), ll_s0 (window 12) and cv_s0 (window 21). One task
per prompt and arm (all three seeds); videos only for drv_dawn_coast and mg0084, seed 10. Any GPU with >= 40 GB and
bf16.                                                                  -> jobs/s33.jsonl
usage: python make_s33_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s33_fresh"
ENV = f"source {H}/jobs/env.sh"
SF = f"{ENV} && cd $H/Self-Forcing"
LL = f"{ENV} && export PYTHONPATH={H}/LongLive/LongLive1.0 && cd {H}/LongLive/LongLive1.0"
CV = f"--ckpt {H}/models/causvid/autoregressive_checkpoint/model.pt --steps 1000,757,522 --warp 0 --shift 8"
ARMS = {"sf06_s0": (SF, "hold_gen_long_rr.py", "--window 6 --sink 0"),
        "sf09_s0": (SF, "hold_gen_long_rr.py", "--window 9 --sink 0"),
        "sf15_s0": (SF, "hold_gen_long_rr.py", "--window 15 --sink 0"),
        "sf18_s0": (SF, "hold_gen_long_rr.py", "--window 18 --sink 0"),
        "ll06_s0": (LL, "ll_hold_gen_long.py", "--window 6 --sink 0"),
        "ll21_s0": (LL, "ll_hold_gen_long.py", "--window 21 --sink 0"),
        "cv12_s0": (SF, "hold_gen_long_rr.py", f"--window 12 --sink 0 {CV}")}
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
        T.append({"id": f"s33_{arm}_{p}", "cmd": f"{env} && mkdir -p {D} && {cmd}",
                  "done": f"{D}/{p}_{arm}_s{SEEDS[-1]}.npz", "est_min": 40, "mem_gb": 40, "gpus": 1, "gpu_mem_gb": 40,
                  "gpu_bf16": True, "cores": 16})
with open("jobs/s33.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
