"""Build the Argon task list of study 36, the attractor offset beyond the released window and the shape of Self
Forcing's step between windows 18 and 21 (docs/preregistrations/36_window_beyond.md).

As studies 30, 32 and 33 (the 40 prompts of prompts/main.jsonl, plain text-to-video, 60 s = 243 latent frames, seeds
10, 11, 12, no sink), five new arms that change only the key-value window (latent frames):
  sf19_s0, sf20_s0, sf24_s0, sf27_s0  Self Forcing DMD, window 19, 20, 24, 27  (hold_gen_long_rr.py --window W --sink 0)
  ll24_s0                             LongLive, window 24                      (ll_hold_gen_long.py --window 24 --sink 0)
to be read with studies 30, 32 and 33. One task per prompt and arm (all three seeds); videos only for drv_dawn_coast
and mg0084, seed 10. Any GPU with >= 40 GB and bf16.                   -> jobs/s36.jsonl
usage: python make_s36_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s36_fresh"
ENV = f"source {H}/jobs/env.sh"
SF = f"{ENV} && cd $H/Self-Forcing"
LL = f"{ENV} && export PYTHONPATH={H}/LongLive/LongLive1.0 && cd {H}/LongLive/LongLive1.0"
ARMS = {"sf19_s0": (SF, "hold_gen_long_rr.py", "--window 19 --sink 0"),
        "sf20_s0": (SF, "hold_gen_long_rr.py", "--window 20 --sink 0"),
        "sf24_s0": (SF, "hold_gen_long_rr.py", "--window 24 --sink 0"),
        "sf27_s0": (SF, "hold_gen_long_rr.py", "--window 27 --sink 0"),
        "ll24_s0": (LL, "ll_hold_gen_long.py", "--window 24 --sink 0")}
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
        T.append({"id": f"s36_{arm}_{p}", "cmd": f"{env} && mkdir -p {D} && {cmd}",
                  "done": f"{D}/{p}_{arm}_s{SEEDS[-1]}.npz", "est_min": 45, "mem_gb": 40, "gpus": 1, "gpu_mem_gb": 40,
                  "gpu_bf16": True, "cores": 16})
with open("jobs/s36.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
