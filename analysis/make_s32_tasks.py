"""Build the Argon task list of study 32, the window-matched attractor offset
(docs/preregistrations/32_window_matched_offset.md).

As study 30 (make_s30_tasks.py: the 40 prompts of prompts/main.jsonl, plain text-to-video, 60 s = 243 latent frames,
seeds 10, 11, 12), two new arms: Self Forcing DMD with LongLive's inference window of 12 latent frames,
  sf12_s0  Self Forcing DMD, window 12, no sink                       (hold_gen_long_rr.py --window 12 --sink 0)
  sf12_s3  Self Forcing DMD, window 12, sink 3 at its original positions (hold_gen_long_rr.py --window 12 --sink 3)
to be compared with study 30's ll_s0 and ll_s3 (LongLive, window 12) and sf_s0 and sf_s3 (Self Forcing, window 21).
One task per prompt and arm (all three seeds); videos only for drv_dawn_coast and mg0084, seed 10. Any GPU with
>= 40 GB and bf16.                                                     -> jobs/s32.jsonl
usage: python make_s32_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s32_fresh"
SF = f"source {H}/jobs/env.sh && cd $H/Self-Forcing"
ARMS = {"sf12_s0": "--window 12 --sink 0", "sf12_s3": "--window 12 --sink 3"}
PROMPTS = [json.loads(l)["id"] for l in open("prompts/main.jsonl")]
SEEDS = (10, 11, 12)
LAT = 243
VIDEO = ("drv_dawn_coast", "mg0084")
T = []
for p in PROMPTS:
    for arm, x in ARMS.items():
        base = (f"$PY $H/hold_gen_long_rr.py --prompts $H/prompts/main.jsonl --prompt_id {p} --tag {arm} {x} --fresh "
                f"--latent_frames {LAT} --outdir {D}")
        if p in VIDEO:
            cmd = f"{base} --seeds {SEEDS[0]} && {base} --seeds {','.join(map(str, SEEDS[1:]))} --no_video"
        else:
            cmd = f"{base} --seeds {','.join(map(str, SEEDS))} --no_video"
        T.append({"id": f"s32_{arm}_{p}", "cmd": f"{SF} && mkdir -p {D} && {cmd}",
                  "done": f"{D}/{p}_{arm}_s{SEEDS[-1]}.npz", "est_min": 40, "mem_gb": 40, "gpus": 1, "gpu_mem_gb": 40,
                  "gpu_bf16": True, "cores": 16})
with open("jobs/s32.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
