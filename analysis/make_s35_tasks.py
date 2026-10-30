"""Build the Argon task list of study 35, one rotary frequency of the sink in LongLive and a single-frequency control
(docs/preregistrations/35_band_hold_ll.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), L* -16 / +16 (m16, p16), generic prompt, sink 3, 60 s (243
latent frames), seed 21 (not used before on these scenes), six arms:
  ll_abs   LongLive (window 12), sink at its original positions          ll_hold_gen_bandhold.py in LongLive-d
  ll_j4    LongLive, temporal frequency 4 of the sink held at distance 20 (--sink_mode band:20:4)
  ll_j5    LongLive, temporal frequency 5 held at 20                     (--sink_mode band:20:5)
  sf_abs   Self Forcing DMD (window 21), sink at its original positions  hold_gen_bandhold.py in Self-Forcing-d
  sf_j4    Self Forcing, frequency 4 held at 20                          (--sink_mode band:20:4)
  sf_j5    Self Forcing, frequency 5 held at 20                          (--sink_mode band:20:5)
One task per scene and arm (both versions; arch_guard keeps a pair on one GPU architecture); videos only for rx00 and
gx00. Any GPU with >= 40 GB and bf16.                                   -> jobs/s35.jsonl
usage: python make_s35_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s35_band"
TREE = {"ll": ("LongLive-d", "ll_hold_gen_bandhold.py"), "sf": ("Self-Forcing-d", "hold_gen_bandhold.py")}
ARMS = {"ll_abs": "", "ll_j4": "--sink_mode band:20:4", "ll_j5": "--sink_mode band:20:5",
        "sf_abs": "", "sf_j4": "--sink_mode band:20:4", "sf_j5": "--sink_mode band:20:5"}
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
SEED, LAT = 21, 243
T = []
for s in SCENES:
    nv = "" if s in ("rx00", "gx00") else " --no_video"
    for arm, x in ARMS.items():
        tree, gen = TREE[arm[:2]]
        cmd = (f"source {H}/jobs/env.sh && export PYTHONPATH={H}/{tree} && cd {H}/{tree} && mkdir -p {D} && "
               f"$PY $H/arch_guard.py {D} {s}_{arm}__ && $PY $H/{gen} --prompts $H/prompts/relax.jsonl --prompt_id {s} "
               f"--tag {arm} --sink 3 {x} --seeds {SEED} --latent_frames {LAT} --versions m16:-16:0,p16:16:0 "
               f"--outdir {D}{nv}")
        T.append({"id": f"s35_{arm}_{s}", "cmd": cmd, "done": f"{D}/{s}_{arm}__p16_s{SEED}.npz",
                  "est_min": 30, "mem_gb": 64, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s35.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
