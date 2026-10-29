"""Build the Argon task list of study 34, single temporal RoPE frequencies of the sink held in generation
(docs/preregistrations/34_band_hold.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), L* -16 / +16 (m16, p16), generic prompt, Self Forcing DMD,
sink 3, window 21, 60 s (243 latent frames), seed 13, hold_gen_bandhold.py (Self-Forcing-d, patch_sf_bandhold.py; it
records motion), five arms:
  bh_abs   sink at its original positions (no flag)
  bh_j4    temporal frequency 4 of the sink held at query-sink distance 20  (--sink_mode band:20:4)
  bh_j3    temporal frequency 3 held at 20                                  (--sink_mode band:20:3)
  bh_j811  temporal frequencies 8-11 held at 20 (control band)              (--sink_mode band:20:8,9,10,11)
  bh_rr    sink re-rotated to sit just before the window (--rerotate)
One task per scene and arm (both versions; arch_guard keeps a pair on one GPU architecture); videos only for rx00 and
gx00. Any GPU with >= 40 GB and bf16.                                   -> jobs/s34.jsonl
usage: python make_s34_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s34_band"
SF = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/Self-Forcing-d && cd {H}/Self-Forcing-d"
ARMS = {"bh_abs": "", "bh_j4": "--sink_mode band:20:4", "bh_j3": "--sink_mode band:20:3",
        "bh_j811": "--sink_mode band:20:8,9,10,11", "bh_rr": "--rerotate"}
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
SEED, LAT = 13, 243
T = []
for s in SCENES:
    nv = "" if s in ("rx00", "gx00") else " --no_video"
    for arm, x in ARMS.items():
        cmd = (f"{SF} && mkdir -p {D} && $PY $H/arch_guard.py {D} {s}_{arm}__ && $PY $H/hold_gen_bandhold.py "
               f"--prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {arm} --sink 3 {x} --seeds {SEED} "
               f"--latent_frames {LAT} --versions m16:-16:0,p16:16:0 --outdir {D}{nv}")
        T.append({"id": f"s34_{arm}_{s}", "cmd": cmd, "done": f"{D}/{s}_{arm}__p16_s{SEED}.npz",
                  "est_min": 30, "mem_gb": 64, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s34.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
