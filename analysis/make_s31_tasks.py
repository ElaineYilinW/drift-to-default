"""Build the Argon task list of study 31, a sink held at a probe-chosen distance as a lever, 180 s
(docs/preregistrations/31_held_sink_lever.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), L* -16 / +16 (m16, p16), generic prompt, Self Forcing DMD,
sink 3, window 21, 180 s (723 latent frames), seed 4, hold_gen_sinkpos.py (Self-Forcing-b; it records motion), four arms:
  sp_d066  sink held at query-sink distance 66   (--sink_mode dist:66)
  sp_d448  sink held at query-sink distance 448  (--sink_mode dist:448)
  sp_rr    sink re-rotated to sit just before the window (--rerotate)
  sp_abs   sink at its original positions (no flag)
One task per scene and arm (both versions; arch_guard keeps a pair on one GPU architecture); videos only for rx00 and
gx00. Any GPU with >= 40 GB and bf16.                                   -> jobs/s31.jsonl
usage: python make_s31_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s31_lever"
SF = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/Self-Forcing-b && cd {H}/Self-Forcing-b"
ARMS = {"sp_d066": "--sink_mode dist:66", "sp_d448": "--sink_mode dist:448", "sp_rr": "--rerotate", "sp_abs": ""}
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
SEED, LAT = 4, 723
T = []
for s in SCENES:
    nv = "" if s in ("rx00", "gx00") else " --no_video"
    for arm, x in ARMS.items():
        cmd = (f"{SF} && mkdir -p {D} && $PY $H/arch_guard.py {D} {s}_{arm}__ && $PY $H/hold_gen_sinkpos.py "
               f"--prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {arm} --sink 3 {x} --seeds {SEED} "
               f"--latent_frames {LAT} --versions m16:-16:0,p16:16:0 --outdir {D}{nv}")
        T.append({"id": f"s31_{arm}_{s}", "cmd": cmd, "done": f"{D}/{s}_{arm}__p16_s{SEED}.npz",
                  "est_min": 60, "mem_gb": 64, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s31.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
