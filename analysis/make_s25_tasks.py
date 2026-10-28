"""Build the Argon task list of study 25, the sink's rotary position and the bursts
(docs/preregistrations/25_sink_position.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), L* -16 / +16, generic prompt, Self Forcing DMD, sink 3,
60 s, seed 0, six arms of hold_gen_sinkpos.py (Self-Forcing-b, patched by jobs/patch_sf_sinkpos.py):
  sp_abs    sink at its original positions (no switch)
  sp_off18  sink 18 latent frames further back      (--sink_mode off:18)
  sp_off36  sink 36 latent frames further back      (--sink_mode off:36)
  sp_d203   sink held at distance 203 (a probe peak) (--sink_mode dist:203)
  sp_d187   sink held at distance 187 (a probe trough) (--sink_mode dist:187)
  sp_d20    sink held at distance 20 (next to the window, as re-rotation) (--sink_mode dist:20)
One task per scene runs all arms and versions (arch_guard over the scene's outputs); videos only for rx00 and gx00;
any GPU with >= 40 GB and bf16.                                                          -> jobs/s25.jsonl
usage: python make_s25_tasks.py
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s25_sinkpos"
SF = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/Self-Forcing-b && cd {H}/Self-Forcing-b"
GEN = "$PY $H/hold_gen_sinkpos.py"
ARMS = {"sp_abs": "", "sp_off18": "--sink_mode off:18", "sp_off36": "--sink_mode off:36",
        "sp_d203": "--sink_mode dist:203", "sp_d187": "--sink_mode dist:187", "sp_d20": "--sink_mode dist:20"}
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
T = []
for s in SCENES:
    nv = "" if s in ("rx00", "gx00") else " --no_video"
    cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{s}_{c}__" for c in ARMS)]
    cmds += [f"{GEN} --prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {c} --sink 3 {x} --seeds 0 "
             f"--latent_frames 243 --versions m16:-16:0,p16:16:0 --outdir {D}{nv}" for c, x in ARMS.items()]
    T.append({"id": f"s25_{s}", "cmd": SF + " && " + " && ".join(cmds), "done": f"{D}/{s}_sp_d20__p16_s0.npz",
              "est_min": 100, "mem_gb": 30, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s25.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
