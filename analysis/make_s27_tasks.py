"""Build the Argon task list of study 27, a sink held at one query-sink distance across 17 distances
(docs/preregistrations/27_held_distance.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), L* -16 / +16, generic prompt, Self Forcing DMD, sink 3,
60 s, seed 0, hold_gen_sinkpos.py (Self-Forcing-b) with --sink_mode dist:D for the 14 distances of s27_distances.py
(from sinkprobe_long_out.json) and the 3 distances of study 25 (20, 187, 203); tags sp_dDDD. One task per scene runs
all 17 arms and both versions (arch_guard over the scene's outputs; an arm whose p16 output exists is skipped, so a
restarted task resumes); videos only for rx00 and gx00; any GPU with >= 40 GB and bf16.          -> jobs/s27.jsonl
usage: python make_s27_tasks.py sinkprobe_long_out.json
"""
import json
import sys

from s27_distances import OLD, select

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s27_held"
SF = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/Self-Forcing-b && cd {H}/Self-Forcing-b"
GEN = "$PY $H/hold_gen_sinkpos.py"
DIST = sorted([d for d, _ in select(json.load(open(sys.argv[1])))] + list(OLD))
ARMS = {f"sp_d{d:03d}": f"--sink_mode dist:{d}" for d in DIST}
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
T = []
for s in SCENES:
    nv = "" if s in ("rx00", "gx00") else " --no_video"
    cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{s}_{c}__" for c in ARMS)]
    cmds += [f"([ -f {D}/{s}_{c}__p16_s0.npz ] || {GEN} --prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {c} "
             f"--sink 3 {x} --seeds 0 --latent_frames 243 --versions m16:-16:0,p16:16:0 --outdir {D}{nv})"
             for c, x in ARMS.items()]
    T.append({"id": f"s27_{s}", "cmd": SF + f" && mkdir -p {D} && " + " && ".join(cmds),
              "done": f"{D}/{s}_{list(ARMS)[-1]}__p16_s0.npz",
              "est_min": 220, "mem_gb": 30, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s27.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks,", len(ARMS), "arms:", list(ARMS))
print(round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
