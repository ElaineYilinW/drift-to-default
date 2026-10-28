"""Build the Argon task list of study 28, the sink's rotary position in LongLive
(docs/preregistrations/28_longlive_sink_position.md).

The 24 scenes of preregistration 06 (rx00-rx11, gx00-gx11), L* -16 / +16, generic prompt, LongLive (sink 3, window 12),
60 s, seed 0, ll_hold_gen_sinkpos.py in LongLive-c (patched by jobs/patch_ll_probe.py and jobs/patch_ll_sinkpos.py):
  ll_abs     sink at its original positions (no switch)
  ll_off18   sink 18 latent frames further back      (--sink_mode off:18)
  ll_off36   sink 36 latent frames further back      (--sink_mode off:36)
  ll_dDDD    sink held at distance DDD               (--sink_mode dist:DDD), the 17 distances of study 27
One task per scene runs all 20 arms and both versions (arch_guard over the scene's outputs; an arm whose p16 output
exists is skipped, so a restarted task resumes); videos only for rx00 and gx00; any GPU with >= 40 GB and bf16.
                                                                                          -> jobs/s28.jsonl
usage: python make_s28_tasks.py sinkprobe_long_out.json
"""
import json
import sys

from s27_distances import OLD, select

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s28_llpos"
LL = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/LongLive-c && cd {H}/LongLive-c"
GEN = "$PY $H/ll_hold_gen_sinkpos.py"
DIST = sorted([d for d, _ in select(json.load(open(sys.argv[1])))] + list(OLD))
ARMS = {"ll_abs": "", "ll_off18": "--sink_mode off:18", "ll_off36": "--sink_mode off:36"}
ARMS.update({f"ll_d{d:03d}": f"--sink_mode dist:{d}" for d in DIST})
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(12)]
T = []
for s in SCENES:
    nv = "" if s in ("rx00", "gx00") else " --no_video"
    cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{s}_{c}__" for c in ARMS)]
    cmds += [f"([ -f {D}/{s}_{c}__p16_s0.npz ] || {GEN} --prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {c} "
             f"--sink 3 {x} --seeds 0 --latent_frames 243 --versions m16:-16:0,p16:16:0 --outdir {D}{nv})"
             for c, x in ARMS.items()]
    T.append({"id": f"s28_{s}", "cmd": LL + f" && mkdir -p {D} && " + " && ".join(cmds),
              "done": f"{D}/{s}_{list(ARMS)[-1]}__p16_s0.npz",
              "est_min": 360, "mem_gb": 30, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s28.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks,", len(ARMS), "arms:", list(ARMS))
print(round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
