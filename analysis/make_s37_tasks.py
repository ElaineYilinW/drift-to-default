"""Build the Argon task list of study 37, calibrating the held-sink distance on two new checkpoints
(docs/preregistrations/37_held_sink_rule.md).

Models: CausVid (autoregressive_checkpoint, 3 steps 1000,757,522, shift 8, no warp) and Self Forcing GAN, both sink 3,
window 21, in Self-Forcing-b (hold_gen_sinkpos.py, frozen). Candidate distances: study 27's rule (s27_distances.select,
K = 14) applied to each model's own exploratory probe (analyze_sinkprobe_long.py output on probe37_cv / probe37_gan),
plus the re-aligned distance 20 as a reference. Calibration scenes: rx00-rx03, gx00-gx03 (8 of the 24 scenes of
preregistration 06), L* -16 / +16, generic prompt, seed 0, 60 s (243 latent frames). One task per model and scene runs
every distance (arch_guard over the scene's outputs; an arm whose p16 output exists is skipped). No video files.
                                                                                     -> jobs/s37.jsonl
usage: python make_s37_tasks.py PROBE_CV.json PROBE_GAN.json
"""
import json
import sys

from s27_distances import select

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s37_calib"
SF = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/Self-Forcing-b && cd {H}/Self-Forcing-b"
GEN = "$PY $H/hold_gen_sinkpos.py"
MODELS = {"cv": "--ckpt $H/models/causvid/autoregressive_checkpoint/model.pt --steps 1000,757,522 --warp 0 --shift 8",
          "gan": "--ckpt $H/models/self_forcing/checkpoints/self_forcing_gan.pt"}
SCENES = [f"{p}{i:02d}" for p in ("rx", "gx") for i in range(4)]
T, cand = [], {}
for m, probe in zip(MODELS, sys.argv[1:3]):
    dist = sorted({d for d, _ in select(json.load(open(probe)))} | {20})
    cand[m] = dist
    arms = {f"{m}_d{d:03d}": f"--sink_mode dist:{d}" for d in dist}
    for s in SCENES:
        cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{s}_{c}__" for c in arms)]
        cmds += [f"([ -f {D}/{s}_{c}__p16_s0.npz ] || {GEN} --prompts $H/prompts/relax.jsonl --prompt_id {s} --tag {c} "
                 f"--sink 3 {MODELS[m]} {x} --seeds 0 --latent_frames 243 --versions m16:-16:0,p16:16:0 "
                 f"--outdir {D} --no_video)" for c, x in arms.items()]
        T.append({"id": f"s37_{m}_{s}", "cmd": SF + f" && mkdir -p {D} && " + " && ".join(cmds),
                  "done": f"{D}/{s}_{list(arms)[-1]}__p16_s0.npz", "est_min": 20 * len(arms), "mem_gb": 30,
                  "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/s37.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
json.dump(cand, open("jobs/s37_candidates.json", "w"))
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)", cand)
