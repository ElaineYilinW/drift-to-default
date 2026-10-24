"""Build the Argon task list of the FramePack study (docs/preregistration_fp_cn.md).

12 scenes of preregistration 08 (SkyReels overlap), generic prompt, seed 0, 20 s, openings L* -16 / +16.
  A  FramePack-F1 (vanilla), shift through both routes, and through the context latent only (paired: one task)
  B  FramePack-F1 (vanilla), shift through the SigLIP embedding only
  C  FramePackI2V (inverted anti-drifting), both routes
Tasks start with arch_guard. usage: python make_fp_tasks.py   (writes jobs/fp.jsonl)
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/fp"
SCENES = ["rx02", "rx06", "rx08", "rx10", "gx02", "gx06", "gx08", "gx10", "rx01", "rx03", "gx01", "gx03"]
PRE = f"source {H}/jobs/env.sh && cd $H"
V = "m16:-16:0,p16:16:0"
FP = f"$H/envs/sky/bin/python $H/fp_gen.py --prompts $H/prompts/relax.jsonl --seconds 20 --seeds 0 --versions {V} --outdir {D}"
GROUPS = {"A": [("fpv_both", "--mode vanilla --channel both"), ("fpv_ctx", "--mode vanilla --channel ctx")],
          "B": [("fpv_emb", "--mode vanilla --channel emb")],
          "C": [("fpi_both", "--mode inverted --channel both")]}
T = []
for s in SCENES:
    for g, arms in GROUPS.items():
        cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{s}_{tag}__" for tag, _ in arms)]
        cmds += [f"{FP} --prompt_id {s} --tag {tag} {extra}" for tag, extra in arms]
        T.append({"id": f"fp{g}_{s}", "cmd": PRE + " && " + " && ".join(cmds), "done": f"{D}/{s}_{arms[-1][0]}__p16_s0.npz",
                  "est_min": 50 * 2 * len(arms), "mem_gb": 60, "gpus": 1, "gpu_mem_gb": 45, "gpu_bf16": True, "cores": 24})
with open("jobs/fp.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h")
