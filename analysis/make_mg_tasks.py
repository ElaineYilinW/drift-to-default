"""Argon task list of the world-model study (docs/preregistration_mg_cn.md): Matrix-Game-2.0 universal, 24 scenes of
the relaxation study, constant forward action, 30 s (189 latent frames), seed 0, openings L* -16 / +16, the shift sent
through both routes, the context latent only, or the CLIP embedding only; all three of a scene in one task with
arch_guard.   usage: python make_mg_tasks.py   (writes jobs/mg.jsonl)"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/mg"
PRE = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/shims:{H}/MatrixGame/Matrix-Game-2 && cd {H}/MatrixGame/Matrix-Game-2"
scenes = [json.loads(l)["id"] for l in open("prompts/relax.jsonl") if json.loads(l)["id"] != "fresh"]
assert len(scenes) == 24
T = []
for s in scenes:
    cmds = [f"$PY $H/arch_guard.py {D} {s}_mg_both__ {s}_mg_ctx__ {s}_mg_emb__"]
    cmds += [f"$PY $H/mg_gen.py --prompts $H/prompts/relax.jsonl --prompt_id {s} --tag mg_{ch} --channel {ch} "
             f"--latent_frames 189 --seeds 0 --versions m16:-16:0,p16:16:0 --outdir {D}" for ch in ("both", "ctx", "emb")]
    T.append({"id": f"mg_{s}", "cmd": PRE + " && " + " && ".join(cmds), "done": f"{D}/{s}_mg_emb__p16_s0.npz",
              "est_min": 20, "mem_gb": 40, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
open("jobs/mg.jsonl", "w").write("".join(json.dumps(t) + "\n" for t in T))
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h")
