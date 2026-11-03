"""Build the Argon task lists of study 39, prompt-only 60-s videos scored on VBench-style dimensions
(docs/preregistrations/39_held_sink_quality.md).

40 prompts (prompts/main.jsonl: 20 MovieGen, 20 driving), no held opening (--fresh), seed 13, 60 s (243 latent
frames), videos written. Models as study 38 (sf, cv, gan in Self-Forcing-b with hold_gen_sinkpos.py; ll in LongLive-c
with ll_hold_gen_sinkpos.py); arms per model with a D* from study 37 (tags {m}_{arm}):
  s0    no sink                                         (--sink 0)
  rr    sink 3 held at the re-aligned distance          (--sink 3 --sink_mode dist:20, LongLive dist:11)
  held  sink 3 held at D*                               (--sink 3 --sink_mode dist:D*)
  abs   sink 3 at its original positions                (--sink 3)
plus Rolling Forcing as it is released (rf_gen_long.py, tag rf) as a reference. One generation task per model and prompt
(all arms; arch_guard over the prompt's outputs; an arm whose npz exists is skipped)        -> jobs/s39.jsonl
One scoring task per model and prompt, vbench_lite.py over '{prompt}_{model}_*_s13.mp4'     -> jobs/s39_feat.jsonl
usage: python make_s39_tasks.py DSTAR.json [MODELS, default sf,ll,cv,gan,rf]
"""
import json
import sys

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s39_quality"
SFB = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/Self-Forcing-b && cd {H}/Self-Forcing-b"
LLC = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/LongLive-c && cd {H}/LongLive-c"
RF = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/RollingForcing && cd {H}/RollingForcing"
MODEL = {"sf": (SFB, "hold_gen_sinkpos.py", "", 10),
         "cv": (SFB, "hold_gen_sinkpos.py",
                "--ckpt $H/models/causvid/autoregressive_checkpoint/model.pt --steps 1000,757,522 --warp 0 --shift 8", 8),
         "gan": (SFB, "hold_gen_sinkpos.py", "--ckpt $H/models/self_forcing/checkpoints/self_forcing_gan.pt", 10),
         "ll": (LLC, "ll_hold_gen_sinkpos.py", "", 10)}
sel = json.load(open(sys.argv[1]))
models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sf", "ll", "cv", "gan", "rf"]
PROMPTS = [json.loads(l)["id"] for l in open("prompts/main.jsonl")]
assert len(PROMPTS) == 40
SEED, LAT = 13, 243
T, F = [], []
for m in models:
    if m == "rf":
        for p in PROMPTS:
            T.append({"id": f"s39_rf_{p}", "cmd": RF + f" && mkdir -p {D} && ([ -f {D}/{p}_rf_s{SEED}.npz ] || $PY $H/"
                      f"rf_gen_long.py --prompts $H/prompts/main.jsonl --prompt_id {p} --tag rf --seeds {SEED} "
                      f"--latent_frames {LAT} --fresh --outdir {D})", "done": f"{D}/{p}_rf_s{SEED}.npz", "est_min": 15,
                      "mem_gb": 30, "gpus": 1, "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
    else:
        if sel.get(m, {}).get("dstar") is None:
            print("no D* for", m, "- left out")
            continue
        pre, gen, extra, mins = MODEL[m]
        arms = {f"{m}_s0": "--sink 0", f"{m}_rr": f"--sink 3 --sink_mode dist:{sel[m]['realign']}",
                f"{m}_held": f"--sink 3 --sink_mode dist:{sel[m]['dstar']}", f"{m}_abs": "--sink 3"}
        for p in PROMPTS:
            cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{p}_{c}_" for c in arms)]
            cmds += [f"([ -f {D}/{p}_{c}_s{SEED}.npz ] || $PY $H/{gen} --prompts $H/prompts/main.jsonl --prompt_id {p} "
                     f"--tag {c} {extra} {x} --seeds {SEED} --latent_frames {LAT} --fresh --outdir {D})"
                     for c, x in arms.items()]
            T.append({"id": f"s39_{m}_{p}", "cmd": pre + f" && mkdir -p {D} && " + " && ".join(cmds),
                      "done": f"{D}/{p}_{m}_abs_s{SEED}.npz", "est_min": 4 * mins, "mem_gb": 30, "gpus": 1,
                      "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
    for p in PROMPTS:
        F.append({"id": f"s39f_{m}_{p}", "cmd": f"source {H}/jobs/env.sh && $PY $H/vbench_lite.py {D} '{p}_{m}_*s{SEED}.mp4' "
                  f"{D}/vbl_done_{p}_{m}.txt", "done": f"{D}/vbl_done_{p}_{m}.txt", "est_min": 5, "mem_gb": 30,
                  "gpus": 1, "gpu_mem_gb": 24, "cores": 16})
sfx = "" if len(sys.argv) <= 2 else "_" + "_".join(models)
for name, L in ((f"jobs/s39{sfx}.jsonl", T), (f"jobs/s39_feat{sfx}.jsonl", F)):
    with open(name, "w") as f:
        for t in L:
            f.write(json.dumps(t) + "\n")
    print(name, len(L), "tasks", round(sum(t["est_min"] for t in L) / 60, 1), "GPU-h (estimate)")
