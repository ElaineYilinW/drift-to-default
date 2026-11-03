"""Build the Argon task list of study 38, the held sink against re-alignment on new scenes and four models
(docs/preregistrations/38_held_sink_test.md).

16 openings never used in a sink study (prompts/heldtest.jsonl: nr01, nr04, nr06, nr09, nr11, nr14, nr16, nr18, ng00-ng07),
generic prompt, L* -16 / +16, seed 5, 120 s (483 latent frames). Models and generators:
  sf   Self Forcing DMD    hold_gen_sinkpos.py (Self-Forcing-b), window 21, sink 3
  cv   CausVid             same generator, --ckpt causvid --steps 1000,757,522 --warp 0 --shift 8
  gan  Self Forcing GAN    same generator, --ckpt self_forcing_gan.pt
  ll   LongLive            ll_hold_gen_sinkpos.py (LongLive-c), window 12, sink 3
Arms per model (tags {m}_held, {m}_rr, {m}_abs): the sink held at the model's D* from study 37 (--sink_mode dist:D*),
held at the re-aligned distance window - 1 (dist:20, or dist:11 for LongLive), and at its original positions (no flag).
A model without a D* is left out. One task per model and opening runs the three arms and both versions (arch_guard over
the opening's outputs; an arm whose p16 output exists is skipped). Videos only for nr01 and ng00. GPUs >= 40 GB, bf16.
                                                                                     -> jobs/s38.jsonl
usage: python make_s38_tasks.py DSTAR.json [MODELS, default sf,ll,cv,gan]
"""
import json
import sys

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/s38_test"
SFB = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/Self-Forcing-b && cd {H}/Self-Forcing-b"
LLC = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/LongLive-c && cd {H}/LongLive-c"
MODEL = {"sf": (SFB, "hold_gen_sinkpos.py", "", 40),
         "cv": (SFB, "hold_gen_sinkpos.py",
                "--ckpt $H/models/causvid/autoregressive_checkpoint/model.pt --steps 1000,757,522 --warp 0 --shift 8", 30),
         "gan": (SFB, "hold_gen_sinkpos.py", "--ckpt $H/models/self_forcing/checkpoints/self_forcing_gan.pt", 40),
         "ll": (LLC, "ll_hold_gen_sinkpos.py", "", 40)}
sel = json.load(open(sys.argv[1]))
models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sf", "ll", "cv", "gan"]
SCENES = [json.loads(l)["id"] for l in open("prompts/heldtest.jsonl")]
assert len(SCENES) == 16
SEED, LAT = 5, 483
T = []
for m in models:
    if sel.get(m, {}).get("dstar") is None:
        print("no D* for", m, "- left out")
        continue
    pre, gen, extra, mins = MODEL[m]
    arms = {f"{m}_held": f"--sink_mode dist:{sel[m]['dstar']}", f"{m}_rr": f"--sink_mode dist:{sel[m]['realign']}",
            f"{m}_abs": ""}
    for s in SCENES:
        nv = "" if s in ("nr01", "ng00") else " --no_video"
        cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{s}_{c}__" for c in arms)]
        cmds += [f"([ -f {D}/{s}_{c}__p16_s{SEED}.npz ] || $PY $H/{gen} --prompts $H/prompts/heldtest.jsonl "
                 f"--prompt_id {s} --tag {c} --sink 3 {extra} {x} --seeds {SEED} --latent_frames {LAT} "
                 f"--versions m16:-16:0,p16:16:0 --outdir {D}{nv})" for c, x in arms.items()]
        T.append({"id": f"s38_{m}_{s}", "cmd": pre + f" && mkdir -p {D} && " + " && ".join(cmds),
                  "done": f"{D}/{s}_{m}_abs__p16_s{SEED}.npz", "est_min": 3 * mins, "mem_gb": 30, "gpus": 1,
                  "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
name = "jobs/s38.jsonl" if len(sys.argv) <= 2 else f"jobs/s38_{'_'.join(models)}.jsonl"
with open(name, "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(name, len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h (estimate)")
