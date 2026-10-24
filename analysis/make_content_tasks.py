"""Build the Argon task lists of the content-relaxation study (docs/preregistration_content_cn.md).

phase a: opening candidates (Self Forcing DMD, 5-s fresh videos of each class prompt, seeds 0..15) and fresh starts of
         the generic prompt (8 seeds) for every configuration                     -> jobs/content_a.jsonl
phase b: continuations of the 24 selected openings under both prompts; all arms of one opening x model family run in
         one task, guarded by arch_guard so paired arms share a GPU architecture  -> jobs/content_b.jsonl
phase c: CLIP embeddings of every video                                           -> jobs/content_c.jsonl
usage: python make_content_tasks.py a|b|c
"""
import json
import sys

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/content"
G = "/nfsscratch/Users/ywang906/vdrift/corr/content_gen"
M = f"{H}/models"
SF = "source /Users/ywang906/vdrift/jobs/env.sh && cd $H/Self-Forcing"
LL = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/LongLive/LongLive1.0 && cd {H}/LongLive/LongLive1.0"
RF = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/RollingForcing && cd {H}/RollingForcing"
CK = {"sid": f"--ckpt {M}/self_forcing/checkpoints/self_forcing_sid.pt",
      "gan": f"--ckpt {M}/self_forcing/checkpoints/self_forcing_gan.pt",
      "cv": f"--ckpt {M}/causvid/autoregressive_checkpoint/model.pt --steps 1000,757,522 --warp 0 --shift 8"}
# configuration -> (program, extra args, latent frames, environment)
CFG = {"sf60_s3": ("hold_gen_long.py", "--sink 3", 243, SF), "sf60_s0": ("hold_gen_long.py", "--sink 0", 243, SF),
       "ll_s3": ("ll_hold_gen.py", "", 126, LL), "ll_s0": ("ll_hold_gen.py", "--sink 0", 126, LL),
       "cv_s3": ("hold_gen.py", CK["cv"], 126, SF), "gan_s3": ("hold_gen.py", CK["gan"], 126, SF),
       "sid_s3": ("hold_gen.py", CK["sid"], 126, SF), "rf": ("rf_gen.py", "", 126, RF)}
FAM = {"sf": ["sf60_s3", "sf60_s0"], "ll": ["ll_s3", "ll_s0"], "sgc": ["cv_s3", "gan_s3", "sid_s3"], "rf": ["rf"]}
EST = {"sf": 8, "ll": 4, "sgc": 4, "rf": 4}    # minutes per video


def run(prompts, pid, cfg, seeds, outdir, fresh=False):
    prog, extra, lat, _ = CFG[cfg]
    return (f"$PY $H/{prog} --prompts {prompts} --prompt_id {pid} --tag {cfg} {extra} {'--fresh' if fresh else ''} "
            f"--seeds {seeds} --latent_frames {lat} --outdir {outdir}")


def task(tid, pre, cmds, done, est):
    return {"id": tid, "cmd": pre + " && " + " && ".join(cmds), "done": done, "est_min": est, "mem_gb": 30, "gpus": 1,
            "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16}


phase = sys.argv[1]
T = []
if phase == "a":
    P = "$H/prompts/content.jsonl"
    for r in map(json.loads, open("prompts/content.jsonl")):
        if r["set"] == "opening_gen":
            T.append(task(f"cg_{r['cls']}", SF, [f"$PY $H/hold_gen.py --prompts {P} --prompt_id {r['id']} --tag sfgen --fresh "
                                                 f"--seeds {','.join(map(str, range(16)))} --latent_frames 21 --outdir {G}"],
                          f"{G}/{r['id']}_sfgen_s15.npz", 20))
    seeds = ",".join(map(str, range(8)))
    for cfg in CFG:
        T.append(task(f"cf_{cfg}", CFG[cfg][3], [run(P, "fresh", cfg, seeds, D, fresh=True)], f"{D}/fresh_{cfg}_s7.npz",
                      8 * (8 if cfg.startswith("sf60") else 4)))
elif phase == "b":
    P = "$H/prompts/content_runs.jsonl"
    ops = sorted({r["id"].rsplit("_", 1)[0] for r in map(json.loads, open("prompts/content_runs.jsonl"))
                  if r["set"] == "content"})
    assert len(ops) == 24, ops
    for o in ops:
        for fam, cfgs in FAM.items():
            pre = [f"{o}_{arm}_{c}_" for c in cfgs for arm in ("mis", "mat")]
            cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(pre)]
            cmds += [run(P, f"{o}_{arm}", c, "0", D) for c in cfgs for arm in ("mis", "mat")]
            T.append(task(f"cs_{fam}_{o}", CFG[cfgs[0]][3], cmds, f"{D}/{o}_mat_{cfgs[-1]}_s0.npz", EST[fam] * 2 * len(cfgs)))
elif phase == "c":
    pre = f"source {H}/jobs/env.sh && cd $H"
    for cls in ("snow", "desert", "forest", "coast", "mountain", "farm", "fresh"):
        g = f"{D}/fresh_*.mp4" if cls == "fresh" else f"{D}/ct{cls}*.mp4"
        T.append(task(f"cc_{cls}", pre, [f"$PY $H/clip_probe.py '{g}' {D}/clip_text.npz {D}/clip_{cls}.done"],
                      f"{D}/clip_{cls}.done", 25))
with open(f"jobs/content_{phase}.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(phase, len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h")
