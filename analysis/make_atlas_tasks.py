"""Build the Argon task list of the 60-s persistence atlas (docs/preregistration_atlas_cn.md).

24 scenes of the relaxation study, generic prompt, seed 0, 60 s (243 latent frames). Each task holds the arms that are
compared with each other and starts with arch_guard, so paired arms share a GPU architecture.
  sfA  Self Forcing DMD sink 3, sink 6, sink 3 re-rotated       (L* -16 / +16)
  sfB  Self Forcing DMD no sink, sink 1, sink 1 re-rotated      (L* -16 / +16)
  ll   LongLive sink 3, no sink                                  (L* -16 / 0 / +16)
  ck1  Self Forcing GAN, SiD                                     (L* -16 / 0 / +16)
  ck2  Self Forcing SiD v2, 10-s                                 (L* -16 / 0 / +16)
  rf   Rolling Forcing                                           (L* -16 / 0 / +16)
usage: python make_atlas_tasks.py   (writes jobs/atlas.jsonl)
"""
import json

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/atlas"
M = f"{H}/models/self_forcing/checkpoints"
SF = f"source {H}/jobs/env.sh && cd $H/Self-Forcing"
LL = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/LongLive/LongLive1.0 && cd {H}/LongLive/LongLive1.0"
RF = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/RollingForcing && cd {H}/RollingForcing"
V2, V3 = "m16:-16:0,p16:16:0", "m16:-16:0,z0:0:0,p16:16:0"
P = "--prompts $H/prompts/relax.jsonl"
# configuration -> (program, extra args, versions)
CFG = {"sf60_s3": ("hold_gen_long_rr.py", "--sink 3", V2), "sf60_s6": ("hold_gen_long_rr.py", "--sink 6", V2),
       "sf60_s3rr": ("hold_gen_long_rr.py", "--sink 3 --rerotate", V2),
       "sf60_s0": ("hold_gen_long_rr.py", "--sink 0", V2), "sf60_s1": ("hold_gen_long_rr.py", "--sink 1", V2),
       "sf60_s1rr": ("hold_gen_long_rr.py", "--sink 1 --rerotate", V2),
       "ll60_s3": ("ll_hold_gen_long.py", "--sink 3", V3), "ll60_s0": ("ll_hold_gen_long.py", "--sink 0", V3),
       "gan60_s3": ("hold_gen_long_rr.py", f"--sink 3 --ckpt {M}/self_forcing_gan.pt", V3),
       "sid60_s3": ("hold_gen_long_rr.py", f"--sink 3 --ckpt {M}/self_forcing_sid.pt", V3),
       "sidv260_s3": ("hold_gen_long_rr.py", f"--sink 3 --ckpt {M}/self_forcing_sid_v2.pt", V3),
       "sf10s60_s3": ("hold_gen_long_rr.py", f"--sink 3 --ckpt {M}/self_forcing_10s.pt", V3),
       "rf60": ("rf_gen_long.py", "", V3)}
FAM = {"sfA": (SF, ["sf60_s3", "sf60_s6", "sf60_s3rr"], 8), "sfB": (SF, ["sf60_s0", "sf60_s1", "sf60_s1rr"], 8),
       "ll": (LL, ["ll60_s3", "ll60_s0"], 7), "ck1": (SF, ["gan60_s3", "sid60_s3"], 8),
       "ck2": (SF, ["sidv260_s3", "sf10s60_s3"], 8), "rf": (RF, ["rf60"], 8)}
scenes = [json.loads(l)["id"] for l in open("prompts/relax.jsonl") if json.loads(l)["id"] != "fresh"]
assert len(scenes) == 24
T = []
for s in scenes:
    for fam, (pre, cfgs, mins) in FAM.items():
        cmds = [f"$PY $H/arch_guard.py {D} " + " ".join(f"{s}_{c}__" for c in cfgs)]
        n = 0
        for c in cfgs:
            prog, extra, vers = CFG[c]
            cmds.append(f"$PY $H/{prog} {P} --prompt_id {s} --tag {c} {extra} --seeds 0 --latent_frames 243 "
                        f"--versions {vers} --outdir {D}")
            n += len(vers.split(","))
        last = cfgs[-1]
        T.append({"id": f"at_{fam}_{s}", "cmd": pre + " && " + " && ".join(cmds),
                  "done": f"{D}/{s}_{last}__p16_s0.npz", "est_min": mins * n, "mem_gb": 30, "gpus": 1,
                  "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16})
with open("jobs/atlas.jsonl", "w") as f:
    for t in T:
        f.write(json.dumps(t) + "\n")
print(len(T), "tasks", round(sum(t["est_min"] for t in T) / 60, 1), "GPU-h")
