"""Build the Argon task lists of the relaxation study (docs/preregistration_relax_cn.md).

Every scene is pinned to one GPU architecture (even scenes A40, odd scenes L40S / RTX 6000 Ada), and all versions of a
scene x configuration run inside one task, so a preempted task that resumes cannot mix architectures.
usage: python make_relax_tasks.py CAUSVID_ARGS   (writes jobs/relax_{a40,l40s}.jsonl)
"""
import json
import sys

H = "/Users/ywang906/vdrift"
D = "/nfsscratch/Users/ywang906/vdrift/corr/relax"
M = f"{H}/models"
CV = sys.argv[1]
VERS = "m16:-16:0,m8:-8:0,z0:0:0,p8:8:0,p16:16:0,bm12:0:-12,bp12:0:12"
scenes = [json.loads(l)["id"] for l in open("prompts/relax.jsonl") if json.loads(l)["id"] != "fresh"]
SF = f"source {H}/jobs/env.sh && cd $H/Self-Forcing"
LL = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/LongLive/LongLive1.0 && cd {H}/LongLive/LongLive1.0"
RF = f"source {H}/jobs/env.sh && export PYTHONPATH={H}/RollingForcing && cd {H}/RollingForcing"
SK = f"source {H}/jobs/env.sh && cd $H"
P = "--prompts $H/prompts/relax.jsonl"
CK = {"sid": f"--ckpt {M}/self_forcing/checkpoints/self_forcing_sid.pt",
      "gan": f"--ckpt {M}/self_forcing/checkpoints/self_forcing_gan.pt",
      "sf10s": f"--ckpt {M}/self_forcing/checkpoints/self_forcing_10s.pt",
      "sidv2": f"--ckpt {M}/self_forcing/checkpoints/self_forcing_sid_v2.pt",
      "cv": CV}


def hg(scene, cfg, extra, seeds="0", vers=VERS, prog="hold_gen.py", lat=126):
    v = f"--versions {vers}" if vers else ""
    return f"$PY $H/{prog} {P} --prompt_id {scene} --tag {cfg} {extra} --seeds {seeds} --latent_frames {lat} {v} --outdir {D}"


def task(tid, pre, cmds, done, est):
    return {"id": tid, "cmd": pre + " && " + " && ".join(cmds), "done": done, "est_min": est, "mem_gb": 30, "gpus": 1,
            "gpu_mem_gb": 40, "gpu_bf16": True, "cores": 16}


def last(scene, cfg, ver="bp12", seed=0):
    return f"{D}/{scene}_{cfg}__{ver}_s{seed}.npz"


out = {"a40": [], "l40s": []}
for i, s in enumerate(scenes):
    arch = "a40" if i % 2 == 0 else "l40s"
    T = out[arch]
    T.append(task(f"rx_sf_{s}", SF, [hg(s, "sf_s3", ""), hg(s, "sf_s0", "--sink 0"),
                                     hg(s, "sf_ttc", "--method ttc --at_step 1 --beta 1.0")], last(s, "sf_ttc"), 70))
    T.append(task(f"rx_sidgan_{s}", SF, [hg(s, "sid_s3", CK["sid"]), hg(s, "gan_s3", CK["gan"])], last(s, "gan_s3"), 45))
    T.append(task(f"rx_sidv2ten_{s}", SF, [hg(s, "sidv2_s3", CK["sidv2"]), hg(s, "sf10s_s3", CK["sf10s"])], last(s, "sf10s_s3"), 45))
    T.append(task(f"rx_cv_{s}", SF, [hg(s, "cv_s3", CK["cv"])], last(s, "cv_s3"), 25))
    T.append(task(f"rx_ll_{s}", LL, [hg(s, "ll_s3", "", prog="ll_hold_gen.py"), hg(s, "ll_s0", "--sink 0", prog="ll_hold_gen.py")],
                  last(s, "ll_s0"), 45))
    T.append(task(f"rx_rf_{s}", RF, [hg(s, "rf", "", prog="rf_gen.py")], last(s, "rf"), 25))
# SkyReels (slow): 12 scenes (every other one), L* -16 / 0 / +16, 20 s
for i, s in enumerate(scenes[::2]):
    arch = "a40" if (2 * i) % 2 == 0 else "l40s"
    out["l40s" if i % 2 else "a40"].append(task(
        f"rx_sky_{s}", SK, [f"$H/envs/sky/bin/python $H/sky_gen.py {P} --prompt_id {s} --tag sky --seconds 20 --seeds 0 "
                            f"--versions m16:-16:0,z0:0:0,p16:16:0 --outdir {D}"], last(s, "sky", "p16"), 110))
# fresh starts: generic prompt, 4 seeds, no opening (one task per configuration family)
F = [("sf", SF, [hg("fresh", "sf_s3", "--fresh", "0,1,2,3", ""), hg("fresh", "sf_s0", "--fresh --sink 0", "0,1,2,3", "")]),
     ("sidgan", SF, [hg("fresh", "sid_s3", "--fresh " + CK["sid"], "0,1,2,3", ""), hg("fresh", "gan_s3", "--fresh " + CK["gan"], "0,1,2,3", "")]),
     ("sidv2ten", SF, [hg("fresh", "sidv2_s3", "--fresh " + CK["sidv2"], "0,1,2,3", ""), hg("fresh", "sf10s_s3", "--fresh " + CK["sf10s"], "0,1,2,3", ""),
                     hg("fresh", "cv_s3", "--fresh " + CK["cv"], "0,1,2,3", "")]),
     ("ll", LL, [hg("fresh", "ll_s3", "--fresh", "0,1,2,3", "", "ll_hold_gen.py"), hg("fresh", "ll_s0", "--fresh --sink 0", "0,1,2,3", "", "ll_hold_gen.py")]),
     ("rf", RF, [hg("fresh", "rf", "--fresh", "0,1,2,3", "", "rf_gen.py")])]
for k, (name, pre, cmds) in enumerate(F):
    tag = {"sf": "sf_s0", "sidgan": "gan_s3", "sidv2ten": "cv_s3", "ll": "ll_s0", "rf": "rf"}[name]
    out["a40" if k % 2 == 0 else "l40s"].append(task(f"rx_fresh_{name}", pre, cmds, f"{D}/fresh_{tag}__fresh_s3.npz", 70))
# long horizon (secondary): 6 scenes, L* -16 / +16, 120 s (480 latent frames)
LD = D + "_long"
for i, s in enumerate(scenes[:3] + scenes[12:15]):
    arch = "a40" if scenes.index(s) % 2 == 0 else "l40s"
    cmds = [hg(s, c, x, vers="m16:-16:0,p16:16:0", lat=480).replace(f"--outdir {D}", f"--outdir {LD}")
            for c, x in (("sf_s3", ""), ("sf_s0", "--sink 0"))]
    out[arch].append(task(f"rxl_sf_{s}", SF, cmds, f"{LD}/{s}_sf_s0__p16_s0.npz", 60))
    out[arch].append(task(f"rxl_ll_{s}", LL, [hg(s, "ll_s3", "", vers="m16:-16:0,p16:16:0", lat=480, prog="ll_hold_gen.py")
                                             .replace(f"--outdir {D}", f"--outdir {LD}")], f"{LD}/{s}_ll_s3__p16_s0.npz", 35))
for arch, T in out.items():
    only = "a40" if arch == "a40" else "l40s,rtx6000ada"
    with open(f"jobs/relax_{arch}.jsonl", "w") as f:
        for t in T:
            f.write(json.dumps(dict(t, gpu_only=only)) + "\n")
    print(arch, len(T), "tasks", sum(t["est_min"] for t in T) / 60, "GPU-h")
