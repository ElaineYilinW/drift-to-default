"""Copy of hold_gen_long_rr.py with --sink_mode (off:N / dist:D, the sink's temporal RoPE position; see
jobs/patch_sf_sinkpos.py) and --probe (attention-to-sink probe over a grid of sink distances). Must run against the
Self-Forcing-b tree (PYTHONPATH and cwd); without --sink_mode/--probe it generates exactly as hold_gen_long_rr.py. The npz
also holds two extra per-frame statistics: motion (mean absolute grey change from the previous frame, 0-100) and
open_diff (mean absolute grey difference from frame 8, the last opening frame).
Original description of hold_gen_long_rr.py follows.
Copy of hold_gen_long.py with --rerotate (sink keys re-rotated to sit just before the window, as in gen.py and
Rolling Forcing); without --rerotate it is identical to hold_gen_long.py.
Long-video copy of hold_gen.py: identical except that the final VAE decode runs in segments with the
temporal cache carried over and the frames moved to CPU (decode_cpu), so that 120-s videos fit in GPU memory.
Held-opening generation for any Self-Forcing-architecture checkpoint (relaxation study; copy of correct_gen2.py).

Adds to correct_gen2.py: --ckpt / --ckpt_key / --steps / --warp / --shift to run other checkpoints of the same
architecture (Self Forcing DMD/SiD/GAN/10s, the ODE-initialised student, CausVid), and a controlled perturbation of the
held opening in CIELAB (--open_dL, --open_da, --open_db: shifts of L*, a*, b* applied to the image before VAE encoding),
so that the opening's brightness/colour can be manipulated within a scene. Without an image (no --init_image and no
'image' key) it is a plain text-to-video fresh start.
Original description of correct_gen2.py follows.

v2 (copy of the frozen correct_gen.py): adds --init_image, a real photo held for the first chunk (9 frames),
encoded with the Wan VAE and used as the first 3 latent frames (= the anchor/sink) of a video continuation.

Re-implements CausalInferencePipeline.inference (text-to-video path) step for step, so that a correction can be
applied to a chunk's predicted clean latent (x0) after denoising step `--at_step` (0..3; 3 = the final output,
which is also what is written into the KV cache by the clean context pass). With `--method none` it must reproduce
gen.py exactly (checked by --check_against).

Correction = a shift of the latent channel means (16-d offset) added to each chunk's predicted x0 at --at_step,
set by closed-loop control on the CLEAN outputs of earlier chunks (see Controller): brightness Lhat = w.m + c is
predicted from channel means (observational fit, for monitoring); the shift per +1 L is u (interventional estimate
from jacobian.py, for acting). The first chunk is the anchor/target and is never corrected.
Outputs as gen.py (mp4 / npz stats / json) plus optional per-chunk latents and a trigger log.
"""
import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--prompt_id", required=True)
p.add_argument("--tag", required=True, help="output name: {prompt}_{tag}_s{seed}")
p.add_argument("--window", type=int, default=21)
p.add_argument("--sink", type=int, default=3)
p.add_argument("--latent_frames", type=int, default=126)
p.add_argument("--seeds", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--method", default="none", choices=["none", "luma", "ttc"])
p.add_argument("--beta", type=float, default=1.0, help="ttc: share of the anchor-context DC difference transferred")
p.add_argument("--at_step", type=str, default="3", help="denoising step(s) to correct at, e.g. 2 or 2,3")
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--trigger", default="always", help="always | every:N | cusum:k:h")
p.add_argument("--luma_dir", default="", help="npz with w (16,), c, sd (residual sd of Lhat per frame)")
p.add_argument("--jump", type=float, default=0.0,
               help="guard: a chunk-to-chunk change in Lhat larger than this (L units) is taken as a legitimate scene "
                    "change and moves the target with it (0 = off)")
p.add_argument("--sd", type=float, default=2.1, help="chunk-to-chunk sd of L (CUSUM scale)")
p.add_argument("--step_cap", type=float, default=2.0)
p.add_argument("--cap", type=float, default=10.0)
p.add_argument("--init_image", default="", help="image file (or images jsonl key 'image') held as the opening chunk")
p.add_argument("--save_latents", action="store_true")
p.add_argument("--no_video", action="store_true")
p.add_argument("--check_against", default="", help="dir with gen.py npz of the same prompt/strategy/seed")
p.add_argument("--check_strategy", default="w21s3")
p.add_argument("--ckpt", default="checkpoints/self_forcing_dmd.pt")
p.add_argument("--ckpt_key", default="auto", help="state-dict key: auto = generator_ema if present, else generator")
p.add_argument("--steps", default="", help="denoising step list, e.g. 1000,757,522 (default: config)")
p.add_argument("--warp", type=int, default=-1, help="warp_denoising_step 1/0 (-1: config)")
p.add_argument("--shift", type=float, default=0.0, help="timestep_shift (0: config)")
p.add_argument("--open_dL", type=float, default=0.0)
p.add_argument("--open_da", type=float, default=0.0)
p.add_argument("--open_db", type=float, default=0.0)
p.add_argument("--fresh", action="store_true", help="ignore any image: plain text-to-video")
p.add_argument("--rerotate", action="store_true", help="re-rotate sink keys to sit just before the window")
p.add_argument("--sink_mode", default="", help="off:N (sink N latent frames further back) | dist:D (held at distance D)")
p.add_argument("--probe", default="", help="json: {out, chunks, grid, nq} (attention-to-sink probe)")
p.add_argument("--versions", default="", help="several openings in one run: name:dL:db,... (tag becomes TAG__name)")
a = p.parse_args()
a.at_steps = sorted({int(v) for v in a.at_step.split(",")})  # list: also goes into the json log
os.environ["SF_SINK_REROTATE"] = "1" if a.rerotate else "0"  # read by the patched causal_model.py at import
assert not (a.rerotate and a.sink_mode), "--rerotate and --sink_mode are exclusive"
os.environ["SF_SINK_MODE"] = a.sink_mode
os.environ["SF_SINK_PROBE"] = a.probe

import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torchvision.io import write_video  # noqa: E402

from pipeline import CausalInferencePipeline  # noqa: E402
from utils.misc import set_seed  # noqa: E402
import wan.modules.causal_model as cm  # noqa: E402
assert cm.SINK_REROTATE == a.rerotate, "Self-Forcing not patched (jobs/patch_sf.py)"
assert getattr(cm, "SINK_MODE", None) == a.sink_mode, "Self-Forcing-b not patched (jobs/patch_sf_sinkpos.py)"
assert "Self-Forcing-b" in cm.__file__, cm.__file__

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gen_stats", os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_util.py"))
gen_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_stats)

_row = next(r for r in map(json.loads, open(a.prompts)) if r["id"] == a.prompt_id)
prompt = _row["prompt"]
if not a.init_image and _row.get("image") and not a.fresh:
    a.init_image = _row["image"]
if a.fresh:
    a.init_image = ""
seeds = [int(s) for s in a.seeds.split(",")]
os.makedirs(a.outdir, exist_ok=True)
stem = lambda s: os.path.join(a.outdir, f"{a.prompt_id}_{a.tag}_s{s}")
VERS = ([(a.tag, a.open_dL, a.open_da, a.open_db)] if not a.versions else
        [(f"{a.tag}__{v.split(':')[0]}", float(v.split(':')[1]), 0.0, float(v.split(':')[2])) for v in a.versions.split(",")])
JOBS = [(t, dL, da, db, s) for t, dL, da, db in VERS for s in seeds
        if not os.path.exists(os.path.join(a.outdir, f"{a.prompt_id}_{t}_s{s}.npz"))]
if not JOBS:
    print("all done")
    raise SystemExit

cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
cfg.model_kwargs.local_attn_size = a.window
cfg.model_kwargs.sink_size = a.sink
if a.steps:
    cfg.denoising_step_list = [int(v) for v in a.steps.split(",")]
if a.warp >= 0:
    cfg.warp_denoising_step = bool(a.warp)
if a.shift > 0:
    cfg.timestep_shift = a.shift
    cfg.model_kwargs.timestep_shift = a.shift
dev = torch.device("cuda")
torch.set_grad_enabled(False)
t0 = time.time()
pipe = CausalInferencePipeline(cfg, device=dev)
sd = torch.load(a.ckpt, map_location="cpu", mmap=True)
key = a.ckpt_key if a.ckpt_key != "auto" else ("generator_ema" if "generator_ema" in sd else "generator")
pipe.generator.load_state_dict(sd[key])
a.ckpt_key_used = key
del sd
pipe = pipe.to(dtype=torch.bfloat16).to(dev)
load_s = time.time() - t0

W = C0 = SD = U = None
if a.luma_dir:
    d = np.load(a.luma_dir)
    W = torch.tensor(d["w"], dtype=torch.float32, device=dev)
    C0, SD = float(d["c"]), float(d["sd"])
    # w/c/sd = observational predictor Lhat = w.m + c (used to monitor); u (optional) = interventional shift per +1 L
    # estimated from designed per-channel perturbations (jacobian.py). Without u, shift along w (prediction direction).
    U = torch.tensor(d["u"], dtype=torch.float32, device=dev) if "u" in d.files else W / (W @ W)


def chan_means(x):  # x: (F, 16, H, W) -> (F, 16)
    return x.float().mean(dim=(2, 3))


def measure_L(x0):
    """Mean CIELAB L* of a chunk, measured by decoding its clean latents (sRGB, D65), as frame_stats does."""
    v = pipe.vae.decode_to_pixel(x0, use_cache=False)[0].float() * 0.5 + 0.5          # (T, 3, H, W) in [0, 1]
    v = v.clamp(0, 1)
    lin = torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    Y = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    f = torch.where(Y > 0.008856, Y ** (1 / 3), 7.787 * Y + 16 / 116)
    return float((116 * f - 16).mean())


class Controller:
    """Closed-loop brightness control on clean chunk outputs, acting along the interventional direction u.

    After each chunk its true mean L* is measured (decode). A jump |L_k - L_{k-1}| > J is taken as a legitimate
    scene change: the target moves with it and the CUSUM resets. Error e_k = target - L_k. The correction offset
    (in L units, applied as offL * u to the next chunk's x0 at --at_step) is updated by the trigger rule:
      always   : integral control, offL += alpha * e_k
      every:N  : every N chunks, offL += e_k
      cusum:k:h: one-sided (downward) CUSUM on e_k / --sd; on an alarm offL += e_k and the CUSUM resets
      never    : offL stays 0
    Anti-windup: each update is clipped to +/- --step_cap and the total offset to +/- --cap (L units).
    """
    def __init__(self, spec, L_anchor):
        self.kind, *args = spec.split(":")
        self.args = [float(v) for v in args]
        self.target = self.prev = L_anchor
        self.offL, self.s, self.n = 0.0, 0.0, 0

    def update(self, x0, log):
        lm = measure_L(x0)
        rec = {}
        if a.jump > 0 and abs(lm - self.prev) > a.jump:
            self.target = min(100.0, max(0.0, self.target + lm - self.prev))
            self.s = 0.0
            rec["jump"] = round(lm - self.prev, 2)
        e = self.target - lm
        self.n += 1
        step = 0.0
        if a.method != "none":
            if self.kind == "always":
                step = a.alpha * e
            elif self.kind == "every" and self.n % int(self.args[0]) == 0:
                step = e
            elif self.kind == "cusum":
                kk, h = self.args
                self.s = max(0.0, self.s + e / a.sd - kk)
                if self.s > h:
                    step, self.s = e, 0.0
        step = max(-a.step_cap, min(a.step_cap, step))
        self.offL = max(-a.cap, min(a.cap, self.offL + step))
        rec.update(L=round(lm, 2), target=round(self.target, 2), err=round(e, 2), fired=step != 0.0,
                   offL=round(self.offL, 2))
        log.append(rec)
        self.prev = lm


def sink_only_x0(x, cond, ts, start):
    """Predicted x0 for the same noisy chunk when the model sees ONLY the sink (clean anchor) frames as context.

    A temporary per-block KV cache holds the sink keys/values copied from the live cache plus room for this chunk;
    the live cache is not touched and no random numbers are drawn, so the run stays paired with the others."""
    fs = pipe.frame_seq_length
    st = a.sink * fs
    tmp = []
    for c in pipe.kv_cache1:
        k = torch.zeros([c["k"].shape[0], st + x.shape[1] * fs, *c["k"].shape[2:]], dtype=c["k"].dtype, device=c["k"].device)
        v = torch.zeros_like(k)
        k[:, :st], v[:, :st] = c["k"][:, :st], c["v"][:, :st]
        tmp.append({"k": k, "v": v,
                    "global_end_index": torch.tensor([start * fs], dtype=torch.long, device=k.device),
                    "local_end_index": torch.tensor([st], dtype=torch.long, device=k.device)})
    _, x0c = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts, kv_cache=tmp,
                            crossattn_cache=pipe.crossattn_cache, current_start=start * fs)
    return x0c


def init_latent_from_image(path):
    """A real photo -> 9-frame still clip (480x832, centre crop) -> 3 Wan VAE latent frames, bf16 on the GPU."""
    import cv2
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    tw = round(h * 832 / 480)
    if tw <= w:
        x0 = (w - tw) // 2
        im = im[:, x0:x0 + tw]
    else:
        th = round(w * 480 / 832)
        y0 = (h - th) // 2
        im = im[y0:y0 + th]
    im = cv2.resize(im, (832, 480), interpolation=cv2.INTER_AREA)
    if a.open_dL or a.open_da or a.open_db:       # controlled shift of the opening in CIELAB (float, D65)
        lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.clip(lab[..., 0] + a.open_dL, 0, 100)
        lab[..., 1] += a.open_da
        lab[..., 2] += a.open_db
        im = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255.0).round().astype(np.uint8)
    t = torch.from_numpy(im).float().div(127.5).sub(1).permute(2, 0, 1)            # (3, H, W) in [-1, 1]
    clip = t[:, None].repeat(1, 9, 1, 1)[None].to(dev, torch.bfloat16)              # (1, 3, 9, H, W)
    return pipe.vae.encode_to_latent(clip).to(dev, torch.bfloat16)                 # (1, 3, 16, 60, 104)


def decode_cpu(lat, seg=12):
    """Decode latents in segments with the VAE's temporal cache carried across segments (equivalent to one full
    decode; conv2 is 1x1x1) and move each segment to CPU, so 120-s videos fit in GPU memory."""
    vae = pipe.vae
    zs = lat.permute(0, 2, 1, 3, 4)
    scale = [vae.mean.to(device=lat.device, dtype=lat.dtype), 1.0 / vae.std.to(device=lat.device, dtype=lat.dtype)]
    vae.model.clear_cache()
    outs = [vae.model.cached_decode(zs[:, :, s:s + seg], scale).float().clamp_(-1, 1).cpu() for s in range(0, zs.shape[2], seg)]
    vae.model.clear_cache()
    return torch.cat(outs, 2).permute(0, 2, 1, 3, 4)


def infer(noise, spec):
    """Mirror of CausalInferencePipeline.inference for T2V (no initial latent) with the correction hook."""
    bsz, nf = noise.shape[:2]
    nb = pipe.num_frame_per_block
    cond = pipe.text_encoder(text_prompts=[prompt])
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(batch_size=bsz, dtype=noise.dtype, device=noise.device)
    pipe._initialize_crossattn_cache(batch_size=bsz, dtype=noise.dtype, device=noise.device)
    out = torch.zeros_like(noise)
    start, ctl, log, lat = 0, None, [], []
    fs = pipe.frame_seq_length
    for blk in range(nf // nb):
        x = noise[:, start:start + nb]
        steps = list(enumerate(pipe.denoising_step_list))
        if blk == 0 and INIT is not None:   # real opening chunk: no denoising, it is given
            steps, x0 = [], INIT
            ts = torch.zeros([bsz, nb], device=noise.device, dtype=torch.int64)
        for i, t in steps:
            ts = torch.ones([bsz, nb], device=noise.device, dtype=torch.int64) * t
            _, x0 = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts, kv_cache=pipe.kv_cache1,
                                   crossattn_cache=pipe.crossattn_cache, current_start=start * fs)
            if i in a.at_steps and ctl is not None and a.method == "luma" and ctl.offL != 0.0:
                x0 = (x0.float() + ctl.offL * U[None, None, :, None, None]).to(x0.dtype)
            if i in a.at_steps and blk > 0 and a.method == "ttc":
                x0c = sink_only_x0(x, cond, ts, start)
                dm = chan_means(x0c[0]) - chan_means(x0[0])                    # (F, 16): DC difference per frame
                x0 = (x0.float() + a.beta * dm[None, :, :, None, None]).to(x0.dtype)
                log.append(dict(blk=blk, dc_norm=round(float(dm.norm(dim=1).mean()), 4),
                                dL_pred=round(float((dm @ W).mean()), 3) if W is not None else None))
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                x = pipe.scheduler.add_noise(x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                                             nt * torch.ones([bsz * nb], device=noise.device, dtype=torch.long)
                                             ).unflatten(0, x0.shape[:2])
        if a.method == "ttc":
            pass
        elif ctl is None:  # first chunk = anchor (never corrected); latent frame 0 is a lone frame, so left out
            ctl = Controller(spec, measure_L(x0[:, 1:]))
        else:
            ctl.update(x0, log)
        out[:, start:start + nb] = x0
        if a.save_latents:
            lat.append(x0[0].float().cpu().numpy().astype(np.float16))
        ctx_t = torch.ones_like(ts) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=x0, conditional_dict=cond, timestep=ctx_t, kv_cache=pipe.kv_cache1,
                       crossattn_cache=pipe.crossattn_cache, current_start=start * fs)
        start += nb
    video = decode_cpu(out)
    return (video * 0.5 + 0.5).clamp(0, 1), log, lat


for vt, dL, da, db, seed in JOBS:
    a.tag, a.open_dL, a.open_da, a.open_db = vt, dL, da, db
    INIT = init_latent_from_image(a.init_image) if a.init_image else None
    set_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.randn([1, a.latent_frames, 16, 60, 104], device=dev, dtype=torch.bfloat16, generator=g)
    t1 = time.time()
    video, log, lat = infer(noise, a.trigger)
    torch.cuda.synchronize()
    gen_s = time.time() - t1
    video = (255.0 * rearrange(video, "b t c h w -> b t h w c")[0].float().cpu()).round().clamp(0, 255).to(torch.uint8)
    out = stem(seed)
    if not a.no_video:
        tmp = f"{out}.{os.getpid()}.tmp.mp4"
        write_video(tmp, video, fps=16, options={"crf": "12"})
        os.replace(tmp, out + ".mp4")
    stats = gen_stats.frame_stats(video.numpy())
    g = video[:, ::4, ::4].float().mean(-1)  # grey, 4x subsampled: frame-to-frame motion and distance to frame 8
    stats["motion"] = np.concatenate([[0.0], (g[1:] - g[:-1]).abs().mean((1, 2)).numpy() * 100 / 255])
    stats["open_diff"] = (g - g[8]).abs().mean((1, 2)).numpy() * 100 / 255
    if a.save_latents:
        np.save(f"{out}.latents.npy", np.concatenate(lat))
    info = dict(vars(a), seed=seed, prompt=None, gpu=torch.cuda.get_device_name(0), video_frames=int(video.shape[0]), load_s=round(load_s, 1),
                gen_s=round(gen_s, 1), n_fired=sum(bool(r.get("fired", True)) for r in log), n_chunks=len(log), log=log)
    if a.check_against:
        ref = np.load(os.path.join(a.check_against, f"{a.prompt_id}_{a.check_strategy}_s{seed}.npz"))
        info["check_max_abs_lab_diff"] = float(np.abs(ref["lab_mean"] - stats["lab_mean"]).max())
    with open(f"{out}.{os.getpid()}.tmp.json", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(f"{out}.{os.getpid()}.tmp.json", out + ".json")
    np.savez(f"{out}.{os.getpid()}.tmp.npz", **stats)
    os.replace(f"{out}.{os.getpid()}.tmp.npz", out + ".npz")
    print(json.dumps({k: v for k, v in info.items() if k != "log"}), flush=True)
