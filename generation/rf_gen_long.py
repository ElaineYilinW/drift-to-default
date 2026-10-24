"""60-s copy of rf_gen.py: identical except that the final VAE decode runs in segments with the temporal cache
carried over and the frames moved to CPU (decode_cpu, as in hold_gen_long.py).
Held-opening generation with Rolling Forcing (TencentARC/RollingForcing), for the relaxation study.

Rolling Forcing denoises a rolling window of 5 blocks at staggered noise levels and keeps the first block as an anchor
(re-aligned RoPE). Its own video-extension path does not offset the rolling window by the given frames, so this script
re-implements CausalInferencePipeline.inference_rolling_forcing with an opening: the held image (9 frames -> 3 latent
frames, optionally shifted in CIELAB as in hold_gen.py) is written into the clean KV cache as block 0 (the anchor), and
the rolling windows run over blocks 1..N-1 at their global positions. Without an image it is the original text-to-video
loop. Run from the RollingForcing repository root (PYTHONPATH=.). Outputs as hold_gen.py (mp4 / npz stats / json).
"""
import argparse
import json
import os
import time
from collections import OrderedDict

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--prompt_id", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--latent_frames", type=int, default=126)
p.add_argument("--seeds", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--ckpt", default="/Users/ywang906/vdrift/models/rolling_forcing/checkpoints/rolling_forcing_dmd.pt")
p.add_argument("--init_image", default="")
p.add_argument("--open_dL", type=float, default=0.0)
p.add_argument("--open_da", type=float, default=0.0)
p.add_argument("--open_db", type=float, default=0.0)
p.add_argument("--fresh", action="store_true")
p.add_argument("--no_video", action="store_true")
p.add_argument("--versions", default="", help="several openings in one run: name:dL:db,... (tag becomes TAG__name)")
a = p.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torchvision.io import write_video  # noqa: E402

from pipeline.rolling_forcing_inference import CausalInferencePipeline  # noqa: E402
from utils.misc import set_seed  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gen_stats", "/Users/ywang906/vdrift/stats_util.py")
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

cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/rolling_forcing_dmd.yaml"))
dev = torch.device("cuda")
torch.set_grad_enabled(False)
t0 = time.time()
pipe = CausalInferencePipeline(cfg, device=dev)
sd = torch.load(a.ckpt, map_location="cpu", mmap=True)
sd = sd["generator_ema"] if "generator_ema" in sd else sd["generator"]
pipe.generator.load_state_dict(OrderedDict((k.replace("_fsdp_wrapped_module.", ""), v) for k, v in sd.items()))
del sd
pipe = pipe.to(dtype=torch.bfloat16).to(dev)
load_s = time.time() - t0


def init_latent_from_image(path):
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
    if a.open_dL or a.open_da or a.open_db:
        lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.clip(lab[..., 0] + a.open_dL, 0, 100)
        lab[..., 1] += a.open_da
        lab[..., 2] += a.open_db
        im = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255.0).round().astype(np.uint8)
    t = torch.from_numpy(im).float().div(127.5).sub(1).permute(2, 0, 1)
    clip = t[:, None].repeat(1, 9, 1, 1)[None].to(dev, torch.bfloat16)
    return pipe.vae.encode_to_latent(clip).to(dev, torch.bfloat16)


def decode_cpu(lat, seg=12):
    """Decode latents in segments with the VAE's temporal cache carried across segments (equivalent to one full
    decode; conv2 is 1x1x1) and move each segment to CPU, so 60-s videos fit in GPU memory."""
    vae = pipe.vae
    zs = lat.permute(0, 2, 1, 3, 4)
    scale = [vae.mean.to(device=lat.device, dtype=lat.dtype), 1.0 / vae.std.to(device=lat.device, dtype=lat.dtype)]
    vae.model.clear_cache()
    outs = [vae.model.cached_decode(zs[:, :, s:s + seg], scale).float().clamp_(-1, 1).cpu() for s in range(0, zs.shape[2], seg)]
    vae.model.clear_cache()
    return torch.cat(outs, 2).permute(0, 2, 1, 3, 4)


def infer(noise):
    bsz, nf = noise.shape[:2]
    nb = pipe.num_frame_per_block
    fs = pipe.frame_seq_length
    cond = pipe.text_encoder(text_prompts=[prompt])
    pipe.kv_cache_clean = None
    pipe._initialize_kv_cache(batch_size=bsz, dtype=noise.dtype, device=noise.device)
    pipe._initialize_crossattn_cache(batch_size=bsz, dtype=noise.dtype, device=noise.device)
    output = torch.zeros_like(noise)
    off = 0
    if INIT is not None:                       # the opening = block 0 = Rolling Forcing's anchor
        output[:, :nb] = INIT
        pipe.generator(noisy_image_or_video=INIT, conditional_dict=cond,
                       timestep=torch.zeros([bsz, nb], device=dev, dtype=torch.int64),
                       kv_cache=pipe.kv_cache_clean, crossattn_cache=pipe.crossattn_cache, current_start=0,
                       updating_cache=True)
        off = 1
    num_blocks = nf // nb - off
    steps = pipe.denoising_step_list
    L = len(steps)
    shared = torch.ones([bsz, L * nb], device=dev, dtype=torch.float32)
    for i, t in enumerate(reversed(steps)):
        shared[:, i * nb:(i + 1) * nb] *= t
    noisy_cache = torch.zeros_like(noise)
    for w in range(num_blocks + L - 1):
        sb, eb = max(0, w - L + 1), min(num_blocks - 1, w)            # local block indices (inclusive)
        cs, ce = (sb + off) * nb, (eb + 1 + off) * nb                  # global frame indices
        n = ce - cs
        full = n == L * nb
        if full or sb == 0:
            x = torch.cat([noisy_cache[:, cs:ce - nb], noise[:, ce - nb:ce]], dim=1)
        else:
            x = noisy_cache[:, cs:ce]
        if full:
            ts = shared
        elif sb == 0:
            ts = shared[:, -n:]
        else:
            ts = shared[:, :n]
        _, pred = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts,
                                 kv_cache=pipe.kv_cache_clean, crossattn_cache=pipe.crossattn_cache,
                                 current_start=cs * fs)
        output[:, cs:ce] = pred
        for bi in range(sb, eb + 1):
            bt = ts[:, (bi - sb) * nb:(bi - sb + 1) * nb].mean().item()
            idx = torch.nonzero(torch.abs(steps - bt) < 1e-4, as_tuple=True)[0]
            if idx == L - 1:
                continue
            nt = steps[idx + 1].to(dev)
            g0 = (bi + off) * nb
            noisy_cache[:, g0:g0 + nb] = pipe.scheduler.add_noise(
                pred.flatten(0, 1), torch.randn_like(pred.flatten(0, 1)),
                nt * torch.ones([bsz * n], device=dev, dtype=torch.long)
            ).unflatten(0, pred.shape[:2])[:, (bi - sb) * nb:(bi - sb + 1) * nb]
        ctx = torch.ones_like(ts[:, :nb]) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=pred[:, :nb], conditional_dict=cond, timestep=ctx,
                       kv_cache=pipe.kv_cache_clean, crossattn_cache=pipe.crossattn_cache,
                       current_start=cs * fs, updating_cache=True)
    video = decode_cpu(output)
    return (video * 0.5 + 0.5).clamp(0, 1)


for vt, dL, da, db, seed in JOBS:
    a.tag, a.open_dL, a.open_da, a.open_db = vt, dL, da, db
    INIT = init_latent_from_image(a.init_image) if a.init_image else None
    set_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.randn([1, a.latent_frames, 16, 60, 104], device=dev, dtype=torch.bfloat16, generator=g)
    t1 = time.time()
    video = infer(noise)
    torch.cuda.synchronize()
    gen_s = time.time() - t1
    video = (255.0 * rearrange(video, "b t c h w -> b t h w c")[0].float().cpu()).round().clamp(0, 255).to(torch.uint8)
    out = stem(seed)
    if not a.no_video:
        tmp = f"{out}.{os.getpid()}.tmp.mp4"
        write_video(tmp, video, fps=16, options={"crf": "12"})
        os.replace(tmp, out + ".mp4")
    stats = gen_stats.frame_stats(video.numpy())
    info = dict(vars(a), seed=seed, prompt=None, model="rolling_forcing", gpu=torch.cuda.get_device_name(0),
                video_frames=int(video.shape[0]), load_s=round(load_s, 1), gen_s=round(gen_s, 1))
    with open(f"{out}.{os.getpid()}.tmp.json", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(f"{out}.{os.getpid()}.tmp.json", out + ".json")
    np.savez(f"{out}.{os.getpid()}.tmp.npz", **stats)
    os.replace(f"{out}.{os.getpid()}.tmp.npz", out + ".npz")
    print(json.dumps(info), flush=True)
