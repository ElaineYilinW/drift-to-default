"""Held-opening generation with SkyReels-V2 Diffusion Forcing 1.3B (diffusers), for the relaxation study.

SkyReels-V2-DF extends a video autoregressively in overlapping segments (diffusion forcing: each segment is denoised
conditioned on the last `overlap_history` frames of the previous one). The opening image (optionally shifted in CIELAB
as in hold_gen.py) is held for 17 frames (0.7 s at 24 fps) and passed as the prefix video; the pipeline extends it.
Without an image it is plain text-to-video with the same long-video settings. Settings follow the model card for long
videos: 544x960, 24 fps, base_num_frames 97, overlap_history 17, addnoise_condition 20, 30 steps, guidance 6.
Outputs as hold_gen.py (mp4 / npz stats / json, json has fps=24).
"""
import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--prompt_id", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--seconds", type=float, default=31.0, help="length of the generated part")
p.add_argument("--seeds", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--model", default="/Users/ywang906/vdrift/models/skyreels_df_1p3b")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--guidance", type=float, default=6.0)
p.add_argument("--init_image", default="")
p.add_argument("--open_dL", type=float, default=0.0)
p.add_argument("--open_da", type=float, default=0.0)
p.add_argument("--open_db", type=float, default=0.0)
p.add_argument("--fresh", action="store_true")
p.add_argument("--no_video", action="store_true")
p.add_argument("--versions", default="", help="several openings in one run: name:dL:db,... (tag becomes TAG__name)")
a = p.parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.io import write_video  # noqa: E402
from diffusers import (AutoencoderKLWan, SkyReelsV2DiffusionForcingPipeline,  # noqa: E402
                       SkyReelsV2DiffusionForcingVideoToVideoPipeline, UniPCMultistepScheduler)

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gen_stats", "/Users/ywang906/vdrift/stats_util.py")
gen_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_stats)

FPS, Hh, Ww, HOLD = 24, 544, 960, 17
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

dev = torch.device("cuda")
t0 = time.time()
vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32)
Pipe = SkyReelsV2DiffusionForcingVideoToVideoPipeline if a.init_image else SkyReelsV2DiffusionForcingPipeline
pipe = Pipe.from_pretrained(a.model, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe = pipe.to(dev)
load_s = time.time() - t0


def opening_frames(path):
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    tw = round(h * Ww / Hh)
    if tw <= w:
        x0 = (w - tw) // 2
        im = im[:, x0:x0 + tw]
    else:
        th = round(w * Hh / Ww)
        y0 = (h - th) // 2
        im = im[y0:y0 + th]
    im = cv2.resize(im, (Ww, Hh), interpolation=cv2.INTER_AREA)
    if a.open_dL or a.open_da or a.open_db:
        lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.clip(lab[..., 0] + a.open_dL, 0, 100)
        lab[..., 1] += a.open_da
        lab[..., 2] += a.open_db
        im = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255.0).round().astype(np.uint8)
    return [Image.fromarray(im)] * HOLD


n_new = int(round(a.seconds * FPS / 4)) * 4 + 1            # 4k+1 frames, as the Wan VAE requires
for vt, dL, da, db, seed in JOBS:
    a.tag, a.open_dL, a.open_da, a.open_db = vt, dL, da, db
    VIDEO = opening_frames(a.init_image) if a.init_image else None
    g = torch.Generator(device=dev).manual_seed(seed)
    t1 = time.time()
    kw = dict(prompt=prompt, height=Hh, width=Ww, num_inference_steps=a.steps, guidance_scale=a.guidance,
              generator=g, base_num_frames=97, overlap_history=17, addnoise_condition=20, ar_step=0, output_type="np")
    if VIDEO is not None:
        frames = pipe(video=VIDEO, num_frames=n_new, **kw).frames[0]
    else:
        frames = pipe(num_frames=n_new, **kw).frames[0]
    torch.cuda.synchronize()
    gen_s = time.time() - t1
    video = torch.from_numpy((np.clip(frames, 0, 1) * 255).round().astype(np.uint8))      # (T, H, W, 3)
    if VIDEO is not None and video.shape[0] < n_new + HOLD:                               # prepend the prefix if omitted
        pre = torch.from_numpy(np.stack([np.asarray(f) for f in VIDEO]))
        video = torch.cat([pre, video])
    out = stem(seed)
    if not a.no_video:
        tmp = f"{out}.{os.getpid()}.tmp.mp4"
        write_video(tmp, video, fps=FPS, options={"crf": "12"})
        os.replace(tmp, out + ".mp4")
    stats = gen_stats.frame_stats(video.numpy())
    info = dict(vars(a), seed=seed, prompt=None, model="skyreels_v2_df_1.3b", fps=FPS, hold_frames=HOLD if VIDEO else 0,
                gpu=torch.cuda.get_device_name(0), video_frames=int(video.shape[0]), raw_frames=int(frames.shape[0]),
                load_s=round(load_s, 1), gen_s=round(gen_s, 1))
    with open(f"{out}.{os.getpid()}.tmp.json", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(f"{out}.{os.getpid()}.tmp.json", out + ".json")
    np.savez(f"{out}.{os.getpid()}.tmp.npz", **stats)
    os.replace(f"{out}.{os.getpid()}.tmp.npz", out + ".npz")
    print(json.dumps(info), flush=True)
