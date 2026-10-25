"""Held-opening rollouts of an action-conditioned world model, Matrix-Game-2.0 (universal distilled model), for the
relaxation study.

Matrix-Game-2.0 is a Wan-architecture image-to-world model without a text branch, distilled for few-step
autoregressive generation (3 latent frames per chunk, 25 fps, 352x640). The opening image reaches the model through two
routes: (i) its VAE latent, concatenated (with a mask) to the first latent frame only, after which it survives only in
the key-value cache, whose local window is 6 latent frames and which has no sink; (ii) its CLIP image embedding, a global
condition of every chunk. The action is held constant: forward, no camera motion.
--channel separates the routes as in fp_gen.py: "both" shifts the image for both routes, "ctx" only the concatenated
latent (CLIP embedding of the unshifted image), "emb" only the CLIP embedding. The opening may be shifted in CIELAB as in
hold_gen.py. 8 copies of the opening image are prepended so that frames 0..8 are the opening (json hold_frames = 9);
the model's own first frame (its reconstruction of the image) follows. Outputs as hold_gen.py (json has fps = 25).
Run from the Matrix-Game-2 directory with vdrift/shims on PYTHONPATH (flash_attn stand-in).
"""
import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True, help="jsonl with id and image (the prompt text is not used)")
p.add_argument("--prompt_id", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--channel", default="both", choices=["both", "ctx", "emb"])
p.add_argument("--latent_frames", type=int, default=375, help="multiple of 3; video frames = 4 * (n - 1) + 1")
p.add_argument("--seeds", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--root", default="/Users/ywang906/vdrift/models/matrixgame2")
p.add_argument("--config", default="configs/inference_yaml/inference_universal.yaml")
p.add_argument("--versions", default="z0:0:0", help="name:dL:db,... (tag becomes TAG__name)")
p.add_argument("--no_video", action="store_true")
a = p.parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torchvision.io import write_video  # noqa: E402

from pipeline import CausalInferencePipeline  # noqa: E402
from wan.vae.wanx_vae import get_wanx_vae_wrapper  # noqa: E402
from demo_utils.vae_block3 import VAEDecoderWrapper  # noqa: E402
from utils.misc import set_seed  # noqa: E402
from utils.wan_wrapper import WanDiffusionWrapper  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gen_stats", "/Users/ywang906/vdrift/stats_util.py")
gen_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_stats)

FPS, Hh, Ww, HOLD = 25, 352, 640, 9
row = next(r for r in map(json.loads, open(a.prompts)) if r["id"] == a.prompt_id)
seeds = [int(s) for s in a.seeds.split(",")]
VERS = [(f"{a.tag}__{v.split(':')[0]}", float(v.split(':')[1]), float(v.split(':')[2])) for v in a.versions.split(",")]
os.makedirs(a.outdir, exist_ok=True)
JOBS = [(t, dL, db, s) for t, dL, db in VERS for s in seeds
        if not os.path.exists(os.path.join(a.outdir, f"{a.prompt_id}_{t}_s{s}.npz"))]
if not JOBS:
    print("all done")
    raise SystemExit
assert a.latent_frames % 3 == 0

dev, wd = torch.device("cuda"), torch.bfloat16
t0 = time.time()
cfg = OmegaConf.load(a.config)
mode = cfg.pop("mode")
generator = WanDiffusionWrapper(**getattr(cfg, "model_kwargs", {}), is_causal=True)
dec = VAEDecoderWrapper()
sd = torch.load(os.path.join(a.root, "Wan2.1_VAE.pth"), map_location="cpu")
dec.load_state_dict({k: v for k, v in sd.items() if "decoder." in k or "conv2" in k})
dec.to(dev, torch.float16).requires_grad_(False).eval()
pipe = CausalInferencePipeline(cfg, generator=generator, vae_decoder=dec)
pipe.generator.load_state_dict(load_file(os.path.join(a.root, "base_distilled_model", "base_distill.safetensors")))
pipe = pipe.to(device=dev, dtype=wd)
pipe.vae_decoder.to(torch.float16)
vae = get_wanx_vae_wrapper(a.root, torch.float16)   # the wrapper forwards .to() to its VAE: move the parts instead
vae.vae.to(dev, wd)
vae.clip = vae.clip.to(dev, wd)
load_s = time.time() - t0


def opening(dL, db):
    """Centre crop to 352:640, resize, optional CIELAB shift; uint8 RGB (352, 640, 3)."""
    im = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    if h / w > Hh / Ww:
        nh = round(w * Hh / Ww)
        im = im[(h - nh) // 2:(h - nh) // 2 + nh]
    else:
        nw = round(h * Ww / Hh)
        im = im[:, (w - nw) // 2:(w - nw) // 2 + nw]
    im = cv2.resize(im, (Ww, Hh), interpolation=cv2.INTER_AREA)
    if dL or db:
        lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.clip(lab[..., 0] + dL, 0, 100)
        lab[..., 2] += db
        im = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255.0).round().astype(np.uint8)
    return im


def to_tensor(im):
    return (torch.from_numpy(im).float().div(127.5).sub(1).permute(2, 0, 1))[None, :, None].to(dev, wd)


for vt, dL, db, seed in JOBS:
    im_s, im_0 = opening(dL, db), opening(0.0, 0.0)
    im_ctx, im_emb = {"both": (im_s, im_s), "ctx": (im_s, im_0), "emb": (im_0, im_s)}[a.channel]
    set_seed(seed)
    n = a.latent_frames
    with torch.no_grad():
        x = to_tensor(im_ctx)
        cond = torch.cat([x, torch.zeros_like(x).repeat(1, 1, 4 * (n - 1), 1, 1)], dim=2)
        lat = vae.encode(cond, device=dev, tiled=True, tile_size=[44, 80], tile_stride=[23, 38]).to(dev)
        mask = torch.ones_like(lat)
        mask[:, :, 1:] = 0
        frames = 4 * (n - 1) + 1
        kb = torch.zeros(1, frames, 4, device=dev, dtype=wd)
        kb[..., 0] = 1                                                               # forward
        cd = {"cond_concat": torch.cat([mask[:, :4], lat], dim=1).to(dev, wd),
              "visual_context": vae.clip.encode_video(to_tensor(im_emb)).to(dev, wd),
              "keyboard_cond": kb, "mouse_cond": torch.zeros(1, frames, 2, device=dev, dtype=wd)}
        noise = torch.randn([1, 16, n, 44, 80], device=dev, dtype=wd)
        t1 = time.time()
        vids = pipe.inference(noise=noise, conditional_dict=cd, return_latents=False, mode=mode)
        torch.cuda.synchronize()
        gen_s = time.time() - t1
    v = rearrange(torch.cat(vids, dim=1), "B T C H W -> B T H W C")[0]
    v = ((v.float() + 1) * 127.5).clip(0, 255).round().to(torch.uint8).cpu()
    video = torch.cat([torch.from_numpy(np.stack([im_ctx] * (HOLD - 1))), v])
    out = os.path.join(a.outdir, f"{a.prompt_id}_{vt}_s{seed}")
    if not a.no_video:
        write_video(f"{out}.{os.getpid()}.tmp.mp4", video, fps=FPS, options={"crf": "12"})
        os.replace(f"{out}.{os.getpid()}.tmp.mp4", out + ".mp4")
    stats = gen_stats.frame_stats(video.numpy())
    info = dict(vars(a), tag=vt, open_dL=dL, open_db=db, seed=seed, model="matrix_game_2_universal", fps=FPS,
                hold_frames=HOLD, open_L_shifted=float(gen_stats.frame_stats(im_s[None])["lab_mean"][0, 0]),
                open_L_unshifted=float(gen_stats.frame_stats(im_0[None])["lab_mean"][0, 0]),
                gpu=torch.cuda.get_device_name(0), video_frames=int(video.shape[0]), load_s=round(load_s, 1),
                gen_s=round(gen_s, 1))
    with open(f"{out}.{os.getpid()}.tmp.json", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(f"{out}.{os.getpid()}.tmp.json", out + ".json")
    np.savez(f"{out}.{os.getpid()}.tmp.npz", **stats)
    os.replace(f"{out}.{os.getpid()}.tmp.npz", out + ".npz")
    print(json.dumps({k: info[k] for k in ("tag", "gpu", "video_frames", "gen_s", "open_L_shifted")}), flush=True)
