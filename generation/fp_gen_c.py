"""FramePack content routes (study 17): copy of fp_gen.py (frozen) in which the second image is a different scene.

Every FramePack section sees the opening twice: its VAE latent as a clean context prefix, and its SigLIP embedding as a
global condition. Here the two routes get either the opening (a snow / desert / ... scene) or a city image (--alt_image,
or the prompt row's "alt_image"):
  both  context = opening, embedding = opening
  ctx   context = opening, embedding = city image
  emb   context = city image, embedding = opening
  alt   context = city image, embedding = city image (the city start that gives the default)
No CIELAB shift. The 8 prepended frames show the context image. Settings and outputs as fp_gen.py (vanilla = F1).
"""
import argparse
import gc
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--prompt_id", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--mode", required=True, choices=["inverted", "vanilla"])
p.add_argument("--seconds", type=float, default=20.0)
p.add_argument("--seeds", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--root", default="/Users/ywang906/vdrift/models/framepack")
p.add_argument("--steps", type=int, default=25)
p.add_argument("--guidance", type=float, default=10.0)
p.add_argument("--init_image", default="")
p.add_argument("--open_dL", type=float, default=0.0)
p.add_argument("--open_da", type=float, default=0.0)
p.add_argument("--open_db", type=float, default=0.0)
p.add_argument("--channel", default="both", choices=["both", "ctx", "emb", "alt"])
p.add_argument("--alt_image", default="")
p.add_argument("--no_video", action="store_true")
p.add_argument("--versions", default="", help="several openings in one run: name:dL:db,... (tag becomes TAG__name)")
a = p.parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.io import write_video  # noqa: E402
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel  # noqa: E402
from transformers import SiglipImageProcessor, SiglipVisionModel  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gen_stats", "/Users/ywang906/vdrift/stats_util.py")
gen_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_stats)

FPS, Hh, Ww, HOLD = 30, 480, 832, 9
_row = next(r for r in map(json.loads, open(a.prompts)) if r["id"] == a.prompt_id)
prompt = _row["prompt"]
if not a.init_image:
    a.init_image = _row["image"]
if not a.alt_image:
    a.alt_image = _row["alt_image"]
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
fe = SiglipImageProcessor.from_pretrained(f"{a.root}/flux_redux_bfl", subfolder="feature_extractor")
ie = SiglipVisionModel.from_pretrained(f"{a.root}/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16)
pipe = HunyuanVideoFramepackPipeline.from_pretrained(f"{a.root}/HunyuanVideo", transformer=None, feature_extractor=fe,
                                                     image_encoder=ie, torch_dtype=torch.float16)
pipe.text_encoder.to(dev)
pipe.text_encoder_2.to(dev)
with torch.no_grad():
    pe, ppe, pam = pipe.encode_prompt(prompt=prompt, device=dev, dtype=torch.bfloat16)
pipe.register_modules(text_encoder=None, text_encoder_2=None)
gc.collect()
torch.cuda.empty_cache()
ck = {"inverted": "FramePackI2V_HY", "vanilla": "FramePack_F1_I2V_HY"}[a.mode]
tr = HunyuanVideoFramepackTransformer3DModel.from_pretrained(f"{a.root}/{ck}", torch_dtype=torch.bfloat16)
pipe.register_modules(transformer=tr)
pipe.to(dev)
pipe.vae.enable_tiling()
load_s = time.time() - t0


def opening(path, shift=True):
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
    if shift and (a.open_dL or a.open_da or a.open_db):
        lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.clip(lab[..., 0] + a.open_dL, 0, 100)
        lab[..., 1] += a.open_da
        lab[..., 2] += a.open_db
        im = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255.0).round().astype(np.uint8)
    return im


for vt, dL, da, db, seed in JOBS:
    a.tag, a.open_dL, a.open_da, a.open_db = vt, dL, da, db
    im_s, im_0 = opening(a.init_image), opening(a.alt_image, shift=False)
    im, emb_im = {"both": (im_s, im_s), "ctx": (im_s, im_0), "emb": (im_0, im_s), "alt": (im_0, im_0)}[a.channel]
    _orig = type(pipe).encode_image.__get__(pipe)
    pipe.encode_image = lambda image, *ar, **kw: _orig(
        pipe.video_processor.preprocess(Image.fromarray(emb_im), Hh, Ww).to(image.device, image.dtype), *ar, **kw)
    t1 = time.time()
    with torch.no_grad():
        frames = pipe(image=Image.fromarray(im), prompt_embeds=pe, pooled_prompt_embeds=ppe, prompt_attention_mask=pam,
                      height=Hh, width=Ww, num_frames=int(round(a.seconds * FPS)), num_inference_steps=a.steps,
                      guidance_scale=a.guidance, generator=torch.Generator().manual_seed(seed),
                      sampling_type="inverted_anti_drifting" if a.mode == "inverted" else "vanilla",
                      output_type="np").frames[0]
    torch.cuda.synchronize()
    gen_s = time.time() - t1
    video = torch.from_numpy((np.clip(frames, 0, 1) * 255).round().astype(np.uint8))      # (T, H, W, 3)
    video = torch.cat([torch.from_numpy(np.stack([im] * (HOLD - 1))), video])
    out = stem(seed)
    if not a.no_video:
        tmp = f"{out}.{os.getpid()}.tmp.mp4"
        write_video(tmp, video, fps=FPS, options={"crf": "12"})
        os.replace(tmp, out + ".mp4")
    stats = gen_stats.frame_stats(video.numpy())
    info = dict(vars(a), seed=seed, prompt=None, model=f"framepack_{a.mode}_content", fps=FPS, hold_frames=HOLD,
                open_L_shifted=float(gen_stats.frame_stats(im_s[None])["lab_mean"][0, 0]),
                open_L_unshifted=float(gen_stats.frame_stats(im_0[None])["lab_mean"][0, 0]),
                gpu=torch.cuda.get_device_name(0), video_frames=int(video.shape[0]), raw_frames=int(frames.shape[0]),
                load_s=round(load_s, 1), gen_s=round(gen_s, 1))
    with open(f"{out}.{os.getpid()}.tmp.json", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(f"{out}.{os.getpid()}.tmp.json", out + ".json")
    np.savez(f"{out}.{os.getpid()}.tmp.npz", **stats)
    os.replace(f"{out}.{os.getpid()}.tmp.npz", out + ".npz")
    print(json.dumps(info), flush=True)
