"""Held-opening generation with Krea Realtime 14B (Self Forcing distilled from Wan2.1-T2V-14B; krea-ai/realtime-video,
inference code of the release), for the relaxation study.

The release generates blocks of 3 latent frames (12 video frames at 16 fps). Before every block it rebuilds the KV
cache from the clean latents of the last kv_cache_num_frames = 3 latent frames (KV-cache recomputation), with rotary
positions restarting at 0 for the first context frame, so nothing leaks from evicted frames. Which latent sits first in
that context is set by --anchor:
  --anchor none   the released default: the first context frame is the oldest decoded frame of the window re-encoded
                  as a single-frame latent; the opening leaves the context after two blocks;
  --anchor first  the release's keep_first_frame option: the opening's first latent stays the first context frame for
                  the whole rollout (an anchor at the position just before the window, i.e. a re-rotated sink).
Image-to-video: the opening image (centre-cropped to 832x480, resized, optionally shifted in CIELAB as in hold_gen.py)
goes through the release's start_frame path (9 copies VAE-encoded as the first 3 latent frames). The release does not
output the start frame itself, so 9 copies of it are prepended to the output: frames 0..8 are the opening and the first
generated frame is frame 9 (json hold_frames = 9). --fresh: text-to-video.
Memory: the text encoder runs on the GPU first and is freed; the 14B transformer is built on the CPU from the Wan2.1-14B
config (no 14B Wan weights needed), loaded from the Krea checkpoint, cast to bf16, its q/k/v projections fused as in the
release (and the unfused copies, which the release keeps but never uses again, deleted) and moved to the GPU, so that it
fits a 48-GB card. Steps and timestep schedule are the release's (GenerateParams defaults).
Outputs as hold_gen.py: {outdir}/{prompt_id}_{tag}_s{seed}.{mp4,npz,json}; json has fps = 16.
"""
import argparse
import gc
import json
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--prompt_id", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--anchor", required=True, choices=["none", "first"])
p.add_argument("--seconds", type=float, default=60.0)
p.add_argument("--seeds", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--repo", default="/Users/ywang906/vdrift/krea-rt")
p.add_argument("--kv_frames", type=int, default=3)
p.add_argument("--init_image", default="")
p.add_argument("--open_dL", type=float, default=0.0)
p.add_argument("--open_da", type=float, default=0.0)
p.add_argument("--open_db", type=float, default=0.0)
p.add_argument("--fresh", action="store_true")
p.add_argument("--no_video", action="store_true")
p.add_argument("--versions", default="", help="several openings in one run: name:dL:db,... (tag becomes TAG__name)")
a = p.parse_args()

os.environ["MODEL_FOLDER"] = f"{a.repo}/wan_models"
os.environ.setdefault("DO_COMPILE", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_{os.getuid()}_{os.getpid()}")   # NFS home cache races between jobs
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor_{os.getuid()}_{os.getpid()}")
outdir = os.path.abspath(a.outdir)
os.chdir(a.repo)
sys.path.insert(0, a.repo)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torchvision.io import write_video  # noqa: E402

torch.set_grad_enabled(False)
import release_server as rs  # noqa: E402
import utils.wan_wrapper as ww  # noqa: E402
import wan.modules.causal_model as cm  # noqa: E402
from utils.scheduler import FlowMatchScheduler  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gen_stats", "/Users/ywang906/vdrift/stats_util.py")
gen_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_stats)

FPS, W, H = 16, 832, 480
_row = next(r for r in map(json.loads, open(a.prompts)) if r["id"] == a.prompt_id)
prompt = _row["prompt"]
if not a.init_image and not a.fresh:
    a.init_image = _row.get("image", f"/Users/ywang906/vdrift/relax_starts/{a.prompt_id}.jpg")
HOLD = 0 if a.fresh else 9
seeds = [int(s) for s in a.seeds.split(",")]
os.makedirs(outdir, exist_ok=True)
stem = lambda s, t: os.path.join(outdir, f"{a.prompt_id}_{t}_s{s}")
VERS = ([(a.tag, a.open_dL, a.open_da, a.open_db)] if not a.versions else
        [(f"{a.tag}__{v.split(':')[0]}", float(v.split(':')[1]), 0.0, float(v.split(':')[2])) for v in a.versions.split(",")])
JOBS = [(t, dL, da, db, s) for t, dL, da, db in VERS for s in seeds if not os.path.exists(stem(s, t) + ".npz")]
if not JOBS:
    print("all done")
    raise SystemExit
n_frames = int(round(a.seconds * FPS))
NUM_BLOCKS = 1 + -(-(n_frames - 6) // 12)          # the first block yields 6 frames, every later one 12

t0 = time.time()
config = rs.load_merge_config("configs/self_forcing_server_14b.yaml")
# text encoder: the release's (umt5-xxl, bf16), loaded from the .pth of the Wan2.1-1.3B folder, run once, then freed
ww.safe_load_file = lambda path, device: torch.load(path.replace(".safetensors", ".pth"), map_location=device,
                                                    weights_only=True)
te = rs.load_text_encoder()
cond = {k: v.to(dtype=torch.bfloat16).contiguous() for k, v in te(text_prompts=[prompt]).items()}
del te
gc.collect()
torch.cuda.empty_cache()
print(f"after text encoder: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)


class CachedText(torch.nn.Module):
    def forward(self, text_prompts):
        return {k: v.clone() for k, v in cond.items()}


# transformer: Wan2.1-14B config, built on the CPU, Krea weights, bf16, to the GPU
cm.CausalWanModel.from_pretrained = classmethod(lambda cls, path, **kw: cls.from_config(cls.load_config(path), **kw))
sd = load_file(config.checkpoint_path, device="cpu")
assert sd["model.blocks.0.self_attn.k.weight"].shape[0] == 5120
transformer = ww.WanDiffusionWrapper(model_name="Wan2.1-T2V-14B", timestep_shift=getattr(config, "timestep_shift", 5.0),
                                     is_causal=True)
transformer.load_state_dict(sd)
del sd
transformer = transformer.to(dtype=torch.bfloat16).eval().requires_grad_(False)
for block in transformer.model.blocks:
    block.self_attn.fuse_projections()
    del block.self_attn.q, block.self_attn.k, block.self_attn.v      # unused once fused; the release keeps both (6 GB)
transformer = transformer.to(torch.cuda.current_device())
print(f"transformer on GPU: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
vae_encoder, vae_decoder = rs.load_vae()
pipeline = rs.load_pipeline(config, torch.cuda.current_device(), transformer, CachedText(), vae_decoder)
models = rs.Models(CachedText(), transformer, pipeline, vae_encoder, vae_decoder)
load_s = time.time() - t0
print(f"all models on GPU: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)


def opening(path, dL, da, db):
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    tw = round(h * W / H)
    if tw <= w:
        x0 = (w - tw) // 2
        im = im[:, x0:x0 + tw]
    else:
        th = round(w * H / W)
        y0 = (h - th) // 2
        im = im[y0:y0 + th]
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    if dL or da or db:
        lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.clip(lab[..., 0] + dL, 0, 100)
        lab[..., 1] += da
        lab[..., 2] += db
        im = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255.0).round().astype(np.uint8)
    return im


for vt, dL, da, db, seed in JOBS:
    im = None if a.fresh else opening(a.init_image, dL, da, db)
    frames = []

    def frame_callback(pixels, frame_ids, event):
        event.synchronize()
        x = pixels[0].float().add(1.0).mul(127.5).clamp(0, 255).round().to(torch.uint8)    # (T, 3, H, W)
        frames.append(x.permute(0, 2, 3, 1).cpu())

    # with a start frame its 3 latents fill the first block's slots of the latent buffer, so the release's buffer of
    # num_blocks blocks holds only num_blocks - 1 generated blocks (the last one would read an empty noise slice): the
    # buffer gets one extra block, and NUM_BLOCKS blocks are generated either way
    params = rs.GenerateParams(prompt=prompt, seed=seed, num_blocks=NUM_BLOCKS + (im is not None),
                               kv_cache_num_frames=a.kv_frames, keep_first_frame=a.anchor == "first")
    params.start_frame = None if im is None else Image.fromarray(im)     # the release types it bytes | str
    t1 = time.time()
    session = rs.GenerationSession(params=params, config=config, frame_callback=frame_callback, models=models)
    for b in range(NUM_BLOCKS):
        session.generate_block(models)
        if b in (0, 1, NUM_BLOCKS - 1):
            torch.cuda.synchronize()
            print(f"block {b + 1}/{NUM_BLOCKS} at {time.time() - t1:.0f} s, {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                  flush=True)
    torch.cuda.synchronize()
    gen_s = time.time() - t1
    session.dispose()
    video = torch.cat(frames, 0)[:n_frames]                                    # (T, H, W, 3) uint8
    raw = int(video.shape[0])
    if HOLD:
        video = torch.cat([torch.from_numpy(np.stack([im] * HOLD)), video])      # the release outputs no start frame
    out = stem(seed, vt)
    if not a.no_video:
        tmp = f"{out}.{os.getpid()}.tmp.mp4"
        write_video(tmp, video, fps=FPS, options={"crf": "12"})
        os.replace(tmp, out + ".mp4")
    stats = gen_stats.frame_stats(video.numpy())
    info = dict(vars(a), tag=vt, open_dL=dL, open_da=da, open_db=db, seed=seed, prompt=None, model="krea_realtime_14b",
                fps=FPS, hold_frames=HOLD, num_blocks=NUM_BLOCKS, keep_first_frame=a.anchor == "first",
                denoising_steps=[int(x) for x in session.denoising_step_list.tolist()],
                open_L_shifted=None if im is None else float(gen_stats.frame_stats(im[None])["lab_mean"][0, 0]),
                gpu=torch.cuda.get_device_name(0), video_frames=int(video.shape[0]), raw_frames=raw,
                max_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), load_s=round(load_s, 1),
                gen_s=round(gen_s, 1))
    with open(f"{out}.{os.getpid()}.tmp.json", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(f"{out}.{os.getpid()}.tmp.json", out + ".json")
    np.savez(f"{out}.{os.getpid()}.tmp.npz", **stats)
    os.replace(f"{out}.{os.getpid()}.tmp.npz", out + ".npz")
    print(json.dumps(info), flush=True)
