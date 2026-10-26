"""Held-opening generation with MAGI-1 4.5B-distill (Sand AI; its own architecture, not Wan), for the relaxation study.

MAGI-1 generates a video autoregressively in chunks of 6 latent frames (24 video frames = 1 s at 24 fps), with a KV
cache of earlier chunks. Which earlier chunks a chunk attends to is set by the runtime config:
  --kv window   the released setting: noise2clean_kvrange [5, 4, 3, 2], clean_chunk_kvrange 1, i.e. a sliding window of
                at most 4 earlier chunks (about 4 s) and no sink; the opening leaves the context after a few seconds;
  --kv full     noise2clean_kvrange [] (the code's default range): every chunk attends to all earlier chunks, so the
                opening stays in the context for the whole rollout.
Image-to-video: the opening image (centre-cropped to the output aspect, resized, optionally shifted in CIELAB as in
hold_gen.py) is VAE-encoded as the clean prefix (MAGI-1's own i2v path, process_image without its ffmpeg resize). 8 copies of it are prepended to the output so
that frames 0..8 are the opening (json hold_frames = 9). --fresh: text-to-video, no opening (hold_frames = 0).
Seeds: the global RNG is reset to the seed before every video, so the versions of one run share the noise.
Outputs as hold_gen.py: {outdir}/{prompt_id}_{tag}_s{seed}.{mp4,npz,json}; json has fps = 24.
"""
import argparse
import json
import os
import random
import time

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--prompt_id", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--kv", required=True, choices=["window", "full"])
p.add_argument("--seconds", type=float, default=30.0)
p.add_argument("--seeds", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--h", type=int, default=432)
p.add_argument("--w", type=int, default=768)
p.add_argument("--root", default="/Users/ywang906/vdrift/models/magi/ckpt")
p.add_argument("--repo", default="/Users/ywang906/vdrift/MAGI-1")
p.add_argument("--init_image", default="")
p.add_argument("--open_dL", type=float, default=0.0)
p.add_argument("--open_da", type=float, default=0.0)
p.add_argument("--open_db", type=float, default=0.0)
p.add_argument("--fresh", action="store_true")
p.add_argument("--no_video", action="store_true")
p.add_argument("--versions", default="", help="several openings in one run: name:dL:db,... (tag becomes TAG__name)")
a = p.parse_args()

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", str(20000 + os.getpid() % 20000))
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OFFLOAD_T5_CACHE", "true")
os.environ.setdefault("OFFLOAD_VAE_CACHE", "true")
os.environ.setdefault("PAD_HQ", "1")
os.environ.setdefault("PAD_DURATION", "1")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_{os.getuid()}_{os.getpid()}")   # NFS home cache races between jobs
import sys  # noqa: E402

sys.path.insert(0, a.repo)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.io import write_video  # noqa: E402

from inference.common import MagiConfig, set_random_seed  # noqa: E402
from inference.infra.distributed import dist_init  # noqa: E402
from inference.model.dit import get_dit  # noqa: E402
from inference.pipeline.prompt_process import get_txt_embeddings  # noqa: E402
from inference.pipeline.video_generate import generate_per_chunk  # noqa: E402
import inference.infra.distributed.parallel_state as mpu  # noqa: E402
from inference.pipeline.video_process import encode_prefix_video, post_chunk_process  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gen_stats", "/Users/ywang906/vdrift/stats_util.py")
gen_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_stats)

FPS = 24
_row = next(r for r in map(json.loads, open(a.prompts)) if r["id"] == a.prompt_id)
prompt = _row["prompt"]
if not a.init_image and not a.fresh:
    a.init_image = _row.get("image", f"/Users/ywang906/vdrift/relax_starts/{a.prompt_id}.jpg")
HOLD = 0 if a.fresh else 9
seeds = [int(s) for s in a.seeds.split(",")]
os.makedirs(a.outdir, exist_ok=True)
stem = lambda s, t: os.path.join(a.outdir, f"{a.prompt_id}_{t}_s{s}")
VERS = ([(a.tag, a.open_dL, a.open_da, a.open_db)] if not a.versions else
        [(f"{a.tag}__{v.split(':')[0]}", float(v.split(':')[1]), 0.0, float(v.split(':')[2])) for v in a.versions.split(",")])
JOBS = [(t, dL, da, db, s) for t, dL, da, db in VERS for s in seeds if not os.path.exists(stem(s, t) + ".npz")]
if not JOBS:
    print("all done")
    raise SystemExit

cfg = json.load(open(f"{a.repo}/example/4.5B/4.5B_distill_config.json"))
rc = cfg["runtime_config"]
rc.update(load=f"{a.root}/magi/4.5B_distill", t5_pretrained=f"{a.root}/t5", vae_pretrained=f"{a.root}/vae",
          video_size_h=a.h, video_size_w=a.w, num_frames=int(round(a.seconds * FPS)), fps=FPS, seed=seeds[0])
if a.kv == "full":
    rc["noise2clean_kvrange"] = []
cfg_path = os.path.join(a.outdir, f".magi_cfg_{os.getpid()}.json")
json.dump(cfg, open(cfg_path, "w"), indent=1)
t0 = time.time()
config = MagiConfig.from_json(cfg_path)
dist_init(config)
caption_embs, emb_masks = get_txt_embeddings(prompt, config)
dit = get_dit(config)
load_s = time.time() - t0


def opening(path, dL, da, db):
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    tw = round(h * a.w / a.h)
    if tw <= w:
        x0 = (w - tw) // 2
        im = im[:, x0:x0 + tw]
    else:
        th = round(w * a.h / a.w)
        y0 = (h - th) // 2
        im = im[y0:y0 + th]
    im = cv2.resize(im, (a.w, a.h), interpolation=cv2.INTER_AREA)
    if dL or da or db:
        lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.clip(lab[..., 0] + dL, 0, 100)
        lab[..., 1] += da
        lab[..., 2] += db
        im = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255.0).round().astype(np.uint8)
    return im


for vt, dL, da, db, seed in JOBS:
    prefix, im = None, None
    if not a.fresh:
        im = opening(a.init_image, dL, da, db)
        prefix = encode_prefix_video(torch.from_numpy(im)[None], FPS, rc["vae_pretrained"], rc["scale_factor"],
                                     parallel_group=mpu.get_tp_group(with_context_parallel=True))
    set_random_seed(seed)
    random.seed(seed)
    t1 = time.time()
    with torch.no_grad():
        chunks = [post_chunk_process(c, config).cpu() for c in generate_per_chunk(
            model=dit, prefix_video=prefix, caption_embs=caption_embs, emb_masks=emb_masks)]
    torch.cuda.synchronize()
    gen_s = time.time() - t1
    video = torch.cat(chunks, 0).permute(0, 2, 3, 1).contiguous()           # (T, H, W, 3) uint8
    raw = int(video.shape[0])
    if HOLD:
        video = torch.cat([torch.from_numpy(np.stack([im] * (HOLD - 1))), video])
    out = stem(seed, vt)
    if not a.no_video:
        tmp = f"{out}.{os.getpid()}.tmp.mp4"
        write_video(tmp, video, fps=FPS, options={"crf": "12"})
        os.replace(tmp, out + ".mp4")
    stats = gen_stats.frame_stats(video.numpy())
    info = dict(vars(a), tag=vt, open_dL=dL, open_da=da, open_db=db, seed=seed, prompt=None, model="magi1_4.5b_distill",
                fps=FPS, hold_frames=HOLD, noise2clean_kvrange=rc["noise2clean_kvrange"],
                clean_chunk_kvrange=rc["clean_chunk_kvrange"],
                open_L_shifted=None if im is None else float(gen_stats.frame_stats(im[None])["lab_mean"][0, 0]),
                gpu=torch.cuda.get_device_name(0), video_frames=int(video.shape[0]), raw_frames=raw,
                load_s=round(load_s, 1), gen_s=round(gen_s, 1))
    with open(f"{out}.{os.getpid()}.tmp.json", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(f"{out}.{os.getpid()}.tmp.json", out + ".json")
    np.savez(f"{out}.{os.getpid()}.tmp.npz", **stats)
    os.replace(f"{out}.{os.getpid()}.tmp.npz", out + ".npz")
    print(json.dumps(info), flush=True)
os.remove(cfg_path)
