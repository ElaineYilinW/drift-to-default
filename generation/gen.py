"""Generate Self Forcing videos for one (prompt, context strategy) over several seeds, plus per-frame colour stats.

Run from the Self-Forcing repo root (it loads wan_models/ by relative path). The model is loaded once per call.
window = KV-cache window in latent frames (-1 = global, max 21); sink = first latent frames kept in the window.
1 latent frame = 4 video frames (+1 for the first). For a given seed the initial noise and the sampler's RNG
stream are identical across strategies, so strategies are paired: videos agree until the window starts rolling.

Outputs per seed in --outdir: {prompt}_{strategy}_s{seed}.mp4, .npz (per-frame stats from the raw frames,
before video compression) and .json (timing). Existing .npz files are skipped.
"""
import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True, help="jsonl with id, prompt")
p.add_argument("--prompt_id", required=True)
p.add_argument("--strategy", required=True, help="name, e.g. w21s3")
p.add_argument("--window", type=int, required=True)
p.add_argument("--sink", type=int, default=0)
p.add_argument("--latent_frames", type=int, required=True)
p.add_argument("--seeds", type=str, required=True, help="comma-separated")
p.add_argument("--outdir", required=True)
p.add_argument("--no_video", action="store_true", help="skip the mp4 (stats only)")
p.add_argument("--rerotate", action="store_true", help="re-rotate sink keys to sit just before the window")
a = p.parse_args()
os.environ["SF_SINK_REROTATE"] = "1" if a.rerotate else "0"  # read by the patched causal_model.py at import

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torchvision.io import write_video  # noqa: E402

from pipeline import CausalInferencePipeline  # noqa: E402
from utils.misc import set_seed  # noqa: E402
import wan.modules.causal_model as cm  # noqa: E402
assert cm.SINK_REROTATE == a.rerotate, "Self-Forcing not patched (jobs/patch_sf.py)"

prompt = next(r["prompt"] for r in map(json.loads, open(a.prompts)) if r["id"] == a.prompt_id)
seeds = [int(s) for s in a.seeds.split(",")]
os.makedirs(a.outdir, exist_ok=True)
stem = lambda s: os.path.join(a.outdir, f"{a.prompt_id}_{a.strategy}_s{s}")
todo = [s for s in seeds if not os.path.exists(stem(s) + ".npz")]
if not todo:
    print("all done")
    raise SystemExit

t0 = time.time()
cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
cfg.model_kwargs.local_attn_size = a.window
cfg.model_kwargs.sink_size = a.sink
dev = torch.device("cuda")
torch.set_grad_enabled(False)
pipe = CausalInferencePipeline(cfg, device=dev)
sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu", mmap=True)
pipe.generator.load_state_dict(sd["generator_ema"])
del sd
pipe = pipe.to(dtype=torch.bfloat16).to(dev)
load_s = time.time() - t0


def frame_stats(video):
    """video: (T, H, W, 3) uint8 RGB -> dict of per-frame arrays."""
    lab_mean, lab_std, sat, val, sharp, hue = [], [], [], [], [], []
    for f in video:
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        lab[:, 0] *= 100 / 255  # 8-bit OpenCV Lab -> L in 0..100, a/b centred at 0
        lab[:, 1:] -= 128
        lab_mean.append(lab.mean(0))
        lab_std.append(lab.std(0))
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        sat.append(hsv[..., 1].mean() / 255)
        val.append(hsv[..., 2].mean() / 255)
        hue.append(np.bincount(hsv[..., 0].ravel(), minlength=180)[:180] / hsv[..., 0].size)
        sharp.append(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())
    return dict(lab_mean=np.array(lab_mean), lab_std=np.array(lab_std), sat_mean=np.array(sat),
                val_mean=np.array(val), sharpness=np.array(sharp), hue_hist=np.array(hue, dtype=np.float16))


for seed in todo:
    set_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.randn([1, a.latent_frames, 16, 60, 104], device=dev, dtype=torch.bfloat16, generator=g)
    pipe.kv_cache1 = None  # fresh caches for every video
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    video = pipe.inference(noise=noise, text_prompts=[prompt])
    torch.cuda.synchronize()
    gen_s = time.time() - t1
    video = (255.0 * rearrange(video, "b t c h w -> b t h w c")[0].float().cpu()).round().clamp(0, 255).to(torch.uint8)
    out = stem(seed)
    if not a.no_video:
        tmp = f"{out}.{os.getpid()}.tmp.mp4"
        write_video(tmp, video, fps=16, options={"crf": "12"})
        os.replace(tmp, out + ".mp4")
    stats = frame_stats(video.numpy())
    info = dict(prompt_id=a.prompt_id, strategy=a.strategy, window=a.window, sink=a.sink, rerotate=a.rerotate, seed=seed,
                latent_frames=a.latent_frames, video_frames=int(video.shape[0]), gpu=torch.cuda.get_device_name(0),
                load_s=round(load_s, 1), gen_s=round(gen_s, 1),
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    with open(f"{out}.{os.getpid()}.tmp.json", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(f"{out}.{os.getpid()}.tmp.json", out + ".json")
    np.savez(f"{out}.{os.getpid()}.tmp.npz", **stats)
    os.replace(f"{out}.{os.getpid()}.tmp.npz", out + ".npz")  # written last: the done marker
    print(json.dumps(info), flush=True)
