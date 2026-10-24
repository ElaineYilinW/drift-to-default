"""Prompt-swap replay: the per-chunk pull of the prompt, measured directly (docs/preregistration_pswap_cn.md).

Self Forcing DMD, text-to-video under prompt A (the generic prompt). Each base video is generated normally; before
chunk k (k in --kpts) the KV cache, RNG state and noise are kept, and chunk k is regenerated with the text conditioning
of another prompt B (its own cross-attention cache), everything else identical; then the cache is restored and the
base video continues. B = A is also replayed and must reproduce the base chunk exactly. The brightness of a chunk is
the mean CIELAB L* of its decoded frames (decoded alone, as in ctx_swap2.py). One .pswap.json per call.
Run from the Self-Forcing root.
usage: python pswap.py --prompts P --base_id A --swap_ids B1,B2,... --sink 3 --seeds 0,1,.. --kpts 8,20,40 --outdir D
"""
import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--base_id", required=True)
p.add_argument("--swap_ids", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--window", type=int, default=21)
p.add_argument("--sink", type=int, default=3)
p.add_argument("--latent_frames", type=int, default=126)
p.add_argument("--seeds", required=True)
p.add_argument("--kpts", default="8,20,40")
p.add_argument("--outdir", required=True)
a = p.parse_args()
os.environ["SF_SINK_REROTATE"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pipeline import CausalInferencePipeline  # noqa: E402
from utils.misc import set_seed  # noqa: E402

rows_p = {r["id"]: r["prompt"] for r in map(json.loads, open(a.prompts))}
swap_ids = a.swap_ids.split(",")
seeds = [int(s) for s in a.seeds.split(",")]
kpts = [int(k) for k in a.kpts.split(",")]
os.makedirs(a.outdir, exist_ok=True)
out_json = os.path.join(a.outdir, f"{a.base_id}_{a.tag}_s{'-'.join(map(str, seeds))}.pswap.json")
if os.path.exists(out_json):
    raise SystemExit("done")

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
NB, FS = pipe.num_frame_per_block, pipe.frame_seq_length
COND = {i: pipe.text_encoder(text_prompts=[rows_p[i]]) for i in [a.base_id] + swap_ids}
pipe.text_encoder.to("cpu")
torch.cuda.empty_cache()


def measure_L(x0):
    v = (pipe.vae.decode_to_pixel(x0, use_cache=False)[0].float() * 0.5 + 0.5).clamp(0, 1)
    lin = torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    Y = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    f = torch.where(Y > 0.008856, Y ** (1 / 3), 7.787 * Y + 16 / 116)
    return float((116 * f - 16).mean())


def new_xcache():
    pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=dev)
    return pipe.crossattn_cache


def denoise_chunk(noise, start, cond, xcache):
    x = noise[:, start:start + NB]
    for i, t in enumerate(pipe.denoising_step_list):
        ts = torch.ones([1, NB], device=dev, dtype=torch.int64) * t
        _, x0 = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts, kv_cache=pipe.kv_cache1,
                               crossattn_cache=xcache, current_start=start * FS)
        if i < len(pipe.denoising_step_list) - 1:
            nt = pipe.denoising_step_list[i + 1]
            x = pipe.scheduler.add_noise(x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                                         nt * torch.ones([NB], device=dev, dtype=torch.long)).unflatten(0, x0.shape[:2])
    return x0


def ctx_pass(x0, start, cond, xcache):
    ts = torch.zeros([1, NB], device=dev, dtype=torch.int64) + pipe.args.context_noise
    pipe.generator(noisy_image_or_video=x0, conditional_dict=cond, timestep=ts, kv_cache=pipe.kv_cache1,
                   crossattn_cache=xcache, current_start=start * FS)


rows, base = [], {}
t0 = time.time()
for seed in seeds:
    set_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.randn([1, a.latent_frames, 16, 60, 104], device=dev, dtype=torch.bfloat16, generator=g)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=dev)
    XA = new_xcache()
    XB = {i: new_xcache() for i in swap_ids}
    chunkL = []
    for blk in range(a.latent_frames // NB):
        if blk in kpts:
            rng = torch.cuda.get_rng_state()
            saved = [{kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in c.items()} for c in pipe.kv_cache1]
            for b in [a.base_id] + swap_ids:
                pipe.kv_cache1 = [{kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in c.items()} for c in saved]
                torch.cuda.set_rng_state(rng)
                xc = XA if b == a.base_id else XB[b]
                xk = denoise_chunk(noise, blk * NB, COND[b], xc)
                rows.append(dict(seed=seed, k=blk, prompt=b, L_k=measure_L(xk)))
            pipe.kv_cache1 = saved
            torch.cuda.set_rng_state(rng)
            del saved
            torch.cuda.empty_cache()
        x0 = denoise_chunk(noise, blk * NB, COND[a.base_id], XA)
        chunkL.append(measure_L(x0))
        ctx_pass(x0, blk * NB, COND[a.base_id], XA)
    base[seed] = chunkL
    for r in rows:
        if r["seed"] == seed:
            r["L_k_uninterrupted"] = chunkL[r["k"]]
    print(seed, round(time.time() - t0), flush=True)
info = dict(base_id=a.base_id, swap_ids=swap_ids, tag=a.tag, window=a.window, sink=a.sink, seeds=seeds, kpts=kpts,
            gpu=torch.cuda.get_device_name(0), secs=round(time.time() - t0, 1), base_chunk_L=base, rows=rows)
json.dump(info, open(out_json + f".{os.getpid()}.tmp", "w"), indent=1)
os.replace(out_json + f".{os.getpid()}.tmp", out_json)
chk = [abs(r["L_k"] - r["L_k_uninterrupted"]) for r in rows if r["prompt"] == a.base_id]
print(json.dumps(dict(secs=info["secs"], n_rows=len(rows), replay_check_max_abs=max(chk) if chk else None)), flush=True)
