"""Context-intervention experiment: how much does the next chunk's brightness follow its context?

While generating a long video normally, pause before chunk k (k in --kpts). For each arm (ds, dw), rebuild the KV
cache by replaying the context passes of chunks 0..k-1 from their stored clean latents, with the anchor chunk 0
shifted by ds L and the chunks currently in the rolling window shifted by dw L along the interventional brightness
direction u (jacobian.py). Then generate chunk k from the same noise and RNG state and measure its mean L* (decode).
The (0, 0) arm must reproduce the uninterrupted chunk k exactly (checked). Afterwards the original cache and RNG
state are restored and generation continues, so the base video is unaffected.

Per video: <prompt>_<tag>_s<seed>.npz/.json (base video stats, as gen.py) and .swap.json with one row per (k, arm):
L_k, the anchor and window context levels (measured by decoding the shifted context), and the base chunk L's.
Run from the Self-Forcing repo root.
"""
import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--prompt_id", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--window", type=int, default=21)
p.add_argument("--sink", type=int, default=3)
p.add_argument("--latent_frames", type=int, default=126)
p.add_argument("--seeds", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--kpts", default="12,22,32", help="chunk indices to intervene before")
p.add_argument("--delta", type=float, default=5.0, help="context shift in L units")
p.add_argument("--u", required=True, help="npz with u (16-d latent shift per +1 L)")
a = p.parse_args()
os.environ["SF_SINK_REROTATE"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pipeline import CausalInferencePipeline  # noqa: E402
from utils.misc import set_seed  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("st", os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_util.py"))
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

prompt = next(r["prompt"] for r in map(json.loads, open(a.prompts)) if r["id"] == a.prompt_id)
seeds = [int(s) for s in a.seeds.split(",")]
kpts = [int(k) for k in a.kpts.split(",")]
os.makedirs(a.outdir, exist_ok=True)
stem = lambda s: os.path.join(a.outdir, f"{a.prompt_id}_{a.tag}_s{s}")
todo = [s for s in seeds if not os.path.exists(stem(s) + ".swap.json")]
if not todo:
    raise SystemExit("all done")

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
U = torch.tensor(np.load(a.u)["u"], dtype=torch.float32, device=dev)
NB, FS = pipe.num_frame_per_block, pipe.frame_seq_length
cond = pipe.text_encoder(text_prompts=[prompt])
pipe.text_encoder.to("cpu")                        # free GPU memory for the cache copies
torch.cuda.empty_cache()
WIN_CHUNKS = (a.window - a.sink) // NB if a.sink else a.window // NB   # chunks of the rolling part


def measure_L(x0):
    v = (pipe.vae.decode_to_pixel(x0, use_cache=False)[0].float() * 0.5 + 0.5).clamp(0, 1)
    lin = torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    Y = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    f = torch.where(Y > 0.008856, Y ** (1 / 3), 7.787 * Y + 16 / 116)
    return float((116 * f - 16).mean())


def shift(x, dl):
    return x if dl == 0 else (x.float() + dl * U[None, None, :, None, None]).to(x.dtype)


def fresh_cache(bsz, dtype):
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(batch_size=bsz, dtype=dtype, device=dev)


def ctx_pass(x0, start):
    ts = torch.zeros([x0.shape[0], NB], device=dev, dtype=torch.int64) + pipe.args.context_noise
    pipe.generator(noisy_image_or_video=x0, conditional_dict=cond, timestep=ts, kv_cache=pipe.kv_cache1,
                   crossattn_cache=pipe.crossattn_cache, current_start=start * FS)


def denoise_chunk(noise, start):
    bsz = noise.shape[0]
    x = noise[:, start:start + NB]
    for i, t in enumerate(pipe.denoising_step_list):
        ts = torch.ones([bsz, NB], device=dev, dtype=torch.int64) * t
        _, x0 = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts, kv_cache=pipe.kv_cache1,
                               crossattn_cache=pipe.crossattn_cache, current_start=start * FS)
        if i < len(pipe.denoising_step_list) - 1:
            nt = pipe.denoising_step_list[i + 1]
            x = pipe.scheduler.add_noise(x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                                         nt * torch.ones([bsz * NB], device=dev, dtype=torch.long)).unflatten(0, x0.shape[:2])
    return x0


def run(noise):
    bsz, nf = noise.shape[:2]
    fresh_cache(bsz, noise.dtype)
    pipe._initialize_crossattn_cache(batch_size=bsz, dtype=noise.dtype, device=dev)
    X, rows, base_L = [], [], []
    arms = [(0.0, 0.0), (0.0, -a.delta), (0.0, a.delta)] + ([(-a.delta, 0.0), (a.delta, 0.0)] if a.sink else [])
    for blk in range(nf // NB):
        start = blk * NB
        if blk in kpts:
            saved = [{kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in c.items()} for c in pipe.kv_cache1]
            rng = torch.cuda.get_rng_state()
            win = list(range(max(1 if a.sink else 0, blk - WIN_CHUNKS), blk))   # chunks in the rolling window
            for ds, dw in arms:
                fresh_cache(bsz, noise.dtype)
                for j in range(blk):
                    d = ds if (a.sink and j == 0) else (dw if j in win else 0.0)
                    ctx_pass(shift(X[j], d), j * NB)
                torch.cuda.set_rng_state(rng)
                xk = denoise_chunk(noise, start)
                L_sink = measure_L(shift(X[0], ds)[:, 1:]) if a.sink else None
                L_win = float(np.mean([measure_L(shift(X[j], dw)) for j in win]))
                rows.append(dict(k=blk, ds=ds, dw=dw, L_k=measure_L(xk), L_sink=L_sink, L_win=L_win, n_win=len(win)))
            pipe.kv_cache1 = saved
            torch.cuda.set_rng_state(rng)
            del saved
            torch.cuda.empty_cache()
        x0 = denoise_chunk(noise, start)
        if blk in kpts:   # the (0, 0) replay must equal the uninterrupted chunk
            ref = [r for r in rows if r["k"] == blk and r["ds"] == 0 and r["dw"] == 0][0]
            ref["L_k_uninterrupted"] = measure_L(x0)
        X.append(x0)
        base_L.append(measure_L(x0) if blk % 1 == 0 else None)
        ctx_pass(x0, start)
    out = torch.cat(X, dim=1)
    video = (pipe.vae.decode_to_pixel(out, use_cache=False) * 0.5 + 0.5).clamp(0, 1)
    return video, rows, base_L


for seed in todo:
    set_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.randn([1, a.latent_frames, 16, 60, 104], device=dev, dtype=torch.bfloat16, generator=g)
    t1 = time.time()
    video, rows, base_L = run(noise)
    video = (255.0 * rearrange(video, "b t c h w -> b t h w c")[0].float().cpu()).round().clamp(0, 255).to(torch.uint8)
    stats = st.frame_stats(video.numpy())
    out = stem(seed)
    info = dict(prompt_id=a.prompt_id, tag=a.tag, seed=seed, window=a.window, sink=a.sink, delta=a.delta, kpts=kpts,
                gpu=torch.cuda.get_device_name(0), gen_s=round(time.time() - t1, 1), base_chunk_L=base_L, swaps=rows)
    np.savez(f"{out}.{os.getpid()}.tmp.npz", **stats)
    os.replace(f"{out}.{os.getpid()}.tmp.npz", out + ".npz")
    json.dump(info, open(f"{out}.{os.getpid()}.tmp.json", "w"), indent=1)
    os.replace(f"{out}.{os.getpid()}.tmp.json", out + ".swap.json")
    chk = [abs(r["L_k"] - r["L_k_uninterrupted"]) for r in rows if "L_k_uninterrupted" in r]
    print(json.dumps(dict(seed=seed, gen_s=info["gen_s"], n_swaps=len(rows), replay_check_max_abs=max(chk))), flush=True)
