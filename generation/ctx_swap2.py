"""Context interventions, round 2 (validity checks for ctx_swap.py).

--mode dose     one seed per call; at each k in --kpts, arms (ds, dw) from --arms: anchor chunk shifted by ds L and
                window chunks by dw L along u (synthetic tone shift), several magnitudes and a joint shift.
--mode natural  several seeds of one prompt in one call. Each base video is generated once (clean latents and RNG
                states at the k's kept); then for every ordered pair (A, B) and k, A's cache is rebuilt with A's own
                chunks except the rolling-window chunks, which are taken from B at the same positions (a NATURAL
                brightness/content difference), and A's chunk k is regenerated from A's noise and RNG state.
                B = A reproduces A exactly (checked).
Outputs per base video: .npz frame stats; one .swap2.json per call with all rows. Run from the Self-Forcing root.
"""
import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dose", "natural"], required=True)
p.add_argument("--prompts", required=True)
p.add_argument("--prompt_id", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--window", type=int, default=21)
p.add_argument("--sink", type=int, default=3)
p.add_argument("--latent_frames", type=int, default=126)
p.add_argument("--seeds", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--kpts", default="12,22,32")
p.add_argument("--arms", default="0:0", help="dose mode: comma list of ds:dw in L units")
p.add_argument("--u", required=True)
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
arms = [tuple(float(v) for v in x.split(":")) for x in a.arms.split(",")]
os.makedirs(a.outdir, exist_ok=True)
out_json = os.path.join(a.outdir, f"{a.prompt_id}_{a.tag}_{a.mode}_s{'-'.join(map(str, seeds))}.swap2.json")
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
U = torch.tensor(np.load(a.u)["u"], dtype=torch.float32, device=dev)
NB, FS = pipe.num_frame_per_block, pipe.frame_seq_length
cond = pipe.text_encoder(text_prompts=[prompt])
pipe.text_encoder.to("cpu")
torch.cuda.empty_cache()
WIN_CHUNKS = (a.window - a.sink) // NB if a.sink else a.window // NB


def measure_L(x0):
    v = (pipe.vae.decode_to_pixel(x0, use_cache=False)[0].float() * 0.5 + 0.5).clamp(0, 1)
    lin = torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    Y = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    f = torch.where(Y > 0.008856, Y ** (1 / 3), 7.787 * Y + 16 / 116)
    return float((116 * f - 16).mean())


def shift(x, dl):
    return x if dl == 0 else (x.float() + dl * U[None, None, :, None, None]).to(x.dtype)


def fresh_cache():
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=dev)


def ctx_pass(x0, start):
    ts = torch.zeros([1, NB], device=dev, dtype=torch.int64) + pipe.args.context_noise
    pipe.generator(noisy_image_or_video=x0, conditional_dict=cond, timestep=ts, kv_cache=pipe.kv_cache1,
                   crossattn_cache=pipe.crossattn_cache, current_start=start * FS)


def denoise_chunk(noise, start):
    x = noise[:, start:start + NB]
    for i, t in enumerate(pipe.denoising_step_list):
        ts = torch.ones([1, NB], device=dev, dtype=torch.int64) * t
        _, x0 = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts, kv_cache=pipe.kv_cache1,
                               crossattn_cache=pipe.crossattn_cache, current_start=start * FS)
        if i < len(pipe.denoising_step_list) - 1:
            nt = pipe.denoising_step_list[i + 1]
            x = pipe.scheduler.add_noise(x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                                         nt * torch.ones([NB], device=dev, dtype=torch.long)).unflatten(0, x0.shape[:2])
    return x0


def window_of(blk):
    return list(range(max(1 if a.sink else 0, blk - WIN_CHUNKS), blk))


def replay(chunks):
    fresh_cache()
    for j, x in enumerate(chunks):
        ctx_pass(x, j * NB)


def base_video(seed, on_kpt=None):
    """Generate one video normally; on_kpt(blk, X, noise, rng) is called before each k in kpts."""
    set_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.randn([1, a.latent_frames, 16, 60, 104], device=dev, dtype=torch.bfloat16, generator=g)
    fresh_cache()
    pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=dev)
    X, rngs, chunkL = [], {}, []
    for blk in range(a.latent_frames // NB):
        if blk in kpts:
            rngs[blk] = torch.cuda.get_rng_state()
            if on_kpt is not None:
                saved = [{kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in c.items()} for c in pipe.kv_cache1]
                on_kpt(blk, X, noise, rngs[blk])
                pipe.kv_cache1 = saved
                torch.cuda.set_rng_state(rngs[blk])
                del saved
                torch.cuda.empty_cache()
        x0 = denoise_chunk(noise, blk * NB)
        X.append(x0)
        chunkL.append(measure_L(x0))
        ctx_pass(x0, blk * NB)
    video = (pipe.vae.decode_to_pixel(torch.cat(X, dim=1), use_cache=False) * 0.5 + 0.5).clamp(0, 1)
    video = (255.0 * rearrange(video, "b t c h w -> b t h w c")[0].float().cpu()).round().clamp(0, 255).to(torch.uint8)
    stem = os.path.join(a.outdir, f"{a.prompt_id}_{a.tag}_s{seed}")
    np.savez(f"{stem}.{os.getpid()}.tmp.npz", **st.frame_stats(video.numpy()))
    os.replace(f"{stem}.{os.getpid()}.tmp.npz", stem + ".npz")
    return X, noise, rngs, chunkL


rows, base = [], {}
t0 = time.time()
if a.mode == "dose":
    for seed in seeds:
        def on_kpt(blk, X, noise, rng, seed=seed):
            win = window_of(blk)
            for ds, dw in arms:
                replay([shift(x, ds if (a.sink and j == 0) else (dw if j in win else 0.0)) for j, x in enumerate(X)])
                torch.cuda.set_rng_state(rng)
                xk = denoise_chunk(noise, blk * NB)
                rows.append(dict(seed=seed, k=blk, ds=ds, dw=dw, L_k=measure_L(xk),
                                 L_sink=measure_L(shift(X[0], ds)[:, 1:]) if a.sink else None,
                                 L_win=float(np.mean([measure_L(shift(X[j], dw)) for j in win]))))
        X, noise, rngs, chunkL = base_video(seed, on_kpt)
        base[seed] = chunkL
        for r in rows:
            if r["seed"] == seed and r["ds"] == 0 and r["dw"] == 0:
                r["L_k_uninterrupted"] = chunkL[r["k"]]
else:
    store = {}
    for seed in seeds:
        X, noise, rngs, chunkL = base_video(seed)
        store[seed] = ([x.cpu() for x in X], noise.cpu(), rngs)
        base[seed] = chunkL
    for A in seeds:
        XA, nA, rA = store[A]
        for blk in kpts:
            win = window_of(blk)
            for B in seeds:
                XB = store[B][0]
                chunks = [(XB[j] if j in win else XA[j]).to(dev) for j in range(blk)]
                replay(chunks)
                torch.cuda.set_rng_state(rA[blk])
                xk = denoise_chunk(nA.to(dev), blk * NB)
                rows.append(dict(A=A, B=B, k=blk, L_k=measure_L(xk),
                                 L_winA=float(np.mean([base[A][j] for j in win])),
                                 L_winB=float(np.mean([base[B][j] for j in win])),
                                 L_k_A=base[A][blk], L_k_B=base[B][blk]))
info = dict(mode=a.mode, prompt_id=a.prompt_id, tag=a.tag, window=a.window, sink=a.sink, seeds=seeds, kpts=kpts,
            arms=arms, gpu=torch.cuda.get_device_name(0), secs=round(time.time() - t0, 1), base_chunk_L=base, rows=rows)
json.dump(info, open(out_json + f".{os.getpid()}.tmp", "w"), indent=1)
os.replace(out_json + f".{os.getpid()}.tmp", out_json)
if a.mode == "dose":
    chk = [abs(r["L_k"] - r["L_k_uninterrupted"]) for r in rows if "L_k_uninterrupted" in r]
else:
    chk = [abs(r["L_k"] - r["L_k_A"]) for r in rows if r["A"] == r["B"]]
print(json.dumps(dict(secs=info["secs"], n_rows=len(rows), replay_check_max_abs=max(chk) if chk else None)), flush=True)
