"""Feasibility test: generate one long Self Forcing video and record time and GPU memory.

Run from the Self-Forcing repo root (it loads wan_models/ by relative path).
local_attn_size = KV-cache window in latent frames (-1 = global, max 21); sink_size = first latent frames kept
in the window (attention sink, i.e. anchoring to the start). 1 latent frame = 4 video frames.
"""
import argparse
import json
import os
import time

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from pipeline import CausalInferencePipeline
from utils.misc import set_seed


def vm(tag):
    # address space (h_vmem counts VmSize) and resident memory, to diagnose SGE kills during loading
    st = dict(l.split(":", 1) for l in open("/proc/self/status") if l.startswith(("VmSize", "VmPeak", "VmRSS")))
    print(f"[mem] {tag}: " + " ".join(f"{k}={int(v.split()[0]) / 2**20:.1f}G" for k, v in st.items()), flush=True)


vm("imports")
import threading


def _vm_loop():
    while True:
        time.sleep(2)
        vm("tick")


threading.Thread(target=_vm_loop, daemon=True).start()
for line in open("/proc/self/cgroup"):
    if "memory" in line:
        path = line.strip().split(":", 2)[2]
        for f in ("memory.limit_in_bytes", "memory.memsw.limit_in_bytes"):
            try:
                print(f"[mem] cgroup {path} {f}={int(open(f'/sys/fs/cgroup/memory{path}/{f}').read()) / 2**30:.1f}G", flush=True)
            except OSError as e:
                print(f"[mem] cgroup {f}: {e}", flush=True)

p = argparse.ArgumentParser()
p.add_argument("--latent_frames", type=int, default=21)
p.add_argument("--local_attn", type=int, default=-1)
p.add_argument("--sink", type=int, default=0)
p.add_argument("--prompt", type=str, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=str, required=True)
a = p.parse_args()

t0 = time.time()
cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
cfg.model_kwargs.local_attn_size = a.local_attn
cfg.model_kwargs.sink_size = a.sink
dev = torch.device("cuda")
set_seed(a.seed)
torch.set_grad_enabled(False)
pipe = CausalInferencePipeline(cfg, device=dev)
vm("pipeline built")
sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu", mmap=True)
pipe.generator.load_state_dict(sd["generator_ema"])
del sd
vm("checkpoint loaded")
pipe = pipe.to(dtype=torch.bfloat16).to(dev)
torch.cuda.synchronize()
vm("on gpu")
t_load = time.time() - t0

noise = torch.randn([1, a.latent_frames, 16, 60, 104], device=dev, dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
video = pipe.inference(noise=noise, text_prompts=[a.prompt])
torch.cuda.synchronize()
t_gen = time.time() - t1

video = (255.0 * rearrange(video, "b t c h w -> b t h w c")[0].float().cpu()).to(torch.uint8)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
write_video(a.out, video, fps=16)
info = dict(vars(a), gpu=torch.cuda.get_device_name(0), torch=torch.__version__, video_frames=int(video.shape[0]),
            load_s=round(t_load, 1), gen_s=round(t_gen, 1), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
print(json.dumps(info))
with open(a.out.replace(".mp4", ".json"), "w") as f:
    json.dump(info, f, indent=1)
