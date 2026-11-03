"""VBench-style quality dimensions per 10-s segment, for every video of one directory (study 39).

Each mp4 (16 fps) is cut after its first chunk (frames 0..8) into 10-s segments, segment k = frames
[9 + 160k, 9 + 160(k + 1)). Per segment:
  imaging      MUSIQ (SPAQ weights, pyiqa musiq-spaq) on every 8th frame, longer side resized to 512 (VBench's protocol)
  aesthetic    LAION aesthetic predictor v1 (linear head sa_0_4_vit_l_14_linear.pth on the L2-normalised CLIP ViT-L/14
               image embedding, as VBench) on every 8th frame, divided by 10
  subject      DINO ViT-B/16 (facebook/dino-vitb16) CLS feature of every 4th frame; mean over the segment's frames
               2..n of (cos(f_i, f_1) + cos(f_i, f_{i-1})) / 2, VBench's subject-consistency formula within a segment
  background   the same formula with CLIP ViT-B/32 image embeddings (VBench's background consistency)
  flow         RAFT-large (torchvision) optical flow between consecutive frames taken every 2nd frame (8 fps), at half
               resolution; per pair the mean of the top 5% of flow magnitudes (VBench's dynamic-degree score before
               its threshold), averaged over the segment
Writes VIDEO.vbl.npz next to each video (per-segment arrays and the per-frame scores); existing files are skipped.
usage: python vbench_lite.py VIDEO_DIR PATTERN DONE_FILE      (PATTERN e.g. '*_s13.mp4')
"""
import glob
import os
import sys
import urllib.request

import cv2
import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel

vdir, pat, done = sys.argv[1], sys.argv[2], sys.argv[3]
FPS, HOLD, SEG = 16, 9, 160
dev = torch.device("cuda")
HEAD = os.path.join(os.environ.get("HF_HOME", "."), "sa_0_4_vit_l_14_linear.pth")
if not os.path.exists(HEAD):
    urllib.request.urlretrieve(
        "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth", HEAD + ".tmp")
    os.replace(HEAD + ".tmp", HEAD)
aes_head = torch.nn.Linear(768, 1)
aes_head.load_state_dict(torch.load(HEAD, map_location="cpu"))
aes_head = aes_head.to(dev).eval()
clipL = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half()
procL = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clipB = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval().half()
procB = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
dino = ViTModel.from_pretrained("facebook/dino-vitb16").to(dev).eval().half()
procD = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
musiq = pyiqa.create_metric("musiq-spaq", device=dev)
raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval()
raft_tf = Raft_Large_Weights.DEFAULT.transforms()


def frames_of(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def embed(model, proc, imgs, bs=32, vit=False):
    out = []
    for i in range(0, len(imgs), bs):
        x = proc(images=imgs[i:i + bs], return_tensors="pt")["pixel_values"].to(dev).half()
        e = model(pixel_values=x).last_hidden_state[:, 0] if vit else model.get_image_features(pixel_values=x)
        out.append(F.normalize(e.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def imaging(imgs):
    s = []
    for f in imgs:
        h, w = f.shape[:2]
        k = 512 / max(h, w)
        g = cv2.resize(f, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else f
        x = torch.from_numpy(g).permute(2, 0, 1)[None].float().div(255).to(dev)
        s.append(float(musiq(x)))
    return np.array(s)


@torch.no_grad()
def flow(imgs, bs=8):
    t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)
    t = F.interpolate(t.float(), scale_factor=0.5, mode="area").to(torch.uint8)
    s = []
    for i in range(0, len(t) - 1, bs):
        a, b = raft_tf(t[i:i + bs][:len(t[i + 1:i + 1 + bs])], t[i + 1:i + 1 + bs])
        f = raft(a.to(dev), b.to(dev))[-1]
        m = f.norm(dim=1).flatten(1)
        k = max(1, int(0.05 * m.shape[1]))
        s += m.topk(k, dim=1).values.mean(1).cpu().tolist()
    return np.array(s)


def consistency(e):
    if len(e) < 2:
        return np.nan
    return float(np.mean([(e[i] @ e[0] + e[i] @ e[i - 1]) / 2 for i in range(1, len(e))]))


videos = sorted(glob.glob(os.path.join(vdir, pat)))
assert videos, f"no videos matching {pat} in {vdir}"
for v in videos:
    out = v[:-4] + ".vbl.npz"
    if os.path.exists(out):
        continue
    fr = frames_of(v)
    nseg = (len(fr) - HOLD) // SEG
    i8, i4, i2 = np.arange(0, len(fr), 8), np.arange(0, len(fr), 4), np.arange(0, len(fr), 2)
    iq = imaging([fr[i] for i in i8])
    aes = (aes_head(torch.from_numpy(embed(clipL, procL, [fr[i] for i in i8])).to(dev)).squeeze(1) / 10).cpu()
    aes = aes.detach().numpy()
    eD = embed(dino, procD, [fr[i] for i in i4], vit=True)
    eB = embed(clipB, procB, [fr[i] for i in i4])
    fl = flow([fr[i] for i in i2])
    seg = {k: np.full(nseg, np.nan) for k in ("imaging", "aesthetic", "subject", "background", "flow")}
    for k in range(nseg):
        a, b = HOLD + k * SEG, HOLD + (k + 1) * SEG
        m8, m4, m2 = (i8 >= a) & (i8 < b), (i4 >= a) & (i4 < b), (i2[:-1] >= a) & (i2[1:] < b)
        seg["imaging"][k], seg["aesthetic"][k] = iq[m8].mean(), aes[m8].mean()
        seg["subject"][k], seg["background"][k] = consistency(eD[m4]), consistency(eB[m4])
        seg["flow"][k] = fl[m2].mean()
    tmp = f"{os.path.dirname(out)}/.{os.path.basename(out)}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **seg, iq_frames=i8, iq=iq, aes=aes, flow_frames=i2[:-1], flow_pairs=fl, n_frames=len(fr))
    os.replace(tmp, out)
    print(os.path.basename(v), {k: np.round(x, 3).tolist() for k, x in seg.items()}, flush=True)
open(done, "w").write("ok\n")
