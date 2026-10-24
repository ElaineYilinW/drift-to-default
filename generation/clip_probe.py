"""CLIP ViT-L/14 embeddings of video frames for the content-relaxation study (docs/preregistration_content_cn.md).

For every video matching the glob: frames 1..8 (the held opening) and every 4th frame from 9 on, embedded with
openai/clip-vit-large-patch14 (standard CLIP preprocessing: shorter side to 224, centre crop), L2-normalised, fp16.
A second set of embeddings is taken after tone normalisation: each frame's CIELAB mean and sd are matched to those of
the video's held opening (mean over frames 1..8), so that global brightness and colour cannot carry the class score.
Fresh starts (no held opening) get no tone-normalised set. One {stem}.clip.npz per video; existing ones are skipped.
The text embeddings of the seven class labels are written once to OUT_TEXT (labels are part of the preregistration).
usage: python clip_probe.py "GLOB" OUT_TEXT [DONE_FILE]
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

LABELS = {
    "city": "a dashcam photo of a city street with buildings and traffic",
    "snow": "a dashcam photo of a snow-covered road in winter",
    "desert": "a dashcam photo of a desert highway with sand dunes",
    "forest": "a dashcam photo of a road through a dense green forest",
    "coast": "a dashcam photo of a coastal road along the sea",
    "mountain": "a dashcam photo of a winding mountain road with rocky cliffs",
    "farm": "a dashcam photo of a rural road through farmland",
}
NAME = "openai/clip-vit-large-patch14"
dev = torch.device("cuda")
model = CLIPModel.from_pretrained(NAME, torch_dtype=torch.float16).to(dev).eval()
proc = CLIPProcessor.from_pretrained(NAME)


@torch.no_grad()
def embed_images(frames, bs=64):
    out = []
    for i in range(0, len(frames), bs):
        px = proc(images=frames[i:i + bs], return_tensors="pt")["pixel_values"].to(dev, torch.float16)
        e = model.get_image_features(pixel_values=px).float()
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float16)


@torch.no_grad()
def embed_text():
    tok = proc(text=list(LABELS.values()), return_tensors="pt", padding=True).to(dev)
    e = model.get_text_features(**tok).float()
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy(), float(model.logit_scale.exp())


def frames_of(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def lab_stats(frames):
    lab = np.concatenate([cv2.cvtColor(f.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3) for f in frames])
    return lab.mean(0), lab.std(0) + 1e-6


def tone_match(f, mu, sd):
    lab = cv2.cvtColor(f.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    lab = (lab - m) / s * sd + mu
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    return (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255.0).round().astype(np.uint8)


if __name__ == "__main__":
    pattern, out_text = sys.argv[1], sys.argv[2]
    if not os.path.exists(out_text):
        T, scale = embed_text()
        np.savez(out_text, text=T, labels=np.array(list(LABELS)), prompts=np.array(list(LABELS.values())), scale=scale,
                 model=NAME)
    for v in sorted(glob.glob(pattern)):
        out = v[:-4] + ".clip.npz"
        if os.path.exists(out):
            continue
        fr = frames_of(v)
        idx = np.array(sorted(set(range(1, 9)) | set(range(9, len(fr), 4))))
        fresh = os.path.basename(v).startswith("fresh")
        emb = embed_images([fr[i] for i in idx])
        res = dict(idx=idx, emb=emb, n_frames=len(fr))
        if not fresh:
            mu, sd = lab_stats(fr[1:9])
            res["emb_tn"] = embed_images([tone_match(fr[i], mu, sd) for i in idx])
        np.savez(out, **res)
        print(os.path.basename(v), len(fr), len(idx), flush=True)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write("ok\n")
