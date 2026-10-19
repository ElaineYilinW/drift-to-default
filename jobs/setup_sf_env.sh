#!/bin/bash
# Self Forcing (Wan2.1-T2V-1.3B) environment + weights on Argon (CentOS 7, glibc 2.17).
# - torch 2.5.1 (needs >=2.5 for flex_attention; 2.6+ wheels need glibc 2.28)
# - pillow/av/etc. from conda-forge (see below)
# - transformers pinned: newer releases disable themselves on torch < 2.6
# - no flash-attn: Wan falls back to torch SDPA
# - training/demo-only deps (tensorrt, pycuda, onnx, flask, wandb, dashscope) skipped
set -ex
H=/Users/ywang906/vdrift
export MAMBA_ROOT_PREFIX=/Users/ywang906/wmsel/mamba
MM=/Users/ywang906/wmsel/bin/micromamba
E=$H/envs/sf
[ -x $E/bin/python ] || $MM create -y -p $E -c conda-forge python=3.10 pip
# compiled deps from conda-forge: the PyTorch index only has a pillow sdist, and new PyPI wheels need glibc >= 2.28
$MM install -y -p $E -c conda-forge pillow numpy=1.24.4 av=13.1.0 sentencepiece tokenizers lmdb scikit-image matplotlib
P=$E/bin/python
$P -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$P -m pip install --no-cache-dir --prefer-binary diffusers==0.31.0 transformers==4.49.0 "tokenizers>=0.20.3" \
  "accelerate>=1.1.1" tqdm imageio imageio-ffmpeg easydict ftfy numpy==1.24.4 omegaconf einops av==13.1.0 \
  opencv-python-headless==4.9.0.80 open_clip_torch lmdb matplotlib sentencepiece pydantic==2.10.6 scikit-image \
  "huggingface_hub[cli]" pandas
# metrics (features.py): pyarrow from conda-forge (no glibc-2.17 wheel), pyiqa with the stack pinned; pyiqa pulls
# opencv-python (needs libGL) and a newer av (breaks torchvision.write_video) -> restore headless OpenCV and av 13.1
$MM install -y -p $E -c conda-forge pyarrow
printf "torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\ntransformers==4.49.0\nnumpy==1.24.4\n" > /tmp/sf_constraints.txt
$P -m pip install --no-cache-dir --prefer-binary -c /tmp/sf_constraints.txt pyiqa
$P -m pip uninstall -y opencv-python && $P -m pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.9.0.80
$MM install -y -p $E -c conda-forge av=13.1.0
cd $H/Self-Forcing && $P -m pip install --no-deps -e . && $P $H/jobs/patch_sf.py
export LD_LIBRARY_PATH=$E/lib
$P -c "import torch, diffusers, transformers, av, cv2; from torch.nn.attention.flex_attention import flex_attention; print('OK', torch.__version__)"
HF=$E/bin/hf  # huggingface-cli was renamed to hf
[ -f wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors ] || \
  $HF download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
[ -f checkpoints/self_forcing_dmd.pt ] || $HF download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
du -sh wan_models checkpoints
touch $H/env_sf.done
