"""Run the frozen sky_gen_x.py unchanged, with diffusers' final video post-processing on the CPU (study 16, deviation).

At 60 s the decoded SkyReels video (about 8.5 GB in fp32) no longer fits on a 48-GB GPU twice, and the pipeline's last
step, VideoProcessor.postprocess_video (x * 0.5 + 0.5, clamp to [0, 1], conversion to numpy), runs out of memory. This
wrapper moves the decoded tensor to the CPU before that step; the operations are the same elementwise float32 arithmetic.
usage: python sky_gen_x_cpupost.py /path/to/sky_gen_x.py ARGS...
"""
import runpy
import sys

import torch
from diffusers.video_processor import VideoProcessor

_orig = VideoProcessor.postprocess_video


def _postprocess_on_cpu(self, video, output_type="np"):
    if torch.is_tensor(video):
        video = video.cpu()
        torch.cuda.empty_cache()
    return _orig(self, video, output_type)


VideoProcessor.postprocess_video = _postprocess_on_cpu
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
