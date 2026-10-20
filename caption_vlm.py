"""Matched-prompt experiment, step 1: caption each opening frame with Qwen2.5-VL-3B-Instruct (greedy) and build the
three prompt conditions per opening. G = the generic prompt of the earlier real-opening runs; M = scene + lighting
from the caption; ML = the generic scene with the caption's lighting sentence only.
usage: python caption_vlm.py OPENINGS_JSONL MODEL_DIR OUT_PROMPTS_JSONL
"""
import json
import re
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

src, mdir, out = sys.argv[1:4]
GENERIC = ("A front-facing dashcam video recorded from a car driving on a city road. Buildings, traffic and the street "
           "ahead are visible, and the camera moves forward steadily with natural lighting and realistic colors.")
Q = ("This is the first frame of a front-facing dashcam video. Describe it for a text-to-video prompt. Reply with JSON "
     "only, with two keys: \"scene\": a short phrase starting with \"on\" or \"through\" that describes the road and "
     "its surroundings (at most 15 words); \"lighting\": one sentence describing the time of day, how bright or dark the "
     "scene is, the sky or weather, and the main light sources (at most 25 words).")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mdir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
proc = AutoProcessor.from_pretrained(mdir, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
with open(out, "w") as f:
    for line in open(src):
        o = json.loads(line)
        inp = proc(text=[text], images=[Image.open(o["image"]).convert("RGB")], return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=160, do_sample=False)
        ans = proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        cap = json.loads(re.search(r"\{.*\}", ans, re.S).group(0))
        scene, light = cap["scene"].strip().rstrip("."), cap["lighting"].strip().rstrip(".")
        prompts = {
            "G": GENERIC,
            "M": (f"A front-facing dashcam video recorded from a car driving {scene}. {light}. "
                  "The camera moves forward steadily with realistic lighting and colors."),
            "ML": ("A front-facing dashcam video recorded from a car driving on a city road. Buildings, traffic and the "
                   f"street ahead are visible. {light}. The camera moves forward steadily with realistic lighting and colors."),
        }
        for k, p in prompts.items():
            f.write(json.dumps(dict(id=f"{o['id']}--{k}", open=o["id"], set=o["set"], cond=k, image=o["image"],
                                    caption=cap, prompt=p)) + "\n")
        print(o["id"], json.dumps(cap), flush=True)
