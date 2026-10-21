"""Calibrated-lighting experiment, step 1 (rule fixed before generation): pick each opening's lighting description from
its measured brightness instead of the VLM's words. Lighting classes and their sentences are the four Part 4 prompts,
whose zero-drift levels on Self Forcing were estimated on the earlier, independent 40 openings
(results/corr/light_out/light_summary.json). Each opening gets the class whose level is nearest to its opening
brightness (Self Forcing G-arm L_open, mean of seeds; the same value is used for both models).
  CL = generic scene ("on a city road") + calibrated lighting = exactly the Part 4 prompt of that class
  C  = VLM scene phrase (as in M) + calibrated lighting
usage: python make_calib_prompts.py MATCHED_JSONL LIGHT_SUMMARY MATCH_VIDEOS_CSV OUT_JSONL
"""
import json
import sys

import pandas as pd

matched, summ, vids, out = sys.argv[1:5]
LIGHT = {
    "sunny": ("on a bright, sunny midday", "Strong sunlight, a clear blue sky and vivid colours"),
    "overcast": ("on a dull, overcast day", "A grey sky, soft diffuse light and muted colours"),
    "dusk": ("at dusk", "The sky fades from orange to dark blue, street lights are switching on and the scene is dim"),
    "night": ("at night", "A dark sky, with street lights and headlights as the only light sources"),
}
level = {k: v["zero_drift_level"] for k, v in json.load(open(summ))["per_prompt"].items() if k in LIGHT}
V = pd.read_csv(vids)
L = V[(V.model == "SF") & (V.arm == "G")].groupby("open").L_open.mean()
rows = {}
for line in open(matched):
    d = json.loads(line)
    if d["cond"] in ("G", "M"):
        rows[(d["open"], d["cond"])] = d
with open(out, "w") as f:
    for o in sorted(L.index):
        cls = min(level, key=lambda k: abs(level[k] - L[o]))
        when, desc = LIGHT[cls]
        m = rows[(o, "M")]
        sc = m["prompt"].split("driving ", 1)[1].split(". ", 1)[0]          # the normalised VLM scene phrase used in M
        base = dict(open=o, set=m["set"], image=m["image"], L_open=round(float(L[o]), 2), light=cls, level=round(level[cls], 2))
        tail = f". {desc}; the camera moves forward steadily with realistic lighting."
        f.write(json.dumps(dict(base, id=f"{o}--CL", cond="CL", prompt=f"A front-facing dashcam video recorded from a car driving on a city road {when}{tail}")) + "\n")
        f.write(json.dumps(dict(base, id=f"{o}--C", cond="C", prompt=f"A front-facing dashcam video recorded from a car driving {sc}, {when}{tail}")) + "\n")
        f.write(json.dumps(dict(rows[(o, "G")], id=f"{o}--G")) + "\n")
        print(o, round(L[o], 1), cls)
