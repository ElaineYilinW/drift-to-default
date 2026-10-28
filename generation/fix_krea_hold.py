"""Study 22 correction: krea_gen.py before 2026-10-27 19:04 UTC prepended 8 copies of the opening instead of 9, so the
first generated frame sat at frame 8 instead of 9 (the release does not output the start frame itself). Per-frame
statistics are computed frame by frame, so prepending one more copy of frame 0 (an opening copy) to every per-frame
array gives exactly what the corrected generator writes. Applies to image-to-video outputs (hold_frames = 9) whose
json says video_frames = 8 + raw_frames; marks the json (hold_fix). Fresh starts (hold_frames = 0) are untouched.
The mp4 files are left as they are (the tone analysis reads only the npz).
usage: python fix_krea_hold.py DIR [DIR ...]
"""
import glob
import json
import os
import sys
import time

import numpy as np

n = 0
for d in sys.argv[1:]:
    for j in sorted(glob.glob(f"{d}/*.json")):
        info = json.load(open(j))
        if info.get("model") != "krea_realtime_14b" or info.get("hold_frames") != 9 or info.get("hold_fix"):
            continue
        if info["video_frames"] != 8 + info["raw_frames"]:
            continue
        p = j[:-5] + ".npz"
        if not os.path.exists(p) or os.path.getmtime(p) < os.path.getmtime(j) or time.time() - os.path.getmtime(p) < 60:
            continue                                                 # the generator writes the json, then the npz
        z = dict(np.load(p))
        assert all(v.shape[0] == info["video_frames"] for v in z.values()), p
        z = {k: np.concatenate([v[:1], v]) for k, v in z.items()}
        np.savez(p[:-4] + ".fix.tmp.npz", **z)
        os.replace(p[:-4] + ".fix.tmp.npz", p)
        info.update(video_frames=info["video_frames"] + 1,
                    hold_fix="one opening copy prepended to the per-frame statistics post hoc (fix_krea_hold.py)")
        with open(j + ".tmp", "w") as f:
            json.dump(info, f, indent=1)
        os.replace(j + ".tmp", j)
        n += 1
print(n, "files fixed")
