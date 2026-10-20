import pickle, glob, os, json, numpy as np
from PIL import Image
B = "/nfsscratch/Users/ywang906/wmsel/dataset"
used = {json.loads(l)["log"] for l in open("/Users/ywang906/vdrift/prompts/real.jsonl")}
def Lstar(img):
    x = np.asarray(img.convert("RGB").resize((320, 180)), dtype=np.float64) / 255
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    Y = x @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(Y > 216 / 24389, np.cbrt(Y), (24389 / 27 * Y + 16) / 116)
    return float((116 * f - 16).mean())
out = []
for p in sorted(glob.glob(f"{B}/navsim_logs/test/*.pkl")):
    log = os.path.basename(p)[:-4]
    fr = pickle.load(open(p, "rb"))
    if len(fr) <= 20: continue
    rel = fr[20]["cams"]["CAM_F0"]["data_path"]
    ip = f"{B}/sensor_blobs/test/{rel}"
    if not os.path.exists(ip): continue
    out.append(dict(log=log, used=log in used, img=ip, L=round(Lstar(Image.open(ip)), 1), loc=fr[20].get("map_location", "")))
json.dump(out, open("/Users/ywang906/vdrift/corr_files/navtest_scan.json", "w"), indent=0)
print(len(out), sum(o["used"] for o in out)); Ls = sorted(o["L"] for o in out if not o["used"]); print(Ls[:10], Ls[-5:])
