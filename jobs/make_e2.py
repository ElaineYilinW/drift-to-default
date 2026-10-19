"""E2 task list: correction arms on the development set (pilot prompts + brightness-transition controls), w21s3,
all on one GPU type so that arms are bit-identical until the first correction. usage: make_e2.py LUMA_NAME OUT.jsonl"""
import json
import sys

luma, out = sys.argv[1], sys.argv[2]
H, R = "/Users/ywang906/vdrift", "/nfsscratch/Users/ywang906/vdrift/corr/e2"
F = f"{H}/corr_files/{luma}.npz"
ARMS = {  # tag: extra args (closed-loop control on clean chunk outputs; see correct_gen.Controller)
    "none": "--method none",
    "luma_s3_a02": f"--method luma --at_step 3 --alpha 0.2 --trigger always --luma_dir {F}",
    "luma_s3_a02_j6": f"--method luma --at_step 3 --alpha 0.2 --trigger always --jump 6 --luma_dir {F}",
    "luma_s3_a02_j3": f"--method luma --at_step 3 --alpha 0.2 --trigger always --jump 3 --luma_dir {F}",
    "luma_s3_a05_j3": f"--method luma --at_step 3 --alpha 0.5 --trigger always --jump 3 --luma_dir {F}",
    "luma_s1_a02_j3": f"--method luma --at_step 1 --alpha 0.2 --trigger always --jump 3 --luma_dir {F}",
    "luma_s2_a02_j3": f"--method luma --at_step 2 --alpha 0.2 --trigger always --jump 3 --luma_dir {F}",
    "luma_s3_cusum_j3": f"--method luma --at_step 3 --trigger cusum:0.5:4 --jump 3 --luma_dir {F}",
    "luma_s3_every4_j3": f"--method luma --at_step 3 --trigger every:4 --jump 3 --luma_dir {F}",
}
sets = [("pilot.jsonl", [json.loads(l)["id"] for l in open("prompts/pilot.jsonl")]),
        ("controls.jsonl", [json.loads(l)["id"] for l in open("prompts/controls.jsonl")])]
with open(out, "w") as f:
    for pfile, ids in sets:
        for pid in ids:
            for tag, extra in ARMS.items():
                cmd = (f"source {H}/jobs/env.sh && cd $H/Self-Forcing && $PY $H/correct_gen.py --prompts $H/prompts/{pfile} "
                       f"--prompt_id {pid} --tag {tag} {extra} --seeds 0,1 --outdir {R}")
                f.write(json.dumps(dict(id=f"e2_{pid}_{tag}", cmd=cmd, done=f"{R}/{pid}_{tag}_s1.npz", est_min=11,
                                        mem_gb=30, gpus=1, gpu_mem_gb=40, gpu_bf16=True, cores=16, gpu_only="a40")) + "\n")
print(sum(1 for _ in open(out)), "tasks")
