"""Make krea-ai/realtime-video run on torch 2.6 (Argon's CentOS 7 glibc has no torch 2.8 wheels). Idempotent.
  release_server.py: torch._dynamo.config.recompile_limit is called cache_size_limit before torch 2.7;
  wan/modules/causal_model.py: drop the flex_attention kernel option BLOCKS_ARE_CONTIGUOUS (a speed hint unknown to 2.6).
usage: python patch_krea.py REPO
"""
import sys

repo = sys.argv[1]
EDITS = {"release_server.py": [("dynamo.config.recompile_limit = 32",
                                "setattr(dynamo.config, 'recompile_limit' if hasattr(dynamo.config, 'recompile_limit') "
                                "else 'cache_size_limit', 32)  # patched for torch 2.6")],
         "wan/modules/causal_model.py": [('"BLOCKS_ARE_CONTIGUOUS": True,', '# "BLOCKS_ARE_CONTIGUOUS": True,  # patched for torch 2.6')]}
for f, eds in EDITS.items():
    p = f"{repo}/{f}"
    s = open(p).read()
    for old, new in eds:
        if new in s:
            continue
        assert s.count(old) == 1, (f, old)
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", f)
