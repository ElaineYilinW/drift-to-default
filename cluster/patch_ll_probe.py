"""Attention-to-sink probe for a SEPARATE copy of LongLive 1.0 (LongLive-b: a symlink tree of LongLive/LongLive1.0 in
which only wan/modules/causal_model.py is a real file), so that running LongLive jobs are untouched. Same probe as
patch_sf_sinkpos.py's SF_SINK_PROBE (read-only, no RNG drawn from the global generators):
  LL_SINK_PROBE  json {"out": path.npz, "chunks": [start latent frames], "grid": [distances], "nq": 128}: for those
                 chunks, every attention call of the cache path (4 denoising passes + 1 context pass, 30 layers) records
                 the share of attention that a fixed random subset of queries would give to the sink if the sink were
                 at each query-sink distance in the grid (sink logsumexp vs window logsumexp). Saved at exit as
                 mass[chunk][call, grid, head].
Unset, the code path is unchanged.
usage: python jobs/patch_ll_probe.py LongLive-b/wan/modules/causal_model.py
"""
import sys

p = sys.argv[1]
s = open(p).read()

head_old = "from utils.debug_option import DEBUG\n"
head_new = head_old + '''import os  # patched: sink probe (patch_ll_probe.py)
SINK_PROBE = os.environ.get("LL_SINK_PROBE", "")


def rotate_sink(sk, freqs, r):
    """Move roped sink keys r latent frames along the temporal RoPE axis (r < 0: further back)."""
    c = sk.size(3) // 2
    ct = c - 2 * (c // 3)
    rot = freqs[abs(r), :ct].to(sk.device)
    if r < 0:
        rot = rot.conj()
    s = torch.view_as_complex(sk.to(torch.float64).reshape(*sk.shape[:3], -1, 2)).clone()
    s[..., :ct] = s[..., :ct] * rot
    return torch.view_as_real(s).flatten(3).type_as(sk)


_PROBE, _PROBE_OUT = None, {}
if SINK_PROBE:
    import atexit
    import json as _json
    import numpy as _np
    _PROBE = _json.loads(SINK_PROBE)

    def _probe_save():
        if _PROBE_OUT:
            _np.savez(_PROBE["out"], grid=_np.array(_PROBE["grid"]), chunks=_np.array(sorted(_PROBE_OUT)),
                      **{f"c{k}": _np.stack(v) for k, v in _PROBE_OUT.items()})
    atexit.register(_probe_save)


def sink_probe(q, sink_k, win_k, freqs, start_frame, end_frame):
    if start_frame not in _PROBE["chunks"]:
        return
    g = torch.Generator(device=q.device).manual_seed(0)
    idx = torch.randperm(q.shape[1], generator=g, device=q.device)[:_PROBE.get("nq", 128)]
    qs = q[:, idx].float()
    scale = qs.shape[-1] ** -0.5
    lse_w = torch.logsumexp(torch.einsum("bqhd,bkhd->bhqk", qs, win_k.float()) * scale, dim=-1)
    res = []
    for D in _PROBE["grid"]:
        sk = rotate_sink(sink_k, freqs, end_frame - 1 - D).float()
        lse_s = torch.logsumexp(torch.einsum("bqhd,bkhd->bhqk", qs, sk) * scale, dim=-1)
        res.append(torch.sigmoid(lse_s - lse_w).mean(-1)[0])
    _PROBE_OUT.setdefault(start_frame, []).append(torch.stack(res).cpu().numpy())
'''

body_old = """                x = attention(
                    roped_query,
                    k_cat,
                    v_cat
                )"""
body_new = """                if _PROBE is not None and local_budget > 0 and current_start_frame >= self.sink_size:
                    sink_probe(roped_query, k_sink, k_local, freqs, current_start_frame, current_end // frame_seqlen)
                x = attention(
                    roped_query,
                    k_cat,
                    v_cat
                )"""
if "patched: sink probe" not in s:
    assert s.count(head_old) == 1 and s.count(body_old) == 1, "unexpected causal_model.py"
    s = s.replace(head_old, head_new).replace(body_old, body_new)
    open(p, "w").write(s)
print("patched", p)
