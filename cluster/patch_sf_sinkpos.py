"""Sink-position patch for a SEPARATE copy of Self Forcing (Self-Forcing-b: a symlink tree of Self-Forcing in which
only wan/modules/causal_model.py is a real file), so that running jobs of the original tree are untouched.

Adds two environment switches, read at import, to the attention of the key-value-cache path (on top of patch_sf.py):
  SF_SINK_MODE   "off:N"  the sink's temporal RoPE position is moved N latent frames further back (query-sink distance
                          = natural distance + N); "dist:D" the sink is held at query-sink distance D (latent frames from
                          the newest query frame to the first sink frame) for every chunk after the opening.
                          Only queries outside the sink are affected (the opening's own context pass is unchanged), and
                          only a copy of the sink keys is rotated (the cache keeps the original keys), as in the
                          re-rotation of patch_sf.py, which is SF_SINK_MODE "dist:20" for chunks past the first window.
  SF_SINK_PROBE  json {"out": path.npz, "chunks": [start latent frames], "grid": [distances], "nq": 128}: for those
                          chunks, every attention call (4 denoising passes + 1 context pass, 30 layers) records the share
                          of attention that a fixed random subset of queries would give to the sink if the sink were at
                          each distance in the grid (sink logsumexp vs window logsumexp; read-only, no RNG drawn from
                          the global generators). Saved at exit as mass[chunk][call, grid, head].
Unset, the code path is identical to patch_sf.py's.
"""
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "wan/modules/causal_model.py"
s = open(p).read()
assert "patched: sink re-rotation" in s, "apply patch_sf.py first"

head_old = 'SINK_REROTATE = os.environ.get("SF_SINK_REROTATE") == "1"\n'
head_new = head_old + '''SINK_MODE = os.environ.get("SF_SINK_MODE", "")  # patched: sink position (patch_sf_sinkpos.py)
SINK_PROBE = os.environ.get("SF_SINK_PROBE", "")


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


_PROBE, _PROBE_CALLS, _PROBE_OUT = None, {}, {}
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

body_old = """            x = attention(
                roped_query,
                attn_k,"""
body_new = """            elif SINK_MODE and self.sink_size > 0 and self.local_attn_size != -1 and current_start_frame >= self.sink_size:
                kind, n = SINK_MODE.split(":")  # patched: sink position
                r = -int(n) if kind == "off" else current_end // frame_seqlen - 1 - int(n)
                if r != 0:
                    assert local_end_index <= self.max_attention_size
                    attn_k = attn_k.clone()
                    attn_k[:, :sink_tokens] = rotate_sink(attn_k[:, :sink_tokens], freqs, r)
            if _PROBE is not None and self.sink_size > 0 and current_start_frame >= self.sink_size:
                sink_probe(roped_query, kv_cache["k"][:, :sink_tokens], attn_k[:, sink_tokens:], freqs,
                           current_start_frame, current_end // frame_seqlen)
            x = attention(
                roped_query,
                attn_k,"""
if "patched: sink position" not in s:
    assert s.count(head_old) == 1 and s.count(body_old) == 1, "unexpected causal_model.py"
    s = s.replace(head_old, head_new).replace(body_old, body_new)
    open(p, "w").write(s)
print("patched", p)
