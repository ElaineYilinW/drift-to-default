"""Band probe patch for a THIRD copy of Self Forcing (Self-Forcing-c: a symlink tree of Self-Forcing whose
wan/modules/causal_model.py is a real copy of Self-Forcing-b's, already patched by patch_sf.py and patch_sf_sinkpos.py),
so that running jobs of the -b tree are untouched.

Exploratory: which temporal RoPE frequencies make the peaks of the attention-to-sink probe? Adds two optional keys to the
SF_SINK_PROBE json of patch_sf_sinkpos.py:
  "bands"  list of lists of temporal frequency indices j (0..21; period 2*pi*10000^(2j/44) latent frames)
  "ref"    reference query-sink distance (latent frames)
With "bands", the sink keys are first moved to distance ref; then, for each band and each grid distance D, only the
frequencies of the band are rotated on to distance D (the others stay at ref), and the probe's share is recorded as
before. Saved as mass[chunk][call, band, grid, head]. A band with all 22 indices reproduces the plain probe (up to bf16 rounding); an empty
band gives the share at distance ref for every D. Without "bands" the probe and generation are unchanged.
"""
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "wan/modules/causal_model.py"
s = open(p).read()
assert "patched: sink position" in s, "apply patch_sf_sinkpos.py first"

f_old = '''def sink_probe(q, sink_k, win_k, freqs, start_frame, end_frame):
    if start_frame not in _PROBE["chunks"]:
        return
    g = torch.Generator(device=q.device).manual_seed(0)
    idx = torch.randperm(q.shape[1], generator=g, device=q.device)[:_PROBE.get("nq", 128)]
    qs = q[:, idx].float()
    scale = qs.shape[-1] ** -0.5
    lse_w = torch.logsumexp(torch.einsum("bqhd,bkhd->bhqk", qs, win_k.float()) * scale, dim=-1)
'''
f_new = '''def rotate_sink_band(sk, freqs, r, band):
    """As rotate_sink, but only the temporal frequencies in band are rotated (patched: band probe)."""
    c = sk.size(3) // 2
    ct = c - 2 * (c // 3)
    rot = freqs[abs(r), :ct].to(sk.device).clone()
    if r < 0:
        rot = rot.conj()
    keep = torch.ones(ct, dtype=torch.bool, device=sk.device)
    keep[list(band)] = False
    rot[keep] = 1
    s = torch.view_as_complex(sk.to(torch.float64).reshape(*sk.shape[:3], -1, 2)).clone()
    s[..., :ct] = s[..., :ct] * rot
    return torch.view_as_real(s).flatten(3).type_as(sk)


def sink_probe(q, sink_k, win_k, freqs, start_frame, end_frame):
    if start_frame not in _PROBE["chunks"]:
        return
    g = torch.Generator(device=q.device).manual_seed(0)
    idx = torch.randperm(q.shape[1], generator=g, device=q.device)[:_PROBE.get("nq", 128)]
    qs = q[:, idx].float()
    scale = qs.shape[-1] ** -0.5
    lse_w = torch.logsumexp(torch.einsum("bqhd,bkhd->bhqk", qs, win_k.float()) * scale, dim=-1)
    if _PROBE.get("bands"):  # patched: band probe
        ref = int(_PROBE["ref"])
        base = rotate_sink(sink_k.to(torch.float64), freqs, end_frame - 1 - ref)
        out = []
        for band in _PROBE["bands"]:
            res = []
            for D in _PROBE["grid"]:
                sk = rotate_sink_band(base, freqs, ref - D, band).float() if band else base.float()
                lse_s = torch.logsumexp(torch.einsum("bqhd,bkhd->bhqk", qs, sk) * scale, dim=-1)
                res.append(torch.sigmoid(lse_s - lse_w).mean(-1)[0])
            out.append(torch.stack(res))
        _PROBE_OUT.setdefault(start_frame, []).append(torch.stack(out).cpu().numpy())
        return
'''
if "patched: band probe" not in s:
    assert s.count(f_old) == 1, "unexpected causal_model.py"
    s = s.replace(f_old, f_new)
    open(p, "w").write(s)
print("patched", p)
