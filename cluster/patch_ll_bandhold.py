"""Band-hold patch for a SEPARATE copy of LongLive 1.0 (LongLive-d: a symlink tree of LongLive/LongLive1.0 whose
wan/modules/causal_model.py is a real copy of LongLive-c's, already patched by patch_ll.py, patch_ll_probe.py and
patch_ll_sinkpos.py), so that the LongLive-c tree is untouched. The LongLive counterpart of patch_sf_bandhold.py.

Adds rotate_sink_band and a third kind to LL_SINK_MODE of patch_ll_sinkpos.py:
  "band:D:j1,j2,..."  for every chunk after the opening, only the temporal RoPE frequencies j1, j2, ... (0..21; period
                      2*pi*10000^(2j/44) latent frames) of a copy of the sink keys are rotated so that the sink sits at
                      query-sink distance D; the other frequencies keep the sink's original positions. With all 22
                      indices it is "dist:D".
"off:N", "dist:D" and the probe are unchanged.
usage: python jobs/patch_ll_bandhold.py LongLive-d/wan/modules/causal_model.py
"""
import sys

p = sys.argv[1]
s = open(p).read()
assert "patched: sink position" in s, "apply patch_ll_sinkpos.py first"
assert "patched: band hold" not in s, "already patched"

head_old = 'SINK_MODE = os.environ.get("LL_SINK_MODE", "")  # patched: sink position (patch_ll_sinkpos.py)\n'
head_new = head_old + ('SINK_BAND = ([int(j) for j in SINK_MODE.split(":")[2].split(",")] if SINK_MODE.startswith("band:")\n'
                       '             else [])  # patched: band hold (patch_ll_bandhold.py)\n')
assert s.count(head_old) == 1
s = s.replace(head_old, head_new)

fn_old = "\n\n_PROBE, _PROBE_OUT = None, {}\n"
fn_new = '''

def rotate_sink_band(sk, freqs, r, band):
    """As rotate_sink, but only the temporal frequencies in band are rotated (patched: band hold)."""
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


_PROBE, _PROBE_OUT = None, {}
'''
assert s.count(fn_old) == 1, s.count(fn_old)
s = s.replace(fn_old, fn_new)

body_old = '''                    kind, n = SINK_MODE.split(":")
                    r = -int(n) if kind == "off" else current_end // frame_seqlen - 1 - int(n)
                    if r != 0:
                        k_sink = rotate_sink(k_sink, freqs, r)
'''
body_new = '''                    kind, n = SINK_MODE.split(":")[:2]
                    r = -int(n) if kind == "off" else current_end // frame_seqlen - 1 - int(n)
                    if r != 0:
                        if kind == "band":  # patched: band hold
                            k_sink = rotate_sink_band(k_sink, freqs, r, SINK_BAND)
                        else:
                            k_sink = rotate_sink(k_sink, freqs, r)
'''
assert s.count(body_old) == 1, s.count(body_old)
s = s.replace(body_old, body_new)
open(p, "w").write(s)
print("patched", p)
