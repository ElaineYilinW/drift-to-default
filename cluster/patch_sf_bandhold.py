"""Band-hold patch for a FOURTH copy of Self Forcing (Self-Forcing-d: a symlink tree of Self-Forcing whose
wan/modules/causal_model.py is a real copy of Self-Forcing-c's, already patched by patch_sf.py, patch_sf_sinkpos.py and
patch_sf_sinkband.py), so that the -b and -c trees are untouched.

Adds a third kind to SF_SINK_MODE of patch_sf_sinkpos.py:
  "band:D:j1,j2,..."  for every chunk after the opening, only the temporal RoPE frequencies j1, j2, ... (0..21; period
                      2*pi*10000^(2j/44) latent frames) of a copy of the sink keys are rotated so that the sink sits at
                      query-sink distance D; the other frequencies keep the sink's original positions. With all 22
                      indices it is "dist:D"; it is the generation counterpart of the band probe's "all but" bands
                      (the band held at the reference distance, the others at the natural distance).
"off:N" and "dist:D" and the probe are unchanged.
"""
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "wan/modules/causal_model.py"
s = open(p).read()
assert "patched: band probe" in s, "apply patch_sf_sinkband.py first"
assert "patched: band hold" not in s, "already patched"

head_old = 'SINK_MODE = os.environ.get("SF_SINK_MODE", "")  # patched: sink position (patch_sf_sinkpos.py)\n'
head_new = head_old + ('SINK_BAND = ([int(j) for j in SINK_MODE.split(":")[2].split(",")] if SINK_MODE.startswith("band:")\n'
                       '             else [])  # patched: band hold (patch_sf_bandhold.py)\n')
assert s.count(head_old) == 1
s = s.replace(head_old, head_new)

body_old = '''                kind, n = SINK_MODE.split(":")  # patched: sink position
                r = -int(n) if kind == "off" else current_end // frame_seqlen - 1 - int(n)
                if r != 0:
                    assert local_end_index <= self.max_attention_size
                    attn_k = attn_k.clone()
                    attn_k[:, :sink_tokens] = rotate_sink(attn_k[:, :sink_tokens], freqs, r)
'''
body_new = '''                kind, n = SINK_MODE.split(":")[:2]  # patched: sink position
                r = -int(n) if kind == "off" else current_end // frame_seqlen - 1 - int(n)
                if r != 0:
                    assert local_end_index <= self.max_attention_size
                    attn_k = attn_k.clone()
                    if kind == "band":  # patched: band hold
                        attn_k[:, :sink_tokens] = rotate_sink_band(attn_k[:, :sink_tokens], freqs, r, SINK_BAND)
                    else:
                        attn_k[:, :sink_tokens] = rotate_sink(attn_k[:, :sink_tokens], freqs, r)
'''
assert s.count(body_old) == 1, s.count(body_old)
s = s.replace(body_old, body_new)
open(p, "w").write(s)
print("patched", p)
