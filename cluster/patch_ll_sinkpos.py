"""Sink-position patch for a SEPARATE copy of LongLive 1.0 (LongLive-c: a symlink tree of LongLive/LongLive1.0 in
which only wan/modules/causal_model.py is a real file), applied on top of jobs/patch_ll_probe.py (which defines
rotate_sink). The switch is the one of patch_sf_sinkpos.py for Self Forcing:
  LL_SINK_MODE   "off:N"  the sink's temporal RoPE position is moved N latent frames further back (query-sink distance
                          = natural distance + N); "dist:D" the sink is held at query-sink distance D (latent frames from
                          the newest query frame to the first sink frame) for every chunk after the opening.
                          Only queries outside the sink are affected (the opening's own context pass is unchanged), and
                          only a copy of the sink keys is rotated (the cache keeps the original keys).
Unset, the code path is unchanged.
usage: python jobs/patch_ll_sinkpos.py LongLive-c/wan/modules/causal_model.py
"""
import sys

p = sys.argv[1]
s = open(p).read()
assert "patched: sink probe" in s, "apply patch_ll_probe.py first"

head_old = 'SINK_PROBE = os.environ.get("LL_SINK_PROBE", "")\n'
head_new = head_old + 'SINK_MODE = os.environ.get("LL_SINK_MODE", "")  # patched: sink position (patch_ll_sinkpos.py)\n'

body_old = """                k_sink = temp_k[:, :sink_tokens]
                v_sink = temp_v[:, :sink_tokens]
"""
body_new = body_old + """                if SINK_MODE and current_start_frame >= self.sink_size:  # patched: sink position
                    kind, n = SINK_MODE.split(":")
                    r = -int(n) if kind == "off" else current_end // frame_seqlen - 1 - int(n)
                    if r != 0:
                        k_sink = rotate_sink(k_sink, freqs, r)
"""
if "patched: sink position" not in s:
    assert s.count(head_old) == 1 and s.count(body_old) == 1, "unexpected causal_model.py"
    s = s.replace(head_old, head_new).replace(body_old, body_new)
    open(p, "w").write(s)
print("patched", p)
