"""Lower host-RAM peak of Self Forcing loading (Argon caps job RAM by slots): build the T5 encoder directly in
bf16 (the pipeline casts it to bf16 anyway) and memory-map the checkpoint instead of copying it."""
p = "utils/wan_wrapper.py"
s = open(p).read()
old = """            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )"""
new = """            dtype=torch.bfloat16,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False, mmap=True)
        )"""
if new not in s:
    assert old in s, "unexpected wan_wrapper.py"
    open(p, "w").write(s.replace(old, new))
print("patched", p)

# No flash-attn on Argon (glibc 2.17): Wan's cross-attention calls flash_attention() directly, which asserts.
# Fall back to torch SDPA with the key-padding mask (text context is padded to 512 tokens; k_lens marks real ones).
p = "wan/modules/attention.py"
s = open(p).read()
old = """    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256
"""
new = """    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):  # patched: SDPA fallback
        mask = None
        if k_lens is not None:
            mask = (torch.arange(k.size(1), device=k.device)[None, :] < k_lens.to(k.device)[:, None])[:, None, None, :]
        if q_scale is not None:
            q = q * q_scale
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).to(dtype), k.transpose(1, 2).to(dtype), v.transpose(1, 2).to(dtype),
            attn_mask=mask, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale)
        return out.transpose(1, 2).contiguous().type(q.dtype)

""" + old
if "patched: SDPA fallback" not in s:
    assert s.count(old) == 1, "unexpected attention.py"
    open(p, "w").write(s.replace(old, new))
print("patched", p)

# Optional sink re-rotation (env SF_SINK_REROTATE=1): Self Forcing keeps sink keys at their absolute RoPE positions,
# so the query-sink distance grows without bound. Re-rotated, the sink sits directly before the rolling window
# (as in Rolling Forcing / Deep Forcing). The cache keeps the original roped sink keys; at attention time only the
# temporal RoPE component of a copy is rotated by delta = max(0, end_frame - window), so no rounding accumulates.
p = "wan/modules/causal_model.py"
s = open(p).read()
old = """            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )"""
new = """            attn_k = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            if SINK_REROTATE and self.sink_size > 0 and self.local_attn_size != -1:  # patched: sink re-rotation
                delta = current_end // frame_seqlen - self.local_attn_size
                if delta > 0:
                    assert local_end_index <= self.max_attention_size
                    c = attn_k.size(3) // 2
                    ct = c - 2 * (c // 3)
                    rot = freqs[delta, :ct].to(attn_k.device)
                    sk = attn_k[:, :sink_tokens].to(torch.float64)
                    sink = torch.view_as_complex(sk.reshape(*sk.shape[:3], -1, 2)).clone()
                    sink[..., :ct] = sink[..., :ct] * rot
                    attn_k = attn_k.clone()
                    attn_k[:, :sink_tokens] = torch.view_as_real(sink).flatten(3).type_as(attn_k)
            x = attention(
                roped_query,
                attn_k,
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )"""
_buggy = """                    sink = torch.view_as_complex(attn_k[:, :sink_tokens].to(torch.float64).reshape(
                        *attn_k.shape[:3], -1, 2)).clone()"""
_fixed = """                    sk = attn_k[:, :sink_tokens].to(torch.float64)
                    sink = torch.view_as_complex(sk.reshape(*sk.shape[:3], -1, 2)).clone()"""
if _buggy in s:  # first version of this patch reshaped with the window's shape instead of the sink's
    open(p, "w").write(s.replace(_buggy, _fixed))
elif "patched: sink re-rotation" not in s:
    assert s.count(old) == 1, "unexpected causal_model.py"
    s = s.replace(old, new)
    s = s.replace("import math\n", "import math\nimport os\nSINK_REROTATE = os.environ.get(\"SF_SINK_REROTATE\") == \"1\"\n", 1)
    assert "SINK_REROTATE = " in s
    open(p, "w").write(s)
print("patched", p)
