"""Runtime patches for LongLive 1.0 on Argon (same as patch_sf.py, minus sink re-rotation): T5 encoder loaded in bf16
with mmap (host-RAM cap), and an SDPA fallback with key-padding mask in flash_attention() (no flash-attn on glibc 2.17).
Each patch is applied only if its target text is found; run from LongLive1.0/."""
import re

p = "utils/wan_wrapper.py"
s = open(p).read()
if "dtype=torch.bfloat16,\n            device=torch.device('cpu')" not in s:
    s2 = s.replace("dtype=torch.float32,\n            device=torch.device('cpu')", "dtype=torch.bfloat16,\n            device=torch.device('cpu')", 1)
    s2 = re.sub(r"(models_t5_umt5-xxl-enc-bf16\.pth\",\s*\n\s*map_location='cpu', weights_only=False)\)", r"\1, mmap=True)", s2, count=1)
    assert s2 != s, "wan_wrapper patch target not found"
    open(p, "w").write(s2)
print("patched", p)

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
    assert s.count(old) == 1, "attention patch target not found"
    open(p, "w").write(s.replace(old, new))
print("patched", p)
