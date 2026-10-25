"""Minimal stand-in for the flash_attn package (flash_attn_func, flash_attn_varlen_func) built on
torch.nn.functional.scaled_dot_product_attention. Same mathematics, no fused kernel. Used only to run Matrix-Game-2,
whose code imports flash_attn unconditionally, in an environment without it."""
import torch
import torch.nn.functional as F

__version__ = "0.0-sdpa-shim"


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **kw):
    """q: (B, Lq, H, D), k, v: (B, Lk, Hk, D) -> (B, Lq, H, D)."""
    assert window_size == (-1, -1), "local windows not supported by the shim"
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return o.transpose(1, 2).contiguous()


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0,
                           softmax_scale=None, causal=False, window_size=(-1, -1), deterministic=False, **kw):
    """q: (total_q, H, D), k, v: (total_k, Hk, D); sequences given by cumulative lengths -> (total_q, H, D)."""
    out = torch.empty_like(q)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        o = flash_attn_func(q[None, cq[i]:cq[i + 1]], k[None, ck[i]:ck[i + 1]], v[None, ck[i]:ck[i + 1]],
                            dropout_p, softmax_scale, causal, window_size)
        out[cq[i]:cq[i + 1]] = o[0]
    return out
