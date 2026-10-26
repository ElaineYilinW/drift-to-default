import torch.nn.functional as F


def silu_and_mul(x, out=None):
    """silu(x[..., :d]) * x[..., d:], d = x.shape[-1] // 2 (flashinfer.activation.silu_and_mul)."""
    d = x.shape[-1] // 2
    y = F.silu(x[..., :d]) * x[..., d:]
    if out is not None:
        out.copy_(y)
        return out
    return y
