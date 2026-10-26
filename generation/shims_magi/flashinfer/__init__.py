"""Minimal stand-in for flashinfer, enough for MAGI-1 inference without FP8: activation.silu_and_mul (same mathematics,
no fused kernel); gemm.bmm_fp8 is only used with fp8_quant and raises."""
from . import activation, gemm  # noqa: F401
