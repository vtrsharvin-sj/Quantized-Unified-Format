"""GEMV over a Quantized_Embedding (task 34.3): ``logits = x @ W.T`` for M <= 16 rows of ``x``.

Importing this module imports Triton; callers go through the Triton_Gate first.

Reads the stored layout directly (see :mod:`quf.quant.embedding`): int4 bytes ``[N, K/2]`` with
column ``2j`` in the low nibble and zero point 8, or int8 ``[N, K]``, and fp16 scales
``[N, K/group]``. The activation is in natural column order (no act-order, no j-major scatter).
Accumulation is fp32 and the output is fp32 ``[M, N]``. One launch, no host sync, so it is
CUDA-graph capturable. ``N`` is the vocabulary (151936 for Qwen2.5), so the grid is very tall:
each program owns ``BLOCK_N`` rows and loops over K.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _embed_gemv_kernel(
    x_ptr, w_ptr, s_ptr, out_ptr,
    N, K,
    GROUP: tl.constexpr, BITS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.program_id(1)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    ng = K // GROUP
    HK: tl.constexpr = BLOCK_K // 2 if BITS == 4 else BLOCK_K
    offs_h = tl.arange(0, HK)
    acc = tl.zeros((BLOCK_N, HK), dtype=tl.float32)
    x_row = x_ptr + m * K
    for k0 in range(0, K, BLOCK_K):
        if BITS == 4:
            col_e = k0 + 2 * offs_h  # column of the low nibble; the high nibble is col_e + 1
            kmask = col_e < K
            b = tl.load(
                w_ptr + offs_n[:, None] * (K // 2) + (k0 // 2 + offs_h)[None, :],
                mask=n_mask[:, None] & kmask[None, :], other=0,
            ).to(tl.int32) & 255
            s = tl.load(
                s_ptr + offs_n[:, None] * ng + (col_e // GROUP)[None, :],
                mask=n_mask[:, None] & kmask[None, :], other=0.0,
            ).to(tl.float32)
            xe = tl.load(x_row + col_e, mask=kmask, other=0.0).to(tl.float32)
            xo = tl.load(x_row + col_e + 1, mask=kmask, other=0.0).to(tl.float32)
            lo = (b & 15).to(tl.float32) - 8.0
            hi = (b >> 4).to(tl.float32) - 8.0
            acc += s * (lo * xe[None, :] + hi * xo[None, :])
        else:
            col = k0 + offs_h
            kmask = col < K
            q = tl.load(
                w_ptr + offs_n[:, None] * K + col[None, :],
                mask=n_mask[:, None] & kmask[None, :], other=0,
            ).to(tl.float32)
            s = tl.load(
                s_ptr + offs_n[:, None] * ng + (col // GROUP)[None, :],
                mask=n_mask[:, None] & kmask[None, :], other=0.0,
            ).to(tl.float32)
            x = tl.load(x_row + col, mask=kmask, other=0.0).to(tl.float32)
            acc += s * q * x[None, :]
    res = tl.sum(acc, axis=1)
    tl.store(out_ptr + m * N + offs_n, res, mask=n_mask)


MAX_GEMV_ROWS = 16


def embed_gemv_config(bits: int, k: int) -> tuple[int, int, int]:
    """Static ``(BLOCK_N, BLOCK_K, num_warps)`` (swept on N = 151936, K = 896 with
    scripts/_embed_gemv_lab.py, RTX 5080 Laptop, L2 flushed per call)."""
    return (32, 128, 4)  # int4 ~0.18-0.23 ms, int8 ~0.29 ms cold for 151936 rows; ties within noise


def embed_gemv(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    out: torch.Tensor | None = None,
    config: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """fp32 ``[M, N]`` = ``x [M, K] @ W.T`` with ``M <= 16``; ``W`` the quantized table."""
    m, k = x.shape
    n = packed.shape[0]
    if m > MAX_GEMV_ROWS:
        raise ValueError(f"embed_gemv supports M <= {MAX_GEMV_ROWS}, got {m}")
    if not x.is_contiguous():
        x = x.contiguous()
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    block_n, block_k, warps = config or embed_gemv_config(bits, k)
    _embed_gemv_kernel[(triton.cdiv(n, block_n), m)](
        x, packed, scales, out, n, k,
        GROUP=group_size, BITS=bits, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=warps, num_stages=2,
    )
    return out
