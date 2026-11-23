"""Fused decode kernels (task 33.2): residual-add + RMSNorm.

Importing this module imports Triton, so callers go through the Triton_Gate first
(``fused_decode_available``) and never import it when ``QUF_NO_TRITON=1``.

The kernels are CUDA-graph capturable: no host sync, no allocation inside the kernel, fixed
launch grid (one program per row).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(
    x_ptr, res_ptr, w_ptr, normed_ptr, newres_ptr,
    idx_ptr, n_cols, eps,
    HAS_RES: tl.constexpr, HAS_IDX: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * n_cols
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    if HAS_RES:
        r = tl.load(res_ptr + base + offs, mask=mask, other=0.0)
        s = x + r  # fp16 add, rounded to fp16 exactly like the torch path `h + out`
        tl.store(newres_ptr + base + offs, s, mask=mask)
    else:
        s = x
    sf = s.to(tl.float32)
    var = tl.sum(sf * sf, axis=0) / n_cols
    y = (sf * tl.rsqrt(var + eps)).to(x.dtype)  # cast to the compute dtype like `xf.to(x.dtype)`
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    outv = w * y
    if HAS_IDX:  # scatter into the consumer's j-major activation layout (decode_gemv)
        dest = tl.load(idx_ptr + offs, mask=mask, other=0)
        tl.store(normed_ptr + base + dest, outv, mask=mask)
    else:
        tl.store(normed_ptr + base + offs, outv, mask=mask)


def add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    out_normed: torch.Tensor | None = None,
    out_residual: torch.Tensor | None = None,
    out_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One pass over each row: ``new_residual = x + residual`` (fp16) and
    ``normed = weight * rmsnorm(new_residual)`` with fp32 reduction.

    ``residual`` None means no add: ``new_residual`` is ``x`` itself and only the norm runs.
    Returns ``(normed, new_residual)`` with the shape of ``x``.
    """
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    rows = x2.shape[0]
    normed = torch.empty_like(x2) if out_normed is None else out_normed.view(rows, n)
    has_res = residual is not None
    if has_res:
        r2 = residual.reshape(-1, n)
        new_res = torch.empty_like(x2) if out_residual is None else out_residual.view(rows, n)
    else:
        r2, new_res = x2, x2
    block = triton.next_power_of_2(n)
    _add_rmsnorm_kernel[(rows,)](
        x2, r2, weight, normed, new_res, out_index if out_index is not None else x2, n, float(eps),
        HAS_RES=has_res, HAS_IDX=out_index is not None, BLOCK=block, num_warps=4 if block <= 2048 else 8,
    )
    return normed.view(x.shape), new_res.view(x.shape)
