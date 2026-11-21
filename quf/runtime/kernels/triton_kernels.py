"""Int4 kernel support: Triton_Gate, Kernel_Layout repack, and a pure-torch layout reference.

Importing this module never imports Triton. ``triton_available()`` imports it lazily and only
when ``QUF_NO_TRITON`` is not ``1``. The W4A16 GEMV kernel is defined lazily by `int4_gemv`.

Kernel_Layout (in memory only, the QUF_File format is unchanged). For a weight with ``N`` output
rows and ``K`` input columns:

* ``packed``: int32 ``[N, K/8]``. Row ``n`` holds the 4-bit values of output ``n`` along the input
  dimension, so a kernel reads each output row contiguously. Nibble ``j`` (lsb first) of word
  ``w`` is input column ``8*w + j`` in kernel column order.
* ``scales``: fp16 ``[N, K/group_size]``.
* ``zeros``: fp16 ``[N, K/group_size]`` holding integer zero points 0..15, or ``None`` when the
  weight is symmetric (implicit zero 8).
* ``perm``: int64 ``[K]`` or ``None``. With act-order, kernel column ``i`` is original column
  ``perm[i]`` (a stable sort by ``g_idx``), so every group is contiguous. The activation must be
  permuted per call: ``x_kernel = x[..., perm]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from quf.quant.dequant import _unpack_last
from quf.quant.quantize import SYMMETRIC_ZERO, QuantizedWeight

SUPPORTED_GROUP_SIZES = (32, 64, 128)


def triton_available() -> tuple[bool, str]:
    """Triton_Gate: return ``(ok, reason)``. Never raises."""
    try:
        if os.environ.get("QUF_NO_TRITON", "") == "1":
            return False, "disabled by QUF_NO_TRITON=1"
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        try:
            import triton  # noqa: PLC0415
        except Exception as exc:  # ImportError or a broken install
            return False, f"triton import failed: {type(exc).__name__}: {exc}"
        return True, f"triton {getattr(triton, '__version__', 'unknown')}"
    except Exception as exc:  # pragma: no cover - defensive, the gate must not raise
        return False, f"triton gate error: {type(exc).__name__}: {exc}"


@dataclass
class Int4Layout:
    packed: torch.Tensor  # int32 [N, K/8]
    scales: torch.Tensor  # fp16 [N, K/group_size]
    zeros: torch.Tensor | None  # fp16 [N, K/group_size] or None (symmetric)
    perm: torch.Tensor | None  # int64 [K] or None
    group_size: int
    in_features: int
    out_features: int


def _pack_last(nib: torch.Tensor) -> torch.Tensor:
    """int values 0..15 ``[..., M * 8]`` -> int32 ``[..., M]`` (lsb first)."""
    n = nib.reshape(*nib.shape[:-1], nib.shape[-1] // 8, 8).to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64, device=nib.device) * 4
    word = (n << shifts).sum(dim=-1)  # 0 .. 2**32 - 1
    word = torch.where(word >= 2**31, word - 2**32, word)
    return word.to(torch.int32)


def act_order_perm(g_idx: torch.Tensor, group_size: int) -> torch.Tensor:
    """Stable permutation that makes every group contiguous and in group order."""
    idx = torch.as_tensor(g_idx).to(torch.int64)
    counts = torch.bincount(idx)
    if counts.numel() * group_size != idx.numel() or bool((counts != group_size).any()):
        raise ValueError("g_idx groups must each hold exactly group_size columns")
    return torch.argsort(idx, stable=True)


def repack_int4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    g_idx: torch.Tensor | None = None,
    group_size: int | None = None,
) -> Int4Layout:
    """Convert the canonical layout to the Kernel_Layout.

    ``qweight`` int32 ``[K/8, N]``, ``scales`` fp16 ``[G, N]``, ``qzeros`` int32 ``[G, N/8]`` or
    ``None``. ``group_size`` defaults to ``K // G``.
    """
    k = qweight.shape[0] * 8
    n = qweight.shape[1]
    if group_size is None:
        group_size = k // scales.shape[0]
    if scales.shape != (k // group_size, n) or k % group_size:
        raise ValueError("scales shape does not match qweight and group_size")

    q = _unpack_last(qweight.t().contiguous())  # [N, K] original column order
    perm = None
    if g_idx is not None and torch.as_tensor(g_idx).numel() > 0:
        if torch.as_tensor(g_idx).numel() != k:
            raise ValueError(f"g_idx has length {torch.as_tensor(g_idx).numel()}, expected {k}")
        perm = act_order_perm(torch.as_tensor(g_idx).to(qweight.device), group_size)
        q = q[:, perm]
    zeros = None
    if qzeros is not None:
        zeros = _unpack_last(qzeros).t().contiguous().to(torch.float16)  # [N, G]
    return Int4Layout(
        packed=_pack_last(q).contiguous(),
        scales=scales.t().contiguous().to(torch.float16),
        zeros=zeros,
        perm=perm,
        group_size=group_size,
        in_features=k,
        out_features=n,
    )


def repack_quantized_weight(qw: QuantizedWeight) -> Int4Layout:
    return repack_int4(qw.qweight, qw.scales, qw.qzeros, qw.g_idx, qw.group_size)


def dequantize_layout(layout: Int4Layout, original_order: bool = True) -> torch.Tensor:
    """Pure-torch dequantization from the Kernel_Layout, fp16 ``[N, K]``.

    With ``original_order`` the columns are returned in the canonical order, so the result
    equals ``reference_dequantize``. Otherwise the kernel column order is kept.
    """
    q = _unpack_last(layout.packed).to(torch.float32)  # [N, K]
    s = layout.scales.to(torch.float32).repeat_interleave(layout.group_size, dim=1)
    if layout.zeros is None:
        z = torch.full_like(s, SYMMETRIC_ZERO)
    else:
        z = layout.zeros.to(torch.float32).repeat_interleave(layout.group_size, dim=1)
    w = ((q - z) * s).to(torch.float16)
    if original_order and layout.perm is not None:
        out = torch.empty_like(w)
        out[:, layout.perm] = w
        w = out
    return w


def permute_activation(x: torch.Tensor, layout: Int4Layout) -> torch.Tensor:
    """Apply the act-order column permutation to the activation (no-op without ``perm``)."""
    return x if layout.perm is None else x[..., layout.perm]


# ---------------------------------------------------------------------------------------------
# W4A16 GEMV (decode shapes, M = 1..16). Triton is imported lazily on first use.
# ---------------------------------------------------------------------------------------------

MAX_GEMV_M = 16
_KERNEL = None
_KERNEL1 = None


def _build_kernel():
    import triton  # noqa: PLC0415
    import triton.language as tl  # noqa: PLC0415

    @triton.jit
    def _int4_gemv_kernel(
        x_ptr, w_ptr, s_ptr, z_ptr, out_ptr,
        M, N, K, K_PER_SPLIT,
        stride_xm,
        GROUP: tl.constexpr, HAS_ZEROS: tl.constexpr, SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        n_mask = offs_n < N
        m_mask = offs_m < M
        k_start = pid_s * K_PER_SPLIT
        k_end = tl.minimum(k_start + K_PER_SPLIT, K)
        k8 = K // 8
        ng = K // GROUP
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        BW: tl.constexpr = BLOCK_K // 8
        GW: tl.constexpr = GROUP // 8
        for k0 in range(k_start, k_end, BLOCK_K):
            offs_w = k0 // 8 + tl.arange(0, BW)  # packed word indices
            w_mask = offs_w < k_end // 8
            wmask = n_mask[:, None] & w_mask[None, :]
            word = tl.load(w_ptr + offs_n[:, None] * k8 + offs_w[None, :], mask=wmask, other=0)
            g = offs_w // GW
            s = tl.load(s_ptr + offs_n[:, None] * ng + g[None, :], mask=wmask, other=0.0).to(tl.float32)
            if HAS_ZEROS:
                z = tl.load(z_ptr + offs_n[:, None] * ng + g[None, :], mask=wmask, other=0.0).to(tl.float32)
            else:
                z = 8.0
            for j in tl.static_range(8):
                q = (word >> (4 * j)) & 15
                w = ((q.to(tl.float32) - z) * s).to(tl.float16)
                x = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + (offs_w * 8 + j)[None, :],
                    mask=m_mask[:, None] & w_mask[None, :], other=0.0,
                )
                acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        o_ptrs = out_ptr + pid_s * M * N + offs_m[:, None] * N + offs_n[None, :]
        o_mask = m_mask[:, None] & n_mask[None, :]
        if SPLIT == 1:
            tl.store(o_ptrs, acc.to(tl.float16), mask=o_mask)
        else:
            tl.store(o_ptrs, acc, mask=o_mask)

    @triton.jit
    def _int4_gemv1_kernel(
        x_ptr, w_ptr, s_ptr, z_ptr, perm_ptr, out_ptr,
        N, K, K_PER_SPLIT,
        GROUP: tl.constexpr, HAS_ZEROS: tl.constexpr, HAS_PERM: tl.constexpr, SPLIT: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # M == 1: out[n] = sum_g s[n,g] * (sum_k q[n,k] x[k] - z[n,g] * sum_k x[k]); no dequantised tile.
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        k_start = pid_s * K_PER_SPLIT
        k_end = tl.minimum(k_start + K_PER_SPLIT, K)
        k8 = K // 8
        ng = K // GROUP
        BW: tl.constexpr = BLOCK_K // 8
        GW: tl.constexpr = GROUP // 8
        shifts = tl.arange(0, 8) * 4
        acc = tl.zeros((BLOCK_N, BW), dtype=tl.float32)
        for k0 in range(k_start, k_end, BLOCK_K):
            offs_w = k0 // 8 + tl.arange(0, BW)
            w_mask = offs_w < k_end // 8
            wmask = n_mask[:, None] & w_mask[None, :]
            word = tl.load(w_ptr + offs_n[:, None] * k8 + offs_w[None, :], mask=wmask, other=0)
            g = offs_w // GW
            s = tl.load(s_ptr + offs_n[:, None] * ng + g[None, :], mask=wmask, other=0.0).to(tl.float32)
            if HAS_ZEROS:
                z = tl.load(z_ptr + offs_n[:, None] * ng + g[None, :], mask=wmask, other=0.0).to(tl.float32)
            else:
                z = 8.0
            d = tl.zeros((BLOCK_N, BW), dtype=tl.float32)
            xs = tl.zeros((BW,), dtype=tl.float32)
            for j in tl.static_range(8):
                offs_k = offs_w * 8 + j
                if HAS_PERM:
                    src = tl.load(perm_ptr + offs_k, mask=w_mask, other=0)
                else:
                    src = offs_k
                xj = tl.load(x_ptr + src, mask=w_mask, other=0.0).to(tl.float32)  # [BW]
                qj = ((word >> (4 * j)) & 15).to(tl.float32)
                d += qj * xj[None, :]
                xs += xj
            acc += s * (d - z * xs[None, :])
        res = tl.sum(acc, axis=1)  # [BN]
        o_ptrs = out_ptr + pid_s * N + offs_n
        if SPLIT == 1:
            tl.store(o_ptrs, res.to(tl.float16), mask=n_mask)
        else:
            tl.store(o_ptrs, res, mask=n_mask)

    return _int4_gemv_kernel, _int4_gemv1_kernel


_SM_COUNT: dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def gemv_config(n: int, k: int, group_size: int, sms: int) -> tuple[int, int, int, int, int]:
    """Static ``(BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)`` for an M = 1 decode matrix.

    A deterministic table keyed on the ``(N, K)`` bucket, tuned on an RTX 5080 with the sweep in
    ``bench_int4``. It is resolved on the host before any CUDA graph capture, so nothing is
    autotuned at run time. BLOCK_K is a multiple of the group size. Split-K (reduced without
    atomics) is used only for tall-K shapes that would otherwise launch too few programs.
    """
    if n <= 512:  # k/v projections: tiny, launch-latency bound
        cfg = (4, 512, 4 if k >= 2048 else 1, 4, 2)
    elif k >= 4096:  # down_proj
        cfg = (16, 512, 8, 2, 2) if n <= 2048 and k >= 8192 else (8, 1024, 1, 4, 2)
    elif n <= 1536:  # q/o
        cfg = (4, 512, 1, 1, 2)
    elif n <= 2048:  # fused qkv
        cfg = (8, 512, 1, 2, 2)
    elif n <= 6144:  # gate/up
        cfg = (8, 256, 1, 1, 2)
    elif n <= 9216:
        cfg = (16, 256, 1, 2, 2)
    else:  # fused gate_up
        cfg = (16, 256, 1, 1, 2)
    block_n, block_k, split, warps, stages = cfg
    block_k = max(block_k, group_size)
    block_k -= block_k % group_size
    k_blocks = -(-k // block_k)
    while split > 1 and k_blocks < split * 2:
        split //= 2
    return block_n, block_k, split, warps, stages


def int4_gemv(x: torch.Tensor, layout: Int4Layout, config: tuple[int, ...] | None = None) -> torch.Tensor:
    """``x @ W.T`` for fp16 ``x`` ``[M, K]`` (M = 1..16); returns fp16 ``[M, N]``.

    Accumulation is fp32. No host synchronisation, so it can be captured in a CUDA graph (run
    it once before capture to compile). ``config`` overrides ``(BLOCK_N, BLOCK_K, SPLIT_K)``,
    optionally followed by ``num_warps, num_stages``.
    """
    global _KERNEL, _KERNEL1
    import triton  # noqa: PLC0415

    if x.dim() != 2 or x.dtype != torch.float16 or not x.is_cuda:
        raise ValueError("x must be a CUDA fp16 tensor of shape [M, K]")
    m, k = x.shape
    n = layout.out_features
    if k != layout.in_features:
        raise ValueError(f"x has K={k}, layout expects {layout.in_features}")
    if not 1 <= m <= MAX_GEMV_M:
        raise ValueError(f"int4_gemv supports M in 1..{MAX_GEMV_M}, got {m}")
    if layout.group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(f"unsupported group size {layout.group_size}")
    if _KERNEL is None:
        _KERNEL, _KERNEL1 = _build_kernel()
    cfg = tuple(config or gemv_config(n, k, layout.group_size, _sm_count(x.device)))
    block_n, block_k, split = cfg[:3]
    warps, stages = (cfg[3], cfg[4]) if len(cfg) >= 5 else (4, 2)
    block_k = max(block_k, layout.group_size)
    k_per_split = -(-(-(-k // block_k)) // split) * block_k
    split = -(-k // k_per_split)
    zeros = layout.zeros if layout.zeros is not None else layout.scales
    out_dtype = torch.float16 if split == 1 else torch.float32
    if m == 1:
        x1 = x if x.is_contiguous() else x.contiguous()
        out = torch.empty((split, n), dtype=out_dtype, device=x.device)
        perm = layout.perm if layout.perm is not None else layout.scales
        _KERNEL1[(triton.cdiv(n, block_n), split)](
            x1, layout.packed, layout.scales, zeros, perm, out,
            n, k, k_per_split,
            GROUP=layout.group_size, HAS_ZEROS=layout.zeros is not None,
            HAS_PERM=layout.perm is not None, SPLIT=split,
            BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=warps, num_stages=stages,
        )
        return (out[0] if split == 1 else out.sum(0).to(torch.float16)).reshape(1, n)
    xk = permute_activation(x, layout)
    if xk.stride(1) != 1:
        xk = xk.contiguous()
    out = torch.empty((split, m, n), dtype=out_dtype, device=x.device)
    _KERNEL[(triton.cdiv(n, block_n), split)](
        xk, layout.packed, layout.scales, zeros, out,
        m, n, k, k_per_split, xk.stride(0),
        GROUP=layout.group_size, HAS_ZEROS=layout.zeros is not None, SPLIT=split,
        BLOCK_M=16, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4,
    )
    return out[0] if split == 1 else out.sum(0).to(torch.float16)
