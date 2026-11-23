"""Decode attention kernels (task 33.2 part B): fused RoPE + KV-cache write, flash-decode GQA.

Importing this module imports Triton; callers go through the Triton_Gate first.

Both kernels read the current position from the device tensor ``cache.position`` and use fixed
launch grids, so they are CUDA-graph capturable and nothing is synchronised on the host.

Numerics follow the torch decode path: RoPE rounds every product and the sum to fp16 exactly as
``q * cos + rotate_half(q) * sin`` does on fp16 tensors (the cos/sin table is the fp16 table of
``QUFModel._rope``), scores are fp32 and divided by ``sqrt(head_dim)``, and the probabilities are
rounded to fp16 before the PV product. The only difference is the softmax normalisation, which
is applied once after the PV sum instead of before the fp16 rounding of the probabilities.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kv_kernel(
    qkv_ptr, cos_ptr, sin_ptr, pos_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr,
    stride_ck, stride_ct,
    H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr,
):
    h = tl.program_id(0)
    pos = tl.load(pos_ptr)
    HALF: tl.constexpr = D // 2
    offs = tl.arange(0, HALF)
    if h < H + KV:  # q head or k head: contiguous in the fused qkv row
        base = h * D
        x1 = tl.load(qkv_ptr + base + offs)
        x2 = tl.load(qkv_ptr + base + HALF + offs)
        c1 = tl.load(cos_ptr + pos * D + offs)
        c2 = tl.load(cos_ptr + pos * D + HALF + offs)
        s1 = tl.load(sin_ptr + pos * D + offs)
        s2 = tl.load(sin_ptr + pos * D + HALF + offs)
        a1 = (x1.to(tl.float32) * c1.to(tl.float32)).to(tl.float16)
        b1 = ((-x2).to(tl.float32) * s1.to(tl.float32)).to(tl.float16)
        a2 = (x2.to(tl.float32) * c2.to(tl.float32)).to(tl.float16)
        b2 = (x1.to(tl.float32) * s2.to(tl.float32)).to(tl.float16)
        o1 = (a1.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
        o2 = (a2.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)
        if h < H:
            tl.store(q_out_ptr + h * D + offs, o1)
            tl.store(q_out_ptr + h * D + HALF + offs, o2)
        else:
            dst = k_cache_ptr + (h - H) * stride_ck + pos * stride_ct
            tl.store(dst + offs, o1)
            tl.store(dst + HALF + offs, o2)
    else:  # v head: plain copy into the cache slot
        kvh = h - H - KV
        d = tl.arange(0, D)
        v = tl.load(qkv_ptr + h * D + d)
        tl.store(v_cache_ptr + kvh * stride_ck + pos * stride_ct + d, v)


def rope_kv_write(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor, num_heads: int, num_kv_heads: int,
    q_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate q and k of the fused projection row ``qkv`` ``[1, (H + 2 KV) * D]`` at ``position``,
    write k and v into slot ``position`` of ``k_cache``/``v_cache`` (``[1, KV, T, D]``) and return
    the rotated q ``[H, D]``. ``cos``/``sin`` are fp16 ``[T, D]`` tables."""
    d = k_cache.shape[-1]
    if q_out is None:
        q_out = torch.empty((num_heads, d), dtype=torch.float16, device=qkv.device)
    _rope_kv_kernel[(num_heads + 2 * num_kv_heads,)](
        qkv, cos, sin, position, q_out, k_cache, v_cache,
        k_cache.stride(1), k_cache.stride(2),
        H=num_heads, KV=num_kv_heads, D=d, num_warps=1,
        enable_fp_fusion=False,  # keep the fp16 rounding after every product, like the torch ops
    )
    return q_out


@triton.jit
def _flash_decode_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, idx_ptr, out_ptr, pm_ptr, pl_ptr, pacc_ptr,
    stride_ck, stride_ct, inv_scale_div,
    CHUNK: tl.constexpr, SPLITS: tl.constexpr, REP: tl.constexpr, REP_PAD: tl.constexpr,
    D: tl.constexpr, BLOCK_T: tl.constexpr, HAS_INDEX: tl.constexpr, H: tl.constexpr,
):
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    pos = tl.load(pos_ptr)
    start = sp * CHUNK
    if start <= pos:
        end = tl.minimum(start + CHUNK, pos + 1)
        rows = tl.arange(0, REP_PAD)
        d = tl.arange(0, D)
        r_mask = rows < REP
        q = tl.load(q_ptr + (kvh * REP + rows)[:, None] * D + d[None, :], mask=r_mask[:, None], other=0.0)
        m_i = tl.full((REP_PAD,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((REP_PAD,), dtype=tl.float32)
        acc = tl.zeros((REP_PAD, D), dtype=tl.float32)
        kbase = k_ptr + kvh * stride_ck
        vbase = v_ptr + kvh * stride_ck
        for t0 in range(start, end, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            t_mask = t < end
            k = tl.load(kbase + t[:, None] * stride_ct + d[None, :], mask=t_mask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) / inv_scale_div
            s = tl.where(t_mask[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(vbase + t[:, None] * stride_ct + d[None, :], mask=t_mask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
            m_i = m_new
        if SPLITS == 1:
            res = (acc / l_i[:, None]).to(tl.float16)
            o_off = (kvh * REP + rows)[:, None] * D + d[None, :]
            if HAS_INDEX:
                dest = tl.load(idx_ptr + o_off, mask=r_mask[:, None], other=0)
                tl.store(out_ptr + dest, res, mask=r_mask[:, None])
            else:
                tl.store(out_ptr + o_off, res, mask=r_mask[:, None])
        else:
            head = kvh * REP + rows
            tl.store(pm_ptr + sp * H + head, m_i, mask=r_mask)
            tl.store(pl_ptr + sp * H + head, l_i, mask=r_mask)
            tl.store(pacc_ptr + (sp * H + head)[:, None] * D + d[None, :], acc, mask=r_mask[:, None])


@triton.jit
def _flash_reduce_kernel(
    pm_ptr, pl_ptr, pacc_ptr, pos_ptr, idx_ptr, out_ptr,
    CHUNK: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HAS_INDEX: tl.constexpr,
):
    head = tl.program_id(0)
    pos = tl.load(pos_ptr)
    d = tl.arange(0, D)
    n_active = pos // CHUNK + 1
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)
    for sp in range(0, n_active):
        m_s = tl.load(pm_ptr + sp * H + head)
        l_s = tl.load(pl_ptr + sp * H + head)
        a_s = tl.load(pacc_ptr + (sp * H + head) * D + d)
        m_new = tl.maximum(m_i, m_s)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_s - m_new)
        l_i = l_i * alpha + l_s * beta
        acc = acc * alpha + a_s * beta
        m_i = m_new
    res = (acc / l_i).to(tl.float16)
    if HAS_INDEX:
        dest = tl.load(idx_ptr + head * D + d)
        tl.store(out_ptr + dest, res)
    else:
        tl.store(out_ptr + head * D + d, res)


def flash_decode(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, position: torch.Tensor,
    num_heads: int, num_kv_heads: int, out_index: torch.Tensor | None = None,
    out: torch.Tensor | None = None, chunk: int = 512, block_t: int = 64,
) -> torch.Tensor:
    """GQA attention of one query position over the static cache.

    ``q`` fp16 ``[H, D]`` (rotated), caches ``[1, KV, T, D]``; keys ``0..position`` are attended.
    Returns fp16 ``[1, H * D]``; with ``out_index`` (int32 ``[H * D]``) element ``i`` is stored at
    ``out[out_index[i]]`` (the j-major layout of the ``o`` projection input)."""
    d = k_cache.shape[-1]
    t_max = k_cache.shape[2]
    rep = num_heads // num_kv_heads
    splits = max(1, triton.cdiv(t_max, chunk))
    if splits == 1:
        chunk = max(t_max, block_t)
    if out is None:
        out = torch.empty((1, num_heads * d), dtype=torch.float16, device=q.device)
    if splits > 1:
        pm = torch.empty((splits, num_heads), dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty((splits, num_heads, d), dtype=torch.float32, device=q.device)
    else:
        pm = pl = pacc = out  # unused
    idx = out_index if out_index is not None else position
    _flash_decode_kernel[(num_kv_heads, splits)](
        q, k_cache, v_cache, position, idx, out, pm, pl, pacc,
        k_cache.stride(1), k_cache.stride(2), math.sqrt(d),
        CHUNK=chunk, SPLITS=splits, REP=rep, REP_PAD=16, D=d, BLOCK_T=block_t,
        HAS_INDEX=out_index is not None, H=num_heads, num_warps=4,
    )
    if splits > 1:
        _flash_reduce_kernel[(num_heads,)](
            pm, pl, pacc, position, idx, out,
            CHUNK=chunk, H=num_heads, D=d, HAS_INDEX=out_index is not None, num_warps=1,
        )
    return out
