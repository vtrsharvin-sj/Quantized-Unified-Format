"""Decode GEMV for the fused path (task 33.2 part B): M = 1, activation in "j-major" kernel order.

Importing this module imports Triton; callers go through the Triton_Gate first.

Why a new kernel: the in-model weights are act-order (every projection has a non-identity
``perm``) with group size 32. The older kernel gathered ``x[perm[k]]`` with an int64 perm load per
element, so each program moved about 4x more bytes of perm and x through L1 than of weights.
Here the *producer* of the activation (the add+RMSNorm kernel, the attention kernel, the
SiLU*mul epilogue) scatters its output once into the consumer's layout, and this kernel reads
only contiguous vectors.

Activation layout ("j-major"). For a layout with ``K`` inputs, kernel column ``i`` is original
column ``perm[i]`` (``i`` itself without perm). Kernel column ``i = 8*w + j`` is nibble ``j`` of
packed word ``w``. The kernel wants, for each ``j``, the vector over ``w`` contiguous, so the
activation is stored as ``xj[j * (K // 8) + w] = x[perm[8 * w + j]]``. ``activation_index`` gives
the destination of every original column: ``xj[dest[c]] = x[c]``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from quf.runtime.kernels.triton_kernels import Int4Layout

def activation_index(layout: Int4Layout) -> torch.Tensor:
    """int32 ``[K]``: destination in the j-major activation of each original column.

    Cached on the layout itself, so it is freed with the layout (a module-level dict keyed by
    ``id(layout)`` kept every layout, and its GPU tensors, alive for the life of the process).
    """
    hit = getattr(layout, "_jmajor_index", None)
    if hit is not None:
        return hit
    k = layout.in_features
    dev = layout.packed.device
    if layout.perm is None:
        inv = torch.arange(k, device=dev, dtype=torch.int64)
    else:
        inv = torch.empty(k, device=dev, dtype=torch.int64)
        inv[layout.perm] = torch.arange(k, device=dev, dtype=torch.int64)
    dest = ((inv % 8) * (k // 8) + inv // 8).to(torch.int32)
    layout._jmajor_index = dest  # type: ignore[attr-defined]
    return dest


def to_jmajor(x: torch.Tensor, layout: Int4Layout) -> torch.Tensor:
    """Torch reference of the producer scatter: fp16 ``[..., K]`` -> j-major ``[..., K]``."""
    out = torch.empty_like(x)
    out[..., activation_index(layout).long()] = x
    return out


@triton.jit
def _gemv_jmajor_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, bias_ptr, out_ptr, ws_ptr, lock_ptr,
    N, K,
    GROUP: tl.constexpr, HAS_ZEROS: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr, NGB: tl.constexpr, SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    k8 = K // 8
    ng = K // GROUP
    GW: tl.constexpr = GROUP // 8
    g_idx = tl.arange(0, NGB)
    gw = tl.arange(0, GW)
    # Split-K: program (pid, sid) covers groups [g_lo, g_hi); SPLIT == 1 is the whole row.
    if SPLIT == 1:
        g_lo = 0
        g_hi = ng
    else:
        sid = tl.program_id(1)
        per = tl.cdiv(tl.cdiv(ng, SPLIT), NGB) * NGB
        g_lo = sid * per
        g_hi = tl.minimum(g_lo + per, ng)
    acc = tl.zeros((BLOCK_N, NGB), dtype=tl.float32)
    for g0 in range(g_lo, g_hi, NGB):
        gg = g0 + g_idx
        g_mask = gg < g_hi
        widx = gg[:, None] * GW + gw[None, :]  # [NGB, GW] packed word index
        m2 = g_mask[:, None] & (gw[None, :] < GW)
        m3 = n_mask[:, None, None] & m2[None, :, :]
        word = tl.load(w_ptr + offs_n[:, None, None] * k8 + widx[None, :, :], mask=m3, other=0)
        m_ng = n_mask[:, None] & g_mask[None, :]
        s = tl.load(s_ptr + offs_n[:, None] * ng + gg[None, :], mask=m_ng, other=0.0).to(tl.float32)
        if HAS_ZEROS:
            z = tl.load(z_ptr + offs_n[:, None] * ng + gg[None, :], mask=m_ng, other=0.0).to(tl.float32)
        else:
            z = 8.0
        d = tl.zeros((BLOCK_N, NGB, GW), dtype=tl.float32)
        xs = tl.zeros((NGB, GW), dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(x_ptr + j * k8 + widx, mask=m2, other=0.0).to(tl.float32)
            q = ((word >> (4 * j)) & 15).to(tl.float32)
            d += q * xj[None, :, :]
            xs += xj
        acc += s * (tl.sum(d, axis=2) - z * tl.sum(xs, axis=1)[None, :])
    part = tl.sum(acc, axis=1)
    if SPLIT == 1:
        res = part.to(tl.float16)
        if HAS_BIAS:
            res = res + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        tl.store(out_ptr + offs_n, res, mask=n_mask)
    else:
        # In-kernel reduce: every split writes its fp32 partial, then takes a ticket on this row
        # block's counter; the last arrival sums the partials in a fixed split order (so the result
        # does not depend on arrival order), writes fp16 + bias, and resets the counter for the
        # next launch (CUDA-graph replay needs no memset).
        tl.store(ws_ptr + sid * N + offs_n, part, mask=n_mask)
        tl.debug_barrier()
        ticket = tl.atomic_add(lock_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if ticket == SPLIT - 1:
            tot = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for t in tl.static_range(SPLIT):
                tot += tl.load(ws_ptr + t * N + offs_n, mask=n_mask, other=0.0, cache_modifier=".cg")
            res = tot.to(tl.float16)
            if HAS_BIAS:
                res = res + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
            tl.store(out_ptr + offs_n, res, mask=n_mask)
            tl.atomic_xchg(lock_ptr + pid, 0, sem="relaxed", scope="gpu")


def jmajor_config(n: int, k: int) -> tuple[int, int, int, int, int]:
    """Static ``(BLOCK_N, NGB, num_warps, num_stages, SPLIT)``: the per-shape entry of
    ``_JMAJOR_CONFIGS`` (chosen by whole-graph replay time, scripts/_jmajor_lab.py, task 39.5),
    else the default of task 33.2: tall-K down projections like one row per program with a long
    K loop, everything else two rows and a short loop."""
    hit = _JMAJOR_CONFIGS.get((n, k))
    if hit is not None:
        return hit
    if k >= 4096:
        return 1, 32, 2, 2, 1
    return 2, 16, 1, 2, 1


# (N, K) -> (BLOCK_N, NGB, num_warps, num_stages, SPLIT); RTX 5080 Laptop whole-graph sweep, task 39.5.
_JMAJOR_CONFIGS: dict[tuple[int, int], tuple[int, int, int, int, int]] = {
    (17920, 1536): (16, 16, 1, 3, 1),  # Qwen2.5-1.5B gate_up
    (1536, 8960): (16, 32, 2, 3, 2),   # 1.5B down, split-K 2 with in-kernel reduce
    (1536, 1536): (2, 16, 1, 3, 1),    # 1.5B q / o
    (896, 896): (1, 16, 1, 3, 1),      # Qwen2.5-0.5B q / o
    (9728, 896): (16, 32, 2, 3, 1),    # 0.5B gate_up
    (896, 4864): (16, 32, 2, 3, 4),    # 0.5B down, split-K 4
}


def splitk_workspace(n: int, block_n: int, split: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 partials ``[split * n]`` and a zeroed int32 counter per row block for split-K."""
    return (torch.empty(split * n, dtype=torch.float32, device=device),
            torch.zeros(triton.cdiv(n, block_n), dtype=torch.int32, device=device))


def gemv_jmajor(
    xj: torch.Tensor,
    layout: Int4Layout,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    config: tuple[int, ...] | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``x @ W.T (+ bias)`` for a j-major fp16 activation ``xj`` of ``K`` elements; fp16 ``[1, N]``.

    ``config`` is ``(BLOCK_N, NGB, num_warps[, num_stages[, SPLIT]])``. With SPLIT > 1 the K
    groups are split over SPLIT programs per row block and reduced in the kernel; ``workspace``
    (from ``splitk_workspace``) must then be private to this call site, and is allocated when None.
    CUDA-graph capturable (one launch, no sync). Run once before capture to compile.
    """
    n, k = layout.out_features, layout.in_features
    if out is None:
        out = torch.empty((1, n), dtype=torch.float16, device=xj.device)
    cfg = tuple(config or jmajor_config(n, k))
    block_n, ngb, warps, stages, split = cfg + (2, 1)[len(cfg) - 3:]  # stages 2, SPLIT 1 if omitted
    if split > 1 and workspace is None:
        workspace = splitk_workspace(n, block_n, split, xj.device)
    ws, lock = workspace if split > 1 else (layout.scales, layout.scales)
    zeros = layout.zeros if layout.zeros is not None else layout.scales
    _gemv_jmajor_kernel[(triton.cdiv(n, block_n), split)](
        xj, layout.packed, layout.scales, zeros, bias if bias is not None else layout.scales, out, ws, lock,
        n, k, GROUP=layout.group_size, HAS_ZEROS=layout.zeros is not None, HAS_BIAS=bias is not None,
        BLOCK_N=block_n, NGB=ngb, SPLIT=split, num_warps=warps, num_stages=stages,
    )
    return out


@triton.jit
def _int8_gemv1_kernel(
    x_ptr, xidx_ptr, w_ptr, s_ptr, bias_ptr, oidx_ptr, out_ptr,
    N, K,
    HAS_XIDX: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_OIDX: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    row = offs_n[:, None].to(tl.int64) * K
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        if HAS_XIDX:
            # weight column c reads the activation at xidx[c] (a producer-scattered layout)
            xk = tl.load(xidx_ptr + kk, mask=k_mask, other=0)
        else:
            xk = kk
        x = tl.load(x_ptr + xk, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + row + kk[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0)
        acc += w.to(tl.float32) * x[None, :]
    # fp32 accumulation, per-output-channel scale applied once after the reduction
    s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    res = (tl.sum(acc, axis=1) * s).to(tl.float16)
    if HAS_BIAS:
        res = res + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    if HAS_OIDX:
        dest = tl.load(oidx_ptr + offs_n, mask=n_mask, other=0)
    else:
        dest = offs_n
    tl.store(out_ptr + dest, res, mask=n_mask)


def int8_config(n: int, k: int) -> tuple[int, int, int]:
    """Static ``(BLOCK_N, BLOCK_K, num_warps)`` for ``gemv_int8`` (see scripts/_int8_gemv_lab.py)."""
    return _INT8_CONFIGS.get((n, k)) or (4, 512, 4)


# Filled from the sweep in scripts/_int8_gemv_lab.py (RTX 5080 Laptop, cold weights).
_INT8_CONFIGS: dict[tuple[int, int], tuple[int, int, int]] = {
    (128, 896): (1, 1024, 4), (896, 896): (1, 512, 2), (4864, 896): (8, 128, 4),
    (9728, 896): (16, 128, 8), (896, 4864): (8, 128, 8), (256, 1536): (1, 1024, 8),
    (1536, 1536): (8, 128, 8), (8960, 1536): (16, 128, 4), (17920, 1536): (16, 128, 4),
    (1536, 8960): (8, 128, 8),
}


def gemv_int8(
    x: torch.Tensor,
    q: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    x_index: torch.Tensor | None = None,
    out_index: torch.Tensor | None = None,
    config: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """``x @ (q * scales[:, None]).T (+ bias)`` for one fp16 activation row; fp16 ``[1, N]``.

    ``q`` is int8 ``[N, K]``, ``scales`` fp16 ``[N]``. ``x_index`` (int32 ``[K]``) is the input
    layout convention of the fast path: weight column ``c`` reads ``x[x_index[c]]`` (the
    ``activation_index`` scatter a producer kernel writes); None means natural order.
    ``out_index`` (int32 ``[N]``) scatters output channel ``n`` to ``out[out_index[n]]``; ``out``
    may be a larger shared buffer (one per Mixed_Group) or a view into one.
    Accumulates in fp32, applies the scale once per channel. One launch, no host sync, so it is
    CUDA-graph capturable (run once before capture to compile).
    """
    n, k = q.shape
    if out is None:
        out = torch.empty((1, n), dtype=torch.float16, device=q.device)
    block_n, block_k, warps = config or int8_config(n, k)
    dummy = scales
    _int8_gemv1_kernel[(triton.cdiv(n, block_n),)](
        x, x_index if x_index is not None else dummy, q, scales,
        bias if bias is not None else dummy, out_index if out_index is not None else dummy, out,
        n, k, HAS_XIDX=x_index is not None, HAS_BIAS=bias is not None, HAS_OIDX=out_index is not None,
        BLOCK_N=block_n, BLOCK_K=block_k, num_warps=warps, num_stages=3,
    )
    return out


class _FastBase:
    """What the model needs of a fast projection: ``index`` is the layout the producer kernels
    write this module's input in (an int32 scatter index, or None for natural order), and
    ``scatter_index`` is the same as a tensor always (identity for natural order), for producers
    that cannot take None."""

    index: torch.Tensor | None
    in_features: int
    out_features: int

    def _set_index(self, index: torch.Tensor | None, device) -> None:
        self.index = index
        self.scatter_index = index if index is not None else torch.arange(
            self.in_features, dtype=torch.int32, device=device)


def _fp16_bias(bias: torch.Tensor | None, device) -> torch.Tensor | None:
    return None if bias is None else bias.to(device=device, dtype=torch.float16).contiguous()


class FastLinear(_FastBase):
    """One int4 projection prepared for the j-major GEMV: layout, bias, and the scatter index
    that producers use to write this module's input."""
    def __init__(self, layout: Int4Layout, bias: torch.Tensor | None):
        self.layout = layout
        self.in_features, self.out_features = layout.in_features, layout.out_features
        self.bias = _fp16_bias(bias, layout.packed.device)
        self._set_index(activation_index(layout), layout.packed.device)
        self.config = jmajor_config(layout.out_features, layout.in_features)
        self._ws_key, self._ws = None, None

    def _workspace(self):
        """The split-K partials and counter of this module (one per module: calls run in stream order)."""
        block_n, split = self.config[0], (tuple(self.config) + (2, 1)[len(self.config) - 3:])[4]
        if split == 1:
            return None
        if self._ws_key != (block_n, split):
            self._ws = splitk_workspace(self.out_features, block_n, split, self.layout.packed.device)
            self._ws_key = (block_n, split)
        return self._ws

    def __call__(self, xj: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return gemv_jmajor(xj, self.layout, self.bias, out=out, config=self.config, workspace=self._workspace())


class FastInt8Linear(_FastBase):
    """One int8 projection on ``gemv_int8``. ``index`` is the input layout the producers write;
    the weight columns are in natural order, so the kernel reads ``x[index[c]]`` for column c."""
    def __init__(self, q: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor | None,
                 index: torch.Tensor | None = None):
        self.q, self.scales = q, scales
        self.out_features, self.in_features = int(q.shape[0]), int(q.shape[1])
        self.bias = _fp16_bias(bias, q.device)
        self._set_index(index, q.device)
        self.config = int8_config(self.out_features, self.in_features)
    def with_index(self, index: torch.Tensor | None) -> "FastInt8Linear":
        return FastInt8Linear(self.q, self.scales, self.bias, index)
    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return gemv_int8(x, self.q, self.scales, self.bias, out=out, x_index=self.index, config=self.config)


class FastFp16Linear(_FastBase):
    """An fp16 projection (cuBLAS) that stays inside the fast layer. Its input arrives in the
    group's scatter layout, so the weight columns are moved to match once at build time."""
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, index: torch.Tensor | None = None):
        self.out_features, self.in_features = int(weight.shape[0]), int(weight.shape[1])
        self.source = weight
        if index is None:
            self.weight = weight
        else:
            self.weight = torch.empty_like(weight)
            self.weight[:, index.long()] = weight  # column c of W reads the activation at index[c]
        self.bias = _fp16_bias(bias, weight.device)
        self._set_index(index, weight.device)
    def with_index(self, index: torch.Tensor | None) -> "FastFp16Linear":
        return FastFp16Linear(self.source, self.bias, index)
    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        if out is None:
            return y
        out.copy_(y)
        return out


class FastGroup(_FastBase):
    """Members that share one input (a Mixed_Group such as q int4 + k, v int8): every member writes
    its outputs into its slice of one buffer, so the consumers see the same ``[1, sum widths]``
    row as a single fused projection. One kernel launch per member."""
    def __init__(self, members: list[_FastBase]):
        self.members = members
        self.widths = [m.out_features for m in members]
        self.offsets = [sum(self.widths[:i]) for i in range(len(members))]
        self.in_features = members[0].in_features
        self.out_features = sum(self.widths)
        self._set_index(members[0].index, self._device_of(members[0]))
    @staticmethod
    def _device_of(m: _FastBase):
        return m.layout.packed.device if isinstance(m, FastLinear) else (
            m.q.device if isinstance(m, FastInt8Linear) else m.weight.device)
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((1, self.out_features), dtype=torch.float16, device=x.device)
        for m, off, w in zip(self.members, self.offsets, self.widths):
            m(x, out[:, off : off + w])
        return out


def fast_linear_of(module) -> FastLinear | None:
    """``FastLinear`` for a QUFLinear or FusedProjection holding a supported int4 weight, else None."""
    prepared = getattr(module, "prepared", None)
    if not isinstance(prepared, Int4Layout):
        return None
    if prepared.group_size not in (32, 64, 128) or prepared.in_features % prepared.group_size:
        return None
    return FastLinear(prepared, getattr(module, "bias", None))


def _member_of(prepared, bias) -> _FastBase | str:
    """The fast member for one prepared weight, or the reason it has none."""
    from quf.runtime.kernels.dispatch import CachedFp16Weight, Int8KernelWeight  # noqa: PLC0415

    if isinstance(prepared, Int4Layout):
        if prepared.group_size not in (32, 64, 128) or prepared.in_features % prepared.group_size:
            return f"int4 module with group size {prepared.group_size} and K={prepared.in_features} is unsupported"
        return FastLinear(prepared, bias)
    if isinstance(prepared, Int8KernelWeight):
        return FastInt8Linear(prepared.q, prepared.scales, bias)
    if isinstance(prepared, CachedFp16Weight):
        return FastFp16Linear(prepared.weight, bias)
    return f"unsupported prepared weight {type(prepared).__name__}"


def fast_group_of(module) -> tuple[_FastBase | None, str]:
    """The fast projection for a QUFLinear or FusedProjection of int4, int8 or fp16 weights, as
    ``(fast, "")``; ``(None, reason)`` when the module needs the unfused path. A projection group
    whose members were not fused into one weight (a mix of precisions) becomes a ``FastGroup``."""
    kernel = getattr(module, "_kernel", None)
    if kernel is not None and not kernel.is_fused:
        parts = [(p, getattr(m, "bias", None)) for p, m in zip(kernel.parts, module._part_modules)]
    else:
        parts = [(getattr(module, "prepared", None), getattr(module, "bias", None))]
    members = []
    for prepared, bias in parts:
        m = _member_of(prepared, bias)
        if isinstance(m, str):
            return None, m
        members.append(m)
    if len(members) == 1:
        return members[0], ""
    # The producer writes one layout for the whole group. int4 members define it (they need the
    # j-major order of their own permutation), so they must agree; int8 and fp16 members adapt.
    int4 = [m for m in members if isinstance(m, FastLinear)]
    index = int4[0].index if int4 else None
    for m in int4[1:]:
        if not torch.equal(m.index, index):
            return None, "int4 members of one projection group have different act-order permutations"
    members = [m if isinstance(m, FastLinear) else m.with_index(index) for m in members]
    return FastGroup(members), ""
@triton.jit
def _silu_mul_scatter_kernel(gu_ptr, idx_ptr, out_ptr, n_half, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_half
    g = tl.load(gu_ptr + offs, mask=mask, other=0.0)
    u = tl.load(gu_ptr + n_half + offs, mask=mask, other=0.0)
    gf = g.to(tl.float32)
    act = (gf / (1.0 + tl.exp(-gf))).to(tl.float16)  # silu rounded to fp16 like F.silu on fp16
    prod = (act.to(tl.float32) * u.to(tl.float32)).to(tl.float16)
    dest = tl.load(idx_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + dest, prod, mask=mask)


def silu_mul_scatter(gate_up: torch.Tensor, index: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``silu(gate) * up`` of a fused ``[1, 2I]`` gate_up output, scattered into the j-major
    activation (``out[index[i]]``) of the down projection."""
    n_half = gate_up.shape[-1] // 2
    if out is None:
        out = torch.empty((1, n_half), dtype=torch.float16, device=gate_up.device)
    block = 1024
    _silu_mul_scatter_kernel[(triton.cdiv(n_half, block),)](gate_up, index, out, n_half, BLOCK=block, num_warps=4)
    return out

@triton.jit
def _scatter_kernel(x_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dest = tl.load(idx_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + dest, tl.load(x_ptr + offs, mask=mask, other=0.0), mask=mask)


def scatter_to(x: torch.Tensor, index: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``out[index[i]] = x[i]`` for a ``[1, K]`` fp16 row (the producer scatter, as its own kernel)."""
    n = x.shape[-1]
    if out is None:
        out = torch.empty((1, n), dtype=x.dtype, device=x.device)
    _scatter_kernel[(triton.cdiv(n, 1024),)](x, index, out, n, BLOCK=1024, num_warps=4)
    return out
