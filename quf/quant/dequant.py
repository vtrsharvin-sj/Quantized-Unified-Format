"""Reference_Dequantizer: pure PyTorch, the source of truth for every Backend."""

from __future__ import annotations

import torch

from quf.quant.quantize import SYMMETRIC_ZERO, QuantizedWeight

_SHIFTS_CACHE: dict[torch.device, torch.Tensor] = {}


def _shifts(device: torch.device) -> torch.Tensor:
    s = _SHIFTS_CACHE.get(device)
    if s is None:
        s = torch.arange(8, dtype=torch.int32, device=device) * 4
        _SHIFTS_CACHE[device] = s
    return s


def _unpack_last(packed: torch.Tensor) -> torch.Tensor:
    """int32 ``[..., M]`` -> int32 ``[..., M * 8]`` with values 0..15 (lsb first)."""
    nib = (packed.unsqueeze(-1) >> _shifts(packed.device)) & 0xF
    return nib.reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def expand_groups(
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    g_idx,
    group_size: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-column ``(scales, zeros)`` as fp32 ``[in, out]``.

    ``scales`` is ``[groups, out]``; ``zeros`` is the unpacked ``[groups, out]`` zero points or
    ``None`` for symmetric schemes (zero 8). Column ``c`` uses group ``g_idx[c]`` when ``g_idx``
    is given, otherwise group ``c // group_size``. This is the single definition of the
    group-to-column mapping shared by the Reference_Dequantizer and every backend.
    """
    s = scales.to(torch.float32)
    if zeros is None:
        z = torch.full_like(s, SYMMETRIC_ZERO)
    else:
        z = zeros.to(torch.float32)
    if g_idx is None or (hasattr(g_idx, "numel") and g_idx.numel() == 0):
        return s.repeat_interleave(group_size, dim=0), z.repeat_interleave(group_size, dim=0)
    idx = torch.as_tensor(g_idx).to(device=s.device, dtype=torch.long)
    if idx.shape[0] != in_features:
        raise ValueError(f"g_idx has length {idx.shape[0]}, expected {in_features}")
    return s[idx], z[idx]


def reference_dequantize(qw: QuantizedWeight, g_idx=None) -> torch.Tensor:
    """Return the fp16 weight ``[out, in]`` computed as ``(q - zero) * scale``.

    When ``g_idx`` (int, length ``in``) is given, column ``c`` uses the scale and zero
    of group ``g_idx[c]``; otherwise it uses group ``c // group_size``.
    Works on whatever device the tensors in ``qw`` live on.
    """
    qweight = qw.qweight
    # [in/8, out] -> [out, in/8] -> [out, in]
    q = _unpack_last(qweight.t().contiguous())
    q = q.t().contiguous()  # [in, out]

    zeros = None if qw.qzeros is None else _unpack_last(qw.qzeros)  # [n_groups, out]
    scales_full, zeros_full = expand_groups(
        qw.scales, zeros, g_idx, qw.group_size, q.shape[0]
    )  # [in, out]
    w = (q.to(torch.float32) - zeros_full) * scales_full  # [in, out]
    return w.t().contiguous().to(torch.float16)
