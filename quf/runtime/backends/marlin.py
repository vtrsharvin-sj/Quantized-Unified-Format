"""Marlin backend (GPTQModel kernels).

Marlin is unverified on native Windows and on compute capability 12.0, so every
GPTQModel import is lazy. When GPTQModel or its kernels are missing, ``prepare``
raises ``MarlinUnavailableError`` and ``marlin_self_test`` returns False.

The canonical Packed_Int4 weight (GPTQ layout, ``[in/8, out]`` int32) is repacked into
the Marlin tile layout in memory only. The QUF_File is never written.
Only symmetric quantization (zero point 8, GPTQModel ``uint4b8``) is supported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import QuantizedWeight

logger = logging.getLogger(__name__)

SELF_TEST_TOLERANCE = 1e-2
_SELF_TEST_GROUP = 128


class MarlinUnavailableError(RuntimeError):
    """GPTQModel or its Marlin kernels cannot be used in this environment."""


@dataclass
class MarlinWeight:
    qweight: torch.Tensor  # Marlin-tiled int32
    scales: torch.Tensor  # Marlin-permuted fp16
    workspace: torch.Tensor
    g_idx: torch.Tensor
    g_idx_sort_indices: torch.Tensor
    in_features: int
    out_features: int

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.scales.dtype


def _load_gptqmodel():
    """Import the GPTQModel pieces lazily. Raises MarlinUnavailableError on any failure."""
    try:
        import importlib

        utils = importlib.import_module("gptqmodel.utils.marlin")
        stype = importlib.import_module("gptqmodel.utils.marlin_scalar_type")
        kernels = importlib.import_module("gptqmodel_marlin_kernels")
        return utils, stype.scalar_types.uint4b8, kernels
    except Exception as exc:  # noqa: BLE001 - ImportError, AttributeError, OSError from native libs
        raise MarlinUnavailableError(f"GPTQModel Marlin kernels unavailable: {type(exc).__name__}: {exc}") from exc


def require_marlin() -> None:
    """Raise ``MarlinUnavailableError`` naming the optional package when it is missing."""
    try:
        _load_gptqmodel()
    except MarlinUnavailableError as exc:
        raise MarlinUnavailableError(
            f"--backend marlin needs the optional package 'gptqmodel' (pip install gptqmodel): {exc}"
        ) from exc


class MarlinBackend:
    name = "marlin"

    def prepare(self, qw: Any, device: Any = "cuda", g_idx: torch.Tensor | None = None):
        dev = torch.device("cuda" if device is None else device)
        if dev.type != "cuda":
            raise ValueError(f"Marlin backend requires a CUDA device, got {dev}")
        if isinstance(qw, torch.Tensor):  # fp16 module: plain matmul
            return qw.to(dev, torch.float16).contiguous()
        if not isinstance(qw, QuantizedWeight):
            raise MarlinUnavailableError(
                "Marlin backend does not support int8 modules; use --backend torch_gpu or torch_fused"
            )
        if g_idx is not None:
            raise MarlinUnavailableError(
                "Marlin backend does not support files with g_idx (act-order); "
                "use --backend torch_gpu or torch_fused"
            )
        if qw.qzeros is not None:
            raise MarlinUnavailableError("Marlin backend supports symmetric quantization only")
        utils, wtype, kernels = _load_gptqmodel()
        try:
            qweight = qw.qweight.to(dev).contiguous()
            scales = qw.scales.to(dev).to(torch.float16).contiguous()
            empty = torch.empty(0, dtype=torch.int32, device=dev)
            repacked = kernels.gptq_marlin_repack(qweight, empty, qw.in_features, qw.out_features, 4)
            m_scales = utils.marlin_permute_scales(scales, qw.in_features, qw.out_features, qw.group_size)
            workspace = utils.marlin_make_workspace(dev)
            g_idx = utils.marlin_make_empty_g_idx(dev)
        except MarlinUnavailableError:
            raise
        except AttributeError as exc:
            raise MarlinUnavailableError(f"GPTQModel Marlin API mismatch: {exc}") from exc
        return MarlinWeight(repacked, m_scales, workspace, g_idx, g_idx, qw.in_features, qw.out_features)

    def matmul(self, x: torch.Tensor, prepared) -> torch.Tensor:
        if isinstance(prepared, torch.Tensor):
            return torch.matmul(x.to(prepared.device, torch.float16), prepared.t()).to(x.dtype)
        utils, wtype, _ = _load_gptqmodel()
        orig_dtype = x.dtype
        x2 = x.to(device=prepared.device, dtype=torch.float16)
        y = utils.apply_gptq_marlin_linear(
            input=x2.reshape(-1, x2.shape[-1]),
            weight=prepared.qweight,
            weight_scale=prepared.scales,
            weight_zp=prepared.g_idx,
            g_idx=prepared.g_idx,
            g_idx_sort_indices=prepared.g_idx_sort_indices,
            workspace=prepared.workspace,
            wtype=wtype,
            output_size_per_partition=prepared.out_features,
            input_size_per_partition=prepared.in_features,
            is_k_full=True,
        )
        return y.reshape(*x.shape[:-1], prepared.out_features).to(orig_dtype)


def marlin_self_test(seed: int = 0) -> bool:
    """Run a small Marlin matmul and compare with the Reference_Dequantizer result.

    Returns False when CUDA, GPTQModel, or its kernels are unavailable, or when the
    result differs from the reference by more than 1e-2 (absolute). Never raises.
    """
    try:
        if not torch.cuda.is_available():
            logger.info("Marlin self-test skipped: CUDA is not available")
            return False
        _load_gptqmodel()
        in_f, out_f = 256, 256
        gen = torch.Generator().manual_seed(seed)
        qweight = torch.randint(
            -(2**31), 2**31 - 1, (in_f // 8, out_f), dtype=torch.int32, generator=gen
        )
        scales = (torch.rand(in_f // _SELF_TEST_GROUP, out_f, generator=gen) * 0.02 + 0.005).to(torch.float16)
        qw = QuantizedWeight(
            qweight=qweight.cuda(),
            scales=scales.cuda(),
            qzeros=None,
            group_size=_SELF_TEST_GROUP,
            in_features=in_f,
            out_features=out_f,
        )
        x = torch.randn(4, in_f, generator=gen).to(torch.float16).cuda()
        expected = torch.matmul(x, reference_dequantize(qw).t())
        be = MarlinBackend()
        actual = be.matmul(x, be.prepare(qw, "cuda"))
        torch.cuda.synchronize()
        if not torch.isfinite(actual.float()).all():
            logger.warning("Marlin self-test failed: non-finite output")
            return False
        err = (actual.float() - expected.float()).abs().max().item()
        if err > SELF_TEST_TOLERANCE:
            logger.warning("Marlin self-test failed: max abs error %.4g > %g", err, SELF_TEST_TOLERANCE)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Marlin self-test unavailable: %s: %s", type(exc).__name__, exc)
        return False
