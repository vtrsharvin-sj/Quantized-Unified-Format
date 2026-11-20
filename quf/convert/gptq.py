"""Native GPTQ conversion: load with transformers, quantize with ``quf.quant.gptq``.

No GPTQModel is involved. ``run_native_gptq`` loads a Hugging Face causal LM,
runs :func:`quf.quant.gptq.quantize_model_sequential` on the calibration
samples, and returns a :class:`GPTQResult` in the canonical Packed_Int4 layout
(see FORMAT.md): ``qweight`` int32 ``[in/8, out]``, ``scales`` fp16
``[in/group_size, out]`` and, for asymmetric schemes, ``qzeros`` int32
``[in/group_size, out/8]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from quf.quant.quantize import QuantizedWeight

BITS = 4
GROUP_SIZE = 128


class MissingDependencyError(ImportError):
    """Raised when an optional conversion dependency is not installed."""


@dataclass
class GPTQResult:
    """Output of a quantization run, already in canonical layout."""

    linears: dict[str, QuantizedWeight] = field(default_factory=dict)
    others: dict[str, np.ndarray] = field(default_factory=dict)
    symmetric: bool = False
    group_size: int = GROUP_SIZE
    bits: int = BITS
    int8: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # name -> (q, scales)
    modules: dict[str, str] = field(default_factory=dict)  # precision per linear; empty = all int4
    act_order: bool = False


def to_tensor_dict(result: GPTQResult) -> dict[str, np.ndarray]:
    """Flatten a ``GPTQResult`` into canonical tensor names for the Writer."""
    out: dict[str, np.ndarray] = dict(result.others)
    for name, qw in result.linears.items():
        out[f"{name}.qweight"] = qw.qweight.numpy()
        out[f"{name}.scales"] = qw.scales.numpy()
        if qw.qzeros is not None:
            out[f"{name}.qzeros"] = qw.qzeros.numpy()
        if qw.g_idx is not None:
            out[f"{name}.g_idx"] = qw.g_idx.numpy().astype(np.int32)
    for name, (q8, s8) in result.int8.items():
        out[f"{name}.qweight_int8"] = q8
        out[f"{name}.scales"] = s8
    return out


def _other_array(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.is_floating_point():
        t = t.to(torch.float16)
    return t.contiguous().numpy()


def run_native_gptq(
    source: str,
    calibration: Sequence[Sequence[int]],
    *,
    group_size: int = GROUP_SIZE,
    symmetric: bool = False,
    act_order: bool = False,
    clip_search: bool = False,
    precision_plan=None,
    device: str | torch.device | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> GPTQResult:
    """Quantize ``source`` (Hub id or local dir) to 4 bits with the native GPTQ.

    ``calibration`` is a list of token-id samples (see ``build_calibration``).
    Raises ``MissingDependencyError`` naming ``quf[convert]`` if transformers
    is not installed.
    """
    try:
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
    except ImportError as exc:
        raise MissingDependencyError(
            "transformers is required for calibrated conversion but is not installed. "
            "Install it with: pip install 'quf[convert]'."
        ) from exc
    from quf.quant.gptq import default_device, quantize_model_sequential  # noqa: PLC0415

    dev = torch.device(device) if device is not None else default_device()
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype)

    plan = precision_plan
    int8: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    quantized = quantize_model_sequential(
        model,
        calibration,
        symmetric=symmetric,
        device=dev,
        group_size=group_size,
        act_order=act_order,
        clip_search=clip_search,
        precision=plan.precision if plan else None,
        int8_out=int8,
        progress=progress,
    )

    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    # lm_head is not a decoder layer: dense fp16 unless the plan asks for int8.
    lm_prec = (plan.explicit("lm_head") if plan else None) or "fp16"
    if lm_prec == "int4":
        raise ValueError("lm_head can be kept at fp16 or int8, not int4")
    if lm_prec == "int8":
        from quf.quant.quantize import quantize_int8  # noqa: PLC0415

        head = model.get_output_embeddings()
        q8, s8 = quantize_int8(head.weight)
        int8["lm_head"] = (q8.numpy(), s8.numpy())
    skip = {f"{n}.weight" for n in quantized} | {f"{n}.weight" for n in int8}
    result = GPTQResult(
        linears=quantized, symmetric=symmetric, group_size=group_size, bits=BITS,
        int8=int8, act_order=act_order,
    )
    if plan:
        decoder = [n for n in model.state_dict() if n.endswith(".weight") and n.startswith("model.layers.")
                   and n[: -len(".weight")] in _linear_names(model)]
        result.modules = {n[: -len(".weight")]: plan.precision(n[: -len(".weight")]) for n in decoder}
        if lm_prec == "int8":
            result.modules["lm_head"] = "int8"
    for name, tensor in model.state_dict().items():
        if name in skip:
            continue
        if tied and name == "lm_head.weight":
            continue  # the runtime falls back to the embedding matrix (or lm_head.qweight_int8)
        result.others[name] = _other_array(tensor)
    return result


def _linear_names(model) -> set[str]:
    return {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
