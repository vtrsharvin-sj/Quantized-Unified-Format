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

from quf.quant.embedding import QuantizedEmbedding
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
    embeddings: dict[str, QuantizedEmbedding] = field(default_factory=dict)  # "model.embed_tokens" / "lm_head"
    embed_method: str = "clip"


EMBED_PRECISIONS = ("fp16", "int8", "int4")
EMBED_TABLES = (("embedding", "model.embed_tokens"), ("lm_head", "lm_head"))


def quantize_embeddings(
    result: GPTQResult, precision: str, group_size: int = 64, method: str = "clip"
) -> GPTQResult:
    """Replace the fp16 embedding table (and an untied lm_head) in ``result`` by Quantized_Embeddings.

    ``precision`` is ``fp16`` (no change), ``int8`` or ``int4``. A lm_head that the precision
    plan already stores as ``lm_head.qweight_int8`` is left as it is. A tied lm_head has no
    tensor of its own and uses the embedding table at run time.
    """
    if precision not in EMBED_PRECISIONS:
        raise ValueError(f"unsupported embed precision {precision!r}: use fp16, int8 or int4")
    if precision == "fp16":
        return result
    from quf.quant.embedding import quantize_embedding  # noqa: PLC0415

    bits = 4 if precision == "int4" else 8
    for _, name in EMBED_TABLES:
        w = result.others.get(f"{name}.weight")
        if w is None:
            continue
        result.embeddings[name] = quantize_embedding(w, bits, group_size, method)
        del result.others[f"{name}.weight"]
    if not result.embeddings:
        raise ValueError("embed precision requested but the model has no embedding table")
    result.embed_method = method
    return result


def embedding_header_specs(result: GPTQResult) -> dict[str, dict]:
    """The ``quantization.embedding`` / ``quantization.lm_head`` objects for the header."""
    specs: dict[str, dict] = {}
    for key, name in EMBED_TABLES:
        qe = result.embeddings.get(name)
        if qe is not None:
            specs[key] = {
                "name": name, "bits": qe.bits, "group_size": qe.group_size,
                "method": "rtn", "clip_search": result.embed_method == "clip",
            }
    return specs


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
    for name, qe in result.embeddings.items():
        suffix = ".embed_q4" if qe.bits == 4 else ".embed_q8"
        out[name + suffix] = qe.packed
        out[name + ".embed_scales"] = qe.scales
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
    damp: float = 0.01,
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
        damp=damp,
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
