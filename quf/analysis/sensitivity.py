"""Sensitivity_Analyzer and Ablation_Harness.

``module_errors`` reports the relative output error of every decoder linear
when quantized to 4 bits on calibration data. ``ablate`` quantizes only a chosen
subset of linears (all others stay in the loaded fp16 weights) and measures
WikiText perplexity with the existing :func:`quf.bench.suite.perplexity`.

Every module is quantized independently from its fp16 weight, using its input
activations from the unmodified model. The relative output error is

    ||X W^T - X Wq^T||_F / ||X W^T||_F = sqrt(tr(E H E^T) / tr(W H W^T))

with ``E = W - Wq`` and ``H = sum X^T X`` over all calibration tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from quf.quant.dequant import reference_dequantize
from quf.quant.gptq import _decoder_layers, gptq_quantize
from quf.quant.quantize import GROUP_SIZE, quantize_groupwise

KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_NAME_RE = re.compile(r"^model\.layers\.(\d+)\.")


@dataclass
class ModuleError:
    name: str  # e.g. model.layers.3.mlp.down_proj
    layer: int
    kind: str  # q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    rel_output_error: float  # ||X W^T - X Wq^T||_F / ||X W^T||_F on calibration data


@dataclass
class Scheme:
    """How a module is quantized: ``rtn`` or native ``gptq`` (needs ``calib``)."""

    method: str = "rtn"
    group_size: int = GROUP_SIZE
    symmetric: bool = False
    damp: float = 0.01
    calib: Any = None  # [n_samples, seq_len] token ids; required for gptq ablation

    def __post_init__(self) -> None:
        if self.method not in ("rtn", "gptq"):
            raise ValueError(f"unknown quantization method {self.method!r}")


# -- helpers ---------------------------------------------------------------


def decoder_linear_names(model) -> list[str]:
    """Names of every ``nn.Linear`` inside the decoder layers, in model order."""
    layers = _decoder_layers(model)
    names = []
    for li, layer in enumerate(layers):
        for n, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear):
                names.append(f"model.layers.{li}.{n}")
    return names


def _parse(name: str) -> tuple[int, str]:
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"not a decoder layer module name: {name!r}")
    return int(m.group(1)), name.rsplit(".", 1)[-1]


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def _collect_hessians(
    model, calib, names: Iterable[str]
) -> dict[str, torch.Tensor]:
    """``H = sum X^T X`` (float32) of each named linear over the calibration set."""
    modules = dict(model.named_modules())
    ids = torch.as_tensor(calib, dtype=torch.long)
    if ids.ndim != 2 or ids.shape[0] == 0:
        raise ValueError("calib must be a non-empty [n_samples, seq_len] token tensor")
    hess: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(n: str):
        def hook(_m, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)
            h = x.t() @ x
            if n in hess:
                hess[n] += h
            else:
                hess[n] = h

        return hook

    for n in names:
        handles.append(modules[n].register_forward_hook(make_hook(n)))
    dev = _device_of(model)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for i in range(ids.shape[0]):
                model(input_ids=ids[i : i + 1].to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return hess


def _quantize_weight(
    w: torch.Tensor, scheme: Scheme, hessian: torch.Tensor | None, name: str
) -> torch.Tensor:
    """Dequantized fp32 weight (on ``w``'s device) of the 4-bit quantization of ``w``."""
    if scheme.method == "gptq":
        assert hessian is not None
        qw = gptq_quantize(
            w,
            hessian,
            group_size=scheme.group_size,
            symmetric=scheme.symmetric,
            damp=scheme.damp,
            layer_name=name,
            device=w.device,
        )
    else:
        qw = quantize_groupwise(w, scheme.group_size, scheme.symmetric)
    return reference_dequantize(qw).to(w.device, torch.float32)


def _needs_hessian(scheme: Scheme) -> bool:
    return scheme.method == "gptq"


# -- public API ------------------------------------------------------------


def module_errors(model, calib, scheme: Scheme | None = None) -> list[ModuleError]:
    """Relative output error of the 4-bit quantized version of every decoder linear."""
    scheme = scheme or Scheme()
    names = decoder_linear_names(model)
    modules = dict(model.named_modules())
    hess = _collect_hessians(model, calib, names)

    out: list[ModuleError] = []
    for name in names:
        layer, kind = _parse(name)
        w = modules[name].weight.detach().to(torch.float32)
        h = hess.pop(name)
        wq = _quantize_weight(w, scheme, h if _needs_hessian(scheme) else None, name)
        e = w - wq
        num = float(((e @ h) * e).sum())
        den = float(((w @ h) * w).sum())
        rel = (max(num, 0.0) / den) ** 0.5 if den > 0 else 0.0
        out.append(ModuleError(name=name, layer=layer, kind=kind, rel_output_error=rel))
    return out


@torch.no_grad()
def apply_ablation(
    model, quantize_names: Iterable[str], scheme: Scheme | None = None
) -> dict[str, torch.Tensor]:
    """Replace each named linear's weight with its dequantized 4-bit weight, in place.

    Modules not named are untouched. Returns the original weights (cloned,
    same dtype and device) so the caller can restore them.
    """
    scheme = scheme or Scheme()
    names = sorted(set(quantize_names))
    modules = dict(model.named_modules())
    for n in names:
        if n not in modules or not isinstance(modules[n], torch.nn.Linear):
            raise ValueError(f"{n!r} is not a linear module of the model")
    hess: dict[str, torch.Tensor] = {}
    if _needs_hessian(scheme):
        if scheme.calib is None:
            raise ValueError("scheme.calib is required for method='gptq'")
        hess = _collect_hessians(model, scheme.calib, names)
    originals: dict[str, torch.Tensor] = {}
    for n in names:
        lin = modules[n]
        originals[n] = lin.weight.detach().clone()
        w32 = lin.weight.detach().to(torch.float32)
        wq = _quantize_weight(w32, scheme, hess.pop(n, None), n)
        lin.weight.copy_(wq.to(lin.weight.dtype))
    return originals


@torch.no_grad()
def restore_weights(model, originals: dict[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    for n, w in originals.items():
        modules[n].weight.copy_(w)


class _LogitsAdapter:
    """Gives a Hugging Face causal LM the ``forward(ids) -> logits`` that ``perplexity`` uses."""

    def __init__(self, model):
        self._m = model
        self._dev = _device_of(model)

    def forward(self, ids):
        with torch.no_grad():
            return self._m(input_ids=ids.to(self._dev), use_cache=False).logits


def ablate(
    model,
    tok,
    quantize_names: Iterable[str],
    scheme: Scheme | None = None,
    eval_cfg: dict | None = None,
) -> float:
    """WikiText perplexity with only ``quantize_names`` quantized and the rest fp16.

    ``eval_cfg`` keys are forwarded to :func:`quf.bench.suite.perplexity`
    (``dataset``, ``ctx_len``, ``text``, ``max_windows``). The model's weights are
    restored before returning.
    """
    from quf.bench.suite import perplexity

    originals = apply_ablation(model, quantize_names, scheme)
    was_training = model.training
    model.eval()
    try:
        return perplexity(_LogitsAdapter(model), tok, **(eval_cfg or {}))
    finally:
        restore_weights(model, originals)
        model.train(was_training)


def standard_subsets(model) -> dict[str, set[str]]:
    """The five standard ablation subsets of decoder linear module names.

    ``first`` and ``last`` are all linears of the first and last decoder layer,
    ``v_proj`` and ``down_proj`` are those modules in every layer, and ``rest``
    is every other decoder linear (in none of the four preceding subsets).
    """
    names = decoder_linear_names(model)
    n_layers = len(_decoder_layers(model))
    first = {n for n in names if _parse(n)[0] == 0}
    last = {n for n in names if _parse(n)[0] == n_layers - 1}
    v = {n for n in names if _parse(n)[1] == "v_proj"}
    down = {n for n in names if _parse(n)[1] == "down_proj"}
    rest = set(names) - first - last - v - down
    return {"first": first, "last": last, "v_proj": v, "down_proj": down, "rest": rest}
