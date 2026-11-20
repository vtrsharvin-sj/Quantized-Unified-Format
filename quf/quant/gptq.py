"""Native_GPTQ core: pure-PyTorch GPTQ for one linear layer (no GPTQModel).

Follows the published GPTQ procedure (Frantar et al., 2022): damp the Hessian,
take the upper Cholesky factor of its inverse, quantize columns in blocks and
propagate the rounding error to the not-yet-quantized columns. Act-order and
MSE clipping are optional.

The result is a :class:`~quf.quant.quantize.QuantizedWeight` in the canonical
Packed_Int4 layout, byte-compatible with ``quantize_groupwise``.
"""

from __future__ import annotations

import numpy as np
import torch

from quf.quant.dequant import reference_dequantize
from quf.quant.pack import pack_int4
from quf.quant.quantize import (
    dequantize_int8,
    quantize_int8,
    GROUP_SIZE,
    SYMMETRIC_ZERO,
    GroupSizeError,
    QuantizedWeight,
)
from quf.quant.clip import scale_zero_for_ratio, search_clip

MAX_DAMP_RETRIES = 5


class GPTQError(RuntimeError):
    """Raised when the Hessian is not positive definite after all damping retries."""

    def __init__(self, layer: str | None, message: str | None = None):
        self.layer = layer
        super().__init__(
            message
            or f"GPTQ failed for layer {layer!r}: Hessian is not positive definite "
            f"after {MAX_DAMP_RETRIES} damping retries"
        )


def default_device() -> torch.device:
    """CUDA when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HessianAccumulator:
    """Running estimate of ``H = 2/n * sum(X^T X)`` over calibration activations."""

    def __init__(self, in_features: int, device: torch.device | str | None = None):
        self.in_features = in_features
        self.device = torch.device(device) if device is not None else default_device()
        self.n_samples = 0
        self._h = torch.zeros(
            (in_features, in_features), dtype=torch.float32, device=self.device
        )

    def add_batch(self, x: torch.Tensor) -> None:
        """Add activations of shape ``[..., in_features]``; rows are samples."""
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"activation width {x.shape[-1]} != in_features {self.in_features}"
            )
        x = x.detach().reshape(-1, self.in_features).to(self.device, torch.float32)
        m = x.shape[0]
        if m == 0:
            return
        n = self.n_samples
        self._h *= n / (n + m)
        self.n_samples = n + m
        x = x * (2.0 / self.n_samples) ** 0.5
        self._h += x.t() @ x

    @property
    def hessian(self) -> torch.Tensor:
        return self._h


def _group_params(
    g: torch.Tensor, symmetric: bool, clip_search: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale (fp16-rounded, as float32) and zero point per row for a ``[out, gs]`` slice.

    With ``clip_search`` the clipping ratio of minimum squared error is chosen per row.
    """
    if clip_search:
        scale16, zero = search_clip(g, symmetric)
    else:
        scale16, zero = scale_zero_for_ratio(g, symmetric, 1.0)
    return scale16.to(torch.float32), zero


def act_order_permutation(
    hessian_diag: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Act-order permutation and g_idx from the Hessian diagonal.

    ``perm`` lists column indices by non-increasing diagonal (ties keep the lower
    index first). ``g_idx[c]`` (int32) is ``rank(c) // group_size`` where
    ``rank`` is the position of column ``c`` in ``perm``.
    """
    d = hessian_diag.detach().to("cpu", torch.float32)
    n = d.shape[0]
    if group_size <= 0 or n % group_size != 0:
        raise GroupSizeError(n, group_size)
    perm = torch.argsort(d, descending=True, stable=True)
    rank = torch.empty(n, dtype=torch.long)
    rank[perm] = torch.arange(n)
    return perm, (rank // group_size).to(torch.int32)


def _inverse_cholesky_upper(
    h: torch.Tensor, damp: float, layer: str | None
) -> torch.Tensor:
    """Upper Cholesky factor of ``H^-1``; raise damping x10 per retry, then GPTQError."""
    n = h.shape[0]
    diag_mean = torch.mean(torch.diag(h)).item()
    if not np.isfinite(diag_mean) or diag_mean <= 0:
        diag_mean = 1.0
    eye_idx = torch.arange(n, device=h.device)
    d = damp
    for _ in range(MAX_DAMP_RETRIES + 1):
        hd = h.clone()
        hd[eye_idx, eye_idx] += d * diag_mean
        try:
            chol = torch.linalg.cholesky(hd)
            if not torch.isfinite(chol).all():
                raise torch.linalg.LinAlgError("non-finite Cholesky factor")
            hinv = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(hinv, upper=True)
        except (torch.linalg.LinAlgError, RuntimeError):
            d *= 10.0
    raise GPTQError(layer)


def gptq_quantize(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    symmetric: bool = False,
    damp: float = 0.01,
    block_size: int = 128,
    layer_name: str | None = None,
    device: torch.device | str | None = None,
    act_order: bool = False,
    clip_search: bool = False,
) -> QuantizedWeight:
    """GPTQ-quantize a ``[out, in]`` weight given its ``[in, in]`` Hessian.

    With ``act_order`` columns are processed by descending Hessian diagonal and
    groups form over the permuted columns; the result carries ``g_idx`` and its
    ``qweight`` stays in original column order. ``clip_search`` picks the
    MSE-optimal clipping ratio per group.

    Runs on ``device`` (default: CUDA if available, else CPU). The returned
    tensors live on the CPU.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [out, in], got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise GroupSizeError(in_features, group_size)
    if out_features % 8 != 0:
        raise ValueError(f"output dimension {out_features} is not a multiple of 8")
    if tuple(hessian.shape) != (in_features, in_features):
        raise ValueError(
            f"hessian shape {tuple(hessian.shape)} does not match in_features {in_features}"
        )
    # Groups must lie inside one block so their parameters see the updated weights.
    if block_size < group_size or block_size % group_size != 0:
        block_size = group_size

    dev = torch.device(device) if device is not None else default_device()
    w = weight.detach().to(dev, torch.float32).clone()
    h = hessian.detach().to(dev, torch.float32).clone()

    dead = torch.diag(h) == 0
    idx = torch.arange(in_features, device=dev)
    h[idx, idx] = torch.where(dead, torch.ones_like(torch.diag(h)), torch.diag(h))
    w[:, dead] = 0

    perm = g_idx = None
    if act_order:
        perm, g_idx = act_order_permutation(torch.diag(h), group_size)
        perm = perm.to(dev)
        w = w[:, perm]
        h = h[perm][:, perm]

    hinv = _inverse_cholesky_upper(h, damp, layer_name)

    n_groups = in_features // group_size
    q_all = torch.zeros((out_features, in_features), dtype=torch.uint8, device=dev)
    scales = torch.ones((n_groups, out_features), dtype=torch.float32, device=dev)
    zeros = torch.full(
        (n_groups, out_features), float(SYMMETRIC_ZERO), dtype=torch.float32, device=dev
    )

    scale = zero = None
    for i1 in range(0, in_features, block_size):
        i2 = min(i1 + block_size, in_features)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group_size == 0:
                scale, zero = _group_params(w1[:, i : i + group_size], symmetric, clip_search)
                scales[col // group_size] = scale
                zeros[col // group_size] = zero
            wc = w1[:, i]
            q = (torch.round(wc / scale) + zero).clamp(0, 15)
            q_all[:, col] = q.to(torch.uint8)
            dq = (q - zero) * scale
            err = (wc - dq) / hinv1[i, i]
            w1[:, i:] -= err.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)
            err1[:, i] = err
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    if perm is not None:
        restored = torch.empty_like(q_all)
        restored[:, perm] = q_all  # back to original column order
        q_all = restored

    qweight = np.ascontiguousarray(pack_int4(q_all.cpu().numpy()).T)  # [in/8, out]
    qzeros = None
    if not symmetric:
        zero_u8 = zeros.to(torch.uint8).cpu().numpy()  # [n_groups, out]
        qzeros = torch.from_numpy(pack_int4(zero_u8))  # [n_groups, out/8]

    return QuantizedWeight(
        qweight=torch.from_numpy(qweight),
        scales=scales.to(torch.float16).cpu().contiguous(),
        qzeros=qzeros,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        g_idx=g_idx,
    )


# ---------------------------------------------------------------------------
# Sequential model quantization
# ---------------------------------------------------------------------------

# Linears of one decoder layer, in "true sequential" order: each group is
# quantized before the activations for the next group are collected.
_SEQUENTIAL_GROUPS: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj"),
    ("o_proj",),
    ("gate_proj", "up_proj"),
    ("down_proj",),
)


class _StopForward(Exception):
    """Raised by the capture hook to abort the model forward after layer 0."""


def _layer_output(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _decoder_layers(model) -> torch.nn.ModuleList:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError("model must expose decoder layers at model.model.layers")
    return layers


def _group_linears(layer: torch.nn.Module) -> list[list[tuple[str, torch.nn.Linear]]]:
    """Split a layer's ``nn.Linear`` modules into sequential groups by name suffix."""
    linears = {
        n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)
    }
    groups: list[list[tuple[str, torch.nn.Linear]]] = []
    used: set[str] = set()
    for suffixes in _SEQUENTIAL_GROUPS:
        group = []
        for suffix in suffixes:
            for n, m in linears.items():
                if n.split(".")[-1] == suffix and n not in used:
                    group.append((n, m))
                    used.add(n)
        if group:
            groups.append(group)
    rest = [(n, m) for n, m in linears.items() if n not in used]
    if rest:
        groups.append(rest)
    return groups


@torch.no_grad()
def quantize_model_sequential(
    model,
    calib,
    *,
    symmetric: bool = False,
    device: torch.device | str | None = None,
    group_size: int = GROUP_SIZE,
    damp: float = 0.01,
    act_order: bool = False,
    clip_search: bool = False,
    precision=None,
    int8_out: dict | None = None,
    progress=None,
) -> dict[str, QuantizedWeight]:
    """Quantize every decoder-layer linear of ``model`` with GPTQ, layer by layer.

    ``model`` is a Hugging Face causal LM (decoder layers at ``model.model.layers``)
    that may live on the CPU. ``calib`` is an int tensor or nested list of token ids
    with shape ``[n_samples, seq_len]``. Only one decoder layer is on ``device``
    at a time, so large models fit in limited GPU memory. Each layer receives the
    outputs of the already quantized preceding layers. After a linear is quantized,
    its weight in ``model`` is replaced by the dequantized values so later groups
    and layers see the quantization error.

    Returns a mapping from module name (``model.layers.0.self_attn.q_proj``) to
    :class:`QuantizedWeight`. ``progress(i, n_layers)`` is called after each layer.

    ``precision(name)`` (optional) returns ``"int4"``, ``"int8"`` or ``"fp16"`` for a
    full module name. ``int8`` modules are quantized per output channel into
    ``int8_out[name] = (int8 [out, in], fp16 scales [out])``; ``fp16`` modules are left
    unchanged and absent from the result. Both are skipped by the Hessian pass.
    """
    dev = torch.device(device) if device is not None else default_device()
    ids = torch.as_tensor(calib, dtype=torch.long)
    if ids.ndim != 2 or ids.shape[0] == 0:
        raise ValueError("calib must be a non-empty [n_samples, seq_len] token tensor")

    layers = _decoder_layers(model)
    base = model.model
    was_training = model.training
    model.eval()

    # Capture the inputs of layer 0 (embedding output) and its call arguments.
    pre_modules = [
        getattr(base, a) for a in ("embed_tokens", "rotary_emb") if hasattr(base, a)
    ]
    for m in pre_modules:
        m.to(dev)
    captured: list[torch.Tensor] = []
    call: dict = {}

    def _capture(_module, args, kwargs):
        kwargs = dict(kwargs)
        hs = args[0] if args else kwargs.pop("hidden_states")
        if not call:
            call["args"] = tuple(args[1:])
            call["kwargs"] = {
                k: v
                for k, v in kwargs.items()
                if k not in ("past_key_value", "past_key_values")
            }
            call["kwargs"]["use_cache"] = False
        captured.append(hs.detach())
        raise _StopForward

    handle = layers[0].register_forward_pre_hook(_capture, with_kwargs=True)
    try:
        for i in range(ids.shape[0]):
            try:
                model(input_ids=ids[i : i + 1].to(dev), use_cache=False)
            except _StopForward:
                pass
    finally:
        handle.remove()
        for m in pre_modules:
            m.to("cpu")
    if len(captured) != ids.shape[0]:
        raise RuntimeError("failed to capture decoder layer inputs")
    inps = captured

    quantized: dict[str, QuantizedWeight] = {}
    n_layers = len(layers)
    for li, layer in enumerate(layers):
        layer.to(dev)
        prefix = f"model.layers.{li}."

        def run(hs):
            return _layer_output(layer(hs, *call["args"], **call["kwargs"]))

        for group in _group_linears(layer):
            # Only int4 modules need a Hessian; int8 and fp16 modules skip it.
            accs = {
                n: HessianAccumulator(m.in_features, dev)
                for n, m in group
                if precision is None or precision(prefix + n) == "int4"
            }
            handles = [
                m.register_forward_hook(
                    lambda _m, inp, _out, acc=accs[n]: acc.add_batch(inp[0])
                )
                for n, m in group
                if n in accs
            ]
            try:
                for hs in inps if accs else ():
                    run(hs)
            finally:
                for h in handles:
                    h.remove()
            for n, m in group:
                prec = precision(prefix + n) if precision is not None else "int4"
                if prec == "fp16":
                    continue  # stays dense fp16; the converter stores it as a plain weight
                if prec == "int8":
                    if int8_out is None:
                        raise ValueError("int8 modules need the int8_out mapping")
                    q8, s8 = quantize_int8(m.weight)
                    int8_out[prefix + n] = (q8.numpy(), s8.numpy())
                    m.weight.copy_(dequantize_int8(q8, s8).to(m.weight.device, m.weight.dtype))
                    continue
                qw = gptq_quantize(
                    m.weight,
                    accs[n].hessian,
                    group_size=group_size,
                    symmetric=symmetric,
                    damp=damp,
                    layer_name=prefix + n,
                    device=dev,
                    act_order=act_order,
                    clip_search=clip_search,
                )
                quantized[prefix + n] = qw
                m.weight.copy_(
                    reference_dequantize(qw, qw.g_idx).to(m.weight.device, m.weight.dtype)
                )
            del accs

        # Outputs of the quantized layer become the next layer's inputs.
        inps = [run(hs).detach() for hs in inps]
        layer.to("cpu")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if progress is not None:
            progress(li + 1, n_layers)

    model.train(was_training)
    return quantized
