"""Qwen2 decoder and KV_Cache. All projections go through QUFLinear."""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from quf.format.constants import G_IDX_SUFFIX, INT8_SUFFIX
from quf.format.reader import QUFReader
from quf.quant.quantize import QuantizedWeight
from quf.runtime.backends.base import Int8Weight
from quf.runtime.errors import GPUMemoryError
from quf.runtime.linear import QUFLinear

logger = logging.getLogger(__name__)


def _available_gpu_bytes() -> int:
    """Free GPU memory in bytes, or 0 when it cannot be queried."""
    try:
        return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001
        return 0

_BACKEND_CLASSES = {
    "cpu": ("quf.runtime.backends.cpu", "CPUBackend"),
    "torch_gpu": ("quf.runtime.backends.torch_gpu", "TorchGPUBackend"),
    "torch_fused": ("quf.runtime.backends.torch_fused", "TorchGPUFusedBackend"),
    "marlin": ("quf.runtime.backends.marlin", "MarlinBackend"),
}


def _make_backend(name: str):
    try:
        module, cls = _BACKEND_CLASSES[name]
    except KeyError:
        raise ValueError(f"unknown backend: {name!r}") from None
    return getattr(importlib.import_module(module), cls)()


def estimate_cached_bytes(path) -> int | None:
    """Bytes the cached backend needs on the GPU: fp16 dequantized weights (4x the packed
    qweight bytes) plus the other tensors and a KV_Cache estimate. None if unreadable."""
    try:
        with QUFReader(path) as reader:
            h = reader.header
            cfg = h.model_config
            def cached(t) -> int:
                if t.name.endswith(".qweight"):
                    return t.nbytes * 4  # int4 -> fp16
                if t.name.endswith(INT8_SUFFIX):
                    return t.nbytes * 2  # int8 -> fp16
                return t.nbytes

            total = sum(cached(t) for t in h.tensors)
            heads = int(cfg["num_attention_heads"])
            kv = (
                2 * int(cfg["num_hidden_layers"]) * int(cfg.get("num_key_value_heads", heads))
                * int(cfg.get("head_dim") or int(cfg["hidden_size"]) // heads)
                * int(cfg["max_position_embeddings"]) * 2
            )
            return int(total + kv)
    except Exception:  # noqa: BLE001
        return None


class KVCache:
    """Per-layer key/value cache. Tensors have shape ``[batch, kv_heads, seq, head_dim]``."""

    def __init__(self, num_layers: int):
        self._k: list[torch.Tensor | None] = [None] * num_layers
        self._v: list[torch.Tensor | None] = [None] * num_layers

    @property
    def length(self) -> int:
        k = self._k[0] if self._k else None
        return 0 if k is None else int(k.shape[2])

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if self._k[layer] is None:
            self._k[layer], self._v[layer] = k, v
        else:
            self._k[layer] = torch.cat([self._k[layer], k], dim=2)
            self._v[layer] = torch.cat([self._v[layer], v], dim=2)

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self._k[layer], self._v[layer]
        if k is None or v is None:
            raise ValueError(f"layer {layer} has no cached keys")
        return k, v


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class _Layer:
    pass


class QUFModel:
    def __init__(self, header, device: torch.device, backend_name: str):
        self.header = header
        self.device = device
        self.backend_name = backend_name
        cfg = header.model_config
        self.config = cfg
        self.hidden_size = int(cfg["hidden_size"])
        self.num_layers = int(cfg["num_hidden_layers"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.num_kv_heads = int(cfg.get("num_key_value_heads", self.num_heads))
        self.head_dim = int(cfg.get("head_dim") or self.hidden_size // self.num_heads)
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(cfg.get("rope_theta", 10000.0))
        self.max_position_embeddings = int(cfg["max_position_embeddings"])
        self.dtype = torch.float32 if device.type == "cpu" else torch.float16
        self.layers: list[_Layer] = []
        self.embed: torch.Tensor
        self.final_norm: torch.Tensor
        self.lm_head: torch.Tensor | QUFLinear

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(
        cls,
        path,
        device_option: str = "cpu",
        backend: str | None = None,
        backend_option: str = "auto",
        prefer_fused: bool = False,
    ) -> "QUFModel":
        """Load a QUF file. ``backend`` names the Backend explicitly (``cpu``, ``torch_gpu``,
        ``torch_fused``, ``marlin``); otherwise the Backend_Selector chooses from
        ``device_option`` (auto, cpu, cuda) and ``backend_option`` (auto, cached, fused, marlin).
        Raises ``NoGPUError`` when a GPU is required and none is available. When the selector
        picked the fused backend and loading it fails, the cached backend is used instead."""
        selected_by_selector = backend is None
        if backend is None:
            from quf.device import detect_devices  # noqa: PLC0415
            from quf.runtime.backends.torch_fused import fused_self_test  # noqa: PLC0415
            from quf.runtime.selector import select_backend  # noqa: PLC0415

            info = detect_devices()
            cached_bytes, free_bytes = (None, None)
            if info.cuda_available:
                cached_bytes, free_bytes = estimate_cached_bytes(path), _available_gpu_bytes() or None
            backend = select_backend(
                device_option,  # type: ignore[arg-type]
                info,
                backend_option=backend_option,  # type: ignore[arg-type]
                cached_bytes=cached_bytes,
                free_bytes=free_bytes,
                prefer_fused=prefer_fused,
                fused_self_test=fused_self_test,
            ).backend
        if backend == "marlin":
            from quf.runtime.backends.marlin import require_marlin  # noqa: PLC0415

            require_marlin()
        try:
            return cls._load_with(path, backend)
        except torch.cuda.OutOfMemoryError:
            raise
        except GPUMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            if backend == "torch_fused" and selected_by_selector:
                logger.debug("fused backend load failure: %s: %s", type(exc).__name__, exc)
                logger.info("fused backend failed to load, using the cached backend")
                return cls._load_with(path, "torch_gpu")
            raise

    @classmethod
    def _load_with(cls, path, backend: str) -> "QUFModel":
        device = torch.device("cpu" if backend == "cpu" else "cuda")
        be = _make_backend(backend)
        with QUFReader(path) as reader:
            model = cls(reader.header, device, backend)
            try:
                model._load_tensors(reader, be)
            except torch.cuda.OutOfMemoryError as exc:
                required = model.estimate_gpu_bytes()
                available = _available_gpu_bytes()
                model.layers.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise GPUMemoryError(required, available) from exc
        return model
    def estimate_gpu_bytes(self) -> int:
        """Tensor_Table bytes plus a KV_Cache estimate for the full context length."""
        table = sum(t.nbytes for t in self.header.tensors)
        kv = 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.max_position_embeddings * 2
        return int(table + kv)

    def _load_tensors(self, reader: QUFReader, be) -> None:
        header = reader.header
        symmetric = bool(header.quantization.get("symmetric", False))
        group_size = int(header.quantization.get("group_size", 128))
        names = set(reader.tensor_names())

        def dense(name: str) -> torch.Tensor:
            t = torch.from_numpy(np.array(reader.tensor(name)))
            return t.to(self.device)

        def norm(name: str) -> torch.Tensor:
            return dense(name).to(self.dtype)

        def linear(prefix: str) -> QUFLinear:
            bias = None
            if f"{prefix}.bias" in names:
                bias = dense(f"{prefix}.bias").to(self.dtype)
            # The precision follows the tensors present (header.modules lists the same).
            if f"{prefix}{INT8_SUFFIX}" in names:
                q = torch.from_numpy(np.array(reader.tensor(f"{prefix}{INT8_SUFFIX}")))
                s = torch.from_numpy(np.array(reader.tensor(f"{prefix}.scales")))
                return QUFLinear(Int8Weight(q.to(self.device), s.to(self.device)), be, self.device, bias=bias)
            if f"{prefix}.qweight" not in names and f"{prefix}.weight" in names:
                w = torch.from_numpy(np.array(reader.tensor(f"{prefix}.weight")))
                return QUFLinear(w.to(self.device), be, self.device, bias=bias)
            qweight = torch.from_numpy(np.array(reader.tensor(f"{prefix}.qweight")))
            scales = torch.from_numpy(np.array(reader.tensor(f"{prefix}.scales")))
            qzeros = None
            if not symmetric:
                qzeros = torch.from_numpy(np.array(reader.tensor(f"{prefix}.qzeros")))
            g_idx = None
            if f"{prefix}{G_IDX_SUFFIX}" in names:
                g_idx = torch.from_numpy(np.array(reader.tensor(f"{prefix}{G_IDX_SUFFIX}"))).to(torch.long)
            qw = QuantizedWeight(
                qweight=qweight.to(self.device),
                scales=scales.to(self.device),
                qzeros=None if qzeros is None else qzeros.to(self.device),
                group_size=group_size,
                in_features=int(qweight.shape[0]) * 8,
                out_features=int(qweight.shape[1]),
            )
            return QUFLinear(
                qw, be, self.device, bias=bias, g_idx=None if g_idx is None else g_idx.to(self.device)
            )

        self.embed = dense("model.embed_tokens.weight")
        self.final_norm = norm("model.norm.weight")
        if "lm_head.qweight" in names or f"lm_head{INT8_SUFFIX}" in names:
            self.lm_head = linear("lm_head")
        elif "lm_head.weight" in names:
            self.lm_head = dense("lm_head.weight")
        else:
            self.lm_head = self.embed  # tied embeddings

        # Dense lm_head weight in the compute dtype, converted once (not per token). On the GPU
        # this is the fp16 tensor itself (a view for tied embeddings, no extra memory). On the
        # CPU it is an fp32 copy (2x the fp16 bytes) so that decode does no per-token cast.
        self._lm_head_weight = None
        if not isinstance(self.lm_head, QUFLinear):
            self._lm_head_weight = self.lm_head.to(self.dtype)

        for i in range(self.num_layers):
            p = f"model.layers.{i}"
            layer = _Layer()
            layer.input_norm = norm(f"{p}.input_layernorm.weight")
            layer.post_norm = norm(f"{p}.post_attention_layernorm.weight")
            layer.q = linear(f"{p}.self_attn.q_proj")
            layer.k = linear(f"{p}.self_attn.k_proj")
            layer.v = linear(f"{p}.self_attn.v_proj")
            layer.o = linear(f"{p}.self_attn.o_proj")
            layer.gate = linear(f"{p}.mlp.gate_proj")
            layer.up = linear(f"{p}.mlp.up_proj")
            layer.down = linear(f"{p}.mlp.down_proj")
            self.layers.append(layer)

    # -- forward ----------------------------------------------------------

    def new_cache(self) -> KVCache:
        return KVCache(self.num_layers)

    def _rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def _attention(self, layer: _Layer, idx: int, x, cos, sin, cache: KVCache | None, past: int):
        b, t, _ = x.shape
        q = layer.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = layer.k(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = layer.v(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if cache is not None:
            cache.append(idx, k, v)
            k, v = cache.get(idx)
        s = k.shape[2]
        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # Query i (absolute position past + i) may attend to keys j <= past + i.
        mask = torch.arange(s, device=x.device)[None, :] <= (past + torch.arange(t, device=x.device))[:, None]
        scores = (q.to(torch.float32) @ k.to(torch.float32).transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        return layer.o(out)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Return logits ``[batch, seq, vocab]`` (float32) for token ids ``[batch, seq]``.

        With a cache, ``ids`` holds only the new tokens and the cache is extended.
        """
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(self.device)
        t = ids.shape[1]
        past = cache.length if cache is not None else 0
        positions = torch.arange(past, past + t, device=self.device)
        cos, sin = self._rope(positions)

        h = F.embedding(ids, self.embed).to(self.dtype)
        for i, layer in enumerate(self.layers):
            h = h + self._attention(layer, i, _rms_norm(h, layer.input_norm, self.eps), cos, sin, cache, past)
            m = _rms_norm(h, layer.post_norm, self.eps)
            h = h + layer.down(F.silu(layer.gate(m)) * layer.up(m))
        h = _rms_norm(h, self.final_norm, self.eps)

        if isinstance(self.lm_head, QUFLinear):
            logits = self.lm_head(h)
        else:
            # fp16 matmul on the GPU (fp32 accumulation inside the kernel), fp32 on the CPU.
            logits = h.to(self.dtype) @ self._lm_head_weight.t()
        return logits.to(torch.float32)
