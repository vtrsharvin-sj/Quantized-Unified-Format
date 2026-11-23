"""Qwen2 decoder and KV_Cache. All projections go through QUFLinear."""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from quf.format.constants import EMBED_Q4_SUFFIX, EMBED_Q8_SUFFIX, EMBED_SCALES_SUFFIX, G_IDX_SUFFIX, INT8_SUFFIX
from quf.format.reader import QUFReader
from quf.quant.quantize import QuantizedWeight
from quf.runtime.backends.base import Int8Weight
from quf.runtime.embedding import RuntimeEmbedding
from quf.runtime.errors import GPUMemoryError
from quf.runtime.linear import FusedProjection, QUFLinear

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
    "triton_int4": ("quf.runtime.backends.triton_int4", "TritonInt4Backend"),
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


class CacheLengthError(ValueError):
    """Tokens to cache would exceed the Static_KV_Cache length."""

    def __init__(self, needed: int, cache_len: int):
        self.needed = needed
        self.cache_len = cache_len
        super().__init__(
            f"sequence length ({needed}) exceeds the static KV cache length ({cache_len})"
        )


class StaticKVCache:
    """Preallocated per-layer key/value cache of shape ``[1, kv_heads, max_len, head_dim]``.

    Allocated once and updated in place with ``index_copy_``. ``length`` is the number of
    cached positions. ``token`` and ``position`` are fixed device tensors that the
    Decode_Step reads, so its tensor shapes are identical on every call.
    """

    def __init__(self, num_layers, kv_heads, head_dim, max_len, device, dtype):
        self.max_len = int(max_len)
        shape = (1, kv_heads, self.max_len, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.token = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.position = torch.zeros((1,), dtype=torch.long, device=device)
        self.length = 0

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int) -> None:
        """Copy ``k``/``v`` (``[1, kv_heads, t, head_dim]``) into positions ``start..start+t``."""
        t = int(k.shape[2])
        if start + t > self.max_len:
            raise CacheLengthError(start + t, self.max_len)
        idx = torch.arange(start, start + t, device=k.device)
        self.k[layer].index_copy_(2, idx, k.to(self.k[layer].dtype))
        self.v[layer].index_copy_(2, idx, v.to(self.v[layer].dtype))


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
        self.fuse_projections = True
        # Fused_Decode_Kernels (task 33.2): set by _load_tensors for triton_int4 when the
        # Triton_Gate passes. False keeps the 0.3.0 decode path.
        self.fused_decode = False
        self._add_rmsnorm = None
        self._fast_layers: list = []
        self._silu_mul_scatter = None
        self._fast_attention = False
        self._rope_kv_write = None
        self._flash_decode = None
        self._rope_tables: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.layers: list[_Layer] = []
        self.embed: torch.Tensor | None
        self.qembed: RuntimeEmbedding | None = None  # Quantized_Embedding (format v3), else None
        self.qlm_head: RuntimeEmbedding | None = None
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
        fuse_projections: bool = True,
        cuda_graph: bool | None = None,
    ) -> "QUFModel":
        """Load a QUF file. ``backend`` names the Backend explicitly (``cpu``, ``torch_gpu``,
        ``torch_fused``, ``marlin``); otherwise the Backend_Selector chooses from
        ``device_option`` (auto, cpu, cuda) and ``backend_option`` (auto, cached, fused, marlin, int4).
        Raises ``NoGPUError`` when a GPU is required and none is available. When the selector
        picked the fused backend and loading it fails, the cached backend is used instead.
        `cuda_graph` False, or QUF_NO_CUDA_GRAPH set (None), makes `auto` prefer torch_gpu over
        triton_int4, since eager int4 decoding is slower."""
        selected_by_selector = backend is None
        if backend is None:
            from quf.device import detect_devices  # noqa: PLC0415
            from quf.runtime.backends.torch_fused import fused_self_test  # noqa: PLC0415
            from quf.runtime.cuda_graph import cuda_graph_disabled_by_env  # noqa: PLC0415
            from quf.runtime.selector import int4_probe, select_backend  # noqa: PLC0415

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
                int4_probe=int4_probe,
                cuda_graph=(not cuda_graph_disabled_by_env()) if cuda_graph is None else bool(cuda_graph),
            ).backend
        if backend == "marlin":
            from quf.runtime.backends.marlin import require_marlin  # noqa: PLC0415

            require_marlin()
        try:
            return cls._load_with(path, backend, fuse_projections)
        except torch.cuda.OutOfMemoryError:
            raise
        except GPUMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            if backend == "triton_int4" and selected_by_selector and backend_option == "auto":
                logger.debug("int4 backend load failure: %s: %s", type(exc).__name__, exc)
                logger.info("int4 backend failed to load, using the cached backend")
                return cls._load_with(path, "torch_gpu", fuse_projections)
            if backend == "torch_fused" and selected_by_selector:
                logger.debug("fused backend load failure: %s: %s", type(exc).__name__, exc)
                logger.info("fused backend failed to load, using the cached backend")
                return cls._load_with(path, "torch_gpu", fuse_projections)
            raise

    @classmethod
    def _load_with(cls, path, backend: str, fuse_projections: bool = True) -> "QUFModel":
        device = torch.device("cpu" if backend == "cpu" else "cuda")
        if backend == "triton_int4":
            from quf.runtime.backends.triton_int4 import triton_int4_self_test  # noqa: PLC0415

            ok, reason = triton_int4_self_test()
            if not ok:
                raise RuntimeError(f"triton_int4 backend unavailable: {reason}")
        be = _make_backend(backend)
        with QUFReader(path) as reader:
            model = cls(reader.header, device, backend)
            model.fuse_projections = fuse_projections
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

        routed: list = []

        def linear(prefix: str) -> QUFLinear:
            m = _linear(prefix)
            routed.append(m.prepared)
            return m

        def _linear(prefix: str) -> QUFLinear:
            bias = None
            if f"{prefix}.bias" in names:
                bias = dense(f"{prefix}.bias").to(self.dtype)
            # The precision follows the tensors present (header.modules lists the same).
            if f"{prefix}{INT8_SUFFIX}" in names:
                q = torch.from_numpy(np.array(reader.tensor(f"{prefix}{INT8_SUFFIX}")))
                s = torch.from_numpy(np.array(reader.tensor(f"{prefix}.scales")))
                return QUFLinear(Int8Weight(q.to(self.device), s.to(self.device)), be, self.device, bias=bias, name=prefix)
            if f"{prefix}.qweight" not in names and f"{prefix}.weight" in names:
                w = torch.from_numpy(np.array(reader.tensor(f"{prefix}.weight")))
                return QUFLinear(w.to(self.device), be, self.device, bias=bias, name=prefix)
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
                qw, be, self.device, bias=bias, name=prefix,
                g_idx=None if g_idx is None else g_idx.to(self.device),
            )

        def quantized_table(name: str) -> RuntimeEmbedding | None:
            """A Quantized_Embedding stored as ``<name>.embed_q4|q8`` + ``.embed_scales``, kept packed."""
            for suffix, bits in ((EMBED_Q4_SUFFIX, 4), (EMBED_Q8_SUFFIX, 8)):
                if name + suffix in names:
                    spec = header.embedding if name == "model.embed_tokens" else header.lm_head_embedding
                    gs = int((spec or {}).get("group_size", 64))
                    return RuntimeEmbedding(
                        reader.tensor(name + suffix), reader.tensor(name + EMBED_SCALES_SUFFIX),
                        bits, gs, self.device, use_kernel=getattr(be, "name", None) == "triton_int4",
                    )
            return None

        self.qembed = quantized_table("model.embed_tokens")
        self.embed = None if self.qembed is not None else dense("model.embed_tokens.weight")
        self.final_norm = norm("model.norm.weight")
        self.qlm_head = quantized_table("lm_head")
        if self.qlm_head is not None:
            self.lm_head = None  # logits come from the quantized lm_head table
        elif "lm_head.qweight" in names or f"lm_head{INT8_SUFFIX}" in names:
            self.lm_head = linear("lm_head")
        elif "lm_head.weight" in names:
            self.lm_head = dense("lm_head.weight")
        elif self.qembed is not None:
            self.qlm_head = self.qembed  # tied: the head reads the same quantized tensor
            self.lm_head = None
        else:
            self.lm_head = self.embed  # tied embeddings

        # Dense lm_head weight in the compute dtype, converted once (not per token). On the GPU
        # this is the fp16 tensor itself (a view for tied embeddings, no extra memory). On the
        # CPU it is an fp32 copy (2x the fp16 bytes) so that decode does no per-token cast.
        self._lm_head_weight = None
        if self.lm_head is not None and not isinstance(self.lm_head, QUFLinear):
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
            # Fused_Projection (cpu and torch_gpu only): one matmul for q/k/v and one for gate/up.
            layer.qkv = layer.gate_up = None
            if self.fuse_projections:
                if FusedProjection.can_fuse([layer.q, layer.k, layer.v]):
                    layer.qkv = FusedProjection([layer.q, layer.k, layer.v])
                if FusedProjection.can_fuse([layer.gate, layer.up]):
                    layer.gate_up = FusedProjection([layer.gate, layer.up])
            self.layers.append(layer)
        if getattr(be, "name", None) == "triton_int4":
            from quf.runtime.kernels.dispatch import route_summary  # noqa: PLC0415

            logger.debug("%s", route_summary(routed))
            self._enable_fused_decode()

    def _enable_fused_decode(self) -> None:
        """Turn on the Fused_Decode_Kernels when the Triton_Gate passes (never imports
        quf.runtime.kernels.fused otherwise). QUF_NO_FUSED=1 keeps the 0.3.0 path."""
        ok, reason = self._fused_gate()
        if not ok:
            logger.info("fused decode kernels unavailable (%s), using the 0.3.0 path", reason)
            self._disable_embed_kernels()
            return
        try:
            from quf.runtime.kernels.fused import add_rmsnorm  # noqa: PLC0415

            self._add_rmsnorm = add_rmsnorm
            self._build_fast_layers()
        except Exception as exc:  # noqa: BLE001
            self._add_rmsnorm, self._fast_layers = None, []
            self._silu_mul_scatter = self._rope_kv_write = self._flash_decode = None
            self._fast_attention = False
            self._disable_embed_kernels()
            logger.info("fused decode kernels failed to load (%s: %s), using the 0.3.0 path", type(exc).__name__, exc)
            return
        self.fused_decode = True

    @staticmethod
    def _fused_gate() -> tuple[bool, str]:
        """The one gate every Fused_Decode_Kernel goes through: QUF_NO_FUSED, then the Triton_Gate
        (QUF_NO_TRITON, CUDA, Triton import). Never imports a kernel module."""
        import os  # noqa: PLC0415

        if os.environ.get("QUF_NO_FUSED", "") == "1":
            return False, "QUF_NO_FUSED=1"
        from quf.runtime.kernels.triton_kernels import triton_available  # noqa: PLC0415

        return triton_available()

    def _disable_embed_kernels(self) -> None:
        for emb in (self.qembed, self.qlm_head):
            if emb is not None:
                emb.use_kernel = False

    def _disable_fused(self, reason: str) -> None:
        """Return the whole model to the 0.3.0 decode path. Logs one INFO line, once."""
        if not self.fused_decode:
            return
        self.fused_decode = False
        self._disable_embed_kernels()
        logger.info("fused decode kernels failed (%s), using the 0.3.0 path", reason)

    def _build_fast_layers(self) -> None:
        """Per layer, the fast projections (qkv, o, gate_up, down): j-major int4 GEMV, int8 GEMV, or
        cuBLAS for an fp16 module, in any mix (a Mixed_Group writes one scattered buffer). None
        keeps that layer on the QUFLinear path; the reason is logged once at DEBUG."""
        from quf.runtime.kernels.decode_gemv import fast_group_of, silu_mul_scatter  # noqa: PLC0415

        self._silu_mul_scatter = silu_mul_scatter

        rep = self.num_heads // self.num_kv_heads
        hd = self.head_dim
        self._fast_attention = (
            self.num_heads % self.num_kv_heads == 0 and rep <= 16 and hd >= 16 and hd & (hd - 1) == 0
        )
        if self._fast_attention:
            from quf.runtime.kernels.attention import flash_decode, rope_kv_write  # noqa: PLC0415

            self._rope_kv_write, self._flash_decode = rope_kv_write, flash_decode
        self._fast_layers = []
        logged: set[str] = set()
        for idx, layer in enumerate(self.layers):
            mods, why = [], ""
            for label, module in (("qkv", layer.qkv), ("o", layer.o), ("gate_up", layer.gate_up), ("down", layer.down)):
                if module is None:
                    fast, why = None, "projection fusion is off" if not self.fuse_projections else "members cannot be fused"
                else:
                    fast, why = fast_group_of(module)
                if fast is None:
                    reason = f"{label}: {why}"
                    if reason not in logged:  # one DEBUG line per reason, not per layer
                        logged.add(reason)
                        logger.debug("fast path off for layer %d (%s); that layer uses the unfused path", idx, reason)
                    break
                mods.append(fast)
            self._fast_layers.append(tuple(mods) if len(mods) == 4 else None)
    # -- forward ----------------------------------------------------------

    def new_cache(self) -> KVCache:
        return KVCache(self.num_layers)

    def new_static_cache(self, max_len: int | None = None) -> StaticKVCache:
        """Allocate a Static_KV_Cache of ``max_len`` positions (default: the context length)."""
        n = self.max_position_embeddings if max_len is None else int(max_len)
        return StaticKVCache(
            self.num_layers, self.num_kv_heads, self.head_dim, n, self.device, self.dtype
        )

    def _rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _qkv(layer: _Layer, x):
        if layer.qkv is not None:
            return layer.qkv(x)
        return layer.q(x), layer.k(x), layer.v(x)

    @staticmethod
    def _mlp(layer: _Layer, m):
        if layer.gate_up is not None:
            gate, up = layer.gate_up(m)
        else:
            gate, up = layer.gate(m), layer.up(m)
        return layer.down(F.silu(gate) * up)

    def _attention(self, layer: _Layer, idx: int, x, cos, sin, cache: KVCache | None, past: int):
        b, t, _ = x.shape
        q, k, v = self._qkv(layer, x)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if isinstance(cache, StaticKVCache):
            cache.write(idx, k, v, past)
            k = cache.k[idx][:, :, : past + t]
            v = cache.v[idx][:, :, : past + t]
        elif cache is not None:
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

        h = self._embed(ids)
        for i, layer in enumerate(self.layers):
            h = h + self._attention(layer, i, _rms_norm(h, layer.input_norm, self.eps), cos, sin, cache, past)
            m = _rms_norm(h, layer.post_norm, self.eps)
            h = h + self._mlp(layer, m)
        if isinstance(cache, StaticKVCache):
            cache.length = past + t
        return self._head(h)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        if self.qembed is not None:
            return self.qembed.lookup(ids).to(self.dtype)
        return F.embedding(ids, self.embed).to(self.dtype)

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        h = _rms_norm(h, self.final_norm, self.eps)
        if self.qlm_head is not None:
            logits = self.qlm_head.logits(h)
        elif isinstance(self.lm_head, QUFLinear):
            logits = self.lm_head(h)
        else:
            # fp16 matmul on the GPU (fp32 accumulation inside the kernel), fp32 on the CPU.
            logits = h.to(self.dtype) @ self._lm_head_weight.t()
        return logits.to(torch.float32)

    def _attention_static(self, layer: _Layer, idx: int, x, cos, sin, cache: StaticKVCache, position, mask):
        """Attention over the whole static cache; the mask hides positions after ``position``."""
        q, k, v = self._qkv(layer, x)
        return layer.o(self._attention_core(idx, q, k, v, cos, sin, cache, position, mask))

    def _attention_core(self, idx: int, q, k, v, cos, sin, cache: StaticKVCache, position, mask):
        """q/k/v ``[1, 1, heads * head_dim]`` projections -> attention output before ``o``."""
        b, t, _ = q.shape
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        cache.k[idx].index_copy_(2, position, k.to(cache.k[idx].dtype))
        cache.v[idx].index_copy_(2, position, v.to(cache.v[idx].dtype))
        k, v = cache.k[idx], cache.v[idx]
        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scores = (q.to(torch.float32) @ k.to(torch.float32).transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(v.dtype)
        return (attn @ v).transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)

    @torch.no_grad()
    def decode_step(self, cache: StaticKVCache) -> torch.Tensor:
        """One fixed-shape decode step. Reads ``cache.token`` ``[1, 1]`` and ``cache.position``
        ``[1]``, writes that position's key/value in place, returns logits ``[1, 1, vocab]``.
        All tensor shapes are identical on every call; no host sync happens here."""
        position = cache.position
        h = self._embed(cache.token)
        if self.fused_decode:
            try:
                return self._decode_layers_fused(h, cache, position)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:  # noqa: BLE001
                if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                    raise  # never switch paths inside a capture; the decoder retries after the failure
                # All-or-nothing: the whole model returns to the 0.3.0 path. Cache slots the fused
                # step already wrote are rewritten by the same position below.
                self._disable_fused(f"{type(exc).__name__}: {exc}")
        cos, sin = self._rope(position)
        mask = torch.arange(cache.max_len, device=position.device)[None, :] <= position[:, None]
        for i, layer in enumerate(self.layers):
            h = h + self._attention_static(
                layer, i, _rms_norm(h, layer.input_norm, self.eps), cos, sin, cache, position, mask
            )
            m = _rms_norm(h, layer.post_norm, self.eps)
            h = h + self._mlp(layer, m)
        return self._head(h)

    def _decode_layers_fused(self, h, cache, position) -> torch.Tensor:
        """Decode_Step body of the Fused_Decode_Kernels (task 33.2).

        Same arithmetic as the 0.3.0 loop: fp16 adds, fp32 norm reduction. Layers with four int4
        projections run the j-major GEMV: each producer (add+RMSNorm, attention output, SiLU*mul)
        scatters its output into the consumer's activation layout, so the GEMV reads only
        contiguous vectors. Other layers keep the QUFLinear calls."""
        from quf.runtime.kernels.decode_gemv import scatter_to  # noqa: PLC0415

        add_rmsnorm = self._add_rmsnorm
        fast = self._fast_layers
        eps = self.eps
        torch_rope = None  # cos/sin/mask of the torch attention, built once and only when a layer needs them
        if self._fast_attention:
            tab_cos, tab_sin = self._rope_table(cache.max_len)
        n, _ = add_rmsnorm(h, None, self.layers[0].input_norm, eps, out_index=fast[0][0].index if fast[0] else None)
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            f = fast[i]
            if f is None or not self._fast_attention:
                if torch_rope is None:
                    cos, sin = self._rope(position)
                    mask = torch.arange(cache.max_len, device=position.device)[None, :] <= position[:, None]
                    torch_rope = (cos, sin, mask)
                cos, sin, mask = torch_rope
            if f is None:
                attn = self._attention_static(layer, i, n, cos, sin, cache, position, mask)
                m, h = add_rmsnorm(attn, h, layer.post_norm, eps)
                out = self._mlp(layer, m)
            else:
                qkv_l, o_l, gu_l, down_l = f
                y = qkv_l(n.view(1, -1))
                if self._fast_attention:
                    q = self._rope_kv_write(y, tab_cos, tab_sin, position, cache.k[i], cache.v[i],
                                            self.num_heads, self.num_kv_heads)
                    att = self._flash_decode(q, cache.k[i], cache.v[i], position, self.num_heads,
                                             self.num_kv_heads, out_index=o_l.index)
                    attn = o_l(att)
                else:
                    wq, wk = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
                    q, k, v = y[:, :wq], y[:, wq : wq + wk], y[:, wq + wk :]
                    att = self._attention_core(i, q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), cos, sin, cache, position, mask)
                    attn = o_l(att.view(1, -1) if o_l.index is None else scatter_to(att.view(1, -1), o_l.index))
                m, h = add_rmsnorm(attn, h, layer.post_norm, eps, out_index=gu_l.index)
                out = down_l(self._silu_mul_scatter(gu_l(m.view(1, -1)), down_l.scatter_index))
                out = out.view(1, 1, -1)
            if i == last:
                n, h = add_rmsnorm(out, h, self.final_norm, eps)
            else:
                nf = fast[i + 1]
                n, h = add_rmsnorm(out, h, self.layers[i + 1].input_norm, eps, out_index=nf[0].index if nf else None)
        return self._head_from_normed(n)

    def _rope_table(self, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """fp16 cos/sin ``[max_len, head_dim]``, the rows ``_rope`` would compute per position."""
        tab = self._rope_tables.get(max_len)
        if tab is None:
            tab = self._rope(torch.arange(max_len, device=self.device))
            self._rope_tables[max_len] = tab
        return tab

    def _head_from_normed(self, h: torch.Tensor) -> torch.Tensor:
        if self.qlm_head is not None:
            logits = self.qlm_head.logits(h)
        elif isinstance(self.lm_head, QUFLinear):
            logits = self.lm_head(h)
        else:
            logits = h.to(self.dtype) @ self._lm_head_weight.t()
        return logits.to(torch.float32)

    @torch.no_grad()
    def decode_token(self, token_id: int, cache: StaticKVCache) -> torch.Tensor:
        """Feed one token at ``cache.length`` through ``decode_step`` and advance the cache."""
        if cache.length + 1 > cache.max_len:
            raise CacheLengthError(cache.length + 1, cache.max_len)
        cache.token.fill_(int(token_id))
        cache.position.fill_(cache.length)
        logits = self.decode_step(cache)
        cache.length += 1
        return logits
