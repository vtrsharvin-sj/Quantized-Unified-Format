"""CUDA_Graph_Decoder: replays a captured fixed-shape Decode_Step, one launch per token."""

from __future__ import annotations

import logging
import os

import torch

from quf.runtime.model import CacheLengthError, QUFModel, StaticKVCache

logger = logging.getLogger("quf.runtime")

ENV_DISABLE = "QUF_NO_CUDA_GRAPH"


def cuda_graph_disabled_by_env() -> bool:
    return os.environ.get(ENV_DISABLE, "").strip().lower() not in ("", "0", "false", "no", "off")


class CUDAGraphDecoder:
    """Captures ``model.decode_step(cache)`` once and replays it per token.

    Inputs are ``cache.token`` and ``cache.position`` (updated in place before each replay);
    the output is the static logits tensor ``[1, 1, vocab]``, overwritten by the next replay.
    """

    def __init__(self, model: QUFModel, cache: StaticKVCache):
        self.model = model
        self.cache = cache
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None

    @staticmethod
    def available(model: QUFModel) -> str | None:
        """Return None if capture can be attempted, else the reason it cannot."""
        if getattr(model.device, "type", "") != "cuda":
            return "the model is not on a CUDA device"
        if not torch.cuda.is_available():
            return "CUDA is not available"
        return None

    def capture(self) -> None:
        """Warm up on a side stream, then capture. Positions written during warm-up and capture
        are the next position to be decoded, so the real step overwrites them. Raises on failure."""
        cache = self.cache
        cache.token.zero_()
        cache.position.fill_(min(cache.length, cache.max_len - 1))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.model.decode_step(cache)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self.model.decode_step(cache)
        torch.cuda.synchronize()
        self.graph, self.logits = graph, logits

    def decode_token(self, token_id: int, cache: StaticKVCache) -> torch.Tensor:
        """Same contract as ``QUFModel.decode_token`` but replays the graph."""
        if cache.length + 1 > cache.max_len:
            raise CacheLengthError(cache.length + 1, cache.max_len)
        cache.token.fill_(int(token_id))
        cache.position.fill_(cache.length)
        self.graph.replay()
        cache.length += 1
        return self.logits


def make_decoder(model: QUFModel, cache: StaticKVCache, enabled: bool | None = None):
    """Return a CUDAGraphDecoder when enabled and capture succeeds, else None (eager).

    ``enabled`` None means the default: on for CUDA models unless ``QUF_NO_CUDA_GRAPH`` is set.
    Any capture failure is logged once at INFO and eager decoding continues.
    """
    if enabled is None:
        enabled = not cuda_graph_disabled_by_env()
    if not enabled:
        return None
    reason = CUDAGraphDecoder.available(model)
    if reason is not None:
        return None  # CPU or no GPU: eager is the normal path, nothing to report
    dec = CUDAGraphDecoder(model, cache)
    try:
        dec.capture()
    except Exception as exc:  # noqa: BLE001
        hint = ""
        if getattr(model, "backend_name", None) == "triton_int4":
            hint = "; eager int4 is slower than torch_gpu, consider --backend cached"
        logger.info("CUDA graph capture failed, using eager decode: %s: %s%s", type(exc).__name__, exc, hint)
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        return None
    return dec
