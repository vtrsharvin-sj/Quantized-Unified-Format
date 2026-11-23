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
        # Greedy argmax inside the graph, only with the Fused_Decode_Kernels (task 33.2).
        self.device_argmax = bool(getattr(model, "fused_decode", False))
        self._next = torch.zeros((1,), dtype=torch.long, device=cache.token.device)
        self._pinned: torch.Tensor | None = None
        self._event: torch.cuda.Event | None = None
        self._count = 0
        self._checked = 0

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
        # The warm-up may have switched the model back to the 0.3.0 path (a fused kernel failed to
        # compile); the graph is then captured whole from that path, with no in-graph argmax.
        self.device_argmax = bool(getattr(self.model, "fused_decode", False))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self.model.decode_step(cache)
            if self.device_argmax:
                # Greedy feed stays on the device: the next token is the argmax of these logits
                # and the position advances, both written into the static inputs of the next replay.
                torch.argmax(logits.view(-1), dim=-1, out=self._next.view(()))
                cache.token.copy_(self._next.view(1, 1))
                cache.position.add_(1)
        torch.cuda.synchronize()
        self.graph, self.logits = graph, logits

    # -- greedy decode with the token kept on the device (Requirement 37.4) -------------------

    @property
    def supports_greedy_device(self) -> bool:
        return self.device_argmax and self.graph is not None

    def begin_greedy(self, first_token: int) -> None:
        """Seed the static inputs; later steps feed themselves from the graph's argmax."""
        cache = self.cache
        cache.token.fill_(int(first_token))
        cache.position.fill_(cache.length)
        self._pinned = torch.zeros((cache.max_len,), dtype=torch.long).pin_memory()
        self._count = 0
        self._checked = 0

    def step_greedy(self) -> None:
        """Launch one replay. No host sync: the sampled token is copied asynchronously into a
        pinned host buffer and read later by ``drain_greedy``."""
        cache = self.cache
        if cache.length + 1 > cache.max_len:
            raise CacheLengthError(cache.length + 1, cache.max_len)
        self.graph.replay()
        self._pinned[self._count : self._count + 1].copy_(cache.token.view(1), non_blocking=True)
        cache.length += 1
        self._count += 1

    def drain_greedy(self) -> list[int]:
        """Wait for the copies issued so far and return the tokens not returned yet."""
        if self._event is None:
            self._event = torch.cuda.Event()
        self._event.record()
        self._event.synchronize()
        out = self._pinned[self._checked : self._count].tolist()
        self._checked = self._count
        return out

    def decode_token(self, token_id: int, cache: StaticKVCache) -> torch.Tensor:
        """Same contract as ``QUFModel.decode_token`` but replays the graph."""
        if cache.length + 1 > cache.max_len:
            raise CacheLengthError(cache.length + 1, cache.max_len)
        cache.token.fill_(int(token_id))
        cache.position.fill_(cache.length)
        self.graph.replay()
        cache.length += 1
        return self.logits


def _reset_after_failed_capture() -> None:
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


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
        try:
            dec.capture()
        except Exception as exc:  # noqa: BLE001
            if not getattr(model, "fused_decode", False):
                raise
            # A fused kernel could not be captured: the whole model goes back to the 0.3.0 path
            # and the capture is tried once more from there (never a partially fused graph).
            _reset_after_failed_capture()
            model._disable_fused(f"capture failed: {type(exc).__name__}: {exc}")
            dec = CUDAGraphDecoder(model, cache)
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
