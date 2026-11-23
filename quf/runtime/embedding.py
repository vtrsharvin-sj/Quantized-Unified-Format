"""Runtime side of a Quantized_Embedding (task 34.3): row gather and the tied / untied lm_head.

The table stays packed on the device (int4 bytes or int8, plus fp16 scales). No full fp16 copy
is ever built:

* ``lookup(ids)`` dequantizes only the requested rows with the reference function
  :func:`quf.quant.embedding.dequantize_embedding_rows` (torch ops, so it also runs inside a
  CUDA graph; the id tensor stays on the device). Every Backend uses it.
* ``logits(h)`` is ``h @ W.T`` in fp32 ``[M, vocab]``. With ``use_kernel`` (the triton_int4
  Backend, Triton_Gate passed) M <= 16 runs the Triton GEMV that reads the stored layout
  directly. Everything else (cpu, torch_gpu, torch_fused, and M > 16 on triton_int4) dequantizes
  the vocabulary in chunks into a bounded buffer and multiplies chunk by chunk. That is the
  deliberate policy for the non-kernel Backends: a cached fp16 copy of a 151936 x 896 table is
  272 MB on the GPU, which is the memory the quantized table exists to save, so it is never kept.
  The cost is a dequantization per token, paid only on the Backends that do not have the GEMV.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from quf.quant.embedding import dequantize_embedding_rows

logger = logging.getLogger("quf.runtime")

CHUNK_ELEMENTS = 1 << 22  # elements of the transient dequantized buffer (8 MB in fp16)


class RuntimeEmbedding:
    def __init__(self, packed, scales, bits: int, group_size: int, device: torch.device, use_kernel: bool = False):
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.device = device
        self.packed = torch.from_numpy(np.array(packed)).to(device)
        self.scales = torch.from_numpy(np.array(scales)).to(device)
        self.rows = int(self.packed.shape[0])
        self.cols = int(self.packed.shape[1]) * (2 if self.bits == 4 else 1)
        self.use_kernel = bool(use_kernel) and device.type == "cuda"

    @property
    def nbytes(self) -> int:
        return int(self.packed.numel() + self.scales.numel() * 2)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """fp16 ``[*ids.shape, cols]``; only the rows in ``ids`` are dequantized."""
        flat = ids.reshape(-1).to(self.device)
        rows = dequantize_embedding_rows(self.packed, self.scales, self.bits, self.group_size, flat)
        return rows.reshape(*ids.shape, self.cols)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """fp32 ``[..., rows]`` = ``h @ W.T`` for ``h`` of shape ``[..., cols]``."""
        lead = h.shape[:-1]
        x = h.reshape(-1, h.shape[-1])
        m = x.shape[0]
        if self.use_kernel and m <= 16:
            try:
                from quf.runtime.kernels.embed_gemv import embed_gemv  # noqa: PLC0415

                out = embed_gemv(x.to(torch.float16), self.packed, self.scales, self.bits, self.group_size)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:  # noqa: BLE001
                if torch.cuda.is_current_stream_capturing():
                    raise  # the decoder handles a failed capture
                self.use_kernel = False
                logger.info("quantized lm_head GEMV failed (%s: %s), using the chunked dequantize path",
                            type(exc).__name__, exc)
                out = self._chunked(x)
        else:
            out = self._chunked(x)
        return out.reshape(*lead, self.rows)

    def _chunked(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.float32 if x.device.type == "cpu" else torch.float16
        xc = x.to(dtype)
        out = torch.empty((x.shape[0], self.rows), dtype=torch.float32, device=x.device)
        step = max(1, CHUNK_ELEMENTS // self.cols)
        for r0 in range(0, self.rows, step):
            r1 = min(self.rows, r0 + step)
            w = dequantize_embedding_rows(
                self.packed[r0:r1], self.scales[r0:r1], self.bits, self.group_size
            ).to(dtype)
            out[:, r0:r1] = (xc @ w.t()).to(torch.float32)
        return out
