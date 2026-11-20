"""lm-evaluation-harness adapter for QUF files.

``lm_eval`` is imported lazily, so importing this module never requires it.
The adapter class ``QUFLM`` subclasses ``lm_eval.api.model.LM`` and is built on
first attribute access (``from quf.bench.lm_eval_adapter import QUFLM``) or via
:func:`get_adapter_class`. It is registered with the harness as ``"quf"``::

    lm_eval --model quf --model_args path=model.quf,device=cpu --tasks hellaswag
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

_LM_CLASS: type | None = None


def _import_lm_base():
    try:
        from lm_eval.api.model import LM
    except ImportError as exc:
        raise ImportError(
            "lm-evaluation-harness is required for the QUF adapter. "
            "Install it with: pip install lm_eval"
        ) from exc
    return LM


def _eos_id(model) -> int | None:
    eos = (model.header.tokenizer or {}).get("eos_token_id")
    return int(eos) if eos is not None else None


def _stop_at(text: str, until: list[str]) -> str:
    cut = len(text)
    for s in until:
        if s:
            idx = text.find(s)
            if idx != -1:
                cut = min(cut, idx)
    return text[:cut]


def score_batched(
    forward,
    items: list[tuple[list[int], list[int]]],
    max_length: int,
    eos_id: int | None = None,
    batch_size: int = 16,
    max_batch_tokens: int = 2048,
    pad_id: int = 0,
) -> list[tuple[float, bool]]:
    """Score ``(context_ids, continuation_ids)`` pairs with batched forward passes.

    ``forward(ids[B, T]) -> logits[B, T, V]``. Sequences are sorted by length and
    right-padded. The model is causal, so padding after a sequence never changes the
    logits at its real positions and no attention mask is needed. A request whose
    input is a prefix of another request's input (identical requests, or choices
    sharing a context) reuses that row, so repeated contexts run once.
    Returns ``(sum log-prob, is_greedy)`` per item, in input order.
    """
    results: list[tuple[float, bool] | None] = [None] * len(items)
    seqs: list[tuple[int, ...]] = []  # input tokens per item
    nconts: list[int] = [0] * len(items)
    targets: list[list[int]] = [[] for _ in items]
    for i, (ctx, cont) in enumerate(items):
        if not cont:
            results[i] = (0.0, True)
            seqs.append(())
            continue
        if not ctx:
            ctx = [eos_id if eos_id is not None else 0]
        full = (list(ctx) + list(cont))[-max_length:]
        n = min(len(cont), len(full) - 1)  # a continuation longer than the window is truncated
        if n < 1:
            results[i] = (0.0, True)
            seqs.append(())
            continue
        nconts[i] = n
        targets[i] = full[-n:]
        seqs.append(tuple(full[:-1]))

    # Pick host rows: longest first; shorter inputs that are prefixes share a host.
    order = sorted((i for i in range(len(items)) if results[i] is None), key=lambda i: -len(seqs[i]))
    hosts: list[tuple[int, ...]] = []
    by_first: dict[int, list[int]] = {}
    host_of: dict[int, int] = {}
    for i in order:
        s = seqs[i]
        found = None
        for h in by_first.get(s[0], ()):
            if hosts[h][: len(s)] == s:
                found = h
                break
        if found is None:
            found = len(hosts)
            hosts.append(s)
            by_first.setdefault(s[0], []).append(found)
        host_of[i] = found
    users: dict[int, list[int]] = {}
    for i, h in host_of.items():
        users.setdefault(h, []).append(i)

    # Batch hosts (already sorted by descending length) under a size and token budget.
    start = 0
    while start < len(hosts):
        end = start + 1
        longest = len(hosts[start])
        while end < len(hosts) and end - start < batch_size and longest * (end - start + 1) <= max_batch_tokens:
            end += 1
        batch = hosts[start:end]
        ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        for b, s in enumerate(batch):
            ids[b, : len(s)] = torch.tensor(s, dtype=torch.long)
        logits = forward(ids)
        for b in range(len(batch)):
            for i in users[start + b]:
                n, length = nconts[i], len(seqs[i])
                rows = F.log_softmax(logits[b, length - n:length].to(torch.float32), dim=-1)
                cont = torch.tensor(targets[i], dtype=torch.long, device=rows.device)
                tok_lp = rows.gather(1, cont.unsqueeze(1)).squeeze(1)
                greedy = bool(torch.equal(rows.argmax(dim=-1), cont))
                results[i] = (float(tok_lp.sum().item()), greedy)
        del logits
        start = end
    return results  # type: ignore[return-value]


def get_adapter_class() -> type:
    """Return the ``QUFLM`` class, importing ``lm_eval`` on first use."""
    global _LM_CLASS
    if _LM_CLASS is not None:
        return _LM_CLASS
    LM = _import_lm_base()

    class QUFLM(LM):
        """Evaluate a QUF_File with lm-evaluation-harness."""

        def __init__(
            self,
            path: str | None = None,
            model: Any = None,
            tokenizer: Any = None,
            device: str = "cpu",
            backend: str | None = None,
            tokenizer_path: str | None = None,
            max_gen_toks: int = 256,
            batch_size: int = 16,
            max_batch_tokens: int = 2048,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            from quf.runtime.model import QUFModel
            from quf.tokenization.wrapper import TokenizerWrapper

            if model is None:
                if path is None:
                    raise ValueError("Provide 'path' to a .quf file or a loaded 'model'")
                model = QUFModel.load(path, device_option=device, backend=backend)
            if tokenizer is None:
                tok_src = tokenizer_path or (model.header.tokenizer or {}).get("path")
                if tok_src is None:
                    raise ValueError("Provide 'tokenizer' or 'tokenizer_path'")
                tokenizer = TokenizerWrapper(tok_src)
            self.model = model
            self.tok = tokenizer
            self._max_gen_toks = int(max_gen_toks)
            self._batch_size = max(1, int(batch_size))
            self._max_batch_tokens = max(1, int(max_batch_tokens))

        # -- properties used by the harness ----------------------------------

        @property
        def eot_token_id(self) -> int | None:
            return _eos_id(self.model)

        @property
        def max_length(self) -> int:
            return int(self.model.max_position_embeddings)

        @property
        def max_gen_toks(self) -> int:
            return self._max_gen_toks

        # -- helpers ---------------------------------------------------------

        def _logprobs(self, ids: list[int]) -> torch.Tensor:
            """Log-probabilities ``[len(ids), vocab]``; row i predicts ids[i + 1]."""
            logits = self.model.forward(torch.tensor([ids], dtype=torch.long))[0]
            return F.log_softmax(logits.to(torch.float32), dim=-1)

        def _score(self, context_ids: list[int], cont_ids: list[int]) -> tuple[float, bool]:
            if not cont_ids:
                return 0.0, True
            if not context_ids:
                eos = self.eot_token_id
                context_ids = [eos if eos is not None else 0]
            full = (context_ids + cont_ids)[-self.max_length:]
            n = len(cont_ids)
            lp = self._logprobs(full[:-1])
            cont = torch.tensor(full[-n:], dtype=torch.long)
            rows = lp[-n:]
            tok_lp = rows.gather(1, cont.unsqueeze(1)).squeeze(1)
            greedy = bool(torch.equal(rows.argmax(dim=-1), cont))
            return float(tok_lp.sum().item()), greedy

        def _split(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
            # Encode context and continuation separately so the boundary is exact,
            # moving trailing whitespace of the context onto the continuation.
            n_spaces = len(context) - len(context.rstrip())
            if n_spaces > 0:
                continuation = context[-n_spaces:] + continuation
                context = context[:-n_spaces]
            return self.tok.encode(context), self.tok.encode(continuation)

        # -- LM interface ----------------------------------------------------

        def loglikelihood(self, requests) -> list[tuple[float, bool]]:
            items = []
            for req in requests:
                context, continuation = req.args[:2]
                items.append(self._split(context, continuation))
            return score_batched(
                self.model.forward,
                items,
                self.max_length,
                eos_id=self.eot_token_id,
                batch_size=self._batch_size,
                max_batch_tokens=self._max_batch_tokens,
            )

        def loglikelihood_rolling(self, requests) -> list[float]:
            out: list[float] = []
            window = self.max_length
            eos = self.eot_token_id
            prefix = eos if eos is not None else 0
            for req in requests:
                ids = self.tok.encode(req.args[0])
                total = 0.0
                pos = 0
                prev: list[int] = [prefix]
                while pos < len(ids):
                    chunk = ids[pos:pos + window - len(prev)]
                    lp, _ = self._score(prev, chunk)
                    total += lp
                    pos += len(chunk)
                    prev = []  # later windows condition on the start of the chunk only
                    if pos < len(ids):
                        prev = [ids[pos - 1]]
                out.append(total)
            return out

        def generate_until(self, requests) -> list[str]:
            from quf.runtime.generate import generate

            out: list[str] = []
            for req in requests:
                context, gen_kwargs = req.args[0], dict(req.args[1] or {})
                until = gen_kwargs.get("until") or []
                if isinstance(until, str):
                    until = [until]
                max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
                max_new = max(1, min(max_new, self.max_length - 1))
                temperature = float(gen_kwargs.get("temperature", 0.0) or 0.0)
                ctx_ids = self.tok.encode(context)[-(self.max_length - max_new):]
                ids = generate(
                    self.model,
                    self.tok,
                    ctx_ids,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    top_k=gen_kwargs.get("top_k"),
                    top_p=gen_kwargs.get("top_p"),
                )
                eos = self.eot_token_id
                if eos is not None and ids and ids[-1] == eos:
                    ids = ids[:-1]
                out.append(_stop_at(self.tok.decode(ids), list(until)))
            return out

    try:
        from lm_eval.api.registry import register_model

        QUFLM = register_model("quf")(QUFLM)
    except Exception:  # registration is best effort (e.g. already registered)
        pass

    _LM_CLASS = QUFLM
    return QUFLM


def __getattr__(name: str):
    if name == "QUFLM":
        return get_adapter_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
