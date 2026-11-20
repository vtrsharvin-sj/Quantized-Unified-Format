"""Sampling and the autoregressive generation loop."""

from __future__ import annotations

from typing import Callable

import torch

from quf.runtime.model import QUFModel


class ContextLengthError(ValueError):
    """Prompt length plus maximum new tokens exceeds the model context length."""

    def __init__(self, prompt_len: int, max_new: int, context_len: int):
        self.prompt_len = prompt_len
        self.max_new = max_new
        self.context_len = context_len
        super().__init__(
            f"prompt length ({prompt_len}) plus max new tokens ({max_new}) = "
            f"{prompt_len + max_new} exceeds the model context length ({context_len})"
        )


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None,
) -> int:
    """Pick the next token id from 1-D ``logits``.

    Temperature 0 is greedy (argmax). Otherwise logits are scaled by the
    temperature, filtered by top-k then top-p, and sampled with ``generator``.
    """
    logits = logits.to(torch.float32)
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k is not None and 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # Drop tokens whose preceding cumulative mass already reaches top_p;
        # the highest-probability token is always kept.
        drop = (cum - probs) >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


def generate(
    model: QUFModel,
    tok,
    prompt: str | list[int],
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = 0,
    stream: Callable[[str], None] | None = None,
) -> list[int]:
    """Generate up to ``max_new_tokens`` token ids after ``prompt``.

    ``prompt`` is text (encoded with ``tok``) or a list of token ids. Generation
    stops after the end-of-sequence token from the Header is produced; that
    token is included in the returned ids but is not streamed. When ``stream``
    is given, each newly decoded text fragment is passed to it as soon as the
    token is generated.

    Raises:
        ContextLengthError: if prompt length + ``max_new_tokens`` exceeds the
            model context length.
    """
    prompt_ids = list(prompt) if not isinstance(prompt, str) else tok.encode(prompt)
    context_len = int(model.max_position_embeddings)
    if len(prompt_ids) + max_new_tokens > context_len:
        raise ContextLengthError(len(prompt_ids), max_new_tokens, context_len)
    if max_new_tokens <= 0 or not prompt_ids:
        return []

    eos = (model.header.tokenizer or {}).get("eos_token_id")
    generator = torch.Generator(device=model.device)
    generator.manual_seed(int(seed))

    cache = model.new_cache()
    out: list[int] = []
    emitted = ""
    inputs = torch.tensor([prompt_ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model.forward(inputs, cache)[0, -1]
        next_id = sample_next(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
        )
        out.append(next_id)
        if eos is not None and next_id == int(eos):
            break
        if stream is not None:
            text = tok.decode(out)
            # Hold back an incomplete multi-byte character until it completes.
            if not text.endswith("\ufffd") and len(text) > len(emitted):
                stream(text[len(emitted):])
                emitted = text
        inputs = torch.tensor([[next_id]], dtype=torch.long)
    return out
