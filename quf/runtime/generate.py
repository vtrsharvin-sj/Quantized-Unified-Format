"""Sampling and the autoregressive generation loop."""

from __future__ import annotations

from typing import Callable

import torch

from quf.runtime.model import QUFModel


GREEDY_CHECK_EVERY = 8  # host EOS check and streaming interval of the on-device greedy loop


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
    static_cache: bool = True,
    cuda_graph: bool | None = None,
) -> list[int]:
    """Generate up to ``max_new_tokens`` token ids after ``prompt``.

    ``prompt`` is text (encoded with ``tok``) or a list of token ids. Generation
    stops after the end-of-sequence token from the Header is produced; that
    token is included in the returned ids but is not streamed. When ``stream``
    is given, each newly decoded text fragment is passed to it as soon as the
    token is generated.

    ``static_cache`` (default True) uses the preallocated Static_KV_Cache and the
    fixed-shape Decode_Step; False uses the growing KVCache path.

    ``cuda_graph`` (default None = on for CUDA models) replays a captured CUDA graph of the
    Decode_Step after prefill. False, or the ``QUF_NO_CUDA_GRAPH`` environment variable, turns
    it off. If capture fails, one INFO line gives the reason and decoding continues eagerly.

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

    use_static = static_cache and hasattr(model, "decode_token")
    if use_static:
        cache = model.new_static_cache(len(prompt_ids) + max_new_tokens)
    else:
        cache = model.new_cache()
    out: list[int] = []
    emitted = ""
    inputs = torch.tensor([prompt_ids], dtype=torch.long)
    decoder = None
    def emit(next_id: int) -> bool:
        """Record one token and stream it. Returns True when generation must stop (EOS)."""
        nonlocal emitted
        out.append(next_id)
        if eos is not None and next_id == int(eos):
            return True
        if stream is not None:
            text = tok.decode(out)
            # Hold back an incomplete multi-byte character until it completes.
            if not text.endswith("\ufffd") and len(text) > len(emitted):
                stream(text[len(emitted):])
                emitted = text
        return False

    for step in range(max_new_tokens):
        if use_static and step == 1 and max_new_tokens > 1:
            from quf.runtime.cuda_graph import make_decoder  # noqa: PLC0415

            decoder = make_decoder(model, cache, cuda_graph)
            if (
                decoder is not None
                and temperature <= 0
                and getattr(decoder, "supports_greedy_device", False)
            ):
                # Greedy: argmax and the next-token feed run inside the graph. Tokens are copied
                # to pinned host memory and checked for EOS every GREEDY_CHECK_EVERY tokens, so
                # extra tokens after EOS may be computed but are never returned or streamed, and
                # streaming lags by at most GREEDY_CHECK_EVERY tokens.
                remaining = max_new_tokens - len(out)
                decoder.begin_greedy(out[-1])
                for issued in range(1, remaining + 1):
                    decoder.step_greedy()
                    if issued % GREEDY_CHECK_EVERY == 0 or issued == remaining:
                        for t in decoder.drain_greedy():
                            if emit(t):
                                return out
                return out
        if use_static and step > 0:
            stepper = decoder if decoder is not None else model
            logits = stepper.decode_token(out[-1], cache)[0, -1]
        else:
            logits = model.forward(inputs, cache)[0, -1]
        next_id = sample_next(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
        )
        if emit(next_id):
            break
        inputs = torch.tensor([[next_id]], dtype=torch.long)
    return out
