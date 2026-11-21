"""Benchmark_Suite: WikiText perplexity, file size, generation speed, results JSON.

The documented target is a relative perplexity increase of at most 3 percent
versus fp16. It is reported next to the measured value and never asserted.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from quf.bench.timing import DEFAULT_REPEATS, DEFAULT_WARMUP, TimingStats, time_repeats

logger = logging.getLogger("quf.bench")

DEFAULT_DATASET = "wikitext-2-raw-v1"
PPL_TARGET_PERCENT = 3.0


class DatasetLoadError(RuntimeError):
    """The evaluation dataset could not be loaded."""

    def __init__(self, dataset: str, reason: str = ""):
        self.dataset = dataset
        msg = f"could not load dataset '{dataset}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


@dataclass
class BenchResult:
    backend: str
    device_name: str
    software_versions: dict[str, str]
    perplexity: float
    file_size_bytes: int
    tokens_per_second: float
    dataset: str = DEFAULT_DATASET
    ppl_target_percent: float = PPL_TARGET_PERCENT
    fp16_perplexity: Optional[float] = None
    relative_ppl_increase: Optional[float] = None  # fraction, 0.02 == 2 percent
    gguf_size_bytes: Optional[int] = None
    gguf_tokens_per_second: Optional[float] = None
    tokens_per_second_min: Optional[float] = None
    tokens_per_second_max: Optional[float] = None
    tokens_per_second_repeats: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchResult":
        return cls(**d)

    @classmethod
    def from_json(cls, text: str) -> "BenchResult":
        return cls.from_dict(json.loads(text))


# -- dataset ---------------------------------------------------------------


def load_wikitext(dataset: str = DEFAULT_DATASET, split: str = "test") -> str:
    """Return the WikiText ``split`` joined into one string.

    Raises DatasetLoadError naming the dataset on any failure.
    """
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("Salesforce/wikitext", dataset, split=split)
        return "\n\n".join(ds["text"])
    except Exception as exc:  # noqa: BLE001 - missing package, network, or cache
        raise DatasetLoadError(dataset, f"{type(exc).__name__}: {exc}") from exc


# -- perplexity ------------------------------------------------------------


def perplexity(
    model,
    tok,
    dataset: str = DEFAULT_DATASET,
    ctx_len: int = 2048,
    *,
    text: str | None = None,
    max_windows: int | None = None,
) -> float:
    """Perplexity of ``model`` over non-overlapping windows of ``ctx_len`` tokens.

    ``model.forward(ids[1, n]) -> logits[1, n, vocab]``. ``text`` overrides
    loading ``dataset`` (used by tests and offline runs).
    """
    import torch
    import torch.nn.functional as F

    if text is None:
        text = load_wikitext(dataset)
    ids = tok.encode(text)
    if ctx_len < 2:
        raise ValueError("ctx_len must be at least 2")
    n_windows = len(ids) // ctx_len
    if n_windows == 0 and len(ids) >= 2:
        ctx_len, n_windows = len(ids), 1
    if n_windows == 0:
        raise ValueError("evaluation text is too short to compute perplexity")
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    total_nll = 0.0
    total_tokens = 0
    for w in range(n_windows):
        window = torch.tensor([ids[w * ctx_len : (w + 1) * ctx_len]], dtype=torch.long)
        logits = model.forward(window)
        logits = logits[0, :-1].to(torch.float32)
        targets = window[0, 1:].to(logits.device)
        nll = F.cross_entropy(logits, targets, reduction="sum")
        total_nll += float(nll.item())
        total_tokens += int(targets.numel())
    return math.exp(total_nll / total_tokens)


# -- speed -----------------------------------------------------------------


def _tps_once(model, tok, prompt: str, max_new_tokens: int) -> float:
    from quf.runtime.generate import generate

    prompt_ids = tok.encode(prompt) or [0]
    budget = max(1, min(max_new_tokens, model.max_position_embeddings - len(prompt_ids)))
    start = time.perf_counter()
    out = generate(model, tok, prompt_ids, max_new_tokens=budget, temperature=0.0)
    elapsed = time.perf_counter() - start
    return len(out) / elapsed if elapsed > 0 and out else 0.0


def measure_tps_stats(
    model,
    tok,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 32,
    *,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> TimingStats:
    """Greedy generation speed: warm-up runs, then ``repeats`` timed runs (median, min, max)."""
    return time_repeats(
        lambda: _tps_once(model, tok, prompt, max_new_tokens), warmup=warmup, repeats=repeats
    )


def measure_tokens_per_second(
    model, tok, prompt: str = "The quick brown fox", max_new_tokens: int = 32
) -> float:
    """Median greedy generation speed in new tokens per second (see ``measure_tps_stats``)."""
    return measure_tps_stats(model, tok, prompt, max_new_tokens).median


class _HFCausalLM:
    """Adapts a transformers causal LM to the ``forward(ids)`` used here."""

    def __init__(self, path: str, device: str):
        import torch
        from transformers import AutoModelForCausalLM  # type: ignore

        self._device = torch.device(device)
        self._m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16 if device != "cpu" else torch.float32
        ).to(self._device)
        self._m.eval()

    def forward(self, ids):
        import torch

        with torch.no_grad():
            return self._m(ids.to(self._device)).logits


def _gguf_speed(path: str, tok_text: str = "The quick brown fox", n: int = 32) -> Optional[float]:
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.warning("llama-cpp-python unavailable, GGUF speed not measured: %r", exc)
        return None
    llm = Llama(model_path=path, verbose=False)
    start = time.perf_counter()
    out = llm(tok_text, max_tokens=n, temperature=0.0)
    elapsed = time.perf_counter() - start
    produced = int(out["usage"]["completion_tokens"])
    return produced / elapsed if elapsed > 0 and produced else 0.0


# -- environment -----------------------------------------------------------


def software_versions() -> dict[str, str]:
    from importlib import metadata

    import quf

    versions = {"python": platform.python_version(), "quf": getattr(quf, "__version__", "unknown")}
    for pkg in ("torch", "gptqmodel", "gigatoken", "tokenizers", "transformers", "datasets"):
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = "not installed"
    try:
        import torch

        versions["cuda"] = str(torch.version.cuda)
    except Exception:  # noqa: BLE001
        versions["cuda"] = "not installed"
    return versions


# -- driver ----------------------------------------------------------------


def _tokenizer_from_header(header):
    from tokenizers import Tokenizer  # type: ignore

    from quf.tokenization.wrapper import TokenizerWrapper

    raw = (header.tokenizer or {}).get("files", {}).get("tokenizer.json")
    if raw is None:
        raise ValueError("QUF header has no embedded tokenizer.json")
    return TokenizerWrapper(hf_tokenizer=Tokenizer.from_str(raw))


def run_bench(
    quf_path,
    *,
    fp16_ref: str | None = None,
    gguf: str | None = None,
    out_json,
    device: str = "auto",
    backend_option: str = "auto",
    prefer_fused: bool = False,
    dataset: str = DEFAULT_DATASET,
    ctx_len: int = 2048,
    max_windows: int | None = None,
    max_new_tokens: int = 32,
    text: str | None = None,
    model_loader: Callable[..., Any] | None = None,
    tok=None,
) -> BenchResult:
    """Run the benchmarks and write the results JSON to ``out_json``.

    Raises DatasetLoadError (naming the dataset) before any model work when the
    dataset cannot be loaded.
    """
    from quf.device import detect_devices
    from quf.runtime.selector import select_backend

    if text is None:
        text = load_wikitext(dataset)  # fail early, naming the dataset

    info = detect_devices()

    cached_bytes = free_bytes = None
    if info.cuda_available:
        from quf.runtime.model import _available_gpu_bytes, estimate_cached_bytes

        cached_bytes, free_bytes = estimate_cached_bytes(quf_path), _available_gpu_bytes() or None
    from quf.runtime.backends.torch_fused import fused_self_test

    choice = select_backend(
        device,
        info,
        backend_option=backend_option,
        cached_bytes=cached_bytes,
        free_bytes=free_bytes,
        prefer_fused=prefer_fused,
        fused_self_test=fused_self_test,
    )
    if model_loader is None:
        from quf.runtime.model import QUFModel

        model_loader = lambda p, backend: QUFModel.load(  # noqa: E731
            p, device_option="cpu" if backend == "cpu" else "cuda", backend=backend
        )
    model = model_loader(os.fspath(quf_path), choice.backend)
    if tok is None:
        tok = _tokenizer_from_header(model.header)

    ppl = perplexity(model, tok, dataset, ctx_len, text=text, max_windows=max_windows)
    stats = measure_tps_stats(model, tok, max_new_tokens=max_new_tokens)
    tps = stats.median

    result = BenchResult(
        backend=choice.backend,
        device_name=(info.name if choice.backend != "cpu" and info.name else "cpu"),
        software_versions=software_versions(),
        perplexity=ppl,
        file_size_bytes=os.path.getsize(quf_path),
        tokens_per_second=tps,
        dataset=dataset,
        tokens_per_second_min=stats.min,
        tokens_per_second_max=stats.max,
        tokens_per_second_repeats=stats.repeats,
    )

    if fp16_ref is not None:
        ref_device = "cpu" if choice.backend == "cpu" else "cuda"
        ref = _HFCausalLM(fp16_ref, ref_device)
        result.fp16_perplexity = perplexity(
            ref, tok, dataset, ctx_len, text=text, max_windows=max_windows
        )
        result.relative_ppl_increase = ppl / result.fp16_perplexity - 1.0
        logger.debug(
            "relative perplexity increase %.2f%% (documented target: at most %.0f%%)",
            100 * result.relative_ppl_increase,
            PPL_TARGET_PERCENT,
        )

    if gguf is not None:
        result.gguf_size_bytes = os.path.getsize(gguf)
        result.gguf_tokens_per_second = _gguf_speed(gguf, n=max_new_tokens)

    parent = os.path.dirname(os.path.abspath(os.fspath(out_json)))
    os.makedirs(parent, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(result.to_json())
    return result
