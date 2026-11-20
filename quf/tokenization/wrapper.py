"""Tokenizer_Wrapper: gigatoken for encoding, Hugging Face tokenizers as fallback.

Gigatoken is used for encoding when it can be imported and wraps the tokenizer
successfully. Decoding always goes through the Hugging Face ``tokenizers``
object, which gigatoken wraps in its compatibility mode. When gigatoken is not
usable, the reason is logged and Hugging Face ``tokenizers`` encodes as well.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("quf.tokenization")


def _load_hf_tokenizer(model_id_or_path: str) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Hugging Face 'tokenizers' is required for tokenization. "
            "Install it with: pip install tokenizers (or quf[convert])."
        ) from exc

    path = os.fspath(model_id_or_path)
    if os.path.isdir(path):
        return Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    if os.path.isfile(path):
        return Tokenizer.from_file(path)
    return Tokenizer.from_pretrained(path)


class TokenizerWrapper:
    """Encode and decode text, preferring gigatoken with an HF fallback.

    Args:
        model_id_or_path: Hugging Face model id, a directory containing
            ``tokenizer.json``, or a ``tokenizer.json`` file path. Ignored when
            ``hf_tokenizer`` is given.
        hf_tokenizer: Optional prebuilt ``tokenizers.Tokenizer`` instance.
    """

    def __init__(self, model_id_or_path: str | None = None, *, hf_tokenizer: Any = None):
        if hf_tokenizer is None:
            if model_id_or_path is None:
                raise ValueError("Provide model_id_or_path or hf_tokenizer")
            hf_tokenizer = _load_hf_tokenizer(model_id_or_path)
        self._hf = hf_tokenizer
        self._gt: Any = None
        self.backend: str = "huggingface"
        self.fallback_reason: str | None = None

        try:
            import gigatoken as gt
        except Exception as exc:  # ImportError or a platform load failure
            self.fallback_reason = f"gigatoken is not importable: {exc!r}"
            logger.info("gigatoken unavailable, using Hugging Face tokenizers")
            logger.debug("Falling back to Hugging Face tokenizers: %s", self.fallback_reason)
            return

        try:
            self._gt = gt.Tokenizer(self._hf)
            self.backend = "gigatoken"
        except Exception as exc:
            self._gt = None
            self.fallback_reason = f"gigatoken could not wrap this tokenizer: {exc!r}"
            logger.info("gigatoken cannot wrap this tokenizer, using Hugging Face tokenizers")
            logger.debug("Falling back to Hugging Face tokenizers: %s", self.fallback_reason)

    def encode(self, text: str) -> list[int]:
        """Return token ids for ``text`` without adding special tokens."""
        if self._gt is not None:
            # gigatoken 0.11: Tokenizer.encode(str) -> numpy uint32 array
            return [int(i) for i in self._gt.encode(text)]
        return list(self._hf.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        """Return the text for ``ids`` (special tokens are kept)."""
        return self._hf.decode([int(i) for i in ids], skip_special_tokens=False)
