"""Converter: Hugging Face Qwen checkpoint -> calibrated 4-bit QUF_File.

Flow: check the output path and architecture (cheap, before any heavy work),
build calibration data, run the native GPTQ quantizer, then write the QUF
file. A failure after the output file is created removes the partial file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from quf.convert import gptq as _gptq
from quf.convert.precision import PrecisionPlan, SizeReport
from quf.format.schema import QUFHeader
from quf.format.writer import write_quf
from quf.tokenization.calibration import CalibrationSource, build_calibration
from quf.tokenization.wrapper import TokenizerWrapper

logger = logging.getLogger("quf.convert")

# Only architectures that quf/runtime/model.py runs correctly. Add one here only after the
# runtime supports it. Qwen3ForCausalLM, for example, first needs q_norm and k_norm (the
# per-head RMSNorm on queries and keys) in quf/runtime/model.py; without them a converted
# Qwen3 file would load and generate wrong text with no error.
SUPPORTED_ARCHITECTURES: tuple[str, ...] = ("Qwen2ForCausalLM",)

DEFAULT_DATASET = "wikitext-2-raw-v1"
# Calibration defaults (task 16.8): wikitext-2 train only, 128 x 1024 tokens. The two-source
# mix (wikitext-2 + C4, 128 x 2048) is opt-in because it measured worse perplexity
# (13.549 vs 13.478) and took 2.8x longer. Legacy 0.1.0 settings: wikitext-2, 128 x 512.
DEFAULT_N_SAMPLES = 128
DEFAULT_SEQ_LEN = 1024
MULTI_SOURCE_SEQ_LEN = 2048
LEGACY_SEQ_LEN = 512
SECOND_SOURCE_DATASET = "allenai/c4"
SECOND_SOURCE_MAX_DOCS = 4000

_MODEL_CONFIG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "tie_word_embeddings",
    "head_dim",
)
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


class ConvertError(Exception):
    """Base class for converter errors."""


class UnsupportedArchitectureError(ConvertError, ValueError):
    """config.json names an architecture outside SUPPORTED_ARCHITECTURES.

    Also a ValueError, so ``quf convert`` prints the message alone on one ``quf: error:`` line.
    """

    def __init__(self, architecture: str):
        self.architecture = architecture
        super().__init__(
            f"Unsupported model architecture '{architecture}'. Supported: "
            f"{', '.join(SUPPORTED_ARCHITECTURES)}."
        )


class OutputExistsError(ConvertError):
    def __init__(self, path):
        self.path = Path(path)
        super().__init__(
            f"Output path '{path}' already exists. Use the overwrite option to replace it."
        )


@dataclass
class ConvertResult:
    output: Path
    size_bytes: int
    fp16_size_bytes: int
    ratio: float  # size_bytes / fp16_size_bytes
    report: SizeReport | None = None

    @property
    def message(self) -> str:
        text = (
            f"Wrote {self.output}: {self.size_bytes} bytes "
            f"({self.ratio:.4f} of the fp16 checkpoint size, {self.fp16_size_bytes} bytes)"
        )
        if self.report is not None:
            text += "\n" + self.report.format()
        return text


def _fetch_file(source: str, filename: str, *, required: bool) -> Path | None:
    """Return a local path to ``filename`` of a local directory or Hub model."""
    if os.path.isdir(source):
        p = Path(source) / filename
        if p.is_file():
            return p
        if required:
            raise ConvertError(f"'{filename}' not found in {source}")
        return None
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:
        raise _gptq.MissingDependencyError(
            "huggingface_hub is required to fetch a model from the Hub. "
            "Install it with: pip install 'quf[convert]'."
        ) from exc
    try:
        return Path(hf_hub_download(source, filename))
    except Exception as exc:
        if required:
            raise ConvertError(f"could not fetch '{filename}' for '{source}': {exc}") from exc
        return None


def read_source_config(source: str) -> dict[str, Any]:
    path = _fetch_file(source, "config.json", required=True)
    return json.loads(path.read_text(encoding="utf-8"))


def check_architecture(config: dict[str, Any]) -> str:
    """Return the architecture named in config.json, or raise UnsupportedArchitectureError.

    convert_checkpoint calls this right after reading config.json, before the tokenizer,
    the calibration data, and the weights are loaded.
    """
    archs = config.get("architectures") or []
    name = archs[0] if archs else "<unknown>"
    if name not in SUPPORTED_ARCHITECTURES:
        raise UnsupportedArchitectureError(name)
    return name


def _first_int(value: Any) -> int | None:
    if isinstance(value, list):
        value = value[0] if value else None
    return None if value is None else int(value)


def build_model_config(config: dict[str, Any], architecture: str) -> dict[str, Any]:
    mc: dict[str, Any] = {"architecture": architecture}
    for key in _MODEL_CONFIG_FIELDS:
        if key in config and config[key] is not None:
            mc[key] = config[key]
    mc.setdefault("num_key_value_heads", config.get("num_attention_heads"))
    mc.setdefault("rope_theta", 10000.0)
    mc.setdefault("tie_word_embeddings", False)
    return mc


def build_tokenizer_info(source: str, config: dict[str, Any]) -> dict[str, Any]:
    files: dict[str, str] = {}
    for name in _TOKENIZER_FILES:
        p = _fetch_file(source, name, required=(name == "tokenizer.json"))
        if p is not None:
            files[name] = p.read_text(encoding="utf-8")
    eos = _first_int(config.get("eos_token_id"))
    gen_cfg = _fetch_file(source, "generation_config.json", required=False)
    if eos is None and gen_cfg is not None:
        eos = _first_int(json.loads(gen_cfg.read_text(encoding="utf-8")).get("eos_token_id"))
    return {
        "model_id": str(source),
        "eos_token_id": eos,
        "bos_token_id": _first_int(config.get("bos_token_id")),
        "files": files,
    }


def load_calibration_texts(dataset: str) -> Iterable[str]:
    """Load calibration documents. ``dataset`` is a text file path, a wikitext
    config name, or a Hugging Face dataset id with a ``text`` column."""
    if os.path.isfile(dataset):
        text = Path(dataset).read_text(encoding="utf-8")
        return [p for p in text.split("\n\n") if p.strip()]
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise _gptq.MissingDependencyError(
            "The 'datasets' package is required to load calibration data. "
            "Install it with: pip install 'quf[convert]'."
        ) from exc
    if dataset.startswith("wikitext"):
        # Recent `datasets` releases require the namespaced repo id.
        ds = load_dataset("Salesforce/wikitext", dataset, split="train")
    else:
        ds = load_dataset(dataset, split="train")
    return [t for t in ds["text"] if t and t.strip()]


def _load_c4(max_docs: int = SECOND_SOURCE_MAX_DOCS) -> list[str]:
    """First ``max_docs`` non-empty documents of the English C4 train split (streamed)."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise _gptq.MissingDependencyError(
            "The 'datasets' package is required to load calibration data. "
            "Install it with: pip install 'quf[convert]'."
        ) from exc
    ds = load_dataset(SECOND_SOURCE_DATASET, "en", split="train", streaming=True)
    out: list[str] = []
    for row in ds:
        t = row.get("text")
        if t and t.strip():
            out.append(t)
            if len(out) >= max_docs:
                break
    return out


def default_calibration_sources() -> list[CalibrationSource]:
    """wikitext-2 train (50 percent) and English C4 (50 percent)."""
    return [
        CalibrationSource("wikitext-2-raw-v1", lambda: load_calibration_texts(DEFAULT_DATASET), 0.5),
        CalibrationSource("c4-en", _load_c4, 0.5),
    ]


def _coerce_other(name: str, arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f":
        return arr.astype(np.float16)  # embeddings, lm_head, norms, biases
    if arr.dtype == np.int32:
        return arr
    raise ConvertError(f"tensor '{name}' has unsupported dtype {arr.dtype}")


def _fp16_size(result: _gptq.GPTQResult) -> int:
    n = sum(q.in_features * q.out_features for q in result.linears.values())
    n += sum(int(a[0].size) for a in result.int8.values())
    n += sum(int(a.size) for a in result.others.values())
    n += sum(qe.rows * qe.cols for qe in result.embeddings.values())
    return 2 * n


def convert_checkpoint(
    source: str,
    output: str | os.PathLike,
    *,
    dataset: str | None = None,
    n_samples: int | None = None,
    seq_len: int | None = None,
    seed: int = 0,
    legacy_calibration: bool = False,
    multi_source_calibration: bool = False,
    overwrite: bool = False,
    symmetric: bool = False,
    group_size: int = _gptq.GROUP_SIZE,
    act_order: bool = False,
    clip_search: bool = False,
    damp: float = 0.01,
    precision_plan: PrecisionPlan | None = None,
    embed_precision: str = "int8",
    embed_group_size: int = 64,
    embed_method: str = "clip",
    texts: Iterable[str] | None = None,
    quantizer: Callable[..., _gptq.GPTQResult] | None = None,
    echo: Callable[[str], None] | None = print,
) -> ConvertResult:
    """Quantize ``source`` (Hub id or local dir) to 4 bits and write ``output``.

    ``group_size`` (32, 64 or 128), ``act_order`` and ``clip_search`` go to the
    quantizer. ``precision_plan`` keeps selected modules at int8 or fp16 and the
    header records the module precision map. With the defaults the file is version 1.

    ``embed_precision`` (``fp16``, ``int8`` or ``int4``) stores the embedding table, and an
    untied lm_head, as a Quantized_Embedding (format version 3) with ``embed_group_size``
    (32 or 64) and ``embed_method`` (``clip`` for MSE-optimal clipping, ``rtn`` for plain
    round-to-nearest). The default is ``int8`` (size study, docs/ACCURACY.md); ``fp16`` leaves the file as before.

    ``texts`` overrides dataset loading, and ``quantizer`` overrides the
    native GPTQ run (both exist for testing). The same ``symmetric`` value is
    passed to the quantizer and recorded in the header.

    Calibration: ``dataset`` set means one source. Otherwise the default is
    128 x 1024 tokens from wikitext-2 train. ``multi_source_calibration`` selects
    wikitext-2 and C4 (50/50, length 2048). With ``legacy_calibration`` the 0.1.0
    settings apply (wikitext-2 only, length 512).
    Explicit ``n_samples`` / ``seq_len`` always win.
    """
    n_samples = DEFAULT_N_SAMPLES if n_samples is None else n_samples
    if seq_len is None:
        seq_len = (LEGACY_SEQ_LEN if legacy_calibration
                   else MULTI_SOURCE_SEQ_LEN if multi_source_calibration and dataset is None
                   else DEFAULT_SEQ_LEN)
    if embed_precision not in _gptq.EMBED_PRECISIONS:
        raise ConvertError(f"unsupported embed precision {embed_precision!r}: use fp16, int8 or int4")
    if embed_group_size not in (32, 64):
        raise ConvertError(f"unsupported embed group size {embed_group_size}: use 32 or 64")
    if embed_method not in ("rtn", "clip"):
        raise ConvertError(f"unsupported embed method {embed_method!r}: use rtn or clip")
    output = Path(output)
    if output.exists() and not overwrite:
        raise OutputExistsError(output)
    config = read_source_config(source)
    architecture = check_architecture(config)

    model_config = build_model_config(config, architecture)
    tokenizer_info = build_tokenizer_info(source, config)

    tok = TokenizerWrapper(source if os.path.isdir(source) else None,
                           hf_tokenizer=_hf_from_info(tokenizer_info, source))
    if texts is not None:
        cal_input = texts
    elif dataset is not None:
        cal_input = load_calibration_texts(dataset)
    elif multi_source_calibration and not legacy_calibration:
        cal_input = default_calibration_sources()
    else:
        cal_input = load_calibration_texts(DEFAULT_DATASET)
    calib = build_calibration(
        cal_input, tok, n_samples, seq_len, seed,
    )

    runner = quantizer or _gptq.run_native_gptq
    extra: dict[str, Any] = {}
    if act_order:
        extra["act_order"] = True
    if clip_search:
        extra["clip_search"] = True
    if damp != 0.01:
        extra["damp"] = damp
    if precision_plan:
        extra["precision_plan"] = precision_plan
    result = runner(source, calib, group_size=group_size, symmetric=symmetric, **extra)

    _gptq.quantize_embeddings(result, embed_precision, embed_group_size, embed_method)
    tensors = _gptq.to_tensor_dict(result)
    tensors = {k: (v if k.endswith((".qweight", ".scales", ".qzeros", ".g_idx", ".qweight_int8", ".embed_q4", ".embed_q8", ".embed_scales")) else _coerce_other(k, v))
               for k, v in tensors.items()}
    quantization: dict[str, Any] = {
        "method": "gptq",
        "bits": _gptq.BITS,
        "group_size": result.group_size,
        "symmetric": bool(result.symmetric),
        "pack_dtype": "int32",
        "pack_order": "lsb_first",
    }
    if result.act_order:
        quantization["act_order"] = True
    if result.modules and any(p != "int4" for p in result.modules.values()):
        quantization["modules"] = dict(result.modules)
    quantization.update(_gptq.embedding_header_specs(result))
    header = QUFHeader(
        model_config=model_config, tokenizer=tokenizer_info, quantization=quantization
    )

    tmp = output.with_name(output.name + ".partial")
    try:
        write_quf(tmp, header, tensors)
        os.replace(tmp, output)
    except BaseException:
        for p in (tmp, output if not _existed_before(output, overwrite) else None):
            if p is not None and Path(p).exists():
                try:
                    Path(p).unlink()
                except OSError:  # pragma: no cover
                    logger.warning("could not delete partial file %s", p)
        raise

    size = output.stat().st_size
    fp16 = _fp16_size(result)
    report = SizeReport.from_tensors(
        [(k, _dtype_name(v), v.shape) for k, v in tensors.items()],
        quantization.get("modules"),
        file_bytes=size,
    )
    res = ConvertResult(
        output=output, size_bytes=size, fp16_size_bytes=fp16, ratio=size / fp16, report=report
    )
    if echo is not None:
        echo(res.message)
    return res


def _dtype_name(arr: np.ndarray) -> str:
    return {"float16": "fp16", "float32": "fp32", "int32": "int32", "int8": "int8"}[arr.dtype.name]


def _existed_before(output: Path, overwrite: bool) -> bool:
    # With overwrite, os.replace is atomic, so a failure leaves any prior file intact.
    return overwrite


def _hf_from_info(info: dict[str, Any], source: str):
    """Build an HF Tokenizer from the tokenizer.json content already read."""
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise _gptq.MissingDependencyError(
            "Hugging Face 'tokenizers' is required. Install it with: pip install 'quf[convert]'."
        ) from exc
    return Tokenizer.from_str(info["files"]["tokenizer.json"])
