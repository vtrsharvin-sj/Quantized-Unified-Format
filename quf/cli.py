"""Command-line interface: ``quf info``, ``convert``, ``run`` and ``bench``.

Every failure ends in a single ``quf: error: ...`` line and a nonzero exit status,
never a Python traceback (use ``--verbose`` to see details).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings
from typing import Sequence

from quf import __version__

COMMANDS = ("info", "convert", "run", "bench")

# Release defaults for ``quf convert``, chosen by perplexity (docs/upgrade_eval.json).
DEFAULT_GROUP_SIZE = 32
DEFAULT_ACT_ORDER = True
DEFAULT_CLIP_SEARCH = True
DEFAULT_PRESET = "size_b"
DEFAULT_MAX_WINDOWS = 40


def _add_toggle(p: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """``--name`` / ``--no-name`` pair. Avoids argparse.BooleanOptionalAction, which a stale
    PyPI ``argparse`` backport that shadows the standard library does not provide."""
    dest = name.replace("-", "_")
    p.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=help_text)
    p.add_argument(f"--no-{name}", dest=dest, action="store_false", help=argparse.SUPPRESS)


class _VersionAction(argparse.Action):
    """Print the version on stdout (some argparse versions use stderr)."""

    def __call__(self, parser, namespace, values, option_string=None):
        sys.stdout.write(f"quf {__version__}\n")
        sys.stdout.flush()
        parser.exit()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quf", description="QUF: quantized unified format tools."
    )
    parser.add_argument("--version", action=_VersionAction, nargs=0,
                        help="show the version and exit")
    sub = parser.add_subparsers(dest="command", metavar="{info,convert,run,bench}")

    p_info = sub.add_parser("info", help="print the header summary of a .quf file")
    p_info.add_argument("path", help="path to a .quf file")

    p_conv = sub.add_parser(
        "convert", help="quantize a Hugging Face Qwen checkpoint into a .quf file"
    )
    p_conv.add_argument("source", help="Hugging Face model id or local directory")
    p_conv.add_argument("output", help="output .quf path")
    p_conv.add_argument("--dataset", default=None,
                        help="calibration dataset: text file, wikitext config, or HF dataset id "
                             "(default: wikitext-2 train; see --multi-source-calibration)")
    p_conv.add_argument("--samples", type=int, default=None,
                        help="number of calibration samples (default: 128)")
    p_conv.add_argument("--seq-len", type=int, default=None,
                        help="calibration sequence length (default: 1024; 512 with --legacy-calibration)")
    p_conv.add_argument("--legacy-calibration", action="store_true",
                        help="use the 0.1.0 calibration: wikitext-2 only, 128 samples of length 512")
    p_conv.add_argument("--multi-source-calibration", action="store_true",
                        help="calibrate on wikitext-2 train and C4 English (50/50), 128 x 2048 tokens "
                             "(measured worse on WikiText-2 perplexity, so not the default)")
    p_conv.add_argument("--seed", type=int, default=0, help="calibration shuffle seed")
    p_conv.add_argument("--group-size", type=int, choices=(32, 64, 128), default=DEFAULT_GROUP_SIZE,
                        help=f"quantization group size (default: {DEFAULT_GROUP_SIZE})")
    _add_toggle(p_conv, "act-order", DEFAULT_ACT_ORDER,
                "quantize columns by descending Hessian diagonal (writes g_idx, format "
                "version 2); default on, --no-act-order turns it off")
    _add_toggle(p_conv, "clip-search", DEFAULT_CLIP_SEARCH,
                "choose the MSE-optimal clipping ratio per group; default on, "
                "--no-clip-search turns it off")
    p_conv.add_argument("--mixed", action="append", default=[], metavar="PATTERN=PRECISION",
                        help="keep modules matching a glob at int4, int8 or fp16, for example "
                             "\"*.down_proj=int8\" or \"lm_head=fp16\"; repeatable or comma separated; "
                             "the last matching rule wins")
    p_conv.add_argument("--int8-modules", action="append", default=[], metavar="PATTERN",
                        help="shortcut for --mixed \"PATTERN=int8\"; repeatable")
    p_conv.add_argument("--preset", action="append", default=[], choices=("down_proj", "sensitive", "size_a", "size_b", "size_c", "none"),
                        help="precision preset from docs/ACCURACY.md: down_proj (all down_proj at int8), "
                             "sensitive (down_proj and the last layer at int8; the v020 plan), "
                             "size_a (down_proj int8, last layer int4), "
                             "size_b (last layer int8, down_proj int4), size_c (all int4 by plan), "
                             "or none (all int4). "
                             f"Default when no --preset, --mixed or --int8-modules is given: {DEFAULT_PRESET}")
    p_conv.add_argument("--overwrite", action="store_true",
                        help="replace the output file if it exists")

    p_run = sub.add_parser("run", help="generate text from a .quf file")
    p_run.add_argument("path", help="path to a .quf file")
    p_run.add_argument("--prompt", required=True, help="prompt text")
    p_run.add_argument("--max-new-tokens", type=int, default=64,
                       help="maximum number of tokens to generate (default: 64)")
    p_run.add_argument("--temperature", type=float, default=0.0,
                       help="sampling temperature; 0 is greedy decoding (default: 0)")
    p_run.add_argument("--top-k", type=int, default=None, help="top-k sampling cutoff")
    p_run.add_argument("--top-p", type=float, default=None, help="top-p (nucleus) cutoff")
    p_run.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    p_run.add_argument("--no-stream", dest="stream", action="store_false",
                       help="print the full text at the end instead of streaming tokens")
    p_run.set_defaults(stream=True)
    p_run.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false",
                       help="decode eagerly instead of replaying a CUDA graph (same as QUF_NO_CUDA_GRAPH=1)")
    p_run.set_defaults(cuda_graph=True)
    p_run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                       help="device to run on; auto picks the best Backend (default: auto)")
    _add_backend_args(p_run)

    p_bench = sub.add_parser("bench", help="benchmark a .quf file (perplexity, size, speed)")
    p_bench.add_argument("path", help="path to a .quf file")
    p_bench.add_argument("--fp16-ref", default=None,
                         help="fp16 Hugging Face model id or directory for the reference perplexity")
    p_bench.add_argument("--gguf", default=None, help="GGUF Q4 file to compare against")
    p_bench.add_argument("--out", required=True, help="path of the results JSON file")
    p_bench.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                         help="device to run on; auto picks the best Backend (default: auto)")
    p_bench.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS,
                         help="evaluate perplexity on at most this many 2048-token WikiText-2 windows "
                              f"(default: {DEFAULT_MAX_WINDOWS}, a few minutes; 0 = the full test split)")
    _add_backend_args(p_bench)
    for p in (p_info, p_conv, p_run, p_bench):
        p.add_argument("--verbose", action="store_true",
                       help="show library warnings, debug messages and error details")
    return parser


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("auto", "cached", "fused", "marlin", "int4"), default="auto",
                   help="GPU backend; auto picks int4 (Triton) when it works, else cached or fused; int4 forces "
                        "the Triton backend; marlin needs the optional gptqmodel package")
    p.add_argument("--prefer-fused", action="store_true",
                   help="in auto mode, use the fused backend when it was measured faster")


class CLIError(Exception):
    """A user-facing error: printed as ``quf: error: <message>``, exit status 1."""


_QUIET_ENV = {
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "DATASETS_VERBOSITY": "error",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
}
_THIRD_PARTY_LOGGERS = ("transformers", "datasets", "huggingface_hub", "httpx", "httpcore",
                        "urllib3", "filelock", "fsspec", "torch", "accelerate")
_HANDLER_TAG = "_quf_cli_handler"


def _configure_logging(verbose: bool, imports_torch: bool = False) -> None:
    """Short ``quf: ...`` INFO lines on stderr; third-party libraries stay quiet unless verbose."""
    import logging  # noqa: PLC0415

    if not verbose:
        for key, value in _QUIET_ENV.items():
            os.environ.setdefault(key, value)
    root = logging.getLogger("quf")
    for h in list(root.handlers):
        if getattr(h, _HANDLER_TAG, False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("quf: %(message)s"))
    setattr(handler, _HANDLER_TAG, True)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not verbose:
        if imports_torch:
            try:  # torch resets its own logger levels on import, so import it before lowering them
                import torch  # noqa: F401, PLC0415
            except Exception:  # noqa: BLE001
                pass
        for name in _THIRD_PARTY_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)


_NETWORK_NAMES = ("connecterror", "connectionerror", "connecttimeout", "readtimeout", "timeout",
                  "maxretryerror", "nameresolutionerror", "gaierror", "offlinemodeisenabled",
                  "proxyerror", "sslerror", "remotedisconnected", "localentrynotfounderror")
_NETWORK_TEXT = ("failed to resolve", "getaddrinfo", "name resolution", "connection error",
                 "connection refused", "connection aborted", "timed out", "network is unreachable",
                 "offline mode", "max retries exceeded", "cannot connect", "couldn't connect",
                 "an error happened while trying to locate the file on the hub")
_NOT_FOUND_NAMES = ("repositorynotfounderror", "revisionnotfounderror", "gatedrepoerror",
                    "repositoryerror", "entrynotfounderror")
_NOT_FOUND_TEXT = ("is not a valid model identifier", "is not a local folder", "repository not found",
                   "404 client error", "401 client error", "does not appear to have a file named",
                   "not found on the hub")


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _first_line(text: str, limit: int = 300) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 3] + "..."
    return ""


def _describe_error(exc: BaseException, args: argparse.Namespace | None = None) -> str:
    """Turn an exception into one short, actionable line."""
    from quf.format.errors import QUFFormatError  # noqa: PLC0415

    source = getattr(args, "source", None)
    path = getattr(args, "path", None)
    if isinstance(exc, CLIError):
        return str(exc)
    if isinstance(exc, QUFFormatError):
        return f"not a valid .quf file ({path}): {_first_line(exc)}" if path else _first_line(exc)
    if isinstance(exc, FileNotFoundError) and not source_is_hub_like(exc):
        return f"file not found: {exc.filename or path}"
    if isinstance(exc, IsADirectoryError):
        return f"expected a file but got a directory: {exc.filename or path}"
    if isinstance(exc, PermissionError):
        return f"permission denied: {exc.filename or path}"
    if isinstance(exc, MemoryError):
        return "out of memory"
    for e in _chain(exc):
        if isinstance(e, ImportError):
            name = (getattr(e, "name", None) or "").split(".")[0]
            if name in ("transformers", "datasets", "accelerate", "safetensors", "gigatoken",
                        "huggingface_hub", "tokenizers", "") or "quf[convert]" in str(e):
                return (f"missing optional package '{name or '?'}'; install the conversion extras with: "
                        "pip install \"quf[convert]\"")
            return f"missing Python package '{name}': {_first_line(e)}"
    if type(exc).__name__ == "OutOfMemoryError":
        return "GPU out of memory; try --device cpu or a smaller model"

    for e in _chain(exc):
        n = type(e).__name__.lower()
        text = str(e).lower()
        if n in _NOT_FOUND_NAMES or any(t in text for t in _NOT_FOUND_TEXT):
            target = source or path or "the model"
            return (f"model '{target}' was not found on Hugging Face Hub (check the id, "
                    "or log in if it is private)")
    for e in _chain(exc):
        n = type(e).__name__.lower()
        text = str(e).lower()
        if n in _NETWORK_NAMES or any(t in text for t in _NETWORK_TEXT):
            return ("could not reach Hugging Face Hub (no network?); check your internet connection "
                    "or pass a local model directory")

    if isinstance(exc, KeyError):
        return f"the file or model config is missing the required field {_first_line(exc)}"
    msg = _first_line(exc)
    if isinstance(exc, (ValueError, RuntimeError, OSError)) and msg:
        return msg
    return f"{type(exc).__name__}: {msg}".rstrip(": ") + " (rerun with --verbose for details)"


def source_is_hub_like(exc: FileNotFoundError) -> bool:
    """A FileNotFoundError raised for something that is not a plain filesystem path (hub cache, etc.)."""
    return exc.filename is None and "hub" in str(exc).lower()


def _quiet_third_party(args: argparse.Namespace) -> None:
    """Silence transformers/datasets/huggingface_hub progress bars and notices unless --verbose."""
    if getattr(args, "verbose", False):
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for mod, fn in (("transformers.utils.logging", "disable_progress_bar"),
                    ("datasets.utils.logging", "disable_progress_bar"),
                    ("huggingface_hub.utils", "disable_progress_bars")):
        try:
            import importlib  # noqa: PLC0415

            getattr(importlib.import_module(mod), fn)()
        except Exception:  # noqa: BLE001 - optional packages or API differences
            pass
    for mod, fn in (("transformers.utils.logging", "set_verbosity_error"),
                    ("datasets.utils.logging", "set_verbosity_error"),
                    ("huggingface_hub.utils.logging", "set_verbosity_error")):
        try:
            import importlib  # noqa: PLC0415

            getattr(importlib.import_module(mod), fn)()
        except Exception:  # noqa: BLE001
            pass


def _require_file(path: str, what: str = ".quf file") -> None:
    if not os.path.exists(path):
        raise CLIError(f"{what} not found: {path}")
    if os.path.isdir(path):
        raise CLIError(f"expected a {what} but got a directory: {path}")


def _check_generation_args(args: argparse.Namespace) -> None:
    if not args.prompt.strip():
        raise CLIError("the prompt is empty; pass some text with --prompt")
    if args.max_new_tokens < 1:
        raise CLIError("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        raise CLIError("--temperature must not be negative")
    if args.top_k is not None and args.top_k < 1:
        raise CLIError("--top-k must be at least 1")
    if args.top_p is not None and not (0 < args.top_p <= 1):
        raise CLIError("--top-p must be greater than 0 and at most 1")


def _gpu_hint(args: argparse.Namespace) -> None:
    """One line on stderr when an NVIDIA GPU exists but torch is CPU-only."""
    if args.device != "auto":
        return
    from quf.device import cpu_only_torch_hint  # noqa: PLC0415

    hint = cpu_only_torch_hint()
    if hint:
        print(f"quf: {hint}", file=sys.stderr)


def _cmd_bench(args: argparse.Namespace) -> int:
    from quf.bench.suite import PPL_TARGET_PERCENT, run_bench  # noqa: PLC0415

    _require_file(args.path)
    if args.max_windows is not None and args.max_windows < 0:
        raise CLIError("--max-windows must be 0 (all windows) or positive")
    windows = None if args.max_windows == 0 else args.max_windows
    _gpu_hint(args)
    _quiet_third_party(args)
    result = run_bench(
        args.path,
        fp16_ref=args.fp16_ref,
        gguf=args.gguf,
        out_json=args.out,
        device=args.device,
        backend_option=args.backend,
        prefer_fused=args.prefer_fused,
        max_windows=windows,
    )
    scope = "all windows" if windows is None else f"first {windows} windows of 2048 tokens"
    print(f"Backend: {result.backend} ({result.device_name})")
    print(f"Perplexity scope: {scope}")
    print(f"Perplexity ({result.dataset}): {result.perplexity:.4f}")
    print(f"File size: {result.file_size_bytes} bytes")
    print(
        f"Tokens per second: {result.tokens_per_second:.2f} (median of "
        f"{result.tokens_per_second_repeats}, min {result.tokens_per_second_min:.2f}, "
        f"max {result.tokens_per_second_max:.2f})"
        if result.tokens_per_second_repeats
        else f"Tokens per second: {result.tokens_per_second:.2f}"
    )
    if result.relative_ppl_increase is not None:
        print(f"Relative perplexity increase vs fp16: {100 * result.relative_ppl_increase:.2f}% "
              f"(documented target: at most {PPL_TARGET_PERCENT:.0f}%)")
    print(f"Results written to {args.out}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from quf.runtime.errors import GPUMemoryError  # noqa: PLC0415
    from quf.runtime.generate import ContextLengthError, generate  # noqa: PLC0415
    from quf.runtime.model import QUFModel  # noqa: PLC0415
    from quf.runtime.selector import NoGPUError  # noqa: PLC0415
    from quf.tokenization.wrapper import TokenizerWrapper  # noqa: PLC0415

    _check_generation_args(args)
    if args.device == "cuda":
        from quf.device import detect_devices  # noqa: PLC0415

        if not detect_devices().cuda_available:
            raise NoGPUError("no NVIDIA GPU was found")
    if os.path.isdir(args.path):
        raise CLIError(f"expected a .quf file but got a directory: {args.path}")
    _gpu_hint(args)
    for stream in (sys.stdout, sys.stderr):  # never crash on characters the console cannot show
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    try:
        model = QUFModel.load(
            args.path, device_option=args.device, backend_option=args.backend, prefer_fused=args.prefer_fused,
            cuda_graph=None if args.cuda_graph else False,
        )
        tok_json = ((model.header.tokenizer or {}).get("files") or {}).get("tokenizer.json")
        if tok_json is None:
            raise CLIError("the .quf file does not contain tokenizer.json")
        from tokenizers import Tokenizer  # noqa: PLC0415

        tok = TokenizerWrapper(hf_tokenizer=Tokenizer.from_str(tok_json))

        def _write(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        ids = generate(
            model,
            tok,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            stream=_write if args.stream else None,
            cuda_graph=None if args.cuda_graph else False,
        )
        if args.stream:
            sys.stdout.write("\n")
        else:
            print(tok.decode(ids))
    except GPUMemoryError as exc:
        raise CLIError(f"not enough GPU memory: {exc}") from exc
    except ContextLengthError as exc:
        raise CLIError(f"context length exceeded: {exc}") from exc
    return 0


_MAX_LISTED_MODULES = 8


def _summarize_modules(quantization: dict, verbose: bool) -> dict:
    """Collapse a long per-module precision map into counts per precision (``--verbose`` lists all)."""
    modules = quantization.get("modules")
    if verbose or not isinstance(modules, dict) or len(modules) <= _MAX_LISTED_MODULES:
        return quantization
    counts: dict = {}
    for precision in modules.values():
        counts[str(precision)] = counts.get(str(precision), 0) + 1
    out = dict(quantization)
    out["modules"] = {"count": len(modules), "by_precision": counts,
                      "note": "per-module list hidden; use --verbose to show it"}
    return out


def _cmd_info(args: argparse.Namespace) -> int:
    from quf.convert.precision import SizeReport  # noqa: PLC0415
    from quf.format.reader import QUFReader  # noqa: PLC0415

    _require_file(args.path)
    with QUFReader(args.path) as reader:
        header = reader.header
        size = reader.file_size
        print(f"File: {args.path}")
        print(f"File size: {size} bytes")
        print("Model config:")
        print(json.dumps(header.model_config, indent=2, sort_keys=True))
        print("Quantization scheme:")
        print(json.dumps(_summarize_modules(header.quantization, args.verbose), indent=2, sort_keys=True))
        print(f"Tensor count: {len(header.tensors)}")
        print(SizeReport.from_header(header, file_bytes=size).format())
    for line in _backend_report():
        print(line)
    return 0


def _backend_report() -> list[str]:
    """Lines for ``quf info``: the Triton_Gate result and the Backend the selector would pick."""
    if os.environ.get("QUF_NO_TRITON", "") == "1":
        gate = "disabled by QUF_NO_TRITON=1"
    else:
        try:
            from quf.device import detect_devices  # noqa: PLC0415

            if not detect_devices().cuda_available:
                gate = "no CUDA device"
            else:
                from quf.runtime.kernels.triton_kernels import triton_available  # noqa: PLC0415

                ok, reason = triton_available()
                gate = f"ok ({reason})" if ok else f"failed ({reason})"
        except Exception as exc:  # noqa: BLE001 - info must not fail on this
            gate = f"unknown ({type(exc).__name__}: {exc})"
    try:
        from quf.device import detect_devices  # noqa: PLC0415
        from quf.runtime.selector import int4_probe, select_backend  # noqa: PLC0415

        choice = select_backend("auto", detect_devices(), int4_probe=int4_probe).backend
    except Exception as exc:  # noqa: BLE001
        choice = f"unknown ({type(exc).__name__}: {exc})"
    return [f"Triton gate: {gate}", f"Selected backend (auto): {choice}"]


def _cmd_convert(args: argparse.Namespace) -> int:
    from quf.convert.converter import convert_checkpoint, read_source_config  # noqa: PLC0415
    from quf.convert.precision import PrecisionPlan  # noqa: PLC0415

    if os.path.exists(args.output) and not args.overwrite:
        raise CLIError(f"output file already exists: {args.output} (use --overwrite to replace it)")
    if os.path.isdir(args.output):
        raise CLIError(f"output path is a directory: {args.output}")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(out_dir):
        raise CLIError(f"output directory does not exist: {out_dir}")
    if args.samples is not None and args.samples < 1:
        raise CLIError("--samples must be at least 1")
    if args.seq_len is not None and args.seq_len < 2:
        raise CLIError("--seq-len must be at least 2")
    _check_convert_extras()
    _quiet_third_party(args)
    import logging  # noqa: PLC0415

    logging.getLogger("quf").info(
        "converting %s (loading the model and calibration data, then quantizing; "
        "this can take several minutes)", args.source)

    presets = [p for p in args.preset if p != "none"]
    if not args.preset and not args.mixed and not args.int8_modules:
        presets = [DEFAULT_PRESET]  # measured default, docs/upgrade_eval.json
    plan = None
    if args.mixed or args.int8_modules or presets:
        num_layers = None
        if presets:
            num_layers = read_source_config(args.source).get("num_hidden_layers")
        plan = PrecisionPlan.from_options(
            args.mixed, presets=presets, int8_modules=args.int8_modules, num_layers=num_layers
        )

    # The converter prints the size and ratio itself (echo=print), so the CLI
    # prints nothing more on success.
    convert_checkpoint(
        args.source,
        args.output,
        dataset=args.dataset,
        legacy_calibration=args.legacy_calibration,
        multi_source_calibration=args.multi_source_calibration,
        n_samples=args.samples,
        seq_len=args.seq_len,
        seed=args.seed,
        overwrite=args.overwrite,
        group_size=args.group_size,
        act_order=args.act_order,
        clip_search=args.clip_search,
        precision_plan=plan,
    )
    return 0


def _check_convert_extras() -> None:
    """Fail early (before any download) when the conversion packages are missing."""
    import importlib.util  # noqa: PLC0415

    def _present(name: str) -> bool:
        if name in sys.modules:
            return True
        try:
            return importlib.util.find_spec(name) is not None
        except (ValueError, ImportError):
            return True

    missing = [m for m in ("transformers", "datasets", "tokenizers") if not _present(m)]
    if missing:
        raise CLIError(f"missing optional package(s) {', '.join(missing)}; install the conversion "
                       "extras with: pip install \"quf[convert]\"")


_HANDLERS = {"info": _cmd_info, "convert": _cmd_convert, "run": _cmd_run, "bench": _cmd_bench}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help and --version exit 0, usage errors exit 2
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)

    if args.command is None:
        parser.print_help(sys.stderr)
        return 1

    handler = _HANDLERS.get(args.command)
    if handler is None:
        print(f"quf: '{args.command}' is not implemented yet", file=sys.stderr)
        return 1
    verbose = bool(getattr(args, "verbose", False))
    try:
        _configure_logging(verbose, imports_torch=args.command != "info")
    except Exception:  # noqa: BLE001 - logging setup must never stop a command
        pass
    try:
        with warnings.catch_warnings():
            if not verbose:
                warnings.simplefilter("ignore")
            return handler(args)
    except KeyboardInterrupt:
        print("\nquf: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 1
    except BaseException as exc:  # noqa: BLE001 - never show a traceback to the user
        if isinstance(exc, SystemExit):
            raise
        print(f"quf: error: {_describe_error(exc, args)}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
