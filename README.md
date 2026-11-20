# QUF: Quantized Unified Format

QUF is a single-file container (`.quf`) for a 4-bit quantized decoder-only LLM. One file holds the model config, tokenizer files, quantization scheme and all tensors. This package provides a reference writer and reader, a native GPTQ converter, a runtime with CPU and CUDA backends, and a benchmark suite.

QUF runs natively on Windows (and any platform with PyTorch). No WSL or Linux environment is needed.

Supported architectures: `Qwen2ForCausalLM` (first), with `Qwen3ForCausalLM` addable through the architecture allow-list. By default, weights are 4-bit with group size 32, act-order and clip search. `down_proj` and the last layer are stored at int8 (the `sensitive` preset). Embeddings and `lm_head` stay fp16.

**Accuracy in one line.** A 4-bit model does not match fp16. The new defaults narrow the gap to +2.35% (0.5B) and +1.01% (1.5B) WikiText-2 perplexity over fp16, down from +7.42% and +6.24% with the 0.1.0 settings. See [Measured results](#measured-results).

**Format version 2.** The new defaults write QUF format version 2 files. Release 0.1.0 readers cannot open them and reject them with an unsupported version error. Version 1 files still open. See [docs/FORMAT.md](docs/FORMAT.md).

## Quickstart

```
pip install quf
quf convert Qwen/Qwen2.5-0.5B out.quf
quf run out.quf --prompt "The capital of France is" --max-new-tokens 32 --temperature 0
quf bench out.quf --out results.json
```

`quf convert` needs the convert extra: `pip install "quf[convert]"`.

## Status: what is and is not verified

| Component | Status |
| --- | --- |
| Format writer/reader, quantizer, packer, CPU backend | Implemented. Covered by the CPU test suite. |
| Native GPTQ converter (`quf convert`) | Verified on native Windows with Qwen/Qwen2.5-0.5B and 1.5B ([docs/RESULTS.md](docs/RESULTS.md)). |
| `torch_gpu` backend on RTX 5080 Laptop (compute capability 12.0) | Verified: `quf run` and `quf bench` work on CUDA with a real converted Qwen2.5-0.5B. |
| `torch_fused` backend | Verified, but slower than `torch_gpu` on this GPU (uses less memory). |
| gigatoken on Windows | Verified: gigatoken 0.11.0 installs and runs, 36.8x faster than HF `tokenizers` on WikiText-2 with identical ids. |
| Perplexity target (3 percent) | With the 0.1.0 settings it was missed (+7.42% on 0.5B). With the new defaults it is reached on WikiText-2 for Qwen2.5-0.5B (+2.35%) and 1.5B (+1.01%). On held-out C4 it is missed on 0.5B (+3.30%) and reached on 1.5B (+1.63%). |
| GGUF comparison | Not run (unverified). |
| lm-evaluation-harness scores | Not run (unverified). |
| Marlin backend | Optional, never selected automatically, not verified on this hardware. |

Details are in [docs/ENV.md](docs/ENV.md) and [docs/RESULTS.md](docs/RESULTS.md).

## Installation

Requires Python 3.10 or newer.

```
pip install quf             # core: format, quantizer, runtime (torch is a base dependency)
pip install "quf[cuda]"     # same as core; kept so the extra name resolves
pip install "quf[convert]"  # adds transformers, datasets, accelerate, gigatoken, tokenizers,
                            # safetensors, huggingface_hub for `quf convert`
pip install "quf[test]"     # pytest and hypothesis
```

From a source checkout, use `pip install .` and the same extras.

The `quf` console script is installed with the package. Run `quf --help` to list subcommands: `info`, `convert`, `run`, `bench`.

For a CUDA build of PyTorch on an RTX 50-series GPU, install a CUDA 12.8 wheel (the version recorded in ENV.md is `2.11.0+cu128`). With no GPU, everything works on CPU.

## Convert

`quf convert` loads a Hugging Face Qwen checkpoint with transformers and quantizes it with QUF's own GPTQ implementation. It runs on native Windows, Linux and macOS.

```
quf convert Qwen/Qwen2.5-0.5B out.quf
quf convert ./local-model out.quf --dataset calib.txt --samples 128 --seq-len 512 --seed 0
```

**Defaults (0.2.0).** Group size 32, act-order on, clip search on, the `sensitive` precision preset (all `down_proj` modules and the last layer at int8, the rest int4), and calibration on the wikitext-2 train split, 128 samples of 1024 tokens. These defaults were chosen from measured runs in [docs/ACCURACY.md](docs/ACCURACY.md).

**Opt-outs and options.**

| Option | Effect |
| --- | --- |
| `--group-size {32,64,128}` | Quantization group size (default 32). `--group-size 128` is the 0.1.0 group size. Pair it with `--no-clip-search`, because clip search alone made the g128 model worse. |
| `--no-act-order` | Turn off act-order (on by default). Act-order writes `g_idx` tensors and needs format version 2. |
| `--no-clip-search` | Turn off the MSE clip search (on by default). |
| `--preset {down_proj,sensitive,none}` | Precision preset. Default is `sensitive`. `--preset none` stores every quantized module at int4. `down_proj` stores only `down_proj` at int8. |
| `--mixed PATTERN=PRECISION`, `--int8-modules PATTERN` | Per-module precision (int4, int8, fp16) by glob. When given, the default preset is not applied. |
| `--legacy-calibration` | 0.1.0 calibration: wikitext-2 only, 128 samples of 512 tokens. |
| `--multi-source-calibration` | Calibrate on wikitext-2 train and English C4 (50/50), 128 samples of 2048 tokens. It is not the default: it measured worse on WikiText-2 perplexity and took longer to convert (see RESULTS.md). It may rank differently on other evaluations. |
| `--dataset`, `--samples` (128), `--seq-len` (1024), `--seed` (0) | Calibration source and size. |
| `--overwrite` | Replace an existing output file. |

To reproduce the 0.1.0 output (a version 1 file), use `quf convert MODEL out.quf --group-size 128 --no-act-order --no-clip-search --preset none --legacy-calibration`.

The command prints the output size and compression ratio. It refuses to replace an existing output unless `--overwrite` is given, rejects unsupported architectures before heavy work, and deletes a partial file if conversion fails.

**Format version 2.** Group size 32 or 64, act-order (`g_idx`) and int8 or fp16 modules need format version 2. The writer picks the version from the features a file uses, so the opt-out command above still produces a version 1 file. 0.1.0 readers reject version 2 files with an unsupported version error. This is intended.

**Multi-source calibration.** With `--multi-source-calibration`, the sample count is split by share between wikitext-2 train and the first 4000 non-empty documents of the English C4 train split (streamed from `allenai/c4`). Each source is shuffled with `seed + source index`. Leftover samples go to the largest fractional parts, and the same sources and seed give identical samples. A source with too few tokens raises an error that names the source and the shortfall.

Inspect the result:

```
quf info out.quf
```

This prints the model config, quantization scheme, tensor count and file size. A corrupt file exits with status 1 and an error message.

## Run

```
quf run out.quf --prompt "Hello" --max-new-tokens 64 --temperature 0
```

Options: `--max-new-tokens` (64), `--temperature` (0 means greedy), `--top-k`, `--top-p`, `--seed` (0), `--no-stream`, `--device {auto,cpu,cuda}` (auto), `--backend {auto,cached,fused,marlin}` (auto), `--prefer-fused`.

Backends:

- `cpu`: used when no NVIDIA GPU is present.
- `torch_gpu` (`--backend cached`): dequantizes weights to fp16 at load. Default on GPU when the weights fit in free GPU memory.
- `torch_fused` (`--backend fused`): keeps packed int4 on the GPU. Chosen automatically when the cached weights do not fit, or with `--prefer-fused`.
- `marlin`: optional, only with `--backend marlin`, never automatic. Unverified on this hardware.

The choice and its reason are logged to stderr. `--device cuda` without an NVIDIA GPU exits with status 1. A prompt that exceeds the context length and GPU out-of-memory also exit with status 1.

## Benchmark

```
quf bench out.quf --out results.json
quf bench out.quf --out results.json --fp16-ref Qwen/Qwen2.5-0.5B --gguf model-q4.gguf --device auto
```

The results JSON records WikiText perplexity, file size, tokens per second, backend, device name and software versions. With `--fp16-ref`, the relative perplexity increase over fp16 is printed next to the documented target of at most 3 percent. The target is reported, not enforced. `--gguf` adds a GGUF Q4 comparison, but no GGUF rows were run for the results below (unverified). An lm-evaluation-harness adapter is in `quf/bench/lm_eval_adapter.py`. No lm-eval scores were produced (unverified).

## Measured results

Machine: RTX 5080 Laptop GPU, native Windows, PyTorch 2.11.0+cu128. Perplexity uses 2048-token windows. WikiText-2 is the test split. C4 is the first ~1.3M characters of the English validation split, never used for calibration. Source: `docs/bench_accuracy.json`. Details: [docs/RESULTS.md](docs/RESULTS.md).

**A 4-bit model does not beat fp16.** Every quantized row below has higher perplexity than fp16. The new defaults narrow the gap, and they cost file size.

Qwen/Qwen2.5-0.5B (fp16 perplexity: WikiText-2 13.0703, C4 19.7930):

| Run | WikiText-2 ppl | vs fp16 | C4 ppl | vs fp16 | File size (bytes) | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | 13.0703 | n/a | 19.7930 | n/a | 988,065,536 (computed) | 15.50 |
| RTN g128 | 15.5417 | +18.91% | 23.5073 | +18.77% | 469,483,072 | 22.92 |
| RTN g32 | 14.3826 | +10.04% | 21.6024 | +9.14% | 490,449,792 | 24.13 |
| 0.1.0 settings (GPTQ g128) | 14.0401 | +7.42% | 21.7476 | +9.87% | 469,483,072 | 26.66 |
| **New defaults (g32 + clip + act-order + sensitive int8)** | **13.3775** | **+2.35%** | **20.4453** | **+3.30%** | 539,605,632 | 16.54 |

Qwen/Qwen2.5-1.5B (fp16 perplexity: WikiText-2 9.2650, C4 14.6605):

| Run | WikiText-2 ppl | vs fp16 | C4 ppl | vs fp16 | File size (bytes) | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | 9.2650 | n/a | 14.6605 | n/a | 3,087,428,608 (computed) | 13.40 |
| RTN g128 | 10.3555 | +11.77% | 16.1390 | +10.09% | 1,158,904,640 | 22.44 |
| RTN g32 | 9.8572 | +6.39% | 15.4676 | +5.51% | 1,235,674,048 | 22.28 |
| 0.1.0 settings (GPTQ g128) | 9.8434 | +6.24% | 15.6935 | +7.05% | 1,158,904,640 | 21.86 |
| **New defaults** | **9.3585** | **+1.01%** | **14.8997** | **+1.63%** | 1,413,323,520 | 22.19 |

What this shows:

- Against the 3% target, the new defaults are within it on WikiText-2 for both models and on C4 for 1.5B. On C4 for 0.5B they are at +3.30%, so the target is **missed** there.
- The calibration data is WikiText-2 train, so the WikiText-2 column is closer to the calibration domain than C4. The C4 column is the held-out check.
- Sizes: the new-default file is 539.6 MB for 0.5B (0.5461 of fp16) and 1,413.3 MB for 1.5B (0.4578 of fp16). The 0.1.0 file was 469.5 MB and 1,158.9 MB.
- Tokens per second are single short decode runs with large run-to-run variation on this laptop. The fp16 row uses Hugging Face generate. Do not read an order into small differences.
- Not run (unverified): GGUF Q4_K_M, Q4_0 and Q8_0 rows, and lm-evaluation-harness scores. The harnesses are included, but no results exist for them.

Backend speed for 0.5B (0.1.0-era file, 64 new tokens): CPU 9.6 tokens/s, `torch_gpu` 20.7, `torch_fused` 9.1 (slower, uses less GPU memory). gigatoken tokenized WikiText-2 train in 0.23 s against 8.44 s for HF tokenizers (36.8x, identical ids). The decode loop is not optimized.
## Documentation

- [docs/FORMAT.md](docs/FORMAT.md): byte layout, header fields, tensor naming, dtypes and the canonical Packed_Int4 order.
- [docs/ENV.md](docs/ENV.md): environment results for native Windows and the backend selection rules.
- [docs/RESULTS.md](docs/RESULTS.md): measured accuracy, size and speed.

README.md, docs/ENV.md and docs/FORMAT.md ship with the package: they are in the source distribution and are installed to `share/doc/quf` by the wheel.

## Tests

```
pytest -m "not gpu and not convert"   # CPU suite, no NVIDIA GPU needed
pytest -m gpu                         # needs an NVIDIA GPU
pytest -m convert                     # needs the convert extra and downloads a model
```

Tests marked `gpu` are skipped automatically when no NVIDIA GPU is available.

## Links and license

Source and issues: https://github.com/vtrsharvin-sj/Quantized-Model-Format

Released under the MIT license. See [LICENSE](LICENSE).
