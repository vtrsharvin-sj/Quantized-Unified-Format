# QUF: Quantized Unified Format

QUF is a single-file container (`.quf`) for a 4-bit quantized decoder-only LLM. One file holds the model config, tokenizer files, quantization scheme and all tensors. This package provides a reference writer and reader, a native GPTQ converter, a runtime with CPU and CUDA backends, and a benchmark suite.

QUF runs natively on Windows (and any platform with PyTorch). No WSL or Linux environment is needed.

Supported architectures: `Qwen2ForCausalLM` (first), with `Qwen3ForCausalLM` addable through the architecture allow-list. By default, weights are 4-bit with group size 32, act-order and clip search. The default preset is `size_b`: the last layer is stored at int8 and everything else at int4 (`down_proj` included). `--preset sensitive` reproduces 0.2.0 (all `down_proj` and the last layer at int8). Embeddings and `lm_head` stay fp16.

## Summary

Qwen2.5 on an RTX 5080 Laptop GPU, native Windows. Perplexity uses the `llama` protocol (positions 1024 to 2046 of each 2048-token window, as `llama-perplexity` does) over the full WikiText-2 test split and a held-out C4 slice. "QUF 0.3.0 defaults" is the `size_b` preset with the int4 backend.

Qwen/Qwen2.5-0.5B:

| | WikiText-2 ppl | C4 ppl | File size (bytes) | GPU decode tok/s |
| --- | --- | --- | --- | --- |
| fp16 | 11.7398 | 18.2394 | 988,065,536 | 45.8 (Hugging Face generate) |
| QUF 0.3.0 defaults (`size_b`, int4 backend) | 12.0769 (+2.87%) | 19.0473 (+4.43%) | 497,728,832 | 284 (QUF, CUDA graph; 508 MiB GPU after load) |
| GGUF Q4_K_M | 12.0425 (+2.58%) | 18.7241 (+2.66%) | 397,807,584 | 764.2 (llama.cpp) |

Qwen/Qwen2.5-1.5B:

| | WikiText-2 ppl | C4 ppl | File size (bytes) | GPU decode tok/s |
| --- | --- | --- | --- | --- |
| fp16 | 8.2964 | 13.4756 | 3,087,428,608 | 30.2 (Hugging Face generate) |
| QUF 0.3.0 defaults (`size_b`, int4 backend) | 8.4442 (+1.78%) | 13.8663 (+2.90%) | 1,257,449,536 | 146 (QUF, CUDA graph; 1320 MiB GPU after load) |
| GGUF Q4_K_M | 8.5998 (+3.66%) | 13.9536 (+3.55%) | 986,048,160 | 450.8 (llama.cpp) |

- **tok/s comes from different engines** (Hugging Face generate, the QUF runtime, llama.cpp) with different measurement setups, so it is a rough guide and not a like-for-like comparison.
- **QUF is still slower than llama.cpp**: 284 against 757 tok/s on 0.5B and 146 against 423 tok/s on 1.5B (llama-bench in the same session), which is 2.6 to 2.9x slower. The default int4 backend reads packed 4-bit weights with Triton kernels. It is 1.21x (0.5B) and 1.46x (1.5B) faster than the cached fp16 backend (`torch_gpu`, 234 and 100 tok/s in the same run) and uses 45 to 54 percent of its GPU memory (508 against 947 MiB, 1320 against 2951 MiB), with the same perplexity.
- **QUF files are larger than GGUF Q4_K_M**: 1.25x on 0.5B and 1.28x on 1.5B. The 0.5B Q4_K_M file is not pure 4-bit K-quant, because several tensors fall back to Q5_0 or Q8_0 (see [docs/RESULTS.md](docs/RESULTS.md)).
- **Perplexity**: QUF is ahead of Q4_K_M on 1.5B (both texts) and behind it on 0.5B (both texts). The 3% target against fp16 is met on WikiText-2 for both models and on C4 for 1.5B (+2.90%). It is missed on C4 for 0.5B (+4.43%).
- **lm-evaluation-harness** (hellaswag and arc_easy, 0-shot, 500 examples each): the 0.2.1 default `size_b` files are within one standard error of fp16 on every score except 0.5B arc_easy acc (0.612 against 0.636, 1.1 standard errors). The 0.2.0 `sensitive` files were within one standard error everywhere. Table below.

| Model | Variant | hellaswag acc | hellaswag acc_norm | arc_easy acc | arc_easy acc_norm |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | fp16 | 0.410 | 0.502 | 0.636 | 0.586 |
| Qwen2.5-0.5B | QUF 0.2.1 `size_b` | 0.410 | 0.500 | 0.612 | 0.588 |
| Qwen2.5-0.5B | QUF 0.2.0 `sensitive` | 0.424 | 0.496 | 0.620 | 0.564 |
| Qwen2.5-1.5B | fp16 | 0.486 | 0.602 | 0.750 | 0.710 |
| Qwen2.5-1.5B | QUF 0.2.1 `size_b` | 0.480 | 0.590 | 0.744 | 0.704 |
| Qwen2.5-1.5B | QUF 0.2.0 `sensitive` | 0.484 | 0.608 | 0.748 | 0.724 |

(stderr is about 0.022 for hellaswag and 0.019 to 0.022 for arc_easy.)
- Not evaluated: int6 and int5 storage. A `torch.compile` decode path was tried and skipped (see [docs/BACKENDS.md](docs/BACKENDS.md)).

Sources: `docs/bench_accuracy.json`, `docs/size_study.json`, [docs/ACCURACY.md](docs/ACCURACY.md), [docs/BACKENDS.md](docs/BACKENDS.md). More tables in [Measured results](#measured-results).

**Format version 2.** The defaults write QUF format version 2 files. Release 0.1.0 readers cannot open them and reject them with an unsupported version error. Version 1 files still open. See [docs/FORMAT.md](docs/FORMAT.md).

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
| `triton_int4` backend | Verified: default on GPU with `pip install "quf[fast]"`. 1.21x (0.5B) and 1.46x (1.5B) faster than `torch_gpu` in graph mode, about half the GPU memory, same perplexity. Slower in eager mode, so `auto` avoids it with `--no-cuda-graph`. |
| CUDA graph decode | Verified: default on GPU. With the default int4 backend (0.3.0): 284.2 tok/s on 0.5B and 145.6 on 1.5B, against 46.7 and 30.9 eager. With `torch_gpu` in the same run: 234.4 and 99.7 with the graph, 58.4 and 40.6 eager. Slower than llama.cpp (756.5 and 422.8 in that session). The 0.2.1 `torch_gpu` figures (255.2 and 82.7 graph) are in Measured results. |
| Perplexity target (3 percent) | `llama` protocol, default `size_b`: reached on WikiText-2 for 0.5B (+2.87%) and 1.5B (+1.78%) and on C4 for 1.5B (+2.90%). Missed on C4 for 0.5B (+4.43%). The 0.1.0 settings missed it. |
| GGUF comparison | Verified: Q4_K_M perplexity from `llama-perplexity` on the GPU, same text and protocol (see Summary). |
| lm-evaluation-harness scores | Verified on the default `size_b` files: within one standard error of fp16 on hellaswag and arc_easy, except 0.5B arc_easy acc (1.1 standard errors). 0.2.0 `sensitive` rows kept for comparison. |
| `torch.compile` decode | Skipped. It needs Triton, and with Triton it measured about 11% slower than the CUDA graph. |
| int6 / int5 storage | Not evaluated. |
| Marlin backend | Optional, never selected automatically, not verified on this hardware. |

Details are in [docs/ENV.md](docs/ENV.md) and [docs/RESULTS.md](docs/RESULTS.md).

## Installation

Requires Python 3.10 or newer.

```
pip install quf             # core: format, quantizer, runtime (torch is a base dependency)
pip install "quf[cuda]"     # same as core; kept so the extra name resolves
pip install "quf[convert]"  # adds transformers, datasets, accelerate, gigatoken, tokenizers,
                            # safetensors, huggingface_hub for `quf convert`
pip install "quf[fast]"     # adds Triton; enables the default int4 GPU backend
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

**Defaults (0.2.1).** Group size 32, act-order on, clip search on, the `size_b` precision preset (the last layer at int8, the rest int4), and calibration on the wikitext-2 train split, 128 samples of 1024 tokens. These defaults were chosen from measured runs in [docs/ACCURACY.md](docs/ACCURACY.md).

**Opt-outs and options.**

| Option | Effect |
| --- | --- |
| `--group-size {32,64,128}` | Quantization group size (default 32). `--group-size 128` is the 0.1.0 group size. Pair it with `--no-clip-search`, because clip search alone made the g128 model worse. |
| `--no-act-order` | Turn off act-order (on by default). Act-order writes `g_idx` tensors and needs format version 2. |
| `--no-clip-search` | Turn off the MSE clip search (on by default). |
| `--preset {down_proj,sensitive,size_a,size_b,size_c,none}` | Precision preset. Default is `size_b`. `sensitive` reproduces 0.2.0 (`down_proj` and the last layer at int8). `size_c` is the smallest candidate (+1.59% and +1.61% C4 perplexity against `sensitive`, so it was not made the default). `none` stores every quantized module at int4. `down_proj` stores only `down_proj` at int8. See [docs/ACCURACY.md](docs/ACCURACY.md). |
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

Options: `--max-new-tokens` (64), `--temperature` (0 means greedy), `--top-k`, `--top-p`, `--seed` (0), `--no-stream`, `--device {auto,cpu,cuda}` (auto), `--backend {auto,int4,cached,fused,marlin}` (auto), `--prefer-fused`, `--no-cuda-graph`.

**CUDA graph decode.** On a CUDA device, decoding replays a captured CUDA graph by default. `--no-cuda-graph` or the environment variable `QUF_NO_CUDA_GRAPH=1` turns it off and decodes eagerly. If graph capture fails, one INFO line gives the reason and decoding continues eagerly. Median of 7 runs, 128 new tokens, RTX 5080 Laptop, default `size_b` files, 0.3.0 measurement. `triton_int4`: 0.5B 46.7 tok/s eager and 284.2 with the graph, 1.5B 30.9 eager and 145.6 with the graph. `torch_gpu`: 0.5B 58.4 eager and 234.4 with the graph, 1.5B 40.6 eager and 99.7 with the graph.

Backends:

- `cpu`: used when no NVIDIA GPU is present.
- `triton_int4` (`--backend int4`): keeps packed int4 on the GPU and reads it with Triton kernels. **Default on GPU** when Triton imports and its self-test passes. Install it with `pip install "quf[fast]"`. It is faster than `torch_gpu` in the default CUDA graph mode (1.21x on 0.5B, 1.46x on 1.5B) and uses about half the GPU memory. Without the extra, with `QUF_NO_TRITON=1`, or when the self-test fails, `auto` uses `torch_gpu`. `--backend int4` reports an error with the reason when it is unavailable. Eager decode is slower with int4 than with `torch_gpu`, so `auto` picks `torch_gpu` when `--no-cuda-graph` or `QUF_NO_CUDA_GRAPH=1` is set, and a failed graph capture logs a hint to use `--backend cached`.
- `torch_gpu` (`--backend cached`): dequantizes weights to fp16 at load. Default on GPU when the int4 backend is unavailable and the weights fit in free GPU memory.
- `torch_fused` (`--backend fused`): keeps packed int4 on the GPU. Chosen automatically when the cached weights do not fit, or with `--prefer-fused`.
- `marlin`: optional, only with `--backend marlin`, never automatic. Unverified on this hardware.

The choice and its reason are logged to stderr. `--device cuda` without an NVIDIA GPU exits with status 1. A prompt that exceeds the context length and GPU out-of-memory also exit with status 1.

## Benchmark

```
quf bench out.quf --out results.json
quf bench out.quf --out results.json --fp16-ref Qwen/Qwen2.5-0.5B --gguf model-q4.gguf --device auto
```

The results JSON records WikiText perplexity, file size, tokens per second, backend, device name and software versions. With `--fp16-ref`, the relative perplexity increase over fp16 is printed next to the documented target of at most 3 percent. The target is reported, not enforced. `--gguf` adds a GGUF Q4 comparison. The GGUF rows in this README were verified against `llama-perplexity` (see [docs/RESULTS.md](docs/RESULTS.md)). An lm-evaluation-harness adapter is in `quf/bench/lm_eval_adapter.py`; its scores are in [docs/RESULTS.md](docs/RESULTS.md).

## Measured results

Machine: RTX 5080 Laptop GPU, native Windows, PyTorch 2.11.0+cu128. Perplexity uses 2048-token windows. WikiText-2 is the test split. C4 is the first ~1.3M characters of the English validation split, never used for calibration. Source: `docs/bench_accuracy.json`. Details: [docs/RESULTS.md](docs/RESULTS.md).

**A 4-bit model does not beat fp16.** Every quantized row has higher perplexity than fp16.

The two tables below use the earlier **full protocol** (every position of each 2048-token window scored), so their perplexities are not comparable with the Summary above. They show the 0.2.0 `sensitive` preset ("0.2.0 defaults"), not the current `size_b` default. The default was changed afterwards to cut file size; against `sensitive`, `size_b` is 7.8% smaller on 0.5B and 11.0% smaller on 1.5B, at +0.37% and +0.58% WikiText-2 perplexity (`llama` protocol, [docs/ACCURACY.md](docs/ACCURACY.md)).

Qwen/Qwen2.5-0.5B (fp16 perplexity: WikiText-2 13.0703, C4 19.7930):

| Run | WikiText-2 ppl | vs fp16 | C4 ppl | vs fp16 | File size (bytes) | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | 13.0703 | n/a | 19.7930 | n/a | 988,065,536 (computed) | 15.50 |
| RTN g128 | 15.5417 | +18.91% | 23.5073 | +18.77% | 469,483,072 | 22.92 |
| RTN g32 | 14.3826 | +10.04% | 21.6024 | +9.14% | 490,449,792 | 24.13 |
| 0.1.0 settings (GPTQ g128) | 14.0401 | +7.42% | 21.7476 | +9.87% | 469,483,072 | 26.66 |
| **0.2.0 defaults (g32 + clip + act-order + sensitive int8)** | **13.3775** | **+2.35%** | **20.4453** | **+3.30%** | 539,605,632 | 16.54 |

Qwen/Qwen2.5-1.5B (fp16 perplexity: WikiText-2 9.2650, C4 14.6605):

| Run | WikiText-2 ppl | vs fp16 | C4 ppl | vs fp16 | File size (bytes) | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | 9.2650 | n/a | 14.6605 | n/a | 3,087,428,608 (computed) | 13.40 |
| RTN g128 | 10.3555 | +11.77% | 16.1390 | +10.09% | 1,158,904,640 | 22.44 |
| RTN g32 | 9.8572 | +6.39% | 15.4676 | +5.51% | 1,235,674,048 | 22.28 |
| 0.1.0 settings (GPTQ g128) | 9.8434 | +6.24% | 15.6935 | +7.05% | 1,158,904,640 | 21.86 |
| **0.2.0 defaults** | **9.3585** | **+1.01%** | **14.8997** | **+1.63%** | 1,413,323,520 | 22.19 |

What this shows:

- Against the 3% target (full protocol), the 0.2.0 defaults are within it on WikiText-2 for both models and on C4 for 1.5B. On C4 for 0.5B they are at +3.30%, so the target is **missed** there.
- The calibration data is WikiText-2 train, so the WikiText-2 column is closer to the calibration domain than C4. The C4 column is the held-out check.
- Sizes: the 0.2.0 file is 539.6 MB for 0.5B (0.5461 of fp16) and 1,413.3 MB for 1.5B (0.4578 of fp16). The 0.1.0 file was 469.5 MB and 1,158.9 MB.
- The Tokens/s column in the tables above is from earlier single short runs on the pre-fast-path decoder (the fp16 row uses Hugging Face generate). Do not read an order into small differences. Current decode speed is in the paragraph below.
- GGUF Q4_K_M and lm-evaluation-harness results are in the Summary and in [docs/RESULTS.md](docs/RESULTS.md).

Decode speed of the 0.2.1 `torch_gpu` fast path (these are 0.2.1 figures, measured before the int4 backend; the 0.3.0 figures are in the Summary and the CUDA graph paragraph above; default `size_b` files, 128 new tokens, median of 7 runs after 2 warm-ups, RTX 5080 Laptop): Qwen2.5-0.5B 57.8 tokens/s eager and 255.2 with the CUDA graph (6.23x the earlier 40.9); Qwen2.5-1.5B 46.6 eager and 82.7 with the CUDA graph (2.70x the earlier 30.7, **below the 3x target**). CPU backend on 0.5B: 22.8. llama.cpp Q4_K_M on the same GPU ran 764 (0.5B) and 451 (1.5B) tokens/s in that earlier session (757 and 423 in the 0.3.0 session), so QUF is slower; it is also a different engine, so the numbers are not like for like. See [docs/BACKENDS.md](docs/BACKENDS.md). gigatoken tokenized WikiText-2 train in 0.23 s against 8.44 s for HF tokenizers (36.8x, identical ids).

## Documentation

- [docs/FORMAT.md](docs/FORMAT.md): byte layout, header fields, tensor naming, dtypes and the canonical Packed_Int4 order.
- [docs/ENV.md](docs/ENV.md): environment results for native Windows and the backend selection rules.
- [docs/RESULTS.md](docs/RESULTS.md): measured accuracy, size and speed.
- [docs/ACCURACY.md](docs/ACCURACY.md): accuracy diagnosis, upgrade evaluation and the size study behind the `size_b` default.
- [docs/BACKENDS.md](docs/BACKENDS.md): backend and decode speed measurements, including the CUDA graph path.

README.md, docs/ENV.md and docs/FORMAT.md ship with the package: they are in the source distribution and are installed to `share/doc/quf` by the wheel.

## Tests

```
pytest -m "not gpu and not convert"   # CPU suite, no NVIDIA GPU needed
pytest -m gpu                         # needs an NVIDIA GPU
pytest -m convert                     # needs the convert extra and downloads a model
```

Tests marked `gpu` are skipped automatically when no NVIDIA GPU is available.

## Links and license

Source and issues: https://github.com/vtrsharvin-sj/Quantized-Unified-Format

Released under the MIT license. See [LICENSE](LICENSE).
