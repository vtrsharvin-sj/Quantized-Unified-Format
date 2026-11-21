# Changelog
## 0.3.0
Int4 GPU backend release. The file format is unchanged (versions 1 and 2).
### Runtime speed and memory
- **New `triton_int4` backend, default on GPU with `pip install "quf[fast]"`.** It keeps packed int4 weights on the GPU and reads them with Triton kernels. Measured against `torch_gpu` (default `size_b` files, CUDA graph, 128 new tokens, median of 7 runs, RTX 5080 Laptop): Qwen2.5-0.5B 284 against 234 tok/s (1.21x), Qwen2.5-1.5B 146 against 100 tok/s (1.46x).
- GPU memory after load is 45 to 54 percent of `torch_gpu`: 508 against 947 MiB (0.5B) and 1320 against 2951 MiB (1.5B).
- Perplexity is identical to `torch_gpu` (full-window WikiText-2 and C4 match bit for bit), and greedy tokens are identical.
- **Eager-mode guard.** Eager decode is slower with int4 than with `torch_gpu` (46.7 against 58.4 tok/s on 0.5B, 30.9 against 40.6 on 1.5B), so `auto` picks `torch_gpu` when `--no-cuda-graph` or `QUF_NO_CUDA_GRAPH=1` is set. A failed graph capture logs a hint to use `--backend cached`.
- `quf run --backend int4` forces the Triton backend and reports an error with the reason when it is unavailable. The environment variable `QUF_NO_TRITON=1` disables it, and `auto` then uses `torch_gpu`. `auto` also falls back to `torch_gpu` when Triton is missing or its self-test fails.
### Logging
- Quieter per-module logs: the per-module routing summary is logged at DEBUG, and INFO shows a single "Selected backend" line with the reason.
### Honest gap to llama.cpp
- QUF is still 2.6 to 2.9x slower than llama.cpp Q4_K_M on the same GPU: 284 against 757 tok/s (0.5B) and 146 against 423 tok/s (1.5B). QUF files are also 1.25x and 1.28x larger than GGUF Q4_K_M. Different engines, so not a like-for-like comparison. Details in `docs/BACKENDS.md` and `docs/bench_accuracy.json`.
### Packaging
- New `fast` extra: Triton (`triton-windows` on Windows, `triton` on Linux).

## 0.2.1
Decode speed and default-size release. The file format is unchanged (versions 1 and 2).

### Runtime speed
- **CUDA graph decode is the default on CUDA.** Measured (default `size_b` files, 128 new tokens, median of 7 runs after 2 warm-ups, RTX 5080 Laptop): Qwen2.5-0.5B 255.2 tok/s with the graph against 57.8 eager and 40.9 before this release; Qwen2.5-1.5B 82.7 tok/s against 46.6 eager and 30.7 before. The 3x speed target is reached on 0.5B and not reached on 1.5B (memory-bound fp16 gemv).
- `--no-cuda-graph` (on `quf run`) and the environment variable `QUF_NO_CUDA_GRAPH=1` turn the graph off and decode eagerly.
- Static KV cache, so tensor shapes and addresses stay fixed for graph replay.
- Fused qkv and fused gate/up projections.
- CPU fp32 weight cache and a cached `lm_head`.
- CPU backend on 0.5B: 22.8 tok/s.

### Converter
- **New default preset `size_b`** (last layer at int8, `down_proj` at int4). Files are 7.8% smaller on 0.5B and 11.0% smaller on 1.5B than the 0.2.0 `sensitive` files, at +0.37% and +0.58% WikiText-2 perplexity. The `sensitive` preset is still available with `--preset sensitive`.

### Evaluation
- Batched lm-evaluation-harness adapter (`quf/bench/lm_eval_adapter.py`); a hellaswag plus arc_easy run takes about a minute per model.
- GGUF Q4_K_M comparison verified against `llama-perplexity` using the llama protocol (same text, same windows, GPU).
- Decode timing reports the median of repeated runs after warm-up instead of a single run.

### Packaging
- The repository was renamed to Quantized-Unified-Format; project URLs now point to it.
## 0.2.0

Accuracy upgrade release.

### Format (version 2)
- Group sizes 32, 64 and 128 (recorded in the header; a missing value is read as 128).
- `g_idx` tensors for act-order (desc_act) quantization.
- Per-module `int8` and `fp16` precision, with an `int8` tensor dtype.
- The Writer emits version 1 when a file uses only group size 128, no `g_idx` and no int8 tensors, so 0.1.0 readers still read those files. Version 2 files are rejected by 0.1.0 readers with the unsupported-version error. The Reader reads versions 1 and 2.

### Converter
- Native GPTQ act-order and MSE clip search.
- A "sensitive" preset that keeps the most sensitive modules at int8.
- New CLI defaults, chosen from measured WikiText-2 perplexity (see docs/ACCURACY.md).
- Multi-source calibration (default: wikitext-2 train plus English C4, 128 samples of length 2048). `--legacy-calibration` restores the 0.1.0 settings.
- Sensitivity diagnosis tooling and evaluation scripts (`scripts/eval_upgrades.py`, `scripts/bench_accuracy.py`).

### Backends
- The PyTorch and GPU backends support group sizes 32/64/128, `g_idx` and int8/fp16 modules.

### Results
- WikiText-2 perplexity versus fp16: +2.35% on Qwen2.5-0.5B and +1.01% on Qwen2.5-1.5B (0.1.0 on 0.5B was +7.4% to +8.5%).

### Packaging
- Added `.gitignore`. The sdist excludes `eval_cache/`, `dist/`, `*.quf` and scratch logs.

## 0.1.0

- Initial release: QUF version 1 file format (4-bit, group size 128), Writer and Reader with validation.
- `quf convert` with native GPTQ, `quf inspect`, `quf verify`, and `quf bench`.
- PyTorch CPU and CUDA backends, device detection, and Windows support.
