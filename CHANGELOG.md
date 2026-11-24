# Changelog
## 0.4.1
Documentation, Python API and packaging release. The file format and the quantization are unchanged: 0.4.1 changes no conversion output for Qwen2 models and reads the same format versions (1, 2 and 3), so the README results, measured with 0.4.0, still apply.
### Python API
- **New `quf.load(path, device="auto", backend="auto", cuda_graph=None)`**, which returns a `quf.Model`. `Model.generate(prompt, max_new_tokens=64, temperature=0.0, top_k=None, top_p=None, seed=0, stream=None)` returns the generated text, and an optional `stream` callback receives each new piece as it is decoded. The handle also has `backend` (the selected backend, for example `triton_int4`), `tokenizer`, `header`, `close()` and `with` support. It runs the same steps as `quf run` (model load, the tokenizer embedded in the file, the same generation loop), so it returns the text that `quf run` prints. `device`, `backend` and the `generate` arguments accept the values and limits of `quf run`; anything outside them raises `ValueError`. `import quf` still imports no torch, tokenizers or `quf.runtime`; `quf.load` and `quf.Model` resolve on first use.
### Conversion
- **Qwen3 is rejected at convert time.** `Qwen2ForCausalLM` is now the only supported architecture. 0.4.0 accepted `Qwen3ForCausalLM`, but the runtime does not apply Qwen3's q_norm and k_norm, so a converted Qwen3 file loaded and generated wrong text without an error. `quf convert` now stops after reading config.json, before it loads the tokenizer, the calibration data or the weights, with one line: `quf: error: Unsupported model architecture 'Qwen3ForCausalLM'. Supported: Qwen2ForCausalLM.` `UnsupportedArchitectureError` now also derives from `ValueError`.
### Runtime
- **Silenced the Triton `TRITON_DEFAULT_FP_FUSION` warning.** Triton 3.9 emits a one-time `UserWarning` when that environment variable is unset. `quf.runtime.kernels` now sets it to `1`, Triton's own default, before any Triton import, so the generated kernels do not change. A value that the user set is kept.
### Documentation
- **README rewritten** as the public landing page for GitHub and PyPI: results at a glance against Q4_K_M with every loss named, install, a quickstart with real output, the Python API, how it works, the full benchmark tables with the methodology and how to reproduce them, a CLI reference, supported models, limitations and troubleshooting. All links and images are absolute, so the page renders the same on PyPI. `tests/test_readme.py` checks the structure, the links, the badges and that every `--option` shown exists in the CLI.
- **Size/perplexity chart** `docs/assets/quf_vs_gguf.png` in the README, drawn from `docs/bench_accuracy.json` by `scripts/make_readme_chart.py`. The script needs matplotlib, which is not a quf dependency.
### Packaging and CI
- **Metadata polish:** a one-line description, the keywords quantization, llm, gptq, int4, 4-bit, triton, cuda, windows and qwen, the classifiers "Environment :: GPU :: NVIDIA CUDA", "Operating System :: POSIX :: Linux" (with "Operating System :: Microsoft :: Windows", replacing "Operating System :: OS Independent") and Python 3.13, and project URLs for the documentation and the changelog next to the homepage, repository and issues.
- **CI** runs the CPU test suite on Python 3.10, 3.11, 3.12 and 3.13 on ubuntu-latest and windows-latest (before: 3.10 and 3.12). The Python classifiers are tested against this matrix.
## 0.4.0
Fused decode, quantized embedding and new defaults release. The default `quf convert` output is now QUF format version 3, so 0.3.0 and older readers reject it. Versions 1 and 2 still open.
### Format and conversion
- **Quantized embedding and `lm_head` (format version 3).** The embedding table, and an untied `lm_head`, are stored per row as int8 (default) or int4 with fp16 group scales. New `quf convert` options: `--embed-precision {fp16,int8,int4}` (default `int8`), `--embed-group-size {32,64}` (default 64) and `--embed-method {clip,rtn}` (default `clip`). `clip` is a conversion option only: the header records `"method": "rtn"` plus `"clip_search": true` (the format allows only `rtn` and `gptq`, unchanged), and `quf info` shows "rtn + clip search". `--embed-precision fp16` writes a version 1 or 2 file as before. The int4 embedding was not chosen: on 1.5B it raised WikiText-2 perplexity to 8.93 against 8.45 for int8. Every backend keeps the table packed (no fp16 copy); `quf info` prints the embedding precision.
- **New default preset `size_f`.** `k_proj`, `v_proj` and the whole last layer at int8, everything else int4 (g32, act-order, clip search). It replaces `size_b`. Presets `size_d`, `size_e`, `size_f` and `size_g` are new; `--preset size_b` still reproduces the 0.3.0 body. Chosen in the size study in `docs/ACCURACY.md`: `k_proj` and `v_proj` are small under GQA, so int8 costs about 2 MB on 0.5B and removes about 1% of perplexity.
- **Clip-search fix (changes conversion output).** The clip search now ranks candidate clipping ratios by the error of the fp16-rounded dequantized weights, the same rounding `reference_dequantize` applies, instead of the unrounded values. Files converted with 0.4.0 differ from files converted with 0.3.0 using identical options. Existing files are not affected.
### Runtime speed
- **Fused decode kernels on `triton_int4`:** add+RMSNorm, RoPE plus KV-cache write, flash-decode attention, and on-device greedy argmax with a lazy EOS check (the end-of-sequence test runs every few tokens, not every token, and surplus tokens are trimmed). They run inside the CUDA graph. `QUF_NO_FUSED=1` disables them and keeps the 0.3.0 decode path.
- **Int8 GEMV and mixed int4/int8 fast path.** A Triton int8 per-output-channel GEMV for M = 1 (fp32 accumulation, bias and scatter epilogues, graph capturable). A fused qkv or gate/up group that mixes int4 and int8 members runs as separate kernel calls into one scattered output buffer, so the 0.4.0 default file keeps the fused RoPE+KV write and flash-decode on every layer instead of falling back to cuBLAS fp16 GEMV.
- **j-major int4 GEMV tuning.** Per-shape `(BLOCK_N, NGB, num_warps, num_stages, SPLIT)` configs, including split-K for the `down` projections, chosen by whole-graph decode time. In the interleaved lab run this is about 14% faster steady decode on 1.5B and about 2% on 0.5B (within noise).
- **Fused-path fallback fixes.** If any fused kernel raises on its first call or at graph capture, the whole decode step returns to the 0.3.0 path with one INFO line and the same greedy tokens. The quantized `lm_head` GEMV falls back on its own to chunked dequantization. Gate and fallback tests are in `tests/test_fused_fallback.py`.
### Tests
- Shared near-tie helper `tests/_greedy_compare.py`: loads a real file with and without the fused path, compares 4 prompts x 64 greedy tokens and the logits, and accepts a token difference only where the top-2 reference logits are closer than the near-tie gap. New tests cover the fused kernels, the int8 GEMV, mixed groups, format v3 and the quantized embedding.
### Measured results (RTX 5080 Laptop, native Windows, Qwen2.5, `llama` perplexity protocol)
Details: `docs/RESULTS.md` and `docs/bench_accuracy.json`. Steady-state decode tok/s, `triton_int4` with CUDA graph, 128 new tokens, median of 7 runs (min-max), all rows in one interleaved session **on battery power**, llama.cpp `llama-bench tg128`.

| Model | QUF 0.4.0 | Q4_K_M | Q4_K_S | IQ4_XS-imat | Q5_K_M |
|---|---|---|---|---|---|
| 0.5B | 662 (658-666) | 494 (459-510) | 496 (486-538) | 496 (489-523) | 506 (483-534) |
| 1.5B | 283 (263-285) | 274 (268-279) | 282 (270-287) | 274 (270-278) | 248 (242-251) |

| Model | Format | Size MB | WikiText-2 | C4 |
|---|---|---:|---:|---:|
| 0.5B | fp16 | 988.1 | 11.740 | 18.239 |
| 0.5B | QUF 0.4.0 | 367.9 | 11.987 (+2.10%) | 18.777 (+2.94%) |
| 0.5B | Q4_K_M | 397.8 | 12.043 | 18.724 |
| 1.5B | fp16 | 3087.4 | 8.296 | 13.476 |
| 1.5B | QUF 0.4.0 | 1040.0 | 8.420 (+1.49%) | 13.763 (+2.13%) |
| 1.5B | Q4_K_M | 986.0 | 8.600 | 13.954 |
| 3B | QUF 0.4.0 | 1987.4 | 7.291 | 12.154 |
| 3B | Q4_K_M | 1929.9 | 7.465 | 12.311 |

- Compared with the 0.3.0 `size_b` default in its own session: 0.5B 284 and 1.5B 146 tok/s (different session and clock state, so not a like-for-like ratio).
- **0.5B:** the file is 7.5% smaller than Q4_K_M, WikiText-2 is 0.46% better and decode is 1.34x faster (ranges do not overlap). **C4 is a loss:** 18.777 against 18.724 (+0.28%) for Q4_K_M, and +0.99% against the imatrix Q4_K_M. Beat_Target is reached only by the WikiText-2 definition.
- **1.5B is larger than Q4_K_M:** 1040.0 against 986.0 MB (+5.5%), also larger than Q4_K_S (+10.6%) and IQ4_XS (+16.1%). Perplexity is better than plain Q4_K_M on both sets (-2.09% and -1.37%) and speed ties with every variant except Q5_K_M (1.14x). Beat_Target is **not reached**. The 1.5B 398 tok/s speed target in the spec was also not reached (283).
- **3B is larger and slower** (one run, on AC power): +3.0% size and 190.7 against 236.0 tok/s end to end (229.1 steady-state, 3% lower), with better perplexity (-2.3% WikiText-2, -1.3% C4). Beat_Target is **not reached**.
- lm-evaluation-harness (hellaswag, arc_easy, 500 examples each): within one standard error of fp16 on every score for 0.5B and 1.5B.
- **Caveats.** The 0.5B and 1.5B speed numbers come from one battery-powered session and the 3B speed from one AC session. Different engines, so speed is not like for like. The same-session ranges overlap for most 1.5B comparisons, which is why they are called ties.
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
