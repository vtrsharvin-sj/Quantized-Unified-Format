# QUF: Quantized Unified Format

QUF is a single-file container (`.quf`) for a 4-bit quantized decoder-only LLM. One file holds the model config, tokenizer files, quantization scheme and all tensors. This package provides a reference writer and reader, a native GPTQ converter, a runtime with CPU and CUDA backends, and a benchmark suite.

QUF runs natively on Windows (and any platform with PyTorch). No WSL or Linux environment is needed.

Supported architectures: `Qwen2ForCausalLM` (first), with `Qwen3ForCausalLM` addable through the architecture allow-list. By default, weights are 4-bit with group size 32, act-order and clip search. The default preset is `size_f` (0.4.0): `k_proj`, `v_proj` and the whole last layer are stored at int8 and everything else at int4 (`down_proj` included). The embedding table, and an untied `lm_head`, are stored as per-row int8 with group size 64 (`--embed-precision int8`, QUF format version 3). `--preset size_b` is the 0.3.0 body (last layer int8 only) and `--preset sensitive` reproduces 0.2.0. `--embed-precision fp16` keeps the embedding and `lm_head` as fp16.

## Summary

Qwen2.5 on an RTX 5080 Laptop GPU, native Windows (the 0.5B and 1.5B speed numbers come from one battery-powered session, and the 3B speed numbers from one session on AC power), QUF 0.4.0 defaults against fp16 and five GGUF variants. Perplexity uses the `llama` protocol over the full WikiText-2 test split and a held-out C4 slice. Full method, per-axis verdicts and caveats: [docs/RESULTS.md](docs/RESULTS.md#quf-040-vs-gguf-task-362).

#### Qwen2.5-0.5B (llama protocol, full WikiText-2 test and the C4 slice)

| Format | Size MB | WikiText-2 PPL | C4 PPL | Decode tok/s (median, min-max) | Δ WikiText-2 vs fp16 | Δ C4 vs fp16 | Pareto front |
|---|---:|---:|---:|---:|---:|---:|:---:|
| fp16 (reference) | 988.1 | 11.740 | 18.239 | not measured | 0 | 0 | n/a |
| QUF 0.4.0 default | 367.9 | 11.987 | 18.777 | 662 (658-666) | +2.10% | +2.94% | yes |
| GGUF Q4_K_M (no imatrix) | 397.8 | 12.043 | 18.724 | 494 (459-510) | +2.58% | +2.66% | yes |
| GGUF Q4_K_M-imat | 397.8 | 11.959 | 18.593 | 494 (459-510) † | +1.87% | +1.94% | yes |
| GGUF Q4_K_S (no imatrix) | 385.5 | 12.207 | 19.002 | 496 (486-538) | +3.98% | +4.18% | no |
| GGUF IQ4_XS-imat | 349.4 | 12.286 | 19.046 | 496 (489-523) | +4.65% | +4.42% | yes |
| GGUF Q5_K_M (no imatrix) | 420.1 | 11.961 | 18.505 | 506 (483-534) | +1.88% | +1.46% | yes |

QUF against each GGUF variant, Qwen2.5-0.5B (size: smaller wins; perplexity: difference under 0.1% is a tie, otherwise lower wins; speed: overlapping min-max ranges are a tie):

| vs | Size | WikiText-2 | C4 | Speed |
|---|---|---|---|---|
| GGUF Q4_K_M (no imatrix) | QUF wins (-7.5%) | QUF wins (-0.46%) | QUF loses (+0.28%) | QUF wins, median ratio 1.340 |
| GGUF Q4_K_M-imat | QUF wins (-7.5%) | QUF loses (+0.23%) | QUF loses (+0.99%) | QUF wins, median ratio 1.340 † |
| GGUF Q4_K_S (no imatrix) | QUF wins (-4.6%) | QUF wins (-1.80%) | QUF wins (-1.18%) | QUF wins, median ratio 1.334 |
| GGUF IQ4_XS-imat | QUF loses (+5.3%) | QUF wins (-2.43%) | QUF wins (-1.41%) | QUF wins, median ratio 1.334 |
| GGUF Q5_K_M (no imatrix) | QUF wins (-12.4%) | QUF loses (+0.22%) | QUF loses (+1.47%) | QUF wins, median ratio 1.308 |

Beat_Target (size ≤ Q4_K_M, WikiText-2 ≤ Q4_K_M, decode tok/s ≥ Q4_K_M): **reached** on 0.5B (size ok, WikiText-2 ok, speed ok).

Pareto front (size, WikiText-2, C4, lower is better on all): QUF 0.4.0 default, GGUF Q4_K_M (no imatrix), GGUF Q4_K_M-imat, GGUF IQ4_XS-imat, GGUF Q5_K_M (no imatrix). Dominated: GGUF Q4_K_S (no imatrix) (by QUF 0.4.0 default).

#### Qwen2.5-1.5B (llama protocol, full WikiText-2 test and the C4 slice)

| Format | Size MB | WikiText-2 PPL | C4 PPL | Decode tok/s (median, min-max) | Δ WikiText-2 vs fp16 | Δ C4 vs fp16 | Pareto front |
|---|---:|---:|---:|---:|---:|---:|:---:|
| fp16 (reference) | 3087.4 | 8.296 | 13.476 | not measured | 0 | 0 | n/a |
| QUF 0.4.0 default | 1040.0 | 8.420 | 13.763 | 283 (263-285) | +1.49% | +2.13% | yes |
| GGUF Q4_K_M (no imatrix) | 986.0 | 8.600 | 13.954 | 274 (268-279) | +3.66% | +3.55% | no |
| GGUF Q4_K_M-imat | 986.0 | 8.497 | 13.757 | 274 (268-279) † | +2.42% | +2.09% | yes |
| GGUF Q4_K_S (no imatrix) | 940.3 | 8.679 | 14.075 | 282 (270-287) | +4.61% | +4.45% | no |
| GGUF IQ4_XS-imat | 895.7 | 8.577 | 13.879 | 274 (270-278) | +3.38% | +3.00% | yes |
| GGUF Q5_K_M (no imatrix) | 1125.1 | 8.389 | 13.633 | 248 (242-251) | +1.12% | +1.17% | yes |

QUF against each GGUF variant, Qwen2.5-1.5B (size: smaller wins; perplexity: difference under 0.1% is a tie, otherwise lower wins; speed: overlapping min-max ranges are a tie):

| vs | Size | WikiText-2 | C4 | Speed |
|---|---|---|---|---|
| GGUF Q4_K_M (no imatrix) | QUF loses (+5.5%) | QUF wins (-2.09%) | QUF wins (-1.37%) | tie (ranges overlap), median ratio 1.033 |
| GGUF Q4_K_M-imat | QUF loses (+5.5%) | QUF wins (-0.91%) | tie (+0.04%) | tie (ranges overlap), median ratio 1.033 † |
| GGUF Q4_K_S (no imatrix) | QUF loses (+10.6%) | QUF wins (-2.99%) | QUF wins (-2.22%) | tie (ranges overlap), median ratio 1.003 |
| GGUF IQ4_XS-imat | QUF loses (+16.1%) | QUF wins (-1.84%) | QUF wins (-0.84%) | tie (ranges overlap), median ratio 1.034 |
| GGUF Q5_K_M (no imatrix) | QUF wins (-7.6%) | QUF loses (+0.37%) | QUF loses (+0.96%) | QUF wins, median ratio 1.142 |

Beat_Target (size ≤ Q4_K_M, WikiText-2 ≤ Q4_K_M, decode tok/s ≥ Q4_K_M): **not reached** on 1.5B (size FAILS, WikiText-2 ok, speed ok).

Pareto front (size, WikiText-2, C4, lower is better on all): QUF 0.4.0 default, GGUF Q4_K_M-imat, GGUF IQ4_XS-imat, GGUF Q5_K_M (no imatrix). Dominated: GGUF Q4_K_M (no imatrix) (by GGUF IQ4_XS-imat), GGUF Q4_K_S (no imatrix) (by GGUF IQ4_XS-imat).

**Where QUF wins and loses (Qwen2.5-0.5B).** Size: smaller than every GGUF variant except IQ4_XS (349.4 MB, 5.3% smaller than QUF). WikiText-2: better than Q4_K_M, Q4_K_S and IQ4_XS, worse than Q4_K_M-imat (+0.23%) and Q5_K_M (+0.22%). C4: better than Q4_K_S and IQ4_XS, worse than Q4_K_M (+0.28%), Q4_K_M-imat (+0.99%) and Q5_K_M (+1.47%). Speed: faster than every variant (ranges do not overlap, median 1.31 to 1.34x). Beat_Target is reached on 0.5B, but only on the WikiText-2 definition: QUF's C4 perplexity is 0.28% worse than Q4_K_M's.

**Where QUF wins and loses (Qwen2.5-1.5B).** Size: larger than Q4_K_M, Q4_K_M-imat, Q4_K_S and IQ4_XS (5.5% to 16.1%), smaller than Q5_K_M. WikiText-2: better than every variant except Q5_K_M (+0.37%). C4: better than Q4_K_M, Q4_K_S, IQ4_XS, tie with Q4_K_M-imat (+0.04%), worse than Q5_K_M (+0.96%). Speed: tie with every variant except Q5_K_M, which QUF beats (1.14x). Beat_Target is not reached on 1.5B because the QUF file is 5.5% larger than Q4_K_M.

**Qwen2.5-3B scaling check (one model, one run per row).** QUF 0.4.0 defaults against GGUF Q4_K_M (no imatrix), llama protocol, full WikiText-2 test split and the C4 slice. This session ran on **AC power**, unlike the 0.5B and 1.5B speed session, so its speed numbers are not comparable with the tables above.

| Format | Size MB | WikiText-2 PPL | C4 PPL | Decode tok/s (median, min-max) |
|---|---:|---:|---:|---|
| GGUF Q4_K_M (no imatrix) | 1929.9 | 7.465 (±0.047) | 12.311 | 236.0 (225.2-244.4), llama-bench tg128 |
| QUF 0.4.0 default | 1987.4 | 7.291 | 12.154 | 190.7 end-to-end (180.5-198.5); 229.1 steady-state |

Per axis against Q4_K_M: size QUF loses (+3.0%); WikiText-2 QUF wins (-2.3%); C4 QUF wins (-1.3%); speed QUF loses (the ranges do not overlap; the end-to-end median is 19% lower and the steady-state median 3% lower). Beat_Target is **not reached** on 3B: the QUF file is larger and decode is slower. Both rows drifted down over the session (probably thermal), so the absolute speeds are indicative only.

**Other caveats.** On 0.5B the only variant QUF beats on size, both perplexities and speed at once is Q4_K_S. On 1.5B no GGUF variant is beaten on all four axes: IQ4_XS and Q4_K_M are smaller and tie on speed. Every tie and win above rests on one speed session and one perplexity run per row (no seeds are involved, but the speed was measured on battery). The QUF file stores `k_proj`, `v_proj` and the last layer at int8 and the embedding at int8, and uses group size 32, which accounts for part of its size against Q4_K_M and IQ4_XS. The 0.5B and 1.5B speed numbers come from one battery-powered session and the 3B speed numbers from one AC session; none was repeated in another session.

- † Q4_K_M-imat speed was not measured; it is given Q4_K_M's speed because the tensor types are the same.
- tok/s is from one interleaved session on battery power with throttled clocks; a tie means the min-max ranges overlap. The perplexity tie band is 0.1%.
- Earlier 0.3.0 numbers are in git history and in the 0.3.0 sections of [docs/RESULTS.md](docs/RESULTS.md).
- **lm-evaluation-harness** (hellaswag and arc_easy, 0-shot, 500 examples each): the 0.4.0 default files are within one standard error of fp16 on every score (largest gap: 1.5B hellaswag acc, 0.470 against 0.486). The older 0.2.1 `size_b` files were within one standard error except 0.5B arc_easy acc (0.612 against 0.636, 1.1 standard errors). The 0.2.0 `sensitive` files were within one standard error everywhere. Table below.

| Model | Variant | hellaswag acc | hellaswag acc_norm | arc_easy acc | arc_easy acc_norm |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | fp16 | 0.410 | 0.502 | 0.636 | 0.586 |
| Qwen2.5-0.5B | QUF 0.4.0 default | 0.416 | 0.504 | 0.636 | 0.578 |
| Qwen2.5-0.5B | QUF 0.2.1 `size_b` | 0.410 | 0.500 | 0.612 | 0.588 |
| Qwen2.5-0.5B | QUF 0.2.0 `sensitive` | 0.424 | 0.496 | 0.620 | 0.564 |
| Qwen2.5-1.5B | fp16 | 0.486 | 0.602 | 0.750 | 0.710 |
| Qwen2.5-1.5B | QUF 0.4.0 default | 0.470 | 0.588 | 0.740 | 0.730 |
| Qwen2.5-1.5B | QUF 0.2.1 `size_b` | 0.480 | 0.590 | 0.744 | 0.704 |
| Qwen2.5-1.5B | QUF 0.2.0 `sensitive` | 0.484 | 0.608 | 0.748 | 0.724 |

(stderr is about 0.022 for hellaswag and 0.019 to 0.022 for arc_easy.)
- Not evaluated: int6 and int5 storage. A `torch.compile` decode path was tried and skipped (see [docs/BACKENDS.md](docs/BACKENDS.md)).

Sources: `docs/bench_accuracy.json`, `docs/size_study.json`, [docs/ACCURACY.md](docs/ACCURACY.md), [docs/BACKENDS.md](docs/BACKENDS.md). More tables in [Measured results](#measured-results).

**Format version 3.** The 0.4.0 defaults write QUF format version 3 files, because the embedding table is quantized. Readers of 0.3.0 and older reject them with an unsupported version error. Version 1 and 2 files still open, and `--embed-precision fp16` writes a version 1 or 2 file again. See [docs/FORMAT.md](docs/FORMAT.md).

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
| `triton_int4` backend | Verified: default on GPU with `pip install "quf[fast]"`. 0.4.0 adds the fused decode kernels, an int8 GEMV and a mixed int4/int8 fast path: 662 tok/s (0.5B) and 283 (1.5B) steady-state, one battery session (see Summary). Older 0.3.0 figures: 1.21x (0.5B) and 1.46x (1.5B) faster than `torch_gpu` in graph mode, about half the GPU memory, same perplexity. Slower in eager mode, so `auto` avoids it with `--no-cuda-graph`. |
| Fused decode kernels (0.4.0) | Verified: add+RMSNorm, RoPE+KV-cache write, flash-decode and on-device greedy argmax with a lazy EOS check, on `triton_int4`. `QUF_NO_FUSED=1` turns them off and keeps the 0.3.0 decode path. |
| Quantized embedding and `lm_head` (format v3) | Verified: int8 per-row, group size 64 by default (`--embed-precision`). Files with the 0.4.0 defaults: 367.9 MB (0.5B), 1040.0 MB (1.5B), 1987.4 MB (3B). |
| CUDA graph decode | Verified: default on GPU. 0.4.0 default files, `triton_int4`, graph, steady-state, one battery session: 662 tok/s on 0.5B and 283 on 1.5B, against llama.cpp Q4_K_M at 494 and 274 in the same session (faster on 0.5B, tie within noise on 1.5B). Older 0.3.0 figures (`size_b` files, a different session where llama.cpp ran 756.5 and 422.8): 284.2 and 145.6 tok/s with the int4 backend, 46.7 and 30.9 eager, and 234.4 and 99.7 with `torch_gpu` (58.4 and 40.6 eager). The 0.2.1 `torch_gpu` figures (255.2 and 82.7 graph) are in Measured results. |
| Perplexity target (3 percent) | `llama` protocol, 0.4.0 default (`size_f`, int8 embedding): reached on WikiText-2 for 0.5B (+2.10%) and 1.5B (+1.49%) and on C4 for 0.5B (+2.94%) and 1.5B (+2.13%). The 0.3.0 `size_b` default missed it on C4 for 0.5B (+4.43%), and the 0.1.0 settings missed it. |
| GGUF comparison | Verified on 0.5B and 1.5B against five GGUF variants and on 3B against Q4_K_M: perplexity from `llama-perplexity` on the GPU, same text and protocol (see Summary). Beat_Target is reached on 0.5B, not reached on 1.5B (file 5.5% larger) or 3B (file 3.0% larger and decode slower). |
| lm-evaluation-harness scores | Verified on the 0.4.0 default files: within one standard error of fp16 on hellaswag and arc_easy for both models. Older `size_b` and `sensitive` rows are kept for comparison. |
| `torch.compile` decode | Skipped. It needs Triton, and with Triton it measured about 11% slower than the CUDA graph. |
| int6 / int5 storage | Not evaluated. |
| Qwen2.5-3B | Converted and evaluated once, on AC power (see Summary): better perplexity than Q4_K_M, larger file, slower decode. |
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

**Defaults (0.4.0).** Group size 32, act-order on, clip search on, the `size_f` precision preset (`k_proj`, `v_proj` and the whole last layer at int8, the rest int4), an int8 embedding table and `lm_head` with group size 64 (`--embed-precision int8 --embed-group-size 64 --embed-method clip`), and calibration on the wikitext-2 train split, 128 samples of 1024 tokens. These defaults were chosen from measured runs in [docs/ACCURACY.md](docs/ACCURACY.md).

**Opt-outs and options.**

| Option | Effect |
| --- | --- |
| `--group-size {32,64,128}` | Quantization group size (default 32). `--group-size 128` is the 0.1.0 group size. Pair it with `--no-clip-search`, because clip search alone made the g128 model worse. |
| `--no-act-order` | Turn off act-order (on by default). Act-order writes `g_idx` tensors and needs format version 2. |
| `--no-clip-search` | Turn off the MSE clip search (on by default). |
| `--preset {down_proj,sensitive,size_a,size_b,size_c,size_d,size_e,size_f,size_g,none}` | Precision preset. Default is `size_f`: `k_proj`, `v_proj` and the last layer at int8, the rest int4. `size_b` is the 0.3.0 default (last layer int8 only). `size_d` to `size_g` add more int8 attention or `down_proj` modules. `sensitive` reproduces 0.2.0 (`down_proj` and the last layer at int8). `size_c` is the smallest candidate (+1.59% and +1.61% C4 perplexity against `sensitive`, so it was not made the default). `none` stores every quantized module at int4. `down_proj` stores only `down_proj` at int8. See [docs/ACCURACY.md](docs/ACCURACY.md). |
| `--embed-precision {fp16,int8,int4}` | Storage of the embedding table and an untied `lm_head` (per-row, with group scales). Default `int8`. `fp16` keeps them unquantized and writes a version 1 or 2 file. The size study chose int8 over int4 because an int4 embedding cost accuracy on 1.5B (WikiText-2 8.93 against 8.45, see [docs/ACCURACY.md](docs/ACCURACY.md)). |
| `--embed-group-size {32,64}` | Group size of a quantized embedding (default 64). |
| `--embed-method {clip,rtn}` | Rounding of a quantized embedding: `clip` (MSE-optimal clipping per group, default) or `rtn`. The header has no `clip` method: the format only allows `rtn` and `gptq` for an embedding, so `--embed-method clip` is written as `"method": "rtn"` plus `"clip_search": true`, and `quf info` shows it as "rtn + clip search". The stored data is plain per-group round-to-nearest with a clip-searched scale. |
| `--mixed PATTERN=PRECISION`, `--int8-modules PATTERN` | Per-module precision (int4, int8, fp16) by glob. When given, the default preset is not applied. |
| `--legacy-calibration` | 0.1.0 calibration: wikitext-2 only, 128 samples of 512 tokens. |
| `--multi-source-calibration` | Calibrate on wikitext-2 train and English C4 (50/50), 128 samples of 2048 tokens. It is not the default: it measured worse on WikiText-2 perplexity and took longer to convert (see RESULTS.md). It may rank differently on other evaluations. |
| `--dataset`, `--samples` (128), `--seq-len` (1024), `--seed` (0) | Calibration source and size. |
| `--overwrite` | Replace an existing output file. |

To reproduce the 0.1.0 output (a version 1 file), use `quf convert MODEL out.quf --group-size 128 --no-act-order --no-clip-search --preset none --embed-precision fp16 --legacy-calibration`.

The command prints the output size and compression ratio. It refuses to replace an existing output unless `--overwrite` is given, rejects unsupported architectures before heavy work, and deletes a partial file if conversion fails.

**Format versions.** A quantized embedding (the 0.4.0 default) needs format version 3. Otherwise group size 32 or 64, act-order (`g_idx`) and int8 or fp16 modules need version 2. The writer picks the version from the features a file uses, so the opt-out command above still produces a version 1 file. Older readers reject a higher version with an unsupported version error. This is intended. Conversion output also changed in 0.4.0: the clip search now ranks candidates by the error of the fp16-rounded weights, so a file converted with 0.4.0 differs from one converted with 0.3.0 using the same options.

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

**Fused decode kernels (0.4.0).** On `triton_int4`, decode uses fused add+RMSNorm, RoPE+KV-cache write, flash-decode and on-device greedy argmax (the end-of-sequence check is lazy). `QUF_NO_FUSED=1` turns them off and decodes with the 0.3.0 path; the output tokens are the same except at near-ties in the fp16 logits.

**CUDA graph decode.** On a CUDA device, decoding replays a captured CUDA graph by default. `--no-cuda-graph` or the environment variable `QUF_NO_CUDA_GRAPH=1` turns it off and decodes eagerly. If graph capture fails, one INFO line gives the reason and decoding continues eagerly. 0.4.0 default files, `triton_int4`, steady-state, median of 7 runs, 128 new tokens, RTX 5080 Laptop, one battery session: 662 tok/s on 0.5B and 283 on 1.5B (3B, on AC: 229, steady-state). Older 0.3.0 measurement (median of 7 runs, 128 new tokens, default `size_b` files): `triton_int4`: 0.5B 46.7 tok/s eager and 284.2 with the graph, 1.5B 30.9 eager and 145.6 with the graph. `torch_gpu`: 0.5B 58.4 eager and 234.4 with the graph, 1.5B 40.6 eager and 99.7 with the graph.

Backends:

- `cpu`: used when no NVIDIA GPU is present.
- `triton_int4` (`--backend int4`): keeps packed int4 on the GPU and reads it with Triton kernels. **Default on GPU** when Triton imports and its self-test passes. Install it with `pip install "quf[fast]"`. It is faster than `torch_gpu` in the default CUDA graph mode (0.3.0 measurement: 1.21x on 0.5B, 1.46x on 1.5B) and uses about half the GPU memory. In 0.4.0 it also holds the quantized embedding packed and runs the int8 modules (`k_proj`, `v_proj`, last layer) on an int8 GEMV. Without the extra, with `QUF_NO_TRITON=1`, or when the self-test fails, `auto` uses `torch_gpu`. `--backend int4` reports an error with the reason when it is unavailable. Eager decode is slower with int4 than with `torch_gpu`, so `auto` picks `torch_gpu` when `--no-cuda-graph` or `QUF_NO_CUDA_GRAPH=1` is set, and a failed graph capture logs a hint to use `--backend cached`.
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

Machine: RTX 5080 Laptop GPU, native Windows, PyTorch 2.11.0+cu128. Perplexity uses 2048-token windows. WikiText-2 is the test split. C4 is the first ~1.3M characters of the English validation split, never used for calibration. Source: `docs/bench_accuracy.json`. Details: [docs/RESULTS.md](docs/RESULTS.md). The sections below this one are older releases unless they say 0.4.0.

**A 4-bit model does not beat fp16.** Every quantized row has higher perplexity than fp16.

The two tables below use the earlier **full protocol** (every position of each 2048-token window scored), so their perplexities are not comparable with the Summary above. They show the 0.2.0 `sensitive` preset ("0.2.0 defaults"), an older release, not the current `size_f` default. The default was changed afterwards to cut file size; against `sensitive`, `size_b` (0.3.0) is 7.8% smaller on 0.5B and 11.0% smaller on 1.5B, at +0.37% and +0.58% WikiText-2 perplexity (`llama` protocol, [docs/ACCURACY.md](docs/ACCURACY.md)). The 0.4.0 numbers are in the Summary.

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

Decode speed of the 0.2.1 `torch_gpu` fast path (these are old 0.2.1 figures, measured before the int4 backend; the 0.4.0 figures are in the Summary and the status table; default `size_b` files, 128 new tokens, median of 7 runs after 2 warm-ups, RTX 5080 Laptop): Qwen2.5-0.5B 57.8 tokens/s eager and 255.2 with the CUDA graph (6.23x the earlier 40.9); Qwen2.5-1.5B 46.6 eager and 82.7 with the CUDA graph (2.70x the earlier 30.7, **below the 3x target**). CPU backend on 0.5B: 22.8. llama.cpp Q4_K_M on the same GPU ran 764 (0.5B) and 451 (1.5B) tokens/s in that earlier session (757 and 423 in the 0.3.0 session), so QUF 0.2.1 was slower; it is also a different engine, so the numbers are not like for like. With 0.4.0 the 0.5B default file is faster than Q4_K_M in the Summary session and the 1.5B file ties. See [docs/BACKENDS.md](docs/BACKENDS.md). gigatoken tokenized WikiText-2 train in 0.23 s against 8.44 s for HF tokenizers (36.8x, identical ids).

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
