# Measured results (Qwen/Qwen2.5-0.5B, native Windows)

All numbers come from real runs on this machine. Raw data: `bench_qwen2.5-0.5b.json`, `bench_qwen2.5-0.5b-128x1024.json`, `real_model_results_qwen2.5-0.5b*.json`, `bench_backends.json` (all in this directory). Produced by `scripts/verify_real_model.py` and `scripts/bench_backends.py`.

## Current summary: QUF 0.2.1 defaults (`size_b`) against fp16 and GGUF Q4_K_M

Sources: `bench_accuracy.json` (`models_llama`, `decode_speed`), `size_study.json`, [ACCURACY.md](ACCURACY.md), [BACKENDS.md](BACKENDS.md). RTX 5080 Laptop GPU, native Windows, PyTorch 2.11.0+cu128. Protocol `llama`: positions 1024 to 2046 of each 2048-token window, over the full WikiText-2 test split and a held-out C4 slice (English validation, first ~1.3M characters, never used for calibration). GGUF perplexity is from `llama-perplexity` on the GPU. The default preset is `size_b` (last layer int8, the rest int4, group size 32, act-order, clip search). `--preset sensitive` reproduces 0.2.0.

Qwen/Qwen2.5-0.5B:

| | WikiText-2 ppl | vs fp16 | C4 ppl | vs fp16 | File size (bytes) | Ratio to fp16 file | GPU decode tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | 11.7398 | n/a | 18.2394 | n/a | 988,065,536 | 1.0000 | 45.8 (HF generate) |
| QUF 0.2.0 (`sensitive`) | 12.0326 | +2.49% | 18.8441 | +3.32% | 539,605,632 | 0.5461 | 50.2 (eager, 7 repeats, 32 tokens) |
| **QUF 0.2.1 (`size_b`, default)** | **12.0769** | **+2.87%** | **19.0473** | **+4.43%** | 497,728,832 | 0.5037 | 255.2 (CUDA graph) |
| GGUF Q4_K_M | 12.0425 | +2.58% | 18.7241 | +2.66% | 397,807,584 | 0.4026 | 764.2 (llama.cpp) |

Qwen/Qwen2.5-1.5B:

| | WikiText-2 ppl | vs fp16 | C4 ppl | vs fp16 | File size (bytes) | Ratio to fp16 file | GPU decode tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | 8.2964 | n/a | 13.4756 | n/a | 3,087,428,608 | 1.0000 | 30.2 (HF generate) |
| QUF 0.2.0 (`sensitive`) | 8.3954 | +1.19% | 13.6774 | +1.50% | 1,413,323,520 | 0.4578 | 39.9 (eager, 7 repeats, 32 tokens) |
| **QUF 0.2.1 (`size_b`, default)** | **8.4442** | **+1.78%** | **13.8663** | **+2.90%** | 1,257,449,536 | 0.4073 | 82.7 (CUDA graph) |
| GGUF Q4_K_M | 8.5998 | +3.66% | 13.9536 | +3.55% | 986,048,160 | 0.3194 | 450.8 (llama.cpp) |

Reading the tables:

- **tok/s is not like for like.** It comes from three engines: Hugging Face generate (fp16), the QUF runtime, and llama.cpp. The 0.2.1 QUF rows and the llama.cpp rows are the median of 7 runs of 128 new tokens ([BACKENDS.md](BACKENDS.md)). The fp16 and 0.2.0 rows are from earlier runs with 32 new tokens.
- **QUF is slower than llama.cpp**: 255.2 against 764.2 tok/s on 0.5B and 82.7 against 450.8 tok/s on 1.5B, with the CUDA graph path (the default on GPU). QUF has no 4-bit kernels, and on 1.5B decode is GPU-bound on reading cached fp16 weights.
- **QUF files are larger than Q4_K_M**: 1.25x on 0.5B and 1.28x on 1.5B (`size_b` bytes over GGUF bytes).
- **Against GGUF Q4_K_M**: QUF `size_b` has lower perplexity on 1.5B (both texts) and higher on 0.5B (both texts). The 0.5B Q4_K_M file is not pure K-quant (see the fallback note near the end), so that comparison is not clean.
- **3% target against fp16** for `size_b`: reached on WikiText-2 for both models and on C4 for 1.5B (+2.90%). **Missed on C4 for 0.5B (+4.43%).**
- `sensitive` to `size_b` (size study, [ACCURACY.md](ACCURACY.md)): 7.8% smaller on 0.5B and 11.0% smaller on 1.5B, at +0.37% and +0.58% WikiText-2 perplexity and +1.08% and +1.38% C4 perplexity. Each is a single conversion, so differences of a few hundredths of a percent are within calibration noise.
- **lm-evaluation-harness**: every QUF score is within one standard error of fp16 (section below). The `size_b` rows are from the 0.2.1 default files; the earlier `sensitive` rows are from 0.2.0.
- **Not evaluated**: int6 and int5 storage. **Skipped**: `torch.compile` for decode (see [BACKENDS.md](BACKENDS.md)).

Decode path controls: the CUDA graph is the default on CUDA. `quf run --no-cuda-graph` or the environment variable `QUF_NO_CUDA_GRAPH=1` decodes eagerly. Eager decode with the same `size_b` files measured 57.8 tok/s on 0.5B and 46.6 tok/s on 1.5B.

## Int4 backend (triton_int4) and the default decision (task 31.2)

Source: `bench_accuracy.json`, key `int4_backend`. Details: [BACKENDS.md](BACKENDS.md). `size_b` files, 128 new tokens, median of 7 runs, CUDA graph on, RTX 5080 Laptop, `triton-windows`.

| Model | Backend | tok/s | GPU memory after load (MiB) | WikiText-2 ppl | C4 ppl |
| --- | --- | --- | --- | --- | --- |
| 0.5B | torch_gpu | 234.4 | 947 | 12.0769 | 19.0473 |
| 0.5B | triton_int4 | 284.2 (1.21x) | 508 | 12.0769 | 19.0473 |
| 1.5B | torch_gpu | 99.7 | 2951 | 8.4442 | 13.8663 |
| 1.5B | triton_int4 | 145.6 (1.46x) | 1320 | 8.4442 | 13.8663 |
| 0.5B | llama.cpp Q4_K_M | 756.5 | n/a | 12.0425 | 18.7241 |
| 1.5B | llama.cpp Q4_K_M | 422.8 | n/a | 8.5998 | 13.9536 |

- Decision: triton_int4 is the default on CUDA. It is faster in graph mode, uses 45 to 54 percent of the memory, and gives the same perplexity (a token-by-token check of the kernels differs by at most 0.024 percent) and the same greedy tokens.
- Speed target (1.5x of 255 and 83 tok/s): met on 1.5B (1.75x), missed on 0.5B (1.11x).
- QUF remains 2.6 to 2.9x slower than llama.cpp. llama.cpp is a different engine, so this is a guide only.
- Eager decode with triton_int4 is slower than torch_gpu (0.80x and 0.76x). `auto` therefore picks torch_gpu when the CUDA graph is disabled.

## Earlier summary: 0.2.0 defaults against fp16, RTN and 0.1.0 (full protocol)

Source: `bench_accuracy.json` (raw data), run on the RTX 5080 Laptop GPU with PyTorch 2.11.0+cu128. Perplexity uses 2048-token windows on the WikiText-2 test split and on a held-out C4 sample (English validation, first ~1.3M characters, never used for calibration). Every position of each window is scored (the `full` protocol), so these numbers are not comparable with the `llama` protocol above. The 3% target is reported against fp16. The "New defaults" rows are the 0.2.0 `sensitive` preset.

**A 4-bit model does not beat fp16.** Every quantized row has higher perplexity than fp16. The 0.2.0 defaults (group size 32, clip search, act-order, `sensitive` int8 preset) narrow the gap: on WikiText-2, +2.35% for 0.5B and +1.01% for 1.5B, against +7.42% and +6.24% for the 0.1.0 settings.

Qwen/Qwen2.5-0.5B:

| Run | WikiText-2 ppl | vs fp16 | C4 ppl | vs fp16 | File size (bytes) | Ratio to fp16 file | Tokens/s | Convert (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | 13.0703 | n/a | 19.7930 | n/a | 988,065,536 (computed) | 1.0000 | 15.50 | n/a |
| RTN g128 | 15.5417 | +18.91% | 23.5073 | +18.77% | 469,483,072 | 0.4752 | 22.92 | 13 |
| RTN g32 | 14.3826 | +10.04% | 21.6024 | +9.14% | 490,449,792 | 0.4964 | 24.13 | 11 |
| 0.1.0 settings (GPTQ g128) | 14.0401 | +7.42% | 21.7476 | +9.87% | 469,483,072 | 0.4752 | 26.66 | 113 |
| **New defaults** | **13.3775** | **+2.35%** | **20.4453** | **+3.30%** | 539,605,632 | 0.5461 | 16.54 | 150 |

Qwen/Qwen2.5-1.5B:

| Run | WikiText-2 ppl | vs fp16 | C4 ppl | vs fp16 | File size (bytes) | Ratio to fp16 file | Tokens/s | Convert (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | 9.2650 | n/a | 14.6605 | n/a | 3,087,428,608 (computed) | 1.0000 | 13.40 | n/a |
| RTN g128 | 10.3555 | +11.77% | 16.1390 | +10.09% | 1,158,904,640 | 0.3754 | 22.44 | 23 |
| RTN g32 | 9.8572 | +6.39% | 15.4676 | +5.51% | 1,235,674,048 | 0.4002 | 22.28 | 24 |
| 0.1.0 settings (GPTQ g128) | 9.8434 | +6.24% | 15.6935 | +7.05% | 1,158,904,640 | 0.3754 | 21.86 | 279 |
| **New defaults** | **9.3585** | **+1.01%** | **14.8997** | **+1.63%** | 1,413,323,520 | 0.4578 | 22.19 | 263 |

Reading the tables:

- 3% target: reached on WikiText-2 for both models and on held-out C4 for 1.5B (+1.63%). **Missed on held-out C4 for 0.5B (+3.30%).**
- The calibration set is WikiText-2 train, so the WikiText-2 column is closer to the calibration domain than the C4 column.
- The new defaults cost size: 0.5B grows from 469.5 MB to 539.6 MB (+14.9%), 1.5B from 1,158.9 MB to 1,413.3 MB (+22.0%). They are still about 1.83x (0.5B) and 2.18x (1.5B) smaller than the fp16 checkpoint.
- The 1.5B fp16 and RTN rows show the same direction as 0.5B. GPTQ with the 0.1.0 settings roughly halves the RTN g128 gap, and the new defaults cut it further.
- Tokens per second are single short decode runs with large run-to-run variation. The fp16 row uses Hugging Face generate, the others use the `torch_gpu` backend. The lower figure for the new 0.5B defaults (16.54) was not investigated, so it may be machine noise. Do not read an order into small differences.
- The `bench_accuracy.json` 0.5B convert times are from separate runs and differ from the 143 s to 259 s in the upgrade table below.

GGUF rows are in the current summary above and in the `llama`-protocol sections below. The `full`-protocol GGUF entries in `bench_accuracy.json` (`models`) were never filled in. The GGUF comparison was run and verified under the `llama` protocol only (`models_llama`), so no full-protocol GGUF numbers are claimed. lm-evaluation-harness scores are in the lm-evaluation-harness section.

Sections below are earlier measurements (0.1.0 settings, then the upgrade study that selected the new defaults). Where they say the 3% target was missed, that refers to the 0.1.0 settings.
## Environment

| Item | Value |
| --- | --- |
| OS | Windows (native, no WSL) |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU, compute capability 12.0 |
| Python | 3.13.0rc2 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| transformers / datasets / tokenizers | 5.23.0 / 5.2.0 / 0.23.3 |
| gigatoken | 0.11.0 |
| quf | 0.1.0 |

## Accuracy and size (WikiText-2 raw, `quf bench --fp16-ref Qwen/Qwen2.5-0.5B --device cuda`)

Two conversions were made with the native GPTQ quantizer (4 bit, group size 128), differing only in calibration (WikiText-2 train).

| Calibration | fp16 perplexity | QUF perplexity | Relative increase | 3% target |
| --- | --- | --- | --- | --- |
| 32 samples x 512 tokens | 13.0703 | 14.1767 | +8.47% | **Missed** |
| 128 samples x 1024 tokens | 13.0703 | 14.0384 | +7.41% | **Missed** |

The documented target of at most 3 percent perplexity increase was **not met**. More calibration data lowered the gap (8.5% to 7.4%) but not near 3%. I did not investigate the cause or tune the quantizer (for example act-order, damping, or symmetric vs asymmetric), so I cannot say whether the gap is inherent to 4-bit g128 on a 0.5B model or due to the implementation. Small models are known to lose more under 4-bit quantization, but that is not tested here.

| Item | Value |
| --- | --- |
| QUF file size | 469,483,072 bytes (447.7 MiB) for both conversions |
| fp16 weights (494,032,768 params x 2 bytes, computed) | about 988,065,536 bytes, so QUF is about 2.1x smaller |

The file is only about 2.1x smaller than fp16 because the embedding and `lm_head` stay fp16 by design, and they are a large share of this small model.

## Speed

`tokens_per_second` in the bench JSON (RTX 5080 Laptop, `torch_gpu` backend): 16.68 (32x512 file) and 14.64 (128x1024 file). These are from the bench run and vary between runs. The decode loop is not optimized.

Backend comparison (`scripts/bench_backends.py`, 64 new tokens, 17-token prompt, greedy):

| Backend | Device | Decode tokens/s | Peak GPU memory |
| --- | --- | --- | --- |
| cpu | CPU | 9.61 | n/a (RSS 2.2 GB) |
| torch_gpu (cached fp16 weights) | RTX 5080 Laptop | 20.71 | 1.56 GB |
| torch_fused | RTX 5080 Laptop | 9.08 | 1.02 GB |

All three backends produced the same greedy tokens. The fused backend was slower than the cached backend, so the selector keeps `torch_gpu` as the default and chooses it with the reason "cached fp16 weights fit and the fused backend is not faster". Fused uses less GPU memory.

### Fast decode path (task 26.8)

Default-preset files (size_b), RTX 5080 Laptop, 128 new tokens, median of 7 runs after 2 warm-ups, end to end including prefill and graph capture. Details and the bottleneck analysis: [BACKENDS.md](BACKENDS.md). Raw data: `bench_accuracy.json`, key `decode_speed`.

| Model | Eager tok/s | CUDA graph tok/s | Pre-change v020 tok/s | Eager speedup | Graph speedup | 3x Speed_Target |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 57.8 | 255.2 | 40.9 | 1.41x | 6.23x | reached (graph) |
| Qwen2.5-1.5B | 46.6 | 82.7 | 30.7 | 1.52x | 2.70x | **not reached** |

CPU backend on 0.5B: 22.8 tok/s. llama.cpp Q4_K_M on the same GPU, for reference only (different engine): 764.2 tok/s on 0.5B and 450.8 tok/s on 1.5B. The 1.5B graph replay itself takes 6.61 ms per token (151 tok/s), dominated by fp16 `gemv` kernels reading the cached weights, so the shortfall is memory-bound kernel work plus about 0.6 s per run of overhead that was not isolated.

## Generation sample (`quf run`, prompt `The capital of France is`, `--temperature 0 --seed 1234 --max-new-tokens 32`)

Both CPU and CUDA exited 0 and printed identical text. From the 128x1024 file:

```
 Paris. The capital of the United States is Washington, D.C. The capital of the United Kingdom is London. The capital of the United States is Washington,
```

Wall time including model load: CPU 10.5 s, CUDA 8.7 s. The 32x512 file produced different (also repetitive, less accurate) text from the same prompt: ` ______.\nA. Paris\nB. London ...`.

## gigatoken on Windows

gigatoken 0.11.0 installs with pip and runs on native Windows. It was used as `gt.Tokenizer(hf_tokenizer)` and `encode(text)`.

On the WikiText-2 train split (10,966,424 characters, 2,518,423 tokens), with the Qwen2.5-0.5B tokenizer:

| Tokenizer | Seconds |
| --- | --- |
| Hugging Face `tokenizers` | 8.44 |
| gigatoken | 0.23 |

gigatoken was 36.8x faster, and its token ids were identical to Hugging Face's (`gigatoken_matches_hf: true`). Note: the software-versions field of the older `bench_backends.json` says gigatoken "not installed" because it was recorded before installation. This is a single measurement of one tokenizer on one text, not a general speed claim.

## Not done

- **Qwen/Qwen2.5-1.5B** was not run in the 0.1.0 measurements above. It was run later; see the summaries at the top.
- The GGUF Q4 comparison was not part of the 0.1.0 measurements above. It was run later under the `llama` protocol and is verified; see the current summary and the GGUF sections below.
- int6 and int5 storage was not evaluated.
- `torch.compile` for decode was skipped (see [BACKENDS.md](BACKENDS.md)).
- Conversion wall time was not recorded: the earlier attempt reused already-written `.quf` files.
- Marlin was not run on this device (optional backend).

## Upgrade evaluation and 0.2.0 defaults (task 16.8)

Same machine and software as above. Perplexity, file size and tokens per second come from `quf bench`-equivalent runs in `scripts/eval_upgrades.py` (raw data `upgrade_eval.json`, reasoning in `ACCURACY.md`). Calibration was wikitext-2 train, 128 x 1024, except the last row.

| Run | Perplexity | vs fp16 (13.0703) | File size (bytes) | Tokens/s | Convert time (s) |
| --- | --- | --- | --- | --- | --- |
| 0.1.0 settings (g128) | 14.0384 | +7.41% | 469,483,072 | 22.5 | 163 |
| g64 | 13.8412 | +5.90% | 476,472,128 | 27.3 | 166 |
| g32 | 13.6386 | +4.35% | 490,449,792 | 24.7 | 149 |
| clip search | 14.0630 | +7.60% | 469,483,072 | 24.6 | 146 |
| act-order | 13.9863 | +7.01% | 470,482,432 | 22.7 | 137 |
| g32 + clip | 13.5528 | +3.69% | 490,449,792 | 23.9 | 178 |
| g32 + act-order | 13.5543 | +3.70% | 491,449,152 | 22.1 | 143 |
| clip + act-order (g128) | 13.9251 | +6.54% | 470,482,432 | 14.6 | 152 |
| g32 + clip + act-order | 13.4777 | +3.12% | 491,449,152 | 17.6 | 259 |
| + `down_proj` int8 | 13.4056 | +2.57% | 535,153,472 | 17.2 | 142 |
| **+ `sensitive` (`down_proj` and last layer int8), the new default** | **13.3775** | **+2.35%** | 539,605,632 | 18.2 | 143 |
| g32 + clip + act-order, wikitext-2 + C4 128 x 2048 | 13.5492 | +3.66% | 491,449,152 | 7.6 | 714 |

The 3% target is reached with the new defaults on WikiText-2 (+2.35%). On held-out C4 the same file is at +3.30%, which misses it (see the summary at the top). The file grows from 469.5 MB to 539.6 MB (+14.9%), still about 1.83x smaller than the fp16 checkpoint (988.1 MB).

Tokens per second is the short 32-token decode from `quf bench` and varies a lot between runs on this laptop GPU. The 7.6 and 14.6 values were not investigated, so I cannot say whether they come from the settings or from machine noise. Use these numbers only as a rough check.

Defaults now in `quf convert`: group size 32, act-order on, clip search on, `sensitive` precision preset, wikitext-2 calibration of 128 x 1024. The opt-outs are `--group-size 128`, `--no-act-order`, `--no-clip-search`, `--preset none`, and `--legacy-calibration`. The two-source calibration is `--multi-source-calibration`.

## lm-evaluation-harness (tasks 17.4, 21.3)
Status: **run**. lm_eval 0.4.13, `scripts/run_lmeval.py`, tasks `hellaswag` and `arc_easy`, 0-shot, `limit=500` each, seeds 1234, CUDA (RTX 5080). HF rows load the checkpoint in fp16 through `HFLM` (batch 16). QUF rows use the batched `quf/bench/lm_eval_adapter.py`. `QUF 0.2.1 size_b` files were converted with the plain 0.2.1 CLI defaults (`quf convert <model> out.quf`, 497,728,832 and 1,257,449,536 bytes). `QUF 0.2.0 sensitive` rows are the earlier files (group size 32, act-order, `sensitive` preset). Values are accuracy ± stderr. Raw outputs are in `eval_cache/lm/`.

| Model | Variant | hellaswag acc | hellaswag acc_norm | arc_easy acc | arc_easy acc_norm | Eval time |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | HF fp16 | 0.410 ± 0.022 | 0.502 ± 0.022 | 0.636 ± 0.022 | 0.586 ± 0.022 | 96 s |
| Qwen2.5-0.5B | **QUF 0.2.1 size_b (default)** | **0.410 ± 0.022** | **0.500 ± 0.022** | **0.612 ± 0.022** | **0.588 ± 0.022** | 70 s |
| Qwen2.5-0.5B | QUF 0.2.0 sensitive | 0.424 ± 0.022 | 0.496 ± 0.022 | 0.620 ± 0.022 | 0.564 ± 0.022 | 33 s |
| Qwen2.5-1.5B | HF fp16 | 0.486 ± 0.022 | 0.602 ± 0.022 | 0.750 ± 0.019 | 0.710 ± 0.020 | 46 s |
| Qwen2.5-1.5B | **QUF 0.2.1 size_b (default)** | **0.480 ± 0.022** | **0.590 ± 0.022** | **0.744 ± 0.020** | **0.704 ± 0.020** | 36 s |
| Qwen2.5-1.5B | QUF 0.2.0 sensitive | 0.484 ± 0.022 | 0.608 ± 0.022 | 0.748 ± 0.019 | 0.724 ± 0.020 | 45 s |

Every QUF score is within one stderr of the fp16 score, except 0.5B `size_b` arc_easy acc (0.612 against 0.636, a gap of 1.1 stderr). The 0.5B `size_b` eval time includes first-run warm-up, so it is not comparable to the other times. Batched scoring removed the earlier slowness: each pair finishes in about a minute of scoring (the first HF run includes dataset download). No adapter change was needed. The HF fp16 0.5B numbers match the earlier fp32 CPU reference run (`eval_cache/lmeval_hf.json`) exactly on acc.

## GGUF validation against llama-perplexity.exe (task 20.1)

Tool: llama.cpp b12567 `llama-bin-win-cuda-13.5-x64` (runs natively on the RTX 5080, sm_120, `-ngl 99`), in `eval_cache/llama_cpp/`. Text: WikiText-2 raw test joined with `"\n\n"` (`wikitext2_test.txt`), Qwen2.5-0.5B GGUFs from `eval_cache/`, context 2048, no BOS (Qwen adds none). Both tokenizers give 299,078 tokens (146 windows). Raw logs and `diag_gguf_validation.json` are in the same folder; the comparison script is `diag_validate.py`.

Two protocols exist. **Full**: score every position 1..2047 of each non-overlapping window (our `perplexity()` and `_gguf_ppl`). **Half**: `llama-perplexity` default, scores only positions 1024..2046 of each window (second half).

| Protocol | Windows | HF fp16 (CUDA) | llama-perplexity f16 | llama-perplexity Q8_0 | `_gguf_ppl` f16 | `_gguf_ppl` Q8_0 |
|---|---|---|---|---|---|---|
| Full | all 146 | 13.0703 | n/a | n/a | not run | not run |
| Full | first 32 | 13.1839 | n/a | n/a | 13.1837 | 13.2086 |
| Half | all 146 | 11.7429 | 11.7409 | 11.7662 | not run | not run |
| Half | first 32 | 11.8092 | 11.8081 | 11.8348 | 11.8091 | 11.8371 |

Findings:
- `_gguf_ppl` agrees with HF fp16 to 0.001% on f16 (13.1837 against 13.1839, full protocol, 32 windows). Mean per-token NLL difference to HF is -1.4e-5. The 0.5B Q8_0 row is +0.19% over fp16 on the same subset (13.2086 against 13.1839), inside the 0.5% target.
- With the half protocol applied to the same logits, `_gguf_ppl` matches llama-perplexity within 0.02% (f16 11.8091 against 11.8081, Q8_0 11.8371 against 11.8348). The scorer, tokenization, batch size and logits_all handling in `_gguf_ppl` are therefore correct for f16 and Q8_0.
- llama-perplexity's default 11.74 against our 13.07 is a protocol difference (second-half scoring drops the high-loss early positions), not an error. HF fp16 scored the same way gives 11.7429, 0.02% from llama-perplexity f16. The two numbers must not be compared across protocols.
- `llama-perplexity --ppl-stride 2048` is not a full-window equivalent. It runs `perplexity_v2` with n_ctx=3072 and a different windowing (f16: 145 chunks), so it was not used. Full scoring in llama.cpp needs a custom loop, as `_gguf_ppl` does.
- Remaining discrepancies in the stored rows:
  1. The GGUF rows are 32-window subset numbers. They must be compared with the fp16 subset value (13.1839 for 0.5B), not the full-text 13.0703. The two differ by 0.9%.
  2. Peak per-token deviation is 0.12 nats at position 0 of the first window for f16 and 1.7 nats somewhere in Q8_0, which is normal quantization noise on single tokens and not systematic. Mean absolute NLL difference to HF is about 0.004 nats at every position band for f16.
  3. An earlier rope.freq_base bug (transformers 5 dropping `rope_theta` from the GGUF) is already patched in `_patch_convert_script` and `_check_rope_base`. The GGUF files in `eval_cache/` carry the correct value, since the f16 matches HF.
  4. `_gguf_ppl` runs on the CPU wheel. Its token/s are not comparable with GPU numbers. llama-perplexity on the GPU runs the 146-window pass in about 0.3 s per pass. Task 20.2 should add the `--protocol` switch (`full` and `llama` half scoring) and a GPU llama.cpp backend, and 20.3 should rerun the Q4 rows.
- Not fixed here, as 20.2 owns the change. No code was modified in this task.

## llama-protocol rows (task 20.3)

Rows are stored as `eval_cache/qwen2_5-{0_5b,1_5b}_<run>_llama.result.json`, separate from the full-protocol rows. Protocol `llama` scores positions 1024..2046 of each 2048-token window over the full WikiText-2 test text and the C4 slice. GGUF perplexity comes from `llama-perplexity.exe` on the GPU (`-ngl 99`). GGUF tokens/s come from `llama-bench.exe` on the GPU (`-p 4 -n 32`, 7 repeats, median), which is the llama.cpp engine and not the QUF runtime, so speed columns across engines are not comparable. The Qwen2.5-1.5B v020 row hit the 25-minute row limit in the first driver pass. It was re-run alone and finished in about 10 minutes (241 s convert, 365 s benchmark); the cause of the first stall was not found (see below).

Qwen/Qwen2.5-1.5B, `llama` protocol, all windows:

| Run | WikiText-2 ppl | C4 ppl | File size (bytes) | Tokens/s | Engine |
| --- | --- | --- | --- | --- | --- |
| fp16 (HF) | 8.2964 | 13.4756 | 3,087,428,608 | 30.24 | HF generate |
| QUF RTN g128 | 9.2096 | 14.7810 | 1,158,904,640 | 41.92 | QUF torch_gpu |
| QUF RTN g32 | 8.7891 | 14.1846 | 1,235,674,048 | 39.78 | QUF torch_gpu |
| QUF 0.1.0 settings | 8.7527 | 14.3473 | 1,158,904,640 | 39.39 | QUF torch_gpu |
| **QUF v020 (new defaults)** | **8.3954** | **13.6774** | 1,413,323,520 | 39.95 (min 38.96, max 40.46, 7 repeats) | QUF torch_gpu |
| GGUF Q4_K_M | 8.5998 | 13.9536 | 986,048,160 | 209.18 | llama-bench |
| GGUF Q4_0 | 8.9146 | 14.3930 | 934,954,656 | 230.85 | llama-bench |
| GGUF Q8_0 | 8.3163 | 13.5018 | 1,646,572,704 | 143.64 | llama-bench |

QUF v020 is +1.19% (WikiText-2) and +1.50% (C4) over fp16 in this protocol, and is better than GGUF Q4_K_M on both texts at 1.43x its size. Tokens/s across engines are not comparable.

First-run stall of the v020 row. The killed run's log (`eval_cache/qwen2_5-1_5b_v020_llama.killed_run.log`) ends right after "converted in 227s" and has no progress output, so the stage that stalled is unknown. The re-run used the same code, backend (`torch_gpu`, chosen by the selector), and file, and showed no slowdown: about 1 s per 2048-token window, 146 + 138 windows. Peak GPU memory was about 7 GB of 16 GB, so memory pressure and the CPU backend are ruled out for the re-run. The most likely cause is external to quf (another GPU or disk consumer during the first pass), but that is not confirmed. `scripts/bench_accuracy.py` now prints a progress line every 20 windows so a future stall is located.

Q4_K_M fallback quant types. Qwen2.5-0.5B has hidden size 896, which is not a multiple of 256, so llama-quantize cannot use K-quant blocks for tensors with 896 input columns and falls back to a legacy type. In the 0.5B Q4_K_M file, `attn_q`, `attn_k`, `attn_output`, `ffn_gate` and `ffn_up` (24 layers each) fall back from Q4_K to Q5_0. `attn_v` falls back to Q8_0 in 12 layers (where Q6_K was requested) and to Q5_0 in the other 12 (where Q4_K was requested). The 0.5B Q4_K_M row is therefore not a pure 4-bit K-quant file. Qwen2.5-1.5B (hidden size 1536) logs no fallbacks.


