# Backend measurements (task 12.5)

Model: Qwen2.5-0.5B converted with `quf convert` (469,483,072 bytes). GPU: NVIDIA GeForce RTX 5080 Laptop (sm_120, 16 GiB). torch 2.11.0+cu128, Windows native. Triton is not installed, so `torch_fused` ran in eager mode (no `torch.compile`).

Command: `python scripts/bench_backends.py model.quf --tokens 64 --out docs/bench_backends.json`
Greedy decode of 64 tokens after a 17-token prompt, batch 1, after an 8-token warm-up. Each backend runs in its own process. Raw numbers are in `docs/bench_backends.json`.

| Backend | Decode tok/s | Prefill | Load | GPU memory after load | GPU peak |
|---|---|---|---|---|---|
| cpu | 9.6 | 140 ms | 2.4 s | none (process RSS 2104 MiB) | none |
| torch_gpu (cached fp16) | 20.7 | 38 ms | 1.0 s | 958 MiB | 1489 MiB |
| torch_fused (eager) | 9.1 | 96 ms | 0.7 s | 446 MiB | 977 MiB |

Findings:
- The fused backend uses about 2.1x less weight memory than the cached backend but decodes about 2.3x slower, since it unpacks and dequantizes on every call. It is also slower than the CPU run by a small margin.
- Fused and cached backends produce identical greedy tokens for the first 16 outputs.
- All backends are slow in absolute terms (batch 1, Python-level layer loop). The numbers are measured, not tuned.
- Selector decision: the fused backend is not faster, so `torch_gpu` (cached) stays the default. `torch_fused` is chosen automatically only when the cached fp16 weights do not fit in free GPU memory (not the case here: about 1.5 GiB needed against 16 GiB) or with `--backend fused`.
- Re-measure with Triton installed and `compile=True` before changing this decision.
- Property 10 and Property 20 GPU tests (torch_gpu and fused variants, excluding Marlin) pass on this machine. Without Triton the compiled variant falls back to eager or is skipped.

## Decode profile for g_idx and int8 modules (task 22.1)

Script: `scripts/profile_decode.py` (profiling only, no runtime code changed). Qwen2.5-0.5B, RTX 5080, Windows native, 17-token prompt, 48-token greedy decode, 2 warm-up runs, then 7 to 9 timed repeats and the median. Files: new defaults (group 32, act-order g_idx, 30 int8 modules, 540 MB) and a 0.1.0-style file (group 128, all int4, 469 MB).

### Findings

1. **No per-call regather or dequantize.** `reference_dequantize` and `dequantize_int8_weight` are called 0 times during decode on both backends. `prepare` already stores fp16 `[out, in]` tensors for all 138 int4 (g_idx) and 30 int8 modules. The cause of the reported slowdown is not in the weight path.
2. **No measurable slowdown from the new defaults on torch_gpu.** Three interleaved runs of the new-defaults file against the 0.1.0-style file gave medians of 41.6 vs 41.0, 36.5 vs 35.4, and 38.9 vs 38.7 tok/s. The gap between the two files (about 0 to 3 percent) is smaller than the drift between rounds (35 to 44 tok/s over a few minutes, with a second task possibly using the GPU). Each file's repeats within one run spread by 3 to 10 percent.
3. **The earlier 16.5 vs 22 to 27 tok/s most likely came from single-run timing.** `scripts/bench_backends.py` takes one 64-token run after an 8-token warm-up and no repeats. Decode is CPU-launch bound (see below), so it follows CPU scheduling and GPU sharing. That is what Requirement 28.1 (warm-up, at least 5 repeats, median) addresses. The values 16.5 and 22 to 27 are also close to the CPU backend figures below, so a backend mix-up in the earlier comparison is possible and cannot be ruled out from this data.
4. **torch_gpu decode is CPU-launch bound.** Over 16 decoded tokens the profiler shows 409 ms of CPU time against 93 ms of CUDA time (about 25 ms vs 6 ms per token). About 3,500 matmul and 10,000 `aten::to` calls are launched for 16 tokens. Per-token wall time for the new-defaults file: attention block with its q/k/v/o linears 14.7 ms, MLP int4 linears 5.8 ms, rms_norm 5.0 ms (49 calls, fp32 round trip), int8 linears 1.2 ms (30 calls, mostly `down_proj`).
5. **Tied lm_head is converted to fp32 on every token (both files).** `forward` computes `h.float() @ lm_head.float().t()`, which copies the 151936 x 896 fp16 embedding to a 545 MB fp32 tensor each step. This shows up as the 1,600-call elementwise copy kernel (25.5 ms of 93 ms CUDA time, 27 percent) and a 0.9 ms `gemv2T` fp32 kernel per token. Isolated, the lm_head costs 1.75 ms per token on the GPU against about 0.1 ms for an fp16 matvec of the same size.
6. **cpu backend: the same pattern, larger.** Medians over 5 repeats of 16 tokens: new defaults 14.2 tok/s, 0.1.0-style 15.9 tok/s (the new file is about 10 percent slower, with repeats of 13.0 to 15.7 vs 15.5 to 16.2, so partly noise). `aten::copy_` is 53 percent of CPU time because `CPUBackend.matmul` casts the cached fp16 weight to fp32 on every call (`prepared.to(torch.float32)`), and the lm_head does the same for the whole embedding (29.6 ms of about 70 ms per token). Per token: int4 linears 24.4 vs 24.0 ms, int8 linears 8.3 ms (new file only), attention block 21.9 vs 14.4 ms, rms_norm 3.6 vs 2.2 ms. The extra int8 time and the lower attention-block throughput of the new file account for the small gap.

### Candidate fixes for task 22.2

- Keep the fp16 Weight_Cache for g_idx and int8 modules (already true). Task 22.2 only needs the regression test that the prepared tensor is float16 and that decode does not call the gather or dequantize path.
- The largest real costs are outside g_idx and int8: store the tied lm_head in fp16 and compute it without the per-token fp32 copy (about 27 percent of GPU kernel time), and on the cpu backend cache an fp32 copy of the prepared weights instead of casting per call (about 50 percent of CPU time, at the price of 2x CPU memory).
- Reducing launch count (fusing rms_norm, rope and the attention cast chain) would help the CPU-launch-bound GPU path but is out of scope for 22.

## Weight caching fixes (task 22.2)

Changes (`quf/runtime/**` only):
- `CPUBackend.prepare` now returns the weight cached once in fp32 (the compute dtype). `matmul` no longer casts the weight on every call. **Memory trade-off:** the cached cpu weights take 2x the fp16 bytes (the fp16 intermediate is released), for example about 2 GiB of weights for the 0.5B model instead of 1 GiB.
- The dense lm_head (tied embedding or stored `lm_head.weight`) is converted once at load into `QUFModel._lm_head_weight` in the compute dtype. On the GPU this is the fp16 embedding itself (no extra memory, no per-token copy). On the CPU it is an fp32 copy (+545 MB for the 0.5B vocabulary) next to the fp16 embedding used for token lookup.
- GPU lm_head logits now come from an fp16 matmul with fp32 accumulation inside the kernel, rounded to fp16 before the cast to fp32 (resolution about 0.015 at logit magnitude 20 to 30). The whole-model GPU tests keep matching the cpu argmax.
- Regression test: `tests/test_decode_weight_cache.py` checks the prepared dtype (fp16 on torch_gpu, fp32 on cpu, for int4 g_idx, int8 and fp16 modules) and that decode never calls `reference_dequantize` or `dequantize_int8_weight`.

Measurements (`out.quf`, 32 tokens, median of 7 to 9 repeats; the GPU was shared with another task and the CPU was loaded, so runs are noisy):

| Backend | Before (tok/s) | After (tok/s) |
|---|---|---|
| cpu | 11.4 (one run) | 8.9 and 17.8 (two runs; the first run had repeats of 5.4 to 18.7) |
| torch_gpu | 42.5 (one run) | 46.5 and 40.3 (two runs) |
| lm_head alone, cpu | 29.6 ms | 7.8 ms |
| lm_head alone, torch_gpu | 1.74 ms | 0.55 ms |

The lm_head cost drops clearly on both devices. End-to-end torch_gpu decode stays inside the run-to-run noise because it is CPU-launch bound (see finding 4); the saved 1 ms of GPU time per token is hidden behind launch overhead. The cpu median is dominated by noise in these runs.

## torch.compile decode path (task 26.4): skipped
Probe on Windows, torch 2.11.0+cu128, RTX 5080 (sm_120), Python 3.13:
- Without Triton (the default install): `torch.compile` of a small function raises `TritonMissing` ("Cannot find a working triton installation"). The compiled path cannot run, so it is skipped (Requirement 31.9).
- With `pip install triton-windows` (3.9.0.post30), compile works. Compiling `QUFModel.decode_step` took about 24 s on first call. Decode medians on `eval_cache/lm/q15.quf`, torch_gpu, 64 tokens, 5 repeats: eager 44.0 tok/s, CUDA graph 144.9 tok/s, torch.compile 129.0 tok/s.
- The compiled step is about 11 percent slower than the CUDA graph path and adds a Triton dependency plus a compile delay, so no opt-in path was added. triton-windows was uninstalled after the probe.

## Decode speed after the fast path (task 26.8)

Script: `scripts/measure_decode_speed.py`. Files: `quf convert` with the new default preset (size_b), 497,728,832 bytes for Qwen2.5-0.5B and 1,257,449,536 bytes for Qwen2.5-1.5B. RTX 5080 Laptop, Windows native, GPU idle before the run (nvidia-smi 0 percent). Prompt `The quick brown fox`, 128 new tokens, greedy, 2 warm-up runs, median of 7 timed runs. Each run covers prefill, decode, and in graph mode the graph capture, so it is an end-to-end rate. Raw numbers: `docs/bench_accuracy.json`, key `decode_speed`.

| Model | Backend | Median tok/s | Min to max | Speedup vs pre-change v020 |
|---|---|---|---|---|
| 0.5B | torch_gpu eager | 57.8 | 56.9 to 59.6 | 1.41x (vs 40.9) |
| 0.5B | torch_gpu CUDA graph | 255.2 | 251.6 to 258.5 | 6.23x |
| 0.5B | cpu | 22.8 | 21.8 to 25.8 | no v020 CPU baseline |
| 1.5B | torch_gpu eager | 46.6 | 43.3 to 48.9 | 1.52x (vs 30.7) |
| 1.5B | torch_gpu CUDA graph | 82.7 | 81.2 to 84.4 | 2.70x |
| 0.5B | llama.cpp Q4_K_M (reference) | 764.2 | 760.1 to 770.3 | n/a |
| 1.5B | llama.cpp Q4_K_M (reference) | 450.8 | 412.4 to 459.4 | n/a |

The llama.cpp rows come from `llama-bench.exe -ngl 99 -p 4 -n 128`. They are a different engine (hand-written CUDA kernels on 4-bit weights) and a different measurement (no Python, no sampling round trip per token), so they show how close the runtime is to GPU speed and are not like for like.

Speed_Target (3x the measured pre-change median, which is the v020 row in `bench_accuracy.json`: 40.9 tok/s for 0.5B and 30.7 tok/s for 1.5B):
- 0.5B: reached with the CUDA graph (6.23x). Eager decode alone is 1.41x.
- 1.5B: **not reached**. The CUDA graph gives 2.70x (82.7 tok/s), short of the 92.0 tok/s that 3x needs. Against a flat 40 tok/s baseline it is 2.07x. Eager decode is 1.52x.

Remaining bottleneck (ad hoc probe, not a repeated benchmark): replaying the captured graph alone takes 3.18 ms per token on 0.5B (314 tok/s) and 6.61 ms per token on 1.5B (151 tok/s). On 1.5B the profiler puts 73 percent of GPU time in two fp16 `gemv` kernels (cuBLAS matvec on the cached fp16 weights, about 3 GB read per token, roughly 480 GB/s). The decode step is now GPU-bound on reading fp16 weights, so the next gain needs 4-bit kernels (as in llama.cpp), which the runtime does not have. The end-to-end 1.5B rate (82.7) is also well below the replay rate (151). The gap is about 0.6 s per 128-token run outside the replay and was not isolated here (candidates: per-token argmax and host sync, prefill, graph capture). The run-to-run spread is small (under 4 percent), so the number is stable.

## Int4 backend measurements and the default decision (tasks 31.1 and 31.2)
The decision is in the last subsection. Script: `scripts/measure_decode_speed.py --compare torch_gpu triton_int4` (speed), `--memory --backend X` (memory, fresh process each), `scripts/bench_accuracy.py --ppl-quf FILE --backend X --protocol llama` (perplexity). Raw numbers: `docs/bench_accuracy.json`, key `int4_backend`; logs in `eval_cache/int4/`. Files: `eval_cache/lm/sb05.quf` (497,728,832 bytes) and `sb15.quf` (1,257,449,536 bytes), size_b preset. RTX 5080 Laptop, Windows, GPU idle before every step (nvidia-smi 0 MiB), triton-windows. Prompt `The quick brown fox`, 128 new tokens (all runs generated exactly 128), greedy, EOS suppressed, 2 warm-up runs, median of 7, the two backends alternate inside every repeat. End-to-end covers prefill, graph capture, and decode. Steady state is the mean gap between consecutive `decode_token` calls (prefill and capture excluded; includes the per-token argmax and host sync).

### Speed
| Model | Backend | Mode | End-to-end tok/s (median, min to max) | Steady decode ms/tok | Steady tok/s |
|---|---|---|---|---|---|
| 0.5B | torch_gpu | eager | 58.4 (57.0 to 59.2) | 17.08 | 58.5 |
| 0.5B | triton_int4 | eager | 46.7 (44.4 to 47.6) | 21.38 | 46.8 |
| 0.5B | torch_gpu | graph | 234.4 (227.9 to 240.1) | 3.625 | 275.9 |
| 0.5B | triton_int4 | graph | 284.2 (275.9 to 295.5) | 2.654 | 376.8 |
| 1.5B | torch_gpu | eager | 40.6 (39.1 to 41.6) | 24.60 | 40.6 |
| 1.5B | triton_int4 | eager | 30.9 (30.1 to 34.3) | 32.33 | 30.9 |
| 1.5B | torch_gpu | graph | 99.7 (98.1 to 101.1) | 9.157 | 109.2 |
| 1.5B | triton_int4 | graph | 145.6 (141.2 to 148.5) | 5.779 | 173.0 |
| 0.5B | llama.cpp Q4_K_M (reference) | n/a | 756.5 (724.7 to 795.5) | n/a | n/a |
| 1.5B | llama.cpp Q4_K_M (reference) | n/a | 422.8 (410.3 to 451.0) | n/a | n/a |

Graph mode was confirmed active (the capture succeeded) for both backends. llama.cpp numbers come from `llama-bench.exe -ngl 99 -p 4 -n 128`, a different engine, for reference only.

triton_int4 in graph mode beats torch_gpu by 1.21x (0.5B) and 1.46x (1.5B) end to end, and by 1.37x and 1.58x in steady state. In eager mode it is slower than torch_gpu (0.80x and 0.76x): the Triton launches add host overhead that the graph removes.

Note: torch_gpu graph measured 234.4 and 99.7 tok/s here, against 255.2 and 82.7 in task 26.8. The 0.5B figure is 8 percent lower and the 1.5B figure 21 percent higher than the earlier run, so the alternating loop and the earlier separate-process runs do not give identical torch_gpu numbers.

### Speed target (1.5x of 255 and 83 tok/s, graph mode, end to end)
| Model | Baseline | Required | Measured triton_int4 graph | Speedup | Met |
|---|---|---|---|---|---|
| 0.5B | 255 | 382.5 | 284.2 | 1.11x | **No** |
| 1.5B | 83 | 124.5 | 145.6 | 1.75x | Yes |

The 0.5B steady-state rate (376.8 tok/s) is within 2 percent of 382.5, but the end-to-end rate is what the target uses, and the 128-token run spends about 0.11 s of 0.45 s outside steady decode (prefill and graph capture).

### GPU memory (MiB)
| Model | Backend | Allocated after load | Device used after load (driver) | Peak allocated, graph generation | Peak reserved | Device used after generation |
|---|---|---|---|---|---|---|
| 0.5B | torch_gpu | 947 | 1554 | 976 | 1530 | 1562 |
| 0.5B | triton_int4 | 508 | 786 | 529 | 750 | 812 |
| 1.5B | torch_gpu | 2951 | 5018 | 2984 | 4994 | 3920 |
| 1.5B | triton_int4 | 1320 | 2170 | 1344 | 2134 | 2120 |

"Device used" is the drop in free memory from `torch.cuda.mem_get_info` and includes the CUDA context. The 1.5B torch_gpu value after load (5018) is above the value after generation (3920) because the allocator released a load-time transient. Generation includes the Static_KV_Cache and the CUDA graph. triton_int4 keeps weights packed in int4, so it uses 45 to 54 percent of the torch_gpu memory.

### Perplexity, llama protocol (second half of each 2048 window), full WikiText-2 test and the C4 slice
| Model | Backend | WikiText-2 | C4 slice | Difference vs torch_gpu |
|---|---|---|---|---|
| 0.5B | torch_gpu | 12.0769 | 19.0473 | n/a |
| 0.5B | triton_int4 | 12.0769 | 19.0473 | 0.000 percent (both) |
| 1.5B | torch_gpu | 8.4442 | 13.8663 | n/a |
| 1.5B | triton_int4 | 8.4442 | 13.8663 | 0.000 percent (both) |

Both models pass the 0.1 percent check. The values are identical because prefill with more than 16 rows dequantizes each weight to fp16 and calls `F.linear`, so a full-window forward does not run the Triton GEMV kernels. To check the kernels themselves, the first two WikiText-2 windows were also scored token by token through `decode_token` (M=1, the Triton GEMV path), same llama scoring:

| Model | torch_gpu | triton_int4 | Difference |
|---|---|---|---|
| 0.5B | 12.6420 | 12.6441 | +0.017 percent |
| 1.5B | 8.7952 | 8.7931 | -0.024 percent |

Both stay within 0.1 percent. This second check covers 2 windows only, not the whole text.

### Result summary
- triton_int4 meets the speed target on 1.5B (1.75x) and misses it on 0.5B (1.11x).
- It is faster than torch_gpu in graph mode on both models and slower in eager mode.
- It passes the 0.1 percent perplexity check on both models and uses about half the GPU memory.

### Default decision (task 31.2)
**triton_int4 is the default backend on CUDA** (`INT4_DEFAULT_ON = True` in `quf/runtime/selector.py`) when Triton imports and the int4 self-test passes. Reasons:
- Speed: in the default graph mode it is faster than torch_gpu on both models, 1.21x (284 against 234 tok/s) on 0.5B and 1.46x (146 against 100 tok/s) on 1.5B.
- Memory: it uses 45 to 54 percent of the torch_gpu GPU memory after load (508 against 947 MiB on 0.5B, 1320 against 2951 MiB on 1.5B).
- Accuracy: perplexity matches torch_gpu (identical on the full-window runs, within 0.024 percent in the token-by-token kernel check), and greedy tokens are identical.
- Target: the 1.5x speed target over 255 and 83 tok/s is met on 1.5B (1.75x) and missed on 0.5B (1.11x). The target was not a condition for the default, which needs only that the backend is faster on both models and passes the logits and perplexity checks.

QUF is still 2.6 to 2.9x slower than llama.cpp Q4_K_M on this GPU (284 against 757 tok/s on 0.5B, 146 against 423 tok/s on 1.5B).

**Eager mode is slower.** Without the CUDA graph, triton_int4 decodes at 0.80x (0.5B) and 0.76x (1.5B) of torch_gpu, so the selector guards that case:
- `auto` selects torch_gpu when the CUDA graph is disabled before selection (`--no-cuda-graph` or `QUF_NO_CUDA_GRAPH=1`). The int4 probe is not called, and the "Selected backend" INFO line says why.
- If the graph capture fails at run time while triton_int4 is loaded, one INFO line reports the failure, says that eager int4 is slower than torch_gpu, and recommends `--backend cached`. The backend is not swapped, because that would reload the weights.
- An explicit `--backend int4` is always honored.

Without Triton (`pip install "quf[fast]"` not done), with `QUF_NO_TRITON=1`, or when the self-test fails, `auto` falls back to torch_gpu (or torch_fused when the cached weights do not fit).

## Decode profile (task 33.1)

Script (removed in task 39.3, merged into `scripts/profile_decode_040.py`): `scripts/profile_decode_graph.py --quf FILE --out eval_cache/profile33/NAME.json` (raw numbers in `eval_cache/profile33/`). The older `scripts/profile_decode.py` (task 22.1, torch_gpu wall-clock components) is unchanged. Setup: RTX 5080 Laptop, Windows, triton-windows, GPU idle (nvidia-smi 0 percent, 0 MiB) before each run, `QUF_NO_TRITON` unset, backend `triton_int4`, files `eval_cache/lm/sb05.quf` and `sb15.quf`, prompt `The quick brown fox` (4 tokens), Static_KV_Cache of prompt + 128 positions as `generate` allocates it, greedy. Each run decodes 32 warm-up tokens and then profiles 64 steady-state tokens (positions 36 to 99) with `torch.profiler` (CPU + CUDA activities).

Method:
- **Graph run.** The real `CUDAGraphDecoder` loop (fill token and position, `graph.replay()`, `argmax(...).item()`) under the profiler. 64 replays were seen, each with the same kernel count (1202 kernels on 0.5B, 1452 on 1.5B).
- **Attribution.** Kernels inside a graph have no op names, so I ran an annotated copy of `QUFModel.decode_step` eagerly (labels through `record_function`, logits identical to `decode_step`, max abs diff 0.0) and mapped each graph kernel to the eager kernel at the same position in the step. The kernel names matched at every one of the 1202 and 1452 positions (100 percent), so the per-group times of the graph run are exact kernel times, not eager estimates. The last column is the same table from the eager run, as a cross-check (within 1 to 7 percent per group, except as noted).
- **Groups.** "int4 GEMV" groups hold everything inside the projection call: the Triton `_int4_gemv1_kernel`, the bias add of QUFLinear, and the cuBLAS fp16 gemv or reduce kernels of the modules that are routed off the int4 kernel (int8 or unsupported shapes, split-K reduce). Kernel count per token does not include the 2 host-loop kernels for token and position fill (counted under copies/casts) and the argmax kernels (sampling).
- Kernel durations are CUPTI durations; they exclude launch gaps, so the sum of kernel times is the GPU busy time, and the pure replay time below includes the gaps.

### Results

#### Qwen2.5-0.5B (sb05.quf): GPU time per token by group, graph mode

| Group | µs/token | % of total | Kernels/token | Eager µs/token (check) |
|---|---|---|---|---|
| int4 GEMV qkv (incl. bias add) | 128.7 | 5.1 | 48 | 129.1 |
| int4 GEMV o | 89.2 | 3.5 | 24 | 89.0 |
| int4 GEMV gate_up | 291.4 | 11.5 | 24 | 276.8 |
| int4 GEMV down | 286.3 | 11.3 | 24 | 267.3 |
| attention (matmul, mask, softmax, matmul) | 202.7 | 8.0 | 168 | 213.3 |
| RMSNorm (2 per layer plus final, with its casts) | 460.7 | 18.2 | 392 | 467.4 |
| RoPE (tables and rotation) | 291.2 | 11.5 | 253 | 306.5 |
| KV-cache writes | 51.4 | 2.0 | 48 | 52.1 |
| residual adds | 48.0 | 1.9 | 48 | 50.0 |
| SiLU*mul | 54.0 | 2.1 | 48 | 54.2 |
| lm_head (fp16 cuBLAS gemv + fp32 cast) | 406.5 | 16.1 | 2 | 382.9 |
| sampling/argmax (outside graph) | 7.8 | 0.3 | 3 | 7.5 |
| copies/casts (kv expand, fp32 casts, reshape, token fill) | 206.5 | 8.2 | 122 | 208.1 |
| other (embedding gather, mask) | 3.2 | 0.1 | 3 | 3.4 |
| **Total GPU kernel time** | **2527** | 100 | **1207** | 2508 |

#### Qwen2.5-1.5B (sb15.quf): GPU time per token by group, graph mode

| Group | µs/token | % of total | Kernels/token | Eager µs/token (check) |
|---|---|---|---|---|
| int4 GEMV qkv (incl. bias add) | 241.0 | 5.6 | 56 | 244.6 |
| int4 GEMV o | 172.9 | 4.0 | 28 | 173.1 |
| int4 GEMV gate_up | 883.2 | 20.7 | 28 | 930.7 |
| int4 GEMV down | 689.9 | 16.1 | 82 | 692.3 |
| attention (matmul, mask, softmax, matmul) | 258.0 | 6.0 | 196 | 267.9 |
| RMSNorm (2 per layer plus final, with its casts) | 546.8 | 12.8 | 456 | 542.3 |
| RoPE (tables and rotation) | 364.3 | 8.5 | 293 | 368.6 |
| KV-cache writes | 76.7 | 1.8 | 56 | 74.9 |
| residual adds | 53.5 | 1.3 | 56 | 52.2 |
| SiLU*mul | 71.5 | 1.7 | 56 | 64.9 |
| lm_head (fp16 cuBLAS gemv + fp32 cast) | 631.6 | 14.8 | 2 | 659.6 |
| sampling/argmax (outside graph) | 7.9 | 0.2 | 3 | 7.5 |
| copies/casts (kv expand, fp32 casts, reshape, token fill) | 273.6 | 6.4 | 142 | 277.3 |
| other (embedding gather, mask) | 3.1 | 0.1 | 3 | 3.4 |
| **Total GPU kernel time** | **4274** | 100 | **1457** | 4359 |

(Run 2 of sb05 in the same session: total GPU kernel time 2463 µs, pure replay 2545 µs, group shares within 1 point of run 1 except gate_up 275 µs and lm_head 380 µs. Run to run noise on a group is about 5 percent.)

### Per-token time against llama.cpp Q4_K_M

llama.cpp rates are from the 31.1 measurements: 756.5 tok/s (1.322 ms/token) for 0.5B and 422.8 tok/s (2.365 ms/token) for 1.5B.

| | 0.5B | 1.5B |
|---|---|---|
| Sum of kernel times (GPU busy) | 2527 µs | 4274 µs |
| Pure graph replay, 64 back-to-back replays, no host work (median of 5) | 2544 µs | 4345 µs |
| Real decode loop per token (fill, replay, argmax, `.item()`), median | 2908 µs (run 2: 2732) | 4717 µs |
| llama.cpp per token | 1322 µs | 2365 µs |
| Pure replay / llama.cpp | 1.92x | 1.84x |
| Real loop / llama.cpp | 2.20x (run 2: 2.07x) | 1.99x |
| Kernel launches per token (graph) | 1207 | 1457 |
| Kernel launches per token (llama.cpp) | not measured | not measured |

The loop here is 0.5B 10 percent slower and 1.5B 18 percent faster than the 31.1 steady-state figures (2.654 ms and 5.779 ms). I did not find the cause; 31.1 measured generate over 128 tokens with alternating backends, this profile times a 64-token window of the bare loop. Use these numbers for the breakdown and the 31.1 table for speed claims.

### Host side, per token (profiler off, median of 64 tokens)

| | 0.5B | 1.5B |
|---|---|---|
| Fill token and position (2 tiny GPU kernels, CPU time) | 43 µs | 50 µs |
| `graph.replay()` CPU time (launch only) | 88 µs | 110 µs |
| `argmax(...).item()` (waits for the replay, then copies one value) | 2775 µs | 4553 µs |
| Loop total | 2908 µs | 4717 µs |
| Pure replay (GPU only) | 2544 µs | 4345 µs |
| **Gap: loop minus pure replay** | **364 µs (run 2: 187 µs)** | **372 µs** |

The gap is the GPU idle time between two replays: the host cannot launch token n+1 until `.item()` returns for token n, then it fills the token, launches, and the GPU has to restart. It is 7 to 14 percent of the loop and does not shrink unless the token stays on the device and the host stops synchronizing every token (Requirement 37.4). The argmax itself costs only 3 kernels and 8 µs.

### What the profile says

- **Kernel count dominates, not data volume.** There are 1207 launches (0.5B) and 1457 (1.5B) per token. The int4 GEMVs are only 24 to 28 launches per projection kind. RMSNorm (392 launches, 18 percent on 0.5B), RoPE (253), attention plus its copies (168 + 122), residual, SiLU*mul and KV-write together are about 1000 tiny elementwise or reduce kernels at 1 to 4 µs each. In a graph each such kernel costs roughly its own run time of 1 to 4 µs plus the dependency latency between kernels, so removing a launch saves about 1 to 3 µs.
- **The int4 GEMVs run far below the bandwidth the card reaches.** The fp16 lm_head runs at about 670 GB/s on 0.5B (272 MB in 406 µs) and 740 GB/s on 1.5B (467 MB in 632 µs). The int4 GEMVs read about 225 MB (0.5B) and 790 MB (1.5B) per token in 796 µs and 1987 µs, which is 280 and 400 GB/s. Per-projection, `_int4_gemv1_kernel` takes 97 + 83 + 253 + 272 µs on 0.5B and 200 + 164 + 805 + 557 µs (qkv, o, gate_up, down) on 1.5B. On 1.5B `down` also launches a split-K reduce (63 µs) and a copy (26 µs), and some modules go through a cuBLAS gemv (the 82 launches).
- **lm_head is 15 to 16 percent** and reads the fp16 embedding table. It becomes a GEMV over int4 or int8 weights only when the file stores a quantized lm_head (task 34).
- **GPU floor.** At 700 GB/s, reading the whole file (498 MB and 1257 MB) takes about 0.71 ms and 1.80 ms per token. llama.cpp (1.32 and 2.37 ms) is at 54 and 76 percent of that floor, QUF at 28 and 41 percent (pure replay).

### Fused kernels to build in 33.2, ranked by expected µs saved per token

Savings are estimated as measured group time minus the time of the replacement, assuming 3 to 4 µs per small fused kernel and the GEMV rate stated below; they are not measured and they overlap slightly, so do not add them to the last decimal. Values are 0.5B / 1.5B.

| Rank | Kernel | Replaces (µs/token now) | Replacement cost | Expected saving (0.5B / 1.5B) |
|---|---|---|---|---|
| 1 | Fused residual-add + RMSNorm (`add_rmsnorm`), one kernel per norm, writing the residual and the normed output | RMSNorm 461 / 547 (392 / 456 launches) + residual adds 48 / 54 (48 / 56) | 49 / 57 kernels at 3 µs = 147 / 171 µs | **about 360 / 430** |
| 2 | GPU argmax in the graph, token kept on the device as the next replay's input, host checks for EOS every N tokens or reads a buffer asynchronously (Requirement 37.4) | Host gap 187 to 364 / 372 µs + the 2 fill kernels + argmax 8 µs | 1 small kernel, about 3 µs, in the graph | **about 270 (190 to 360) / 370** |
| 3 | Fused RoPE + KV-cache write (`rope_kv_write`), with cos and sin read from a table indexed by position instead of computed in the graph | RoPE 291 / 364 (253 / 293 launches) + KV writes 51 / 77 (48 / 56) | 24 / 28 kernels at 4 µs = 96 / 112 µs | **about 245 / 330** |
| 4 | Flash-decode GQA attention, split-K over the sequence, reading the cache with the KV-head repeat done by indexing (no `repeat_interleave`, no fp32 copies of K, no mask tensor) | Attention 203 / 258 (168 / 196 launches) + the attention copies and casts 204 / 271 (all but 2 of the 122 / 142 copies launches) | split + merge, about 8 µs per layer = 190 / 225 µs | **about 215 / 305** |
| 5 | lm_head through the int4 or int8 GEMV with a quantized table (33.2 routing, needs the file from task 34) | fp16 lm_head 406 / 632 | int4 at today's GEMV rate (about 280 to 400 GB/s) 70 MB: about 190 / 300 µs; at 650 GB/s: about 110 / 180 µs | **about 215 / 330** (at the faster GEMV rate: 295 / 450) |
| 6 | SiLU*mul and the QKV bias add fused into the int4 GEMV prologue or epilogue | SiLU*mul 54 / 72 + bias add 24 / 29 (inside the qkv group) | 0 extra launches | **about 60 to 75 / 90 to 100** |

Not on the 33.2 list but larger than ranks 3 to 6 on 1.5B: **tuning the int4 GEMV itself** (`_int4_gemv1_kernel` and the split-K reduce on `down`). The GEMV groups hold 796 / 1987 µs; at the 650 GB/s that cuBLAS reaches on the same card the 225 / 790 MB would take about 350 / 1215 µs, a saving of about **450 / 770 µs** (about 17 and 18 percent of the token time). I recommend adding it to 33.2 or as its own task; ranks 1 to 5 alone do not reach llama.cpp.

Projection if ranks 1 to 5 land: 0.5B real loop 2.91 ms minus about 1.3 ms is about 1.6 ms per token (about 620 tok/s, llama.cpp 757); 1.5B 4.72 ms minus about 1.7 ms is about 3.0 ms (about 330 tok/s, llama.cpp 423). Adding the GEMV tuning gives about 1.15 ms and 2.25 ms, which is at or just under the llama.cpp per-token time on both models. These are projections from the profile, not measurements.

Order of work: ranks 1, 3, and 4 are independent kernels and each shows its saving in the profile table; rank 2 needs a change to `CUDAGraphDecoder` and `generate` (token on device, deferred EOS check) and gives the largest single gain per line of code; rank 5 depends on task 34.1.

### Fused decode kernels, part B (task 33.2)

Kernels: `quf/runtime/kernels/decode_gemv.py` (j-major int4 GEMV with bias epilogue, SiLU*mul scatter) and `quf/runtime/kernels/attention.py` (fused RoPE + KV-cache write, flash-decode GQA with split-K over the sequence). All are CUDA-graph capturable, read the position from `cache.position`, and run only when `model.fused_decode` is true. A layer whose four projections are not all supported int4 weights (the last layer of sb05 and sb15 is int8/fp16) keeps the 0.3.0 calls.

What changed in the int4 GEMV: the in-model weights are act-order (non-identity `perm`) with group size 32. The old M=1 kernel gathered `x[perm[k]]` with an int64 perm load per element, which moved about 4 times more bytes through L1 than the weights themselves. Now each producer (add+RMSNorm, flash-decode output, SiLU*mul) scatters its output once into the consumer's "j-major" layout (`xj[j*(K/8)+w] = x[perm[8w+j]]`), and the GEMV reads contiguous vectors, one scale/zero per group, with no perm load. The qkv bias is added in the GEMV epilogue (fp16 add after the fp16 cast, same rounding as the torch path). The config is `(BLOCK_N, groups per step, warps)` = `(1, 32, 2)` for K >= 4096 and `(2, 16, 1)` otherwise.

Measured kernel times in the real graph (median per call, µs):

| | 0.5B old | 0.5B new | 1.5B old (task 33.1) | 1.5B new |
|---|---|---|---|---|
| qkv (incl. bias) | 5.4 | 3.8 | 8.6 | 6.7 |
| o | 3.7 | 3.3 | 6.2 | 5.6 |
| gate_up | 12.1 | 13.8 | 31.5 | 39.7 |
| down (incl. split-K reduce) | 11.9 | 8.1 | 24.6 | 25.2 |

The new kernel is slower on gate_up and unchanged on 1.5B down: the cold-weight lab sweep (one weight copy per layer, graph-timed) predicted larger gains than the real graph shows, because on this 55 W laptop GPU a GEMV-only loop is power limited in a way the mixed decode graph is not. The cold sweep results should not be used to pick configs; sweep inside the real decode graph (whole-graph replay time) instead. Neither the in-kernel split-K reduce nor SiLU*mul in the gate_up epilogue was tried.

End-to-end steady-state ms/token (`scripts/measure_decode_speed.py --compare triton_int4`, separate processes, GPU idle before each run; run-to-run spread is 3 to 6 percent):

| | 0.3.0 path (`QUF_NO_FUSED=1`) | part A | + int4 GEMV | + RoPE/KV write + flash-decode (final) |
|---|---|---|---|---|
| sb05 | 3.83 | 3.26 | 3.03 | 2.56 to 2.74 |
| sb15 | 7.86 to 7.91 | 7.6 | 7.29 | 7.49 to 7.81 |

Raw runs: `eval_cache/profile33/b0_*.json` (before part B), `b1_*` (GEMV), `b2_*`, `b3_*` (all of part B). On 0.5B the GPU busy time per token after part B is about 2.05 ms (profiler) against 2.6 ms wall, so launch gaps and clock behaviour now matter; on 1.5B the profiler shows about 5.4 ms of kernels against 7.5 ms wall. lm_head (and the int8/fp16 last layer) is about 0.7 ms of 0.5B and 1.5 ms of 1.5B kernel time and is left for the quantized embedding (34.x).

Greedy tokens equal the 0.3.0 path on sb05 and sb15 for 4 prompts x 64 tokens (no flips). Flash-decode normalises the softmax after the PV sum instead of rounding normalised probabilities to fp16, so near-ties could flip in principle.

## Quantized_Embedding on the Backends (task 34.3)

Code: `quf/runtime/embedding.py` (`RuntimeEmbedding`), `quf/runtime/kernels/embed_gemv.py` (Triton GEMV). Tests: `tests/test_embed_runtime.py`.

The table stays packed on the device (int4 bytes or int8, plus fp16 scales) on every Backend. No Backend builds a full fp16 copy of it.

| Backend | row lookup | tied or untied quantized lm_head, M = 1 to 16 | lm_head, M > 16 (prefill) |
|---|---|---|---|
| cpu | reference row dequant (`dequantize_embedding_rows`) | chunked dequant + fp32 matmul | same |
| torch_gpu, torch_fused | same function on the GPU, ids stay on the device | chunked dequant + fp16 matmul, fp32 output | same |
| triton_int4 | same function (torch ops, graph capturable) | Triton GEMV on the stored layout, fp32 accumulation, fp32 output | chunked dequant + matmul |

Chunked dequant: the vocabulary is dequantized `CHUNK_ELEMENTS` (4M elements, 8 MB fp16) at a time into a transient buffer and multiplied chunk by chunk, so the extra memory is bounded and independent of vocabulary size. Choice for the non-kernel Backends: the cached-fp16 policy of `torch_gpu` applies to the projection weights, not to the embedding, because a cached fp16 table (272 MB for Qwen2.5-0.5B) would cancel the saving the quantized table exists for. The price is a dequantization per decoded token on those Backends. v1 and v2 files with an fp16 embedding take the old path unchanged (`model.embed` is the fp16 tensor, `model.qembed` is None).

GEMV: one program per `BLOCK_N` = 32 vocabulary rows (grid `[ceil(N / 32), M]`), loop over K in blocks of 128 columns, int4 nibbles unpacked in registers, scale gathered per column group, fp32 accumulation. One launch, no host sync; it runs inside the CUDA graph of the fused decode path. Config sweep: `scripts/_embed_gemv_lab.py` (N = 151936, K = 896, L2 flushed per call): int4 about 0.18 to 0.23 ms, int8 about 0.29 ms; many configs tie within noise.

## Decode speed against the Requirement 37 targets (task 33.5)
Data: `docs/bench_accuracy.json` key `speed_040`; raw runs in `eval_cache/speed040/`. Scripts: `scripts/bench_speed040.py`, `scripts/profile_decode_040.py`.
Method: Windows, RTX 5080 laptop, graph mode, backend `triton_int4`, prompt "The quick brown fox", 128 new tokens, EOS suppressed, warm-up 2, median of 7, runs alternate between configurations inside each round. llama.cpp: `llama-bench -p 0 -n 128 -ngl 99 -r 1`, repeated in alternating rounds (9 rounds, first 2 discarded). Steady ms/token excludes prefill and graph capture. The GPU was idle before the run.

**Power state.** The requested state was power-capped (about 55 W, about 900 MHz). The sampled state was not that cap: QUF runs drew a median 112 W (0.5B) and 127 W (1.5B) at median SM clocks of 2497 and 2310 MHz; llama.cpp runs drew a median 48 W (0.5B) and 64 W (1.5B) at 1627 and 1541 MHz. Throttle reasons seen: 0x400 (power brake), 0x4 (SW power cap), 0x20 (SW thermal slowdown, 1.5B QUF run only). So these are laptop numbers under varying power limits, not the 55 W cap, and QUF and llama.cpp did not run at the same clocks. Treat cross-engine ratios as indicative.

| Model | Config | file MB | e2e tok/s | steady ms/token | steady tok/s |
|---|---|---|---|---|---|
| 0.5B | QUF 0.4.0 (q05.quf) | 368 | 343.9 | 1.990 | 502.5 |
| 0.5B | 0.3.0 file (sb05.quf), 0.3.0 decode path (QUF_NO_FUSED=1) | 498 | 287.6 | 2.528 | 395.6 |
| 0.5B | 0.3.0 file, current code with fused kernels | 498 | 579.4 | 1.240 | 806.4 |
| 1.5B | QUF 0.4.0 (q15.quf) | 1040 | 203.5 | 3.777 | 264.8 |
| 1.5B | 0.3.0 file (sb15.quf), 0.3.0 decode path (QUF_NO_FUSED=1) | 1257 | 171.7 | 4.631 | 215.9 |
| 1.5B | 0.3.0 file, current code with fused kernels | 1257 | 290.4 | 2.827 | 353.7 |

| GGUF (llama.cpp tg128, tok/s) | 0.5B | 1.5B |
|---|---|---|
| Q4_K_M | 773.1 | 448.8 |
| Q4_K_S | 781.9 | 458.1 |
| IQ4_XS | 791.4 | 447.0 |
| Q4_K_M-imat | 774.5 | 448.3 |
| Q5_K_M | 789.5 | 410.8 |

| Target | 0.5B | 1.5B | Met |
|---|---|---|---|
| Required: 2x the 0.3.0 speed (0.4.0 / 0.3.0 path, e2e) | 1.20x | 1.19x | No |
| Required, steady-state ms/token | 1.27x | 1.23x | No |
| Stretch: at least Q4_K_M tok/s (steady 502 vs 773; 265 vs 449) | 0.65x | 0.59x | No |

Both targets are missed for the 0.4.0 defaults. The 0.3.0 file run through the new fused kernels is faster than the 0.4.0 file (1.24 vs 1.99 ms and 2.83 vs 3.78 ms), and the 0.5B case (806 tok/s steady) would pass the stretch target, but that file is 35% larger than the 0.4.0 one and a different quantization. So the 0.4.0 default layout (size_f routing plus int8 embedding) does not get the full fused-kernel speed.

Per-token GPU profile, 1.5B 0.4.0 file, graph mode (3792 us of GPU time per token, matches the 3.78 ms steady rate, so decode is GPU-bound, not launch-bound):
| # | Cost | us/token | share | launches/token |
|---|---|---|---|---|
| 1 | `_int4_gemv1_kernel` (all int4 projections) | 1738 | 45.8% | 111 |
| 2 | `_embed_gemv_kernel` (int8 g64 lm_head) | 319 | 8.4% | 1 |
| 3 | cuBLAS fp16 GEMV (about 2 matrices per layer not on the int4 kernel) | 310 | 8.2% | 60 |
| 4 | torch elementwise/copy kernels (add, copy, neg, cat, index_copy: eager RoPE and KV write, residual adds) | about 750 | about 20% | several hundred |
| 5 | `_add_rmsnorm_kernel` | 80 | 2.1% | 59 |
The fused RoPE plus KV-write kernel does not appear in this profile: `neg`, `cat` and `index_copy` kernels run about twice per layer, which is the unfused RoPE and cache write. Next steps in order: (a) find why about 2 matrices per layer go to cuBLAS fp16 GEMV and put them on the int4 kernel, (b) get the fused RoPE plus KV-cache write used for the 0.4.0 layout, (c) tune `_int4_gemv1_kernel` (1738 us for about 0.9 GB of weights is about 0.5 TB/s of traffic at the clocks above), (d) the lm_head GEMV.
Note (superseded by task 39.3, which deleted `scripts/profile_decode_graph.py`): that script could not run on the 0.4.0 file (its annotated eager copy reads `m.embed`, which is None for a quantized embedding), so `scripts/profile_decode_040.py` aggregates graph-replay kernels by name instead.

## Decode profile after the int8 fast path (task 39.3)

Script: `scripts/profile_decode_040.py --quf FILE --out eval_cache/profile393/NAME.json` (raw numbers in `eval_cache/profile393/`). It is now the only decode profiler: `scripts/profile_decode_graph.py` was deleted because its eager copy of `decode_step` read `m.embed`, which is None when `m.qembed` is set. The script works on any file, aggregates graph-replay kernels by name, and prints a Requirement 42.6 check. Prefill is excluded: it profiles `generate` for 64 new tokens and for 1 new token, subtracts the second from the first, and divides by the 63 graph-replayed decode tokens. Setup: RTX 5080 Laptop, Windows, triton-windows, `QUF_NO_TRITON` unset, backend `triton_int4`, prompt `The quick brown fox`, greedy, `fused_decode` true, files `eval_cache/lm040/q15.quf` and `q05.quf` (0.4.0 defaults). The GPU was shared with the small property-test run of task 39.4, so absolute times are a few percent noisy (an earlier unsubtracted q15 run measured 2347 µs/token with prefill included). Read the composition, not the absolute numbers.

### Requirement 42.6 check

| File | cuBLAS fp16 GEMV kernels | neg / cat / index_copy kernels in steady decode |
|---|---|---|
| q15.quf | none | none (one stray `CatArrayBatchedCopy` launch in all 63 tokens, 0.016 per token, 0.014 µs per token: a one-off from the 1-token baseline run, not a per-token kernel) |
| q05.quf | none | same one-off launch, 0.017 µs per token |

The int8 modules run on `_int8_gemv1_kernel`; the RoPE and KV-cache write run in `_rope_kv_kernel`. No module falls back to the unfused path, so there was no offending module to report. In the earlier run that included prefill, the cuBLAS `gemmSN_TN_kernel` (cublasGemv), `index_copy`, `CatArrayBatchedCopy` and one elementwise kernel (about 0.9 launches per token) came from the prefill pass only.

### Qwen2.5-1.5B (q15.quf): GPU time per token, graph mode

| Kernel | µs per token | share | launches per token |
|---|---|---|---|
| `_gemv_jmajor_kernel` (int4 decode GEMV) | 1482.8 | 58.5% | 113.1 |
| `_embed_gemv_kernel` (tied lm_head, quantized embedding) | 358.6 | 14.1% | 1.0 |
| `_int8_gemv1_kernel` | 265.4 | 10.5% | 60.8 |
| `_add_rmsnorm_kernel` | 164.9 | 6.5% | 59.7 |
| `_flash_decode_kernel` | 118.7 | 4.7% | 29.3 |
| `_silu_mul_scatter_kernel` | 88.0 | 3.5% | 29.3 |
| `_rope_kv_kernel` | 31.8 | 1.3% | 29.3 |
| 23 other kernels (argmax, token fill, copies) | 24.4 | 1.0% | 12.7 |
| Total | 2534.6 | 100% | 335.3 |

### Qwen2.5-0.5B (q05.quf): GPU time per token, graph mode

| Kernel | µs per token | share | launches per token |
|---|---|---|---|
| `_gemv_jmajor_kernel` (int4 decode GEMV) | 455.9 | 42.8% | 96.4 |
| `_embed_gemv_kernel` (tied lm_head, quantized embedding) | 191.0 | 17.9% | 1.0 |
| `_int8_gemv1_kernel` | 126.8 | 11.9% | 52.4 |
| `_add_rmsnorm_kernel` | 104.6 | 9.8% | 51.3 |
| `_flash_decode_kernel` | 77.4 | 7.3% | 25.1 |
| `_silu_mul_scatter_kernel` | 59.7 | 5.6% | 25.1 |
| `_rope_kv_kernel` | 26.7 | 2.5% | 25.1 |
| 23 other kernels (argmax, token fill, copies) | 22.5 | 2.1% | 12.7 |
| Total | 1064.7 | 100% | 289.2 |

### Top remaining costs

1.5B: (1) int4 GEMV 1482.8 µs (58.5%), (2) embed GEMV 358.6 µs (14.1%), (3) int8 GEMV 265.4 µs (10.5%), (4) add+RMSNorm 164.9 µs (6.5%), (5) flash decode 118.7 µs (4.7%).
0.5B: (1) int4 GEMV 455.9 µs (42.8%), (2) embed GEMV 191.0 µs (17.9%), (3) int8 GEMV 126.8 µs (11.9%), (4) add+RMSNorm 104.6 µs (9.8%), (5) flash decode 77.4 µs (7.3%).

`_int4_gemv1_kernel` (in `triton_kernels.py`) does not appear in either profile: 0 µs per token, 0 launches. The fused Fast_Path decodes int4 modules with `_gemv_jmajor_kernel` (`quf/runtime/kernels/decode_gemv.py`), and the older `_int4_gemv_kernel` appeared only in the prefill-inclusive run. The int4 decode GEMV that does cost time is therefore `_gemv_jmajor_kernel`: 58.5% of the 1.5B token and 42.8% of the 0.5B token, about 13.1 µs per launch on 1.5B and 4.7 µs on 0.5B. Task 39.5 should tune `_gemv_jmajor_kernel` (and, second, `_embed_gemv_kernel` and `_int8_gemv1_kernel`), not `_int4_gemv1_kernel`.

## j-major int4 GEMV tuning (task 39.5)

`_gemv_jmajor_kernel` was 58.5% of a 1.5B token and 42.8% of a 0.5B token (task 39.3). Per-shape configs
`(BLOCK_N, NGB, num_warps, num_stages, SPLIT)` now live in `_JMAJOR_CONFIGS` in
`quf/runtime/kernels/decode_gemv.py`. They were chosen by whole-graph steady decode time
(`scripts/_jmajor_lab.py graph`, configs alternated in one process, 5 to 7 repeats), not by isolated
microbenchmarks, on the freshly re-converted 0.4.0 default files (q05.quf 367,911,680 bytes,
q15.quf 1,040,011,648 bytes; sizes unchanged by the clip-search fix). RTX 5080 Laptop, power-capped.

| Shape (N x K) | Model | Before | After |
|---|---|---|---|
| 17920x1536 gate_up | 1.5B | 2,16,1,2 | 16,16,1,3 |
| 1536x8960 down | 1.5B | 1,32,2,2 | 16,32,2,3, split-K 2 (in-kernel reduce) |
| 1536x1536 q, o | 1.5B | 2,16,1,2 | 2,16,1,3 |
| 9728x896 gate_up | 0.5B | 2,16,1,2 | 16,32,2,3 |
| 896x4864 down | 0.5B | 1,32,2,2 | 16,32,2,3, split-K 4 |
| 896x896 q, o | 0.5B | 2,16,1,2 | 1,16,1,3 |

| Measurement | 0.5B before | 0.5B after | 1.5B before | 1.5B after |
|---|---|---|---|---|
| `_gemv_jmajor_kernel` GPU time per token (profile_decode_040) | 668.7 us (96.4 launches, 6.9 us each) | 663.4 us (6.9 us each) | 2320.3 us (113.1 launches, 20.5 us each) | 2046.6 us (18.1 us each) |
| Steady ms/token, interleaved lab graph run | 1.78 | 1.75 | 5.68 | 4.86 |
| Steady tok/s, interleaved lab graph run | 560 | 573 | 176 | 206 |
| Steady ms/token, `bench_speed040.py` (sequential runs) | 1.95 (513 tok/s) | 1.52 (658 tok/s) | 6.08 (165 tok/s) | 3.67 (273 tok/s) |

Caveats. The GPU clock state drifts between processes (the task-39.3 profile showed 13 us per launch,
this one 20 us before tuning), so the sequential `bench_speed040.py` before/after pair overstates the gain.
The interleaved lab numbers are the reliable ones: about 14% faster steady decode on 1.5B and about 2% on 0.5B
(within noise). On 0.5B the sweep found nothing clearly better than the default for gate_up and down; the
0.5B entries are kept because none was worse. Wider loads and loading scales/zeros once per group were not
implemented; the block-shape, warps, stages and split-K sweep was the whole search.

## Final decode speed, QUF 0.4.0 vs llama.cpp (task 39.5)

Power source during the run: **BATTERY (not on AC)** (PowerOnline=False, battery 71%). All numbers below are battery numbers.

One session, rows interleaved in rotated order, 2 warm-up rounds discarded, 7 measured rounds. QUF: triton_int4, CUDA graph, 128 new tokens, EOS suppressed, prompt "The quick brown fox". llama-bench: `-p 0 -n 128 -ngl 99 -r 1`. Compared figure is QUF steady tok/s (no prefill, no graph capture) against llama-bench tg128. Verdict: faster/slower when the min-max ranges do not overlap, otherwise equal within noise.

| Model | Engine | tok/s median | min | max | QUF e2e median | power W | SM MHz | QUF vs row |
|---|---|---|---|---|---|---|---|---|
| 0.5B | QUF 0.4.0 | 662.2 | 658.4 | 666.1 | 371.0 | 48.1 | 1447 | - |
| 0.5B | Q4_K_M | 494.2 | 458.5 | 510.0 | - | 38.5 | 1864 | faster (1.34x) |
| 0.5B | Q4_K_S | 496.2 | 485.7 | 538.1 | - | 35.8 | 1792 | faster (1.33x) |
| 0.5B | IQ4_XS-imat | 496.3 | 488.5 | 522.6 | - | 34.2 | 1796 | faster (1.33x) |
| 0.5B | Q5_K_M | 506.1 | 482.7 | 533.6 | - | 34.8 | 1804 | faster (1.31x) |
| 1.5B | QUF 0.4.0 | 282.8 | 263.0 | 285.2 | 200.3 | 43.9 | 1668 | - |
| 1.5B | Q4_K_M | 273.9 | 268.1 | 279.1 | - | 45.0 | 1507 | equal within noise (1.03x) |
| 1.5B | Q4_K_S | 281.9 | 270.3 | 286.5 | - | 39.8 | 1582 | equal within noise (1.00x) |
| 1.5B | IQ4_XS-imat | 273.5 | 270.4 | 278.4 | - | 37.8 | 1590 | equal within noise (1.03x) |
| 1.5B | Q5_K_M | 247.6 | 242.1 | 250.5 | - | 41.4 | 1541 | faster (1.14x) |

Targets: 0.5B QUF >= Q4_K_M: **met**; 1.5B QUF >= 398 tok/s: **not met**; 1.5B QUF vs Q4_K_M: equal within noise.
