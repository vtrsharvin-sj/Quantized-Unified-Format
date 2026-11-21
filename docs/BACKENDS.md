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
