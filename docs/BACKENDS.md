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
