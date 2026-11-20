# Environment results

Produced with `python scripts/env_spike.py` (Requirement 2) and the real-model runs in [RESULTS.md](RESULTS.md). Statuses: **verified** (ran and passed), **failed** (ran, failure recorded), **not verified** (not run, reason given).

Reference hardware: RTX 5080 Laptop GPU, compute capability 12.0 (sm_120), 16303 MiB, driver 621.14.

## Windows (native)

QUF runs natively on Windows. No WSL or Linux environment is used.

| Item | Result |
| --- | --- |
| OS / Python | Windows 11 (10.0.26300), Python 3.13.0rc2 |
| CUDA toolkit (`nvcc`) | Not installed. Not needed for PyTorch wheels. |
| PyTorch build | 2.11.0+cu128 (CUDA 12.8 runtime). Arch list includes `sm_120`. |
| transformers / datasets / tokenizers | 5.23.0 / 5.2.0 / 0.23.3 |
| gigatoken | 0.11.0, installed and working |

### PyTorch CUDA kernels on compute capability 12.0
**Verified: pass.** An fp16 256x256 matmul executed on the GPU and returned a finite result. `quf run` and `quf bench` also ran on CUDA with a converted Qwen2.5-0.5B.

### gigatoken on Windows
**Verified: pass.** gigatoken 0.11.0 installs with pip and runs on native Windows (`gt.Tokenizer(hf_tokenizer)`, `encode(text)`). On the WikiText-2 train split it produced the same token ids as Hugging Face `tokenizers` and was 36.8x faster in one measurement (0.23 s vs 8.44 s). See [RESULTS.md](RESULTS.md). The Hugging Face `tokenizers` fallback remains available when gigatoken is not installed.

### Marlin on compute capability 12.0
**Not verified.** No Marlin kernel has run on this device. Marlin is an optional backend, is never selected automatically, and is not needed for conversion or normal use.

## Backend designation

**`torch_gpu` is the GPU backend for this device.** It was measured at 20.71 tokens/s on Qwen2.5-0.5B, against 9.08 for `torch_fused` and 9.61 on CPU (see [RESULTS.md](RESULTS.md)).

## Backend selection

`quf run` and `quf bench` choose the backend with `--device {auto,cpu,cuda}` and `--backend {auto,cached,fused,marlin}`.

- `auto` selects CPU when no NVIDIA GPU is present. With a GPU it selects the cached backend (`torch_gpu`, fp16 weights dequantized at load) when those weights fit in free GPU memory. It selects the fused backend (`torch_fused`, packed int4 kept on the GPU) when the cached weights do not fit, or when `--prefer-fused` is given. If the fused self-test or load fails, the error is logged and the cached backend is used.
- `--backend cached` and `--backend fused` force that GPU backend.
- `--backend marlin` forces the optional Marlin backend. It needs an extra package that is not part of any install extra. Without it, the command exits with status 1 and an error naming the missing package.
