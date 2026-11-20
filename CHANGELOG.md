# Changelog

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
