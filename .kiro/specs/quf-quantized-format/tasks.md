# Implementation Plan: QUF Quantized Unified Format

## Overview

The implementation language is Python (as used in the design). Property-based tests use Hypothesis with at least 100 examples per property, and each test carries the tag `Feature: quf-quantized-format, Property N: <property text>`. Work proceeds in ten stages. Each stage ends in something runnable, and later stages build on earlier ones. Marlin on compute capability 12.0 is unverified until the spike in task 1, and Torch_GPU is the fallback GPU Backend. Qwen2ForCausalLM is supported first, with Qwen3ForCausalLM added through the architecture allow-list.

## Tasks

- [x] 1. Package scaffold, device detection, and environment spike
  - Demo: `python -c "import quf; print(quf.__version__)"` works with no extras installed, `quf.device.detect_devices()` prints the RTX 5080 details, and docs/ENV.md records the spike results.
  - [x] 1.1 Create the package scaffold
    - Create `pyproject.toml` with the `quf` package, the `quf` console script, and extras `cuda` and `convert`
    - Create `quf/__init__.py` with `__version__` and lazy optional imports
    - Create `tests/conftest.py` with the Hypothesis profile (`max_examples >= 100`) and the `gpu` and `convert` markers, where `gpu` tests skip when no NVIDIA GPU is available
    - _Requirements: 1.1, 1.2, 1.5, 13.1, 13.6_

  - [x] 1.2 Implement device detection
    - Create `quf/device.py` with `DeviceInfo` and `detect_devices()`, importing `torch` inside the function
    - Return `DeviceInfo(False, None, None, None)` when torch is missing or CUDA is unavailable
    - _Requirements: 1.3, 1.4_

  - [x]* 1.3 Write unit tests for the package and device detection
    - Test `__version__`, declared extras, and import with extras absent
    - Test `detect_devices` with mocked `torch.cuda` for the GPU and no-GPU cases
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.4 Implement the environment spike script and docs/ENV.md
    - Create `scripts/env_spike.py` that reports CUDA toolkit, PyTorch, GPTQModel, and gigatoken versions, runs a PyTorch CUDA kernel, a Marlin matmul, and a gigatoken tokenize call
    - Write docs/ENV.md with the results for Windows and WSL, including the failure message and the Torch_GPU designation if Marlin fails on compute capability 12.0
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2. QUF v1 format: writer, reader, and validation
  - Demo: write a small tensor set to a `.quf` file, read it back bit-identically, and see a clear error for each kind of corrupted file.
  - [x] 2.1 Define constants, header schema, and errors
    - Create `quf/format/constants.py` (magic `QUF1`, version 1, align 64, dtype codes)
    - Create `quf/format/schema.py` with `QUFHeader`, `TensorEntry`, and JSON (de)serialization with required-field checks
    - Define `QUFFormatError` and its subclasses `InvalidMagicError`, `UnsupportedVersionError`, `TruncatedHeaderError`, `TensorBoundsError`, `HeaderSchemaError`
    - _Requirements: 3.2, 3.3, 3.7, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 2.2 Implement the Writer
    - Create `quf/format/writer.py` with `write_quf(path, header, tensors)`
    - Write magic, version, header length, JSON header, zero padding, and tensors at absolute offsets that are multiples of 64
    - Store quantized weights as int32 Packed_Int4 with fp16 scales, include qzeros when asymmetric, and keep embedding and lm_head as fp16
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 2.3 Implement the Reader
    - Create `quf/format/reader.py` with `QUFReader` that validates in this order: magic, version, header length, JSON parse, required fields, per-tensor alignment and bounds
    - Return the parsed header and zero-copy memory-mapped tensor views, opening the file read-only
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 9.5_

  - [x]* 2.4 Write property test for the written file layout
    - **Property 1: Written file layout invariant**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 7.3, 7.4**

  - [x]* 2.5 Write property test for the write/read round trip
    - **Property 2: Write/read round trip**
    - **Validates: Requirements 3.5, 4.1, 4.7**

  - [x]* 2.6 Write property test for corrupted file rejection
    - **Property 3: Reader rejects corrupted files**
    - **Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6**

  - [x] 2.7 Write FORMAT.md
    - Specify the byte layout, header fields, tensor naming, dtypes, and the canonical Packed_Int4 order (value `i` in bits `4*(i%8)`, least significant first)
    - _Requirements: 3.8, 5.2_

- [x] 3. 4-bit group quantizer, packer, and reference dequantizer
  - Demo: quantize a random 512x256 matrix, pack it, dequantize it, and print the maximum error against half the group scale.
  - [x] 3.1 Implement pack and unpack
    - Create `quf/quant/pack.py` with `pack_int4` and `unpack_int4` using the canonical order (8 values per int32)
    - _Requirements: 5.2_

  - [x] 3.2 Implement the quantizer and Reference_Dequantizer
    - Create `quf/quant/quantize.py` with `QuantizedWeight`, `GroupSizeError`, and `quantize_groupwise` (deterministic round-to-nearest, group size 128, symmetric or asymmetric)
    - Create `quf/quant/dequant.py` with `reference_dequantize` returning fp16 of the original shape, using `w = (q - zero) * scale` with `zero = 8` for symmetric schemes
    - _Requirements: 5.1, 5.4, 5.6_

  - [x]* 3.3 Write property test for pack/unpack
    - **Property 4: Pack/unpack round trip**
    - **Validates: Requirements 5.2, 5.3**

  - [x]* 3.4 Write property test for quantizer output shape and range
    - **Property 5: Quantizer output shape and range**
    - **Validates: Requirements 5.1, 5.4**

  - [x]* 3.5 Write property test for the quantization error bound
    - **Property 6: Quantization error bound**
    - **Validates: Requirements 5.5**

  - [x]* 3.6 Write property test for group-size violations
    - **Property 7: Group-size violations rejected**
    - **Validates: Requirements 5.6**

- [x] 4. Tokenizer wrapper and calibration builder
  - Demo: encode and decode a sentence with gigatoken (or the logged HF fallback) and build 4 reproducible calibration samples of length 512.
  - [x] 4.1 Implement TokenizerWrapper
    - Create `quf/tokenization/wrapper.py` using gigatoken, falling back to Hugging Face `tokenizers` and logging the reason when gigatoken cannot be imported
    - _Requirements: 6.1, 6.2_

  - [x] 4.2 Implement the Calibration_Builder
    - Create `quf/tokenization/calibration.py` with `build_calibration` (concatenate tokens, shuffle documents with `random.Random(seed)`, slice non-overlapping windows) and `CalibrationShortfallError`
    - _Requirements: 6.4, 6.5, 6.6_

  - [x]* 4.3 Write property test for the tokenizer round trip
    - **Property 8: Tokenizer round trip**
    - **Validates: Requirements 6.3**

  - [x]* 4.4 Write property test for calibration shape, determinism, and shortfall
    - **Property 9: Calibration shape, determinism, and shortfall**
    - **Validates: Requirements 6.4, 6.5, 6.6**

  - [x]* 4.5 Write unit test for the tokenizer fallback
    - Mock the gigatoken import to fail and assert the fallback and the logged reason
    - _Requirements: 6.2_

- [x] 5. Calibrated conversion with GPTQModel and the `info` and `convert` commands
  - Demo: in WSL, run `quf convert Qwen/Qwen2.5-0.5B out.quf`, see the size and ratio printed, then run `quf info out.quf`.
  - [x] 5.1 Implement GPTQModel invocation and tensor mapping
    - Create `quf/convert/gptq.py` that runs GPTQModel at 4 bits and group size 128 and maps `qweight`, `scales`, and `qzeros` to the canonical Packed_Int4 layout
    - Raise an error naming `quf[convert]` when GPTQModel is missing
    - _Requirements: 7.1, 7.2_

  - [x] 5.2 Implement the Converter
    - Create `quf/convert/converter.py` with `convert_checkpoint` and `ConvertResult`
    - Check the output path and the architecture allow-list (`Qwen2ForCausalLM`, extensible to `Qwen3ForCausalLM`) before heavy work
    - Store model config, tokenizer info, and fp16 embedding and lm_head in the header, and delete the partial file on failure after creation
    - Define `UnsupportedArchitectureError` and `OutputExistsError`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

  - [x] 5.3 Implement the CLI skeleton with `info` and `convert`
    - Create `quf/cli.py` with `main(argv) -> int`, subparsers, `--help`, and error mapping to exit status 1
    - Implement `quf info` (model config, quantization scheme, tensor count, file size) and `quf convert` (dataset, sample count, sequence length, seed, overwrite options)
    - _Requirements: 7.1, 7.2, 7.5, 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x]* 5.4 Write property test for output preservation
    - **Property 15: Existing output is preserved**
    - **Validates: Requirements 7.7**

  - [x]* 5.5 Write property test for `info` output completeness
    - **Property 16: Info output completeness**
    - **Validates: Requirements 12.2**

  - [x]* 5.6 Write unit tests for the Converter
    - Use a stubbed GPTQModel to test size and ratio output, non-Qwen rejection, and partial-file cleanup with an injected failure
    - _Requirements: 7.5, 7.6, 7.8_

- [x] 6. CPU runtime and generation
  - Demo: `quf run tiny.quf --device cpu --prompt "Hello" --max-new-tokens 16 --temperature 0` streams text on a machine without a GPU.
  - [x] 6.1 Implement the Backend protocol, CPU Backend, and QUFLinear
    - Create `quf/runtime/backends/base.py`, `quf/runtime/backends/cpu.py`, and `quf/runtime/linear.py`
    - The CPU Backend dequantizes with the Reference_Dequantizer and multiplies in PyTorch
    - _Requirements: 8.2_

  - [x] 6.2 Implement the Qwen2 decoder and KV_Cache
    - Create `quf/runtime/model.py` with `KVCache`, and `QUFModel.load` and `forward` (RMSNorm, rotary embeddings, grouped-query attention, SwiGLU, all projections through QUFLinear)
    - _Requirements: 8.1, 8.3, 8.10_

  - [x] 6.3 Implement sampling and the generation loop
    - Create `quf/runtime/generate.py` with `generate`, `ContextLengthError`, seeded `torch.Generator` sampling, greedy decoding at temperature 0, top-k, top-p, streaming callback, and EOS stop
    - _Requirements: 8.1, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9_

  - [x] 6.4 Add the `run` command to the CLI
    - Add `quf run` with options for maximum new tokens, temperature, top-k, top-p, seed, and streaming
    - _Requirements: 8.1, 8.4, 8.5, 8.8, 12.1_

  - [x]* 6.5 Write property test for backend output versus the reference (CPU)
    - **Property 10: Backend output matches the reference**
    - Parametrize over available Backends, with the CPU variant here and GPU variants added in task 7
    - **Validates: Requirements 8.2, 9.2, 9.4**

  - [x]* 6.6 Write property test for KV_Cache equivalence
    - **Property 11: KV_Cache equivalence**
    - **Validates: Requirements 8.3, 8.10**

  - [x]* 6.7 Write property test for sampling determinism and greedy decoding
    - **Property 12: Sampling determinism and greedy decoding**
    - **Validates: Requirements 8.6, 8.7**

  - [x]* 6.8 Write property test for the context length check
    - **Property 13: Context length check**
    - **Validates: Requirements 8.8**

  - [x]* 6.9 Write unit tests for the runtime
    - Test end-to-end `quf run` on a tiny random Qwen-shaped QUF_File, streaming, option parsing, and EOS stop
    - _Requirements: 8.1, 8.4, 8.5, 8.9_

- [x] 7. CUDA backends: Torch_GPU and Marlin
  - Demo: on the RTX 5080, `quf run` generates tokens through Torch_GPU, and the Marlin self-test reports pass or fail with its message.
  - [x] 7.1 Implement the Torch_GPU Backend
    - Create `quf/runtime/backends/torch_gpu.py` that dequantizes with the Reference_Dequantizer on the GPU and multiplies on the GPU
    - _Requirements: 9.3, 9.4_

  - [x] 7.2 Implement the Marlin Backend, self-test, and GPU out-of-memory handling
    - Create `quf/runtime/backends/marlin.py` that repacks canonical Packed_Int4 into the Marlin layout in memory only and provides a Marlin self-test comparing against the reference result
    - In `quf/runtime/model.py`, catch CUDA out-of-memory at load and raise `GPUMemoryError(required, available)` with required bytes computed from the Tensor_Table plus a KV_Cache estimate
    - _Requirements: 9.1, 9.2, 9.5, 9.6_

  - [x]* 7.3 Extend the Property 10 test with GPU variants
    - **Property 10: Backend output matches the reference**
    - Add Torch_GPU and Marlin variants marked `gpu`
    - **Validates: Requirements 8.2, 9.2, 9.4**

  - [x]* 7.4 Write GPU unit and integration tests
    - Test the Marlin repack at load, the unchanged file hash after load, and the out-of-memory message with a mocked allocator
    - _Requirements: 9.1, 9.5, 9.6_

- [x] 8. Automatic backend selection
  - Demo: `quf run --device auto` logs the selected Backend and reason, and `--device cuda` on a machine without a GPU exits with "no NVIDIA GPU was found".
  - [x] 8.1 Implement the Backend_Selector
    - Create `quf/runtime/selector.py` with the pure function `select_backend`, `BackendChoice`, and `NoGPUError`
    - Call the Marlin self-test only when a GPU is present and compute capability is 8.0 or higher, and treat an exception as failure
    - _Requirements: 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [x] 8.2 Wire selection into model loading and the CLI
    - Add `--device {auto,cpu,cuda}` to `run` and `bench`, log the selected Backend and reason, and load the chosen Backend in `QUFModel.load`
    - _Requirements: 10.1, 10.8_

  - [x]* 8.3 Write property test for backend selection rules
    - **Property 14: Backend selection rules**
    - **Validates: Requirements 10.2, 10.3, 10.4, 10.5, 10.6, 10.7**

  - [x]* 8.4 Write unit tests for selection logging and the CLI option
    - Test the log message, accepted `--device` values, and the nonzero exit for `cuda` without a GPU
    - _Requirements: 10.1, 10.7, 10.8_

- [x] 9. Benchmarks and accuracy evaluation
  - Demo: `quf bench model.quf --out results.json` writes WikiText perplexity, file size, tokens per second, Backend, device name, and software versions.
  - [x] 9.1 Implement the Benchmark_Suite
    - Create `quf/bench/suite.py` with `perplexity`, `run_bench`, `BenchResult`, `DatasetLoadError`, optional fp16 reference and GGUF Q4 comparison, and JSON output
    - Report the relative perplexity increase next to the documented 3 percent target without asserting it
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.6, 11.7_

  - [x] 9.2 Implement the lm-evaluation-harness adapter
    - Create `quf/bench/lm_eval_adapter.py` subclassing the harness `LM` base with `loglikelihood`, `loglikelihood_rolling`, and `generate_until`
    - _Requirements: 11.5_

  - [x] 9.3 Add the `bench` command to the CLI
    - Add `quf bench` with options for the fp16 reference, the GGUF file, the output JSON path, and `--device`
    - _Requirements: 11.1, 11.6, 12.1_

  - [x]* 9.4 Write property test for the benchmark results JSON round trip
    - **Property 17: Benchmark results JSON round trip**
    - **Validates: Requirements 11.6**

  - [x]* 9.5 Write unit tests for the Benchmark_Suite
    - Use a stub model to test reported fields and the dataset load failure message
    - _Requirements: 11.1, 11.2, 11.3, 11.7_

- [x] 10. Packaging, documentation, and CI
  - Demo: `pip install .` followed by `pytest -m "not gpu and not convert"` passes on a CPU-only machine, and the CI workflow builds a wheel.
  - [x] 10.1 Write the README
    - Describe installation (including `quf[cuda]` and `quf[convert]`), conversion, running, benchmarking, and the 3 percent perplexity target
    - Confirm docs/ENV.md and FORMAT.md are included in the package
    - _Requirements: 13.1, 13.2, 13.3_

  - [x] 10.2 Create the CI workflow
    - Create `.github/workflows/ci.yml` that builds a wheel and runs `pytest -m "not gpu and not convert"` on a CPU runner
    - _Requirements: 13.4, 13.5, 13.6_

  - [x]* 10.3 Write CLI unit tests
    - Test the four subcommands, `--help` exiting with status 0, the unknown subcommand listing valid subcommands, and `info` on a corrupt file
    - _Requirements: 12.1, 12.3, 12.4, 12.5_

  - [x] 10.4 Final checkpoint - Ensure all tests pass
    - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Native GPTQ quantizer and converter (Windows-native, no GPTQModel)
  - Demo: `quf convert Qwen/Qwen2.5-0.5B out.quf` runs on native Windows using transformers and gigatoken-tokenized calibration, then `quf info out.quf` reads the result.
  - [x] 11.1 Implement the Native_GPTQ core
    - Create `quf/quant/gptq.py` with `HessianAccumulator`, `gptq_quantize` (damping, group size 128, asymmetric and symmetric, act-order off, blockwise error propagation, Cholesky retry with `GPTQError`), and device choice (CUDA if available, else CPU)
    - Return a `QuantizedWeight` in the canonical Packed_Int4 layout
    - _Requirements: 14.1, 14.2, 14.3, 14.8, 14.9_

  - [x] 11.2 Implement sequential model quantization
    - Add `quantize_model_sequential` that captures layer inputs, fills Hessians with forward hooks, quantizes each decoder layer's linears in groups, and feeds quantized outputs to the next layer
    - _Requirements: 14.4_

  - [x] 11.3 Replace GPTQModel in the Converter and packaging extras
    - Update `quf/convert/converter.py` to load with transformers, build calibration with `build_calibration`, and call `quantize_model_sequential`
    - Remove `quf/convert/gptq.py` GPTQModel usage and drop GPTQModel from the `convert` extra in `pyproject.toml` (list transformers, datasets, accelerate, gigatoken)
    - Update existing converter unit tests that stub GPTQModel to stub the native quantizer
    - _Requirements: 14.4, 14.5, 14.6_

  - [x] 11.4 Write property test for GPTQ versus round-to-nearest
    - **Property 18: GPTQ error is no worse than round-to-nearest**
    - **Validates: Requirements 14.7**

  - [x] 11.5 Write property test for GPTQ output shape, range, and layout
    - **Property 19: GPTQ output shape, range, and layout**
    - **Validates: Requirements 14.1, 14.2, 14.8**

  - [x] 11.6 Write unit tests for the Native_GPTQ
    - Test the Hessian accumulator against `2 X^T X / n`, the Cholesky retry path, and sequential quantization of a tiny random Qwen-shaped model written through the Converter and read by `quf info`
    - _Requirements: 14.4, 14.5, 14.9_

- [x] 12. Windows-native GPU backend and real-model verification
  - Demo: `quf run` and `quf bench` on a converted Qwen2.5-0.5B work on CPU and on the RTX 5080, with real numbers in docs/RESULTS.md.
  - [x] 12.1 Implement the TorchGPUFused backend
    - Create `quf/runtime/backends/torch_fused.py` that keeps packed int32, scales, and zeros on the GPU and unpacks, dequantizes, and multiplies in one function, optionally wrapped in `torch.compile` and falling back to eager when compile or Triton is unavailable
    - Leave `torch_gpu.py` output unchanged
    - _Requirements: 15.2, 15.3, 15.6_

  - [x] 12.2 Update the Backend_Selector, CLI, and Marlin handling
    - Extend `select_backend` with the memory estimate, the `prefer_fused` flag, and fallback to the cached backend when the fused load or self-test fails
    - Make Marlin a lazily imported optional module selectable only with `--backend marlin`, with an error naming the optional package when it is missing
    - Update existing selector tests and docs/ENV.md accordingly
    - _Requirements: 15.1, 15.4, 15.5, 15.7_

  - [x] 12.3 Write property test for the fused backend
    - **Property 20: Fused backend matches the reference**
    - Eager variant on CPU, plus a `gpu`-marked CUDA variant
    - **Validates: Requirements 15.2, 15.3, 15.6**

  - [x] 12.4 Write property test for extended backend selection
    - **Property 21: Extended backend selection rules**
    - **Validates: Requirements 15.1, 15.4, 15.5, 15.7**

  - [x] 12.5 Measure backends on the RTX 5080
    - Create `scripts/bench_backends.py` that measures decode tokens per second for the cached and fused backends, run it on the RTX 5080 (sm_120), confirm the Torch_GPU Property 10 test still passes, and record the numbers and the selector decision
    - If the fused path is not faster, keep the cached backend as the default and say so in the results
    - _Requirements: 15.4, 15.6_

  - [x] 12.6 Run the real-model verification
    - Create `scripts/verify_real_model.py`, convert Qwen/Qwen2.5-0.5B, run `quf run` on CPU and CUDA with a fixed prompt and seed, and run `quf bench` for WikiText-2 perplexity against fp16, file size, and tokens per second
    - Verify gigatoken installs and works on Windows, or record the fallback
    - Repeat with Qwen/Qwen2.5-1.5B if time permits
    - Write docs/RESULTS.md and add the results table to the README
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_

- [x] 13. PyPI packaging for native Windows
  - Demo: a wheel built from the repository installs into a fresh venv, and `quf --help` and `quf info` run.
  - [x] 13.1 Update pyproject metadata and add LICENSE
    - Move `torch` to base dependencies, add the MIT license and `LICENSE` file, set the author to vtrsharvin-sj, add `[project.urls]` for https://github.com/vtrsharvin-sj/Quantized-Model-Format, add classifiers, and remove the `cuda` extra (or leave it empty) without bumping the version
    - Add a unit test that reads `pyproject.toml` and checks these fields
    - _Requirements: 17.1, 17.2_

  - [x] 13.2 Update the README and docs for Windows-native use
    - State that QUF runs natively on Windows, remove all WSL instructions and GPTQModel references from the README, docs/ENV.md, and CI notes, and describe the native converter and backends
    - _Requirements: 17.3_

  - [x] 13.3 Build and smoke test the package
    - Build the sdist and wheel with `python -m build`, run `twine check dist/*`, install the wheel into a fresh venv, and run `quf --help` and `quf info` on a small QUF_File
    - Do not upload to PyPI
    - _Requirements: 17.4, 17.5, 17.6_

- [x] 14. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Accuracy diagnosis
  - Demo: `python scripts/diagnose_accuracy.py` prints a ranked table of per-module quantization error and the perplexity of each ablation subset for Qwen2.5-0.5B, and docs/ACCURACY.md records the findings.
  - [x] 15.1 Implement the Sensitivity_Analyzer and Ablation_Harness
    - Create `quf/analysis/__init__.py` and `quf/analysis/sensitivity.py` with `ModuleError`, `module_errors`, `ablate`, and `standard_subsets`
    - `ablate` quantizes only the named modules, keeps all others in fp16, and calls the existing `perplexity`
    - _Requirements: 18.1, 18.2, 18.3_

  - [x] 15.2 Write tests for the analyzer and harness
    - **Property 22: Ablation changes only the selected modules**
    - Unit test that `module_errors` returns one entry per decoder linear module on a tiny random Qwen-shaped model
    - **Validates: Requirements 18.1, 18.2**

  - [x] 15.3 Audit `quf/quant/gptq.py` and fix defects
    - Check Hessian accumulation, damping and dead columns, calibration length, sequential and true-sequential ordering, fp16 scale rounding before error propagation, and tied embedding handling
    - Add a unit test for each checked item, and a regression test for each defect found
    - Record perplexity before and after each fix on Qwen2.5-0.5B
    - _Requirements: 18.5, 18.6_

  - [x] 15.4 Run the diagnosis on Qwen2.5-0.5B and write docs/ACCURACY.md
    - Create `scripts/diagnose_accuracy.py` that runs `module_errors` and `ablate` on first layer, last layer, all `v_proj`, all `down_proj`, remaining linears, and reports whether the tied embedding is fp16 in the converted file
    - Write docs/ACCURACY.md with the audit result, the tables, the modules ranked by perplexity impact, and the Upgrades selected from the ranking
    - _Requirements: 18.3, 18.4, 18.5, 18.7_

- [x] 16. Quantizer upgrades and format 2
  - Demo: `quf convert Qwen/Qwen2.5-0.5B out.quf --group-size 64 --act-order --clip-search --mixed "lm_head=fp16"` writes a version 2 file that `quf info` and `quf run` read on cpu and cuda, and a default-settings conversion without upgrades still writes a version 1 file.
  - [x] 16.1 Extend the format for group size, g_idx, and int8 and fp16 modules
    - Add the `int8` dtype and version 2 to `quf/format/constants.py`, `group_size`, `act_order`, and `modules` fields to `schema.py`, and `required_version` to the Writer
    - Update the Reader to read versions 1 and 2, default a missing group size to 128, and validate group size, g_idx range and length, and module precision strings
    - Update FORMAT.md with the version 2 additions
    - Add a version 1 fixture written by 0.1.0 and a test that the Reader reads it as group size 128
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.7, 19.8, 20.2, 20.6_

  - [x] 16.2 Add group size, MSE clipping, g_idx dequantization, and int8 to the quantizer
    - Update `quantize_groupwise` and `GroupSizeError` for group sizes 32, 64, 128 and remove the fixed 128 checks in the writer and reader paths
    - Create `quf/quant/clip.py` with the clipping-ratio grid that includes 1.0, and add the `clip_search` option
    - Update `reference_dequantize` to take `g_idx`, and add `quantize_int8` and `dequantize_int8`
    - _Requirements: 19.1, 19.6, 20.4, 21.1, 21.2, 21.4, 22.3_

  - [x] 16.3 Add act-order, group size, and clipping to Native_GPTQ
    - Update `gptq_quantize` and `quantize_model_sequential` with `group_size`, `act_order`, and `clip_search`, returning `g_idx` and keeping stored weights in original column order
    - Add a helper that returns the act-order permutation and g_idx from the Hessian diagonal
    - _Requirements: 20.1, 20.2, 21.3, 21.4_

  - [x] 16.4 Support g_idx, group sizes, int8, and fp16 modules in all backends
    - Add the shared `expand_groups` helper and update `QUFLinear.prepare` and the CPU, Torch_GPU cached, and Torch_GPU_Fused backends
    - Keep Marlin explicit-only and have it raise a clear error for g_idx and int8 modules
    - Keep Torch_GPU output for version 1 files unchanged
    - _Requirements: 20.4, 20.5, 22.3, 22.4_

  - [x] 16.5 Implement mixed precision in the Converter and `info`
    - Create `PrecisionPlan` with glob patterns, add `--mixed`, `--group-size`, `--act-order`, and `--clip-search` to `quf convert`, and write the module precision map
    - Create `SizeReport`, print it at conversion and in `quf info`
    - _Requirements: 19.1, 22.1, 22.2, 22.5, 22.6_

  - [x] 16.6 Implement multi-source calibration and new defaults
    - Add `CalibrationSource` and multi-source `build_calibration` with per-source shares and the shortfall error naming the source
    - Set the default to 128 samples of length 2048 from wikitext-2 train and one second English text source, add `--legacy-calibration` for the 0.1.0 settings, and document the sources
    - _Requirements: 23.1, 23.2, 23.3, 23.4_

  - [x] 16.7 Write property and unit tests for the new format features
    - **Property 23: Header group size and version round trip** (includes the g_idx round trip)
    - **Property 24: Group-size and g_idx validation**
    - **Property 25: Backend output matches the reference for all precisions** (cpu and torch_fused eager here, `gpu` variants for torch_gpu and torch_fused on CUDA)
    - **Property 26: MSE clipping is no worse than round-to-nearest and keeps the layout**
    - **Property 27: Act-order processes columns in descending Hessian diagonal**
    - **Property 28: Size accounting equals tensor bytes**
    - **Property 29: Multi-source calibration shape, shares, and determinism**
    - Unit tests for the 0.1.0 fixture, the default of 128 by 2048, and the module-pattern selection
    - **Validates: Requirements 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 21.1, 21.2, 21.4, 22.1, 22.3, 22.4, 22.5, 22.6, 23.2, 23.3, 23.4**

  - [x] 16.8 Evaluate each Upgrade by perplexity and set defaults
    - Create `scripts/eval_upgrades.py` that converts Qwen2.5-0.5B with the 0.1.0 settings plus one Upgrade at a time (act-order, clip search, group size 64 and 32, mixed precision from the diagnosis, new calibration) and then the combination of the kept Upgrades, and measures WikiText-2 perplexity and file size with all other settings fixed
    - Set each Upgrade default on only if it lowered perplexity, record both values for every Upgrade in docs/ACCURACY.md, and record the reason when an Upgrade stays off
    - _Requirements: 23.5, 24.1, 24.2_

- [x] 17. Evaluation and results
  - Demo: `python scripts/bench_accuracy.py` writes JSON and docs/RESULTS.md shows perplexity, size, and tokens per second for Qwen2.5-0.5B and Qwen2.5-1.5B under fp16, 0.1.0 settings, 0.2.0 settings, and RTN.
  - [x] 17.1 Implement the evaluation matrix script
    - Create `scripts/bench_accuracy.py` that converts and benchmarks fp16 reference, 0.1.0 settings, 0.2.0 defaults, and the RTN_Baseline through the Converter, and writes JSON with device names and software versions
    - Add the GGUF_Baseline step with Q4_K_M and Q4_0 through llama-cpp-python only if a prebuilt wheel installs on Windows, and otherwise record "not run" with the pip error
    - _Requirements: 24.3, 24.4_

  - [x] 17.2 Run the matrix on Qwen2.5-0.5B
    - Record perplexity, file size, tokens per second, relative perplexity increase against fp16, and whether the 3 percent target was reached
    - _Requirements: 24.3, 24.6_

  - [x] 17.3 Convert and run the matrix on Qwen2.5-1.5B
    - Convert Qwen2.5-1.5B with the 0.1.0 settings, the 0.2.0 defaults, and RTN, and record the same fields
    - _Requirements: 24.3, 24.6_

  - [x] 17.4 Run lm-evaluation-harness on one or two tasks if available
    - Use the existing adapter on fp16 and the 0.2.0 default file only if lm-eval is installed or installs with one command, and otherwise record that it was not run and why
    - _Requirements: 24.5_

  - [x] 17.5 Update README and docs/RESULTS.md
    - Replace or extend the results tables with the measured numbers only, state the 3 percent target as reached or not reached, and document the new options
    - _Requirements: 24.3, 24.4, 24.5, 24.6_

- [x] 18. Release 0.2.0
  - Demo: a wheel built from the repository installs into a fresh venv and `quf --help` and `quf info` run on a version 1 file and a version 2 file.
  - [x] 18.1 Bump version and write the changelog
    - Set version 0.2.0 in `pyproject.toml` and `quf/__init__.py`, create CHANGELOG.md with the Upgrades kept, the measured perplexity change, and the format version note, and add a unit test that the two version strings match
    - _Requirements: 25.1, 25.2_

  - [x] 18.2 Run the full test suite
    - Run `pytest -m "not gpu and not convert"` and the `gpu` tests on the RTX 5080, and fix failures
    - _Requirements: 25.5_

  - [x] 18.3 Build, check, and smoke test the package
    - Remove old `dist/`, run `python -m build`, run `twine check dist/*`, install the wheel into a fresh venv, and run `quf --help` and `quf info` on a version 1 file and a version 2 file
    - Do not upload to PyPI and do not write any token to disk
    - _Requirements: 25.3, 25.4, 25.6_

- [x] 19. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 20. GGUF comparison (verified)
  - Demo: `python scripts/bench_accuracy.py --protocol llama` reports GGUF perplexity matching llama-perplexity.exe.
  - [x] 20.1 Validate the GGUF path against llama-perplexity.exe
    - Download the llama.cpp Windows release (CUDA if available), run it on the same text and context, and add an f16 or Q8_0 row that must be within 0.5% of fp16
    - _Requirements: 26.1, 26.2, 26.5_

  - [x] 20.2 Fix the discrepancy
    - Check BOS handling, context and batch settings, second-half scoring, logits_all, and the GGUF tokenizer, then add the `--protocol` switch
    - _Requirements: 26.1, 26.3_

  - [x] 20.3 Re-run Q4_K_M, Q4_0, and Q8_0 for 0.5B and 1.5B
    - Use one Eval_Protocol for QUF and GGUF, and record file size and GPU tokens per second
    - _Requirements: 26.3, 26.4_

- [x] 21. Batched lm-evaluation-harness adapter
  - Demo: hellaswag and arc_easy run on Qwen2.5-0.5B in minutes and report accuracy with stderr.
  - [x] 21.1 Batch loglikelihood requests
    - Pad batches, reuse repeated contexts, and run on GPU in `quf/bench/lm_eval_adapter.py`
    - _Requirements: 27.1, 27.2_

  - [x] 21.2 Write property test for batched equivalence
    - **Property 30: Batched log-likelihood equals single-request log-likelihood**
    - **Validates: Requirements 27.1, 27.2, 27.3**

  - [x] 21.3 Run hellaswag and arc_easy
    - Use limit 500 and a fixed seed for fp16 and QUF defaults on 0.5B, and on 1.5B if under about 20 minutes, and record accuracy with stderr
    - _Requirements: 27.4_

- [x] 22. Decode speed on 0.5B
  - Demo: the new defaults decode at a speed consistent with the other 0.5B settings.
  - [x] 22.1 Profile g_idx and int8 decode
    - Profile torch_gpu and cpu for re-gathering or per-call dequantization
    - _Requirements: 28.2_

  - [x] 22.2 Cache fp16 weights and add a regression test
    - Fix any real slowdown and test that the prepared torch_gpu weight for g_idx and int8 modules is a cached fp16 tensor
    - _Requirements: 28.2_

  - [x] 22.3 Stabilize the tokens per second measurement
    - Add warm-up, at least 5 repeats, and the median, then re-measure all rows for both models
    - _Requirements: 28.1, 28.3_

  - [x] 22.4 Write property test for median timing
    - **Property 31: Median timing is order-independent**
    - **Validates: Requirements 28.1**

- [x] 23. CI and GitHub
  - Demo: the GitHub Actions run is green on ubuntu-latest and windows-latest.
  - [x] 23.1 Initialise git and scan for secrets
    - Confirm `.gitignore` excludes .quf files, eval_cache, dist, logs, venvs, and secrets, and scan for `pypi-` before staging
    - _Requirements: 29.1, 29.2_

  - [x] 23.2 Commit and push to main
    - Run `git ls-remote` first and stop and report if the remote has commits, never force-push, and store no tokens
    - _Requirements: 29.3, 29.4_

  - [x] 23.3 Confirm GitHub Actions passes
    - Fix failures on ubuntu-latest and windows-latest and push again
    - _Requirements: 29.5_

- [x] 24. Release 0.2.1
  - Demo: the 0.2.1 wheel installs in a fresh venv and `quf --help` and `quf info` run.
  - [x] 24.1 Update docs with verified numbers
    - Update README, docs, and RESULTS.md and remove the "not run (unverified)" markers now verified
    - Depends on tasks 26 and 27, so the docs report the measured speed and size numbers from those tasks
    - _Requirements: 30.3_

  - [x] 24.2 Bump version, changelog, build, and smoke test
    - Set 0.2.1 in `pyproject.toml` and `quf/__init__.py`, update CHANGELOG.md, rebuild, run `twine check`, and smoke test; do not upload
    - _Requirements: 30.1, 30.2, 30.4_

- [x] 25. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 26. Fast decode path
  - Demo: `quf bench` on Qwen2.5-0.5B and Qwen2.5-1.5B v020 reports median tokens per second for the eager and CUDA graph decoders, and docs/BACKENDS.md and docs/RESULTS.md show the speedup over the 40 to 50 tokens per second baseline.
  - Runs before task 24.1 so that release 0.2.1 includes it.
  - [x] 26.1 Add the Static_KV_Cache and fixed-shape Decode_Step
    - Preallocate the per-layer cache once, update it in place with `index_copy_`, mask by position, copy prefill keys and values into it, and raise an error naming both lengths when the cache length is exceeded
    - Use it on both the Torch_GPU and CPU Backends, keeping the CPU Backend correct
    - _Requirements: 31.1, 31.2, 31.10_

  - [x] 26.2 Add Fused_Projection for qkv and gate/up
    - Concatenate cached fp16 weights and biases at `prepare` time, split the output by head widths, and free the unfused tensors
    - _Requirements: 31.5_

  - [x] 26.3 Add the CUDA_Graph_Decoder with eager fallback
    - Warm up on a side stream, capture with `torch.cuda.graphs`, replay per token with static input and output tensors, and on any capture failure log at INFO and continue with the Eager_Decoder
    - _Requirements: 31.3, 31.4_

  - [x] 26.4 Probe and optionally add the torch.compile decode path
    - Add a one-time probe that compiles a small function; enable the compiled path only if the probe passes on Windows without Triton, otherwise skip it and write the reason to docs/BACKENDS.md
    - _Requirements: 31.8, 31.9_

  - [x] 26.5 Write property test for graph versus eager decode
    - **Property 32: Graph decode equals eager decode**
    - Random prompts, lengths, and the shapes of both models with random weights; the CUDA graph variant carries the `gpu` marker and the CPU static-cache variant runs in CI
    - **Validates: Requirements 31.2, 31.3, 31.6, 31.10**

  - [x] 26.6 Write property test for fused projections
    - **Property 33: Fused projection equals separate projections**
    - **Validates: Requirements 31.5, 31.7**

  - [x] 26.7 Write unit tests for the decode path
    - Cache overflow error, INFO log and eager continuation when capture is forced to fail, fused weights replacing unfused ones, and the compile probe skip path
    - _Requirements: 31.1, 31.4, 31.5, 31.9_

  - [x] 26.8 Measure speed and update docs
    - Run the 22.3 timing helper (warm-up, at least 5 repeats, median) on 0.5B and 1.5B v020 for the eager and graph decoders and report each speedup against the 40 to 50 tokens per second baseline
    - Target at least 3x; if a model misses the target, report the measured number and the remaining bottleneck in docs/BACKENDS.md and docs/RESULTS.md
    - _Requirements: 31.11, 31.12_

- [x] 27. Size without losing precision
  - Demo: `python scripts/size_study.py` prints a size and llama-protocol perplexity table for the v020 preset and every candidate on both models, and docs/RESULTS.md records the chosen default.
  - Runs before task 24.1 so that release 0.2.1 includes it. Keep a candidate only if it helps.
  - [x] 27.1 Add the size study script and candidate presets
    - Add candidates A (last layer int4 g32), B (down_proj int4 g32 with act-order instead of int8), and C (A and B) as named mixed precision plans, and a script that converts, sizes, and evaluates them with `--protocol llama`
    - _Requirements: 32.1, 32.2_

  - [x] 27.2 Cost check and optional int6 or int5 group-wise option
    - Check whether a new dtype code, packer, unpacker, reference dequantizer, and dequantization in both Backends fit in a small change that leaves Packed_Int4 tensors unchanged; implement it and add it as candidate D only if so, otherwise record candidate D as not evaluated with the reason
    - If implemented, bump the format version, keep every older file readable, and document it in FORMAT.md
    - _Requirements: 32.3, 32.6, 32.7_

  - [x] 27.3 Run the study on both models and choose the default
    - Convert and evaluate the v020 preset and all candidates on Qwen2.5-0.5B and Qwen2.5-1.5B under the llama protocol, adopt the smallest candidate within 1 percent of v020 perplexity on both models, and otherwise keep v020; record every size and perplexity in docs/RESULTS.md
    - _Requirements: 32.1, 32.4, 32.5, 32.8_

  - [x] 27.4 Write tests for the new default and any new format feature
    - Test that the chosen preset name resolves to the expected per-module plan and that `quf info` reports it, and for a new dtype add Property 34 and an older-fixture read test
    - **Property 34: New bit width round trip (only if a new bit width is added)**
    - **Validates: Requirements 32.6, 32.7**
    - _Requirements: 32.4, 32.5, 32.6, 32.7_

- [x] 28. Triton int4 kernels
  - Demo: with `quf[fast]` installed, `quf run --backend int4` decodes from packed int4 weights and the kernel matches the reference dequantizer.
  - [x] 28.1 Add the kernel module, layout repack, and Triton gating
    - Add `quf/backends/triton_kernels.py` with `triton_available()` (Triton_Gate, never raises) and `repack_int4` that builds the Kernel_Layout, including the one-time column permutation for act-order
    - Add the `fast` extra to pyproject.toml with `triton-windows` on Windows and `triton` on Linux through platform markers, and keep the wheel pure Python
    - _Requirements: 33.5, 34.1, 36.1, 36.2, 36.3_
  - [x] 28.2 Add the W4A16 GEMV kernel
    - Support group sizes 32, 64, 128, symmetric and asymmetric, M from 1 to 16, fp32 accumulation, fp16 output, writing into a preallocated output
    - _Requirements: 33.1, 33.2, 33.3, 33.4_
  - [x] 28.3 Add int8 handling
    - Route int8 modules and unsupported shapes or group sizes to the cached fp16 path and log the module name at INFO level
    - _Requirements: 33.6, 33.11_
  - [x] 28.4 Add the prefill path
    - For M above 16, dequantize into one reused fp16 buffer allocated at prepare time and run a matmul
    - _Requirements: 33.7_
  - [x] 28.5 Add fused projection support
    - Concatenate packed weights, scales, and zero points along the output dimension for qkv and gate/up, split the output by head widths, and keep only Kernel_Layout tensors for int4 modules
    - _Requirements: 33.8, 33.10_
  - [x] 28.6 Check CUDA graph capture
    - Capture and replay the kernel with no host synchronization and no allocation during capture, and compare replay output with eager output
    - _Requirements: 33.9_
- [x] 29. Backend integration
  - Demo: `quf info` reports the selected Backend and the Triton_Gate result, and the selector falls back to torch_gpu with one INFO line when Triton does not work.
  - [x] 29.1 Add the triton_int4 backend and startup self-test
    - Add `quf/backends/triton_int4.py` with `prepare`, `linear`, and a Self_Test against the Reference_Dequantizer for group sizes 32, 64, 128 in both zero point modes, run before any weights load
    - _Requirements: 33.1, 33.10, 34.1_
  - [x] 29.2 Add the selector rule, `--backend int4`, and QUF_NO_TRITON
    - Default to the Int4_Backend on CUDA only when Triton works, the Self_Test passes, and the default decision is on, otherwise use torch_gpu with one INFO line; add `--backend int4` and the QUF_NO_TRITON=1 variable, which prevents any Triton import; report both in `quf info`
    - _Requirements: 34.2, 34.3, 34.4, 34.5, 34.6, 34.7, 34.8_
  - [x] 29.3 Write property test for selector rules
    - **Property 36: Backend selection rules with the int4 backend**
    - Runs on CPU in CI
    - **Validates: Requirements 34.2, 34.3, 34.4, 34.5, 34.6, 34.8**
- [x] 30. Correctness tests
  - Demo: `pytest -m triton` passes on the GPU machine and `pytest` passes in CPU CI without Triton.
  - [x] 30.1 Write property test for kernel equivalence
    - **Property 35: Int4 kernel equals the fp32 reference**
    - Random shapes, group sizes, symmetric and asymmetric, g_idx, fused layouts, and M from 1 to 16, marked `triton`
    - **Validates: Requirements 33.1, 33.2, 33.3, 33.4, 33.5, 33.7, 33.8**
  - [x] 30.2 Write property test for greedy token equality
    - **Property 37: Int4 backend greedy tokens equal Torch_GPU**
    - Tiny models with random quantized weights, with and without the CUDA graph, marked `triton`
    - **Validates: Requirements 35.1**
  - [x] 30.3 Add the whole-model logits check and the triton marker
    - Compare logits on the real 0.5B file against torch_gpu, register the `triton` marker, skip Triton tests when Triton is missing, and confirm the CPU CI suite passes without Triton
    - _Requirements: 35.2, 36.3, 36.4_
- [x] 31. Measure and decide
  - Demo: docs/RESULTS.md and docs/BACKENDS.md hold the measured int4 numbers and the default decision, and version 0.3.0 is built and pushed.
  - [x] 31.1 Measure speed, memory, and perplexity
    - Run the median timing helper for eager and graph decoders on 0.5B and 1.5B size_b, record GPU memory after load, and run llama-protocol perplexity, which must stay within 0.1% of torch_gpu
    - Target at least 1.5x of 255 and 83 tok/s; report the measured speedup honestly if the target is missed
    - _Requirements: 35.3, 35.4, 35.5, 35.6_
  - [x] 31.2 Decide the default and update the docs
    - Make the Int4_Backend the default on CUDA only if it is faster on both models and the logits and perplexity checks pass, otherwise keep it opt-in and record why; update docs/BACKENDS.md, docs/RESULTS.md, and the README
    - _Requirements: 35.7, 35.8, 34.3_
  - [x] 31.3 Release 0.3.0
    - Update the CHANGELOG, bump the version to 0.3.0, build, run `twine check`, smoke test in a fresh virtual environment with and without the fast extra, commit, push, and confirm CI passes; do not upload to PyPI
    - _Requirements: 36.5, 36.6, 36.7_
- [x] 32. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
- [x] 33. Decode profiling and fused kernels (speed)
  - Demo: graph-mode decode on both models runs through the fused kernels and the per-kernel profile and measured speedup are in docs/BACKENDS.md.
  - [x] 33.1 Profile graph-mode decode per kernel
    - Add `scripts/profile_decode.py` that runs torch.profiler on graph-mode decode for Qwen2.5-0.5B and Qwen2.5-1.5B and write the per-kernel breakdown to docs/BACKENDS.md
    - _Requirements: 37.1_
  - [x] 33.2 Add the fused Triton kernels in the order the profile justifies
    - Add under `quf/runtime/kernels/`: fused residual-add plus RMSNorm, fused RoPE plus KV-cache write, a flash-decode GQA attention kernel for M equal to 1 over the static cache with split-K over sequence length, an int4 and int8 lm_head GEMV so the output layer is no longer fp16, and greedy argmax on the GPU inside the graph with no host sync per token
    - Sampling with temperature above 0 may stay on the host or use the Gumbel-max trick with a seeded Philox generator if it stays deterministic
    - _Requirements: 37.2, 37.3, 37.4, 37.5_
  - [x] 33.3 Gate, capture, and fall back
    - Make every fused kernel capturable in a CUDA graph, fall back to the 0.3.0 path with one INFO line when unavailable, gate it by the same Triton_Gate, and never import it when QUF_NO_TRITON equals 1
    - _Requirements: 37.6, 37.7, 37.8_
  - [x] 33.4 Write property tests for the fused kernels
    - **Property 38: Fused kernel equals unfused reference**
    - **Property 39: Quantized lm_head GEMV equals the dequantized reference**
    - **Property 40: Fused decode greedy tokens equal the 0.3.0 path**
    - **Property 42: Seeded sampling is deterministic**
    - Mark `triton`, and handle near ties as in tests/test_int4_greedy.py
    - **Validates: Requirements 37.2, 37.3, 37.4, 37.5, 37.6, 37.7, 38.6**
  - [x] 33.5 Measure speed against the targets
    - Measure median graph-mode decode on both models and compare with 2 times the 0.3.0 rate (required) and the Q4_K_M rate (stretch); report the measured speedup honestly in docs/BACKENDS.md
    - _Requirements: 37.9_
- [x] 34. Embedding and lm_head quantization (size)
  - Demo: `quf convert --embed-precision int8` writes a smaller QUF_V3 file that runs on all Backends, and fp16 files are unchanged.
  - [x] 34.1 Add the Quantized_Embedding to the format
    - Support per-row group-wise int4 (g32 or g64) and int8 tables with fp16 scales, row gather that dequantizes on the fly, and a tied lm_head through the int4 or int8 GEMV; write format version 3 only when needed, reject it cleanly in v1 and v2 readers, and keep v1 and v2 files loading
    - _Requirements: 38.1, 38.2, 38.3, 38.4, 38.5, 38.6_
  - [x] 34.2 Add the quantizer and converter options
    - Add `--embed-precision fp16|int8|int4` to `quf convert` with the default set in 34.4, and RTN or GPTQ-like error compensation for the embedding and lm_head where cheap, recording the method in the file metadata
    - _Requirements: 38.7, 38.8_
  - [x] 34.3 Support all Backends and test the format
    - Support the Quantized_Embedding in the cpu, torch_gpu, and triton_int4 Backends, and write property and unit tests for the round trip, dequantization equivalence, v1 and v2 loading, and unchanged fp16 files
    - **Property 41: Quantized embedding round trip and row dequantization**
    - **Validates: Requirements 38.1, 38.2, 38.5, 38.9, 38.10, 38.11**
  - [x] 34.4 Run the size study and choose the defaults
    - Under the llama protocol on both models, search Embed_Precision, mixed presets, and the int8 last-layer rule of size_b, choose the default so size and WikiText-2 Perplexity are at most Q4_K_M and C4 rises by at most 1 percent against 0.3.0, and record the best Pareto points honestly if no combination meets the target
    - _Requirements: 39.1, 39.2, 39.3, 39.4_
- [x] 35. Accuracy guard and improvements
  - Demo: docs/RESULTS.md holds the lm-eval table and the effect of each quantizer improvement.
  - [x] 35.1 Run the lm-eval accuracy guard
    - With the new defaults, run hellaswag and arc_easy (limit 500, 0-shot, seed 1234) on both models for fp16 and QUF 0.4.0 and for Q4_K_M through the lm_eval gguf or llama-cpp backend if feasible on Windows, otherwise record "not run" with the reason
    - _Requirements: 39.5, 39.6_
  - [x] 35.2 Evaluate cheap quantizer improvements
    - Evaluate 256 samples of 2048 tokens, a GPTQ damping sweep of 0.005, 0.01, and 0.05, per-layer group size selection by sensitivity, and true-sequential quantization within attention; keep only those that lower WikiText-2 Perplexity and record every result
    - _Requirements: 39.7, 39.8_
- [x] 36. Broader comparison
  - Demo: docs/RESULTS.md and the README hold a Pareto_Table with a per-axis beats, ties, or loses statement.
  - [x] 36.1 Produce the GGUF variants
    - With eval_cache/llama_cpp tools, produce Q4_K_S, IQ4_XS with an imatrix from llama-imatrix on the WikiText-2 train calibration text, and Q5_K_M for both models
    - _Requirements: 40.1, 40.2_
  - [x] 36.2 Measure and write the Pareto table
    - Measure WikiText-2 and C4 Perplexity under the llama protocol, file size, and GPU tok/s with llama-bench for every GGUF variant and QUF 0.4.0 on both models; write the Pareto_Table in docs/RESULTS.md and the README, state per axis whether QUF beats, ties, or loses, and state whether the Beat_Target was reached with no blanket claims
    - _Requirements: 40.3, 40.4, 40.5, 40.8_
  - [x] 36.3 Check scaling on Qwen2.5-3B
    - If 16 GB of GPU memory allows, run QUF and Q4_K_M on Qwen2.5-3B; skip with a recorded reason if conversion or evaluation is too slow
    - _Requirements: 40.6, 40.7_
- [x] 37. Release 0.4.0
  - Update the CHANGELOG, bump the version to 0.4.0, build, run `twine check`, smoke test in a fresh virtual environment with and without the fast extra, commit, push, and confirm CI passes; do not upload to PyPI
  - _Requirements: 41.1, 41.2, 41.3, 41.4, 41.5_
- [x] 38. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
- [x] 39. Fast decode path for int8 modules
  - Demo: the 0.4.0 default file decodes with no cuBLAS fp16 GEMV and no neg/cat/index_copy kernels in steady decode, and docs/BACKENDS.md holds the before and after tokens per second.
  - Evidence (task 33.5, docs/BACKENDS.md "Decode speed against the Requirement 37 targets"): the 0.4.0 default preset size_f stores k_proj, v_proj, and the last layer as int8, and the fused fast path in quf/runtime/model.py (_build_fast_layers, _decode_layers_fused) handles only int4. Measured steady decode is 502 tok/s (0.5B) and 265 tok/s (1.5B) for the 0.4.0 file, against 806 and 354 for the 0.3.0 all-int4 file and 773 and 449 for Q4_K_M.
  - [x] 39.1 Implement the Triton int8 per-output-channel GEMV for M equal to 1
    - Add the Int8_Kernel in quf/runtime with the same epilogues as decode_gemv (bias add and scatter output index), fp32 accumulation, and CUDA graph capture support
    - _Requirements: 42.1, 42.2_
  - [x] 39.2 Add fast-path support for mixed int4 and int8 groups
    - A qkv group that mixes int4 and int8 members runs as separate kernel calls into one scattered output buffer, or as one launch where that is simple, and the layer keeps the fused RoPE plus KV-cache write, add+RMSNorm, and Flash_Decode_Kernel
    - Treat all-int8 layers (the last layer) and int8 gate_up and down modules as fast-path
    - Log each remaining fallback once at DEBUG with the reason
    - _Requirements: 42.3, 42.4, 42.5_
  - [x] 39.3 Fix scripts/profile_decode_graph.py for quantized-embedding files and re-profile
    - Handle m.embed equal to None for quantized-embedding files, re-profile the 1.5B 0.4.0 file, and record that no cuBLAS fp16 GEMV and no neg, cat, or index_copy kernel remains in steady decode
    - _Requirements: 42.6_
  - [x] 39.4 Write the equivalence property tests
    - **Property 43: Fused fast path equals the unfused path on mixed-precision modules**
    - On tiny models with mixed int4, int8, and fp16 modules, compare fused fast-path logits and greedy tokens with the unfused path using the near-tie helper tests/_greedy_compare.py
    - Add a kernel-level property that compares the int8 GEMV with the fp32 reference
    - **Validates: Requirements 42.1, 42.3, 42.4**
  - [x] 39.5 Re-measure 0.4.0 decode speed and report the result
    - Repeat the 33.5 method: alternating runs, median of 7, same session as the llama.cpp rows, power state recorded
    - Targets: 0.5B at or above 773 tok/s (Q4_K_M), and 1.5B at or above 398 tok/s (1.5 times the pre-fix 265)
    - If the int4 GEMV then dominates, tune _int4_gemv1_kernel for the in-model shapes and record before and after
    - Record the measured numbers in docs/BACKENDS.md whether or not the targets are reached
    - _Requirements: 42.7, 42.8_
- [ ] 40. Deployment README and release 0.4.1
  - Demo: README.md renders the same on GitHub and in the PyPI preview with absolute links and the QUF-versus-GGUF chart, `quf.load(path).generate(prompt)` returns the text that `quf run` prints, `quf convert` rejects a Qwen3 checkpoint before any heavy work, and version 0.4.1 is built, checked, committed, and pushed without a PyPI upload.
  - Context: quf 0.4.0 is on PyPI. pyproject.toml has `readme = "README.md"`, so README.md is both the GitHub README and the PyPI description, and PyPI shows the new README only after 0.4.1 is uploaded. Measured results are in docs/bench_accuracy.json (pareto_040, speed_040_final, scaling_3b, lm_eval quf_0_4_0, and models_llama for the fp16 rows), and the GGUF build details are in docs/gguf_variants.json.
  - [x] 40.1 Reject non-Qwen2 architectures and add the `quf.load` Python API
    - In quf/convert/converter.py, set SUPPORTED_ARCHITECTURES to ("Qwen2ForCausalLM",), rewrite the comment (an architecture is added only after the runtime supports it; Qwen3 first needs q_norm and k_norm in quf/runtime/model.py), and let UnsupportedArchitectureError also derive from ValueError so that `quf convert` prints only "Unsupported model architecture 'X'. Supported: Qwen2ForCausalLM."; check_architecture already runs before the tokenizer, the calibration data, and the weights, so keep that order
    - Replace test_qwen3_accepted_and_non_qwen_rejected in tests/test_converter_unit.py: Qwen3ForCausalLM, LlamaForCausalLM, and a config without `architectures` are rejected with a message that names Qwen2ForCausalLM before a failing tokenizer stub is reached, Qwen2ForCausalLM is accepted, and a CLI test with `_check_convert_extras` patched out checks the single `quf: error:` line
    - Add quf/api.py with `load(path, device="auto", backend="auto", cuda_graph=None)` and the `Model` handle (`generate`, `tokenizer`, `backend`, `header`, `close`, and context-manager support) as in the design, built only from QUFModel.load, the header tokenizer construction of `_cmd_run`, and quf.runtime.generate.generate; leave `quf run` unchanged
    - Expose `quf.load` and `quf.Model` lazily in quf/__init__.py next to the lazy submodules (and add `api` to them), so that `import quf` still imports no torch, tokenizers, or quf.runtime module
    - Write CPU unit tests in tests/test_api.py: the attributes, streaming, argument errors and boundary values, `close` twice, the context manager, a file without tokenizer.json, the device and backend checks, and the lazy import in a subprocess
    - **Property 44: The Python API returns the same text as `quf run`**
    - Write it in tests/test_api.py with Hypothesis (at least 100 examples, CPU, runs in CI), tiny files from tests/test_runtime_unit.make_quf including one whose EOS id greedy decoding emits, and stdout captured with contextlib.redirect_stdout
    - **Validates: Requirements 43.3, 43.5, 43.6, 43.7**
    - _Requirements: 43.1, 43.2, 43.3, 43.4, 43.5, 43.6, 43.7, 43.8, 43.9, 43.10, 43.11, 43.12, 43.13, 43.14_
  - [x] 40.2 Add scripts/make_readme_chart.py and generate the chart
    - Write a pure `collect_points(bench)` that reads pareto_040, scaling_3b where present, and the fp16 WikiText-2 value from models_llama with the row labels of the design, and a `main` that imports matplotlib (Agg backend) only inside the function and writes docs/assets/quf_vs_gguf.png
    - One panel per model with data: x = file size in MB (10^6 bytes), y = WikiText-2 Perplexity (llama protocol); every measured format is a labelled point; QUF gets its own color and marker; fp16 is a dashed horizontal line, or the panel says "fp16 not measured" (3B); add the note "lower-left is better" and a small caption that names docs/bench_accuracy.json and the GPU from `environment.device`
    - Keep matplotlib out of pyproject.toml; when it is missing, exit with status 1 and a message that names `pip install matplotlib`
    - Run the script to write the PNG, and add tests/test_readme_chart.py: `collect_points` on the real JSON (runs in CI), the missing-matplotlib path, and a PNG smoke test with `pytest.importorskip("matplotlib")`
    - _Requirements: 43.15, 43.16, 43.17, 43.18, 43.19, 43.20, 43.21, 43.22_
  - [x] 40.3 Polish the package metadata and the CI matrix
    - In pyproject.toml: a one-line description of at most 100 characters; the keywords quantization, llm, gptq, int4, 4-bit, triton, cuda, windows, and qwen; Python classifiers 3.10, 3.11, 3.12, and 3.13 (3.13 only because CI runs it); "Environment :: GPU :: NVIDIA CUDA"; "Operating System :: Microsoft :: Windows" and "Operating System :: POSIX :: Linux", with "Operating System :: OS Independent" removed because CI tests only those two; and [project.urls] Homepage, Repository, Documentation (`tree/main/docs`), Changelog (`blob/main/CHANGELOG.md`), and Issues
    - In .github/workflows/ci.yml, set the matrix to Python 3.10, 3.11, 3.12, and 3.13 on ubuntu-latest and windows-latest, and keep the wheel artifact upload on the ubuntu-latest Python 3.12 job only
    - Check every classifier against the trove-classifiers package
    - Extend tests/test_pyproject_metadata.py: the description length, the keywords, the URLs, the CUDA and operating system classifiers, no matplotlib dependency, and the Python classifiers against the ci.yml versions (skipped when the workflow file is absent); leave the version assertion to 40.6
    - _Requirements: 43.21, 43.23, 43.24, 43.25, 43.26, 43.27, 43.28, 43.29, 43.30_
  - [x] 40.4 Rewrite README.md as the public landing page for GitHub and PyPI
    - Sections in this order: title and one-sentence tagline; badges (shields.io: PyPI version, Python versions, license, CI workflow status); Why QUF (4 or 5 bullets); Results at a glance; Install; Quickstart; Python API; How it works; Benchmarks; CLI reference; Supported models; Limitations; Troubleshooting; File format; Development; Changelog; License; Acknowledgements
    - Rendering rules: plain GitHub-flavoured Markdown only (no raw HTML, no mermaid, no "> [!NOTE]" alerts, no `#anchor` links); every repository link uses `https://github.com/vtrsharvin-sj/Quantized-Unified-Format/blob/main/<path>` (or `tree/main/<folder>`); every repository image uses `https://raw.githubusercontent.com/vtrsharvin-sj/Quantized-Unified-Format/main/<path>`; every code block has a language tag
    - Results at a glance: QUF 0.4.0 against Q4_K_M, and Q4_K_M imatrix where measured, for 0.5B, 1.5B, and 3B, with the columns size MB, WikiText-2 PPL, C4 PPL, and decode tok/s; bold the best value per model and column except comparisons marked "equal within noise"; then one plain sentence that names the wins and every loss in the design's glance verdict table (0.5B C4 against Q4_K_M and both Q4_K_M imatrix Perplexities, 1.5B and 3B file size, 1.5B C4 against Q4_K_M imatrix, 3B decode, and Q5_K_M being more accurate on 0.5B and 1.5B); then the chart; then a link to the full benchmarks
    - Install: quf, quf[fast], and quf[convert]; the CUDA 12.8 PyTorch wheel for RTX 50-series through the cu128 index URL; CPU-only works. Quickstart: convert, info, run, and bench with real abbreviated output, plus the note that `quf bench` scores non-overlapping 2048-token windows (40 by default) and times end-to-end generation. Python API: `quf.load`, `generate`, and streaming. How it works: the file format, the quantization pipeline, the runtime backends, and automatic selection in at most 40 lines
    - Benchmarks: per-model tables with fp16 and every GGUF variant (Q4_K_M, Q4_K_M imatrix, Q4_K_S, IQ4_XS imatrix, Q5_K_M); the lm-eval table with Q4_K_M "not run" and its reason; the methodology (llama protocol, 2048-token windows scoring the second half, WikiText-2 test plus a held-out C4 validation slice, calibration on WikiText-2 train 128x1024, imatrix variants from the same calibration text, QUF steady-state decode against llama-bench tg128 in one interleaved session, with the end-to-end medians next to it); the GPU, the OS, and the power state (battery for the 0.5B and 1.5B speed rows, AC for 3B); and how to reproduce, saying so where a result came from a script that is not in the repository
    - CLI reference: concise option tables for convert, run, bench, and info that match `--help`; the presets with their rules (size_f default: k_proj, v_proj, and the last layer int8, the rest int4; size_b, sensitive, none, and the others); and QUF_NO_TRITON, QUF_NO_CUDA_GRAPH, and QUF_NO_FUSED
    - Supported models (Qwen2ForCausalLM; tested Qwen2.5 0.5B, 1.5B, and 3B; Instruct variants share the architecture but `quf run` does plain completion with no chat template; other architectures are rejected at convert time), Limitations (an honest list), Troubleshooting (missing Triton, the CPU-only torch hint, triton-windows in a --system-site-packages venv falling back to torch_gpu, turning off the fast paths, format version errors where v3 files need quf 0.4.0 or later, GPU out of memory), File format (versions 1, 2, and 3 with a link to FORMAT.md), Development (clone, editable install, test markers gpu, triton, realmodel, and convert), Changelog (one "Release history" line linking to CHANGELOG.md), License (MIT), and Acknowledgements (GPTQ by Frantar et al., llama.cpp and ggml, Triton and triton-windows, gigatoken, the Qwen team, and Hugging Face transformers and datasets)
    - Honesty: every number from docs/bench_accuracy.json in the formats and source keys of the design; no unmeasured superlatives ("fastest", "best", "state of the art"); no internal task numbers or spec terms; older-release numbers only in the Release history line; 300 to 450 lines. Use real output from runs on this machine for now; 40.5 replaces it with the Copy_Paste_Test output where it differs
    - _Requirements: 43.31, 43.32, 43.33, 43.34, 43.35, 43.36, 43.37, 43.38, 43.39, 43.40, 43.41, 43.42, 43.43, 43.44, 43.45, 43.46, 43.47, 43.48, 43.49, 43.50, 43.51, 43.52, 43.53, 43.54, 43.55, 43.56, 43.57, 43.58, 43.59, 43.60, 43.61, 43.62, 43.63, 43.64_
  - [x] 40.5 Verify the README end to end and fix every mismatch
    - Add tests/test_readme.py with the static checks of the design (section order, rendering rules, absolute links and images with existing targets, language tags, badges, the Why QUF and How it works limits, line count, no banned superlatives, no task numbers or spec terms, and every `--option` exists in `quf.cli._build_parser()`), and make it pass in the CPU suite
    - Build the sdist and the wheel, run `twine check --strict dist/*` and a readme_renderer Markdown render of README.md, and fix every warning
    - Copy_Paste_Test: create a fresh virtual environment (--system-site-packages is allowed to reuse torch; record whether it was used), install the built wheel with `[convert,fast]`, and from a folder outside the repository run every Quickstart command and the Python API snippet exactly as written; put the real abbreviated output into the README
    - Compare every option and default in the README with `quf <command> --help` and every number with docs/bench_accuracy.json; fix each mismatch in the README and re-run the checks until all pass
    - _Requirements: 43.31, 43.32, 43.33, 43.34, 43.35, 43.36, 43.37, 43.38, 43.42, 43.45, 43.52, 43.61, 43.62, 43.63, 43.65, 43.66, 43.67_
  - [-] 40.6 Release 0.4.1
    - Add a 0.4.1 CHANGELOG entry: the README rewrite, `quf.load` and `quf.Model`, the rejection of Qwen3ForCausalLM, and the metadata and CI changes
    - Bump the version to 0.4.1 in pyproject.toml, quf/__init__.py, and tests/test_pyproject_metadata.py
    - Run the full `python -m pytest -q` with no marker filter on this machine, and the CI command `python -m pytest -m "not gpu and not convert" -q` with QUF_NO_TRITON=1; both must pass
    - Build with `python -m build`, run `twine check --strict dist/*`, and scan the sdist and wheel members for .quf files, eval_cache, logs, and secrets (for example `pypi-`); stop and report on any hit
    - Commit and push normally (no force-push) and confirm that every CI job passes; do not upload to PyPI, which the orchestrator does
    - _Requirements: 43.68, 43.69, 43.70, 43.71, 43.72, 43.73_
- [~] 41. Final checkpoint - Ensure all tests pass (GitHub Actions green on every job)
  - Ensure all tests pass, ask the user if questions arise.
  - Confirm that every GitHub Actions job of the CI matrix (Python 3.10 to 3.13 on ubuntu-latest and windows-latest) is green for the release commit.

## Notes

- Tasks 26 and 27 add the speed and size work. They run before task 24.1, so release 0.2.1 includes them, and all of their sub-tasks are required
- Tasks 20 to 25 add the 0.2.1 verification work and all sub-tasks are required
- Tasks 15 to 19 add the 0.2.0 accuracy upgrade. All of their sub-tasks are required, including tests
- Tasks 11 to 14 extend the plan for native Windows and supersede the WSL and GPTQModel notes below
- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests validate the universal correctness properties from the design, and unit tests validate specific examples and edge cases
- Requirements 6.1 (gigatoken tokenization) and Requirement 2 are verified manually through the spike and recorded in docs/ENV.md
- Conversion and Marlin tests run in WSL and are marked `convert` or `gpu`, so CI skips them

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["1.3", "1.4", "2.2", "2.3", "3.2", "4.2"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6", "2.7", "3.3", "3.4", "3.5", "3.6", "4.3", "4.4", "4.5", "5.1", "6.1", "8.1", "10.2"] },
    { "id": 4, "tasks": ["5.2", "6.2", "6.5", "7.1", "8.3"] },
    { "id": 5, "tasks": ["5.3", "6.3", "6.6", "7.2"] },
    { "id": 6, "tasks": ["5.4", "5.5", "5.6", "6.4", "6.7", "6.8", "7.3", "7.4", "9.1", "9.2"] },
    { "id": 7, "tasks": ["6.9", "8.2", "9.4"] },
    { "id": 8, "tasks": ["8.4", "9.3"] },
    { "id": 9, "tasks": ["9.5", "10.1", "10.3"] },
    { "id": 10, "tasks": ["11.1", "12.1", "13.1"] },
    { "id": 11, "tasks": ["11.2", "12.2", "11.4", "11.5", "12.3"] },
    { "id": 12, "tasks": ["11.3", "12.4"] },
    { "id": 13, "tasks": ["11.6", "12.5"] },
    { "id": 14, "tasks": ["12.6"] },
    { "id": 15, "tasks": ["13.2"] },
    { "id": 16, "tasks": ["13.3"] },
    { "id": 17, "tasks": ["15.1", "15.3", "16.1"] },
    { "id": 18, "tasks": ["15.2", "16.2", "16.6"] },
    { "id": 19, "tasks": ["15.4", "16.3", "16.4"] },
    { "id": 20, "tasks": ["16.5"] },
    { "id": 21, "tasks": ["16.7"] },
    { "id": 22, "tasks": ["16.8"] },
    { "id": 23, "tasks": ["17.1"] },
    { "id": 24, "tasks": ["17.2"] },
    { "id": 25, "tasks": ["17.3", "17.4"] },
    { "id": 26, "tasks": ["17.5", "18.1"] },
    { "id": 27, "tasks": ["18.2"] },
    { "id": 28, "tasks": ["18.3"] },
    { "id": 29, "tasks": ["20.1", "21.1", "22.1", "23.1"] },
    { "id": 30, "tasks": ["20.2", "21.2", "22.2", "23.2"] },
    { "id": 31, "tasks": ["20.3", "21.3", "22.3", "22.4", "23.3"] },
    { "id": 32, "tasks": ["26.1", "27.1"] },
    { "id": 33, "tasks": ["26.2", "27.2"] },
    { "id": 34, "tasks": ["26.3", "27.3"] },
    { "id": 35, "tasks": ["26.4", "26.5", "26.6", "26.7", "27.4"] },
    { "id": 36, "tasks": ["26.8"] },
    { "id": 37, "tasks": ["24.1"] },
    { "id": 38, "tasks": ["24.2"] },
    { "id": 39, "tasks": ["25"] },
    { "id": 40, "tasks": ["28.1"] },
    { "id": 41, "tasks": ["28.2"] },
    { "id": 42, "tasks": ["28.3", "28.4", "28.5"] },
    { "id": 43, "tasks": ["28.6"] },
    { "id": 44, "tasks": ["29.1"] },
    { "id": 45, "tasks": ["29.2"] },
    { "id": 46, "tasks": ["29.3", "30.1", "30.2"] },
    { "id": 47, "tasks": ["30.3"] },
    { "id": 48, "tasks": ["31.1"] },
    { "id": 49, "tasks": ["31.2"] },
    { "id": 50, "tasks": ["31.3"] },
    { "id": 51, "tasks": ["32"] },
    { "id": 52, "tasks": ["33.1"] },
    { "id": 53, "tasks": ["33.2", "34.1"] },
    { "id": 54, "tasks": ["33.3", "34.2"] },
    { "id": 55, "tasks": ["33.4", "34.3"] },
    { "id": 56, "tasks": ["34.4", "35.2"] },
    { "id": 57, "tasks": ["33.5", "35.1", "36.1"] },
    { "id": 58, "tasks": ["39.1"] },
    { "id": 59, "tasks": ["39.2"] },
    { "id": 60, "tasks": ["39.3", "39.4"] },
    { "id": 61, "tasks": ["39.5"] },
    { "id": 62, "tasks": ["36.2"] },
    { "id": 63, "tasks": ["36.3"] },
    { "id": 64, "tasks": ["37"] },
    { "id": 65, "tasks": ["38"] },
    { "id": 66, "tasks": ["40.1", "40.2", "40.3"] },
    { "id": 67, "tasks": ["40.4"] },
    { "id": 68, "tasks": ["40.5"] },
    { "id": 69, "tasks": ["40.6"] },
    { "id": 70, "tasks": ["41"] }
  ]
}
```
