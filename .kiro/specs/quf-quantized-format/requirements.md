# Requirements Document

## Introduction

QUF (Quantized Unified Format) is a Python package named `quf` that defines a GGUF-like single-file format for 4-bit quantized decoder-only LLMs, together with a converter and an inference runtime. The goal is fast and accurate inference at low storage. The runtime uses an NVIDIA GPU when available and otherwise uses the CPU.

The v1 target is Qwen decoder-only models with 4-bit weights and group size 128. Development and testing use Qwen2.5-0.5B and Qwen2.5-1.5B, then Qwen2.5-7B. The reference hardware is a Windows laptop with an RTX 5080 (Blackwell, compute capability 12.0) and 16 GB VRAM. WSL (Ubuntu with CUDA) is used for components that do not build or run on Windows.

QUF owns the file format and the converter. The runtime reuses existing kernels and repacks the canonical weights at load time. GPTQModel is the single quantizer and the source of Marlin GPU kernels and CPU kernels. Gigatoken provides tokenization for calibration data, evaluation, and batch jobs, with a Hugging Face tokenizers fallback. lm-evaluation-harness provides accuracy evaluation.

## Glossary

- **QUF**: Quantized Unified Format, the single-file format defined by this feature.
- **QUF_File**: A file in QUF v1 layout.
- **QUF_Package**: The pip-installable Python package `quf`.
- **Header**: The JSON section of a QUF_File containing model config, tokenizer info, quantization scheme, and the Tensor_Table.
- **Tensor_Table**: The Header list of entries giving name, dtype, shape, and byte offset for each tensor.
- **Tensor_Data**: The 64-byte aligned, memory-mappable region of a QUF_File holding tensor bytes.
- **Writer**: The QUF_Package component that serializes a model into a QUF_File.
- **Reader**: The QUF_Package component that parses and validates a QUF_File.
- **Quantizer**: The QUF_Package component that performs 4-bit group quantization and packing.
- **Group_Size**: The number of consecutive weights along the input dimension sharing one scale, fixed at 128 in v1.
- **Packed_Int4**: Canonical weight storage with 8 four-bit values per int32.
- **Scales**: fp16 per-group scale values.
- **Zeros**: Per-group zero points stored when the quantization scheme is asymmetric.
- **Reference_Dequantizer**: The pure PyTorch function that converts Packed_Int4, Scales, and Zeros into fp16 weights.
- **GPTQModel**: The ModelCloud/GPTQModel library used as the calibrated quantizer and kernel provider.
- **Calibration_Builder**: The component that creates calibration samples from text datasets.
- **Tokenizer_Wrapper**: The component that exposes tokenization using gigatoken, with a Hugging Face tokenizers fallback.
- **Converter**: The component behind `quf convert` that turns a Hugging Face Qwen checkpoint into a QUF_File.
- **Runtime**: The component that loads a QUF_File and generates tokens.
- **QUFLinear**: The Runtime linear layer that holds quantized weights and runs the matrix multiply on a selected Backend.
- **KV_Cache**: The Runtime store of attention keys and values for already processed tokens.
- **Backend**: An execution path for QUFLinear, one of Marlin, Torch_GPU, or CPU.
- **Marlin**: The GPTQModel Marlin GPU kernel path, requiring compute capability 8.0 or higher.
- **Torch_GPU**: The fallback Backend that dequantizes weights with PyTorch on an NVIDIA GPU.
- **CPU**: The Backend that runs on the host processor.
- **Backend_Selector**: The component that chooses a Backend for a given device option.
- **CLI**: The `quf` command-line program with subcommands `info`, `convert`, `run`, and `bench`.
- **Benchmark_Suite**: The component behind `quf bench` that measures perplexity, size, and speed.
- **Perplexity**: The WikiText perplexity of a model measured with a fixed context length.
- **Environment_Spike**: The documented investigation of CUDA, PyTorch, and Marlin support on the reference hardware, recorded in docs/ENV.md.

## Requirements

### Requirement 1: Package scaffold and device detection

**User Story:** As a developer, I want an installable package that reports available compute devices, so that I can tell which Backends can run on my machine.

#### Acceptance Criteria

1. THE QUF_Package SHALL provide the importable module `quf` with a version string.
2. THE QUF_Package SHALL declare the optional dependency extras `cuda` and `convert`.
3. WHEN device detection is invoked, THE QUF_Package SHALL report whether an NVIDIA GPU is available, its name, its compute capability, and its total memory.
4. IF no NVIDIA GPU is available, THEN THE QUF_Package SHALL report the CPU as the only available device without raising an error.
5. WHEN the `quf` module is imported on a machine without optional extras installed, THE QUF_Package SHALL complete the import without error.

### Requirement 2: Environment spike

**User Story:** As a developer, I want the CUDA, PyTorch, and Marlin compatibility on the RTX 5080 documented, so that I know which Backends work before building on them.

#### Acceptance Criteria

1. THE Environment_Spike SHALL record in docs/ENV.md the installed versions of the CUDA toolkit, the PyTorch build, GPTQModel, and gigatoken on Windows and on WSL.
2. THE Environment_Spike SHALL record in docs/ENV.md whether PyTorch CUDA kernels execute on compute capability 12.0.
3. THE Environment_Spike SHALL record in docs/ENV.md whether the Marlin kernel executes a matrix multiply on compute capability 12.0.
4. THE Environment_Spike SHALL record in docs/ENV.md whether gigatoken installs and tokenizes on Windows.
5. IF the Marlin kernel fails on compute capability 12.0, THEN THE Environment_Spike SHALL record the failure message and designate Torch_GPU as the GPU Backend for that device.

### Requirement 3: QUF v1 file layout

**User Story:** As a model distributor, I want a single-file format with a self-describing header, so that one file carries everything needed to run a quantized model.

#### Acceptance Criteria

1. THE Writer SHALL begin every QUF_File with the 4-byte magic `QUF1`, followed by a version number, followed by the Header length in bytes, followed by the Header.
2. THE Writer SHALL encode the Header as JSON containing the model config, the tokenizer info, the quantization scheme, and the Tensor_Table.
3. THE Writer SHALL record for each Tensor_Table entry the tensor name, dtype, shape, and byte offset into Tensor_Data.
4. THE Writer SHALL place every tensor at a byte offset that is a multiple of 64 within the QUF_File.
5. THE Writer SHALL store quantized linear weights as Packed_Int4 with 8 four-bit values per int32, Scales as fp16, and Zeros when the scheme is asymmetric.
6. THE Writer SHALL store embedding and lm_head tensors as fp16.
7. THE Writer SHALL record the Group_Size of 128 in the quantization scheme of the Header.
8. THE QUF_Package SHALL provide a FORMAT.md document that specifies the QUF v1 byte layout and Header fields.

### Requirement 4: Format reading and validation

**User Story:** As a runtime author, I want strict validation when opening a file, so that corrupt or incompatible files fail with clear errors.

#### Acceptance Criteria

1. WHEN a valid QUF_File is opened, THE Reader SHALL return the parsed Header and a memory-mapped view of each tensor.
2. IF the first 4 bytes of a file differ from `QUF1`, THEN THE Reader SHALL raise an error identifying an invalid magic.
3. IF the file version is higher than the highest version supported by the Reader, THEN THE Reader SHALL raise an error stating the file version and the supported version.
4. IF the Header length extends beyond the end of the file, THEN THE Reader SHALL raise an error identifying a truncated Header.
5. IF a Tensor_Table entry has an offset that is not a multiple of 64, or has an offset plus byte size beyond the end of the file, THEN THE Reader SHALL raise an error naming the tensor.
6. IF the Header is not valid JSON or lacks a required field, THEN THE Reader SHALL raise an error naming the problem.
7. FOR ALL valid in-memory models, writing to a QUF_File and then reading it SHALL produce an equivalent Header and bit-identical tensors (round-trip property).

### Requirement 5: 4-bit group quantization and packing

**User Story:** As a developer, I want a correct reference quantizer and packer, so that I can verify every kernel against known-good results.

#### Acceptance Criteria

1. WHEN a weight matrix whose input dimension is a multiple of the Group_Size is quantized, THE Quantizer SHALL produce 4-bit integer values in the range 0 to 15, one fp16 scale per group, and zero points when asymmetric.
2. THE Quantizer SHALL pack 8 four-bit values into each int32 in the canonical order defined in FORMAT.md.
3. FOR ALL arrays of 4-bit values with a length that is a multiple of 8, unpacking after packing SHALL return the original values (round-trip property).
4. WHEN Packed_Int4, Scales, and Zeros are provided, THE Reference_Dequantizer SHALL return an fp16 weight matrix of the original shape.
5. FOR ALL weight matrices, the absolute difference between each dequantized element and the original element SHALL be at most one half of the scale of its group, plus fp16 rounding error.
6. IF the input dimension is not a multiple of the Group_Size, THEN THE Quantizer SHALL raise an error stating the dimension and the Group_Size.

### Requirement 6: Tokenization and calibration data

**User Story:** As a developer, I want fast tokenization with a fallback, so that calibration and evaluation run quickly on every platform.

#### Acceptance Criteria

1. WHEN text is provided, THE Tokenizer_Wrapper SHALL return token ids using gigatoken.
2. IF gigatoken is not importable on the current platform, THEN THE Tokenizer_Wrapper SHALL use Hugging Face tokenizers and log the reason for the fallback.
3. FOR ALL text strings, decoding the token ids returned by encoding SHALL reproduce the original text (round-trip property).
4. WHEN a text dataset, a sample count, and a sequence length are provided, THE Calibration_Builder SHALL return exactly the requested number of samples each of exactly the requested sequence length.
5. WHEN the same dataset, seed, and parameters are provided twice, THE Calibration_Builder SHALL return identical samples.
6. IF the dataset yields fewer tokens than the requested samples require, THEN THE Calibration_Builder SHALL raise an error stating the shortfall.

### Requirement 7: Calibrated conversion

**User Story:** As a model distributor, I want to convert a Hugging Face Qwen checkpoint into a QUF_File, so that I can publish a small, accurate model.

#### Acceptance Criteria

1. WHEN `quf convert` is run with a Hugging Face Qwen model identifier or path and an output path, THE Converter SHALL quantize the model with GPTQModel at 4 bits and Group_Size 128 and write a QUF_File.
2. THE Converter SHALL accept options for the calibration dataset, sample count, sequence length, and seed.
3. THE Converter SHALL store the tokenizer info and model config of the source checkpoint in the Header.
4. THE Converter SHALL store embedding and lm_head tensors as fp16.
5. WHEN conversion completes, THE Converter SHALL print the output size in bytes and the ratio to the fp16 checkpoint size.
6. IF the source model architecture is not a Qwen decoder-only architecture, THEN THE Converter SHALL exit with a nonzero status and a message naming the architecture.
7. IF the output path already exists and no overwrite option is given, THEN THE Converter SHALL exit with a nonzero status without modifying the existing file.
8. IF conversion fails after the output file is created, THEN THE Converter SHALL delete the partial output file.

### Requirement 8: CPU runtime and generation

**User Story:** As a user without an NVIDIA GPU, I want to run a QUF_File on the CPU, so that I can generate text on any machine.

#### Acceptance Criteria

1. WHEN `quf run` is run with a QUF_File and a prompt, THE Runtime SHALL generate tokens autoregressively and print the generated text.
2. THE QUFLinear SHALL compute the same output on the CPU Backend as the Reference_Dequantizer weights followed by a matrix multiply, within an absolute tolerance of 1e-2 in fp16.
3. WHILE generation is in progress, THE Runtime SHALL store keys and values of processed tokens in the KV_Cache and compute each new token using the cached values.
4. WHILE streaming is enabled, THE Runtime SHALL emit each decoded text fragment to standard output as soon as the token is generated.
5. THE Runtime SHALL accept options for maximum new tokens, temperature, top-k, top-p, and seed.
6. WHEN temperature is 0, THE Runtime SHALL select the highest-probability token at each step.
7. WHEN the same QUF_File, prompt, and options including seed are provided twice on the same Backend, THE Runtime SHALL generate identical token ids.
8. IF the prompt length plus maximum new tokens exceeds the model context length in the Header, THEN THE Runtime SHALL exit with a nonzero status and a message stating both numbers.
9. WHEN the end-of-sequence token is generated, THE Runtime SHALL stop generation.
10. FOR ALL prompts, generating with the KV_Cache SHALL produce the same logits as recomputing the full sequence without the KV_Cache, within an absolute tolerance of 1e-2 (invariant).

### Requirement 9: CUDA backend

**User Story:** As a user with an NVIDIA GPU, I want GPU-accelerated inference, so that generation is fast.

#### Acceptance Criteria

1. WHERE the Marlin Backend is selected, THE Runtime SHALL repack the canonical Packed_Int4 weights into the Marlin layout at load time.
2. THE QUFLinear SHALL compute the same output on the Marlin Backend as on the CPU Backend, within an absolute tolerance of 1e-2 in fp16.
3. WHERE the Torch_GPU Backend is selected, THE QUFLinear SHALL dequantize weights with the Reference_Dequantizer on the GPU and compute the matrix multiply on the GPU.
4. THE QUFLinear SHALL compute the same output on the Torch_GPU Backend as on the CPU Backend, within an absolute tolerance of 1e-2 in fp16.
5. THE Runtime SHALL keep the QUF_File on disk unmodified when repacking weights.
6. IF the GPU runs out of memory while loading, THEN THE Runtime SHALL exit with a nonzero status and a message stating the required and available GPU memory.

### Requirement 10: Automatic backend selection

**User Story:** As a user, I want the best Backend chosen automatically, so that I do not need to know my hardware details.

#### Acceptance Criteria

1. THE CLI SHALL accept the option `--device` with the values `auto`, `cpu`, and `cuda`.
2. WHEN `--device auto` is given and an NVIDIA GPU with compute capability 8.0 or higher is available and the Marlin kernel passes a self-test, THE Backend_Selector SHALL select Marlin.
3. WHEN `--device auto` is given and an NVIDIA GPU is available and Marlin is not selected, THE Backend_Selector SHALL select Torch_GPU.
4. WHEN `--device auto` is given and no NVIDIA GPU is available, THE Backend_Selector SHALL select CPU.
5. WHEN `--device cpu` is given, THE Backend_Selector SHALL select CPU.
6. WHEN `--device cuda` is given and an NVIDIA GPU is available, THE Backend_Selector SHALL select Marlin or Torch_GPU by the same rules as `auto`.
7. IF `--device cuda` is given and no NVIDIA GPU is available, THEN THE Backend_Selector SHALL exit with a nonzero status and a message stating that no NVIDIA GPU was found.
8. WHEN a Backend is selected, THE Backend_Selector SHALL log the selected Backend and the reason.

### Requirement 11: Benchmarks and accuracy evaluation

**User Story:** As a maintainer, I want repeatable measurements of accuracy, size, and speed, so that I can confirm QUF meets its goals against fp16 and GGUF Q4.

#### Acceptance Criteria

1. WHEN `quf bench` is run with a QUF_File, THE Benchmark_Suite SHALL report WikiText Perplexity, file size in bytes, and generation speed in tokens per second.
2. WHERE a fp16 reference model is provided, THE Benchmark_Suite SHALL report the Perplexity of the fp16 model and the relative Perplexity increase of the QUF_File.
3. WHERE a GGUF Q4 file is provided, THE Benchmark_Suite SHALL report its file size and generation speed.
4. THE Benchmark_Suite SHALL report the Perplexity increase of QUF_File models relative to fp16, with a documented target of at most 3 percent.
5. THE Benchmark_Suite SHALL provide an adapter that lets lm-evaluation-harness evaluate a QUF_File.
6. WHEN `quf bench` completes, THE Benchmark_Suite SHALL write the results as a JSON file containing the Backend, device name, and software versions.
7. IF the WikiText dataset cannot be loaded, THEN THE Benchmark_Suite SHALL exit with a nonzero status and a message naming the dataset.

### Requirement 12: Command-line interface

**User Story:** As a user, I want a single `quf` command, so that I can inspect, convert, run, and benchmark models.

#### Acceptance Criteria

1. THE CLI SHALL provide the subcommands `info`, `convert`, `run`, and `bench`.
2. WHEN `quf info` is run with a QUF_File, THE CLI SHALL print the model config, quantization scheme, tensor count, and total file size.
3. IF `quf info` is run with a file that fails validation, THEN THE CLI SHALL exit with a nonzero status and print the Reader error message.
4. WHEN any subcommand is run with `--help`, THE CLI SHALL print its usage and exit with status 0.
5. IF an unknown subcommand is given, THEN THE CLI SHALL exit with a nonzero status and print the list of valid subcommands.

### Requirement 13: Packaging, documentation, and CI

**User Story:** As a maintainer, I want a documented, tested, installable package, so that others can install and trust it.

#### Acceptance Criteria

1. THE QUF_Package SHALL be installable with pip, with the extras `quf[cuda]` and `quf[convert]`.
2. THE QUF_Package SHALL include a README that describes installation, conversion, running, and benchmarking.
3. THE QUF_Package SHALL include docs/ENV.md and FORMAT.md.
4. WHEN CI runs, THE CI pipeline SHALL build a wheel and run the CPU test suite.
5. THE CI pipeline SHALL run the CPU test suite without requiring an NVIDIA GPU.
6. WHERE a test requires an NVIDIA GPU, THE test suite SHALL skip that test when no NVIDIA GPU is available.

## Windows-Native Extension

This extension supersedes the WSL and GPTQModel assumptions above. Every feature SHALL work on native Windows without WSL and without GPTQModel.

### Extension Glossary

- **Native_GPTQ**: The pure-PyTorch layer-wise sequential GPTQ quantizer in `quf/quant/gptq.py`.
- **Hessian**: The matrix H = 2 X^T X accumulated from calibration activations for one linear layer.
- **Fused_Backend**: The Windows-native GPU Backend that multiplies activations by 4-bit weights without storing a full dequantized weight matrix in GPU memory permanently.
- **Cached_Dequant_Backend**: The Torch_GPU variant that dequantizes each weight once at load and keeps the fp16 weight cached.
- **Results_Doc**: The file docs/RESULTS.md that records measured numbers.
- **Reference_GPU**: The NVIDIA GeForce RTX 5080 with compute capability 12.0.

### Requirement 14: Native GPTQ quantizer

**User Story:** As a model distributor on Windows, I want a pure-PyTorch GPTQ quantizer, so that I can convert models without WSL or GPTQModel.

#### Acceptance Criteria

1. WHEN a linear layer weight and calibration activations are provided, THE Native_GPTQ SHALL accumulate the Hessian, apply damping, and return a QuantizedWeight with Group_Size 128 in the canonical Packed_Int4 layout.
2. WHERE the scheme is asymmetric, THE Native_GPTQ SHALL produce scales and zero points, and WHERE the scheme is symmetric, THE Native_GPTQ SHALL produce scales with zero point 8.
3. WHEN a CUDA device is available, THE Native_GPTQ SHALL run on CUDA, and WHEN no CUDA device is available, THE Native_GPTQ SHALL run on the CPU.
4. WHEN a model is converted, THE Converter SHALL quantize decoder layers sequentially, feeding each layer the activations produced by the already quantized preceding layers.
5. WHEN `quf convert` is run on native Windows with a Hugging Face Qwen identifier, THE Converter SHALL write a QUF_File using transformers and gigatoken-tokenized calibration data.
6. THE QUF_Package SHALL declare the `convert` extra without GPTQModel.
7. FOR ALL small random layers and calibration data, THE Native_GPTQ SHALL produce a layer output error on the calibration data less than or equal to the error of round-to-nearest quantization with the same scheme (metamorphic property).
8. FOR ALL weight matrices whose input dimension is a multiple of 128, THE Native_GPTQ SHALL return integer values in 0 to 15, fp16 scales of shape `[out, in/128]`, and a reference-dequantized matrix of the original shape.
9. IF the Hessian is not positive definite after damping, THEN THE Native_GPTQ SHALL increase the damping and retry, and IF the retries are exhausted, THEN THE Native_GPTQ SHALL raise an error naming the layer.

### Requirement 15: Windows-native GPU speed path

**User Story:** As a Windows user with an NVIDIA GPU, I want a GPU Backend that builds and runs natively, so that I get fast generation without Marlin.

#### Acceptance Criteria

1. THE Backend_Selector SHALL select among Fused_Backend, Cached_Dequant_Backend, and CPU without requiring GPTQModel or Marlin.
2. WHEN the Fused_Backend is selected, THE Fused_Backend SHALL compute outputs within an absolute tolerance of 1e-2 in fp16 of the matrix multiply with Reference_Dequantizer weights.
3. WHERE `torch.compile` is available and enabled, THE Fused_Backend SHALL use it, and WHERE it is unavailable, THE Fused_Backend SHALL run in eager PyTorch.
4. WHEN the Fused_Backend is not faster than the Cached_Dequant_Backend on the Reference_GPU, THE Backend_Selector SHALL select the Cached_Dequant_Backend and THE Results_Doc SHALL record the measured tokens per second of each.
5. IF a selected GPU Backend raises an error at load or during its self-test, THEN THE Backend_Selector SHALL log the error and select the Cached_Dequant_Backend.
6. THE QUF_Package SHALL keep the existing Torch_GPU output unchanged, so that the Property 10 equivalence test passes on the Reference_GPU.
7. THE Marlin Backend SHALL be optional and SHALL NOT be required by any installation extra.

### Requirement 16: Real-model verification on Windows

**User Story:** As a maintainer, I want measured results from real models on Windows, so that the documented numbers are real.

#### Acceptance Criteria

1. WHEN Qwen/Qwen2.5-0.5B is converted on native Windows, THE Converter SHALL produce a QUF_File that `quf run` loads on CPU and on CUDA.
2. WHEN `quf run` is run with a prompt on the converted file, THE Runtime SHALL print a generated text sample.
3. WHEN `quf bench` is run, THE Benchmark_Suite SHALL report WikiText-2 Perplexity of the QUF_File and of the fp16 model, the file size, and tokens per second.
4. THE Results_Doc and the README SHALL record the measured numbers, the device names, and the software versions.
5. WHERE time permits, THE maintainer SHALL repeat conversion and benchmarking with Qwen/Qwen2.5-1.5B and record the numbers.
6. IF gigatoken cannot be installed or run on Windows, THEN THE Tokenizer_Wrapper SHALL use the Hugging Face tokenizers fallback and THE Results_Doc SHALL document the failure.

### Requirement 17: PyPI packaging

**User Story:** As a user, I want to install `quf` from PyPI with pip, so that I can use it without cloning the repository.

#### Acceptance Criteria

1. THE QUF_Package SHALL declare torch as a base dependency.
2. THE QUF_Package SHALL declare the MIT license, include a LICENSE file, list the author vtrsharvin-sj, declare the project URL https://github.com/vtrsharvin-sj/Quantized-Model-Format, and declare classifiers.
3. THE README SHALL state that QUF runs natively on Windows and SHALL contain no WSL instructions.
4. WHEN the package is built, THE build SHALL produce an sdist and a wheel that pass `twine check`.
5. WHEN the wheel is installed into a fresh virtual environment, THE `quf --help` and `quf info` commands SHALL run successfully.
6. THE packaging task SHALL NOT upload to PyPI.

## Accuracy Upgrade Extension (release 0.2.0)

Release 0.1.0 is published. Native GPTQ on Qwen2.5-0.5B (4-bit, group size 128) measures WikiText-2 perplexity of 14.04 to 14.18 against 13.07 for fp16, an increase of 7.4 to 8.5 percent. The documented target is at most 3 percent. A 4-bit model cannot score a lower perplexity than fp16, so this extension narrows the gap as far as the measurements allow. The Results_Doc SHALL report the measured numbers whether or not the 3 percent target is reached.

### Accuracy Glossary

- **Accuracy_Doc**: The file docs/ACCURACY.md that records diagnosis findings and the perplexity result of each upgrade.
- **Sensitivity_Analyzer**: The component that measures per-layer and per-module quantization error and the perplexity impact of quantizing a subset of modules.
- **Ablation_Harness**: The part of the Sensitivity_Analyzer that quantizes a chosen subset of modules and keeps all others in fp16, then measures Perplexity.
- **Act_Order**: GPTQ column processing in descending order of Hessian diagonal, also called desc_act.
- **G_Idx**: An int32 tensor of length `in` that maps each input column to its group index, stored per quantized linear layer when Act_Order is used.
- **MSE_Clipping**: A per-group search over scale (and zero point) candidates that minimizes the squared weight error of the group.
- **Mixed_Precision**: A quantization plan that stores selected sensitive modules as Int8_Tensor or fp16 and all others as Packed_Int4.
- **Int8_Tensor**: A quantized weight stored as int8 values with fp16 per-output-channel or per-group Scales.
- **RTN_Baseline**: Round-to-nearest quantization from `quantize_groupwise` with Group_Size 128 and no calibration.
- **GGUF_Baseline**: A GGUF Q4_K_M or Q4_0 model evaluated through llama-cpp-python.
- **Upgrade**: One of Act_Order, MSE_Clipping, a non-default Group_Size, Mixed_Precision, or a new calibration default.
- **Release_0_2_0**: The `quf` package version 0.2.0.

### Requirement 18: Accuracy diagnosis

**User Story:** As a maintainer, I want to know which layers and modules cause the perplexity gap, so that I choose upgrades by evidence.

#### Acceptance Criteria

1. WHEN the Sensitivity_Analyzer is run on a model and calibration data, THE Sensitivity_Analyzer SHALL report for every decoder layer and every linear module the relative output error of the 4-bit quantized module on the calibration data.
2. WHEN the Ablation_Harness is given a subset of modules, THE Ablation_Harness SHALL quantize only that subset, keep every other module in fp16, and report the WikiText-2 Perplexity.
3. THE Ablation_Harness SHALL measure the Perplexity impact of quantizing each of these subsets alone: the first decoder layer, the last decoder layer, all `v_proj` modules, all `down_proj` modules, and all remaining linear modules.
4. WHERE the model ties the embedding and lm_head weights, THE Sensitivity_Analyzer SHALL report whether the tied weight is kept in fp16 in the converted file.
5. THE Accuracy_Doc SHALL record the result of auditing `quf/quant/gptq.py` for Hessian accumulation correctness, damping, calibration data distribution and length, sequential ordering, fp16 scale rounding, and tied embedding handling.
6. IF the audit finds a defect in `quf/quant/gptq.py`, THEN THE maintainer SHALL fix the defect, add a regression test, and record the Perplexity before and after the fix in the Accuracy_Doc.
7. THE Accuracy_Doc SHALL list the modules ranked by measured Perplexity impact and name the Upgrades selected from that ranking.

### Requirement 19: Configurable group size and backward-compatible format

**User Story:** As a model distributor, I want to choose the group size and keep older files readable, so that I can trade size for accuracy without breaking existing users.

#### Acceptance Criteria

1. WHEN `quf convert` is run with a group size option, THE Converter SHALL accept the values 32, 64, and 128 and record the chosen value in the quantization scheme of the Header.
2. THE Writer SHALL emit a file as version 1 when the file uses Group_Size 128, no G_Idx, and no Int8_Tensor, so that readers of release 0.1.0 read the file unchanged.
3. THE Writer SHALL emit a file as version 2 when the file uses a Group_Size other than 128, a G_Idx, or an Int8_Tensor.
4. THE Reader SHALL read version 1 files and version 2 files.
5. IF the Header records a Group_Size other than 32, 64, or 128, THEN THE Reader SHALL raise an error stating the recorded value and the supported values.
6. IF the input dimension of a quantized module is not a multiple of the Group_Size, THEN THE Quantizer SHALL raise an error stating the dimension and the Group_Size.
7. WHERE a version 1 file omits the group size field from the Header, THE Reader SHALL treat the Group_Size as 128.
8. THE QUF_Package SHALL document the version 2 additions in FORMAT.md, including the G_Idx tensor, the Int8_Tensor layout, and the group size field.

### Requirement 20: Act-order with stored permutation

**User Story:** As a model distributor, I want act-order quantization, so that columns with large activations are quantized with less error.

#### Acceptance Criteria

1. WHERE Act_Order is enabled, THE Native_GPTQ SHALL process input columns in descending order of the Hessian diagonal.
2. WHERE Act_Order is enabled, THE Writer SHALL store a G_Idx tensor of dtype int32 and length `in` for each affected quantized linear module.
3. FOR ALL valid G_Idx tensors, writing a QUF_File and reading it SHALL return a bit-identical G_Idx (round-trip property).
4. WHEN a G_Idx is present, THE Reference_Dequantizer SHALL select the scale and zero point of each input column by the value of G_Idx at that column.
5. THE CPU, Cached_Dequant_Backend, and Fused_Backend SHALL compute outputs within an absolute tolerance of 1e-2 in fp16 of the multiplication with Reference_Dequantizer weights for modules with and without a G_Idx.
6. IF a G_Idx value is outside the range 0 to the group count minus 1, THEN THE Reader SHALL raise an error naming the tensor.

### Requirement 21: MSE-optimal clipping

**User Story:** As a model distributor, I want a per-group scale search, so that outlier weights do not inflate the quantization step.

#### Acceptance Criteria

1. WHERE MSE_Clipping is enabled, THE Quantizer SHALL evaluate a fixed grid of clipping ratios for each group and select the ratio that minimizes the squared weight error of that group.
2. FOR ALL weight groups, THE squared error of the MSE_Clipping result SHALL be less than or equal to the squared error of the unclipped round-to-nearest result for the same scheme (metamorphic property).
3. WHERE MSE_Clipping is enabled, THE Native_GPTQ SHALL use the searched scales for each group while propagating the quantization error.
4. THE output of MSE_Clipping SHALL remain in the canonical Packed_Int4 layout, so that all Backends read the result unchanged.

### Requirement 22: Mixed precision with size accounting

**User Story:** As a model distributor, I want to keep the most sensitive modules at higher precision, so that I can recover accuracy at a known size cost.

#### Acceptance Criteria

1. WHERE Mixed_Precision is enabled, THE Converter SHALL keep the modules selected by the diagnosis at Int8_Tensor or fp16 and quantize every other linear module to Packed_Int4.
2. THE Converter SHALL accept a module selection by name pattern, so that the Accuracy_Doc diagnosis can be applied without code changes.
3. WHEN a QUF_File with an Int8_Tensor is run, THE Reference_Dequantizer SHALL return `int8 value * scale` in fp16 of the original shape.
4. THE CPU, Cached_Dequant_Backend, and Fused_Backend SHALL compute outputs for Int8_Tensor modules within an absolute tolerance of 1e-2 in fp16 of the multiplication with Reference_Dequantizer weights.
5. WHEN conversion completes, THE Converter SHALL print the bytes used by Packed_Int4 modules, Int8_Tensor modules, fp16 modules, and the total, and the ratio to the fp16 checkpoint size.
6. WHEN `quf info` is run on a mixed precision file, THE CLI SHALL print the precision of each module group and the byte totals.

### Requirement 23: Calibration defaults

**User Story:** As a model distributor, I want larger and more varied calibration data by default, so that the Hessians represent real text.

#### Acceptance Criteria

1. WHERE the user gives no calibration options, THE Converter SHALL build 128 samples of sequence length 2048.
2. THE Calibration_Builder SHALL accept more than one text source and SHALL draw a documented share of the samples from each source.
3. WHEN the same sources, seed, and parameters are provided twice, THE Calibration_Builder SHALL return identical samples.
4. IF the sources yield fewer tokens than requested, THEN THE Calibration_Builder SHALL raise an error stating the shortfall.
5. THE Accuracy_Doc SHALL record the Perplexity of the old defaults and the new defaults measured on the same model.

### Requirement 24: Evaluation of upgrades and release numbers

**User Story:** As a maintainer, I want every upgrade measured against baselines, so that only upgrades that help are kept and claims match measurements.

#### Acceptance Criteria

1. WHEN an Upgrade is evaluated, THE maintainer SHALL measure WikiText-2 Perplexity with and without the Upgrade on Qwen2.5-0.5B with all other settings fixed and SHALL record both values in the Accuracy_Doc.
2. IF an Upgrade does not lower the measured Perplexity, THEN THE Converter SHALL leave that Upgrade disabled by default and the Accuracy_Doc SHALL state the measured result.
3. THE Results_Doc SHALL report, for Qwen2.5-0.5B and Qwen2.5-1.5B, the Perplexity, file size, and tokens per second of fp16, the 0.1.0 settings, the 0.2.0 default settings, and the RTN_Baseline.
4. WHERE llama-cpp-python installs on Windows without a source build of more than 10 minutes, THE Results_Doc SHALL report GGUF_Baseline Q4_K_M and Q4_0 Perplexity and file size, and WHERE it does not install, THE Results_Doc SHALL state that the comparison was not run and give the reason.
5. WHERE lm-evaluation-harness is installed, THE Results_Doc SHALL report accuracy on one or two tasks for fp16 and the 0.2.0 default settings.
6. THE Results_Doc and the README SHALL report the relative Perplexity increase against fp16 and state whether the 3 percent target was reached, using only measured values.

### Requirement 25: Release 0.2.0

**User Story:** As a maintainer, I want a verified 0.2.0 build, so that the release can be uploaded with confidence.

#### Acceptance Criteria

1. THE QUF_Package SHALL declare version 0.2.0 in `pyproject.toml` and in `quf.__version__`.
2. THE QUF_Package SHALL include a changelog entry for 0.2.0 listing the Upgrades kept and the measured Perplexity change.
3. WHEN the package is built, THE build SHALL produce an sdist and a wheel that pass `twine check`.
4. WHEN the wheel is installed into a fresh virtual environment, THE `quf --help` and `quf info` commands SHALL run successfully on a version 1 file and on a version 2 file.
5. WHEN the test suite `pytest -m "not gpu and not convert"` is run, THE test suite SHALL pass.
6. THE release tasks SHALL NOT upload to PyPI and SHALL NOT write any access token to disk.

## Verification Extension (release 0.2.1)

### Glossary additions

- **Llama_Perplexity_Tool**: The prebuilt `llama-perplexity.exe` binary from the llama.cpp GitHub releases, used as the reference for GGUF perplexity.
- **Eval_Protocol**: The window length, stride, scored token range, and BOS handling used to compute Perplexity.
- **LM_Eval_Adapter**: The class in `quf/bench/lm_eval_adapter.py` that exposes a QUF model to lm-evaluation-harness.
- **Weight_Cache**: The per-module cached fp16 weight tensor held by the Torch_GPU Backend after preparation.
- **Release_0_2_1**: The `quf` package version 0.2.1.

### Requirement 26: Verified GGUF comparison

**User Story:** As a maintainer, I want GGUF rows measured with a validated protocol, so that the comparison against QUF is fair.

#### Acceptance Criteria

1. WHEN the GGUF perplexity path in `scripts/bench_accuracy.py` is run on a GGUF file, THE script SHALL report a Perplexity within 1 percent of the Llama_Perplexity_Tool on the same text, context length, and model file.
2. WHEN the GGUF f16 or Q8_0 file is evaluated, THE script SHALL report a Perplexity within 0.5 percent of the fp16 Hugging Face model under the same Eval_Protocol.
3. THE script SHALL evaluate GGUF and QUF models under one Eval_Protocol, or SHALL evaluate QUF under the llama.cpp protocol as well and label both.
4. THE Results_Doc SHALL report Q4_K_M, Q4_0, and Q8_0 for Qwen2.5-0.5B and Qwen2.5-1.5B with file size and tokens per second, naming the device class used for each row.
5. IF the Llama_Perplexity_Tool cannot be downloaded or run, THEN THE Results_Doc SHALL record the GGUF rows as not run with the reason.

### Requirement 27: Batched lm-evaluation-harness adapter

**User Story:** As a maintainer, I want a fast lm-eval adapter, so that task accuracy can be measured in reasonable time.

#### Acceptance Criteria

1. WHEN loglikelihood requests are received, THE LM_Eval_Adapter SHALL process them in padded batches on the active device.
2. WHEN several requests share the same context, THE LM_Eval_Adapter SHALL compute that context once.
3. FOR ALL request lists, THE batched LM_Eval_Adapter SHALL return the same log-likelihoods, within 1e-3 absolute tolerance, as processing each request alone.
4. THE Results_Doc SHALL report hellaswag and arc_easy accuracy with standard error for fp16 and the QUF defaults on Qwen2.5-0.5B, using limit 500 and a fixed seed.

### Requirement 28: Stable decode speed

**User Story:** As a maintainer, I want decode speed measured stably and free of avoidable slowdowns, so that reported tokens per second are trustworthy.

#### Acceptance Criteria

1. WHEN a tokens per second value is measured, THE benchmark SHALL run a warm-up and at least 5 timed repeats and report the median.
2. WHEN the Torch_GPU Backend prepares a module with a G_Idx tensor or an Int8_Tensor, THE Backend SHALL store the Weight_Cache as an fp16 tensor and SHALL NOT regather or dequantize that module on each call.
3. THE Results_Doc SHALL report re-measured tokens per second for every row of both models.

### Requirement 29: Repository and continuous integration

**User Story:** As a maintainer, I want the project on GitHub with passing CI, so that contributors can trust the build.

#### Acceptance Criteria

1. THE repository SHALL exclude `.quf` files, eval_cache, dist, logs, virtual environments, and secrets through `.gitignore`.
2. WHEN the first commit is prepared, THE maintainer script SHALL scan the staged files for the string `pypi-` and stop on any match.
3. IF the remote repository already contains commits, THEN THE push task SHALL stop and report without overwriting.
4. THE push task SHALL NOT force-push and SHALL NOT write credentials to any file.
5. WHEN the GitHub Actions workflow runs, THE workflow SHALL pass on ubuntu-latest and windows-latest.

### Requirement 30: Release 0.2.1

**User Story:** As a maintainer, I want a verified 0.2.1 build, so that the orchestrator can upload it.

#### Acceptance Criteria

1. WHERE code changed in Requirements 26 to 28, THE QUF_Package SHALL declare version 0.2.1 in `pyproject.toml` and `quf.__version__`, with a changelog entry.
2. THE build SHALL produce an sdist and a wheel that pass `twine check`, and the installed wheel SHALL run `quf --help` and `quf info`.
3. THE README, docs, and RESULTS.md SHALL contain only measured numbers and SHALL drop the "not run (unverified)" marker from every row now verified.
4. THE release tasks SHALL NOT upload to PyPI.

## Speed and Size Extension (release 0.2.1)

QUF decode on Torch_GPU measures about 40 to 50 tokens per second for Qwen2.5-0.5B and Qwen2.5-1.5B, against about 460 to 560 (0.5B) and 140 to 230 (1.5B) tokens per second for llama.cpp GGUF on the same GPU. Profiling in task 22.1 shows decode is CPU kernel-launch bound, at about 25 ms of CPU time against 6 ms of GPU time per token. Perplexity is already competitive under the llama protocol: 0.5B v020 scores 12.033 against 12.043 for Q4_K_M, and 1.5B v020 scores 8.395 against 8.600. File size is the other gap: v020 is 540 MB against 398 MB for Q4_K_M on 0.5B, and 1413 MB against 986 MB on 1.5B. This extension targets decode speed and file size without losing precision. The Results_Doc SHALL report the measured numbers whether or not the targets are reached.

### Glossary additions

- **Static_KV_Cache**: A KV_Cache whose key and value tensors are allocated once for a fixed maximum sequence length and updated in place.
- **Decode_Step**: The single-token forward pass that reads the Static_KV_Cache and returns the next-token logits, with tensor shapes that do not change between calls.
- **CUDA_Graph_Decoder**: The Torch_GPU path that captures the Decode_Step with `torch.cuda.graphs` and replays it for each generated token.
- **Eager_Decoder**: The Torch_GPU and CPU path that runs the Decode_Step as ordinary PyTorch calls.
- **Fused_Projection**: A single linear layer built at load time by concatenating the cached weights of the query, key, and value projections, or of the gate and up projections.
- **Sensitive_Preset**: The v020 mixed precision plan that keeps selected modules at int8.
- **Size_Candidate**: One alternative to the Sensitive_Preset evaluated in Requirement 32.
- **Llama_Protocol**: The Eval_Protocol that matches llama-perplexity.exe, selected with `--protocol llama`.
- **Speed_Target**: A median decode rate of at least 3 times the measured pre-change rate, on each of the two models.

### Requirement 31: Fast decode path

**User Story:** As a user with an NVIDIA GPU, I want decode to run close to GPU speed, so that generation is fast.

#### Acceptance Criteria

1. THE Runtime SHALL allocate a Static_KV_Cache once per generation session with a fixed maximum sequence length and update the Static_KV_Cache in place.
2. THE Runtime SHALL execute each generated token through a Decode_Step whose tensor shapes are identical on every call.
3. WHEN the Torch_GPU Backend starts a generation session, THE CUDA_Graph_Decoder SHALL capture the Decode_Step with `torch.cuda.graphs` and replay the capture for each subsequent token.
4. IF CUDA graph capture fails, THEN THE Runtime SHALL log the failure at INFO level and continue with the Eager_Decoder.
5. WHEN the Torch_GPU Backend prepares a layer, THE Backend SHALL build a Fused_Projection for query, key, and value, and a Fused_Projection for gate and up, from the Weight_Cache.
6. FOR ALL prompts, prompt lengths, generation lengths, and the shapes of both Qwen2.5-0.5B and Qwen2.5-1.5B, THE CUDA_Graph_Decoder SHALL select the same greedy tokens as the Eager_Decoder and SHALL produce logits within an absolute tolerance of 1e-2 (equivalence property).
7. FOR ALL prompts, THE Fused_Projection output SHALL equal the concatenation of the separate projection outputs within an absolute tolerance of 1e-3.
8. WHERE `torch.compile` runs on Windows without Triton, THE Torch_GPU Backend SHALL offer a compiled Decode_Step as an optional path.
9. IF `torch.compile` does not run on Windows without Triton, THEN THE Results_Doc SHALL record the path as skipped with the reason.
10. THE CPU Backend SHALL use the Static_KV_Cache and SHALL produce the same logits as before the change within an absolute tolerance of 1e-2.
11. WHEN the decode rate is measured, THE benchmark SHALL use the median timing helper from Requirement 28 on Qwen2.5-0.5B v020 and Qwen2.5-1.5B v020 and SHALL record the rates for Eager_Decoder and CUDA_Graph_Decoder.
12. THE Results_Doc SHALL report whether each model reached the Speed_Target and SHALL state the measured speedup when the Speed_Target is not reached.

### Requirement 32: Smaller files at equal precision

**User Story:** As a model distributor, I want smaller files at the same perplexity, so that QUF is competitive with GGUF on size.

#### Acceptance Criteria

1. THE evaluation script SHALL measure file size and WikiText-2 Perplexity under the Llama_Protocol on Qwen2.5-0.5B and Qwen2.5-1.5B for the Sensitive_Preset and for each Size_Candidate.
2. THE Size_Candidate set SHALL include int4 with group size 32 on the last layer only, and down_proj at group size 32 int4 with act-order in place of int8.
3. WHERE an int6 or int5 group-wise option can be added to the format and the Backends without changing the Packed_Int4 layout of existing tensors, THE evaluation SHALL include that option as a Size_Candidate.
4. WHEN a Size_Candidate has a WikiText-2 Perplexity within 1 percent of the v020 Perplexity on both models and a smaller file size than v020 on both models, THE converter SHALL adopt the candidate with the smallest file size as the default preset.
5. IF no Size_Candidate meets the criteria in 32.4, THEN THE converter SHALL keep the v020 preset as the default and THE Results_Doc SHALL record the measured size and Perplexity of every Size_Candidate.
6. WHERE a new tensor layout or Header field is added, THE Writer and Reader SHALL round-trip the new layout bit-identically, THE Reader SHALL read every older file unchanged, and FORMAT.md SHALL document the addition.
7. WHERE a new quantization bit width is added, THE Backends SHALL dequantize the new layout to the same values as the reference dequantizer within an absolute tolerance of 1e-3.
8. THE Results_Doc SHALL report the size and Perplexity table of all Size_Candidates and the default chosen.
