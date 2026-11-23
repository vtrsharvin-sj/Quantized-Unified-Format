# QUF file format (versions 1, 2 and 3)

QUF (Quantized Unified Format) is a single-file container for a 4-bit quantized decoder-only LLM. One file holds the model configuration, tokenizer information, the quantization scheme, and every tensor.

This document matches the reference writer and reader in `quf/format/`. All multi-byte integers are little-endian.

## 1. Byte layout

```
offset  size   field
0       4      magic: the ASCII bytes "QUF1"
4       4      version: uint32 (1, 2 or 3, see sections 8 and 9)
8       8      header_len: uint64, byte length of the JSON header
16      N      header: UTF-8 JSON, N = header_len
16+N    pad    zero bytes up to the next multiple of 64
...            tensor data region
```

The fixed prefix (magic, version, header_len) is 16 bytes. The JSON header follows immediately.

Tensor data starts at the first multiple of 64 at or after `16 + header_len`. Each tensor is stored at its `offset` from the Tensor_Table. Rules:

- `offset` is an absolute file offset, counted from byte 0 of the file.
- Every `offset` is a multiple of 64.
- Gaps between tensors, and after the last tensor, are zero padding up to the next multiple of 64.
- A tensor occupies `prod(shape) * element_size` bytes, stored C-contiguous (row-major) with no per-tensor framing. The size is computed from `dtype` and `shape` and is not stored.
- Tensors are written in Tensor_Table order, so offsets increase.

Because a tensor's `offset` is a multiple of 64 and elements are at most 4 bytes, tensors can be memory-mapped and viewed in place without copying.

## 2. Header JSON

The header is a single JSON object. The writer emits it compactly (no extra whitespace); readers must accept any valid JSON.

```json
{
  "model_config": {
    "architecture": "Qwen2ForCausalLM",
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": true
  },
  "tokenizer": {
    "model_id": "Qwen/Qwen2.5-0.5B",
    "eos_token_id": 151643,
    "bos_token_id": null,
    "files": { "tokenizer.json": "<embedded string or tensor name>" }
  },
  "quantization": {
    "method": "gptq",
    "bits": 4,
    "group_size": 128,
    "symmetric": false,
    "pack_dtype": "int32",
    "pack_order": "lsb_first"
  },
  "tensors": [
    { "name": "model.layers.0.self_attn.q_proj.qweight", "dtype": "int32",
      "shape": [112, 896], "offset": 4096 }
  ]
}
```

### 2.1 Top-level fields

All four fields are required. A missing field is a header schema error.

| Field | Type | Meaning |
|---|---|---|
| `model_config` | object | Architecture and dimensions needed to build the decoder (see the example). |
| `tokenizer` | object | Tokenizer information: `model_id`, `eos_token_id`, `bos_token_id`, and `files`. |
| `quantization` | object | The quantization scheme. |
| `tensors` | list | The Tensor_Table, one entry per tensor. |

`model_config`, `tokenizer`, and `quantization` must be JSON objects. `tensors` must be a list.

### 2.2 `quantization`

| Field | Value in v1 | Meaning |
|---|---|---|
| `group_size` | `128` | Required when writing. 32, 64, and 128 are supported; version 1 files always use 128. A reader treats a missing value as 128. |
| `bits` | `4` | Bits per weight. |
| `symmetric` | `true` / `false` | When `false`, each quantized layer has a `.qzeros` tensor. Treated as `false` if absent. |
| `method` | e.g. `"gptq"` | How the weights were produced. Informational. |
| `pack_dtype` | `"int32"` | Container type of Packed_Int4 values. |
| `pack_order` | `"lsb_first"` | The canonical order in section 4. |

### 2.3 Tensor_Table entries

Each entry has these required fields:

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Unique tensor name. |
| `dtype` | string | One of the codes in section 3. |
| `shape` | list of non-negative integers | Dimensions, outermost first. |
| `offset` | non-negative integer | Absolute file offset, a multiple of 64. |

Tensor names must be unique within a file.

## 3. Data types

| Code | Element | Size |
|---|---|---|
| `fp16` | IEEE 754 half, little-endian | 2 bytes |
| `fp32` | IEEE 754 single, little-endian | 4 bytes |
| `int32` | signed 32-bit integer, little-endian | 4 bytes |
| `int8` | signed 8-bit integer (version 2) | 1 byte |

An unknown dtype code is a header schema error.

## 4. Canonical Packed_Int4 order

Eight 4-bit values (each 0 to 15) are packed into one `int32`. Value `i` of a group of eight occupies bits `4*(i % 8)` through `4*(i % 8) + 3`, so value 0 is in the least significant nibble:

```
packed = v0 | (v1 << 4) | (v2 << 8) | (v3 << 12)
       | (v4 << 16) | (v5 << 20) | (v6 << 24) | (v7 << 28)
```

Consecutive values are consecutive elements along the packed dimension. Runtimes that need another layout (for example Marlin) repack in memory at load time. The file is never rewritten.

## 5. Tensor naming and layout

Names follow Hugging Face Qwen2 conventions, for example `model.layers.0.self_attn.q_proj`.

For each quantized linear layer `<name>` with weight shape `[out, in]`, where `in % 128 == 0` and `out % 8 == 0`:

| Tensor | dtype | shape | Notes |
|---|---|---|---|
| `<name>.qweight` | `int32` | `[in/8, out]` | Packed_Int4 along the input dimension, in the order of section 4. |
| `<name>.scales` | `fp16` | `[in/128, out]` | One scale per group of 128 input elements. |
| `<name>.qzeros` | `int32` | `[in/128, out/8]` | Packed_Int4 zero points along the output dimension. Present only when `symmetric` is `false`. |

Dequantization of element `(o, i)` with group `g = i // 128`:

```
w[o, i] = (q[o, i] - zero[g, o]) * scales[g, o]
```

where `q` is the unpacked 4-bit value and `zero` is the unpacked `qzeros` value, or the constant `8` when the scheme is symmetric. The result is fp16 with the original shape `[out, in]`.

Tensors that are not quantized linear weights are stored unquantized:

- `model.embed_tokens.weight` and `lm_head.weight` (when not tied) must be `fp16`. The writer rejects any other dtype for names ending in `embed_tokens.weight` or `lm_head.weight`.
- Normalization weights and biases are stored as `fp16`.

The writer requires every `.qweight` to have an `fp16` `.scales` tensor, and a `.qzeros` tensor when the scheme is asymmetric.

## 6. Reader validation

A reader opens the file read-only and validates in this order, stopping at the first failure:

1. **Magic.** The first 4 bytes must be `QUF1`. Otherwise: invalid magic error, reporting the bytes found.
2. **Version.** A version greater than the highest supported version (3) is an unsupported version error that reports the file's version and the supported version. A file too short to hold the version field is a truncated header error.
3. **Header length.** The file must hold the 16-byte prefix, and `16 + header_len` must not exceed the file size. Otherwise: truncated header error.
4. **JSON parse.** The header bytes must be valid UTF-8 JSON. Otherwise: header schema error.
5. **Required fields.** All top-level and per-tensor required fields must be present and well typed. Otherwise: header schema error naming the field.
6. **Tensor bounds.** For each tensor, `offset` must be a multiple of 64 and `offset + nbytes` must not exceed the file size. Otherwise: tensor bounds error naming the tensor. Duplicate names are a header schema error.
7. **Version 2 fields.** `group_size` must be 32, 64, or 128 (a missing value is read as 128), `quantization.modules` values must be `int4`, `int8`, or `fp16`, and each `<module>.g_idx` must be int32 with length equal to the module's input dimension, with every value in `0` to `in/group_size - 1`. Otherwise: header schema error stating the offending value and, for group size and precision, the supported values, or naming the g_idx tensor.

All of these errors derive from `QUFFormatError`. Reads return memory-mapped, zero-copy views of the tensor data.

## 7. Compatibility

- A reader must reject any file whose version is higher than it supports.
- Version 1 files use `group_size` 128, no g_idx, and no int8 tensors. Release 0.1.0 reads only these files.
- Unknown extra keys in header objects are ignored by the reference reader, which leaves room for additive changes.

## 8. Version 2 additions

The writer picks the version from the features a file uses. A file is version 1 exactly when it has `group_size` 128, no `g_idx` tensor, no `int8` tensor, and no `int8` or `fp16` entry in `quantization.modules`. Every other file is version 2. Version 1 files are byte-compatible with release 0.1.0. A 0.1.0 reader rejects a version 2 file with the unsupported version error. This is intended.

| Feature | Header field | Tensors |
|---|---|---|
| Group size 32 or 64 | `quantization.group_size` | `<name>.scales` is `[in/group_size, out]`, `<name>.qzeros` is `[in/group_size, out/8]` |
| Act-order | `quantization.act_order = true` | `<name>.g_idx` |
| Int8 module | `quantization.modules[<name>] = "int8"` | `<name>.qweight_int8`, `<name>.scales` |
| fp16 module | `quantization.modules[<name>] = "fp16"` | `<name>.weight` |

`quantization.modules` is optional. It maps a module name (such as `model.layers.0.mlp.down_proj`) to `"int4"`, `"int8"`, or `"fp16"`. A module that is absent from the map is `int4`.

### 8.1 Group size

`group_size` is one of 32, 64, or 128. The input dimension of each int4 module must be a multiple of it. Smaller groups store more scales and zeros and usually lose less accuracy.

### 8.2 G_Idx tensor

With act-order, columns are quantized in descending order of the Hessian diagonal, so the group of a column is not `i // group_size`. The file stores the group of each column:

| Tensor | dtype | shape | Notes |
|---|---|---|---|
| `<name>.g_idx` | `int32` | `[in]` | `g_idx[i]` is the group of input column `i`, in `0` to `in/group_size - 1`. |

Weights stay in the original column order, so the runtime needs no input permutation. With a `g_idx`, element `(o, i)` is dequantized with `g = g_idx[i]` instead of `g = i // group_size`:

```
w[o, i] = (q[o, i] - zero[g, o]) * scales[g, o]
```

### 8.3 Int8_Tensor layout

| Tensor | dtype | shape | Notes |
|---|---|---|---|
| `<name>.qweight_int8` | `int8` | `[out, in]` | Signed values, one per weight. |
| `<name>.scales` | `fp16` | `[out]` | One scale per output channel. |

Dequantization is `w[o, i] = qweight_int8[o, i] * scales[o]`, returned as fp16 with shape `[out, in]`.

### 8.4 fp16 modules

A module marked `"fp16"` stores its weight as `<name>.weight`, dtype `fp16`, shape `[out, in]`.

## 9. Version 3 additions: Quantized_Embedding

A file is version 3 exactly when it holds a Quantized_Embedding: `quantization.embedding` or `quantization.lm_head` is present, or any tensor ends in `.embed_q4`, `.embed_q8`, or `.embed_scales`. This rule takes precedence over section 8. Every other file keeps version 1 or 2 with unchanged bytes. A 0.3.0 reader (highest version 2) rejects a version 3 file with the unsupported version error. The current reader loads versions 1, 2, and 3, and rejects a version 1 or 2 file that declares a quantized embedding.

Header fields, each an object with the same keys:

| Field | Meaning |
|---|---|
| `quantization.embedding` | The embedding table. `name` is the tensor prefix (such as `model.embed_tokens`), `bits` is 4 or 8, `group_size` is 32 or 64, `method` is `rtn` or `gptq` (default `rtn`). |
| `quantization.lm_head` | Optional. Same keys, for an untied lm_head with its own tensors (`name` such as `lm_head`). When absent and the model ties its weights, the lm_head reuses the `embedding` tensors. |

The embedding `group_size` is independent of `quantization.group_size`.

Tensors for a table named `<name>` with `rows` vocabulary entries and `cols` hidden size:

| Tensor | dtype | shape | Notes |
|---|---|---|---|
| `<name>.embed_q4` (bits 4) | `int8` | `[rows, cols/2]` | Raw bytes. Column `2j` is the low nibble and column `2j+1` the high nibble of byte `j`. Nibbles are unsigned 0 to 15 with zero point 8. |
| `<name>.embed_q8` (bits 8) | `int8` | `[rows, cols]` | Signed values -127 to 127, zero point 0. |
| `<name>.embed_scales` | `fp16` | `[rows, cols/group_size]` | One scale per row and group. |

Dequantization: `w[r, c] = (q[r, c] - zero) * scales[r, c // group_size]`, returned as fp16 `[rows, cols]`. Storage is row-major with one row per token, so a row gather reads `cols/2` (or `cols`) contiguous bytes plus `cols/group_size` scales per requested token, and the tied lm_head GEMV `logits = x @ W.T` walks the same tensors in the same `[out, in]` orientation. `cols` must be a multiple of `group_size`. The reference implementation is `quf.quant.embedding.dequantize_embedding_rows`.

The reader checks the bits, group size, method, tensor presence, dtypes, and shapes, and rejects tensors with an embedding suffix that no header field declares. An fp16 embedding (`<name>.weight`) is unchanged and does not make a file version 3.
