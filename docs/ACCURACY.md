# Accuracy diagnosis (Qwen/Qwen2.5-0.5B, 4-bit, group size 128, asymmetric)

Produced by `python scripts/diagnose_accuracy.py --model Qwen/Qwen2.5-0.5B --out docs/accuracy_diagnosis.json` on the RTX 5080 Laptop GPU (native Windows). Raw data: `accuracy_diagnosis.json` (RTN and GPTQ module errors and ablation perplexities). The GPTQ stage log is `diagnose_gptq.log`. The RTN stage came from an earlier run of the same script and was kept. GPTQ used 128 calibration samples x 1024 tokens from the WikiText-2 train split. Perplexity is WikiText-2 raw test, 2048-token windows, same as `quf bench`.

fp16 perplexity: **13.0703**. Full 4-bit GPTQ model: **14.0320 (+7.36%)** in memory, matching the converted file's 14.0384 (+7.41%) in `RESULTS.md`. The 3% target is missed.

## Audit result (task 15.3)

`tests/test_gptq_audit.py` covers Hessian accumulation, damping, dead columns, GPTQ vs. a naive unblocked implementation, identity-Hessian equals RTN, true-sequential ordering with the full calibration set, and tied-embedding handling. No defect was found in `quf/quant/gptq.py`, so no fix was made and there is no before/after perplexity. The GPTQ error numbers below (about 35% lower than RTN at every module kind) agree with a correct implementation. The gap is a property of 4-bit g128 on this model, not a bug.

## Tied embedding

Qwen2.5-0.5B has `tie_word_embeddings = true`. The converter stores `model.embed_tokens.weight` as fp16, shape [151936, 896] (136.0M parameters, 27.5% of the model), and stores no separate `lm_head`. It is not quantized, so it is not a source of the gap. It is a source of size: it takes about 272 MB of the 469 MB file.

## Per-module error (relative output error on calibration activations)

Mean over 24 layers, by module kind:

| Module | RTN | GPTQ | GPTQ / RTN |
| --- | --- | --- | --- |
| v_proj | 0.1226 | 0.0762 | 0.62 |
| o_proj | 0.1215 | 0.0633 | 0.52 |
| up_proj | 0.1168 | 0.0760 | 0.65 |
| down_proj | 0.0997 | 0.0656 | 0.66 |
| q_proj | 0.0968 | 0.0606 | 0.63 |
| k_proj | 0.0697 | 0.0437 | 0.63 |
| gate_proj | 0.0692 | 0.0451 | 0.65 |

Highest single modules under GPTQ: layer 16 v_proj 0.102, layer 20 v_proj 0.097, layer 18 v_proj 0.096, layer 3 up_proj 0.095, layer 17 v_proj 0.090, layer 22 q_proj 0.089. Under RTN: layer 0 v_proj 0.157, layer 16 v_proj 0.156, layer 18 v_proj 0.149, layer 14 o_proj 0.149.

By layer, mean GPTQ error is 0.044 in layer 0 and 0.052 in layer 23. Per-layer values for all layers are in the JSON. Per-module error is broadly flat across depth, with `v_proj`, `up_proj`, and `o_proj` highest and `k_proj`/`gate_proj` lowest. Error does not predict perplexity impact well, as the next table shows (`gate_proj` has the lowest error, yet `down_proj` has a high impact).

## Ablation: only the subset is quantized, everything else stays fp16

Subsets are disjoint. `first` and `last` are every linear in layer 0 and layer 23. `v_proj` and `down_proj` are those modules in every layer. `rest` is the other 110 linears.

| Subset | Modules | RTN ppl | RTN increase | GPTQ ppl | GPTQ increase | GPTQ ppl added per module |
| --- | --- | --- | --- | --- | --- | --- |
| first layer | 7 | 13.1555 | +0.65% | 13.0988 | +0.22% | 0.0041 |
| last layer | 7 | 13.3909 | +2.45% | 13.1570 | +0.66% | 0.0124 |
| all v_proj | 24 | 13.3583 | +2.20% | 13.2305 | +1.23% | 0.0067 |
| all down_proj | 24 | 13.7666 | +5.33% | 13.3114 | +1.84% | 0.0100 |
| rest | 110 | 14.1645 | +8.37% | 13.4864 | +3.18% | 0.0038 |
| all 168 | 168 | 15.5416 | +18.9% | 14.0320 | +7.36% | 0.0057 |

The subset effects add up almost linearly under GPTQ (0.029 + 0.087 + 0.160 + 0.241 + 0.416 = 0.93 against 0.96 for all together), so interactions are small.

## Modules ranked by perplexity impact (GPTQ, absolute ppl added)

1. rest (110 linears: q, k, o, gate, up in layers 1 to 22): +0.416 (43% of the gap)
2. all down_proj: +0.241 (25%)
3. all v_proj: +0.160 (17%)
4. last layer: +0.087 (9%)
5. first layer: +0.029 (3%)

Per module, the order is: last layer (0.0124), down_proj (0.0100), v_proj (0.0067), first layer (0.0041), rest (0.0038).

## Findings

- The gap is spread across the model. No single subset explains it. Keeping `first`, `last`, all `v_proj`, and all `down_proj` in fp16 would remove 0.52 of the 0.96 ppl added (54%) but would still leave the other 110 linears at +3.18%, which is above the 3% target by itself.
- `down_proj` and the last layer are the most sensitive per module. Under RTN, `down_proj` alone adds +5.33%, and the last layer adds +2.45% from 7 modules. GPTQ cuts those to +1.84% and +0.66%.
- The first layer is not special once GPTQ is used (+0.22%).
- The embedding is fp16 and the tied `lm_head` is not quantized, so neither contributes.
- GPTQ already recovers 61% of the RTN gap (+18.9% to +7.36%). Further gains need a finer or better-fitted quantization grid, not a fix to GPTQ.

## Upgrades selected for task 16.8

In order of expected value per cost. Each should be measured on the same benchmark (`quf bench`, WikiText-2) and the 3% target checked.

1. **Group size 64 and 32** (with `rest` as the main target). The 110 `rest` linears carry 43% of the gap, and a smaller group is the direct way to reduce per-weight error across all modules. Cost: larger scales and zeros. Evaluate 64 first, then 32.
2. **Clip search (MSE clipping)**. Cheap, no format cost, and applies to every module. Evaluate alone and combined with group size 64.
3. **Act-order**. Helps GPTQ most when a few input channels dominate the Hessian, which fits `down_proj` (inputs from the SwiGLU product) and `v_proj`. Evaluate alone and combined with group size.
4. **8-bit for sensitive modules**: all `down_proj` and the last layer (and `v_proj` if budget allows). These three subsets have the highest per-module impact (0.0124, 0.0100, 0.0067 ppl) and cover 51% of the gap with 38 of 168 modules. 8-bit `down_proj` in all layers adds about 52 MB (104.6M parameters x 0.5 byte, computed, scales excluded). Measure the size report from task 16.5 next to perplexity.
5. **fp16 embeddings**: already the default, with no gain available. Keep it as is. The only open question is the size, not accuracy.

Do not expect any single upgrade to reach 3%. The plausible route is a combination: group size 64 plus clip search plus act-order for all modules, with 8-bit `down_proj` and last layer if still short.

## Reproduce

```
python scripts/diagnose_accuracy.py --model Qwen/Qwen2.5-0.5B --out docs/accuracy_diagnosis.json
```

The GPTQ stage took about 20 minutes on the RTX 5080 Laptop (module errors about 3 minutes, each ablation 1 to 4 minutes).

## Upgrade evaluation (task 16.8)

Produced by `python scripts/eval_upgrades.py --drive`. Raw data: `upgrade_eval.json`. Every run converts Qwen2.5-0.5B with the native GPTQ quantizer, using the same cached calibration tokens (wikitext-2 train, 128 x 1024, seed 0), and measures WikiText-2 test perplexity with the `quf bench` code (2048-token windows, `torch_gpu` backend). Each conversion ran in its own process with a 25 minute cap; none hit it. fp16 perplexity is 13.0703. The reference run (group size 128, no act-order, no clip search, all int4) reproduced the earlier 14.0384 (+7.41%) exactly.

| Run | Settings changed from the reference | Perplexity | vs fp16 | File size (bytes) |
| --- | --- | --- | --- | --- |
| baseline | none | 14.0384 | +7.41% | 469,483,072 |
| g64 | group size 64 | 13.8412 | +5.90% | 476,472,128 |
| g32 | group size 32 | 13.6386 | +4.35% | 490,449,792 |
| clip | clip search | 14.0630 | +7.60% | 469,483,072 |
| actorder | act-order | 13.9863 | +7.01% | 470,482,432 |
| g32_clip | g32 + clip | 13.5528 | +3.69% | 490,449,792 |
| g32_actorder | g32 + act-order | 13.5543 | +3.70% | 491,449,152 |
| clip_actorder | clip + act-order (g128) | 13.9251 | +6.54% | 470,482,432 |
| all3_g32 | g32 + clip + act-order | 13.4777 | +3.12% | 491,449,152 |
| best_down_int8 | all3_g32 + `down_proj` int8 | 13.4056 | +2.57% | 535,153,472 |
| best_sensitive | all3_g32 + `down_proj` and last layer int8 | **13.3775** | **+2.35%** | 539,605,632 |
| best_newcalib | all3_g32 with wikitext-2 + C4, 128 x 2048 | 13.5492 | +3.66% | 491,449,152 |

### Decisions

A setting is a default only if it lowered perplexity by more than 0.02 against its reference run.

| Upgrade | Before | After | Default |
| --- | --- | --- | --- |
| Group size 64 | 14.0384 | 13.8412 | no, replaced by 32 |
| Group size 32 | 14.0384 | 13.6386 | **on** (+21 MB, +4.5%) |
| Act-order | 14.0384 (g128), 13.6386 (g32) | 13.9863, 13.5543 | **on** |
| Clip search | 13.6386 (g32), 13.5543 (g32 + act-order) | 13.5528, 13.4777 | **on** |
| Mixed precision, `down_proj` int8 | 13.4777 | 13.4056 | no, `sensitive` is better |
| Mixed precision, `sensitive` (`down_proj` and last layer int8) | 13.4777 | 13.3775 | **on** (+48 MB, +9.8%) |
| New calibration (wikitext-2 + C4, 128 x 2048) | 13.4777 | 13.5492 | **off** |

- Clip search alone made the g128 model worse (14.0384 to 14.0630). It helps once the group size is 32 and in the full combination, so it is on in the default combination only. With `--group-size 128` and no act-order, pass `--no-clip-search`.
- The new calibration is off because it raised perplexity by 0.07 and took 714 s to convert instead of 259 s. This is a WikiText-2 test, and the reference calibration is WikiText-2 train, so the comparison favors the reference calibration. A C4 or downstream evaluation could rank them differently. It stays available as `--multi-source-calibration`.
- 3% target: **reached** with the default settings (+2.35%). Without mixed precision the best result is +3.12%, so the 3% target needs the int8 `down_proj` and last layer.
- Each result is one conversion. The differences between g32 variants (for example 13.5528 and 13.5543) are well inside what a different calibration seed could change, so do not read an order into them.

## Size candidate D: int6 or int5 group-wise

Not evaluated: needs runtime changes.

Cost check (task 27.2). A new precision needs changes in more than `quf/format` and `quf/quant`:

- `quf/runtime/model.py` selects the weight loader by tensor suffix (`.qweight`, `INT8_SUFFIX`) and sizes its GPU cache by those suffixes, so a new suffix needs a new load branch there.
- `quf/runtime/linear.py` derives `precision` from `QuantizedWeight` or `Int8Weight` only, and `quf/runtime/backends/*` (`prepare`, including the Marlin int4-only check) would need a new weight type or dispatch.
- Format, writer validation, quantizer, GPTQ and converter would also need a new dtype code, packer, unpacker and version bump.

The runtime changes cannot be avoided, and `quf/runtime/**` was off limits for this task because task 26.2 is editing it. No code was changed and Packed_Int4 tensors are unaffected.

## Size study and default preset (task 27.3, Requirement 32)

Llama protocol (`--protocol llama`), full WikiText-2 test split and the held-out C4 slice, group size 32, act-order, clip search, GPTQ. v020 rows are reused from `docs/bench_accuracy.json` (`models_llama`); candidate rows come from `scripts/size_study.py` and are saved in `docs/size_study.json`.

| Model | Run | Description | File bytes | vs v020 size | wikitext2 ppl | vs v020 | c4 ppl | vs v020 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-0.5B | v020 | sensitive preset (down_proj and last layer int8) | 539605632 | +0.00% | 12.0326 | +0.00% | 18.8441 | +0.00% |
| Qwen/Qwen2.5-0.5B | size_a | A: last layer int4 g32 | 535153472 | -0.83% | 12.0546 | +0.18% | 18.9052 | +0.32% |
| Qwen/Qwen2.5-0.5B | size_b | B: down_proj int4 g32 with act-order | 497728832 | -7.76% | 12.0769 | +0.37% | 19.0473 | +1.08% |
| Qwen/Qwen2.5-0.5B | size_c | C: A and B together | 491449152 | -8.92% | 12.1133 | +0.67% | 19.1446 | +1.59% |
| Qwen/Qwen2.5-1.5B | v020 | sensitive preset (down_proj and last layer int8) | 1413323520 | +0.00% | 8.3954 | +0.00% | 13.6774 | +0.00% |
| Qwen/Qwen2.5-1.5B | size_a | A: last layer int4 g32 | 1399383936 | -0.99% | 8.4062 | +0.13% | 13.7026 | +0.18% |
| Qwen/Qwen2.5-1.5B | size_b | B: down_proj int4 g32 with act-order | 1257449536 | -11.03% | 8.4442 | +0.58% | 13.8663 | +1.38% |
| Qwen/Qwen2.5-1.5B | size_c | C: A and B together | 1237729024 | -12.42% | 8.4580 | +0.75% | 13.8980 | +1.61% |


Candidates: A = last layer int4; B = down_proj int4 with act-order, last layer stays int8; C = A and B (every decoder linear int4). D (int6/int5) was not evaluated, see below.

### Decision

- Requirement 32.4 (wikitext2 within 1% of v020 on both models, smaller file on both): A, B and C all qualify. C is the smallest (-8.9% on 0.5B, -12.4% on 1.5B; wikitext2 +0.67% and +0.75%).
- Extra check on C4 (should not degrade more than about 1.5%): C reaches +1.59% and +1.61%, so C fails it. B stays at +1.08% and +1.38%.
- **Default changed from `sensitive` (v020) to `size_b`**: the smallest candidate that also passes the C4 check. It is 7.8% smaller on 0.5B (497,728,832 bytes) and 11.0% smaller on 1.5B (1,257,449,536 bytes), at +0.37% and +0.58% wikitext2 perplexity. `--preset sensitive` restores v020. `--preset size_c` gives the smallest files if the C4 loss of about 1.6% is acceptable.
- Single conversion per row, so differences of a few hundredths of a percent are within calibration noise.

## Quantizer improvement study (task 35.2, Requirement 39.7 and 39.8)

Qwen2.5-0.5B, base `b_e8g64` (size_b body, group 32, act-order, clip search, int8 g64 embedding, 365.8 MB).
Quick screen = first 16 llama-protocol windows. Keep rule: WikiText-2 down by at least 0.1 percent and C4 up by at most 0.1 percent.
Raw rows: `docs/size_study.json` -> `v040` -> `Qwen/Qwen2.5-0.5B` (keys `q_*@w16`).

| Candidate | Wiki-2 (16 win) | vs base | C4 (16 win) | vs base | Kept |
|---|---|---|---|---|---|
| b_e8g64 (128 x 1024, damp 0.01) | 11.8994 | - | 20.9241 | - | base |
| calibration 256 x 2048 | 11.8937 | -0.05% | 20.8740 | -0.24% | no (Wiki gain under 0.1%) |
| damp 0.005 | 11.9001 | +0.01% | 20.8764 | -0.23% | no |
| damp 0.05 | 11.8976 | -0.02% | 20.8479 | -0.37% | no |
| 256 x 2048 + damp 0.05 | 11.9453 | +0.38% | 20.8526 | -0.35% | no (worse) |
| true-sequential within attention | not run | | | | already implemented |
| per-layer group size by sensitivity | not evaluated | | | | format limitation |

Notes:
- 256 x 1024 and 128 x 2048 were not needed: 256 x 2048 converted in 325 s.
- `quantize_model_sequential` already quantizes q/k/v, re-collects activations, then quantizes o_proj, then gate/up, then down_proj (`_SEQUENTIAL_GROUPS` in `quf/quant/gptq.py`).
- Per-module group size is not supported. The header has one `quantization.group_size` for all int4 modules (`check_group_size` in `quf/format/validate.py`), and the backends read that single value. Only the embedding has its own group size. The size_b body is already group 32, so "32 for sensitive, 64 for the rest" would also shrink the file less than it costs in a format change. No format change was made.
- Run-to-run differences on 16 windows are about 0.4% on Wiki-2, larger than the 0.1% keep threshold, so these screens cannot resolve small gains. The C4 drops (0.2 to 0.4%) appear for every changed setting and are not specific to one of them.
- No improvement met the keep rule, so no converter or CLI default changed. The only code change is an opt-in `damp` argument on `convert_checkpoint` and `run_native_gptq` (default 0.01, unchanged).

Full llama protocol (all windows), Qwen2.5-0.5B:

| Config | Size | Wiki-2 | C4 |
|---|---|---|---|
| QUF b_e8g64 (final, no improvement kept) | 365.8 MB | 12.0783 | 19.0493 |
| QUF 0.3.0 size_b (earlier run) | 497.7 MB | 12.077 | |
| GGUF Q4_K_M | 398 MB | 12.043 | |

Result: 0.5B is smaller than Q4_K_M (365.8 MB, under the 397.8 MB limit) but WikiText-2 12.078 is 0.29% above Q4_K_M 12.043. The accuracy target is still missed.

## Size study and default choice (task 34.4, Requirement 39.1 to 39.4)
Full llama protocol (all windows) unless marked 16w. Body is int4 g32 with act-order and clip search, embedding int8 g64 (clip) unless the name says otherwise. Raw rows: `docs/size_study.json` -> `v040`; decision: `v040_decision`.
Targets: 0.5B size at most 397.8 MB, WikiText-2 at most 12.043, C4 at most 19.238 (0.3.0 size_b 19.047 + 1%). 1.5B size at most 986.0 MB, WikiText-2 at most 8.600, C4 at most 14.005.

Step 1, per-module probe (Qwen2.5-0.5B, windows 16 to 31, `scripts/mixed_precision_study.py`, `docs/mixed_precision_0.5B.json`, `docs/mixed_plans_0.5B.json`). Making one module int8 and dividing the perplexity gain by the extra bytes ranks k_proj and v_proj far above everything else: they are small under GQA, so int8 costs about 45 KB per module. Whole plans against the size_b body on those windows:

| Plan (int8 on top of size_b) | Extra MB | WikiText-2 | C4 |
|---|---:|---:|---:|
| k_proj, v_proj | 2.1 | -1.08% | -1.54% |
| k, v, o_proj | 9.8 | -1.33% | -1.73% |
| all attention | 17.6 | -1.37% | -1.78% |
| all attention + down_proj layers 0 to 5 | 28.5 | -1.52% | -2.03% |
| all attention + down_proj layers 17 to 22 | 28.5 | -1.55% | -2.06% |
| all attention + all down_proj | 59.5 | -1.93% | -2.46% |

So almost all of the attention gain comes from k_proj and v_proj for 12% of the bytes. The rule is by module type, not a list for one model.

Step 1 and 2, full protocol:

| Model | Config | Size MB | vs Q4_K_M size | WikiText-2 | C4 |
|---|---|---:|---:|---:|---:|
| 0.5B | b_e8g64 (size_b, 0.4.0 base) | 365.8 | -8.0% | 12.0783 | 19.0493 |
| 0.5B | b_bg64_e8g64 (body g64) | 352.5 | -11.4% | 12.2273 | 19.3079 |
| 0.5B | **f_e8g64 (size_f, chosen)** | **367.9** | **-7.5%** | **11.9644** | **18.7459** |
| 0.5B | g_e8g64 (size_g, k v o) | 375.7 | -5.6% | 11.9550 | 18.7413 |
| 0.5B | d_e8g64 (size_d, all attention) | 383.4 | -3.6% | 11.9554 | 18.7146 |
| 0.5B | e_e8g64 (size_e, attention + tail down_proj) | 392.5 | -1.3% | 11.9295 | 18.6644 |
| 0.5B | Q4_K_M | 397.8 | 0 | 12.043 | |
| 1.5B | b_e4g32 (int4 embedding) | 922.0 | -6.5% | 8.9255 | 14.7787 |
| 1.5B | b_bg64_e8g64 (body g64) | 982.0 | -0.4% | 8.4957 | 13.9736 |
| 1.5B | d_e4g32 | 984.2 | -0.2% | 8.8805 | 14.6386 |
| 1.5B | b_e8g64 | 1031.4 | +4.6% | 8.4470 | 13.8700 |
| 1.5B | **f_e8g64 (size_f, chosen)** | **1040.0** | **+5.5%** | **8.4141** | **13.8094** |
| 1.5B | d_e8g64 | 1093.6 | +10.9% | 8.4066 | 13.7488 |
| 1.5B | Q4_K_M | 986.0 | 0 | 8.600 | 14.005 (limit) |

### Decision
- Default is now `size_f` with the int8 g64 embedding, body g32: k_proj and v_proj int8 in every layer, the whole last layer int8, everything else int4. `quf convert` default `--preset` changed from `size_d` to `size_f` (embed int8 g64 unchanged).
- Qwen2.5-0.5B: **all targets met**. 367.9 MB (29.9 MB under Q4_K_M), WikiText-2 11.9644 (0.65% under Q4_K_M 12.043), C4 18.7459 (1.6% better than 0.3.0 size_b).
- Qwen2.5-1.5B: WikiText-2 8.4141 (target 8.600) and C4 13.8094 (target 14.005) are met. **Size is missed**: 1040.0 MB against 986.0 MB, +54.0 MB (+5.5%).
- No single configuration meets every target on both models. The 1.5B size target needs body group size 64 (b_bg64_e8g64, 982.0 MB, all 1.5B targets met), which costs 0.5B WikiText-2 12.2273 and C4 +1.4%. The format has one group size for all int4 modules, so it cannot be applied to the 1.5B only. Every g32 body config that meets the 0.5B targets is at least 1031 MB on the 1.5B.
- Why size_f over d, e, g: g and d are 8 MB and 16 MB larger on 0.5B for at most 0.01 WikiText-2 gain, and on 1.5B d adds 53 MB. e reaches 11.9295 at 392.5 MB, but only buys 0.03 WikiText-2 over f for 24.6 MB, and would push the 1.5B further from 986 MB.
- Last-layer int8 rule (39.3): size_c (last layer int4) saves 6.3 MB on 0.5B and 19.7 MB on 1.5B but raised C4 by 1.6% in the earlier study, so the rule stays.
- The 0.5B and 1.5B differences between neighbouring rows of a few hundredths are within the single-conversion noise noted above.
