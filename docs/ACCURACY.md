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
