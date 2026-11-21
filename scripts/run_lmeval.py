"""Run hellaswag and arc_easy (limit 500, 0-shot, fixed seed) for an HF fp16 model or a QUF file.

    python scripts/run_lmeval.py hf  Qwen/Qwen2.5-0.5B  out.json
    python scripts/run_lmeval.py quf path/to/model.quf  out.json  [tokenizer_id]
"""
import json
import sys
import time

import lm_eval

kind, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
if kind == "hf":
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=src, dtype="float16", device="cuda", batch_size=16)
else:
    from quf.bench.lm_eval_adapter import get_adapter_class

    tok = sys.argv[4] if len(sys.argv) > 4 else None
    lm = get_adapter_class()(path=src, device="cuda", tokenizer_path=tok)
t = time.time()
r = lm_eval.simple_evaluate(
    model=lm, tasks=["hellaswag", "arc_easy"], num_fewshot=0, limit=500,
    random_seed=1234, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234,
    log_samples=False,
)
res = dict(r["results"])
res["_seconds"] = time.time() - t
json.dump(res, open(out, "w"), indent=1, default=str)
print("DONE", kind, json.dumps(res, default=str))
