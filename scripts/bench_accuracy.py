"""Evaluation matrix (task 17.1, Requirements 24.3, 24.4, 24.6).

For each model the matrix measures, with the same evaluation code and text:

* ``fp16``      reference (Hugging Face model in fp16),
* ``v010``      the 0.1.0 settings (g128, no act-order, no clip, int4 everywhere,
                wikitext-2 calibration 128 x 512),
* ``rtn_g128`` / ``rtn_g32``  RTN_Baseline: round-to-nearest from quf's own group-wise
                quantizer (no calibration, no Hessian),
* ``v020``      the 0.2.0 defaults (g32 + clip + act-order + ``sensitive`` preset,
                wikitext-2 calibration 128 x 1024),
* ``gguf_q4_k_m`` / ``gguf_q4_0``  GGUF_Baseline through llama-cpp-python, only when a
                prebuilt wheel installs; otherwise recorded as "not run" with the pip error.

Every row reports perplexity on the WikiText-2 test split (the calibration domain) and on a
held-out slice of the C4 English *validation* split (a second domain that is never used
for calibration), plus file size and tokens per second, the relative increase against
fp16, and whether the 3 percent target was reached. JSON also records device names and
software versions.

Driver (each run is a subprocess with a 25 minute cap, results are merged into
docs/bench_accuracy.json after every run, logs are in eval_cache/):

    python scripts/bench_accuracy.py --drive --models Qwen/Qwen2.5-0.5B
    python scripts/bench_accuracy.py --drive --models Qwen/Qwen2.5-1.5B --only fp16 v010 v020 rtn_g128

Single run (what the driver launches):

    python scripts/bench_accuracy.py --run v020 --model Qwen/Qwen2.5-0.5B
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "eval_cache"
OUT = ROOT / "docs" / "bench_accuracy.json"
TIME_CAP_S = 25 * 60
PIP_CAP_S = 5 * 60
TARGET_PCT = 3.0
C4_TEXT_CHARS = 1_300_000  # about as many tokens as the WikiText-2 test split
C4_DATASET = "allenai/c4"

QUF_RUNS = {
    # name: (group_size, act_order, clip_search, preset, calibration, method)
    "v010": (128, False, False, None, "legacy", "gptq"),
    "rtn_g128": (128, False, False, None, "none", "rtn"),
    "rtn_g32": (32, False, False, None, "none", "rtn"),
    "v020": (32, True, True, "sensitive", "default", "gptq"),
}
GGUF_RUNS = {"gguf_q4_k_m": ("Q4_K_M", 15), "gguf_q4_0": ("Q4_0", 2), "gguf_q8_0": ("Q8_0", 7)}
ALL_RUNS = ["fp16", "v010", "rtn_g128", "rtn_g32", "v020", "gguf_q4_k_m", "gguf_q4_0", "gguf_q8_0"]
CONVERT_SCRIPT_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/b6100/convert_hf_to_gguf.py"


def slug(model: str) -> str:
    return model.split("/")[-1].lower().replace(".", "_")


# ----------------------------------------------------------------------------- shared helpers
def c4_heldout_text() -> str:
    """A fixed slice of the C4 English validation split, cached on disk."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / "c4_validation_slice.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    from datasets import load_dataset

    ds = load_dataset(C4_DATASET, "en", split="validation", streaming=True)
    docs, n = [], 0
    for row in ds:
        t = row.get("text")
        if t and t.strip():
            docs.append(t)
            n += len(t)
            if n >= C4_TEXT_CHARS:
                break
    text = "\n\n".join(docs)
    p.write_text(text, encoding="utf-8")
    return text


def _env() -> dict:
    from quf.bench.suite import software_versions

    v = software_versions()
    try:
        from importlib import metadata

        v["llama-cpp-python"] = metadata.version("llama-cpp-python")
    except Exception:  # noqa: BLE001
        v["llama-cpp-python"] = "not installed"
    dev = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return {"device": dev, "platform": platform.platform(), "software_versions": v}


def _hf_tokenizer(model: str):
    from quf.convert.converter import _hf_from_info, build_tokenizer_info, read_source_config
    from quf.tokenization.wrapper import TokenizerWrapper

    cfg = read_source_config(model)
    return TokenizerWrapper(None, hf_tokenizer=_hf_from_info(build_tokenizer_info(model, cfg), model))


PROTOCOLS = ("full", "llama")
CTX = 2048
LLAMA_EXE = CACHE / "llama_cpp" / "llama-perplexity.exe"


def score_start(n_tokens: int, protocol: str) -> int:
    """First scored target index (into ``ids[1:]``) of a window.

    ``full`` scores every position 1..n-1. ``llama`` matches llama-perplexity, which scores
    only the second half of each window: logits at positions n//2 .. n-2.
    """
    if protocol == "full":
        return 0
    if protocol == "llama":
        return n_tokens // 2
    raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def window_nll(logits, ids, protocol: str = "full") -> tuple[float, int]:
    """Summed negative log-likelihood and token count of one window under ``protocol``.

    ``logits`` is ``[n, vocab]`` (numpy or torch), ``ids`` the n window token ids; row j
    predicts ``ids[j + 1]``.
    """
    import numpy as np

    n = len(ids)
    first = score_start(n, protocol)
    nll, cnt = 0.0, 0
    for s in range(first, n - 1, 256):
        e = min(s + 256, n - 1)
        blk = logits[s:e]
        if hasattr(blk, "detach"):
            blk = blk.detach().float().cpu().numpy()
        blk = np.asarray(blk, dtype=np.float32)
        mx = blk.max(1, keepdims=True)
        lse = np.log(np.exp(blk - mx).sum(1)) + mx[:, 0]
        tgt = np.asarray(ids[s + 1 : e + 1])
        nll += float((lse - blk[np.arange(len(blk)), tgt]).sum())
        cnt += len(blk)
    return nll, cnt


def window_ids(ids: list[int], windows: int = 0, ctx: int = CTX) -> list[list[int]]:
    """Non-overlapping ``ctx``-token windows; ``windows`` > 0 keeps the first N (0 means all)."""
    n = len(ids) // ctx
    if windows:
        n = min(n, windows)
    return [ids[i * ctx:(i + 1) * ctx] for i in range(n)]


def ppl_protocol(model, tok, text: str, protocol: str, windows: int = 0) -> float:
    """Perplexity of a ``forward(ids) -> logits`` model under ``protocol`` on a fixed window set."""
    import math

    import torch

    nll, cnt = 0.0, 0
    for w in window_ids(tok.encode(text), windows):
        logits = model.forward(torch.tensor([w], dtype=torch.long))[0]
        a, b = window_nll(logits, w, protocol)
        nll, cnt = nll + a, cnt + b
    return math.exp(nll / cnt)


def _domain_ppls(model, tok, protocol: str = "full", windows: int = 0) -> dict:
    """WikiText-2 test and held-out C4 perplexity, same protocol and window set for every row."""
    from quf.bench.suite import load_wikitext

    return {"ppl_wikitext2": ppl_protocol(model, tok, load_wikitext(), protocol, windows),
            "ppl_c4_heldout": ppl_protocol(model, tok, c4_heldout_text(), protocol, windows),
            "protocol": protocol, "windows": windows or "all"}

# ----------------------------------------------------------------------------- workers
def rtn_quantizer(source, calibration, *, group_size, symmetric=False, **_ignored):
    """RTN_Baseline: quf's own ``quantize_groupwise`` on every decoder linear. No Hessian."""
    import torch
    from transformers import AutoModelForCausalLM

    from quf.convert import gptq as g
    from quf.quant.quantize import quantize_groupwise

    try:
        model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float16)
    linears = {}
    for name, m in model.named_modules():
        if name.startswith("model.layers.") and isinstance(m, torch.nn.Linear):
            linears[name] = quantize_groupwise(m.weight, group_size=group_size, symmetric=symmetric)
    res = g.GPTQResult(linears=linears, symmetric=symmetric, group_size=group_size, bits=g.BITS)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    skip = {f"{n}.weight" for n in linears}
    for name, t in model.state_dict().items():
        if name in skip or (tied and name == "lm_head.weight"):
            continue
        res.others[name] = g._other_array(t)
    return res


def run_fp16(a) -> int:
    import torch

    from quf.bench.suite import _HFCausalLM

    tok = _hf_tokenizer(a.model)
    m = _HFCausalLM(a.model, "cuda")
    ppls = _domain_ppls(m, tok, a.protocol, a.windows)
    ids = torch.tensor([tok.encode("The quick brown fox") or [0]], device="cuda")
    t = time.perf_counter()
    with torch.no_grad():
        out = m._m.generate(ids, max_new_tokens=32, min_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()
    tps = (out.shape[1] - ids.shape[1]) / (time.perf_counter() - t)
    size = sum(p.numel() * 2 for p in m._m.parameters())
    rec = {"run": a.run, "status": "done", **ppls, "file_size_bytes": size,
           "tokens_per_second": tps, "backend": "huggingface fp16 generate (cuda)",
           "size_note": "fp16 parameter bytes (tied embeddings counted once)"}
    _write_result(a, rec)
    return 0


def run_quf(a) -> int:
    import gc

    import quf.convert.converter as conv
    from quf.bench.suite import _tokenizer_from_header, run_bench
    from quf.convert.precision import PrecisionPlan

    gs, act, clip, preset, calib_kind, method = QUF_RUNS[a.run]
    CACHE.mkdir(exist_ok=True)
    quf_path = CACHE / f"{slug(a.model)}_{a.run}.quf"
    kw = dict(group_size=gs, act_order=act, clip_search=clip, overwrite=True)
    if preset:
        from quf.convert.converter import read_source_config

        kw["precision_plan"] = PrecisionPlan.from_options(
            presets=[preset], num_layers=read_source_config(a.model).get("num_hidden_layers"))
    if method == "rtn":
        conv.build_calibration = lambda *x, **y: [[0]]  # unused by RTN
        kw.update(texts=["x"], quantizer=rtn_quantizer, n_samples=1, seq_len=1)
    elif calib_kind == "legacy":
        kw["legacy_calibration"] = True
    t1 = time.perf_counter()
    res = conv.convert_checkpoint(a.model, quf_path, **kw)
    convert_s = time.perf_counter() - t1
    print(f"converted in {convert_s:.0f}s", flush=True)
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    box = {}

    def loader(p, backend):
        from quf.runtime.model import QUFModel

        box["m"] = QUFModel.load(p, device_option="cpu" if backend == "cpu" else "cuda", backend=backend)
        return box["m"]

    t2 = time.perf_counter()
    r = run_bench(quf_path, out_json=CACHE / f"{slug(a.model)}_{a.run}.bench.json",
                  device="cuda", model_loader=loader)
    model = box["m"]
    ppls = _domain_ppls(model, _tokenizer_from_header(model.header), a.protocol, a.windows)
    rec = {"run": a.run, "status": "done", "group_size": gs, "act_order": act, "clip_search": clip,
           "preset": preset, "calibration": calib_kind, "method": method, **ppls,
           "file_size_bytes": r.file_size_bytes, "tokens_per_second": r.tokens_per_second,
           "backend": r.backend, "convert_seconds": round(convert_s),
           "bench_seconds": round(time.perf_counter() - t2),
           "size_report": res.report.format() if res.report else None}
    _write_result(a, rec)
    quf_path.unlink(missing_ok=True)
    return 0


def _gguf_ppl(llm, windows_ids, n_ctx: int = CTX, protocol: str = "full") -> float:
    """llama-cpp-python scorer; same ``window_nll`` as the HF and QUF rows."""
    import math

    import numpy as np

    nll, cnt = 0.0, 0
    for ids in windows_ids:
        llm.reset()
        llm.eval(ids)
        a, b = window_nll(np.asarray(llm.scores[: len(ids)], dtype=np.float32), ids, protocol)
        nll, cnt = nll + a, cnt + b
    return math.exp(nll / cnt)


def _exe_ppl(gguf: Path, text: str, windows: int, tag: str, timeout: int = 900) -> float:
    """Half-window (``llama`` protocol) perplexity from llama-perplexity.exe on the GPU."""
    import re

    txt = CACHE / f"ppl_text_{tag}.txt"
    txt.write_text(text, encoding="utf-8")
    cmd = [str(LLAMA_EXE), "-m", str(gguf), "-f", str(txt), "-c", str(CTX), "-ngl", "99"]
    if windows:
        cmd += ["--chunks", str(windows)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(LLAMA_EXE.parent))
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", p.stdout + p.stderr)
    if not m:
        raise RuntimeError(f"llama-perplexity gave no PPL (rc={p.returncode}): {(p.stderr or p.stdout)[-500:]}")
    return float(m.group(1))

_HPARAMS_LINE = "config = AutoConfig.from_pretrained(dir_model, trust_remote_code=False).to_dict()"
_HPARAMS_PATCH = ("; config.update({k: v for k, v in json.load(open(dir_model / 'config.json', encoding='utf-8'))"
                  ".items() if k not in config})  # quf: transformers>=5 moves rope_theta into rope_parameters")


def _patch_convert_script(script: Path) -> None:
    """Merge raw config.json keys into the converter's hparams.

    transformers 5 moves ``rope_theta`` into ``rope_parameters`` in
    ``AutoConfig.to_dict()``, so the llama.cpp converter writes no
    ``rope.freq_base`` and llama.cpp falls back to 10000 (Qwen2.5 uses 1e6).
    """
    src = script.read_text(encoding="utf-8")
    if "# quf:" in src:
        return
    if _HPARAMS_LINE not in src:
        raise RuntimeError("convert_hf_to_gguf.py layout changed; cannot apply the rope_theta patch")
    script.write_text(src.replace(_HPARAMS_LINE, _HPARAMS_LINE + _HPARAMS_PATCH, 1), encoding="utf-8")


def _check_rope_base(gguf_path: Path, local: str) -> None:
    from gguf import GGUFReader

    theta = json.loads((Path(local) / "config.json").read_text(encoding="utf-8")).get("rope_theta")
    if theta is None:
        return
    keys = [k for k in GGUFReader(str(gguf_path)).fields if k.endswith(".rope.freq_base")]
    if not keys:
        gguf_path.unlink(missing_ok=True)
        raise RuntimeError(f"{gguf_path.name} has no rope.freq_base (expected {theta})")


def run_gguf(a) -> int:
    import ctypes

    import llama_cpp
    from huggingface_hub import snapshot_download

    from quf.bench.suite import load_wikitext

    label, ftype = GGUF_RUNS[a.run]
    CACHE.mkdir(exist_ok=True)
    f16 = CACHE / f"{slug(a.model)}-f16.gguf"
    if not f16.exists():
        script = CACHE / "convert_hf_to_gguf.py"
        if not script.exists():
            import urllib.request

            urllib.request.urlretrieve(CONVERT_SCRIPT_URL, script)
        _patch_convert_script(script)
        local = snapshot_download(a.model)
        subprocess.run([sys.executable, str(script), local, "--outtype", "f16", "--outfile", str(f16)],
                       check=True)
        _check_rope_base(f16, local)
    out = CACHE / f"{slug(a.model)}-{label}.gguf"
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = ftype
    rc = llama_cpp.llama_model_quantize(str(f16).encode(), str(out).encode(), ctypes.byref(params))
    if rc != 0:
        raise RuntimeError(f"llama_model_quantize returned {rc}")
    wiki, c4 = load_wikitext(), c4_heldout_text()
    use_exe = a.protocol == "llama" and a.gguf_backend in ("auto", "exe") and LLAMA_EXE.exists()
    if a.gguf_backend == "exe" and not use_exe:
        raise RuntimeError(f"--gguf-backend exe needs --protocol llama and {LLAMA_EXE}")
    if use_exe:
        # llama-perplexity only implements half-window scoring; it runs on the GPU (-ngl 99).
        ppls = {"ppl_wikitext2": _exe_ppl(out, wiki, a.windows, "wikitext2"),
                "ppl_c4_heldout": _exe_ppl(out, c4, a.windows, "c4")}
        backend = "llama-perplexity.exe (cuda, -ngl 99)"
    else:
        tok = _hf_tokenizer(a.model)
        gpu = bool(llama_cpp.llama_supports_gpu_offload())
        llm = llama_cpp.Llama(model_path=str(out), n_ctx=CTX, n_batch=CTX, logits_all=True,
                              n_gpu_layers=-1 if gpu else 0, verbose=False)
        ppls = {"ppl_wikitext2": _gguf_ppl(llm, window_ids(tok.encode(wiki), a.windows), CTX, a.protocol),
                "ppl_c4_heldout": _gguf_ppl(llm, window_ids(tok.encode(c4), a.windows), CTX, a.protocol)}
        backend = "llama-cpp-python (cuda wheel)" if gpu else "llama-cpp-python (cpu wheel)"
        del llm
    llm = llama_cpp.Llama(model_path=str(out), n_gpu_layers=-1 if llama_cpp.llama_supports_gpu_offload() else 0,
                          verbose=False)
    rec = {"run": a.run, "status": "done", "quant": label, **ppls, "protocol": a.protocol,
           "windows": a.windows or "all", "file_size_bytes": out.stat().st_size, "backend": backend}
    t = time.perf_counter()
    o = llm("The quick brown fox", max_tokens=32, temperature=0.0)
    n = int(o["usage"]["completion_tokens"])
    rec["tokens_per_second"] = n / (time.perf_counter() - t) if n else 0.0
    _write_result(a, rec)
    return 0


def _write_result(a, rec: dict) -> None:
    (CACHE / f"{slug(a.model)}_{a.run}.result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("RESULT", json.dumps(rec), flush=True)


# ----------------------------------------------------------------------------- driver
def _load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"environment": {}, "target_percent": TARGET_PCT, "time_cap_seconds": TIME_CAP_S,
            "eval": {"domain_1": "WikiText-2 test split (calibration domain)",
                     "domain_2": f"{C4_DATASET} en validation, first ~{C4_TEXT_CHARS} chars (never used for calibration)",
                     "context": 2048}, "gguf_setup": None, "models": {}}


def _finalize(d: dict) -> None:
    """Add relative increases against fp16 and the 3 percent verdict to every finished row."""
    for m in d["models"].values():
        ref = m.get("fp16")
        for name, r in m.items():
            if name == "fp16" or r.get("status") != "done" or not ref or ref.get("status") != "done":
                continue
            if (r.get("protocol"), r.get("windows")) != (ref.get("protocol"), ref.get("windows")):
                r["comparison"] = "not comparable: protocol or window set differs from the fp16 row"
                continue
            for key, label in (("ppl_wikitext2", "wikitext2"), ("ppl_c4_heldout", "c4_heldout")):
                if key in r and key in ref:
                    inc = 100 * (r[key] / ref[key] - 1)
                    r[f"increase_pct_{label}"] = inc
                    r[f"target_reached_{label}"] = inc <= TARGET_PCT


def _save(d: dict) -> None:
    _finalize(d)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _pip(args: list[str]) -> tuple[bool, str]:
    try:
        p = subprocess.run([sys.executable, "-m", "pip", "install", *args], capture_output=True, text=True,
                           timeout=PIP_CAP_S)
        return p.returncode == 0, (p.stdout + p.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return False, f"pip timed out after {PIP_CAP_S}s"


def setup_gguf(d: dict) -> bool:
    """Install llama-cpp-python from a prebuilt wheel only (never a source build)."""
    g = d.get("gguf_setup")
    if g:
        return g["status"] == "installed"
    tries = [["--only-binary=:all:", "llama-cpp-python"],
             ["--only-binary=:all:", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu",
              "llama-cpp-python"]]
    errs = []
    for t in tries:
        ok, log = _pip(t)
        if ok:
            ok2, log2 = _pip(["gguf"])
            d["gguf_setup"] = {"status": "installed" if ok2 else "not run", "command": "pip install " + " ".join(t),
                               "pip_error": None if ok2 else log2}
            _save(d)
            return ok2
        errs.append({"command": "pip install " + " ".join(t), "error": log})
    d["gguf_setup"] = {"status": "not run", "reason": "no prebuilt llama-cpp-python wheel installed", "attempts": errs}
    _save(d)
    return False


def launch(model: str, name: str, d: dict, protocol: str = "full", windows: int = 0,
           gguf_backend: str = "auto"):
    rows = d["models"].setdefault(model, {})
    old = rows.get(name, {})
    if old.get("status") in ("done", "timeout") and (old.get("protocol"), old.get("windows")) == (
            protocol, windows or "all"):
        return
    cmd = [sys.executable, "-u", str(Path(__file__)), "--run", name, "--model", model,
           "--protocol", protocol, "--windows", str(windows), "--gguf-backend", gguf_backend]
    CACHE.mkdir(exist_ok=True)
    log = CACHE / f"{slug(model)}_{name}.log"
    rf = CACHE / f"{slug(model)}_{name}.result.json"
    rf.unlink(missing_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] start {model} {name}", flush=True)
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
        t = time.time()
        while p.poll() is None and time.time() - t < TIME_CAP_S:
            time.sleep(5)
        if p.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            p.wait()
            rec = {"run": name, "status": "timeout", "note": f"not completed: killed after {TIME_CAP_S // 60} min"}
        elif p.returncode == 0 and rf.exists():
            rec = json.loads(rf.read_text(encoding="utf-8"))
        else:
            tail = log.read_text(encoding="utf-8", errors="replace")[-800:] if log.exists() else ""
            rec = {"run": name, "status": "error", "returncode": p.returncode, "log_tail": tail}
    rows[name] = rec
    _save(d)
    print(f"[{time.strftime('%H:%M:%S')}] {model} {name}: {rec['status']} wiki={rec.get('ppl_wikitext2')} "
          f"c4={rec.get('ppl_c4_heldout')}", flush=True)


def drive(models: list[str], only: list[str] | None, protocol: str = "full", windows: int = 0,
          gguf_backend: str = "auto", with_gguf: bool = False) -> int:
    d = _load()
    d["environment"] = _env()
    _save(d)
    for model in models:
        for name in ALL_RUNS:
            if only and name not in only:
                continue
            if name in GGUF_RUNS and with_gguf:
                launch(model, name, d, protocol, windows, gguf_backend)
                continue
            if name in GGUF_RUNS:
                # GGUF rows are unverified and skipped in this matrix (scope cut); no pip install attempted.
                rows = d["models"].setdefault(model, {})
                if rows.get(name, {}).get("status") != "done":
                    rows[name] = {"run": name, "status": "not run (unverified)",
                                  "reason": "GGUF rows skipped by scope decision; harness not verified end to end"}
                    _save(d)
                continue
            launch(model, name, d, protocol, windows, gguf_backend)
    print("driver finished", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--only", nargs="+", choices=ALL_RUNS)
    ap.add_argument("--run", choices=ALL_RUNS)
    ap.add_argument("--protocol", choices=PROTOCOLS, default="full",
                    help="full: score every position 1..2047 of each window; llama: second half only, "
                         "as llama-perplexity does. Applied to every row.")
    ap.add_argument("--windows", type=int, default=0,
                    help="score only the first N 2048-token windows in every row (0 = whole text)")
    ap.add_argument("--gguf-backend", choices=["auto", "exe", "python"], default="auto",
                    help="auto: llama-perplexity.exe (GPU) for --protocol llama, else llama-cpp-python")
    ap.add_argument("--with-gguf", action="store_true", help="include the GGUF rows in --drive")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    a = ap.parse_args()
    if a.drive:
        return drive(a.models, a.only, a.protocol, a.windows, a.gguf_backend, a.with_gguf)
    if a.run == "fp16":
        return run_fp16(a)
    if a.run in QUF_RUNS:
        return run_quf(a)
    if a.run in GGUF_RUNS:
        return run_gguf(a)
    ap.error("use --drive or --run NAME")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
