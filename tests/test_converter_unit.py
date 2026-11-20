"""Unit tests for the Converter (Requirements 7.5, 7.6, 7.7, 7.8) with a stubbed quantizer."""
import json

import pytest
import torch

pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers

from quf.convert import converter as C
from quf.convert.gptq import GPTQResult
from quf.format.reader import QUFReader
from quf.quant.quantize import quantize_groupwise

TEXTS = ["a b c d e f g h i j k l"] * 10


def _model_dir(tmp_path, arch="Qwen2ForCausalLM", extra=None):
    d = tmp_path / "model"
    d.mkdir()
    vocab = {chr(97 + i): i for i in range(26)}
    vocab["[UNK]"] = 26
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    cfg = {
        "architectures": [arch], "hidden_size": 128, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 256,
        "vocab_size": 27, "max_position_embeddings": 64, "eos_token_id": 3,
    }
    cfg.update(extra or {})
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _runner(record=None):
    def run(source, calib, *, group_size, symmetric):
        if record is not None:
            record.update(group_size=group_size, symmetric=symmetric, n=len(calib))
        torch.manual_seed(0)
        qw = quantize_groupwise(torch.randn(16, 128), group_size=128, symmetric=symmetric)
        return GPTQResult(
            linears={"model.layers.0.mlp.up_proj": qw},
            others={
                "model.embed_tokens.weight": torch.randn(27, 128).numpy(),
                "lm_head.weight": torch.randn(27, 128).numpy(),
            },
            symmetric=symmetric,
        )
    return run


def _convert(tmp_path, out, **kw):
    d = kw.pop("model_dir", None) or _model_dir(tmp_path)
    kw.setdefault("n_samples", 2)
    kw.setdefault("seq_len", 8)
    kw.setdefault("texts", TEXTS)
    kw.setdefault("quantizer", _runner())
    kw.setdefault("echo", None)
    return C.convert_checkpoint(str(d), out, **kw)


def test_size_and_ratio_reported(tmp_path):
    out = tmp_path / "o.quf"
    msgs = []
    res = _convert(tmp_path, out, echo=msgs.append)
    assert res.size_bytes == out.stat().st_size
    expected_fp16 = 2 * (16 * 128 + 27 * 128 * 2)
    assert res.fp16_size_bytes == expected_fp16
    assert res.ratio == pytest.approx(res.size_bytes / expected_fp16)
    assert msgs == [res.message]
    assert str(res.size_bytes) in res.message and f"{res.ratio:.4f}" in res.message


def test_runner_receives_4bit_group128_and_options(tmp_path):
    rec = {}
    _convert(tmp_path, tmp_path / "o.quf", quantizer=_runner(rec), symmetric=True)
    assert rec == {"group_size": 128, "symmetric": True, "n": 2}
    r = QUFReader(tmp_path / "o.quf")
    q = r.header.quantization
    assert q["bits"] == 4 and q["group_size"] == 128 and q["symmetric"] is True
    r.close()


def test_header_stores_config_and_tokenizer(tmp_path):
    out = tmp_path / "o.quf"
    _convert(tmp_path, out)
    r = QUFReader(out)
    mc = r.header.model_config
    assert mc["hidden_size"] == 128 and mc["num_key_value_heads"] == 1
    assert mc["architecture"] == "Qwen2ForCausalLM"
    tk = r.header.tokenizer
    assert tk["eos_token_id"] == 3
    assert "tokenizer.json" in tk["files"]
    r.close()


def test_embedding_and_lm_head_stored_fp16(tmp_path):
    out = tmp_path / "o.quf"
    _convert(tmp_path, out)
    r = QUFReader(out)
    assert r.tensor("model.embed_tokens.weight").dtype.name == "float16"
    assert r.tensor("lm_head.weight").dtype.name == "float16"
    r.close()


def test_qwen3_accepted_and_non_qwen_rejected():
    assert C.check_architecture({"architectures": ["Qwen3ForCausalLM"]}) == "Qwen3ForCausalLM"
    for cfg in ({"architectures": ["LlamaForCausalLM"]}, {}):
        with pytest.raises(C.UnsupportedArchitectureError) as ei:
            C.check_architecture(cfg)
        assert ei.value.architecture in str(ei.value)


def test_non_qwen_creates_no_output(tmp_path):
    d = _model_dir(tmp_path, "MistralForCausalLM")
    out = tmp_path / "x.quf"
    with pytest.raises(C.UnsupportedArchitectureError, match="MistralForCausalLM"):
        _convert(tmp_path, out, model_dir=d)
    assert not out.exists() and not (tmp_path / "x.quf.partial").exists()


def test_existing_output_untouched_without_overwrite(tmp_path):
    out = tmp_path / "e.quf"
    out.write_bytes(b"keep")
    with pytest.raises(C.OutputExistsError):
        _convert(tmp_path, out)
    assert out.read_bytes() == b"keep"


def test_overwrite_replaces_existing(tmp_path):
    out = tmp_path / "e.quf"
    out.write_bytes(b"old")
    _convert(tmp_path, out, overwrite=True)
    QUFReader(out).close()


@pytest.mark.parametrize("overwrite", [False, True])
def test_write_failure_cleans_partial(tmp_path, monkeypatch, overwrite):
    out = tmp_path / "f.quf"
    prior = b"prior"
    if overwrite:
        out.write_bytes(prior)

    def boom(path, header, tensors):
        open(path, "wb").write(b"x")
        raise RuntimeError("injected")

    monkeypatch.setattr(C, "write_quf", boom)
    with pytest.raises(RuntimeError, match="injected"):
        _convert(tmp_path, out, overwrite=overwrite)
    assert not (tmp_path / "f.quf.partial").exists()
    if overwrite:
        assert out.read_bytes() == prior
    else:
        assert not out.exists()


def test_runner_failure_leaves_no_output(tmp_path):
    def bad(*a, **k):
        raise RuntimeError("quantizer died")

    out = tmp_path / "g.quf"
    with pytest.raises(RuntimeError, match="quantizer died"):
        _convert(tmp_path, out, quantizer=bad)
    assert not out.exists() and not (tmp_path / "g.quf.partial").exists()


def test_eos_falls_back_to_generation_config(tmp_path):
    d = _model_dir(tmp_path)
    cfg = json.loads((d / "config.json").read_text())
    del cfg["eos_token_id"]
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 8]}))
    info = C.build_tokenizer_info(str(d), cfg)
    assert info["eos_token_id"] == 7


def test_load_calibration_texts_from_file(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("one\n\n\n\ntwo\n\nthree", encoding="utf-8")
    assert list(C.load_calibration_texts(str(f))) == ["one", "two", "three"]
