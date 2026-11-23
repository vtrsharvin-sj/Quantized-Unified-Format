import json

import pytest
import torch

pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers

from quf.convert import converter as C
from quf.convert.gptq import GPTQResult
from quf.format.reader import QUFReader
from quf.quant.quantize import quantize_groupwise


def _model_dir(tmp_path, arch="Qwen2ForCausalLM"):
    d = tmp_path / "model"
    d.mkdir()
    vocab = {chr(97 + i): i for i in range(26)}
    vocab["[UNK]"] = 26
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    (d / "config.json").write_text(json.dumps({
        "architectures": [arch], "hidden_size": 128, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 256,
        "vocab_size": 27, "max_position_embeddings": 64, "eos_token_id": 3,
    }))
    return d


def _fake_gptq(source, calib, *, group_size, symmetric):
    assert len(calib) == 2 and all(len(s) == 8 for s in calib)
    torch.manual_seed(0)
    qw = quantize_groupwise(torch.randn(16, 128), group_size=128, symmetric=symmetric)
    return GPTQResult(
        linears={"model.layers.0.mlp.up_proj": qw},
        others={"model.embed_tokens.weight": torch.randn(27, 128).numpy()},
        symmetric=symmetric,
    )


TEXTS = ["a b c d e f g h i j k l"] * 10


def test_convert_roundtrip(tmp_path):
    out = tmp_path / "o.quf"
    msgs = []
    res = C.convert_checkpoint(str(_model_dir(tmp_path)), out, n_samples=2, seq_len=8,
                               texts=TEXTS, quantizer=_fake_gptq, echo=msgs.append, embed_precision="fp16")
    assert res.size_bytes == out.stat().st_size and msgs and "bytes" in msgs[0]
    assert not (tmp_path / "o.quf.partial").exists()
    r = QUFReader(out)
    assert r.header.quantization["symmetric"] is False
    assert r.header.model_config["architecture"] == "Qwen2ForCausalLM"
    assert r.tensor("model.embed_tokens.weight").dtype.name == "float16"
    r.close()


def test_rejects_non_qwen_and_existing(tmp_path):
    d = _model_dir(tmp_path, "LlamaForCausalLM")
    with pytest.raises(C.UnsupportedArchitectureError, match="LlamaForCausalLM"):
        C.convert_checkpoint(str(d), tmp_path / "x.quf", texts=TEXTS)
    out = tmp_path / "e.quf"
    out.write_bytes(b"keep")
    with pytest.raises(C.OutputExistsError):
        C.convert_checkpoint(str(d), out, texts=TEXTS)
    assert out.read_bytes() == b"keep"


def test_failure_leaves_no_partial(tmp_path, monkeypatch):
    out = tmp_path / "f.quf"

    def boom(path, header, tensors):
        open(path, "wb").write(b"x")
        raise RuntimeError("injected")

    monkeypatch.setattr(C, "write_quf", boom)
    with pytest.raises(RuntimeError):
        C.convert_checkpoint(str(_model_dir(tmp_path)), out, n_samples=2, seq_len=8,
                             texts=TEXTS, quantizer=_fake_gptq)
    assert not out.exists() and not (tmp_path / "f.quf.partial").exists()
