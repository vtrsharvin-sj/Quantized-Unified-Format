"""Unit tests for the Native_GPTQ (Requirements 14.4, 14.5, 14.9). CPU only."""
import contextlib
import io
import json

import pytest
import torch

from quf.quant import gptq as G
from quf.quant.dequant import reference_dequantize

CPU = torch.device("cpu")


# --- Hessian accumulator (14.9) -------------------------------------------

def test_hessian_matches_two_xtx_over_n():
    torch.manual_seed(0)
    x = torch.randn(50, 16)
    acc = G.HessianAccumulator(16, CPU)
    acc.add_batch(x)
    assert acc.n_samples == 50
    torch.testing.assert_close(acc.hessian, 2.0 * x.t() @ x / 50, rtol=1e-4, atol=1e-5)


def test_hessian_incremental_batches_and_3d_input():
    torch.manual_seed(1)
    x = torch.randn(4, 5, 16)  # [batch, seq, features] -> 20 rows
    acc = G.HessianAccumulator(16, CPU)
    acc.add_batch(x[:1])
    acc.add_batch(x[1:3])
    acc.add_batch(x[3:])
    flat = x.reshape(-1, 16)
    assert acc.n_samples == 20
    torch.testing.assert_close(acc.hessian, 2.0 * flat.t() @ flat / 20, rtol=1e-4, atol=1e-5)


def test_hessian_rejects_wrong_width():
    with pytest.raises(ValueError):
        G.HessianAccumulator(8, CPU).add_batch(torch.zeros(2, 7))


# --- Cholesky retry path (14.5) -------------------------------------------

def test_cholesky_retry_recovers_with_larger_damping():
    # Indefinite matrix: fails at damp=0.01 but succeeds once damping is raised.
    h = torch.diag(torch.tensor([1.0, 1.0, 1.0, -0.5]))
    u = G._inverse_cholesky_upper(h, 0.01, "layer")
    assert torch.isfinite(u).all()
    assert torch.allclose(u, torch.triu(u))


def test_cholesky_failure_raises_gptq_error_naming_layer():
    h = torch.diag(torch.tensor([1.0, 1.0, 1.0, -1e9]))
    with pytest.raises(G.GPTQError) as ei:
        G._inverse_cholesky_upper(h, 0.01, "model.layers.3.mlp.down_proj")
    assert ei.value.layer == "model.layers.3.mlp.down_proj"
    assert "model.layers.3.mlp.down_proj" in str(ei.value)


def test_gptq_quantize_raises_gptq_error_for_non_finite_hessian():
    w = torch.randn(8, 128)
    h = torch.full((128, 128), float("nan"))
    with pytest.raises(G.GPTQError) as ei:
        G.gptq_quantize(w, h, layer_name="bad.layer", device=CPU)
    assert "bad.layer" in str(ei.value)


def test_gptq_quantize_handles_dead_columns():
    torch.manual_seed(2)
    w = torch.randn(8, 128)
    x = torch.randn(64, 128)
    x[:, :4] = 0  # columns never activated -> zero Hessian diagonal
    acc = G.HessianAccumulator(128, CPU)
    acc.add_batch(x)
    qw = G.gptq_quantize(w, acc.hessian, device=CPU)
    assert reference_dequantize(qw).shape == (8, 128)


# --- Sequential quantization through the Converter (14.4) ------------------

transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from quf.cli import main  # noqa: E402
from quf.convert import converter as C  # noqa: E402
from quf.format.reader import QUFReader  # noqa: E402


def _tiny_qwen_dir(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=27, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        tie_word_embeddings=False, eos_token_id=3, bos_token_id=1,
    )
    model = Qwen2ForCausalLM(cfg)
    d = tmp_path / "tiny"
    model.save_pretrained(d)
    vocab = {chr(97 + i): i for i in range(26)}
    vocab["[UNK]"] = 26
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    return d, model


TEXTS = ["a b c d e f g h i j k l m n o p"] * 8


def test_quantize_model_sequential_covers_all_linears(tmp_path):
    _, model = _tiny_qwen_dir(tmp_path)
    calib = torch.randint(0, 27, (2, 8))
    seen = []
    q = G.quantize_model_sequential(
        model, calib, device=CPU, progress=lambda i, n: seen.append((i, n))
    )
    expected = {
        f"model.layers.{i}.{p}.{n}"
        for i in range(2)
        for p, names in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for n in names
    }
    assert set(q) == expected
    assert seen == [(1, 2), (2, 2)]
    w = q["model.layers.0.mlp.up_proj"]
    assert w.in_features == 128 and w.out_features == 256
    # Model weights were replaced by the dequantized values.
    lin = model.model.layers[0].mlp.up_proj
    torch.testing.assert_close(
        lin.weight.detach(), reference_dequantize(w).to(lin.weight.dtype)
    )


def test_quantize_model_sequential_rejects_bad_calibration(tmp_path):
    _, model = _tiny_qwen_dir(tmp_path)
    with pytest.raises(ValueError):
        G.quantize_model_sequential(model, torch.zeros(0, 8, dtype=torch.long), device=CPU)


def test_convert_tiny_qwen_then_info(tmp_path, monkeypatch):
    d, _ = _tiny_qwen_dir(tmp_path)
    out = tmp_path / "tiny.quf"
    # Force CPU so this test does not touch the GPU.
    from quf.convert import gptq as cg

    orig = cg.run_native_gptq
    monkeypatch.setattr(
        cg, "run_native_gptq", lambda *a, **k: orig(*a, **{**k, "device": "cpu"})
    )
    res = C.convert_checkpoint(
        str(d), out, n_samples=2, seq_len=8, texts=TEXTS, echo=None, embed_precision="fp16"
    )
    assert out.exists() and not (tmp_path / "tiny.quf.partial").exists()
    assert 0 < res.ratio < 2

    with QUFReader(out) as r:
        names = set(r.header.tensors) if not isinstance(r.header.tensors, list) else {
            t["name"] if isinstance(t, dict) else t.name for t in r.header.tensors
        }
        assert "model.layers.1.mlp.down_proj.qweight" in names
        assert "model.layers.0.self_attn.q_proj.scales" in names
        assert "model.layers.0.self_attn.q_proj.qzeros" in names
        assert "model.embed_tokens.weight" in names
        assert r.header.quantization["group_size"] == 128
        assert r.header.quantization["bits"] == 4

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["info", str(out)])
    text = buf.getvalue()
    assert code == 0
    assert json.dumps(C.build_model_config(json.loads((d / "config.json").read_text()),
                                          "Qwen2ForCausalLM")["hidden_size"]) in text
    assert '"group_size": 128' in text
