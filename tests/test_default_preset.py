"""Default precision preset of `quf convert` (Requirement 32, task 27.4).

The default is size_f (task 34.4): k_proj, v_proj and the last layer int8, the other MLP
modules (including down_proj) int4, with group size 32, act-order and an int8 embedding (group 64). No new bit width was added, so there is no new dtype test.
"""
import contextlib
import io

import pytest

from quf import cli
from quf.convert.precision import PrecisionPlan

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from quf.cli import main  # noqa: E402
from quf.convert import converter as C  # noqa: E402
from quf.convert import gptq as cg  # noqa: E402
from quf.format.reader import QUFReader  # noqa: E402

TEXTS = ["a b c d e f g h i j k l m n o p"] * 8
DOWN0 = "model.layers.0.mlp.down_proj"
DOWN1 = "model.layers.1.mlp.down_proj"
Q0 = "model.layers.0.self_attn.q_proj"
Q1 = "model.layers.1.self_attn.q_proj"
K0 = "model.layers.0.self_attn.k_proj"


def _tiny_dir(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=27, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        tie_word_embeddings=False, eos_token_id=3, bos_token_id=1,
    )
    d = tmp_path / "tiny"
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    vocab = {chr(97 + i): i for i in range(26)}
    vocab["[UNK]"] = 26
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    return d


@pytest.fixture
def cpu_run(monkeypatch):
    orig = cg.run_native_gptq
    monkeypatch.setattr(cg, "run_native_gptq", lambda *a, **k: orig(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(C, "load_calibration_texts", lambda ds: TEXTS)


def _convert(tmp_path, name, *extra):
    out = tmp_path / name
    src = tmp_path / "tiny"
    if not src.exists():
        _tiny_dir(tmp_path)
    code = main(["convert", str(src), str(out), "--dataset", "x", "--samples", "2", "--seq-len", "8", *extra])
    assert code == 0
    return out


def test_default_preset_constant_and_plan():
    assert cli.DEFAULT_PRESET == "size_f" and cli.DEFAULT_EMBED_PRECISION == "int8"
    plan = PrecisionPlan.from_options(presets=[cli.DEFAULT_PRESET], num_layers=4)
    assert plan.precision("model.layers.3.mlp.down_proj") == "int8"  # last layer
    assert plan.precision("model.layers.3.self_attn.q_proj") == "int8"  # last layer
    assert plan.precision("model.layers.0.mlp.down_proj") == "int4"
    assert plan.precision("model.layers.0.mlp.up_proj") == "int4"
    assert plan.precision("model.layers.0.self_attn.q_proj") == "int4"
    assert plan.precision("model.layers.0.self_attn.o_proj") == "int4"
    assert plan.precision("model.layers.0.self_attn.k_proj") == "int8"  # k and v are int8 in every layer
    assert plan.precision("model.layers.0.self_attn.v_proj") == "int8"


def test_help_names_size_f_default(capsys):
    try:
        main(["convert", "--help"])
    except SystemExit:
        pass
    out = " ".join(capsys.readouterr().out.split())
    assert "Default when no --preset, --mixed or --int8-modules is given: size_f" in out
    assert "default: int8" in out
    assert "sensitive (down_proj and the last layer at int8; the v020 plan)" in out


def test_default_convert_applies_size_f(tmp_path, cpu_run):
    out = _convert(tmp_path, "default.quf")
    with QUFReader(out) as r:
        q = r.header.quantization
        names = {t.name for t in r.header.tensors}
        assert q["group_size"] == 32 and q["act_order"] is True
        mods = q["modules"]
        # k_proj and v_proj are int8 in every layer, last layer (index 1) is int8, the rest is int4
        assert mods[Q1] == "int8" and mods[DOWN1] == "int8" and mods[K0] == "int8" and mods.get(Q0, "int4") == "int4"
        assert mods.get(DOWN0, "int4") == "int4" and mods.get("model.layers.0.mlp.up_proj", "int4") == "int4"
        assert f"{Q1}.qweight_int8" in names and f"{K0}.qweight_int8" in names
        assert q["embedding"]["bits"] == 8 and q["embedding"]["group_size"] == 64  # int8 g64 embedding
        # down_proj in layer 0 is int4 group 32 with act-order (g_idx present)
        assert f"{DOWN0}.qweight" in names and f"{DOWN0}.g_idx" in names
        d0 = next(t for t in r.header.tensors if t.name == f"{DOWN0}.scales")
        assert d0.shape[0] == 256 // 32  # intermediate_size / group size

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["info", str(out)]) == 0
    assert "Size by precision" in buf.getvalue()


def test_sensitive_preset_reproduces_v020(tmp_path, cpu_run):
    out = _convert(tmp_path, "sens.quf", "--preset", "sensitive")
    with QUFReader(out) as r:
        mods = r.header.quantization["modules"]
    assert mods[DOWN0] == "int8" and mods[DOWN1] == "int8"
    assert mods[Q1] == "int8"
    assert mods.get(Q0, "int4") == "int4"
    assert mods.get("model.layers.0.mlp.up_proj", "int4") == "int4"


def test_none_preset_is_all_int4(tmp_path, cpu_run):
    out = _convert(tmp_path, "none.quf", "--preset", "none")
    with QUFReader(out) as r:
        q = r.header.quantization
        names = {t.name for t in r.header.tensors}
        assert all(v == "int4" for v in q.get("modules", {}).values())
        assert not any(n.endswith(".qweight_int8") for n in names)
        assert f"{Q1}.qweight" in names and f"{DOWN1}.qweight" in names
