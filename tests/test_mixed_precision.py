"""Mixed precision in the Converter and `quf info` (Requirements 19.1, 22.1, 22.2, 22.5, 22.6)."""
import contextlib
import io

import pytest

from quf.convert.precision import PrecisionPlan, PrecisionRuleError, SizeReport
from quf.format.reader import QUFReader

# --- PrecisionPlan ----------------------------------------------------------


def test_plan_default_int4_and_last_rule_wins():
    plan = PrecisionPlan.from_options(["*.down_proj=int8,lm_head=fp16", "*.layers.0.*.down_proj=int4"])
    assert plan.precision("model.layers.3.mlp.down_proj") == "int8"
    assert plan.precision("model.layers.0.mlp.down_proj") == "int4"
    assert plan.precision("model.layers.3.self_attn.q_proj") == "int4"
    assert plan.explicit("lm_head") == "fp16" and plan.explicit("model.layers.1.mlp.up_proj") is None


def test_plan_presets_and_int8_modules():
    plan = PrecisionPlan.from_options(
        ["*.v_proj=fp16"], presets=["sensitive"], int8_modules=["*.o_proj"], num_layers=4
    )
    assert plan.precision("model.layers.3.self_attn.q_proj") == "int8"  # last layer
    assert plan.precision("model.layers.2.mlp.down_proj") == "int8"
    assert plan.precision("model.layers.1.self_attn.o_proj") == "int8"
    assert plan.precision("model.layers.3.self_attn.v_proj") == "fp16"  # explicit rule overrides
    assert not PrecisionPlan()


@pytest.mark.parametrize("bad", ["down_proj", "=int8", "x=int3"])
def test_plan_rejects_bad_rules(bad):
    with pytest.raises(PrecisionRuleError):
        PrecisionPlan.from_options([bad])
    with pytest.raises(PrecisionRuleError):
        PrecisionPlan.from_options(presets=["nope"])


# --- SizeReport -------------------------------------------------------------


def test_size_report_sums_tensor_bytes_by_class():
    tensors = [
        ("a.qweight", "int32", (16, 64)),
        ("a.scales", "fp16", (1, 64)),
        ("a.qzeros", "int32", (1, 8)),
        ("b.qweight_int8", "int8", (32, 64)),
        ("b.scales", "fp16", (32,)),
        ("c.weight", "fp16", (8, 8)),
        ("model.embed_tokens.weight", "fp16", (10, 8)),
    ]
    rep = SizeReport.from_tensors(tensors, {"c": "fp16"}, file_bytes=10_000)
    assert rep.int4 == 16 * 64 * 4 + 64 * 2 + 8 * 4
    assert rep.int8 == 32 * 64 + 32 * 2
    assert rep.fp16 == 8 * 8 * 2 + 10 * 8 * 2
    assert rep.tensor_bytes == sum(
        {"fp16": 2, "int32": 4, "int8": 1}[d] * _n(s) for _, d, s in tensors
    )
    assert rep.fp16_equivalent == 2 * (16 * 8 * 64 + 32 * 64 + 64 + 80)
    text = rep.format()
    assert "int4:" in text and "int8:" in text and "fp16:" in text and "ratio" in text
    assert rep.groups["fp16"] == {"c": 1}


def _n(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# --- End to end on a tiny Qwen2 model ---------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from quf.cli import main  # noqa: E402
from quf.convert import converter as C  # noqa: E402
from quf.convert import gptq as cg  # noqa: E402

TEXTS = ["a b c d e f g h i j k l m n o p"] * 8


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


def _convert(tmp_path, out, **kw):
    msgs = []
    res = C.convert_checkpoint(
        str(_tiny_dir(tmp_path)), out, n_samples=2, seq_len=8, texts=TEXTS, echo=msgs.append, **kw
    )
    return res, msgs


def test_default_conversion_stays_version_1(tmp_path, cpu_run):
    out = tmp_path / "d.quf"
    res, msgs = _convert(tmp_path, out)
    assert out.read_bytes()[4:8] == (1).to_bytes(4, "little")
    assert "modules" not in QUFReader(out).header.quantization
    assert res.report.int8 == 0 and "Size by precision" in msgs[0]


def test_mixed_precision_conversion_writes_modules_and_tensors(tmp_path, cpu_run):
    out = tmp_path / "m.quf"
    plan = PrecisionPlan.from_options(
        ["*.v_proj=fp16", "lm_head=int8"], int8_modules=["*.down_proj"]
    )
    res, msgs = _convert(tmp_path, out, precision_plan=plan, group_size=64, act_order=True, clip_search=True)
    assert out.read_bytes()[4:8] == (2).to_bytes(4, "little")
    with QUFReader(out) as r:
        q = r.header.quantization
        names = {t.name for t in r.header.tensors}
        assert q["group_size"] == 64 and q["act_order"] is True
        assert q["modules"]["model.layers.1.mlp.down_proj"] == "int8"
        assert q["modules"]["model.layers.0.self_attn.v_proj"] == "fp16"
        assert q["modules"]["model.layers.0.self_attn.q_proj"] == "int4"
        assert q["modules"]["lm_head"] == "int8"
        assert "model.layers.1.mlp.down_proj.qweight_int8" in names
        assert "model.layers.1.mlp.down_proj.scales" in names
        assert "model.layers.0.self_attn.v_proj.weight" in names
        assert "model.layers.0.self_attn.q_proj.g_idx" in names
        assert "lm_head.qweight_int8" in names and "lm_head.weight" not in names
        assert "model.layers.0.self_attn.v_proj.qweight" not in names
        rep = SizeReport.from_header(r.header, file_bytes=r.file_size)
    assert rep.int8 > 0 and rep.int4 > 0
    assert rep.tensor_bytes == sum(t.nbytes for t in QUFReader(out).header.tensors)
    assert "int8:" in msgs[0] and "int8: 0 bytes" not in msgs[0]
    assert res.report.int8 == rep.int8 and res.report.int4 == rep.int4

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["info", str(out)]) == 0
    text = buf.getvalue()
    assert "Size by precision" in text and "down_proj x2" in text and "v_proj x2" in text


def test_cli_convert_passes_options(tmp_path, cpu_run, monkeypatch):
    d = _tiny_dir(tmp_path)
    out = tmp_path / "c.quf"
    monkeypatch.setattr(C, "load_calibration_texts", lambda ds: TEXTS)
    code = main([
        "convert", str(d), str(out), "--dataset", "x", "--samples", "2", "--seq-len", "8",
        "--group-size", "32", "--act-order", "--clip-search", "--preset", "sensitive",
        "--mixed", "lm_head=int8",
    ])
    assert code == 0
    with QUFReader(out) as r:
        q = r.header.quantization
        assert q["group_size"] == 32 and q["act_order"] is True
        # the last layer (index 1) and every down_proj are int8
        assert q["modules"]["model.layers.1.self_attn.q_proj"] == "int8"
        assert q["modules"]["model.layers.0.mlp.down_proj"] == "int8"
        assert q["modules"]["model.layers.0.mlp.up_proj"] == "int4"


# --- Size_Candidates (task 27.1) ----------------------------------------------

def _int8_modules(preset, names, num_layers=4):
    plan = PrecisionPlan.from_options(presets=[preset], num_layers=num_layers)
    return {n for n in names if plan.precision(n) == "int8"}


def test_size_candidate_presets_select_expected_modules():
    names = [f"model.layers.{i}.{p}" for i in range(4)
             for p in ("self_attn.q_proj", "mlp.up_proj", "mlp.down_proj")]
    last = {n for n in names if ".layers.3." in n}
    down = {n for n in names if n.endswith("down_proj")}
    assert _int8_modules("sensitive", names) == down | last
    assert _int8_modules("size_a", names) == down  # last layer int4
    assert _int8_modules("size_b", names) == last  # down_proj int4
    assert _int8_modules("size_c", names) == set()  # A and B together
    assert _int8_modules("size_a", names) | _int8_modules("size_b", names) == _int8_modules("sensitive", names)


def test_cli_accepts_size_presets_and_records_modules(tmp_path, cpu_run, monkeypatch):
    d = _tiny_dir(tmp_path)
    monkeypatch.setattr(C, "load_calibration_texts", lambda ds: TEXTS)
    expect = {  # preset -> (layer 0 down_proj, layer 1 q_proj)
        "size_a": ("int8", "int4"), "size_b": ("int4", "int8"), "size_c": ("int4", "int4"),
    }
    for preset, (down, last_q) in expect.items():
        out = tmp_path / f"{preset}.quf"
        code = main(["convert", str(d), str(out), "--dataset", "x", "--samples", "2", "--seq-len", "8",
                     "--group-size", "32", "--act-order", "--preset", preset])
        assert code == 0
        with QUFReader(out) as r:
            q = r.header.quantization
            assert q["group_size"] == 32 and q["act_order"] is True
            mods = q.get("modules", {})
            assert mods.get("model.layers.0.mlp.down_proj", "int4") == down
            assert mods.get("model.layers.1.self_attn.q_proj", "int4") == last_q
            assert "model.layers.0.self_attn.q_proj.g_idx" in {t.name for t in r.header.tensors}
