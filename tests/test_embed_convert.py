"""Task 34.2: --embed-precision in the converter, CLI, SizeReport and ``quf info``."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from quf.cli import _embedding_line, _build_parser  # noqa: E402
from quf.convert import converter as C  # noqa: E402
from quf.convert import gptq as cg  # noqa: E402
from quf.convert.precision import SizeReport  # noqa: E402
from quf.format.reader import QUFReader  # noqa: E402
from quf.quant.embedding import (  # noqa: E402
    quantize_embedding_clip,
    quantize_embedding_rtn,
)

TEXTS = ["a b c d e f g h i j k l m n o p"] * 8


def _tiny_dir(tmp_path, tie):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=27, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        tie_word_embeddings=tie, eos_token_id=3, bos_token_id=1,
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


def _convert(tmp_path, out, tie=False, **kw):
    return C.convert_checkpoint(
        str(_tiny_dir(tmp_path, tie)), out, n_samples=2, seq_len=8, texts=TEXTS, echo=None, **kw
    )


def _version(path):
    return int.from_bytes(path.read_bytes()[4:8], "little")


def test_fp16_default_keeps_version_and_tensors(tmp_path, cpu_run):
    out = tmp_path / "f.quf"
    _convert(tmp_path, out, embed_precision="fp16")
    assert _version(out) == 1
    with QUFReader(out) as r:
        names = {t.name for t in r.header.tensors}
        assert "model.embed_tokens.weight" in names and "lm_head.weight" in names
        assert "embedding" not in r.header.quantization
    assert _embedding_line(r.header.quantization) == "Embedding precision: fp16"


@pytest.mark.parametrize("precision,bits,gs", [("int8", 8, 64), ("int4", 4, 64), ("int4", 4, 32)])
def test_quantized_embedding_untied(tmp_path, cpu_run, precision, bits, gs):
    out = tmp_path / "q.quf"
    res = _convert(tmp_path, out, embed_precision=precision, embed_group_size=gs)
    assert _version(out) == 3
    suffix = ".embed_q4" if bits == 4 else ".embed_q8"
    with QUFReader(out) as r:
        q = r.header.quantization
        names = {t.name: t for t in r.header.tensors}
        for key, name in (("embedding", "model.embed_tokens"), ("lm_head", "lm_head")):
            assert q[key]["name"] == name and q[key]["bits"] == bits
            assert q[key]["group_size"] == gs and q[key]["clip_search"] is True
            assert name + suffix in names and name + ".embed_scales" in names
            assert name + ".weight" not in names
            assert names[name + ".embed_scales"].shape == [27, 128 // gs]
        line = _embedding_line(q)
    assert f"embedding int{bits} group {gs}" in line and "lm_head int" in line
    rep = res.report
    emb_bytes = 2 * (27 * (64 if bits == 4 else 128) + 27 * (128 // gs) * 2)
    assert SizeReport.from_header(r.header).tensor_bytes == rep.tensor_bytes
    assert (rep.int4 if bits == 4 else rep.int8) >= emb_bytes


def test_size_report_counts_embedding_by_precision(tmp_path, cpu_run):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for p in (a, b, c):
        p.mkdir()
    r16 = _convert(a, a / "x.quf", embed_precision="fp16").report
    r8 = _convert(b, b / "x.quf", embed_precision="int8").report
    r4 = _convert(c, c / "x.quf", embed_precision="int4").report
    n = 27 * 128
    # tensors that moved out of fp16 are two tables of n elements
    assert r16.fp16 - r8.fp16 == 2 * (2 * n)
    assert r8.int8 - r16.int8 == 2 * (n + 27 * (128 // 64) * 2)
    assert r4.int4 - r16.int4 == 2 * (n // 2 + 27 * (128 // 64) * 2)
    assert r16.fp16_equivalent == r8.fp16_equivalent == r4.fp16_equivalent
    assert r4.tensor_bytes < r8.tensor_bytes < r16.tensor_bytes
    assert "embed_tokens" in r4.groups["int4"] and "embed_tokens" in r8.groups["int8"]


def test_tied_embedding_has_single_table(tmp_path, cpu_run):
    out = tmp_path / "t.quf"
    _convert(tmp_path, out, tie=True, embed_precision="int8")
    with QUFReader(out) as r:
        assert "embedding" in r.header.quantization and "lm_head" not in r.header.quantization
        assert not any(t.name.startswith("lm_head") for t in r.header.tensors)


def test_rtn_method_recorded(tmp_path, cpu_run):
    out = tmp_path / "r.quf"
    _convert(tmp_path, out, embed_precision="int4", embed_method="rtn")
    with QUFReader(out) as r:
        spec = r.header.quantization["embedding"]
        assert spec["method"] == "rtn" and spec["clip_search"] is False
        t = {t.name: r.tensor(t.name) for t in r.header.tensors}
    w = torch.from_numpy(t["model.embed_tokens.embed_q4"].copy())
    assert w.dtype == torch.int8


def test_invalid_options_rejected(tmp_path):
    with pytest.raises(C.ConvertError, match="embed precision"):
        _convert(tmp_path, tmp_path / "x.quf", embed_precision="int2")
    with pytest.raises(C.ConvertError, match="group size"):
        _convert(tmp_path, tmp_path / "x.quf", embed_group_size=128)


def test_cli_flags_parse():
    p = _build_parser()
    a = p.parse_args(["convert", "src", "o.quf"])
    assert (a.embed_precision, a.embed_group_size, a.embed_method) == ("int8", 64, "clip")  # defaults chosen in task 34.4
    a = p.parse_args(["convert", "src", "o.quf", "--embed-precision", "int4", "--embed-group-size", "32"])
    assert (a.embed_precision, a.embed_group_size) == ("int4", 32)
    with pytest.raises(SystemExit):
        p.parse_args(["convert", "src", "o.quf", "--embed-precision", "int2"])


def _deq32(qe):
    """Dequantize in float32, so the fp16 output rounding does not blur the comparison."""
    p = torch.from_numpy(qe.packed.copy())
    if qe.bits == 4:
        u = p.view(torch.uint8)
        q = torch.stack((u & 0xF, u >> 4), dim=-1).reshape(u.shape[0], -1).float() - 8
    else:
        q = p.float()
    return q * torch.from_numpy(qe.scales.copy()).float().repeat_interleave(qe.group_size, dim=-1)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("gs", [32, 64])
def test_clip_error_not_above_rtn(bits, gs):
    g = torch.Generator().manual_seed(bits * 100 + gs)
    w = torch.randn(200, 128, generator=g) * 0.05
    w[::7, 3] *= 12  # outliers make clipping useful
    rtn = _deq32(quantize_embedding_rtn(w, bits, gs))
    clip = _deq32(quantize_embedding_clip(w, bits, gs))
    e_rtn = ((rtn - w) ** 2).reshape(200, -1, gs).sum(-1)
    e_clip = ((clip - w) ** 2).reshape(200, -1, gs).sum(-1)
    assert (e_clip <= e_rtn + 1e-9).all()
    if bits == 4:  # at int8 the unclipped range is almost always already best
        assert e_clip.sum() < e_rtn.sum()
