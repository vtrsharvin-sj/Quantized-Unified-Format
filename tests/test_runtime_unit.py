"""Unit tests for the runtime: `quf run` end to end, streaming, options, EOS stop."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from quf import cli
from quf.format.reader import QUFReader
from quf.format.writer import write_quf
from quf.runtime.generate import ContextLengthError, generate
from quf.runtime.model import QUFModel
from quf.tokenization.wrapper import TokenizerWrapper

from test_model import VOCAB, build_tiny_quf

WORDS = ["<eos>", "hello", "world", "foo", "bar", "baz", "qux", "<unk>"]


def _tokenizer() -> Tokenizer:
    vocab = {w: i for i, w in enumerate(WORDS)}
    for i in range(len(WORDS), VOCAB):
        vocab[f"w{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def make_quf(path, eos_token_id: int = 0):
    """Tiny random Qwen-shaped QUF file that also embeds tokenizer.json."""
    raw = path.parent / (path.stem + "_raw.quf")
    build_tiny_quf(raw)
    with QUFReader(raw) as r:
        tensors = {n: r.tensor(n).copy() for n in r.tensor_names()}
        header = r.header
    tokenizer = dict(header.tokenizer)
    tokenizer["eos_token_id"] = eos_token_id
    tokenizer["files"] = {"tokenizer.json": _tokenizer().to_str()}
    write_quf(path, dataclasses.replace(header, tokenizer=tokenizer, tensors=[]), tensors)
    return path


@pytest.fixture()
def quf_path(tmp_path):
    # eos id 31 is unlikely to be generated greedily in the run tests below
    return make_quf(tmp_path / "tiny.quf", eos_token_id=VOCAB - 1)


def _tok():
    return TokenizerWrapper(hf_tokenizer=_tokenizer())


def test_cli_run_end_to_end(quf_path, capsys):
    code = cli.main(["run", str(quf_path), "--device", "cpu", "--prompt", "hello world",
                     "--max-new-tokens", "5", "--temperature", "0"])
    out = capsys.readouterr().out
    assert code == 0
    model = QUFModel.load(quf_path, "cpu")
    ids = generate(model, _tok(), "hello world", max_new_tokens=5)
    assert out.strip() == _tok().decode(ids).strip()
    assert len(ids) >= 1


def test_cli_run_no_stream_matches_stream(quf_path, capsys):
    base = ["run", str(quf_path), "--device", "cpu", "--prompt", "foo bar",
            "--max-new-tokens", "4"]
    assert cli.main(base) == 0
    streamed = capsys.readouterr().out
    assert cli.main(base + ["--no-stream"]) == 0
    full = capsys.readouterr().out
    assert streamed.strip() == full.strip()


def test_streaming_fragments_concatenate_to_decoded_text(quf_path):
    model = QUFModel.load(quf_path, "cpu")
    tok = _tok()
    pieces: list[str] = []
    ids = generate(model, tok, [1, 2, 3], max_new_tokens=6, stream=pieces.append)
    assert len(pieces) >= 1
    assert "".join(pieces) == tok.decode(ids)


def test_cli_run_context_length_error(quf_path, capsys):
    code = cli.main(["run", str(quf_path), "--device", "cpu", "--prompt", "hello",
                     "--max-new-tokens", "100000"])
    assert code == 1
    assert "context length" in capsys.readouterr().err


def test_cli_run_option_parsing():
    args = cli._build_parser().parse_args(
        ["run", "m.quf", "--prompt", "hi", "--max-new-tokens", "7", "--temperature", "0.5",
         "--top-k", "10", "--top-p", "0.9", "--seed", "3", "--no-stream", "--device", "cpu"])
    assert (args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.seed) == (
        7, 0.5, 10, 0.9, 3)
    assert args.stream is False and args.device == "cpu"

    defaults = cli._build_parser().parse_args(["run", "m.quf", "--prompt", "hi"])
    assert defaults.temperature == 0.0 and defaults.stream is True
    assert defaults.top_k is None and defaults.top_p is None and defaults.seed == 0


def test_cli_run_rejects_bad_options(quf_path):
    assert cli.main(["run", str(quf_path)]) == 2  # --prompt is required
    assert cli.main(["run", str(quf_path), "--prompt", "x", "--device", "tpu"]) == 2
    assert cli.main(["run", str(quf_path), "--prompt", "x", "--max-new-tokens", "abc"]) == 2


def test_eos_stops_generation(tmp_path):
    first = make_quf(tmp_path / "a.quf", eos_token_id=VOCAB - 1)
    model = QUFModel.load(first, "cpu")
    tok = _tok()
    prompt = tok.encode("hello world")
    free = generate(model, tok, prompt, max_new_tokens=6)
    assert len(free) == 6

    # Make the second greedy token the EOS token: generation must stop right after it.
    eos = free[1]
    stop_at = free.index(eos) + 1
    model2 = QUFModel.load(make_quf(tmp_path / "b.quf", eos_token_id=eos), "cpu")
    pieces: list[str] = []
    out = generate(model2, tok, prompt, max_new_tokens=6, stream=pieces.append)
    assert out == free[:stop_at]
    assert out[-1] == eos
    # The EOS token is not streamed.
    assert "".join(pieces) == tok.decode(out[:-1]) or "".join(pieces) == tok.decode(out[:-1]).strip()


def test_max_new_tokens_zero_and_context_check(quf_path):
    model = QUFModel.load(quf_path, "cpu")
    assert generate(model, _tok(), [1, 2], max_new_tokens=0) == []
    with pytest.raises(ContextLengthError):
        generate(model, _tok(), [1] * 250, max_new_tokens=10)


def test_seeded_sampling_is_reproducible(quf_path):
    model = QUFModel.load(quf_path, "cpu")
    kw = dict(max_new_tokens=6, temperature=1.0, top_k=5, top_p=0.9)
    a = generate(model, _tok(), [1, 2, 3], seed=7, **kw)
    b = generate(model, _tok(), [1, 2, 3], seed=7, **kw)
    assert a == b
    assert all(isinstance(t, int) and 0 <= t < VOCAB for t in a)
    assert torch.is_tensor(torch.tensor(a))
