"""Property 15: Existing output is preserved.

**Validates: Requirements 7.7**
"""
import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings, strategies as st
from tokenizers import Tokenizer, models, pre_tokenizers

from quf.convert import converter as C


def _model_dir(root: Path) -> Path:
    d = root / "model"
    d.mkdir()
    vocab = {chr(97 + i): i for i in range(26)}
    vocab["[UNK]"] = 26
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    (d / "config.json").write_text(json.dumps({
        "architectures": ["Qwen2ForCausalLM"], "hidden_size": 128, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 256,
        "vocab_size": 27, "max_position_embeddings": 64, "eos_token_id": 3,
    }))
    return d


def _must_not_run(*args, **kwargs):
    raise AssertionError("quantizer must not run when output exists")


@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(content=st.binary(min_size=0, max_size=512))
def test_existing_output_preserved(content):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model = _model_dir(root)
        out = root / "existing.quf"
        out.write_bytes(content)
        with pytest.raises(C.OutputExistsError):
            C.convert_checkpoint(str(model), out, texts=["a b c"], quantizer=_must_not_run,
                                 echo=None)
        assert out.read_bytes() == content
        assert not (root / "existing.quf.partial").exists()
