"""Tests for the Sensitivity_Analyzer and Ablation_Harness (quf/analysis/sensitivity.py)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quf.analysis.sensitivity import (
    Scheme,
    apply_ablation,
    decoder_linear_names,
    module_errors,
    restore_weights,
    standard_subsets,
)
from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import quantize_groupwise


def _tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=27, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def model():
    return _tiny()


def test_module_errors_one_entry_per_decoder_linear(model):
    calib = torch.randint(0, 27, (2, 16))
    errs = module_errors(model, calib)
    names = decoder_linear_names(model)
    assert len(names) == 2 * 7
    assert [e.name for e in errs] == names
    assert {e.kind for e in errs} == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    assert {e.layer for e in errs} == {0, 1}
    assert all(0.0 < e.rel_output_error < 1.0 for e in errs)


def test_standard_subsets_partition(model):
    subs = standard_subsets(model)
    assert set(subs) == {"first", "last", "v_proj", "down_proj", "rest"}
    assert all(n.startswith("model.layers.0.") for n in subs["first"])
    assert subs["rest"].isdisjoint(subs["first"] | subs["last"] | subs["v_proj"] | subs["down_proj"])
    assert set().union(*subs.values()) == set(decoder_linear_names(model))


# Feature: quf-quantized-format, Property 22: Ablation changes only the selected modules
# **Validates: Requirements 18.1, 18.2**
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_ablation_changes_only_selected(model, data):
    names = decoder_linear_names(model)
    subset = set(data.draw(st.lists(st.sampled_from(names), max_size=4, unique=True)))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    mods = dict(model.named_modules())
    originals = apply_ablation(model, subset, Scheme())
    try:
        assert set(originals) == subset
        for n, p in model.named_parameters():
            mod_name = n.rsplit(".", 1)[0]
            if mod_name in subset and n.endswith(".weight"):
                w = before[n].to(torch.float32)
                expected = reference_dequantize(quantize_groupwise(w, 128, False))
                assert torch.equal(p.detach(), expected.to(p.dtype))
            else:
                assert torch.equal(p.detach(), before[n]), n
        assert all(isinstance(mods[n], torch.nn.Linear) for n in subset)
    finally:
        restore_weights(model, originals)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n])


def test_apply_ablation_rejects_non_linear(model):
    with pytest.raises(ValueError):
        apply_ablation(model, ["model.embed_tokens"])
