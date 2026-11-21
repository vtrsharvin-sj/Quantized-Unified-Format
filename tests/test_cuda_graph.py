"""Property 32: CUDA graph decode equals eager decode (GPU)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import test_model as tm
from quf.runtime.cuda_graph import make_decoder
from quf.runtime.generate import generate
from quf.runtime.model import QUFModel

pytestmark = pytest.mark.gpu

# (hidden, heads, kv_heads, inter, layers): tiny, MHA, and the GQA head ratios of
# Qwen2.5-0.5B (14:2) and Qwen2.5-1.5B (12:2) with small depth.
SHAPES = {
    "tiny": (128, 4, 2, 128, 2),
    "mha": (128, 2, 2, 128, 1),
    "qwen0.5b": (896, 14, 2, 256, 2),
    "qwen1.5b": (1536, 12, 2, 256, 2),
}

_MODELS: dict[str, QUFModel] = {}


def _model(name: str, tmp_root) -> QUFModel:
    if name not in _MODELS:
        hidden, heads, kv, inter, layers = SHAPES[name]
        saved = (tm.HIDDEN, tm.HEADS, tm.KV_HEADS, tm.INTER, tm.LAYERS)
        tm.HIDDEN, tm.HEADS, tm.KV_HEADS, tm.INTER, tm.LAYERS = hidden, heads, kv, inter, layers
        try:
            path = tmp_root / f"{name}.quf"
            tm.build_tiny_quf(path, seed=len(name))
        finally:
            tm.HIDDEN, tm.HEADS, tm.KV_HEADS, tm.INTER, tm.LAYERS = saved
        _MODELS[name] = QUFModel.load(path, "cuda")
    return _MODELS[name]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("graph")
    yield {n: _model(n, root) for n in SHAPES}
    _MODELS.clear()


@settings(max_examples=100, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    shape=st.sampled_from(sorted(SHAPES)),
    prompt=st.lists(st.integers(1, tm.VOCAB - 1), min_size=1, max_size=24),
    new=st.integers(2, 24),
)
def test_graph_decode_equals_eager(models, shape, prompt, new):
    """Feature: quf-quantized-format, Property 32: Graph decode equals eager decode

    **Validates: Requirements 31.2, 31.3, 31.6, 31.10**
    """
    model = models[shape]
    eager = generate(model, None, prompt, max_new_tokens=new, cuda_graph=False)
    graph = generate(model, None, prompt, max_new_tokens=new, cuda_graph=True)
    assert graph == eager

    # Logits: teacher-force the same tokens through both decoders.
    n = len(prompt) + new
    ca, cb = model.new_static_cache(n), model.new_static_cache(n)
    ids = torch.tensor([prompt])
    la, lb = model.forward(ids, ca), model.forward(ids, cb)
    assert torch.allclose(la.float(), lb.float(), atol=1e-2)
    dec = make_decoder(model, cb, True)
    assert dec is not None, "CUDA graph capture failed"
    tok = int(la[0, -1].argmax())
    for _ in range(new - 1):
        ea = model.decode_token(tok, ca).float().clone()
        gb = dec.decode_token(tok, cb).float().clone()
        assert torch.allclose(ea, gb, atol=1e-2), float((ea - gb).abs().max())
        assert int(ea[0, -1].argmax()) == int(gb[0, -1].argmax())
        tok = int(ea[0, -1].argmax())
