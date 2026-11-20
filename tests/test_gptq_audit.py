"""Audit tests for quf/quant/gptq.py (task 15.3, Requirements 18.5, 18.6). CPU only.

One test per audited item: Hessian accumulation, damping, dead columns,
calibration length, sequential / true-sequential ordering, fp16 scale rounding,
and tied embedding handling.
"""
import numpy as np
import pytest
import torch

from quf.quant import gptq as G
from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import quantize_groupwise

CPU = torch.device("cpu")


def _hessian(x):
    x = x.reshape(-1, x.shape[-1]).double()
    return 2.0 * x.t() @ x / x.shape[0]


# --- Hessian accumulation ---------------------------------------------------

def test_hessian_is_symmetric_and_scale_invariant_to_batching():
    torch.manual_seed(0)
    x = torch.randn(7, 9, 32)
    one = G.HessianAccumulator(32, CPU)
    one.add_batch(x)
    many = G.HessianAccumulator(32, CPU)
    for i in range(7):
        many.add_batch(x[i])
    torch.testing.assert_close(one.hessian, many.hessian, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(one.hessian, one.hessian.t())
    assert one.n_samples == many.n_samples == 63


def test_hessian_accumulates_fp16_input_in_fp32():
    x = torch.randn(40, 16).half()
    acc = G.HessianAccumulator(16, CPU)
    acc.add_batch(x)
    assert acc.hessian.dtype == torch.float32
    torch.testing.assert_close(acc.hessian.double(), _hessian(x.float()), rtol=1e-4, atol=1e-5)


# --- Damping ----------------------------------------------------------------

def test_damping_is_fraction_of_mean_diagonal():
    torch.manual_seed(1)
    a = torch.randn(24, 12)
    h = a.t() @ a  # PSD
    damp = 0.05
    got = G._inverse_cholesky_upper(h, damp, "l")
    hd = h.double() + damp * torch.diag(h).double().mean() * torch.eye(12, dtype=torch.float64)
    want = torch.linalg.cholesky(torch.linalg.inv(hd), upper=True)
    torch.testing.assert_close(got.double(), want, rtol=1e-3, atol=1e-4)


def test_rank_deficient_hessian_quantizes_finitely():
    torch.manual_seed(2)
    x = torch.randn(4, 128)  # rank 4 << 128
    acc = G.HessianAccumulator(128, CPU)
    acc.add_batch(x)
    qw = G.gptq_quantize(torch.randn(16, 128), acc.hessian, device=CPU)
    assert torch.isfinite(reference_dequantize(qw).float()).all()


# --- Dead columns -----------------------------------------------------------

def test_dead_columns_dequantize_to_exact_zero():
    torch.manual_seed(3)
    w = torch.randn(8, 256)
    x = torch.randn(64, 256)
    dead = [0, 5, 130, 255]
    x[:, dead] = 0
    acc = G.HessianAccumulator(256, CPU)
    acc.add_batch(x)
    for sym in (False, True):
        qw = G.gptq_quantize(w, acc.hessian, symmetric=sym, device=CPU)
        dq = reference_dequantize(qw).float()
        assert (dq[:, dead] == 0).all()


# --- fp16 scale rounding ----------------------------------------------------

@pytest.mark.parametrize("symmetric", [False, True])
def test_identity_hessian_equals_round_to_nearest(symmetric):
    """With H = I there is no error propagation, so GPTQ must equal RTN exactly.

    This only holds if GPTQ quantizes with the fp16-rounded scale that is stored.
    """
    torch.manual_seed(4)
    w = torch.randn(16, 256) * 0.3
    qw = G.gptq_quantize(w, torch.eye(256), symmetric=symmetric, device=CPU)
    rtn = quantize_groupwise(w, symmetric=symmetric)
    assert torch.equal(qw.qweight, rtn.qweight)
    assert torch.equal(qw.scales, rtn.scales)
    if symmetric:
        assert qw.qzeros is None
    else:
        assert torch.equal(qw.qzeros, rtn.qzeros)


# --- Naive reference --------------------------------------------------------

def _naive_gptq(w, h, damp, gs):
    """Column-by-column GPTQ without blocking, fp64, asymmetric."""
    w = w.double().clone()
    h = h.double().clone()
    n = w.shape[1]
    h += damp * torch.diag(h).mean() * torch.eye(n, dtype=torch.float64)
    u = torch.linalg.cholesky(torch.linalg.inv(h), upper=True)
    out = torch.zeros_like(w)
    for j in range(n):
        if j % gs == 0:
            g = w[:, j : j + gs]
            wmin, wmax = g.amin(1).clamp(max=0), g.amax(1).clamp(min=0)
            scale = ((wmax - wmin) / 15).half().double()
            zero = torch.round(-wmin / scale).clamp(0, 15)
        q = (torch.round(w[:, j] / scale) + zero).clamp(0, 15)
        dq = (q - zero) * scale
        out[:, j] = dq
        err = (w[:, j] - dq) / u[j, j]
        w[:, j:] -= err[:, None] * u[j, j:][None, :]
    return out


@pytest.mark.parametrize("block_size", [128, 256])
def test_matches_naive_unblocked_gptq(block_size):
    torch.manual_seed(5)
    w = torch.randn(16, 256) * 0.2
    x = torch.randn(512, 256) * (torch.rand(256) + 0.2)
    h = _hessian(x).float()
    qw = G.gptq_quantize(w, h, block_size=block_size, device=CPU)
    got = reference_dequantize(qw).double()
    want = _naive_gptq(w, h, 0.01, 128)
    # fp32 vs fp64 may flip a rare rounding tie; allow a tiny fraction of columns.
    mismatch = ((got - want).abs() > 1e-2).double().mean().item()
    assert mismatch < 0.01


# --- Sequential ordering, calibration length, tied embeddings ---------------

transformers = pytest.importorskip("transformers")


def _tiny(tie):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=27, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        tie_word_embeddings=tie, eos_token_id=3, bos_token_id=1,
    )
    return Qwen2ForCausalLM(cfg).eval()


def test_sequential_hessians_see_quantized_predecessors_and_full_calibration(monkeypatch):
    """Each linear's Hessian equals 2/N X^T X of its input in the final quantized
    model: layers (and groups inside a layer) are quantized in true-sequential order,
    and N is n_samples * seq_len."""
    model = _tiny(False)
    calib = torch.randint(0, 27, (3, 10))
    seen: dict[str, tuple[torch.Tensor, int]] = {}
    real = G.gptq_quantize

    def spy(weight, hessian, **kw):
        seen[kw["layer_name"]] = (hessian.clone(), 0)
        return real(weight, hessian, **kw)

    monkeypatch.setattr(G, "gptq_quantize", spy)
    order: list[str] = []
    monkeypatch.setattr(
        G, "_group_linears",
        (lambda orig: lambda layer: (lambda gs: (order.extend(n for g in gs for n, _ in g), gs)[1])(orig(layer)))(G._group_linears),
    )
    G.quantize_model_sequential(model, calib, device=CPU)

    # True-sequential order inside layer 0.
    names0 = [n.split(".")[-1] for n in order[:7]]
    assert names0 == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    inputs: dict[str, list[torch.Tensor]] = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        for n, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear):
                key = f"model.layers.{li}.{n}"
                inputs[key] = []
                hooks.append(m.register_forward_hook(
                    lambda _m, i, _o, k=key: inputs[k].append(i[0].detach().reshape(-1, i[0].shape[-1]))
                ))
    with torch.no_grad():
        for i in range(calib.shape[0]):
            model(input_ids=calib[i : i + 1])
    for h in hooks:
        h.remove()
    assert set(inputs) == set(seen)
    for k, xs in inputs.items():
        x = torch.cat(xs)
        assert x.shape[0] == 3 * 10
        torch.testing.assert_close(seen[k][0].double(), _hessian(x), rtol=1e-3, atol=1e-4)


def test_tied_embedding_stays_fp16_and_lm_head_is_not_stored(monkeypatch, tmp_path):
    from quf.convert import gptq as cg

    model = _tiny(True)
    d = tmp_path / "tied"
    model.save_pretrained(d)
    res = cg.run_native_gptq(str(d), [[1, 2, 3, 4, 5, 6, 7, 8]] * 2, device="cpu")
    assert "lm_head.weight" not in res.others
    emb = res.others["model.embed_tokens.weight"]
    assert emb.dtype == np.float16 and emb.shape == (27, 128)
    assert not any(k.startswith(("lm_head", "model.embed_tokens")) for k in res.linears)
