"""Numerical regressions for SDNQ's FP32 and weight-decay compensation."""
from importlib.metadata import version

import pytest
import torch

from trainer.training.config import OptimizerConfig
from trainer.training.optim import build_optimizer

pytestmark = pytest.mark.skipif(version('sdnq') != '0.2.9', reason='SDNQ 0.2.9 migration regression')


@pytest.mark.parametrize('quantized_buffers', [False, True])
@pytest.mark.parametrize('gradient', [0., 1.])
def test_fp32_kahan_matches_compensated_adamw_with_decay(quantized_buffers, gradient):
    param = torch.nn.Parameter(torch.ones(128, 128))
    optimizer = build_optimizer([{'params': [param]}], OptimizerConfig(
        kind='adamw8bit', lr=.01, weight_decay=.1, use_kahan=True,
        quantize_state=quantized_buffers))
    expected = 1.
    for step in range(20):
        lr = .01 / (1 + step)
        optimizer.param_groups[0]['lr'] = lr
        param.grad = torch.full_like(param, gradient)
        optimizer.step()
        expected = expected * (1 - lr * .1) - lr * gradient
    torch.testing.assert_close(param, torch.full_like(param, expected), rtol=0, atol=3e-6)


def test_fp32_small_updates_do_not_accumulate_twice():
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = build_optimizer([{'params': [param]}], OptimizerConfig(
        kind='adamw8bit', lr=1e-5, weight_decay=0., use_kahan=True))
    for _ in range(100):
        param.grad = torch.ones_like(param)
        optimizer.step()
    torch.testing.assert_close(param, torch.tensor([.999]), rtol=0, atol=1e-7)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('cautious', [False, True])
def test_compensated_value_includes_decay_after_normalization(dtype, cautious):
    from sdnq.optim import optimizer, utils

    from trainer.training.sdnq_compat import fix_kahan_weight_decay
    assert fix_kahan_weight_decay()
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(32, 32, dtype=dtype))
    residual = torch.zeros_like(p)
    grad = torch.randn_like(p).float()
    update = torch.randn_like(grad) * 3
    expected_update = utils.apply_norm_to_update_(update.clone(), p.float(), 'clip', (1., 1e-3, 1e-3))
    if cautious:
        mask = (expected_update * grad > 0).float()
        expected_update *= mask / mask.mean().clamp_min(1e-3)
    target = p.detach().float() * .99 - .1 * expected_update
    optimizer.update_param_(p, p.detach().float().clone(), grad, update, residual,
                            learning_rate=.1, weight_decay=.1, clips=(1., 1e-3, 1e-3),
                            use_cautious=cautious, use_stochastic_rounding=False,
                            use_stochastic_buffers=False)
    # Residual uses parameter precision, so allow its own BF16 rounding error.
    torch.testing.assert_close(p.float() + residual.float(), target, rtol=0,
                               atol=4e-5 if dtype == torch.bfloat16 else 4e-7)


def test_legacy_quantized_optimizer_buffers_resume_without_requantization():
    from sdnq.training import SDNQTensor
    param = torch.nn.Parameter(torch.ones(128, 128, dtype=torch.bfloat16))
    cfg = OptimizerConfig(kind='adamw8bit', lr=1e-4, quantize_state=True, use_kahan=True)
    optimizer = build_optimizer([{'params': [param]}], cfg)
    param.grad = torch.full_like(param, .1)
    optimizer.step()
    state = optimizer.state_dict()
    originals = {}
    for key, value in state['state'][0].items():
        if isinstance(value, SDNQTensor):
            originals[key] = (value.weight.clone(), value.dequantize(torch.float32).clone())
            for field in ('scale_dtype', 'zero_point_dtype', 'use_codebook_scale',
                          'quantized_scale_shape', 'quantized_zero_point_shape'):
                delattr(value.sdnq_dequantizer, field)
    for group in state['param_groups']:
        for key in ('quantized_buffers_scale_dtype', 'quantized_buffers_zero_point_dtype',
                    'quantized_buffers_use_codebook_scale'):
            group.pop(key)
    restored = build_optimizer([{'params': [param]}], cfg)
    restored.load_state_dict(state)
    assert originals
    for key, (codes, dense) in originals.items():
        value = restored.state[param][key]
        assert torch.equal(value.weight, codes)
        assert torch.equal(value.dequantize(torch.float32), dense)
    param.grad = torch.full_like(param, .1)
    restored.step()
    assert torch.isfinite(param).all()


def test_restored_kahan_group_enables_decay_fix(monkeypatch):
    from sdnq.optim import optimizer, utils

    p = torch.nn.Parameter(torch.ones(16))
    source = build_optimizer([{'params': [p]}], OptimizerConfig(
        kind='adamw8bit', lr=.1, weight_decay=.1, use_kahan=True))
    p.grad = torch.zeros_like(p)
    source.step()
    state = source.state_dict()
    # Simulate a fresh process whose configuration did not request Kahan.
    monkeypatch.setattr(optimizer, 'update_param_', utils.update_param_)
    monkeypatch.setattr(optimizer, 'update_param_compiled_', utils.update_param_compiled_)
    restored = build_optimizer([{'params': [p]}], OptimizerConfig(
        kind='adamw8bit', use_kahan=False))
    restored.load_state_dict(state)
    for _ in range(9):
        p.grad = torch.zeros_like(p)
        restored.step()
    torch.testing.assert_close(p, torch.full_like(p, .99 ** 10), rtol=0, atol=3e-7)
