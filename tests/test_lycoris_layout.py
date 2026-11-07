"""Guard the pinned LoCon atomic backward against non-dense input layouts."""
import os
from types import SimpleNamespace

import pytest
import torch

from trainer.training.lycoris import algorithm_classes


@pytest.mark.parametrize('allocator,device,use_graph,expected', [
    ('cudaMallocAsync', 'cuda', True, False), ('native', 'cuda', True, True),
    ('native', 'cuda', False, False), ('cudaMallocAsync', 'xpu', True, True),
])
def test_allocator_safe_tuner_preserves_options(monkeypatch, allocator, device, use_graph, expected):
    from lycoris.kernels.tuner import measure
    from trainer.training.lycoris import install_allocator_safe_timing
    calls = []
    monkeypatch.setattr(measure, 'time_candidates', lambda *a, **kw: calls.append((a, kw)) or [1.0])
    monkeypatch.setattr(torch.cuda.memory, 'get_allocator_backend', lambda: allocator)
    install_allocator_safe_timing()
    wrapped = measure.time_candidates
    install_allocator_safe_timing()
    assert measure.time_candidates is wrapped
    errors = []
    dev = SimpleNamespace(kind=device)
    assert wrapped([], 2, dev, rounds=3, use_graph=use_graph, prune=1.5, errors=errors) == [1.0]
    assert calls == [(([], 2, dev), dict(rounds=3, use_graph=expected, prune=1.5, errors=errors))]


@pytest.mark.parametrize('layout', ['slice', 'broadcast', 'transpose', 'contiguous'])
def test_locon_normalizes_only_adapter_input(layout, monkeypatch):
    from lycoris.modules import LoConModule
    base = torch.nn.Linear(16, 16, bias=False)
    adapter = algorithm_classes()['locon']('test', base, lora_dim=8, alpha=8, bypass_mode=True)
    x = torch.randn(3, 32)[:, ::2]
    if layout == 'broadcast':
        x = torch.randn(1, 16).expand(3, 16)
    elif layout == 'transpose':
        x = torch.randn(16, 3).T
    elif layout == 'contiguous':
        x = x.contiguous()
    seen = []
    monkeypatch.setattr(LoConModule, 'bypass_forward_diff',
                        lambda self, value, scale=1: seen.append(value) or value * scale)
    out = adapter.bypass_forward_diff(x, scale=2)
    assert seen[0].is_contiguous()
    torch.testing.assert_close(out, x * 2)
    if layout == 'contiguous':
        assert seen[0] is x


@pytest.mark.skipif(os.environ.get('BLUFLOW_GPU_TESTS') != '1', reason='GPU opt-in')
@pytest.mark.parametrize('rank', [8, 16])
def test_locon_strided_atomic_backward(rank, monkeypatch):
    from lycoris.kernels.triton.lora import stream
    from lycoris.kernels.plans import tune
    monkeypatch.setenv('LYCORIS_KERNEL_BACKEND', 'triton')
    # Exercise the problematic candidate regardless of the local tuning cache.
    monkeypatch.setattr(stream, 'select', lambda *a, **kw: stream.ATOMIC)
    monkeypatch.setattr(tune, 'tuned', lambda name, key, shortlist, factory:
                        next(p for p in shortlist() if p.limiter != 'eager'))
    torch.manual_seed(123)
    torch.set_num_threads(2)
    base = torch.nn.Linear(256, 256, bias=False, device='cuda', dtype=torch.bfloat16)
    adapter = algorithm_classes()['locon']('test', base, lora_dim=rank, alpha=rank,
                                          bypass_mode=True).cuda().bfloat16()
    with torch.no_grad():
        adapter.lora_up.weight.normal_(std=.01)
    for fn in [adapter.bypass_forward_diff,
               torch.compile(adapter.bypass_forward_diff, fullgraph=True, dynamic=True)]:
        for length in [33, 65]:
            x = torch.randn(length, 512, device='cuda', dtype=torch.bfloat16)[:, ::2].detach().requires_grad_()
            up, down = adapter.lora_up.weight, adapter.lora_down.weight
            grad = torch.randn(length, 256, device='cuda', dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(torch.nn.functional.linear(x, down), up)
            ref = torch.autograd.grad(expected, (x, up, down), grad)
            actual = fn(x)
            got = torch.autograd.grad(actual, (x, up, down), grad)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=.002, rtol=.02)
            for a, b in zip(got, ref):
                assert torch.isfinite(a).all()
                # Atomic FP32 reductions and cuBLAS BF16 reductions differ
                # around cancellation; compare error against the tensor scale.
                error = (a.float() - b.float()).abs()
                assert error.norm() < .01 * b.float().norm()
                assert error.max() < .01 * b.float().abs().max() + 1e-5
