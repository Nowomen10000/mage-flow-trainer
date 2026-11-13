"""Concept sliders on SDNQ-quantized MOSS bases (GPU opt-in: BLUFLOW_GPU_TESTS=1).

Slider steps only scale the adapter branch. These tests pin that down on the quantized paths the
trainer offers -- frozen SDNQ storage and the frozen INT8 matmul presets -- with gradient
checkpointing, eager and compiled: strength 0 reproduces the base model, the quantized storage is
bit-identical after slider steps and optimizer updates, and every adapter gets finite gradients.
"""
import os

import pytest
import torch

GPU = pytest.mark.skipif(os.environ.get('BLUFLOW_GPU_TESTS') != '1', reason='GPU opt-in')
ROLES = ('positive', 'neutral', 'negative', 'anchor')


def _base_storage(model):
    """Every frozen stored tensor (quantized payloads, scales, rotations, norms) by name."""
    from trainer.training.distributed import model_storage
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    return {name: value.detach().clone() for name, value in model_storage(model)
            if name not in trainable}


@GPU
@pytest.mark.parametrize('compile_mode', [None, 'default'])
@pytest.mark.parametrize('preset,kind', [
    ('standard', 'lora'), ('standard', 'lokr'), ('standard', 'lycoris_lora'),
    ('balanced', 'lora'), ('balanced', 'lycoris_lora'),
    ('fast', 'lora'), ('fast', 'lycoris_lora'),
])
def test_slider_steps_leave_quantized_base_untouched(preset, kind, compile_mode):
    from trainer.modeling.moss_image import MOSSImage, MOSSParams
    from trainer.training.params import AdapterConfig, apply_adapter
    from trainer.training.quant import QuantConfig, quantize_module, quantized_layer_report
    from trainer.training.slider import AdapterStrength, SliderConfig, slider_step

    torch.manual_seed(0)
    # Four layers: MOSS quantization protects the first and the last two blocks.
    m = MOSSImage(MOSSParams(dim=256, ffn_dim=512, num_heads=4, num_layers=4, text_dim=128,
                             latent_channels=8)).bfloat16()
    qcfg = QuantConfig(preset=preset)
    m = quantize_module(m, qcfg, torch.device('cuda'), torch.bfloat16, qcfg.use_quantized_matmul is True)
    assert quantized_layer_report(m)[0] > 0
    apply_adapter(m, AdapterConfig(kind=kind, rank=8, alpha=8, dtype='bfloat16', moss_train_text_refiner=False))
    m.cuda().bfloat16()
    m.configure_execution(True, compile_mode=compile_mode)
    m.train()

    x = torch.randn(2, 8, 1, 8, 8, device='cuda', dtype=torch.bfloat16)
    t = torch.tensor([.3, .7], device='cuda')
    mask = torch.ones(2, 9, device='cuda', dtype=torch.bool)
    contexts = {role: (torch.randn(1, 9, 128, device='cuda', dtype=torch.bfloat16).expand(2, -1, -1).contiguous(), mask)
                for role in ROLES}

    def forward(role):
        return m(x, t, contexts[role])[0]

    params = [p for p in m.parameters() if p.requires_grad]
    with torch.no_grad():
        base = forward('neutral').float()   # adapters start at zero delta: this is the base model
        for p in params:                     # give every adapter a real delta
            p.add_(torch.randn_like(p) * 0.05)
        adapted = forward('neutral').float()
    assert (adapted - base).abs().max() > 1e-2

    strength = AdapterStrength(m)
    strength.set(0.0)
    with torch.no_grad():
        off = forward('neutral').float()
    strength.restore()
    if compile_mode is None:
        assert torch.equal(off, base)
    else:
        torch.testing.assert_close(off, base, rtol=1e-2, atol=1e-2)

    before = _base_storage(m)
    assert before
    opt = torch.optim.AdamW(params, lr=1e-3)
    cfg = SliderConfig(enabled=True, target_class='a', positive_prompt='b', negative_prompt='c',
                       anchor_class='d')
    for _ in range(2):
        loss, stats = slider_step(cfg, strength, forward, early_backward=lambda term: term.backward())
        assert strength.value == -1.0
        loss.backward()
        strength.restore()
        assert torch.isfinite(loss) and set(stats) == {'slider_pos', 'slider_neg', 'slider_anchor'}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert any(p.grad.abs().sum() > 0 for p in params)
        opt.step()
        opt.zero_grad(set_to_none=True)

    after = _base_storage(m)
    assert before.keys() == after.keys()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert not changed, f'slider steps modified quantized base storage: {changed[:5]}'
