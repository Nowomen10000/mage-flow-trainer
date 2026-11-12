import json
import os
from types import SimpleNamespace as NS

import pytest
import torch

from trainer.training.block_lr import BlockLRConfig
from trainer.training.config import Config, load_config, validate_model_options
from trainer.training.params import AdapterConfig, ComponentLRs, apply_adapter, build_adapter_param_groups
from trainer.training.optim import build_optimizer, build_scheduler
from trainer.training.config import OptimizerConfig, ScheduleConfig
from test_moss_image import tiny, inputs


def plan(first=.75, last=1.5):
    return BlockLRConfig(True, [dict(start=0, end=0, lr_mul=first), dict(start=1, end=1, lr_mul=last)])


@pytest.mark.parametrize('ranges', [
    {}, [dict(start=-1, end=2, lr_mul=1)], [dict(start=0, end=42, lr_mul=1)],
    [dict(start=2, end=1, lr_mul=1)], [dict(start=True, end=2, lr_mul=1)],
    [dict(start=0., end=2, lr_mul=1)], [dict(start=0, end=2, lr_mul=-1)],
    [dict(start=0, end=2, lr_mul=float('nan'))], [dict(start=0, end=2, lr_mul=float('inf'))],
    [dict(start=0, end=2, lr_mul=True)], [dict(start=0, end=2)],
    [dict(start=0, end=2, lr_mul=1, typo=1)],
    [dict(start=0, end=2, lr_mul=1), dict(start=2, end=3, lr_mul=1)],
])
def test_invalid_ranges_fail_early(ranges):
    with pytest.raises(ValueError, match='block_lr'):
        BlockLRConfig(True, ranges)


def test_disabled_defaults_gaps_and_equivalent_resume():
    assert BlockLRConfig().expanded() == [1.] * 42
    p = plan()
    assert p.expanded() == [.75, 1.5] + [1.] * 40
    p.check_resume(p.expanded())
    for old in (None, [1.] * 42):
        with pytest.raises(ValueError, match='weights-only'):
            p.check_resume(old)
    p.enabled = False
    assert p.expanded() == [1.] * 42
    p.check_resume(None)
    BlockLRConfig(True, []).check_resume(None)


@pytest.mark.parametrize('kind', ['lora', 'lycoris_lora'])
def test_real_adapters_receive_block_and_component_lr_and_updates(kind):
    torch.manual_seed(123)
    model = tiny()
    model.configure_execution(False)
    apply_adapter(model, AdapterConfig(kind=kind, rank=2, alpha=2, moss_train_text_refiner=True))
    report = build_adapter_param_groups(model, ComponentLRs(mlp=.02), .01, block_lr=plan())
    # The ordinary optimizer path strips bookkeeping before SDNQ's strict schema.
    came = build_optimizer(report.groups, OptimizerConfig(kind='came', lr=.01))
    assert all('block_lr_mul' not in g and 'component' not in g for g in came.param_groups)
    rates = {id(p): g['lr'] for g in report.groups for p in g['params']}
    assert len(rates) == sum(p.requires_grad for p in model.parameters())
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert id(p) not in rates
            continue
        base = .02 if '.ffn_' in name else .01
        mul = .75 if name.startswith('layers.0.') else 1.5 if name.startswith('layers.1.') else 1.
        assert rates[id(p)] == pytest.approx(base * mul), name
    optimizer = torch.optim.SGD(report.groups)
    xs, t, text, mask = inputs()
    loss = sum(y.square().mean() for y in model(xs, t, (text, mask))[0])
    loss.backward()
    before = {id(p): p.detach().clone() for p in model.parameters() if p.requires_grad}
    gradients = {id(p): p.grad.clone() for p in model.parameters() if p.requires_grad}
    assert sum(g.abs().sum() for g in gradients.values()) > 0
    optimizer.step()
    for p in model.parameters():
        if p.requires_grad:
            torch.testing.assert_close(p, before[id(p)] - rates[id(p)] * gradients[id(p)])


@pytest.mark.parametrize('kind', ['lora', 'lycoris_lora'])
def test_freezing_and_disabled_group_order(kind):
    model = tiny()
    apply_adapter(model, AdapterConfig(kind=kind, rank=2, alpha=2, moss_train_text_refiner=False))
    baseline = build_adapter_param_groups(model, ComponentLRs(), .001)
    disabled = plan()
    disabled.enabled = False
    for cfg in (disabled, BlockLRConfig(True, [])):
        actual = build_adapter_param_groups(model, ComponentLRs(), .001, block_lr=cfg)
        assert [[id(p) for p in g['params']] for g in actual.groups] == [[id(p) for p in g['params']] for g in baseline.groups]
    report = build_adapter_param_groups(model, ComponentLRs(), .001, block_lr=plan(0, 1))
    assert all(not p.requires_grad for n, p in model.named_parameters() if n.startswith('layers.0.'))
    assert any(p.requires_grad for n, p in model.named_parameters() if n.startswith('layers.1.'))
    with pytest.raises(ValueError, match='nothing to train'):
        build_adapter_param_groups(model, ComponentLRs(), .001, block_lr=plan(0, 0))


@pytest.mark.parametrize('schedule', ['constant', 'rex', 'rerex'])
@pytest.mark.parametrize('staged', [False, True])
def test_schedule_stage_and_curriculum_multiply_without_compounding(schedule, staged):
    from trainer.training.train import Trainer
    model = tiny()
    apply_adapter(model, AdapterConfig(kind='lora', rank=2, alpha=2))
    report = build_adapter_param_groups(model, ComponentLRs(), .01, block_lr=plan())
    opt = torch.optim.SGD(report.groups)
    sched = build_scheduler(opt, ScheduleConfig(kind=schedule, warmup_steps=2), 8)
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = NS(resolution_schedule=NS(enabled=staged, stages=[NS(lr_mul=.8), NS(lr_mul=1.)]))
    trainer.optimizer, trainer.scheduler = opt, sched
    bases = [g['lr'] for g in report.groups]
    for step in range(8):
        mul = .8 if step < 4 else 1.
        trainer.stage_index = int(step >= 4)
        trainer.phase = NS(lr_mul=mul)
        scheduled = list(sched.get_last_lr())
        for _ in range(3):
            trainer._apply_lr_mul()
            assert [g['lr'] for g in opt.param_groups] == pytest.approx([v*mul for v in scheduled])
        if scheduled[0]:
            assert [g['lr']/opt.param_groups[0]['lr'] for g in opt.param_groups] == pytest.approx([v/bases[0] for v in bases])
        opt.step()
        sched.step()


@pytest.mark.parametrize('family,kind,algo', [('mage_flow', 'lora', 'locon'), ('moss_image', 'none', 'locon'), ('moss_image', 'lokr', 'locon'), ('moss_image', 'lycoris_lora', 'lokr')])
def test_unsupported_modes_rejected(family, kind, algo):
    from trainer.data.dataset import DatasetConfig
    cfg = Config(dataset=DatasetConfig(path='/images'))
    cfg.train.model_family = family
    cfg.adapter = AdapterConfig(kind=kind, lycoris_algo=algo)
    cfg.block_lr = plan()
    with pytest.raises(ValueError, match='block LR'):
        validate_model_options(cfg)


def test_metadata_recovery_and_full_resume_guard(tmp_path):
    from test_multires_checkpoint_save import make_trainer
    from trainer.tools.recover_config import recover_file
    trainer = make_trainer(tmp_path, 1, kind='lora')
    trainer.cfg.block_lr = plan()
    trainer.cfg.train.save_optimizer_state = True
    trainer.flow_rng = NS(version=1)
    trainer.save_full_state = lambda tag: (tmp_path/'multires-state'/'accelerator').mkdir(parents=True)
    trainer.save('final')
    state = json.loads((tmp_path/'multires-state'/'state.json').read_text())
    assert state['block_lr_multipliers'] == plan().expanded()
    target, notes = recover_file(tmp_path/'multires.safetensors')
    assert not notes
    assert load_config(target).block_lr == plan()
    trainer.cfg.block_lr = plan(.5)
    # Refuse before loading optimizer/scheduler state, even with the same group count.
    with pytest.raises(ValueError, match='block LR multipliers changed'):
        trainer._resume(str(tmp_path/'multires.safetensors'))


@pytest.mark.skipif(os.environ.get('BLUFLOW_GPU_TESTS') != '1', reason='GPU opt-in')
@pytest.mark.parametrize('kind', ['lora', 'lycoris_lora'])
def test_bf16_came_cuda_update_and_optimizer_resume(kind):
    import copy
    torch.manual_seed(42)
    model = tiny().to('cuda', torch.bfloat16)
    model.configure_execution(False)
    apply_adapter(model, AdapterConfig(kind=kind, rank=2, alpha=2, dtype='bfloat16',
                                      moss_train_text_refiner=False))
    groups = build_adapter_param_groups(model, ComponentLRs(), 1e-4, block_lr=plan())
    cfg = OptimizerConfig(kind='came', lr=1e-4)
    optimizer = build_optimizer(groups.groups, cfg)
    scheduler = build_scheduler(optimizer, ScheduleConfig(kind='rex', warmup_steps=1), 4)
    before = {n:p.clone() for n,p in model.named_parameters() if p.requires_grad}
    for _ in range(2):
        xs,t,text,mask = inputs()
        output = model([x.to('cuda',torch.bfloat16) for x in xs], t.cuda(),
                       (text.to('cuda',torch.bfloat16),mask.cuda()))[0]
        loss = sum(y.float().square().mean() for y in output)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    torch.cuda.synchronize()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n in before)
    expected = [g['lr'] for g in optimizer.param_groups]
    # Rebuild in the same deterministic order used during full-state resume.
    fresh = build_adapter_param_groups(model, ComponentLRs(), 1e-4, block_lr=plan())
    restored = build_optimizer(fresh.groups, cfg)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert [g['lr'] for g in restored.param_groups] == expected
