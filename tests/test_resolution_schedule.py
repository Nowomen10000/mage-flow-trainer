"""Staged budgets, sampling, config, and the real training loop on a tiny MOSS."""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from accelerate import Accelerator

from trainer.data.staged import StagedBatchSampler
from trainer.training.config import Config, load_config
from trainer.training.resolution_schedule import ResolutionSchedule


def plan():
    return ResolutionSchedule(enabled=True, stages=[dict(resolution=r, share=s) for r, s in
                                                    ((512, .2), (768, .2), (1024, .6))])


class Samples(torch.utils.data.Dataset):
    def __init__(self, root, count=7):
        self.entries = []
        for tier in (512, 768, 1024):
            for i in range(count):
                path = root / f'{tier}-{i}.bin'
                if not path.exists():
                    path.write_bytes(b'latent')
                self.entries.append(NS(resolution=tier, bucket=(tier, tier), texture_ok=False,
                                       latent_path=path, tags=str(i), nl=None))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        i, cycle, _, _ = index
        entry = self.entries[i]
        n = entry.resolution // 256
        return dict(latents=torch.full((8, 1, n, n + i % 2), (i % 7 + 1) / 10),
                    caption=f'{i}:{cycle}', bucket=entry.bucket, path=str(entry.latent_path))


def test_rounding_and_zero_stage():
    assert plan().budgets(560) == [112, 112, 336]
    assert plan().budgets(7) == [2, 1, 4]
    with pytest.raises(ValueError, match='at least one'):
        plan().budgets(2)


@pytest.mark.parametrize('rank', [0, 1])
def test_sampler_full_batches_and_resume(tmp_path, rank):
    ds = Samples(tmp_path)
    sampler = StagedBatchSampler(ds, {512: 4, 768: 3, 1024: 2}, plan(), 10,
                                world=2, rank=rank, accumulation=3)
    for stage, bs in enumerate((4, 3, 2)):
        sampler.configure(stage, 0, 4)
        uninterrupted = list(sampler)
        assert len(uninterrupted) == 12
        assert all(len(b) == bs for b in uninterrupted)
        assert all(ds.entries[i].resolution == (512, 768, 1024)[stage] for b in uninterrupted for i, *_ in b)
        sampler.configure(stage, 6, 2)
        assert list(sampler) == uninterrupted[6:]


def test_ranks_share_shuffle_without_double_sharding(tmp_path):
    ds = Samples(tmp_path, count=24)
    batches = []
    for rank in (0, 1):
        sampler = StagedBatchSampler(ds, {512: 4, 768: 3, 1024: 2}, plan(), 10, world=2, rank=rank)
        sampler.configure(0, 0, 3)
        batches.extend(list(sampler))
    indices = [i for batch in batches for i, *_ in batch]
    assert len(indices) == len(set(indices)) == 24


def config_text():
    return '''
[dataset]
path = "/test"
source = "latents"
resolutions = [512,768,1024]
[train]
model_family = "moss_image"
max_steps = 560
batch_size = {512=24,768=12,1024=7}
pack_resolutions = true
attention_backend = "torch_varlen"
[adapter]
kind = "lora"
[resolution_schedule]
enabled = true
stages = [{resolution=512,share=0.2},{resolution=768,share=0.2},{resolution=1024,share=0.6}]
'''


def test_config_and_epoch_rejection(tmp_path):
    p = tmp_path/'test.toml'
    p.write_text(config_text())
    c = load_config(p)
    assert c.train.save_every_epochs is None
    for key in ('epochs', 'save_every_epochs'):
        p.write_text(config_text().replace('[train]', f'[train]\n{key} = 10'))
        with pytest.raises(ValueError, match='step-driven'):
            load_config(p)


@pytest.mark.parametrize('change', [
    ('kind = "lora"', 'kind = "none"'),
    ('share=0.6', 'share=0.5'),
    ('model_family = "moss_image"', 'model_family = "mage_flow"'),
    ('pack_resolutions = true', 'pack_resolutions = false'),
])
def test_invalid_modes_rejected(tmp_path, change):
    p = tmp_path/'test.toml'
    p.write_text(config_text().replace(*change))
    with pytest.raises(ValueError):
        load_config(p)


def make_trainer(root, kind='lora', workers=0, gpu=False, accumulation=2):
    from trainer.training.train import Trainer
    from trainer.training.params import AdapterConfig, apply_adapter
    from test_moss_image import tiny
    torch.manual_seed(42)
    acc = Accelerator(cpu=not gpu, gradient_accumulation_steps=accumulation)
    from trainer.data.dataset import DatasetConfig
    cfg = Config(dataset=DatasetConfig(path=str(root)))
    cfg.resolution_schedule = plan()
    cfg.train.batch_size = {512: 3, 768: 2, 1024: 1}
    cfg.train.max_steps = 5
    cfg.train.num_workers = workers
    cfg.train.gradient_accumulation_steps = accumulation
    cfg.train.log_every = 1
    cfg.train.save_every_epochs = None
    cfg.train.skip_final_save = True
    cfg.optimizer.lr = 1e-3
    cfg.optimizer.weight_decay = 0
    cfg.schedule.kind = 'rex'
    cfg.adapter = AdapterConfig(kind=kind, rank=2, alpha=2, dtype='bfloat16' if gpu else 'float32')
    cfg.train.model_family = 'moss_image'
    model = tiny().to(device=acc.device, dtype=torch.bfloat16 if gpu else torch.float32)
    apply_adapter(model, cfg.adapter)
    model.configure_execution(True, attention_backend='torch_varlen' if gpu else 'sdpa')
    trainer = Trainer.__new__(Trainer)
    trainer.cfg, trainer.accelerator = cfg, acc
    trainer.dataset = Samples(root)
    trainer.sampler = StagedBatchSampler(trainer.dataset, cfg.train.batch_size, plan(), 5,
                                        world=acc.num_processes, rank=acc.process_index,
                                        accumulation=accumulation, seed=42)
    trainer.stage_index, trainer.stage_cursor = -1, 0
    trainer.stage_images = [0]*3
    trainer.stage_attempts = [0]*3
    trainer.total_steps = 5
    trainer.steps_per_epoch = 0
    trainer.groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]
    trainer._build_optimizer()
    trainer.transformer, trainer.optimizer, trainer.scheduler = acc.prepare(model, trainer.optimizer, trainer.scheduler)
    trainer.text_encoder = None
    trainer.self_flow_ema = trainer.perceptual = trainer.adversarial = trainer.bar = trainer._eta_anchor = None
    trainer.start_epoch = trainer.resume_skip_batches = trainer.global_step = 0
    trainer.successful_updates = trainer.skipped_updates = trainer.consecutive_nonfinite = 0
    trainer.progress_mode = 'off'
    trainer._report = trainer._set_phase = trainer._set_rti_budget = trainer._apply_lr_mul = lambda: None
    trainer.trace = []
    def step(batch):
        xs = [x.to(device=acc.device, dtype=next(model.parameters()).dtype) for x in batch['latents']]
        count = len(xs)
        times = torch.full((count,), .4, device=acc.device)
        text = torch.ones(count, 3, 16, device=acc.device, dtype=xs[0].dtype)
        mask = torch.ones(count, 3, device=acc.device, dtype=torch.bool)
        pred = trainer.transformer(xs, times, (text, mask))[0]
        trainer.trace.append((trainer.stage_index, tuple(batch['captions'])))
        return sum(y.float().square().mean() for y in pred) / count, None
    trainer._step = step
    trainer.saved = []
    trainer.save = lambda tag: trainer.saved.append((tag, copy.deepcopy(trainer._resolution_state())))
    trainer._log = lambda *args: None
    return trainer


@pytest.mark.parametrize('kind', ['lora', 'lycoris_lora'])
@pytest.mark.parametrize('workers', [0, 2])
def test_real_loop_transitions_and_prefetch(tmp_path, kind, workers, monkeypatch):
    monkeypatch.setenv('LYCORIS_KERNEL_BACKEND', 'torch')
    t = make_trainer(tmp_path, kind, workers)
    t.train()
    assert t.global_step == t.successful_updates == 5
    assert t.stage_attempts == [1, 1, 3]
    assert [s for s, _ in t.trace] == [0]*2 + [1]*2 + [2]*6
    assert [tag for tag, _ in t.saved] == ['step000001', 'step000002']
    assert t.stage_images == [6, 4, 6]
    assert t.scheduler.scheduler.last_epoch == 5
    t.accelerator.free_memory()


def test_skipped_update_refills_same_stage(tmp_path):
    t = make_trainer(tmp_path)
    step = t._step
    def one_bad_batch(batch):
        loss, hf = step(batch)
        if len(t.trace) == 1:
            loss = loss * float('nan')
        return loss, hf
    t._step = one_bad_batch
    t.train()
    assert t.global_step == t.successful_updates == 5
    assert t.skipped_updates == 1
    assert t.stage_attempts == [2, 1, 3]
    assert [s for s, _ in t.trace] == [0]*4 + [1]*2 + [2]*6
    assert t.scheduler.scheduler.last_epoch == 5
    t.accelerator.free_memory()


def enable_checkpoint_saving(t, root):
    from trainer.training.train import Trainer
    from trainer.training.timing import TrainingTiming
    from trainer.training.flow_rng import FlowRNG
    from types import MethodType
    t.cfg.train.run_name = 'staged-test'
    t.cfg.train.save_optimizer_state = True
    t.cfg.train.save_every_steps = 1
    t.out_dir = root / 'output'
    t.out_dir.mkdir(exist_ok=True)
    t.dtype = next(t.transformer.parameters()).dtype
    t.timing = TrainingTiming()
    t.source_checkpoint = t.adapter_initialization = t.rti_interface = None
    t.flow_rng = FlowRNG(42, t.accelerator.process_index, t.accelerator.num_processes, t.accelerator.device)
    t.save = MethodType(Trainer.save, t)


@pytest.mark.parametrize('checkpoint_step', [1, 2, 3])
@pytest.mark.parametrize('kind', ['lora', 'lycoris_lora'])
def test_full_state_resume_matches_uninterrupted(tmp_path, checkpoint_step, kind, monkeypatch):
    monkeypatch.setenv('LYCORIS_KERNEL_BACKEND', 'torch')
    first = make_trainer(tmp_path, kind=kind)
    enable_checkpoint_saving(first, tmp_path)
    first.train()
    final = {k: v.detach().clone() for k, v in first._unwrap(first.transformer).state_dict().items()}
    trace = list(first.trace)
    first.accelerator.free_memory()
    resumed = make_trainer(tmp_path, kind=kind)
    enable_checkpoint_saving(resumed, tmp_path)
    resumed._resume(str(tmp_path/'output'/f'staged-test-step{checkpoint_step:06d}-state'))
    resumed.train()
    assert resumed.trace == trace[checkpoint_step * 2:]
    assert resumed.stage_images == [6, 4, 6]
    assert resumed.successful_updates == 5
    for k, value in resumed._unwrap(resumed.transformer).state_dict().items():
        torch.testing.assert_close(value, final[k], rtol=0, atol=0)
    resumed.accelerator.free_memory()


def test_staged_fingerprint_detects_changed_plan(tmp_path):
    ds = Samples(tmp_path)
    def fingerprint(**kw):
        return StagedBatchSampler(ds, {512: 4, 768: 3, 1024: 2}, plan(), **kw).identity
    original = fingerprint(steps=10)
    assert original != fingerprint(steps=15)
    assert original != fingerprint(steps=10, world=2)
    assert original != fingerprint(steps=10, accumulation=2)


def test_collapsed_sources_remain_in_every_stage(tmp_path):
    from trainer.parity.test_multires import make_fixture
    from trainer.data.dataset import DatasetConfig, MageFlowDataset
    make_fixture(tmp_path, [(256,256), (1200,1200)], [512,768,1024])
    cfg = DatasetConfig(path=str(tmp_path), source='latents', resolutions=[512,768,1024], tier_collapse='dedup')
    normal = MageFlowDataset(cfg)
    staged = MageFlowDataset(replace(cfg, tier_collapse='repeat'))
    assert len(normal.entries) == 4
    assert len(staged.entries) == 6
    assert all(sum(e.resolution == r for e in staged.entries) == 2 for r in cfg.tiers)
    small = [e for e in staged.entries if 'img000' in str(e.latent_path)]
    assert len({e.latent_path for e in small}) == 1


@pytest.mark.parametrize('args,budgets', [
    (['--dry-run'], [1, 1, 1]),
    (['--no-save', '--max-steps', '10'], [2, 2, 6]),
])
def test_cli_staged_budgets_and_no_saves(tmp_path, monkeypatch, args, budgets):
    import sys
    import trainer.training.train as entry
    path = tmp_path / 'staged.toml'
    path.write_text(config_text())
    seen = []
    class FakeTrainer:
        def __init__(self, cfg, **kwargs):
            seen.append(cfg)
        def train(self):
            pass
    monkeypatch.setattr(entry, 'Trainer', FakeTrainer)
    monkeypatch.setattr(sys, 'argv', ['train', str(path), *args])
    entry.main()
    cfg, = seen
    assert cfg.resolution_schedule.budgets(cfg.train.max_steps) == budgets
    assert not cfg.resolution_schedule.save_stage_end
    assert cfg.train.save_every_steps is cfg.train.save_every_epochs is None
    assert cfg.train.skip_final_save


def test_cli_rejects_too_short_override_before_building(tmp_path, monkeypatch):
    import sys
    import trainer.training.train as entry
    path = tmp_path / 'staged.toml'
    path.write_text(config_text())
    monkeypatch.setattr(sys, 'argv', ['train', str(path), '--max-steps', '2'])
    with pytest.raises(ValueError, match='at least one'):
        entry.main()
