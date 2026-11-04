"""Exercise Trainer.save itself, including tiered batch metadata and real exports."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_moss_image import tiny
from trainer.data.dataset import DatasetConfig
from trainer.tools.inspect_checkpoint import read_metadata
from trainer.training.config import Config
from trainer.training.params import AdapterConfig, apply_adapter
from trainer.training.train import Trainer


def make_trainer(folder, batch_size, world=2, accumulation=1, kind='lycoris_lora', native=True):
    cfg = Config(dataset=DatasetConfig(path='/test/images', resolutions=[512, 768, 1024]))
    cfg.train.model_family = 'moss_image'
    cfg.train.run_name = 'multires'
    cfg.train.batch_size = batch_size
    cfg.train.gradient_accumulation_steps = accumulation
    cfg.train.save_native = native
    cfg.adapter = AdapterConfig(kind=kind, rank=2, alpha=2)
    model = tiny()
    model.configure_execution(False)
    if cfg.is_lora:
        apply_adapter(model, cfg.adapter)
    trainer = Trainer.__new__(Trainer)
    from trainer.training.timing import TrainingTiming
    trainer.timing = TrainingTiming()
    trainer.cfg = cfg
    trainer.accelerator = SimpleNamespace(
        is_main_process=True, num_processes=world, wait_for_everyone=lambda: None,
        unwrap_model=lambda model, **kw: model,
    )
    trainer.transformer = model
    trainer.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    trainer.out_dir = Path(folder)
    trainer.dtype = torch.float32
    trainer.global_step = trainer.successful_updates = 5
    trainer.skipped_updates = 0
    trainer.source_checkpoint = None
    trainer.rti_interface = None
    return trainer


@pytest.mark.parametrize('batch_size,world,accumulation,expected', [
    (8, 2, 1, 16),
    ({512: 24, 768: 12, 1024: 8}, 1, 1, {'512': 24, '768': 12, '1024': 8}),
    ({512: 24, 768: 12, 1024: 8}, 2, 1, {'512': 48, '768': 24, '1024': 16}),
    ({512: 24, 768: 12, 1024: 8}, 2, 3, {'512': 144, '768': 72, '1024': 48}),
])
@pytest.mark.parametrize('kind,native', [('lycoris_lora', True), ('none', True), ('none', False)])
def test_save_preserves_batch_metadata(tmp_path, batch_size, world, accumulation, expected, kind, native):
    trainer = make_trainer(tmp_path, batch_size, world, accumulation, kind, native)
    before = dict(batch_size) if isinstance(batch_size, dict) else batch_size
    for tag in ('epoch001', 'step000005', 'final'):
        trainer.save(tag)
        stem = 'multires' if tag == 'final' else 'multires-' + tag
        path = (tmp_path / (stem + '.safetensors') if native else
                tmp_path / stem / 'transformer/diffusion_pytorch_model.safetensors')
        metadata = read_metadata(path)
        state = metadata['training_state']
        assert state['effective_batch_size'] == expected
        assert state['world_size'] == world
        assert state['global_step'] == 5
        assert state['save_tag'] == tag
        assert state['training_started_at'] == trainer.timing.training_started_at
        assert state['checkpoint_saved_at'].endswith('+00:00')
        assert state['training_elapsed_seconds'] >= state['session_elapsed_seconds'] >= 0
        stored_batch = metadata['training_config']['train']['batch_size']
        assert stored_batch == ({str(k): v for k, v in before.items()} if isinstance(before, dict) else before)
    assert trainer.cfg.train.batch_size == before


def test_timing_round_trip_through_export_and_resume_state(tmp_path):
    import json
    from trainer.training.timing import TrainingTiming
    trainer = make_trainer(tmp_path, 6)
    trainer.cfg.train.save_optimizer_state = True
    trainer.flow_rng = SimpleNamespace(version=1, load=lambda *a, **k: None)
    trainer.save_full_state = lambda tag: (tmp_path / 'multires-state' / 'accelerator').mkdir(parents=True)
    trainer.timing.previous_seconds = 120
    trainer.save('final')
    saved = json.loads((tmp_path / 'multires-state/state.json').read_text())['training_timing']
    exported = read_metadata(tmp_path / 'multires.safetensors')['training_state']
    assert all(exported[key] == value for key, value in saved.items())
    trainer.timing = TrainingTiming()
    trainer.accelerator.load_state = lambda path: None
    trainer.accelerator.print = lambda *args: None
    trainer.steps_per_epoch = 10
    trainer._resume(str(tmp_path / 'multires.safetensors'))
    assert trainer.timing.previous_seconds == saved['training_elapsed_seconds']
    assert trainer.timing.snapshot()['training_elapsed_seconds'] >= 120
    assert trainer.timing.training_started_at == saved['training_started_at']


def test_final_save_notification_only_after_successful_export(tmp_path, monkeypatch, capsys):
    import json
    from trainer.modeling import export
    trainer = make_trainer(tmp_path, 6)
    trainer.save('epoch001')
    assert 'BLUFLOW_FINAL_SAVE' not in capsys.readouterr().out
    trainer.save('final')
    line = next(s for s in capsys.readouterr().out.splitlines() if s.startswith('BLUFLOW_FINAL_SAVE '))
    assert json.loads(line.split(' ', 1)[1]) == {'output_dir': str(tmp_path.resolve()), 'step': 5}
    def fail(*args, **kwargs):
        raise OSError('export failed')
    monkeypatch.setattr(export, 'export_checkpoint', fail)
    with pytest.raises(OSError, match='export failed'):
        trainer.save('final')
    assert 'BLUFLOW_FINAL_SAVE' not in capsys.readouterr().out
