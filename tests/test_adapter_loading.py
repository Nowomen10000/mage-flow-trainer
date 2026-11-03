"""Weights-only continuation must preserve the exported function and frozen base."""
import copy

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_moss_image import tiny, inputs
from trainer.data.dataset import DatasetConfig
from trainer.modeling.export import export_checkpoint
from trainer.training.adapter_loading import load_adapter_weights, validate_weights_source, validate_new_phase_output
from trainer.training.config import Config, TrainConfig, OptimizerConfig, ScheduleConfig
from trainer.training.optim import build_optimizer, build_scheduler
from trainer.training.params import AdapterConfig, apply_adapter


def setup_pair(tmp_path, kind='lycoris_lora', algo='locon'):
    torch.manual_seed(8)
    model = tiny()
    target = copy.deepcopy(model)
    cfg = Config(dataset=DatasetConfig(path=str(tmp_path)))
    cfg.train.model_family = 'moss_image'
    cfg.adapter = AdapterConfig(kind=kind, lycoris_algo=algo, rank=2, alpha=1.7,
                                moss_train_text_refiner=False)
    apply_adapter(model, cfg.adapter)
    apply_adapter(target, cfg.adapter)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * .04)
    return model, target, cfg


@pytest.mark.parametrize('kind,algo', [('lora', 'locon'), ('lycoris_lora', 'locon'),
                                      ('lycoris_lora', 'lokr'), ('lycoris_lora', 'dora'), ('lokr', 'locon')])
@pytest.mark.parametrize('layout', ['auto', 'separated'])
def test_export_continue_preserves_outputs_gradients_and_base(tmp_path, kind, algo, layout):
    model, target, cfg = setup_pair(tmp_path, kind, algo)
    frozen = {n: p.detach().clone() for n, p in target.named_parameters() if not p.requires_grad}
    cfg.train.moss_qkv_format = layout
    export_checkpoint(model, tmp_path, 'source', cfg, torch.float32, 42)
    source = tmp_path / 'source.safetensors'
    identity = load_adapter_weights(target, source)
    assert identity['source_step'] == '42'
    assert identity['optimizer_restored'] is False
    for n, p in target.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
    xs, t, text, mask = inputs()
    model.eval(); target.eval()
    a, b = model(xs, t, (text, mask))[0], target(xs, t, (text, mask))[0]
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    sum(x.square().mean() for x in a).backward()
    sum(x.square().mean() for x in b).backward()
    for (n, p), (m, q) in zip(model.named_parameters(), target.named_parameters()):
        assert n == m
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    # Fresh moments and a new schedule regardless of the source step.
    optim = build_optimizer([{'params': [p for p in target.parameters() if p.requires_grad]}],
                            OptimizerConfig(lr=3e-4))
    build_scheduler(optim, ScheduleConfig(kind='constant'), 5)
    assert not optim.state
    assert optim.param_groups[0]['lr'] == 3e-4
    optim.step()


@pytest.mark.parametrize('damage', ['alpha', 'rank', 'missing', 'nan', 'cross_block'])
def test_invalid_file_fails_before_any_parameters_change(tmp_path, damage):
    _, target, cfg = setup_pair(tmp_path)
    export_checkpoint(target, tmp_path, 'source', cfg, torch.float32, 2)
    state = load_file(tmp_path / 'source.safetensors')
    prefix = 'diffusion_model.layers.0.proj.qkv'
    if damage == 'alpha':
        state[prefix + '.alpha'] *= 2
    elif damage == 'rank':
        state[prefix + '.lora_down.weight'] = state[prefix + '.lora_down.weight'][:1]
    elif damage == 'missing':
        del state[prefix + '.alpha']
    elif damage == 'nan':
        state[prefix + '.lora_down.weight'][0, 0] = float('nan')
    else:
        state[prefix + '.lora_up.weight'][0, -1] = 1
    save_file(state, tmp_path / 'bad.safetensors')
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with pytest.raises(ValueError):
        load_adapter_weights(target, tmp_path / 'bad.safetensors')
    for k, v in target.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_flattened_names_and_config_validation(tmp_path):
    model, target, cfg = setup_pair(tmp_path)
    export_checkpoint(model, tmp_path, 'source', cfg, torch.float32, 2)
    state = load_file(tmp_path / 'source.safetensors')
    flat = {}
    for k, v in state.items():
        split = k.index('.lora_') if '.lora_' in k else k.index('.alpha')
        flat['lora_unet_' + k[:split].removeprefix('diffusion_model.').replace('.', '_') + k[split:]] = v
    save_file(flat, tmp_path / 'flat.safetensors')
    load_adapter_weights(target, tmp_path / 'flat.safetensors')
    assert TrainConfig().resume_mode == 'exact'
    with pytest.raises(ValueError, match='resume_mode'):
        TrainConfig(resume_mode='bogus')
    with pytest.raises(ValueError, match='resume_from'):
        TrainConfig(resume_mode='weights')
    cfg.train.resume_mode = 'weights'
    cfg.train.resume_from = str(tmp_path / 'flat.safetensors')
    cfg.train.output_dir = str(tmp_path)
    cfg.train.run_name = 'new-phase'
    validate_weights_source(cfg)
    validate_new_phase_output(cfg)
    (tmp_path / 'new-phase').mkdir()
    (tmp_path / 'new-phase' / 'old.toml').touch()
    with pytest.raises(ValueError, match='new run_name'):
        validate_new_phase_output(cfg)


def test_gui_weights_only_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from trainer.gui.app import TrainingGUI
    from trainer.gui import bridge
    from trainer.training.config import load_config
    app = QApplication.instance() or QApplication([])
    model, _, cfg = setup_pair(tmp_path)
    export_checkpoint(model, tmp_path, 'source', cfg, torch.float32, 1)
    gui = TrainingGUI()
    try:
        gui._apply(bridge.defaults() | {
            'dataset.path': str(tmp_path), 'train.output_dir': str(tmp_path),
            'train.run_name': 'phase2', 'train.model_family': 'moss_image',
            'train.resume_from': str(tmp_path / 'source.safetensors'),
            'train.resume_mode': 'weights', 'adapter.kind': 'lycoris_lora',
            'adapter.rank': 2, 'adapter.alpha': 1.7, 'adapter.moss_train_text_refiner': False,
        })
        assert gui.editors['train.resume_mode'].get() == 'weights'
        assert gui.editors['train.resume_mode'].widget.isEnabled()
        path = tmp_path / 'phase2.toml'
        bridge.write_toml(path, gui.collect())
        loaded = load_config(path)
        assert loaded.train.resume_mode == 'weights'
        assert loaded.train.resume_from == str(tmp_path / 'source.safetensors')
        assert not loaded.train.save_optimizer_state
    finally:
        gui.close()
        app.processEvents()
