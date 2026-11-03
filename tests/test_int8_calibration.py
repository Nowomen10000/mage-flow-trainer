"""Calibration cache, routing, GUI/CLI integration and opt-in CUDA correctness."""
import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch

from trainer.training import int8_calibration as c
from trainer.training.quant import QuantConfig

TILE = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_warps=4, num_stages=2)


def identity():
    return dict(schema=c.SCHEMA, gpu='Mock GPU', uuid='GPU-test', torch='test', driver='test',
                shapes=[[256, 256, False]], tokens=[512, 4096], candidates=[TILE])


def profile():
    return dict(identity=identity(), entries=[dict(m=m, n=256, k=256, contiguous=False,
                tile=copy.deepcopy(TILE), ms=.1) for m in (512, 4096)])


def test_cache_atomic_roundtrip_and_invalidation(tmp_path):
    data = profile()
    path = c.profile_path(data['identity'], tmp_path)
    c.write_profile(path, data)
    assert c.read_profile(path, data['identity']) == data
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    for key in ('uuid', 'torch', 'driver'):
        changed = dict(data['identity'], **{key: 'changed'})
        assert c.profile_path(changed, tmp_path) != path
        assert c.read_profile(path, changed) is None
    path.write_text('{broken')
    assert c.read_profile(path, identity()) is None


@pytest.mark.parametrize('damage', ['tile', 'nan', 'layout', 'duplicate', 'empty'])
def test_reject_invalid_cache(tmp_path, damage):
    data = profile()
    if damage == 'tile':
        data['entries'][0]['tile']['BLOCK_SIZE_M'] = -1
    elif damage == 'nan':
        data['entries'][0]['ms'] = float('nan')
    elif damage == 'layout':
        data['entries'][0]['k'] = 128
    elif damage == 'duplicate':
        data['entries'].append(copy.deepcopy(data['entries'][0]))
    else:
        data['entries'] = []
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(data))
    assert c.read_profile(path, identity()) is None


def test_nearest_anchor_layout_device_and_override(monkeypatch):
    c.clear_active()
    data = profile()
    data['entries'][1]['tile'] = 'torch'
    c.activate(torch.device('cuda:1'), data)
    try:
        assert c.select_tile(torch.device('cuda:1'), 513, 256, 256, False) == TILE
        assert c.select_tile(torch.device('cuda:1'), 8000, 256, 256, False) == 'torch'
        assert c.select_tile(torch.device('cuda:0'), 513, 256, 256, False) is None
        assert c.select_tile(torch.device('cuda:1'), 513, 256, 256, True) is None
        assert c.select_tile(torch.device('cuda:1'), 513, 128, 256, False) is None
        monkeypatch.setenv('SDNQ_TRITON_MM_BLOCK_SIZE_M_LIST', '128')
        assert c.select_tile(torch.device('cuda:1'), 513, 256, 256, False) is None
    finally:
        c.clear_active()


def test_quant_options_and_projection_plan(monkeypatch):
    assert QuantConfig(preset='fast').matmul_tuning == 'auto'
    for mode in ('auto', 'cached', 'off'):
        assert QuantConfig(preset='balanced', matmul_tuning=mode).matmul_tuning == mode
    with pytest.raises(ValueError, match='matmul_tuning'):
        QuantConfig(matmul_tuning='bogus')
    from sdnq import kernel_wrappers
    monkeypatch.setattr(kernel_wrappers, 'use_contiguous_int8_mm', False)
    shapes = c.moss_shapes('/missing-local-model')
    assert (16384, 3072, False) in shapes
    assert (3072, 16384, False) in shapes
    assert (8192, 3072, False) in shapes
    monkeypatch.setattr(kernel_wrappers, 'use_contiguous_int8_mm', True)
    assert all(layout for _, _, layout in c.moss_shapes('/missing-local-model'))


def test_ddp_coordination_and_failure_broadcast(monkeypatch):
    import accelerate.utils
    calls = []
    monkeypatch.setattr(c, 'moss_shapes', lambda path: [(256, 256, False)])
    monkeypatch.setattr(c, 'prepare', lambda device, shapes, **kw: calls.append(('prepare', device)))
    monkeypatch.setattr(accelerate.utils, 'broadcast_object_list',
                        lambda error, from_process: calls.append(('broadcast', from_process, error[0])))
    cfg = SimpleNamespace(quant=QuantConfig(preset='fast'), train=SimpleNamespace(model_path='unused'))
    acc = SimpleNamespace(num_processes=2, process_index=1, device='cuda:1')
    c.prepare_training(cfg, acc)
    assert calls == [('broadcast', 0, None), ('prepare', 'cuda:1'), ('broadcast', 1, None)]
    def fail(*a, **kw):
        raise RuntimeError('test failure')
    monkeypatch.setattr(c, 'prepare', fail)
    with pytest.raises(RuntimeError, match='rank 1: test failure'):
        c.prepare_training(cfg, acc)
    assert calls[-1][0:2] == ('broadcast', 1)
    assert 'test failure' in calls[-1][2]


def test_gui_launch_and_control(monkeypatch, tmp_path):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from trainer.gui import bridge
    from trainer.gui.app import TrainingGUI
    from trainer.gui.preferences import Preferences
    from trainer.gui.process import calibration_launch
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(TrainingGUI, '_load_presets', lambda *a, **kw: None)
    gui = TrainingGUI(preferences=Preferences(tmp_path/'prefs.json'), config_dir=str(tmp_path))
    try:
        gui._apply(bridge.defaults() | {'dataset.path':'/images', 'train.model_family':'moss_image',
                    'adapter.kind':'lycoris_lora', 'adapter.dtype':'bfloat16', 'quant.preset':'fast'})
        assert gui.calibrate_btn.isEnabled()
        assert gui.editors['quant.matmul_tuning'].widget.isEnabled()
        gui.editors['quant.matmul_tuning'].set('cached')
        import toml
        assert toml.loads(bridge.dump_toml(gui.collect()))['quant']['matmul_tuning'] == 'cached'
        launches = []
        monkeypatch.setattr(gui, '_run', lambda launch, **kw: launches.append((launch, kw)))
        gui._calibrate_int8()
        launch, kwargs = launches[0]
        assert launch.is_training is False and kwargs['training'] is False
        assert '--all-devices' in launch.argv and '--force' in launch.argv
        gui.editors['quant.preset'].set('standard'); gui._on_edit()
        assert not gui.calibrate_btn.isEnabled()
        assert not gui.editors['quant.matmul_tuning'].widget.isEnabled()
        path = tmp_path/'path with spaces.toml'
        assert str(path) in calibration_launch(path).argv
    finally:
        gui.close(); gui.deleteLater(); app.processEvents()


GPU = pytest.mark.skipif(os.environ.get('BLUFLOW_GPU_TESTS') != '1', reason='GPU opt-in')


@GPU
def test_real_calibration_cached_routing_and_rng(tmp_path, monkeypatch):
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    before = torch.cuda.get_rng_state(device).clone()
    shapes = [(256, 256, False), (256, 512, True)]
    data = c.prepare(device, shapes, root=tmp_path, tokens=(32, 96))
    assert torch.equal(before, torch.cuda.get_rng_state(device))
    assert len(data['entries']) == 4
    monkeypatch.setattr(c, 'calibrate', lambda *a, **kw: pytest.fail('cache miss'))
    assert c.prepare(device, shapes, root=tmp_path, tokens=(32, 96)) == data
    from trainer.training import frozen_matmul as f
    monkeypatch.setattr(f, 'int_scaled_mm_func', lambda *a, **kw: pytest.fail('autotuner fallback'))
    for n, k, contiguous in shapes:
        m = 71  # unseen token count
        a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=device)
        b = torch.randint(-127, 128, (k,n) if contiguous else (n,k), dtype=torch.int8, device=device)
        if not contiguous: b = b.T
        sa, sb = torch.rand(m, device=device)*.01, torch.rand(n, device=device)*.01
        out = f._gemm((a,b,sa,sb,None,None,torch.bfloat16,(m,n)))
        reference = (a.cpu().long() @ b.cpu().long()).float() * sa.cpu()[:,None]
        reference = (reference * sb.cpu()[None,:]).bfloat16()
        assert torch.equal(out.cpu(), reference)
        fallback = c.launch_scaled_mm(a,b,sa,sb,torch.bfloat16,'torch')
        assert torch.equal(fallback.cpu(), reference)
    c.clear_active()


def test_prepare_cache_modes_and_explicit_overrides(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from sdnq import kernel_wrappers
    monkeypatch.setattr(c, '_ACTIVE', {})
    monkeypatch.setattr(kernel_wrappers, 'use_triton_scaled_mm', True)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(c, 'fingerprint', lambda *a: identity())
    calls = []
    monkeypatch.setattr(c, 'calibrate', lambda *a: calls.append('calibrate') or profile())
    device = torch.device('cuda:1')
    args = dict(root=tmp_path, tokens=[512,4096], log=lambda *a, **kw: None)
    shapes = [(256,256,False)]
    assert c._prepare(device, shapes, mode='cached', **args) is None
    assert calls == []
    assert c._prepare(device, shapes, **args) == profile()
    assert calls == ['calibrate'] and c.has_profile(device)
    assert c._prepare(device, shapes, mode='cached', **args) == profile()
    assert calls == ['calibrate']
    assert c._prepare(device, shapes, mode='off', **args) is None
    assert not c.has_profile(device)
    c._prepare(device, shapes, force=True, **args)
    assert calls == ['calibrate', 'calibrate']
    monkeypatch.setenv('SDNQ_TRITON_MM_BLOCK_SIZE_M_LIST', '64')
    assert c._prepare(device, shapes, **args) is None
    assert not c.has_profile(device)
    with pytest.raises(ValueError, match='matmul_tuning'):
        c._prepare(device, shapes, mode='bad', **args)
