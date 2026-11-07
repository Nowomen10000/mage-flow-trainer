import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets

from trainer.gui import bridge
from trainer.gui.fields import ResolutionStagesEditor
from trainer.gui.metrics import LiveMetricsWidget
from test_resolution_schedule import config_text
from test_gui_dataset_experiments import gui


def test_bridge_staged_round_trip_omits_epochs(tmp_path):
    import tomllib
    from trainer.training.config import load_config
    flat = {**bridge.defaults(), **bridge.flatten(tomllib.loads(config_text()))}
    flat['train.epochs'] = 100
    flat['train.save_every_epochs'] = 2
    text = bridge.dump_toml(flat)
    assert 'epochs' not in text
    p = tmp_path/'staged.toml'
    p.write_text(text)
    cfg = load_config(p)
    assert cfg.resolution_schedule.budgets(cfg.train.max_steps) == [112,112,336]
    assert bridge.validate(flat)[0]
    flat['resolution_schedule.enabled'] = False
    assert 'epochs = 100' in bridge.dump_toml(flat)


def test_stage_editor_budget_and_batch_preview():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    editor = ResolutionStagesEditor()
    stages = [dict(resolution=r,share=s) for r,s in ((512,.2),(768,.2),(1024,.6))]
    editor.set(stages)
    assert editor.get() == stages
    editor.preview(560,{512:24,768:12,1024:7},2,1)
    assert [editor.table.item(r,3).text() for r in range(3)] == ['112','112','336']
    assert editor.table.item(0,4).text() == '48'
    editor.widget.deleteLater()
    app.processEvents()


def test_stage_metrics_reach_graph_and_header():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = LiveMetricsWidget()
    widget.parse_and_update('stage2 step 113/560 loss 0.3 lr 1e-4 2s/it peak_alloc 9.0GiB')
    widget.parse_and_update('BLUFLOW_STAGE ' + json.dumps(dict(stage=2,resolution=768,completed=1,total=112,images=24,effective_batch=24)))
    widget._flush()
    assert widget.latest['stage'] == 2
    assert 'epoch' not in widget.latest
    assert widget.latest['step'] == 113
    assert '768px' in widget.header.text()
    widget.deleteLater()
    app.processEvents()


def test_gui_enabling_stages_hides_epochs_and_seeds_valid_plan(gui):
    gui._apply(bridge.defaults() | {
        'dataset.path': '/images', 'dataset.source': 'latents', 'train.model_family': 'moss_image',
        'adapter.kind': 'lora', 'train.batch_size': 2,
        'dataset.resolutions': [256, 384, 512, 640, 768, 1024],
    })
    gui.editors['resolution_schedule.enabled'].set(True)
    gui._on_edit()
    flat = gui.collect()
    assert flat['train.batch_size'] == {r: 2 for r in [256, 384, 512, 640, 768, 1024]}
    assert flat['train.pack_resolutions']
    assert flat['train.max_steps'] == 560
    assert abs(sum(s['share'] for s in flat['resolution_schedule.stages']) - 1) < 1e-9
    assert all(w.isHidden() for w in gui.rows['train.epochs'])
    assert not gui.editors['dataset.tier_collapse'].widget.isEnabled()
    ok, error = bridge.validate(flat)
    assert ok, error
    gui.editors['resolution_schedule.enabled'].set(False)
    gui._on_edit()
    assert not any(w.isHidden() for w in gui.rows['train.epochs'])
