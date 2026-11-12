import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from trainer.gui import bridge
from trainer.training.config import load_config
from test_gui_dataset_experiments import gui


def test_experimental_block_table_roundtrip(gui, tmp_path):
    flat = bridge.defaults() | {'dataset.path':'/images', 'train.model_family':'moss_image', 'adapter.kind':'lora'}
    gui._apply(flat)
    editor = gui.editors['block_lr.ranges']
    assert not editor.widget.isEnabled()
    assert editor.get() == flat['block_lr.ranges']
    gui.editors['block_lr.enabled'].set(True)
    gui._on_edit()
    assert gui.tab_bar.isTabVisible(gui.experimental_tab)
    assert editor.widget.isEnabled()
    editor.table.cellWidget(0, 2).setValue(.75)
    editor.table.cellWidget(2, 2).setValue(1.5)
    config = gui.collect()
    ok, error = bridge.validate(config)
    assert ok, error
    path = tmp_path/'blocks.toml'
    bridge.write_toml(path, config)
    assert load_config(path).block_lr.expanded() == [.75]*25+[1.]*8+[1.5]*9
    gui._apply(bridge.defaults() | bridge.flatten(bridge.read_toml(path)))
    assert gui.collect()['block_lr.ranges'] == config['block_lr.ranges']
    assert gui.tab_bar.isTabVisible(gui.experimental_tab)
    gui.editors['block_lr.enabled'].set(False)
    gui._on_edit()
    bridge.write_toml(path, gui.collect())
    cfg = load_config(path)
    assert cfg.block_lr.expanded() == [1.]*42
    assert cfg.block_lr.ranges[0]['lr_mul'] == .75


def test_overlaps_do_not_silently_change_and_unsupported_can_disable(gui):
    gui._apply(bridge.defaults() | {'dataset.path':'/images', 'train.model_family':'moss_image', 'adapter.kind':'lora', 'block_lr.enabled':True})
    editor = gui.editors['block_lr.ranges']
    editor.table.cellWidget(1,0).setValue(24)
    ok, error = bridge.validate(gui.collect())
    assert not ok and 'overlap' in error
    editor.table.cellWidget(1,0).setValue(25)
    editor.table.selectRow(1)
    editor._remove()
    editor._add()
    assert editor.get()[-1] == dict(start=25,end=25,lr_mul=1.)
    gui.editors['adapter.kind'].set('none')
    gui._on_edit()
    assert gui.editors['block_lr.enabled'].widget.isEnabled()
    ok, error = bridge.validate(gui.collect())
    assert not ok and 'block LR' in error
