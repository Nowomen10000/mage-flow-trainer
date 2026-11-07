import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
import toml
from PySide6 import QtWidgets
from trainer.gui import bridge
from trainer.gui.app import TrainingGUI
from trainer.gui.preferences import Preferences
from trainer.training.config import load_config
from trainer.training.quant import QUANT_PRESETS


@pytest.mark.parametrize('preset',['standard','balanced','fast'])
@pytest.mark.parametrize('kind',['lycoris_lora','lora'])
def test_preset_controls_save_and_reload(monkeypatch,tmp_path,preset,kind):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(TrainingGUI,'_load_presets',lambda self,**kw:None)
    gui=TrainingGUI(preferences=Preferences(tmp_path/"gui.json"), config_dir=str(tmp_path))
    try:
        gui._apply(bridge.defaults() | {'dataset.path':'/images','train.model_family':'moss_image',
            'adapter.kind':kind,'adapter.dtype':'bfloat16'})
        gui.editors['quant.preset'].set(preset);gui._on_edit()
        flat=gui.collect()
        for key,value in QUANT_PRESETS[preset].items():
            assert flat[f'quant.{key}']==value
            assert not gui.editors[f'quant.{key}'].widget.isEnabled()
        path=tmp_path/'preset.toml';path.write_text(toml.dumps(bridge.nest(bridge.prune_defaults(flat))))
        cfg=load_config(path);assert cfg.quant.preset==preset
        gui._apply(bridge.defaults() | bridge.flatten(bridge.read_toml(path)))
        assert gui.collect()['quant.preset']==preset
        gui.editors['quant.preset'].set('custom');gui._on_edit()
        assert gui.editors['quant.mode'].widget.isEnabled()
        assert gui.editors['quant.weights_dtype'].widget.isEnabled()
        assert gui.collect()['quant.weights_dtype']==cfg.quant.weights_dtype
    finally:
        gui.close();gui.deleteLater();app.processEvents()
