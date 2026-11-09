import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtCore, QtWidgets
from trainer.gui import app as gui_module, bridge
from trainer.gui.app import TrainingGUI
from trainer.gui.preferences import Preferences


def test_console_tab_footer_preferences_and_utility_launches(monkeypatch, tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(TrainingGUI, '_load_presets', lambda self: None)
    monkeypatch.setattr(gui_module, 'detect_gpus', lambda: [(0, 'Test GPU')])
    launches = []
    monkeypatch.setattr(QtCore.QProcess, 'startDetached', lambda *args: (launches.append(args) or (True, 123)))
    window = TrainingGUI(preferences=Preferences(tmp_path/'prefs.json'))
    window._apply(bridge.defaults() | {'dataset.path': str(tmp_path)})
    window.show(); app.processEvents()
    try:
        assert window.tab_bar.tabText(window.console_tab) == 'Console'
        assert window.stack.count() == window.tab_bar.count()
        assert window.tab_bar.tabText(window.metrics_tab) == 'Live Metrics'
        assert window.preferences_btn.parent() == window.start_btn.parent()
        window._show_console()
        assert window.stack.currentWidget() == window.console.parentWidget()
        assert window.console.isVisible()
        window.log('console persists across tab switches')
        widget = window.console
        window.tab_bar.setCurrentIndex(window.metrics_tab)
        assert not window.console.isVisible()
        window.tab_bar.setCurrentIndex(window.console_tab)
        assert window.console.isVisible() and window.console is widget
        for action in window.utils_menu.actions():
            action.trigger()
        assert [args[1] for args in launches] == [
            ['-m', 'trainer.tools.lora_alpha_edit'],
            ['-m', 'trainer.tools.lora_qkv_convert'],
            ['-m', 'trainer.tools.lora_resize'],
        ]
    finally:
        window.close(); window.deleteLater(); app.processEvents()


def test_resize_sv_fro_controls():
    from trainer.tools.lora_resize import gui
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    errors = []

    def inspect():
        window = None
        try:
            window = next(w for w in app.topLevelWidgets() if w.windowTitle() == 'LoRA rank resize')
            assert window.method.currentData() == 'sv_fro'
            assert window.threshold.value() == 0.99
            assert window.threshold.isEnabled()
            assert not window.cap.isChecked() and not window.rank.isEnabled()
            window.cap.setChecked(True)
            assert window.rank.isEnabled()
            window.cap.setChecked(False)
            window.method.setCurrentIndex(window.method.findData('fixed'))
            assert window.rank.isEnabled() and not window.threshold.isEnabled()
            assert not window.cap.isEnabled()
            window.method.setCurrentIndex(window.method.findData('sv_fro'))
            assert not window.rank.isEnabled() and window.threshold.isEnabled()
        except Exception as exc:
            errors.append(exc)
        finally:
            if window is not None: window.close()
            app.quit()

    QtCore.QTimer.singleShot(100, inspect)
    assert gui() == 0
    if errors: raise errors[0]
