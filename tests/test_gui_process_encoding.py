"""GUI subprocess logs must not depend on the Windows ANSI code page."""
import os
import subprocess
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6 import QtWidgets

from trainer.gui.process import Launch, ProcessRunner


@pytest.mark.parametrize('inherited_encoding', ['cp932', 'utf-8'])
def test_unicode_child_logs_survive_japanese_parent_locale(tmp_path, monkeypatch, inherited_encoding):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Model a Japanese non-Unicode Windows locale without changing the host OS.
    monkeypatch.setattr(subprocess, '_text_encoding', lambda: 'cp932')
    env = dict(os.environ, PYTHONIOENCODING=inherited_encoding)
    text = '日本語の出力 📈'
    marker = 'BLUFLOW_FINAL_SAVE {"output_dir":"日本語の出力"}'
    script = f'print({text!r}, flush=True); print({marker!r}, flush=True)'
    runner = ProcessRunner(Launch([sys.executable, '-u', '-c', script], 'unicode test'), str(tmp_path), env)
    lines, errors, exits = [], [], []
    runner.metricsSignal.connect(lines.append)
    runner.errorSignal.connect(errors.append)
    runner.finishedSignal.connect(exits.append)
    runner.run()
    assert exits == [0], errors
    assert not errors
    assert lines == [text, marker]
    assert env['PYTHONIOENCODING'] == inherited_encoding  # Only the child's copy changes.
    assert app is not None


def test_non_utf8_native_output_does_not_hide_final_notification(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(subprocess, '_text_encoding', lambda: 'cp932')
    # A native library or build tool may still write legacy-encoded bytes.
    script = 'import os; os.write(1, b"native: \\xff\\nBLUFLOW_FINAL_SAVE {}\\n")'
    runner = ProcessRunner(Launch([sys.executable, '-u', '-c', script], 'legacy bytes'), str(tmp_path))
    lines, errors, exits = [], [], []
    runner.metricsSignal.connect(lines.append)
    runner.errorSignal.connect(errors.append)
    runner.finishedSignal.connect(exits.append)
    runner.run()
    assert exits == [0], errors
    assert not errors
    assert lines == ['native: \ufffd', 'BLUFLOW_FINAL_SAVE {}']
    assert app is not None
