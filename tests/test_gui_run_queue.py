"""Exercise the real GUI job chain without starting GPU training."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from trainer.gui import app as gui_module, bridge
from trainer.gui.app import TrainingGUI
from trainer.gui.preferences import Preferences


class FakeRunner(QtCore.QObject):
    logSignal = QtCore.Signal(str)
    errorSignal = QtCore.Signal(str)
    progressSignal = QtCore.Signal(str, bool)
    metricsSignal = QtCore.Signal(str)
    finishedSignal = QtCore.Signal(int)
    instances = []

    def __init__(self, launch, cwd, env):
        super().__init__()
        self.launch, self.cwd, self.env = launch, cwd, env
        self.active = False
        self.instances.append(self)

    def isRunning(self):
        return self.active

    def start(self):
        self.active = True

    def complete(self, code=0):
        self.active = False
        self.finishedSignal.emit(code)

    def stop(self):
        self.complete(-15)


@pytest.fixture
def queue_gui(monkeypatch, tmp_path):
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(TrainingGUI, "_load_presets", lambda self, **kwargs: None)
    monkeypatch.setattr(gui_module, "ProcessRunner", FakeRunner)
    monkeypatch.setattr(gui_module, "prevent_sleep", lambda *args: None)
    FakeRunner.instances = []
    gui = TrainingGUI(preferences=Preferences(tmp_path / 'gui.json'))
    gui._apply(bridge.defaults() | {"dataset.path": "/editor-dataset"})
    paths = []
    for i in range(2):
        path = tmp_path / f"run {i}.toml"
        path.write_text(f'[dataset]\npath="/dataset{i}"\n[train]\nrun_name="run{i}"\n')
        paths.append(path)
    gui.run_queue.add_paths(paths)
    yield gui, paths, qapp
    gui._stop()
    qapp.processEvents()
    gui.close()
    gui.deleteLater()
    qapp.processEvents()


def test_queue_add_remove_and_reorder(queue_gui):
    gui, paths, _ = queue_gui
    queue = gui.run_queue
    queue.add_paths(paths)
    assert queue.paths() == paths
    queue.list.item(1).setSelected(True)
    queue.move(-1)
    assert queue.paths() == paths[::-1]
    queue.remove_selected()
    assert queue.paths() == paths[:1]


def test_configs_cache_then_train_in_order_without_rewriting(queue_gui):
    gui, paths, qapp = queue_gui
    originals = [p.read_bytes() for p in paths]
    gui.run_queue.start_btn.click()
    assert gui._batch_queue_active
    assert not gui.run_queue.start_btn.isEnabled()
    snapshots = [e.snapshot for e in gui.run_queue.entries()]
    expected = [(False, '/dataset0'), (True, str(snapshots[0])), (False, '/dataset1'), (True, str(snapshots[1]))]
    for i, (training, argument) in enumerate(expected):
        runner = FakeRunner.instances[i]
        assert argument in runner.launch.argv
        assert gui._running_training == training
        if training:
            assert gui.metrics.elapsed_timer.isActive()
        runner.complete()
        qapp.processEvents()
    assert not gui._batch_queue_active and not gui._queue and gui.runner is None
    assert gui.run_queue.status.text() == 'Queue completed.'
    assert not gui.metrics.elapsed_timer.isActive()
    assert len(FakeRunner.instances) == 4
    assert [p.read_bytes() for p in paths] == originals
    assert gui.collect()['dataset.path'] == '/editor-dataset'


@pytest.mark.parametrize('stop', [False, True])
def test_failed_cache_or_stop_cancels_later_training(queue_gui, stop):
    gui, _, qapp = queue_gui
    gui._start_queue()
    if stop:
        gui._stop()
    else:
        gui.runner.complete(1)
    qapp.processEvents()
    assert len(FakeRunner.instances) == 1
    assert not gui._queue and not gui._batch_queue_active
    assert 'Skipped' in gui.run_queue.list.item(1).text()
    assert not gui.run_queue.start_btn.isEnabled()
    gui.run_queue.list.item(1).setSelected(True)
    gui.run_queue.retry_selected()
    assert gui.run_queue.start_btn.isEnabled()


def test_all_configs_validated_before_launch(queue_gui):
    gui, paths, _ = queue_gui
    paths[1].write_text('[train]\nwrong_key=3\n')
    gui.run_queue.list.clear()
    gui.run_queue.add_paths(paths)
    gui._start_queue()
    assert not FakeRunner.instances
    assert 'validation failed' in gui.run_queue.status.text()


def test_duplicate_outputs_rejected(queue_gui):
    gui, paths, _ = queue_gui
    paths[1].write_bytes(paths[0].read_bytes())
    gui.run_queue.list.clear()
    gui.run_queue.add_paths(paths)
    gui._start_queue()
    assert not FakeRunner.instances
    assert 'Duplicate output' in gui.run_queue.status.text()


def test_source_changed_between_cache_and_training_uses_snapshot(queue_gui):
    gui, paths, qapp = queue_gui
    gui._start_queue()
    paths[0].write_text(paths[0].read_text() + '\n# edited\n')
    gui.runner.complete()
    qapp.processEvents()
    assert len(FakeRunner.instances) == 2
    snapshot = gui.run_queue.entries()[0].snapshot
    assert str(snapshot) in gui.runner.launch.argv
    assert '# edited' not in snapshot.read_text()
    assert gui._batch_queue_active


def test_stop_between_jobs_cancels_scheduled_start(queue_gui):
    gui, _, qapp = queue_gui
    gui._start_queue()
    gui.runner.complete()
    gui._stop()  # next job is scheduled but has not started yet
    qapp.processEvents()
    assert len(FakeRunner.instances) == 1
    assert not gui._batch_queue_active


def test_stop_after_completed_run_preserves_its_status(queue_gui):
    gui, _, qapp = queue_gui
    gui._start_queue()
    gui.runner.complete()  # cache
    qapp.processEvents()
    gui.runner.complete()  # training
    gui._stop()  # before second config's cache callback
    qapp.processEvents()
    assert len(FakeRunner.instances) == 2
    assert gui.run_queue.list.item(0).text().startswith('Completed')
    assert gui.run_queue.list.item(1).text().startswith('Skipped')


def test_queue_tab_has_one_unambiguous_start_action(queue_gui):
    gui, _, _ = queue_gui
    gui.tab_bar.setCurrentIndex(gui.queue_tab)
    assert gui.start_btn.isHidden()
    assert gui.run_queue.start_btn.isEnabled()


@pytest.mark.parametrize('training', [False, True])
def test_continue_after_failure_skips_dependent_steps(queue_gui, training):
    gui, _, qapp = queue_gui
    queue = gui.run_queue
    queue.continue_on_error.setChecked(True)
    gui._start_queue()
    if training:
        gui.runner.complete()
        qapp.processEvents()
    gui.runner.errorSignal.emit('example failure details')
    gui.runner.progressSignal.emit('traceback location', False)
    gui.runner.complete(1)
    qapp.processEvents()
    assert '/dataset1' in gui.runner.launch.argv
    first = queue.entries()[0]
    assert first.status == 'Failed'
    assert 'example failure details' in first.error
    assert 'traceback location' in first.error
    queue.list.setCurrentRow(0)
    assert 'example failure details' in queue.details.toPlainText()
    gui.runner.complete()
    qapp.processEvents()
    gui.runner.complete()
    qapp.processEvents()
    assert not gui._batch_queue_active
    assert [e.status for e in queue.entries()] == ['Failed', 'Completed']
    assert '1 failed' in queue.status.text()


def test_continue_skips_invalid_snapshot_at_preflight(queue_gui):
    gui, paths, _ = queue_gui
    paths[0].write_text('[train]\nwrong_key=3\n')
    gui.run_queue.list.clear()
    gui.run_queue.add_paths(paths)
    gui.run_queue.continue_on_error.setChecked(True)
    gui._start_queue()
    assert '/dataset1' in gui.runner.launch.argv
    assert gui.run_queue.entries()[0].status == 'Failed'


def test_stop_overrides_continue_on_error(queue_gui):
    gui, _, qapp = queue_gui
    gui.run_queue.continue_on_error.setChecked(True)
    gui._start_queue()
    gui._stop()
    qapp.processEvents()
    assert len(FakeRunner.instances) == 1
    assert [e.status for e in gui.run_queue.entries()] == ['Stopped', 'Skipped']


def test_live_add_reorder_and_remove_pending_keeps_active_entry(queue_gui, tmp_path):
    gui, _, qapp = queue_gui
    queue = gui.run_queue
    gui._start_queue()
    active = queue.entries()[0]
    queue.list.item(0).setSelected(True)
    queue.remove_selected()
    assert queue.entries()[0].id == active.id
    queue.list.clearSelection()
    extra = []
    for i in (2, 3):
        path = tmp_path / f'live{i}.toml'
        path.write_text(f'[dataset]\npath="/dataset{i}"\n[train]\nrun_name="live{i}"\n')
        extra.append(path)
    queue.add_paths(extra)
    assert queue.add_btn.isEnabled()
    queue.list.item(2).setSelected(True)
    queue.move(-1)  # put dataset2 before the original pending dataset1
    queue.list.clearSelection()
    queue.list.item(3).setSelected(True)
    queue.remove_selected()  # remove dataset3
    gui.runner.complete()
    qapp.processEvents()
    assert gui._active_queue_index == active.id
    gui.runner.complete()
    qapp.processEvents()
    assert '/dataset2' in gui.runner.launch.argv
    assert queue.entries()[0].status == 'Completed'
    assert queue.entries()[1].status == 'Caching'
    assert all('/dataset3' not in r.launch.argv for r in FakeRunner.instances)


def test_source_edit_and_delete_before_start_do_not_change_snapshot(queue_gui):
    gui, paths, _ = queue_gui
    paths[0].write_text('invalid replacement')
    paths[1].unlink()
    gui._start_queue()
    assert '/dataset0' in gui.runner.launch.argv
    assert gui._batch_queue_active


def test_save_and_add_while_running_ignores_autosave_toggle(queue_gui, tmp_path):
    gui, _, _ = queue_gui
    gui.config_dir = tmp_path / 'managed'
    gui._start_queue()
    running = gui.runner
    gui.autosave_chk.setChecked(False)
    gui._apply(gui.collect() | {'train.run_name': 'fresh', 'dataset.path': '/fresh'})
    gui.add_queue_btn.click()
    saved = gui.config_dir / 'fresh.toml'
    assert saved.is_file()
    entry = gui.run_queue.entries()[-1]
    assert entry.source == saved.resolve()
    assert entry.snapshot.read_bytes() == saved.read_bytes()
    assert entry.snapshot != saved
    saved.write_text('changed after adding')
    assert 'changed after adding' not in entry.snapshot.read_text()
    assert gui.runner is running and running.isRunning()


def test_live_invalid_config_does_not_stop_queue_with_continue(queue_gui, tmp_path):
    gui, _, qapp = queue_gui
    queue = gui.run_queue
    queue.continue_on_error.setChecked(True)
    gui._start_queue()
    bad = tmp_path / 'bad.toml'
    bad.write_text('[train]\nwrong_key=3\n')
    good = tmp_path / 'good.toml'
    good.write_text('[train]\nrun_name="good"\n[dataset]\npath="/good"\n')
    queue.add_paths([bad, good])
    for _ in range(4):  # finish the two original cache/train pairs
        gui.runner.complete()
        qapp.processEvents()
    assert '/good' in gui.runner.launch.argv
    assert queue.entries()[2].status == 'Failed'


def test_pending_output_reservation_released_after_removal(queue_gui):
    gui, paths, qapp = queue_gui
    queue = gui.run_queue
    gui._start_queue()
    queue.list.item(1).setSelected(True)
    queue.remove_selected()
    queue.add_paths([paths[1]])
    gui.runner.complete()
    qapp.processEvents()
    gui.runner.complete()
    qapp.processEvents()
    assert '/dataset1' in gui.runner.launch.argv


def test_snapshot_tampering_skips_config_when_continuing(queue_gui):
    gui, _, qapp = queue_gui
    gui.run_queue.continue_on_error.setChecked(True)
    gui._start_queue()
    gui.run_queue.entries()[0].snapshot.write_text('changed')
    gui.runner.complete()
    qapp.processEvents()  # failed config's training must not launch
    qapp.processEvents()  # continue to the next config
    assert '/dataset1' in gui.runner.launch.argv
    assert gui.run_queue.entries()[0].status == 'Failed'


def test_retry_failed_entry_uses_saved_snapshot(queue_gui):
    gui, paths, qapp = queue_gui
    queue = gui.run_queue
    gui._start_queue()
    gui.runner.complete(1)
    qapp.processEvents()
    paths[0].write_text('source changed')
    queue.list.item(0).setSelected(True)
    queue.retry_selected()
    gui._start_queue()
    assert '/dataset0' in gui.runner.launch.argv
    assert queue.entries()[0].status == 'Caching'


def test_failed_subset_cache_skips_other_caches_and_training(queue_gui):
    gui, paths, qapp = queue_gui
    paths[0].write_text('[train]\nrun_name="subsets"\n'
                        '[[dataset.subsets]]\npath="/a"\n'
                        '[[dataset.subsets]]\npath="/b"\n')
    gui.run_queue.list.clear()
    gui.run_queue.add_paths(paths)
    gui.run_queue.continue_on_error.setChecked(True)
    gui._start_queue()
    assert '/a' in gui.runner.launch.argv
    gui.runner.complete(1)
    qapp.processEvents()
    assert '/dataset1' in gui.runner.launch.argv
    assert len(FakeRunner.instances) == 2


@pytest.mark.parametrize('keep_going', [False, True])
def test_live_duplicate_output_is_rejected(queue_gui, tmp_path, keep_going):
    gui, paths, qapp = queue_gui
    queue = gui.run_queue
    queue.continue_on_error.setChecked(keep_going)
    gui._start_queue()
    duplicate = tmp_path / 'duplicate.toml'
    duplicate.write_bytes(paths[0].read_bytes())
    good = tmp_path / 'last.toml'
    good.write_text('[train]\nrun_name="last"\n[dataset]\npath="/last"\n')
    queue.add_paths([duplicate, good])
    for _ in range(4):
        gui.runner.complete()
        qapp.processEvents()
    assert queue.entries()[2].status == 'Failed'
    assert 'Duplicate output' in queue.entries()[2].error
    if keep_going:
        assert '/last' in gui.runner.launch.argv
    else:
        assert gui.runner is None and not gui._batch_queue_active
        assert queue.entries()[3].status == 'Skipped'
