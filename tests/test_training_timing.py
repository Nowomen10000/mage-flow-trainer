import pytest

from trainer.training import timing


def test_resume_accumulates_sessions_without_downtime(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: clock[0])
    first = timing.TrainingTiming('2026-11-04T10:00:00+00:00')
    clock[0] = 160
    saved = first.snapshot()
    assert saved['training_elapsed_seconds'] == 60
    clock[0] = 10000
    second = timing.TrainingTiming('2026-11-05T10:00:00+00:00')
    clock[0] += 20  # Loading the resume state counts as session time.
    second.resume(saved)
    clock[0] += 10
    resumed = second.snapshot()
    assert resumed['training_elapsed_seconds'] == 90
    assert resumed['session_elapsed_seconds'] == 30
    assert resumed['training_started_at'] == saved['training_started_at']
    assert resumed['session_started_at'] != saved['session_started_at']
    assert resumed['timing_history_complete'] is True
    assert second.snapshot()['training_elapsed_seconds'] == 90  # Saving never double-counts.


def test_legacy_resume_marks_unknown_history(monkeypatch):
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: 30)
    timer = timing.TrainingTiming(started_counter=10)
    timer.resume(None)
    saved = timer.snapshot()
    assert saved['training_elapsed_seconds'] == 20
    assert saved['timing_history_complete'] is False
    resumed = timing.TrainingTiming()
    resumed.resume(saved)
    assert resumed.snapshot()['timing_history_complete'] is False


@pytest.mark.parametrize('seconds', [-1, float('nan'), float('inf')])
def test_invalid_resume_duration_rejected(seconds):
    with pytest.raises(ValueError):
        timing.TrainingTiming().resume({'training_elapsed_seconds': seconds})
