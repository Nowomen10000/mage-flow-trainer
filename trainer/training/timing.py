"""Wall-clock checkpoint timing; monotonic durations exclude gaps between sessions."""
from datetime import datetime, timezone
import math
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TrainingTiming:
    def __init__(self, started_at=None, started_counter=None):
        self.session_started_at = started_at or utc_now()
        self.started_counter = time.perf_counter() if started_counter is None else started_counter
        self.training_started_at = self.session_started_at
        self.previous_seconds = 0.0
        self.history_complete = True

    def resume(self, saved):
        # Legacy checkpoints have unknown historical duration. Do not invent it.
        if not saved:
            self.history_complete = False
            return
        seconds = float(saved.get("training_elapsed_seconds", 0))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid elapsed training time in resume state")
        self.previous_seconds = seconds
        self.training_started_at = saved.get("training_started_at", self.session_started_at)
        self.history_complete = saved.get("timing_history_complete", False)

    def snapshot(self):
        elapsed = max(0.0, time.perf_counter() - self.started_counter)
        return {
            "training_started_at": self.training_started_at,
            "session_started_at": self.session_started_at,
            "checkpoint_saved_at": utc_now(),
            "session_elapsed_seconds": round(elapsed, 3),
            "training_elapsed_seconds": round(self.previous_seconds + elapsed, 3),
            "timing_history_complete": self.history_complete,
        }
