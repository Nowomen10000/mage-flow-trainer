"""Session-local training queue with independent config snapshots."""
from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile

from PySide6 import QtCore, QtWidgets

from .widgets import make_btn, usable_dialog_start


@dataclass
class QueueEntry:
    id: int
    source: Path
    snapshot: Path
    digest: str
    status: str = "Pending"
    error: str = ""


class RunQueueWidget(QtWidgets.QWidget):
    startRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        self._next_id = 0
        self._snapshots = tempfile.TemporaryDirectory(prefix="mageflow-queue-")
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "Each entry keeps a copy of its TOML as added. Editing the source file will not change "
            "queued settings. Add jobs while running; move or remove pending jobs with the buttons. "
            "Stop cancels the remaining runs. Relative paths use the trainer folder.")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.continue_on_error = QtWidgets.QCheckBox("Continue queue after errors")
        self.continue_on_error.setToolTip(
            "Mark invalid or failed configs as Failed and continue with the next config. "
            "A failed cache always skips training for that config. Stop still stops the queue.")
        layout.addWidget(self.continue_on_error)
        self.summary = QtWidgets.QLabel()
        layout.addWidget(self.summary)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.list, 1)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(110)
        self.details.setPlaceholderText("Select a run to inspect its source, snapshot and error.")
        layout.addWidget(self.details)
        buttons = QtWidgets.QHBoxLayout()
        self.add_btn = make_btn("Add TOMLs", self._add)
        self.edit_buttons = [
            self.add_btn,
            make_btn("Move up", lambda: self.move(-1)),
            make_btn("Move down", lambda: self.move(1)),
            make_btn("Remove selected", self.remove_selected),
            make_btn("Clear pending", self.clear_pending),
            make_btn("Retry selected", self.retry_selected),
        ]
        for button in self.edit_buttons:
            buttons.addWidget(button)
        buttons.addStretch()
        self.start_btn = make_btn("Start queue", self.startRequested.emit, style="accent")
        buttons.addWidget(self.start_btn)
        layout.addLayout(buttons)
        self.status = QtWidgets.QLabel("No runs queued.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.list.itemSelectionChanged.connect(self._show_details)
        for signal in (self.list.model().rowsInserted, self.list.model().rowsRemoved,
                       self.list.model().modelReset, self.list.model().rowsMoved):
            signal.connect(self._changed)
        self.set_busy(False)

    def entries(self):
        return [self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                for i in range(self.list.count())]

    def entry(self, entry_id):
        return next(e for e in self.entries() if e.id == entry_id)

    def pending(self):
        return [e for e in self.entries() if e.status == "Pending"]

    def paths(self):
        return [e.source for e in self.entries()]

    def _changed(self, *args):
        counts = Counter(e.status for e in self.entries())
        self.start_btn.setEnabled(not self._busy and bool(counts["Pending"]))
        self.summary.setText(
            f"Running: {counts['Caching'] + counts['Training']} · Pending: {counts['Pending']} · "
            f"Completed: {counts['Completed']} · Failed: {counts['Failed']} · "
            f"Stopped/skipped: {counts['Stopped'] + counts['Skipped']}")

    def _show_details(self):
        item = self.list.currentItem()
        if item is None:
            self.details.clear()
            return
        entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.details.setPlainText(
            f"Source: {entry.source}\nSnapshot: {entry.snapshot}\n{entry.status}"
            + (f"\n{entry.error}" if entry.error else ""))

    def add_paths(self, paths):
        added = []
        errors = []
        for path in (Path(p).resolve() for p in paths):
            try:
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if any(e.source == path and e.digest == digest for e in self.entries()):
                    continue
                snapshot = Path(self._snapshots.name) / f"{self._next_id:04d}-{path.name}"
                snapshot.write_bytes(data)
            except OSError as exc:
                errors.append(f"Could not queue {path}: {exc}")
                continue
            entry = QueueEntry(self._next_id, path, snapshot, digest)
            self._next_id += 1
            item = QtWidgets.QListWidgetItem(f"Pending — {path}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(f"{path}\nSaved snapshot: {snapshot}")
            self.list.addItem(item)
            added.append(entry)
        if added or errors:
            self.status.setText("\n".join([
                f"Added {len(added)} configuration(s). Queued settings are saved separately.", *errors]))
        return added

    def _add(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add training configs", usable_dialog_start("configs"), "TOML configs (*.toml)")
        self.add_paths(paths)

    def remove_selected(self):
        for item in self.list.selectedItems():
            entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if entry.status in ("Caching", "Training") or (self._busy and entry.status != "Pending"):
                continue
            self.list.takeItem(self.list.row(item))
            entry.snapshot.unlink(missing_ok=True)

    def clear_pending(self):
        for i in reversed(range(self.list.count())):
            entry = self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            if entry.status == "Pending":
                self.list.takeItem(i)
                entry.snapshot.unlink(missing_ok=True)

    def retry_selected(self):
        for item in self.list.selectedItems():
            entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if entry.status in ("Failed", "Stopped", "Skipped"):
                self.mark(entry.id, "Pending")
        self.status.setText("Selected failed/stopped/skipped runs are pending again, using their saved settings.")

    def move(self, direction):
        rows = sorted((self.list.row(item) for item in self.list.selectedItems()), reverse=direction > 0)
        selected = set(rows)
        for row in rows:
            target = row + direction
            entry = self.list.item(row).data(QtCore.Qt.ItemDataRole.UserRole)
            if entry.status != "Pending" or not 0 <= target < self.list.count() or target in selected:
                continue
            neighbor = self.list.item(target).data(QtCore.Qt.ItemDataRole.UserRole)
            if neighbor.status != "Pending":
                continue
            item = self.list.takeItem(row)
            self.list.insertItem(target, item)
            item.setSelected(True)
            selected.remove(row)
            selected.add(target)

    def set_busy(self, busy):
        self._busy = busy
        # Dragging is unrestricted only while idle; live edits use the guarded buttons.
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.NoDragDrop if busy
                                  else QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.continue_on_error.setEnabled(not busy)
        self._changed()

    def mark(self, entry_id, status, error=""):
        entry = self.entry(entry_id)
        entry.status, entry.error = status, error
        item = next(self.list.item(i) for i in range(self.list.count())
                    if self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole).id == entry_id)
        item.setText(f"{status} — {entry.source}")
        self.status.setText(f"{status} — {entry.source.name}" + (f": {error.splitlines()[0]}" if error else ""))
        self._changed()
        self._show_details()

    def end(self, message):
        for entry in self.pending():
            self.mark(entry.id, "Skipped")
        self.set_busy(False)
        self.status.setText(message)
