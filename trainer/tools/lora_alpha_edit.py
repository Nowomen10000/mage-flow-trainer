#!/usr/bin/env python
"""A small PySide6 tool for baking inference weight into a finished LoRA safetensors file.

Alpha is not a label: sd-scripts applies `dW * (alpha / dim)`, so alpha *is* a plain LoRA's
baked-in strength. A dim 16 / alpha 8 LoRA (scale 0.5) that looks best at weight 1.5 becomes
alpha 12 (scale 0.75), and afterwards weight 1.0 behaves like 1.5 did.

That equivalence holds only WITHOUT weight decomposition. For a DoRA, alpha enters the update
before the magnitude renormalisation and the inference weight is applied after it:

    lora_diff *= alpha                       # alpha here
    weight_calc = W + lora_diff
    weight_calc *= dora_scale / ||W||
    weight += strength * (weight_calc - W)   # strength here

so with k = dora_scale / ||W|| and D = up @ down,

    baked alpha (s*alpha), weight 1.0 :        k*W + s*k*alpha*D
    original alpha,        weight s   : (s*k - s + 1)*W + s*k*alpha*D

The LoRA term matches; the base weight is left scaled differently, by (s-1)(1-k) per output
channel. This tool therefore bakes a DoRA exactly instead, by rewriting the two tensors that can
express it -- alpha is left alone:

    dora_scale' = s*dora_scale - (s-1)*||W||
    lora_up'    = lora_up * (s*k / (s*k - s + 1))   per output channel

That needs ||W|| from the base model, which is why an exact DoRA bake asks for it. The result is
specific to that base model.

Everything else is copied through byte for byte, so untouched weights stay bit-identical.

    ./venv/bin/python lora_alpha_edit.py [file.safetensors]
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# safetensors layout: <u64 header_len><header json><data>; header offsets are relative to data.
_HEADER_LEN_BYTES = 8

_STRUCT_FMT = {"F64": "<d", "F32": "<f", "F16": "<e"}
_ITEM_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
_NP_DTYPE = {"F64": np.float64, "F32": np.float32, "F16": np.float16}

_RANK_KEYS = (".lora_down.weight", ".lora_A.weight", ".hada_w1_b", ".hada_w2_b", ".lokr_w1")
_BASE_PREFIXES = ("net.", "model.diffusion_model.", "diffusion_model.", "model.", "")
_LORA_PREFIXES = ("lora_unet_", "lora_te_", "lora_")


def _decode_scalar(raw: bytes, dtype: str) -> float:
    if dtype == "BF16":
        return struct.unpack("<f", b"\x00\x00" + raw[:2])[0]
    return struct.unpack(_STRUCT_FMT[dtype], raw[: _ITEM_SIZE[dtype]])[0]


def _encode_scalar(value: float, dtype: str) -> bytes:
    if dtype == "BF16":
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16  # round half to even at bit 16
        return struct.pack("<H", bits & 0xFFFF)
    return struct.pack(_STRUCT_FMT[dtype], value)


def _array_from_bytes(raw: bytes, dtype: str, shape) -> np.ndarray:
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    return np.frombuffer(raw, dtype=_NP_DTYPE[dtype]).reshape(shape).astype(np.float32)


def _array_to_bytes(arr: np.ndarray, dtype: str) -> bytes:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if dtype == "BF16":
        u = a.view(np.uint32)
        u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16  # round half to even
        return np.ascontiguousarray(u.astype("<u2")).tobytes()
    return np.ascontiguousarray(a.astype(_NP_DTYPE[dtype])).tobytes()


def _read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        raw_len = f.read(_HEADER_LEN_BYTES)
        if len(raw_len) < _HEADER_LEN_BYTES:
            raise ValueError("file is too short to be a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        if not 0 < header_len < 512 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header length ({header_len})")
        blob = f.read(header_len)
        if len(blob) < header_len:
            raise ValueError("truncated safetensors header")
    try:
        return json.loads(blob), _HEADER_LEN_BYTES + header_len
    except json.JSONDecodeError as e:
        raise ValueError(f"header is not valid JSON: {e}") from e


class BaseModel:
    """Per-output-channel weight norms from the base checkpoint, read one layer at a time."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.header, self.data_start = _read_header(path)
        self.lookup: dict[str, str] = {}
        for key in self.header:
            if key == "__metadata__" or not key.endswith(".weight"):
                continue
            core = key[: -len(".weight")]
            for prefix in _BASE_PREFIXES:
                if core.startswith(prefix):
                    self.lookup.setdefault(core[len(prefix):].replace(".", "_"), key)

    def row_norms(self, module: str) -> np.ndarray | None:
        for prefix in _LORA_PREFIXES:
            if module.startswith(prefix):
                module = module[len(prefix):]
                break
        key = self.lookup.get(module)
        if key is None:
            return None
        info = self.header[key]
        begin, end = info["data_offsets"]
        with self.path.open("rb") as f:
            f.seek(self.data_start + begin)
            raw = f.read(end - begin)
        W = _array_from_bytes(raw, info["dtype"], info["shape"])
        return np.linalg.norm(W.reshape(W.shape[0], -1), axis=1)


class LoraFile:
    """A LoRA safetensors file: its alpha scalars, and any DoRA magnitude vectors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.header, self.data_start = _read_header(path)
        self.metadata = self.header.get("__metadata__", {})
        self.dora_modules = sum(1 for k in self.header if k.endswith(".dora_scale"))

        # alpha name -> (dtype, absolute offset, current value, rank or None)
        self.alphas: dict[str, tuple[str, int, float, int | None]] = {}
        with path.open("rb") as f:
            for name, info in self.header.items():
                if name == "__metadata__" or not name.endswith(".alpha"):
                    continue
                dtype = info.get("dtype", "")
                if dtype not in _ITEM_SIZE:
                    raise ValueError(f"{name} has unsupported dtype {dtype!r}")
                begin, end = info["data_offsets"]
                if end - begin != _ITEM_SIZE[dtype]:
                    raise ValueError(f"{name} is not a scalar ({end - begin} bytes)")
                f.seek(self.data_start + begin)
                value = _decode_scalar(f.read(end - begin), dtype)
                self.alphas[name] = (dtype, self.data_start + begin, value, self._rank_of(name))

        if not self.alphas:
            # PEFT-style exports keep alpha in an adapter_config.json, so there is nothing here
            # to edit. Say that, rather than implying the file is broken.
            n_lora = sum(1 for k in self.header if k != "__metadata__" and ".lora_" in k)
            raise ValueError(
                "this file has no '.alpha' tensors, so there is no stored alpha to change.\n\n"
                f"({n_lora} LoRA weight tensors found. PEFT-style exports keep alpha outside the "
                "file; such a LoRA already applies at alpha = dim, i.e. scale 1.0.)"
            )

    def _rank_of(self, alpha_name: str) -> int | None:
        prefix = alpha_name[: -len(".alpha")]
        for suffix in _RANK_KEYS:
            info = self.header.get(prefix + suffix)
            if info and info.get("shape"):
                return int(info["shape"][0])
        return None

    def _read_array(self, name: str) -> np.ndarray:
        info = self.header[name]
        begin, end = info["data_offsets"]
        with self.path.open("rb") as f:
            f.seek(self.data_start + begin)
            raw = f.read(end - begin)
        return _array_from_bytes(raw, info["dtype"], info["shape"])

    def groups(self) -> list[tuple[int | None, float, int]]:
        counts: dict[tuple[int | None, float], int] = {}
        for _, _, value, rank in self.alphas.values():
            counts[(rank, value)] = counts.get((rank, value), 0) + 1
        return sorted(
            ((rank, alpha, n) for (rank, alpha), n in counts.items()),
            key=lambda g: (-g[2], g[0] or 0),
        )

    # ── exact DoRA bake ──
    def dora_bake(self, strength: float, base: BaseModel) -> tuple[dict[str, np.ndarray], list[str]]:
        """Tensor replacements that make weight 1.0 reproduce `strength`. Also returns warnings.

        The arrays come back in fp32 and are written as fp32. Rounding them into the file's native
        bf16 would cost ~2^-8 relative error -- larger than the systematic error this whole
        operation exists to remove -- so the two rewritten tensors are promoted instead.
        """
        patches: dict[str, np.ndarray] = {}
        unmatched, unstable, wrong_axis = [], [], []
        amplification = 0.0

        for alpha_name in self.alphas:
            module = alpha_name[: -len(".alpha")]
            ds_name, up_name = module + ".dora_scale", module + ".lora_up.weight"
            ds_info, up_info = self.header.get(ds_name), self.header.get(up_name)
            if ds_info is None or up_info is None:
                continue

            # The derivation assumes Forge's wd_on_output_axis branch, where the normalising
            # factor is ||W|| and therefore independent of alpha. Otherwise it is computed from
            # the adapted weight and the relationship is genuinely non-linear.
            if int(ds_info["shape"][0]) != int(up_info["shape"][0]):
                wrong_axis.append(module)
                continue

            norms = base.row_norms(module)
            if norms is None:
                unmatched.append(module)
                continue
            ds = self._read_array(ds_name).reshape(-1)
            if ds.shape[0] != norms.shape[0]:
                unmatched.append(module)
                continue

            k = ds / np.maximum(norms, 1e-12)
            k_new = strength * k - strength + 1.0
            # A negative k' is fine and is exactly what the original does at this weight: the
            # extrapolation genuinely inverts that channel. Only k' ~ 0 is singular, because the
            # base-weight term vanishes while lora_up would have to grow without bound.
            if np.any(np.abs(k_new) < 1e-3):
                unstable.append(module)
                continue

            up = self._read_array(up_name)
            factor = (strength * k) / k_new
            amplification = max(amplification, float(np.abs(factor).max()))
            new_up = up.reshape(up.shape[0], -1) * factor[:, None]

            patches[ds_name] = (k_new * norms).reshape(ds_info["shape"]).astype(np.float32)
            patches[up_name] = new_up.reshape(up_info["shape"]).astype(np.float32)

        warnings = []
        if wrong_axis:
            warnings.append(f"{len(wrong_axis)} module(s) decompose on the input axis; left unchanged")
        if unmatched:
            warnings.append(f"{len(unmatched)} module(s) had no matching base weight; left unchanged")
        if unstable:
            warnings.append(
                f"{len(unstable)} module(s) have a channel at k'~0 and cannot represent this "
                "weight; left unchanged"
            )
        if amplification > 10:
            warnings.append(f"lora_up amplified up to {amplification:.0f}x on some channels")
        return patches, warnings

    # ── writing ──
    def save_as(self, dest: Path, new_alpha_for=None, patches=None, extra_metadata=None) -> None:
        """Copy to `dest`, replacing alpha scalars and any `patches` (name -> fp32 array).

        Patched arrays are written as F32, which changes their byte length, so the data section is
        rebuilt with recomputed offsets rather than copied wholesale. Untouched tensors are still
        copied byte for byte and stay bit-identical.
        """
        new_bytes: dict[str, bytes] = {}
        new_dtype: dict[str, str] = {}
        resulting: dict[str, float] = {}

        if new_alpha_for is not None:
            for name, (dtype, _, value, rank) in self.alphas.items():
                new = new_alpha_for(name, value, rank)
                resulting[name] = new
                if new != value:
                    new_bytes[name] = _encode_scalar(new, dtype)  # same dtype, same length

        for name, arr in (patches or {}).items():
            if name not in self.header:
                raise ValueError(f"patch for unknown tensor {name}")
            new_bytes[name] = _array_to_bytes(arr, "F32")
            new_dtype[name] = "F32"

        # Preserve the original layout order so unchanged tensors stay contiguous.
        entries = sorted(
            ((n, i) for n, i in self.header.items() if n != "__metadata__"),
            key=lambda kv: kv[1]["data_offsets"][0],
        )

        header: dict = {}
        plan: list[tuple[str, int, int]] = []  # (name, src_begin, src_end); begin<0 => new bytes
        offset = 0
        for name, info in entries:
            begin, end = info["data_offsets"]
            if name in new_bytes:
                length = len(new_bytes[name])
                plan.append((name, -1, -1))
            else:
                length = end - begin
                plan.append((name, begin, end))
            header[name] = {
                "dtype": new_dtype.get(name, info["dtype"]),
                "shape": info["shape"],
                "data_offsets": [offset, offset + length],
            }
            offset += length

        metadata = dict(self.metadata)
        distinct = set(resulting.values())
        if len(distinct) == 1:
            # Only meaningful when every module agrees; on a mixed result leave it alone rather
            # than recording a value true of just some layers.
            metadata["ss_network_alpha"] = str(next(iter(distinct)))
        metadata.update(extra_metadata or {})
        if metadata:
            header["__metadata__"] = metadata

        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        blob += b" " * (-len(blob) % 8)  # keep the data section 8-byte aligned

        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            with self.path.open("rb") as src, tmp.open("wb") as out:
                out.write(struct.pack("<Q", len(blob)))
                out.write(blob)
                for name, begin, end in plan:
                    if begin < 0:
                        out.write(new_bytes[name])
                        continue
                    src.seek(self.data_start + begin)
                    remaining = end - begin
                    while remaining:
                        chunk = src.read(min(remaining, 8 * 1024 * 1024))
                        if not chunk:
                            raise ValueError(f"unexpected end of file reading {name}")
                        out.write(chunk)
                        remaining -= len(chunk)
            tmp.replace(dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


class AlphaEditor(QtWidgets.QWidget):
    BAKE, SET, TO_DIM = 0, 1, 2
    MODES = ("Bake in inference weight", "Set alpha to", "Set alpha = dim (scale 1.0)")

    def __init__(self) -> None:
        super().__init__()
        self.lora: LoraFile | None = None
        self.base_path: Path | None = None
        self.setWindowTitle("LoRA Alpha Editor")
        self.setAcceptDrops(True)
        self.resize(820, 560)
        self._build()
        self._refresh()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        grid = QtWidgets.QGridLayout()
        self.path_edit = QtWidgets.QLineEdit(
            placeholderText="Drop a .safetensors LoRA here, or browse..."
        )
        self.path_edit.setReadOnly(True)
        lora_btn = QtWidgets.QPushButton("Browse...")
        lora_btn.clicked.connect(self._browse_lora)
        grid.addWidget(QtWidgets.QLabel("LoRA"), 0, 0)
        grid.addWidget(self.path_edit, 0, 1)
        grid.addWidget(lora_btn, 0, 2)

        self.base_edit = QtWidgets.QLineEdit(
            placeholderText="Only needed to bake a DoRA exactly"
        )
        self.base_edit.setReadOnly(True)
        self.base_btn = QtWidgets.QPushButton("Browse...")
        self.base_btn.clicked.connect(self._browse_base)
        self.base_label = QtWidgets.QLabel("Base model")
        grid.addWidget(self.base_label, 1, 0)
        grid.addWidget(self.base_edit, 1, 1)
        grid.addWidget(self.base_btn, 1, 2)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Modules", "Dim", "Alpha", "Scale", "New alpha → scale"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.table, 1)

        form = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(self.MODES)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.value = QtWidgets.QDoubleSpinBox()
        self.value.setDecimals(4)
        self.value.setRange(0.0001, 1_000_000.0)
        self.value.setValue(1.5)
        self.value.setSingleStep(0.05)
        self.value.valueChanged.connect(self._refresh)
        form.addWidget(self.mode)
        form.addWidget(self.value, 1)
        layout.addLayout(form)

        self.note = QtWidgets.QLabel(wordWrap=True)
        self.note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.note)

        self.out_label = QtWidgets.QLabel(wordWrap=True)
        self.out_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.out_label)

        buttons = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self._save)
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.save_btn)
        layout.addLayout(buttons)

    # ── loading ──
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        urls = e.mimeData().urls()
        if len(urls) == 1 and urls[0].toLocalFile().endswith(".safetensors"):
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        self._load(Path(e.mimeData().urls()[0].toLocalFile()))

    def _browse_lora(self) -> None:
        name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open LoRA", self.path_edit.text() or "", "safetensors (*.safetensors)"
        )
        if name:
            self._load(Path(name))

    def _browse_base(self) -> None:
        name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open base model", self.base_edit.text() or "", "safetensors (*.safetensors)"
        )
        if name:
            self.base_path = Path(name)
            self.base_edit.setText(name)
            self._refresh()

    def _load(self, path: Path) -> None:
        try:
            self.lora = LoraFile(path)
        except (ValueError, OSError) as e:
            self.lora = None
            self.path_edit.clear()
            QtWidgets.QMessageBox.critical(self, "Could not read LoRA", f"{path.name}:\n\n{e}")
        else:
            self.path_edit.setText(str(path))
        self._refresh()

    def _mode_changed(self) -> None:
        # Each mode wants a different kind of number; don't carry over one that meant something
        # else in the previous mode.
        if self.mode.currentIndex() == self.BAKE:
            self.value.setValue(1.5)
        elif self.mode.currentIndex() == self.SET and self.lora:
            ranks = {r for _, _, _, r in self.lora.alphas.values() if r is not None}
            if len(ranks) == 1:
                self.value.setValue(float(next(iter(ranks))))
        self._refresh()

    # ── state ──
    def _exact_dora(self) -> bool:
        """True when this save will be an exact DoRA bake rather than an alpha edit."""
        return bool(
            self.lora
            and self.lora.dora_modules
            and self.mode.currentIndex() == self.BAKE
            and self.base_path
        )

    def _new_alpha(self, current: float, rank: int | None) -> float | None:
        mode = self.mode.currentIndex()
        if mode == self.BAKE:
            return current * self.value.value()
        if mode == self.SET:
            return self.value.value()
        return float(rank) if rank is not None else None

    def _out_path(self) -> Path | None:
        if self.lora is None:
            return None
        src = self.lora.path
        if self._exact_dora():
            return src.with_name(f"{src.stem}-bake{self.value.value():g}{src.suffix}")
        results = {self._new_alpha(v, r) for _, _, v, r in self.lora.alphas.values()}
        results.discard(None)
        if len(results) == 1:
            tag = f"{next(iter(results)):g}"
        elif self.mode.currentIndex() == self.BAKE:
            tag = f"x{self.value.value():g}"  # alphas differ, so name it by the multiplier
        else:
            tag = "mixed"
        return src.with_name(f"{src.stem}-alpha{tag}{src.suffix}")

    # ── preview ──
    def _refresh(self) -> None:
        mode = self.mode.currentIndex()
        self.value.setEnabled(mode != self.TO_DIM)
        self.value.setSuffix("×" if mode == self.BAKE else "")
        self.save_btn.setEnabled(self.lora is not None)
        is_dora = bool(self.lora and self.lora.dora_modules)
        for w in (self.base_label, self.base_edit, self.base_btn):
            w.setEnabled(is_dora)

        if self.lora is None:
            self.table.setRowCount(0)
            self.note.setText("")
            self.out_label.setText("")
            self.status.setText("No file loaded.")
            return

        exact = self._exact_dora()
        groups = self.lora.groups()
        self.table.setRowCount(len(groups))
        unknown_rank = 0
        for row, (rank, alpha, count) in enumerate(groups):
            new = self._new_alpha(alpha, rank)
            if rank is None:
                unknown_rank += count
            scale = f"{alpha / rank:.4g}" if rank else "?"
            if exact:
                result = "alpha unchanged (exact DoRA bake)"
            elif new is None:
                result = "unchanged (dim unknown)"
            else:
                result = f"{new:g} → {new / rank:.4g}" if rank else f"{new:g} → ?"
            for col, text in enumerate(
                (str(count), str(rank or "?"), f"{alpha:g}", scale, result)
            ):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))

        self.status.setText(
            f"{len(self.lora.alphas)} module(s), {len(groups)} distinct dim/alpha group(s)"
            + (f", {self.lora.dora_modules} DoRA" if is_dora else "")
        )

        notes = []
        if exact:
            notes.append(
                f"<b>Exact DoRA bake.</b> alpha is left alone; <code>dora_scale</code> and "
                f"<code>lora_up</code> are rewritten so weight <b>1.0</b> reproduces "
                f"<b>{self.value.value():g}</b> exactly. The result is specific to this base model."
            )
        elif is_dora:
            notes.append(
                f"<b>⚠ This is a DoRA ({self.lora.dora_modules} dora_scale tensors)</b> and alpha "
                "is not interchangeable with inference weight: alpha applies before the DoRA "
                "magnitude renormalisation, weight after it. "
                + (
                    "Pick a base model above to bake it exactly."
                    if mode == self.BAKE
                    else "Editing alpha here will not match the weight you tuned."
                )
            )
        else:
            notes.append(
                "sd-scripts applies <b>dW × (alpha / dim)</b>, so alpha is this LoRA's baked-in "
                "strength. The weight tensors are copied through untouched."
            )
            if mode == self.BAKE:
                notes.append(
                    f"After saving, inference weight <b>1.0</b> will behave like "
                    f"<b>{self.value.value():g}</b> does now."
                )
        if unknown_rank:
            notes.append(
                f"<b>{unknown_rank} module(s) have no recognisable rank tensor.</b> Their scale "
                "cannot be shown, and 'Set alpha = dim' leaves them unchanged."
            )
        self.note.setText("<br><br>".join(notes))

        out = self._out_path()
        exists = " <b>(exists — will ask before overwriting)</b>" if out and out.exists() else ""
        self.out_label.setText(f"Saves to: <code>{out.name}</code>{exists}" if out else "")

    # ── saving ──
    def _save(self) -> None:
        if self.lora is None:
            return
        dest = self._out_path()
        if dest is None:
            return
        if dest == self.lora.path:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to do",
                "That would overwrite the source file in place. Pick a different value.",
            )
            return
        if dest.exists():
            reply = QtWidgets.QMessageBox.question(
                self, "Overwrite?", f"{dest.name} already exists.\n\nOverwrite it?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        exact = self._exact_dora()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            if exact:
                self.status.setText("Reading base model weights...")
                QtWidgets.QApplication.processEvents()
                strength = self.value.value()
                patches, warnings = self.lora.dora_bake(strength, BaseModel(self.base_path))
                if not patches:
                    QtWidgets.QMessageBox.critical(
                        self, "Could not bake",
                        "No modules could be matched to the base model.\n\n"
                        "Is this the checkpoint the LoRA was trained on?",
                    )
                    return
                self.lora.save_as(
                    dest,
                    patches=patches,
                    extra_metadata={
                        "ss_dora_baked_weight": str(strength),
                        "ss_dora_bake_base": self.base_path.name,
                    },
                )
                done = f"Wrote {dest.name} ({len(patches) // 2} module(s) baked at {strength:g}×)"
                if warnings:
                    done += " — " + "; ".join(warnings)
            else:
                self.lora.save_as(dest, new_alpha_for=lambda n, cur, r: self._new_alpha(cur, r) or cur)
                done = f"Wrote {dest.name}"
        except (ValueError, OSError) as e:
            QtWidgets.QMessageBox.critical(self, "Could not save", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.status.setText(done)
        QtWidgets.QMessageBox.information(self, "Saved", f"{done}\n\n{dest}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    from trainer.gui.widgets import STYLESHEET
    app.setStyleSheet(STYLESHEET)
    w = AlphaEditor()
    if len(sys.argv) > 1:
        w._load(Path(sys.argv[1]))
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
