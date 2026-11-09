"""Resize plain LoRA/LoCon ranks with SVD, preserving the effective alpha scaling.

Uses QR of the existing factors followed by a small SVD instead of constructing
an entire diffusion-model weight matrix. CPU only; no training GPU is required.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile


def resize_state(state, rank=None, progress=None, *, method='fixed', threshold=0.99):
    import torch
    from trainer.modeling.moss_export import separate_adapter_qkv, fuse_qkv

    if method not in ('fixed', 'sv_fro'):
        raise ValueError('Resize method must be fixed or sv_fro')
    if method == 'fixed' and rank is None:
        rank = 8  # Preserve the original fixed-rank default.
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
        raise ValueError('Target rank must be a positive integer')
    if method == 'sv_fro' and (not math.isfinite(threshold) or not 0 < threshold <= 1):
        raise ValueError('sv_fro threshold must be greater than 0 and at most 1')
    pairs = (('.lora_down.weight', '.lora_up.weight'),
             ('.lora_A.weight', '.lora_B.weight'))
    # Validate every tensor before doing work. DoRA/LoKr/LoHa/LoCon mid tensors
    # cannot be copied through a rank change without changing their meaning.
    allowed = set()
    for key in state:
        for ds, us in pairs:
            if key.endswith(ds):
                prefix = key[:-len(ds)]
                if prefix + us not in state:
                    raise ValueError(f'Missing up/B factor: {prefix}')
                allowed.update((key, prefix + us, prefix + '.alpha'))
    unsupported = set(state) - allowed
    if unsupported or not allowed:
        raise ValueError('Only plain LoRA/LoCon factor pairs are supported; unsupported tensors: '
                         + ', '.join(sorted(unsupported)[:4]))
    state, split = separate_adapter_qkv(state, compact=True)
    fused = bool(split['joint_blocks'])
    result = dict(state)
    modules = [(key, ds, us) for key in state for ds, us in pairs if key.endswith(ds)]
    reports = {}
    with torch.no_grad():
        for index, (key, ds, us) in enumerate(modules):
            prefix = key[:-len(ds)]
            down, up = state[key], state[prefix + us]
            old_rank = down.shape[0]
            if down.ndim not in (2, 4) or up.ndim != down.ndim or up.shape[1] != old_rank:
                raise ValueError(f'{prefix}: invalid LoRA factor shapes')
            if down.ndim == 4 and up.shape[2:] != (1, 1):
                raise ValueError(f'{prefix}: only LoCon with a 1x1 up projection is supported')
            alpha = state.get(prefix + '.alpha')
            scale = float(alpha.item()) / old_rank if alpha is not None else 1.0
            if not math.isfinite(scale) or not down.is_floating_point() or not up.is_floating_point():
                raise ValueError(f'{prefix}: requires finite alpha and floating-point factors')
            d, u = down.float().flatten(1), up.float().reshape(up.shape[0], old_rank)
            if not torch.isfinite(d).all() or not torch.isfinite(u).all():
                raise ValueError(f'{prefix}: nonfinite weights')
            qu, ru = torch.linalg.qr(u, mode='reduced')
            qd, rd = torch.linalg.qr(d.T, mode='reduced')
            left, singular, right = torch.linalg.svd(ru @ rd.T, full_matrices=False)
            squared = singular.double().square()
            energy = float(squared.sum())
            available = min(old_rank, len(singular))
            needed = available
            if method == 'sv_fro':
                # sd-scripts uses a Frobenius NORM ratio, not squared energy:
                # 0.99 therefore requests 0.9801 of cumulative squared singular values.
                if singular[0] <= 1e-6 or scale == 0:
                    needed = 1
                else:
                    cumulative = squared.cumsum(0) / energy
                    # Compare norm ratios directly so e.g. 4/5 meets 0.8 rather
                    # than overshooting because Python rounds 0.8**2 upward.
                    needed = min(available, int(torch.searchsorted(cumulative.sqrt(), threshold)) + 1)
            k = min(needed, rank if rank is not None else available)
            # Bake the old scale into the up factor; new alpha=rank gives scale 1.
            new_up = (qu @ left[:, :k]) * (singular[:k] * scale)
            new_down = right[:k] @ qd.T
            result[key] = new_down.reshape(k, *down.shape[1:]).to(down.dtype).contiguous()
            result[prefix + us] = new_up.reshape(up.shape[0], k, *up.shape[2:]).to(up.dtype).contiguous()
            result[prefix + '.alpha'] = torch.tensor(float(k))
            retained = min(1.0, float(squared[:k].sum()) / energy) if energy and scale else 1.0
            reports[prefix] = dict(old_rank=old_rank, rank=k, retained_energy=retained,
                                   retained_fro=math.sqrt(retained),
                                   rank_cap_limited=(method == 'sv_fro' and k < needed))
            if progress:
                progress(index + 1, len(modules), prefix)
    if fused:
        result, _ = fuse_qkv(result, adapter=True)
    return result, dict(target_rank=rank, selection_method=method,
                        fro_threshold=threshold if method == 'sv_fro' else None,
                        capped_modules=sum(v['rank_cap_limited'] for v in reports.values()),
                        modules=reports, fused_qkv=fused,
                        method='factor-QR/SVD; alpha baked into up, output alpha=rank')


def output_suffix(rank=None, method='fixed', threshold=0.99):
    if method == 'sv_fro':
        return f'-sv_fro{threshold:g}' + (f'-maxrank{rank}' if rank is not None else '')
    return f'-rank{8 if rank is None else rank}'


def report_summary(report):
    values = report['modules'].values()
    worst_fro = min(v['retained_fro'] for v in values)
    worst_energy = min(v['retained_energy'] for v in values)
    ranks = [v['rank'] for v in values]
    summary = (f'Ranks per projection: {min(ranks)}–{max(ranks)}. '
               f'Lowest retained Frobenius norm: {worst_fro:.2%}; squared energy: {worst_energy:.2%} '
               '(before output rounding).')
    if report['capped_modules']:
        summary += (f" WARNING: rank cap prevents reaching the requested retention in "
                    f"{report['capped_modules']} module(s).")
    return summary


def resize_file(source, destination=None, rank=None, progress=None, *, method='fixed', threshold=0.99):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    source = Path(source).resolve()
    destination = (Path(destination).resolve() if destination else
                   source.with_name(source.stem + output_suffix(rank, method, threshold) + '.safetensors'))
    if destination == source or destination.exists():
        raise ValueError('Choose a new output filename; existing files are never overwritten')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with safe_open(source, framework='pt', device='cpu') as reader:
        metadata = dict(reader.metadata() or {})
    state = load_file(source, device='cpu')
    input_bytes = sum(t.numel() * t.element_size() for t in state.values())
    resized, report = resize_state(state, rank, progress, method=method, threshold=threshold)
    report.update(source=source.name, input_tensor_bytes=input_bytes,
                  output_tensor_bytes=sum(t.numel() * t.element_size() for t in resized.values()))
    metadata['bluflow_rank_resize'] = json.dumps(report)
    for key in ('sshs_model_hash', 'sshs_legacy_hash', 'ss_network_dim', 'ss_network_alpha'):
        metadata.pop(key, None)
    ranks = {v['rank'] for v in report['modules'].values()}
    if len(ranks) == 1 and not report['fused_qkv']:
        metadata['ss_network_dim'] = metadata['ss_network_alpha'] = str(next(iter(ranks)))
    if report['fused_qkv']:
        metadata['moss_qkv_format'] = 'fused'
        metadata.pop('moss_qkv_conversion', None)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        save_file(resized, temporary, metadata=metadata)
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination, report


def gui():
    from PySide6 import QtCore, QtWidgets

    class Worker(QtCore.QThread):
        progress = QtCore.Signal(str)
        result = QtCore.Signal(str)

        def __init__(self, source, target, rank, method, threshold, parent):
            super().__init__(parent)
            self.source, self.target, self.rank = source, target, rank
            self.method, self.threshold = method, threshold

        def run(self):
            try:
                out, report = resize_file(self.source, self.target, self.rank,
                    lambda i, n, name: self.progress.emit(f'{i}/{n}: {name}'),
                    method=self.method, threshold=self.threshold)
                self.result.emit(f'Saved {out}\nTensor size: {report["input_tensor_bytes"]/2**20:.1f}'
                                 f' → {report["output_tensor_bytes"]/2**20:.1f} MiB.'
                                 f'\n{report_summary(report)}')
            except Exception as exc:
                self.result.emit(f'Resize failed: {exc}')

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.worker = None
            self.setWindowTitle('LoRA rank resize')
            layout = QtWidgets.QVBoxLayout(self)
            note = QtWidgets.QLabel('Reduce rank with SVD. Lower ranks are an approximation and may lose style or detail.\n'
                'MOSS joint Q/K/V are resized separately, then returned to the original layout.\n'
                'Rank is a cap per projection; it does not increase smaller modules. Alpha scaling is preserved.\n'
                'CPU only. Plain LoRA/LoCon supported; DoRA, LoKr and LoHa are not supported.')
            note.setWordWrap(True); layout.addWidget(note)
            row = QtWidgets.QHBoxLayout()
            self.path = QtWidgets.QLineEdit()
            browse = QtWidgets.QPushButton('Choose LoRA…')
            row.addWidget(self.path); row.addWidget(browse); layout.addLayout(row)
            self.rank = QtWidgets.QSpinBox(); self.rank.setRange(1, 16384); self.rank.setValue(8)
            self.method = QtWidgets.QComboBox()
            self.method.addItem('sv_fro — retain Frobenius norm', 'sv_fro')
            self.method.addItem('Fixed rank', 'fixed')
            self.threshold = QtWidgets.QDoubleSpinBox()
            self.threshold.setDecimals(6); self.threshold.setRange(0.000001, 1.0)
            self.threshold.setSingleStep(0.001); self.threshold.setValue(0.99)
            self.threshold.setToolTip('sd-scripts convention: 0.99 retains at least 98.01% of squared energy, unless limited by the rank cap. This does not guarantee visual quality.')
            self.cap = QtWidgets.QCheckBox('Limit rank (may prevent reaching retention target)')
            form = QtWidgets.QFormLayout()
            form.addRow('Method', self.method)
            form.addRow('Retained Frobenius norm', self.threshold)
            form.addRow('', self.cap)
            form.addRow('Maximum rank per projection', self.rank)
            layout.addLayout(form)
            def update_controls():
                dynamic = self.method.currentData() == 'sv_fro'
                self.threshold.setEnabled(dynamic); self.cap.setEnabled(dynamic)
                self.rank.setEnabled(not dynamic or self.cap.isChecked())
            self.method.currentIndexChanged.connect(update_controls)
            self.cap.toggled.connect(update_controls)
            update_controls()
            self.save = QtWidgets.QPushButton('Save resized copy…'); layout.addWidget(self.save)
            self.status = QtWidgets.QLabel('Original file is preserved.'); self.status.setWordWrap(True); layout.addWidget(self.status)
            browse.clicked.connect(self.pick); self.save.clicked.connect(self.write)
            self.resize(740, 410)

        def pick(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose LoRA', self.path.text(), 'Safetensors (*.safetensors)')
            if path: self.path.setText(path)

        def write(self):
            source = Path(self.path.text())
            if not source.is_file():
                self.status.setText('Choose an existing LoRA file.'); return
            method = self.method.currentData()
            threshold = self.threshold.value()
            rank = self.rank.value() if method == 'fixed' or self.cap.isChecked() else None
            target, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save a new resized copy',
                str(source.with_name(source.stem + output_suffix(rank, method, threshold) + '.safetensors')), 'Safetensors (*.safetensors)')
            if not target: return
            self.save.setEnabled(False)
            self.worker = Worker(str(source), target, rank, method, threshold, self)
            self.worker.progress.connect(self.status.setText)
            self.worker.result.connect(self.status.setText)
            self.worker.finished.connect(lambda: self.save.setEnabled(True))
            self.worker.start()

        def closeEvent(self, event):
            if self.worker and self.worker.isRunning():
                self.status.setText('Wait for the current resize to finish before closing.'); event.ignore()
            else: event.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from trainer.gui.widgets import STYLESHEET
    app.setStyleSheet(STYLESHEET)
    window = Window(); window.show()
    return app.exec()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path)
    parser.add_argument('--rank', type=int, default=None,
                        help='Maximum rank per projection; default 8 for fixed, no extra cap for sv_fro')
    parser.add_argument('--method', choices=('fixed', 'sv_fro'), default='fixed')
    parser.add_argument('--threshold', type=float, default=0.99,
                        help='sv_fro retained norm target in (0,1], default 0.99 (98.01%% squared energy)')
    parser.add_argument('--output', '-o', type=Path)
    args = parser.parse_args(argv)
    if args.input is None: return gui()
    try:
        out, report = resize_file(args.input, args.output, args.rank,
                             lambda i, n, name: print(f'{i}/{n}: {name}', flush=True),
                             method=args.method, threshold=args.threshold)
    except Exception as exc:
        parser.exit(1, f'Resize failed: {exc}\n')
    print(f'Saved {out}')
    print(report_summary(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
