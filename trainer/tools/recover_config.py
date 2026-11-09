"""Recover a BluFlow training TOML from safetensors metadata, without loading weights."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import toml
import tomllib
from safetensors import safe_open


def recover_settings(source):
    with safe_open(str(source), framework='pt', device='cpu') as reader:
        metadata = reader.metadata() or {}
    try:
        config = json.loads(metadata['training_config'])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('No readable BluFlow training_config metadata found; '
                         'this file cannot recover a complete training recipe.') from exc
    if not isinstance(config, dict) or not all(
            isinstance(config.get(key), dict) for key in ('train', 'dataset')):
        raise ValueError('training_config must contain train and dataset sections.')
    config = copy.deepcopy(config)
    notes = []
    curriculum = config.get('curriculum', [])
    if isinstance(curriculum, dict) and set(curriculum) == {'phases'}:
        curriculum = curriculum['phases']
    if not isinstance(curriculum, list) or not all(isinstance(p, dict) for p in curriculum):
        raise ValueError('Unrecognized curriculum metadata; expected a list of phases.')
    config['curriculum'] = curriculum
    train, dataset = config['train'], config['dataset']
    if train.get('resume_from') or train.get('resume_mode', 'exact') != 'exact':
        notes.append('Cleared resume settings: this TOML starts a fresh training run.')
    train.pop('resume_from', None)
    train['resume_mode'] = 'exact'
    # LyCORIS defaults to compile when the key is omitted. Preserve explicitly off.
    if 'compile' in train and train['compile'] is None:
        train['compile'] = False
    # Resolved metadata includes both the active setting and inactive defaults.
    if dataset.get('resolutions'):
        dataset.pop('resolution', None)
    else:
        dataset.pop('resolutions', None)
    if dataset.get('subsets'):
        dataset.pop('path', None)
    else:
        dataset.pop('subsets', None)
    if config.get('resolution_schedule', {}).get('enabled'):
        train.pop('epochs', None)
        train.pop('save_every_epochs', None)
    if any(key in metadata for key in (
            'bluflow_rank_resize', 'inference_conversion', 'ss_dora_baked_weight')):
        notes.append('This adapter was edited after training. Recovered settings describe '
                     'the original training recipe, not necessarily its current tensors.')
    return config, notes


def recover_file(source, destination=None):
    source = Path(source).resolve()
    target = (Path(destination).resolve() if destination else
              source.with_name(source.stem + '-recovered.toml'))
    if target == source or target.exists():
        raise ValueError('Choose a new output filename; existing files are never overwritten.')
    config, notes = recover_settings(source)
    # Keep saved defaults instead of pruning against today's defaults. None is
    # unset in TOML; unknown fields are retained and reported by loader validation.
    text = ('# Recovered BluFlow training recipe. Review local paths and output naming.\n'
            '# Starts fresh; optimizer state and GPU selection are not restored.\n'
            + toml.dumps(config))
    tomllib.loads(text)  # Never publish syntactically invalid TOML.
    from trainer.training.config import load_config
    with tempfile.TemporaryDirectory(prefix='bluflow-recover-') as folder:
        check = Path(folder) / 'config.toml'
        check.write_text(text, encoding='utf-8')
        try:
            load_config(check)
        except (ValueError, TypeError, OSError) as exc:
            notes.append(f'Review before training: {exc}')
    # Exclusive creation also protects against a destination appearing after the check.
    with target.open('x', encoding='utf-8') as stream:
        stream.write(text)
    return target, notes


def gui():
    from PySide6 import QtWidgets
    from trainer.gui.widgets import STYLESHEET
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(STYLESHEET)
    window = QtWidgets.QWidget()
    window.setWindowTitle('Recover training TOML')
    layout = QtWidgets.QVBoxLayout(window)
    info = QtWidgets.QLabel(
        'Recover the original training recipe from a BluFlow safetensors file.\n'
        'Includes texture curriculum and dataset settings. Review local paths, output naming\n'
        'and GPU selection before training. Resume instructions are cleared for a fresh run.')
    info.setWordWrap(True)
    layout.addWidget(info)
    row = QtWidgets.QHBoxLayout()
    source = QtWidgets.QLineEdit()
    row.addWidget(source)
    browse = QtWidgets.QPushButton('Choose checkpoint…')
    row.addWidget(browse)
    layout.addLayout(row)
    save = QtWidgets.QPushButton('Save recovered TOML…')
    layout.addWidget(save)
    status = QtWidgets.QLabel('The checkpoint and existing TOMLs are preserved.')
    status.setWordWrap(True)
    layout.addWidget(status)

    def pick():
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            window, 'Choose checkpoint', source.text(), 'Safetensors (*.safetensors)')
        if selected:
            source.setText(selected)

    def write():
        path = Path(source.text())
        if not path.is_file():
            status.setText('Choose an existing safetensors file.')
            return
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            window, 'Save recovered TOML', str(path.with_name(path.stem + '-recovered.toml')),
            'TOML configs (*.toml)')
        if not selected:
            return
        try:
            target, notes = recover_file(path, selected)
            status.setText(f'Saved {target}\nOpen it with the trainer’s Open button.\n' + '\n'.join(notes))
        except Exception as exc:
            status.setText(f'Recovery failed: {exc}')

    browse.clicked.connect(pick)
    save.clicked.connect(write)
    window.resize(680, 240)
    window.show()
    return app.exec()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path)
    parser.add_argument('--output', '-o', type=Path)
    args = parser.parse_args(argv)
    if args.input is None:
        return gui()
    try:
        target, notes = recover_file(args.input, args.output)
    except Exception as exc:
        parser.exit(1, f'Recovery failed: {exc}\n')
    print(f'Saved {target}')
    for note in notes:
        print(note)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
