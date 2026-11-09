import toml
import tomllib

from trainer.gui import bridge


def test_read_existing_inline_texture_canvases(tmp_path):
    text = '''[dataset]
path = "images"
texture = { canvases = [ [ 832, 1216,], [ 1216, 832,], [ 1024, 1024,],] }
'''
    path = tmp_path / 'texture.toml'
    path.write_text(text, encoding='utf-8')
    assert bridge.read_toml(path) == tomllib.loads(text)


def test_texture_canvases_roundtrip_without_inline_section(tmp_path):
    values = bridge.defaults() | {
        'dataset.path': 'images',
        'dataset.texture.canvases': [[832, 1216], [1216, 832], [1024, 1024]],
        'dataset.caption.mixed_weights': {'tags': 70, 'nl': 30},
        'train.batch_size': {512: 24, 1024: 7},
    }
    path = tmp_path / 'texture.toml'
    bridge.write_toml(path, values)
    text = path.read_text(encoding='utf-8')
    assert '[dataset.texture]' in text
    assert 'texture = {' not in text
    # Existing GUIs using the old parser can also read newly saved configs.
    assert toml.loads(text) == tomllib.loads(text) == bridge.read_toml(path)
    restored = bridge.flatten(bridge.read_toml(path))
    assert restored['dataset.texture.canvases'] == values['dataset.texture.canvases']
    assert restored['dataset.caption.mixed_weights'] == values['dataset.caption.mixed_weights']
    assert 'batch_size = {' in text
