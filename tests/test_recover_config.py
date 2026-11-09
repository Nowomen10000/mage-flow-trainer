import json
import tomllib

import pytest
import torch
from safetensors.torch import save_file

from trainer.modeling.checkpoint_metadata import encode_settings
from trainer.tools.recover_config import recover_file, main
from trainer.training.config import load_config


def checkpoint(tmp_path, text, **metadata):
    original = tmp_path / 'original.toml'
    original.write_text(text, encoding='utf-8')
    cfg = load_config(original)
    source = tmp_path / 'adapter.safetensors'
    save_file({'weight': torch.ones(1)}, source,
              metadata={'training_config': encode_settings(cfg), **metadata})
    return source, cfg


def test_texture_recipe_roundtrip_and_source_preserved(tmp_path):
    source, original = checkpoint(tmp_path, '''
[train]
model_family = "moss_image"
compile = false
max_steps = 180
[adapter]
kind = "lycoris_lora"
[dataset]
path = "画像/example"
source = "encode"
[dataset.texture]
trigger = "paint"
canvases = [[832,1216], [1216,832], [1024,1024]]
[[curriculum]]
at = 0.0
mode = "fullres"
[[curriculum]]
at = 0.5
mode = "texture"
t_range = [0.0, 0.6]
lr_mul = 0.8
''')
    before = source.read_bytes()
    target, notes = recover_file(source)
    assert notes == []
    restored = load_config(target)
    assert json.loads(encode_settings(restored)) == json.loads(encode_settings(original))
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match='never overwritten'):
        recover_file(source, target)
    with pytest.raises(ValueError, match='never overwritten'):
        recover_file(source, source)


def test_staged_multires_and_subsets_roundtrip(tmp_path):
    source, original = checkpoint(tmp_path, '''
[train]
model_family = "moss_image"
max_steps = 560
pack_resolutions = true
attention_backend = "torch_varlen"
batch_size = {512 = 24, 768 = 12, 1024 = 8}
[adapter]
kind = "lora"
[dataset]
resolutions = [512,768,1024]
source = "latents"
[[dataset.subsets]]
path = "images/a"
[[dataset.subsets]]
path = "images/b"
[resolution_schedule]
enabled = true
stages = [{resolution=512,share=0.2,lr_mul=0.8}, {resolution=768,share=0.2,lr_mul=0.9}, {resolution=1024,share=0.6}]
''')
    target, notes = recover_file(source)
    assert notes == []
    raw = tomllib.loads(target.read_text())
    assert 'resolution' not in raw['dataset'] and 'path' not in raw['dataset']
    assert 'epochs' not in raw['train'] and 'save_every_epochs' not in raw['train']
    assert json.loads(encode_settings(load_config(target))) == json.loads(encode_settings(original))


def test_fresh_run_and_edited_adapter_notice(tmp_path, capsys):
    source, _ = checkpoint(tmp_path, '''
[train]
model_family = "moss_image"
[adapter]
kind = "lora"
[dataset]
path = "images"
''', bluflow_rank_resize='{}')
    from safetensors import safe_open
    with safe_open(source, framework='pt') as reader:
        metadata = dict(reader.metadata())
    config = json.loads(metadata['training_config'])
    config['train'].update(resume_from='previous.safetensors', resume_mode='weights')
    metadata['training_config'] = json.dumps(config)
    save_file({'weight': torch.ones(1)}, source, metadata=metadata)
    target = tmp_path / 'recipe.toml'
    assert main([str(source), '-o', str(target)]) == 0
    cfg = load_config(target)
    assert cfg.train.resume_from is None and cfg.train.resume_mode == 'exact'
    output = capsys.readouterr().out
    assert 'Cleared resume' in output and 'edited after training' in output


@pytest.mark.parametrize('config', [None, 'bad json', '[]', '{"train": {}}'])
def test_missing_or_invalid_metadata_rejected(tmp_path, config):
    source = tmp_path / 'legacy.safetensors'
    save_file({'weight': torch.ones(1)}, source,
              metadata={} if config is None else {'training_config': config})
    target = tmp_path / 'recovered.toml'
    with pytest.raises(ValueError):
        recover_file(source, target)
    assert not target.exists()


def test_unknown_settings_retained_and_reported(tmp_path):
    source = tmp_path / 'future.safetensors'
    save_file({'weight': torch.ones(1)}, source, metadata={'training_config': json.dumps({
        'train': {'model_family': 'moss_image'}, 'dataset': {'path': 'images'},
        'future_feature': {'value': 123}})})
    target, notes = recover_file(source)
    assert tomllib.loads(target.read_text())['future_feature']['value'] == 123
    assert any('Review before training:' in note and 'future_feature' in note for note in notes)
