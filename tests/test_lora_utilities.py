from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file, load_file

from trainer.modeling.moss_export import fuse_qkv, separate_adapter_qkv
from trainer.tools.lora_resize import resize_state, resize_file


def factors(prefix='layer', rank=4, conv=False):
    torch.manual_seed(17)
    down = torch.randn(rank, 9, dtype=torch.float32)
    up = torch.randn(6, rank, dtype=torch.float32)
    if conv:
        down = down.reshape(rank, 1, 3, 3)
        up = up.reshape(6, rank, 1, 1)
    return {prefix+'.lora_down.weight': down, prefix+'.lora_up.weight': up,
            prefix+'.alpha': torch.tensor(rank / 2)}


def delta(state, prefix):
    d = state[prefix+'.lora_down.weight']
    u = state[prefix+'.lora_up.weight']
    return u.float().reshape(u.shape[0], -1) @ d.float().flatten(1) * (state[prefix+'.alpha'].item() / d.shape[0])


@pytest.mark.parametrize('conv', [False, True])
def test_resize_matches_dense_svd_and_scaling(conv):
    state = factors(conv=conv)
    weight = delta(state, 'layer')
    u, s, v = torch.linalg.svd(weight, full_matrices=False)
    resized, report = resize_state(state, 2)
    torch.testing.assert_close(delta(resized, 'layer'), (u[:, :2]*s[:2]) @ v[:2], atol=1e-5, rtol=1e-5)
    assert resized['layer.alpha'].item() == 2
    assert report['modules']['layer']['rank'] == 2
    retained, _ = resize_state(state, 16)
    torch.testing.assert_close(delta(retained, 'layer'), weight, atol=1e-5, rtol=1e-5)
    assert retained['layer.lora_down.weight'].shape[0] == 4


def test_compact_split_restores_factors_and_smaller_size():
    original = {}
    for name in ('q', 'k', 'v'):
        original.update(factors('diffusion_model.layers.0.proj.'+name))
    fused, _ = fuse_qkv(original, adapter=True)
    separated, _ = separate_adapter_qkv(fused, compact=True)
    for name in ('q', 'k', 'v'):
        prefix = 'diffusion_model.layers.0.proj.'+name
        assert separated[prefix+'.lora_down.weight'].shape[0] == 4
        torch.testing.assert_close(delta(separated, prefix), delta(original, prefix), rtol=0, atol=0)
    assert sum(t.numel() for t in separated.values()) < sum(t.numel() for t in fused.values())
    resized, _ = resize_state(fused, 2)
    assert resized['diffusion_model.layers.0.proj.qkv.lora_down.weight'].shape[0] == 6
    split, _ = separate_adapter_qkv(resized, compact=True)
    for name in ('q', 'k', 'v'):
        prefix = 'diffusion_model.layers.0.proj.'+name
        expected, _ = resize_state({k: v for k,v in original.items() if k.startswith(prefix+'.')}, 2)
        torch.testing.assert_close(delta(split, prefix), delta(expected, prefix))


def test_dense_fused_split_retains_all_columns():
    state = factors('diffusion_model.layers.0.proj.qkv')
    split, _ = separate_adapter_qkv(state, compact=True)
    combined = torch.cat([delta(split, 'diffusion_model.layers.0.proj.'+p) for p in ('q','k','v')])
    torch.testing.assert_close(combined, delta(state, 'diffusion_model.layers.0.proj.qkv'))
    assert all(split['diffusion_model.layers.0.proj.'+p+'.lora_down.weight'].shape[0] == 4 for p in ('q','k','v'))


def test_resize_rejects_unsupported_and_preserves_source(tmp_path):
    state = factors()
    source = tmp_path/'input.safetensors'
    save_file(state, source, metadata={'training_config': 'history'})
    original = source.read_bytes()
    out, _ = resize_file(source, rank=2)
    assert source.read_bytes() == original
    assert load_file(out)['layer.lora_down.weight'].shape[0] == 2
    with pytest.raises(ValueError, match='never overwritten'):
        resize_file(source, rank=2)
    with pytest.raises(ValueError, match='never overwritten'):
        resize_file(source, source, rank=2)
    for name in ('layer.dora_scale', 'layer.lokr_w1', 'layer.lora_mid.weight'):
        with pytest.raises(ValueError, match='unsupported'):
            resize_state(state | {name: torch.ones(2)}, 2)


def test_bundled_alpha_editor_preserves_factors(tmp_path):
    from trainer.tools.lora_alpha_edit import LoraFile
    source = tmp_path/'input.safetensors'; out = tmp_path/'scaled.safetensors'
    state = factors()
    save_file(state, source)
    editor = LoraFile(source)
    editor.save_as(out, new_alpha_for=lambda name, alpha, rank: alpha*2)
    saved = load_file(out)
    torch.testing.assert_close(delta(saved, 'layer'), delta(state, 'layer')*2)
    assert torch.equal(saved['layer.lora_down.weight'], state['layer.lora_down.weight'])


def diagonal_factors(prefix, spectrum):
    rank = len(spectrum)
    return {prefix+'.lora_down.weight': torch.eye(rank),
            prefix+'.lora_up.weight': torch.diag(torch.tensor(spectrum, dtype=torch.float32)),
            prefix+'.alpha': torch.tensor(rank / 2)}


@pytest.mark.parametrize('threshold,expected', [(0.8, 1), (0.9, 2), (1.0, 2)])
def test_sv_fro_uses_norm_not_energy(threshold, expected):
    state = diagonal_factors('layer', [4, 3])
    resized, report = resize_state(state, method='sv_fro', threshold=threshold)
    module = report['modules']['layer']
    assert module['rank'] == expected
    assert not module['rank_cap_limited']
    # One component retains 4/5 of the norm, but only 16/25 of the energy.
    assert module['retained_fro'] == pytest.approx(0.8 if expected == 1 else 1)
    assert module['retained_energy'] == pytest.approx(0.64 if expected == 1 else 1)
    assert torch.linalg.vector_norm(delta(resized, 'layer')) / torch.linalg.vector_norm(delta(state, 'layer')) == pytest.approx(module['retained_fro'])


def test_sv_fro_reports_cap_and_handles_zero_updates():
    from trainer.tools.lora_resize import report_summary
    state = diagonal_factors('layer', [1, 1, 1, 1])
    _, report = resize_state(state, 2, method='sv_fro', threshold=0.99)
    assert report['modules']['layer']['rank'] == 2
    assert report['capped_modules'] == 1
    assert 'rank cap prevents' in report_summary(report)
    for spectrum, alpha in (([0, 0, 0], 1.5), ([4, 3, 2], 0.0)):
        state = diagonal_factors('layer', spectrum)
        state['layer.alpha'] = torch.tensor(alpha)
        resized, report = resize_state(state, method='sv_fro')
        assert report['modules']['layer']['rank'] == 1
        assert report['modules']['layer']['retained_fro'] == 1
        assert torch.count_nonzero(delta(resized, 'layer')) == 0


def test_sv_fro_fused_qkv_can_select_different_ranks():
    state = {}
    for p, spectrum in zip(('q', 'k', 'v'), ([5, 0, 0], [4, 3, 0], [3, 3, 3])):
        state.update(diagonal_factors('diffusion_model.layers.0.proj.'+p, spectrum))
    fused, _ = fuse_qkv(state, adapter=True)
    resized, report = resize_state(fused, method='sv_fro', threshold=0.9)
    assert [report['modules']['diffusion_model.layers.0.proj.'+p]['rank'] for p in ('q','k','v')] == [1, 2, 3]
    assert resized['diffusion_model.layers.0.proj.qkv.lora_down.weight'].shape[0] == 6
    split, _ = separate_adapter_qkv(resized, compact=True)
    for p in ('q','k','v'):
        prefix = 'diffusion_model.layers.0.proj.'+p
        torch.testing.assert_close(delta(split, prefix), delta(state, prefix))


@pytest.mark.parametrize('threshold', [0, -0.1, 1.1, float('nan'), float('inf')])
def test_sv_fro_rejects_invalid_targets(threshold):
    with pytest.raises(ValueError, match='threshold'):
        resize_state(factors(), method='sv_fro', threshold=threshold)


def test_sv_fro_file_metadata_and_cli(tmp_path, capsys):
    import json
    from safetensors import safe_open
    from trainer.tools.lora_resize import main
    source = tmp_path/'original.safetensors'
    save_file(diagonal_factors('layer', [4, 3]), source)
    assert main([str(source), '--method', 'sv_fro', '--threshold', '0.9', '--rank', '1']) == 0
    assert 'rank cap prevents' in capsys.readouterr().out
    out = tmp_path/'original-sv_fro0.9-maxrank1.safetensors'
    with safe_open(out, framework='pt') as reader:
        report = json.loads(reader.metadata()['bluflow_rank_resize'])
        assert report['selection_method'] == 'sv_fro'
        assert report['fro_threshold'] == 0.9
        assert report['capped_modules'] == 1
