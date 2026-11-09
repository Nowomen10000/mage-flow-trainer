from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trainer.data.caption import CaptionConfig
from trainer.data.caption_variations import CaptionVariationCache
from trainer.data.text_cache import validate_static_captions
from trainer.modeling.moss_image import MOSSImage, MOSSParams
from trainer.training.config import load_config
from trainer.training.flow import FlowConfig, flow_loss, hf_loss, prepare_flow_batch


def test_short_curriculum_uses_step_budget_across_epochs():
    from collections import Counter
    from trainer.data.dataset import BucketBatchSampler
    from trainer.training.curriculum import Curriculum, Phase

    curriculum = Curriculum([Phase(at=0), Phase(at=0.5, mode='texture'),
                             Phase(at=0.7), Phase(at=0.9, mode='texture')])
    sampler = BucketBatchSampler.__new__(BucketBatchSampler)
    sampler._batches = lambda: [[i] for i in range(81)]
    sampler.set_curriculum(curriculum, 10, total_steps=180)
    counts = Counter()
    for step in range(180):
        sampler.epoch, ordinal = divmod(step, 81)
        phase = sampler._phase_for(ordinal)
        assert phase is curriculum.resolve(step / 180)
        counts[phase.at] += 1
    assert counts == {0: 90, 0.5: 36, 0.7: 36, 0.9: 18}


def test_curriculum_accumulation_and_partial_epoch_group():
    from trainer.data.dataset import BucketBatchSampler
    from trainer.training.curriculum import Curriculum, Phase

    curriculum = Curriculum([Phase(at=0), Phase(at=0.5, mode='texture')])
    sampler = BucketBatchSampler.__new__(BucketBatchSampler)
    sampler._batches = lambda: [[i] for i in range(5)]
    sampler.set_curriculum(curriculum, 2, gradient_accumulation_steps=2)
    # Six updates: [2,2,1] microbatches in each epoch. All of epoch two is texture.
    for epoch, expected in enumerate(('fullres', 'texture')):
        sampler.epoch = epoch
        assert [sampler._phase_for(i).mode for i in range(5)] == [expected] * 5
    # A boundary inside the epoch must not split an accumulation group.
    sampler.set_curriculum(curriculum, 2, total_steps=2, gradient_accumulation_steps=2)
    sampler.epoch = 0
    assert [sampler._phase_for(i).mode for i in range(4)] == [
        'fullres', 'fullres', 'texture', 'texture']


def test_moss_curriculum_config_and_existing_restrictions(tmp_path):
    text = '''
[train]
model_family = "moss_image"
cache_text_embeddings = true
[dataset]
source = "encode"
path = "images"
[[curriculum]]
at = 0.0
mode = "texture"
'''
    path = tmp_path / 'texture.toml'
    path.write_text(text)
    cfg = load_config(path)
    validate_static_captions(cfg)
    assert cfg.train.flux2_vae
    assert not cfg.train.allow_multi_gpu_texture
    for invalid in (
        text.replace('moss_image', 'flux3'),
        text.replace('"encode"', '"latents"'),
        text + '\n[paired_texture]\nscale = 0.1\n',
    ):
        path.write_text(invalid)
        with pytest.raises(ValueError):
            load_config(path)


def test_texture_trigger_cached_without_changing_caption_slots(tmp_path):
    cache = CaptionVariationCache(tmp_path / 'captions.sqlite', 'encoder')
    entry = SimpleNamespace(path=Path('image.png'), tags='subject, background', nl='')
    caption = CaptionConfig()
    cache.prepare([entry], caption, 42, 3, extra_captions=['style trigger'])
    assert {text for _, text in cache.pending()} == {'', entry.tags, 'style trigger'}
    assert cache.statistics['slots'] == 3
    for key, _ in cache.pending():
        cache.add(key, torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool))
    cache.prepare([entry], caption, 42, 3, extra_captions=['style trigger'])
    assert not list(cache.pending())
    assert cache.get(['style trigger'], 'cpu', torch.float32)[0].shape == (1, 2, 4)
    cache.prepare([entry], caption, 42, 3, extra_captions=['changed trigger'])
    assert [text for _, text in cache.pending()] == ['changed trigger']
    cache.close()


def test_moss_texture_masked_forward_backward():
    # The f8 VAE packs 2x2 into channels: texture masks and MOSS both use f16 grids.
    torch.set_num_threads(2)
    model = MOSSImage(MOSSParams(dim=32, ffn_dim=48, num_heads=2,
                                num_layers=2, text_dim=16, latent_channels=8))
    model.configure_execution(False)
    clean = torch.randn(2, 8, 1, 4, 4)
    noisy, times, target = prepare_flow_batch(clean, FlowConfig(
        timestep_sample_method='moss', shift=1.0), t_range=(0.0, 0.6))
    text = (torch.randn(2, 5, 16), torch.ones(2, 5, dtype=torch.bool))
    pred = model(hidden_states=noisy, timestep=times, encoder_hidden_states=text,
                 return_dict=False)[0]
    mask = torch.zeros(2, 1, 1, 4, 4)
    mask[..., 1:3, 1:3] = 1
    mse = flow_loss(pred, target, mask=mask)
    expected = (pred - target).square()[..., 1:3, 1:3].mean()
    torch.testing.assert_close(mse, expected)
    loss = mse + 0.4 * hf_loss(pred, noisy, clean, times, 1, 1.0, mask=mask)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
