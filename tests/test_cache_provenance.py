import json
from types import SimpleNamespace as NS
import pytest
import torch
from safetensors.torch import save_file
from trainer.data.cache_provenance import vae_identity, validate_cache, validate_dataset_caches, atomic_save


def test_content_identity_move_change_and_mode(tmp_path):
    vae = tmp_path / 'vae.safetensors'
    vae.write_bytes(b'weights-a')
    expected = vae_identity('', vae, 'moss_image')
    moved = tmp_path / 'moved.safetensors'
    moved.write_bytes(vae.read_bytes())
    assert expected == vae_identity('', moved, 'moss_image')
    cache = tmp_path / 'cache.safetensors'
    atomic_save({'latents': torch.ones(1)}, cache, {'cache_provenance': json.dumps(expected)})
    validate_cache(cache, expected, 'strict')
    vae.write_bytes(b'weights-b')
    with pytest.raises(ValueError, match='vae_sha256 mismatch'):
        validate_cache(cache, vae_identity('', vae, 'moss_image'))
    with pytest.raises(ValueError, match='normalization mismatch'):
        validate_cache(cache, vae_identity('', moved, 'mage_flow'))


def test_legacy_policy_and_unavailable_vae(tmp_path):
    cache = tmp_path / 'cache.safetensors'
    save_file({'latents': torch.ones(1)}, cache)
    with pytest.warns(UserWarning, match='unverified'):
        validate_cache(cache, {}, 'warn')
    with pytest.raises(ValueError, match='unverified'):
        validate_cache(cache, {}, 'strict')
    validate_cache(cache, {}, 'off')
    expected = vae_identity('', tmp_path / 'missing', 'moss_image')
    known = dict(expected, vae_sha256='abc')
    atomic_save({'latents': torch.ones(1)}, cache, {'cache_provenance': json.dumps(known)})
    with pytest.warns(UserWarning):
        validate_cache(cache, expected)
    with pytest.raises(ValueError, match='unverified'):
        validate_cache(cache, expected, 'strict')


def test_source_change_and_mixed_cache_rejected(tmp_path):
    from trainer.data.cache_provenance import file_sha256
    source = tmp_path / 'image.png'
    source.write_bytes(b'first')
    expected = vae_identity('', tmp_path / 'missing', 'moss_image')
    paths = [tmp_path / 'a.safetensors', tmp_path / 'b.safetensors']
    for path, digest in zip(paths, ('a', 'b')):
        atomic_save({'latents': torch.ones(1)}, path, {
            'cache_provenance': json.dumps(dict(expected, vae_sha256=digest)),
            'source_sha256': file_sha256(source)})
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source image content changed'):
        validate_cache(paths[0], source=source)
    data = NS(require_cache=True, entries=[NS(latent_path=p) for p in paths])
    train = NS(model_path='', vae_path=str(tmp_path / 'missing'), model_family='moss_image', flux2_vae=False)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match='different VAEs'):
        validate_dataset_caches(data, train)


def test_writer_records_actual_vae_and_source_and_refuses_reuse(tmp_path):
    from PIL import Image
    from trainer.data.cache import LatentCacher, read_cache_metadata
    from trainer.data.bucket import BucketManager
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def encode(self, x):
            return torch.zeros(len(x), 128, x.shape[-2] // 16, x.shape[-1] // 16)
    image = tmp_path / 'image.png'
    Image.new('RGB', (256, 256)).save(image)
    provenance = dict(vae_sha256='test-vae', normalization='mage-native')
    cacher = LatentCacher(VAE(), device='cpu', provenance=provenance, cache_validation='strict')
    from trainer.data.dataset import DatasetConfig
    manager = DatasetConfig(path=str(tmp_path), resolution=256, min_bucket_reso=256).build_bucket_manager(256)
    path, _ = cacher.cache_image(image, manager)
    meta = read_cache_metadata(path)
    assert json.loads(meta['cache_provenance'])['encode_dtype'] == 'float32'
    assert meta['source_sha256']
    cacher.cache_image(image, manager)
    cacher.provenance['vae_sha256'] = 'different'
    with pytest.raises(ValueError, match='mismatch'):
        cacher.cache_batch([image], manager)
    cacher.cache_image(image, manager, overwrite=True)
    assert json.loads(read_cache_metadata(path)['cache_provenance'])['vae_sha256'] == 'different'


def test_atomic_save_keeps_open_handle_and_roundtrips(tmp_path, monkeypatch):
    import safetensors.torch
    from safetensors import safe_open
    folder = tmp_path / 'cache with spaces ü'
    folder.mkdir()
    path = folder / 'image.safetensors'
    # Do not reopen the temporary path with the Rust file writer.
    monkeypatch.setattr(safetensors.torch, 'save_file', lambda *a, **kw: pytest.fail('path reopened'))
    value = torch.arange(12).reshape(3, 4)
    atomic_save({'latents': value}, path, {'source': 'test'})
    assert torch.equal(safetensors.torch.load_file(path)['latents'], value)
    with safe_open(path, framework='pt') as reader:
        assert reader.metadata() == {'source': 'test'}
    assert list(folder.iterdir()) == [path]


def test_atomic_save_retries_windows_replace_without_deleting_destination(tmp_path, monkeypatch):
    from trainer.data import cache_provenance as cp
    path = tmp_path / 'cache.safetensors'
    path.write_bytes(b'original')
    replace = cp.os.replace
    attempts, sleeps = [], []
    def transient(source, target):
        assert path.read_bytes() == b'original'
        attempts.append(source)
        if len(attempts) < 3:
            raise PermissionError(13, 'Access is denied')
        replace(source, target)
    monkeypatch.setattr(cp, '_WINDOWS', True)
    monkeypatch.setattr(cp.os, 'replace', transient)
    monkeypatch.setattr(cp.time, 'sleep', sleeps.append)
    cp.atomic_save({'latents': torch.ones(1)}, path, {})
    from safetensors.torch import load_file
    assert torch.equal(load_file(path)['latents'], torch.ones(1))
    assert len(attempts) == 3 and sleeps == [.05, .1]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('windows,errno,attempts', [(True,13,6), (False,13,1), (True,28,1)])
def test_atomic_save_persistent_errors_preserve_old_cache(tmp_path, monkeypatch, windows, errno, attempts):
    from trainer.data import cache_provenance as cp
    path = tmp_path / 'cache.safetensors'
    path.write_bytes(b'original')
    calls, sleeps = [], []
    def denied(*args):
        calls.append(args)
        raise OSError(errno, 'simulated write failure')
    monkeypatch.setattr(cp, '_WINDOWS', windows)
    monkeypatch.setattr(cp.os, 'replace', denied)
    monkeypatch.setattr(cp.time, 'sleep', sleeps.append)
    with pytest.raises(OSError, match='Could not write latent cache') as caught:
        cp.atomic_save({'latents': torch.ones(1)}, path, {})
    assert caught.value.errno == errno and caught.value.filename == str(path)
    assert 'without overwrite' in str(caught.value)
    assert len(calls) == attempts and len(sleeps) == attempts - 1
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_save_serialization_error_leaves_no_partial_files(tmp_path, monkeypatch):
    import safetensors.torch
    path = tmp_path / 'cache.safetensors'
    path.write_bytes(b'original')
    def invalid(*a, **kw):
        raise ValueError('invalid tensor')
    monkeypatch.setattr(safetensors.torch, 'save', invalid)
    with pytest.raises(ValueError, match='invalid tensor'):
        atomic_save({'latents': torch.ones(1)}, path, {})
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_save_partial_write_preserves_destination_and_closes_handle(tmp_path, monkeypatch):
    from trainer.data import cache_provenance as cp
    path = tmp_path / 'cache.safetensors'
    path.write_bytes(b'original')
    fdopen = cp.os.fdopen
    streams = []
    class BrokenWriter:
        def __init__(self, fd, mode):
            self.stream = fdopen(fd, mode)
            streams.append(self.stream)
        def __enter__(self):
            return self
        def write(self, payload):
            self.stream.write(payload[:8])
            raise OSError(28, 'No space left on device')
        def __exit__(self, *args):
            self.stream.close()
    monkeypatch.setattr(cp.os, 'fdopen', BrokenWriter)
    monkeypatch.setattr(cp.os, 'replace', lambda *args: pytest.fail('published partial file'))
    with pytest.raises(OSError, match='No space left on device') as caught:
        cp.atomic_save({'latents': torch.ones(1)}, path, {})
    assert caught.value.filename == str(path)
    assert streams and all(stream.closed for stream in streams)
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_save_cleanup_failure_preserves_original_error(tmp_path, monkeypatch):
    from trainer.data import cache_provenance as cp
    path = tmp_path / 'cache.safetensors'
    path.write_bytes(b'original')
    def replace_failure(*args):
        raise OSError(28, 'original disk error')
    def cleanup_failure(*args, **kwargs):
        raise PermissionError(13, 'cleanup locked')
    monkeypatch.setattr(cp.os, 'replace', replace_failure)
    monkeypatch.setattr(cp.Path, 'unlink', cleanup_failure)
    with pytest.raises(OSError, match='original disk error') as caught:
        cp.atomic_save({'latents': torch.ones(1)}, path, {})
    assert caught.value.errno == 28
    assert path.read_bytes() == b'original'
