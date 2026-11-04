from types import SimpleNamespace

import pytest

from trainer.training import allocator


@pytest.mark.parametrize("platform,expected", [
    ("linux", {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ("win32", {}), ("darwin", {}),
])
def test_platform_allocator_default(monkeypatch, platform, expected):
    env = {}
    monkeypatch.setattr(allocator.os, "environ", env)
    monkeypatch.setattr(allocator, "sys", SimpleNamespace(platform=platform))
    allocator.configure_allocator()
    allocator.configure_allocator()
    assert env == expected


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("key", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
@pytest.mark.parametrize("value", ["", "backend:cudaMallocAsync", "expandable_segments:False"])
def test_user_allocator_settings_are_preserved(monkeypatch, platform, key, value):
    env = {key: value}
    monkeypatch.setattr(allocator.os, "environ", env)
    monkeypatch.setattr(allocator, "sys", SimpleNamespace(platform=platform))
    allocator.configure_allocator()
    assert env == {key: value}
