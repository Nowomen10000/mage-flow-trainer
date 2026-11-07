from pathlib import Path

import pytest

pytest.importorskip("triton")
from triton import knobs
from triton.runtime.cache import get_cache_manager
from trainer.training.triton_cache import RecoveringFileCacheManager


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path))
    return RecoveringFileCacheManager("test-kernel")


@pytest.mark.parametrize("contents", [b"", b'{"child_paths":', b"\xff"])
def test_corrupt_group_is_rebuilt_without_deleting_files(manager, contents, caplog):
    child = manager.put(b"kernel bytes", "kernel.cubin")
    index = Path(manager.put(contents, "__grp__kernel.json"))
    assert manager.get_group("kernel.json") is None
    assert "rebuilding the kernel" in caplog.text
    assert index.read_bytes() == contents
    assert Path(child).read_bytes() == b"kernel bytes"
    manager.put_group("kernel.json", {"kernel.cubin": child})
    assert manager.get_group("kernel.json") == {"kernel.cubin": child}


def test_valid_and_missing_groups_keep_upstream_behavior(manager, caplog):
    assert manager.get_group("absent.json") is None
    child = manager.put(b"kernel", "kernel.cubin")
    manager.put_group("kernel.json", {"kernel.cubin": child, "missing": child + ".absent"})
    assert manager.get_group("kernel.json") == {"kernel.cubin": child}
    assert not caplog.records


def test_io_errors_still_propagate(manager, monkeypatch):
    from triton.runtime.cache import FileCacheManager

    def denied(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(FileCacheManager, "get_group", denied)
    with pytest.raises(PermissionError):
        manager.get_group("kernel.json")


def test_triton_loads_manager_from_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path))
    monkeypatch.setenv("TRITON_CACHE_MANAGER",
                       "trainer.training.triton_cache:RecoveringFileCacheManager")
    # Clear a previously cached knob value so this exercises Triton's env loader.
    knobs.cache.__dict__.pop("manager_class", None)
    manager = get_cache_manager("ab" * 32)
    assert isinstance(manager, RecoveringFileCacheManager)
    manager.put(b"", "__grp__kernel.json")
    assert manager.get_group("kernel.json") is None
