import os

import pytest
import torch

from trainer.training import attention_check


def test_sdpa_does_not_probe_varlen(monkeypatch):
    monkeypatch.setattr(attention_check, "_probe_varlen", lambda *a: pytest.fail("unexpected probe"))
    attention_check.check_attention_backend("sdpa", "cpu", torch.bfloat16)


@pytest.mark.parametrize("error", [ImportError("missing flash_attn"), OSError("DLL load failed"),
                                  RuntimeError("no kernel image is available")])
def test_unavailable_external_flash_reports_compatible_wheel(monkeypatch, error):
    def fail(*args):
        raise error
    monkeypatch.setattr(attention_check, "_probe_flash_varlen", fail)
    with pytest.raises(RuntimeError, match="wheel matching your OS, Python") as caught:
        attention_check.check_attention_backend("flash_attn_2", "cuda:0", torch.bfloat16)
    assert caught.value.__cause__ is error


def test_external_flash_does_not_probe_native_varlen(monkeypatch):
    calls = []
    monkeypatch.setattr(attention_check, "_probe_varlen", lambda *a: pytest.fail("wrong backend"))
    monkeypatch.setattr(attention_check, "_probe_flash_varlen", lambda *a: calls.append(a))
    attention_check.check_attention_backend("flash_attn_2", "cuda:0", torch.bfloat16)
    assert len(calls) == 1


@pytest.mark.parametrize("error", [ImportError("missing varlen module"),
                                  RuntimeError("USE_FLASH_ATTENTION was not enabled for build.")])
def test_unavailable_varlen_reports_actionable_error(monkeypatch, error):
    def fail(*args):
        raise error
    monkeypatch.setattr(attention_check, "_probe_varlen", fail)
    with pytest.raises(RuntimeError, match="train.attention_backend='sdpa'") as caught:
        attention_check.check_attention_backend("torch_varlen", "cuda:0", torch.bfloat16)
    assert str(error) in str(caught.value)
    assert caught.value.__cause__ is error


@pytest.mark.skipif(os.environ.get("MAGEFLOW_GPU_TESTS") != "1", reason="GPU opt-in")
def test_varlen_probe_preserves_rng_and_leaves_no_live_allocations():
    torch.cuda.init()
    before_rng = torch.cuda.get_rng_state().clone()
    before_memory = torch.cuda.memory_allocated()
    attention_check.check_attention_backend("torch_varlen", "cuda:0", torch.bfloat16)
    assert torch.equal(before_rng, torch.cuda.get_rng_state())
    assert torch.cuda.memory_allocated() == before_memory
