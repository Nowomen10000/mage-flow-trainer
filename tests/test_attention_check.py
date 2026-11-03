import os

import pytest
import torch

from trainer.training import attention_check


def test_sdpa_does_not_probe_varlen(monkeypatch):
    monkeypatch.setattr(attention_check, "_probe_varlen", lambda *a: pytest.fail("unexpected probe"))
    attention_check.check_attention_backend("sdpa", "cpu", torch.bfloat16)


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
