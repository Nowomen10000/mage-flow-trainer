"""Run GPU cases with torchrun --nproc_per_node=2 -m pytest on both ranks."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from trainer.training.distributed import configure_moss_checkpoint_ddp


@pytest.mark.parametrize("family,world,compiled,checkpoint,inside,disabled", [
    ("moss_image", 2, "default", True, False, True),
    ("moss_image", 1, "default", True, False, False),
    ("moss_image", 2, None, True, False, False),
    ("moss_image", 2, "default", False, False, False),
    ("moss_image", 2, "default", True, True, False),
    ("flux3", 2, "default", True, False, False),
])
def test_ddp_partition_guard_is_scoped(monkeypatch, family, world, compiled, checkpoint, inside, disabled):
    monkeypatch.setattr(torch._dynamo.config, "optimize_ddp", True)
    cfg = SimpleNamespace(model_family=family, compile=compiled,
                          gradient_checkpointing=checkpoint, moss_checkpoint_inside_compile=inside)
    configure_moss_checkpoint_ddp(cfg, world)
    assert torch._dynamo.config.optimize_ddp is (not disabled)


@pytest.fixture(scope="module")
def ddp_device():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    yield torch.device("cuda", rank)
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("MAGEFLOW_GPU_TESTS") != "1" or os.environ.get("WORLD_SIZE") != "2",
                    reason="requires two-rank GPU opt-in")
@pytest.mark.parametrize("preset,inside,backend", [
    (None, False, "sdpa"),
    ("standard", False, "sdpa"),
    ("standard", True, "sdpa"),
    ("standard", False, "torch_varlen"),
    ("balanced", False, "torch_varlen"),
    ("fast", False, "torch_varlen"),
])
def test_moss_compiled_ddp_changes_shapes_and_accumulates(monkeypatch, ddp_device, preset, inside, backend):
    from trainer.modeling.moss_image import MOSSImage, MOSSParams
    from trainer.training.distributed import configure_quantized_ddp, sync_quantized_model
    from trainer.training.params import AdapterConfig, apply_adapter
    from trainer.training.quant import QuantConfig, quantize_module

    device = ddp_device
    rank = device.index
    torch.set_num_threads(2)
    monkeypatch.setattr(torch._dynamo.config, "optimize_ddp", True)
    try:
        torch.manual_seed(42 + rank)
        model = MOSSImage(MOSSParams(dim=256, ffn_dim=512, num_heads=4,
                                    num_layers=4, text_dim=64, latent_channels=8)).to(device, torch.bfloat16)
        quant = QuantConfig(preset=preset) if preset else None
        if quant:
            model = quantize_module(model, quant, device, torch.bfloat16, quant.frozen_matmul)
        apply_adapter(model, AdapterConfig(kind="lycoris_lora", rank=4, alpha=4,
                                          dtype="bfloat16", moss_train_text_refiner=False))
        model.to(device)
        cfg = SimpleNamespace(model_family="moss_image", compile="default",
                              gradient_checkpointing=True, moss_checkpoint_inside_compile=inside)
        configure_moss_checkpoint_ddp(cfg, 2)
        kwargs = {"find_unused_parameters": True}
        if quant and quant.frozen_matmul:
            kwargs = configure_quantized_ddp().to_kwargs()
            sync_quantized_model(model)
        model.configure_execution(True, compile_mode="default", attention_backend=backend,
                                  moss_checkpoint_inside_compile=inside)
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], **kwargs)
        parameters = [p for p in model.parameters() if p.requires_grad]
        initial = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        for step, batch in enumerate((2, 1, 2)):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                with ddp.no_sync() if micro == 0 else nullcontext():
                    xs = [torch.randn(1, 8, 1, 3 + (step + i) % 2, 4, device=device,
                                      dtype=torch.bfloat16) for i in range(batch)]
                    text = torch.randn(batch, 19, 64, device=device, dtype=torch.bfloat16)
                    lengths = torch.arange(batch, device=device) + 9 + step + micro + rank
                    mask = torch.arange(19, device=device)[None] < lengths[:, None]
                    out = ddp(xs, torch.full((batch,), .5, device=device), (text, mask))[0]
                    loss = sum(x.float().square().mean() for x in out) / (2 * batch)
                    assert torch.isfinite(loss)
                    loss.backward()
            for p in parameters:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                reference = p.grad.clone()
                dist.broadcast(reference, 0)
                assert torch.equal(reference, p.grad)
            optimizer.step()
            for p in parameters:
                reference = p.detach().clone()
                dist.broadcast(reference, 0)
                assert torch.equal(reference, p)
        assert any(not torch.equal(before, after) for before, after in zip(initial, parameters))
    finally:
        torch._dynamo.reset()
