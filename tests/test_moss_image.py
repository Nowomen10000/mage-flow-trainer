import importlib.util
import os
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from trainer.modeling.moss_image import MOSSImage, MOSSParams
from trainer.training.params import (
    ComponentLRs,
    adapter_target_names,
    build_param_groups,
)


def tiny():
    torch.set_num_threads(2)
    return MOSSImage(
        MOSSParams(
            dim=32,
            ffn_dim=48,
            num_heads=2,
            num_layers=2,
            text_dim=16,
            latent_channels=8,
        )
    )


def inputs():
    return (
        [torch.randn(1, 8, 1, 3, 4), torch.randn(1, 8, 1, 2, 3)],
        torch.tensor([0.2, 0.7]),
        torch.randn(2, 7, 16),
        torch.tensor([[1] * 7, [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool),
    )


def test_bf16_rmsnorm_accumulates_input_gradient_in_fp32():
    from trainer.modeling.moss_image import _RMSNorm

    torch.manual_seed(42)
    norm = _RMSNorm(64).bfloat16().requires_grad_(False)
    x = torch.randn(2, 17, 64, dtype=torch.bfloat16, requires_grad=True)
    # Independent FP32 input is the gradient oracle: round only after the
    # complete normalization derivative has been accumulated.
    reference_input = x.detach().float().requires_grad_(True)
    reference = reference_input * torch.rsqrt(
        reference_input.square().mean(-1, keepdim=True) + 1e-6
    )
    reference = reference.bfloat16() * norm.weight
    actual = norm(x)
    upstream_gradient = torch.randn_like(x)
    actual.backward(upstream_gradient)
    reference.backward(upstream_gradient)
    assert torch.equal(actual, reference)
    assert torch.equal(x.grad, reference_input.grad.bfloat16())


def test_reference_forward_and_gradients():
    path = Path("moss-image/pipeline_moss_image.py")
    if not path.exists():
        pytest.skip("Optional local upstream parity reference")
    spec = importlib.util.spec_from_file_location("moss_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(12)
    model = tiny()
    model.configure_execution(False)
    ref = module.MOSSImageTransformer(**asdict(model.params))
    ref.load_state_dict(model.state_dict())
    xs, t, text, mask = inputs()
    actual = model(xs, t, (text, mask))[0]
    expected = [
        ref(
            x.squeeze(2),
            t[i : i + 1] * 1000,
            text[i : i + 1],
            mask[i : i + 1],
            return_dict=False,
        )[0].unsqueeze(2)
        for i, x in enumerate(xs)
    ]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    sum(y.square().mean() for y in actual).backward()
    sum(y.square().mean() for y in expected).backward()
    for (name, p), (other, q) in zip(model.named_parameters(), ref.named_parameters()):
        assert name == other
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)


def test_packed_checkpoint_isolation(monkeypatch):
    from torch.nn.attention import varlen

    def attn(q, k, v, cq, ck, mq, mk):
        bounds = cq.tolist()
        return torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(
                    q[a:b].transpose(0, 1)[None],
                    k[a:b].transpose(0, 1)[None],
                    v[a:b].transpose(0, 1)[None],
                )[0].transpose(0, 1)
                for a, b in pairwise(bounds)
            ]
        )

    monkeypatch.setattr(varlen, "varlen_attn", attn)
    model = tiny()
    xs, t, text, mask = inputs()
    model.configure_execution(False)
    expected = model(xs, t, (text, mask))[0]
    sum(y.square().mean() for y in expected).backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.configure_execution(True, attention_backend="torch_varlen")
    actual = model(xs, t, (text, mask))[0]
    sum(y.square().mean() for y in actual).backward()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[n])
    text[1] *= 100
    torch.testing.assert_close(model(xs, t, (text, mask))[0][0], actual[0])
    assert model.bfloat16().rope.bands.dtype == torch.float32


@pytest.mark.parametrize("mode", ["default", "reduce-overhead", "max-autotune"])
def test_compile_keeps_mode_options_with_safe_reductions(monkeypatch, mode):
    calls = []

    def compile_stub(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", compile_stub)
    model = tiny()
    model.configure_execution(compile_mode=mode, attention_backend="torch_varlen")
    expected = dict(torch._inductor.list_mode_options(mode))
    expected["triton.mix_order_reduction"] = False
    assert calls == [{"options": expected, "dynamic": True}]


@pytest.mark.parametrize("checkpointing", [True, False])
def test_compiled_checkpoint_matches_eager_training_and_eval(monkeypatch, checkpointing):
    # Exercise the captured checkpoint operation with AOTAutograd on CPU;
    # full-size BF16/Inductor parity is covered by the GPU replay benchmark.
    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real_compile(
        fn, backend="aot_eager", dynamic=True, fullgraph=True
    ))
    torch.manual_seed(42)
    model = tiny()
    xs, t, text, mask = inputs()
    model.configure_execution(checkpointing)
    expected = model(xs, t, (text, mask))[0]
    sum(y.square().mean() for y in expected).backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.configure_execution(checkpointing, compile_mode="default")
    actual = model(xs, t, (text, mask))[0]
    sum(y.square().mean() for y in actual).backward()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[n])
    model.eval()
    with torch.no_grad():
        inference = model(xs, t, (text, mask))[0]
    for a, b in zip(inference, expected):
        torch.testing.assert_close(a, b)


@pytest.mark.skipif(os.environ.get("MAGEFLOW_GPU_TESTS") != "1", reason="GPU opt-in")
@pytest.mark.parametrize("backend", ["sdpa", "torch_varlen"])
@pytest.mark.parametrize("quant_mode", ["frozen", "training"])
def test_compiled_moss_quantized_checkpoint_backward(backend, quant_mode):
    from trainer.training.params import AdapterConfig, apply_adapter
    from trainer.training.quant import QuantConfig, quantize_module

    model = MOSSImage(MOSSParams(dim=128, ffn_dim=192, num_heads=4,
                               num_layers=2, text_dim=64, latent_channels=8))
    model = model.to("cuda", torch.bfloat16)
    model = quantize_module(model, QuantConfig(mode=quant_mode, weights_dtype="uint8",
                            use_quantized_matmul=False), torch.device("cuda"),
                            torch.bfloat16, False)
    if quant_mode == "frozen":
        apply_adapter(model, AdapterConfig(kind="lycoris_lora", rank=4,
                                          alpha=4, dtype="bfloat16"))
    model.configure_execution(True, compile_mode="default", attention_backend=backend)
    xs = [torch.randn(1, 8, 1, h, w, device="cuda", dtype=torch.bfloat16)
          for h, w in [(4, 4), (3, 4)]]
    text = torch.randn(2, 7, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.tensor([[1] * 7, [1, 1, 1, 0, 0, 0, 0]], device="cuda", dtype=torch.bool)
    outputs = model(xs, torch.tensor([.2, .7], device="cuda"), (text, mask))[0]
    loss = sum(y.float().square().mean() for y in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)


def test_training_targets():
    model = tiny()
    report = build_param_groups(model, ComponentLRs(), 1e-5)
    assert sum(p.numel() for g in report.groups for p in g["params"]) == sum(
        p.numel() for p in model.parameters()
    )
    targets = adapter_target_names(model, ["image_attn", "text_attn", "mlp"])
    assert "layers.0.proj.q" in targets
    assert all("modulation" not in name for name in targets)


@pytest.mark.parametrize("kind", ["lora", "lokr", "lycoris_lora"])
@pytest.mark.parametrize("train_text", [False, True])
def test_moss_text_refiner_adapter_selection(kind, train_text):
    from trainer.training.params import AdapterConfig, apply_adapter

    model = tiny()
    cfg = AdapterConfig(kind=kind, rank=2, alpha=2,
                        moss_train_text_refiner=train_text)
    targets = adapter_target_names(model, cfg.components,
                                  moss_train_text_refiner=train_text)
    assert len(targets) == 12 + (10 if train_text else 0)
    apply_adapter(model, cfg)
    assert any(p.requires_grad for p in model.layers.parameters())
    assert any(p.requires_grad for p in model.text_embedder.blocks.parameters()) == train_text
    assert not any(p.requires_grad for p in model.text_embedder.proj.parameters())


def test_sharded_load_identity_export_and_reload(tmp_path):
    import json

    from safetensors.torch import save_file

    from trainer.data.dataset import DatasetConfig
    from trainer.modeling.checkpoint_metadata import checkpoint_identity
    from trainer.modeling.export import export_checkpoint
    from trainer.modeling.moss_loader import load_transformer
    from trainer.training.config import Config

    model = tiny()
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(asdict(model.params)))
    items = list(model.state_dict().items())
    mapping = {}
    for i, chunk in enumerate((items[: len(items) // 2], items[len(items) // 2 :])):
        name = f"shard{i}.safetensors"
        save_file(dict(chunk), folder / name)
        mapping.update({k: name for k, v in chunk})
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    loaded = load_transformer(tmp_path, None, torch.float32)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, loaded.state_dict()[k])
    identity = checkpoint_identity(folder)
    assert identity["model_type"] == "moss_image" and len(identity["shards"]) == 2
    cfg = Config(dataset=DatasetConfig(path=str(tmp_path)))
    cfg.train.model_family = "moss_image"
    cfg.adapter.kind = "none"
    export_checkpoint(model, tmp_path, "export", cfg, torch.float32, 2)
    reloaded = load_transformer(
        tmp_path, tmp_path / "export.safetensors", torch.float32
    )
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, reloaded.state_dict()[k])


def test_moss_config_and_text_cache_namespace(tmp_path):
    import json

    from trainer.data.caption_variations import _encoder_fingerprint_payload
    from trainer.data.dataset import DatasetConfig
    from trainer.modeling.loader import resolve_model_family
    from trainer.training.config import Config, validate_model_options

    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "MOSSImagePipeline"})
    )
    assert resolve_model_family(tmp_path) == "moss_image"
    cfg = Config(dataset=DatasetConfig(path=str(tmp_path)))
    cfg.train.model_path = str(tmp_path)
    validate_model_options(cfg)
    assert cfg.train.flux2_vae
    payload = _encoder_fingerprint_payload(cfg)
    assert (
        payload["model_family"] == "moss_image"
        and payload["layers"] == "last_hidden_state"
    )
    cfg.rti.enabled = True
    with pytest.raises(ValueError, match="supports text-to-image flow training"):
        validate_model_options(cfg)


@pytest.mark.parametrize(
    "kind,algo",
    [
        ("lora", "locon"),
        ("lycoris_lora", "locon"),
        ("lycoris_lora", "lokr"),
        ("lycoris_lora", "dora"),
    ],
)
def test_adapter_backward_without_modulation(kind, algo):
    from trainer.training.params import AdapterConfig, apply_adapter

    model = tiny()
    apply_adapter(
        model,
        AdapterConfig(kind=kind, lycoris_algo=algo, rank=4, alpha=4, dtype="float32"),
    )
    xs, t, text, mask = inputs()
    sum(y.square().mean() for y in model(xs, t, (text, mask))[0]).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(
        not p.requires_grad for n, p in model.named_parameters() if "modulation" in n
    )


def test_fused_export_loads_exact_native_weights(tmp_path):
    import json
    from safetensors.torch import save_file
    from trainer.modeling.moss_loader import load_transformer

    model = tiny()
    folder = tmp_path / 'transformer'
    folder.mkdir()
    (folder / 'config.json').write_text(json.dumps(asdict(model.params)))
    state = dict(model.state_dict())
    for i in range(model.params.num_layers):
        prefix = f'layers.{i}.proj.'
        state[prefix + 'qkv.weight'] = torch.cat([
            state.pop(prefix + name + '.weight') for name in ('q', 'k', 'v')
        ])
    source = tmp_path / 'fused.safetensors'
    save_file({'model.diffusion_model.' + k: v for k, v in state.items()}, source)
    loaded = load_transformer(tmp_path, source, torch.float32)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    state['layers.0.proj.q.weight'] = model.layers[0].proj['q'].weight
    save_file(state, source)
    with pytest.raises(ValueError, match='Ambiguous MOSS checkpoint'):
        load_transformer(tmp_path, source, torch.float32)
    del state['layers.0.proj.q.weight']
    state['layers.0.proj.qkv.weight'] = state['layers.0.proj.qkv.weight'][:-1].clone()
    save_file(state, source)
    with pytest.raises(ValueError, match='expected'):
        load_transformer(tmp_path, source, torch.float32)
