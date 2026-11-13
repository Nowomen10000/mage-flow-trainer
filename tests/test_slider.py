"""Concept-slider configuration, signed adapters, objectives and trainer wiring."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import toml
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from trainer.training.config import load_config
from trainer.training.params import AdapterConfig, apply_adapter
from trainer.training.slider import (
    AdapterStrength,
    ConceptSlider,
    SliderConfig,
    mean_mse,
    norm_like,
    per_sample,
    slider_passes,
    slider_step,
    slider_targets,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    """Keep tiny matrix operations cheap without changing other tests' RNGs."""
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        yield
    torch.set_num_threads(threads)


def _config_data(tmp_path):
    return {
        "train": {"model_family": "mage_flow"},
        "dataset": {"path": str(tmp_path)},
        "adapter": {"kind": "lora"},
        "slider": {
            "enabled": True,
            "target_class": "",
            "positive_prompt": "a cheerful person",
            "negative_prompt": "a gloomy person",
        },
    }


def _load(tmp_path, data):
    path = tmp_path / "slider.toml"
    path.write_text(toml.dumps(data), encoding="utf-8")
    return load_config(path)


def test_config_defaults_and_prompt_roles(tmp_path):
    cfg = _load(tmp_path, _config_data(tmp_path))
    assert cfg.slider.enabled
    assert cfg.slider.guidance_strength == 3.0
    assert cfg.slider.anchor_strength == 1.0
    assert not cfg.slider.use_anchor
    assert cfg.slider.prompts() == {
        "positive": "a cheerful person", "neutral": "", "negative": "a gloomy person",
    }
    cfg.slider.anchor_class = "a tree"
    assert cfg.slider.use_anchor
    assert cfg.slider.prompts()["anchor"] == "a tree"


@pytest.mark.parametrize("cache", [False, True])
def test_config_accepts_unused_caption_options(tmp_path, cache):
    # A reused normal config keeps its caption settings; a slider ignores them rather than refusing.
    data = _config_data(tmp_path)
    data["train"].update(caption_variations=2, cache_text_embeddings=cache)
    data["dataset"]["caption"] = {"shuffle_tags": True}
    assert _load(tmp_path, data).train.caption_variations == 2
    if not cache:
        data["slider"]["enabled"] = False
        with pytest.raises(ValueError, match="cache_text_embeddings"):
            _load(tmp_path, data)


@pytest.mark.parametrize("incompatible", [
    "identical", "full_finetune", "dora", "gradient_release", "hf_scale", "preserve",
])
def test_config_rejects_incompatible_options(tmp_path, incompatible):
    data = _config_data(tmp_path)
    if incompatible == "identical":
        data["slider"]["negative_prompt"] = "  " + data["slider"]["positive_prompt"] + " "
    elif incompatible == "full_finetune":
        data["adapter"]["kind"] = "none"
    elif incompatible == "dora":
        data["adapter"].update(kind="lycoris_lora", lycoris_algo="dora")
    elif incompatible == "gradient_release":
        # Both settings are needed to reach slider validation rather than optimizer validation.
        data["optimizer"] = {"kind": "optimi_adamw", "gradient_release": True, "max_grad_norm": 0}
    elif incompatible == "hf_scale":
        data["flow"] = {"hf_scale": 0.1}
    else:
        pairs = tmp_path / "pairs.txt"
        pairs.write_text("a person with glasses ||| a person\n", encoding="utf-8")
        from trainer.training.preserve import load_pairs
        assert load_pairs(pairs) == [("a person with glasses", "a person")]
        data["preserve"] = {"prompts": str(pairs)}
    with pytest.raises(ValueError, match=r"\[slider\]"):
        _load(tmp_path, data)
    # Prove that unrelated validation does not account for the rejection.
    if incompatible != "identical":
        data["slider"]["enabled"] = False
        assert not _load(tmp_path, data).slider.enabled


def test_static_caption_check_is_skipped_only_for_slider(tmp_path):
    data = _config_data(tmp_path)
    data["train"]["cache_text_embeddings"] = True
    data["dataset"]["caption"] = {"shuffle_tags": True}
    assert _load(tmp_path, data).dataset.caption.shuffle_tags
    data["slider"]["enabled"] = False
    with pytest.raises(ValueError, match="fixed captions.*shuffle_tags"):
        _load(tmp_path, data)


class _TinyModel(nn.Module):
    """One real adapter target, with the same qualified name as Mage-Flow."""

    def __init__(self, width=8):
        super().__init__()
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.to_q = nn.Linear(width, width)
        self.transformer_blocks = nn.ModuleList([block])

    @property
    def projection(self):
        return self.transformer_blocks[0].attn.to_q

    def add_adapter(self, config):
        from peft import inject_adapter_in_model
        inject_adapter_in_model(config, self)

    def forward(self, x):
        return self.projection(x)


def _adapted(kind="lora", algo="locon", bypass=True, width=8):
    model = _TinyModel(width)
    apply_adapter(model, AdapterConfig(
        kind=kind, rank=2, alpha=1.7, components=["image_attn"],
        lycoris_algo=algo, lycoris_bypass=bypass,
    ))
    # Include zero-initialized up factors: otherwise signed-output checks are vacuous.
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.normal_(0, 0.2)
    return model.eval()


def _scale_snapshot(model):
    from peft.tuners.tuners_utils import BaseTunerLayer
    return [
        (module, dict(module.scaling) if isinstance(module, BaseTunerLayer) else module.multiplier)
        for name, module in model.named_modules()
        if (isinstance(module, BaseTunerLayer) and isinstance(getattr(module, "scaling", None), dict))
        or name.endswith("lycoris_adapter")
    ]


def _assert_scales_restored(snapshot):
    for module, original in snapshot:
        actual = module.scaling if isinstance(original, dict) else module.multiplier
        if isinstance(original, dict):
            assert actual.keys() == original.keys()
            pairs = [(actual[key], value) for key, value in original.items()]
        else:
            pairs = [(actual, original)]
        for value, before in pairs:
            # hex also distinguishes signed zero; equality alone is weaker than bit identity.
            assert float(value).hex() == float(before).hex()


@pytest.mark.parametrize("kind,algo,bypass", [
    pytest.param("lora", "locon", True, id="peft-lora"),
    pytest.param("lokr", "locon", True, id="peft-lokr"),
    pytest.param("lycoris_lora", "locon", True, id="lycoris-locon-bypass"),
    pytest.param("lycoris_lora", "locon", False, id="lycoris-locon-weight"),
    pytest.param("lycoris_lora", "lokr", True, id="lycoris-lokr"),
])
def test_adapter_strength_off_mirror_and_restore(kind, algo, bypass):
    model = _adapted(kind, algo, bypass)
    layer = model.projection
    if kind == "lycoris_lora":
        # Restoration must preserve a pre-existing multiplier, not just reset to 1.
        layer.lycoris_adapter.multiplier = 0.625
    snapshot = _scale_snapshot(model)
    assert snapshot
    if kind == "lycoris_lora":
        assert layer.lycoris_adapter.bypass_mode is bypass
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    strength = AdapterStrength(model)
    assert len(strength) == len(snapshot) > 0
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        # Independent adapter-off reference; never obtained through AdapterStrength.
        base = layer.lycoris_adapter.org_forward(x) if kind == "lycoris_lora" else layer.base_layer(x)
        strength.set(1.0)
        positive = model(x)
        delta = positive - base
        assert delta.abs().max() > 1e-5
        strength.set(0.0)
        assert torch.equal(model(x), base)
        strength.set(-1.0)
        torch.testing.assert_close(model(x) - base, -delta, atol=3e-7, rtol=1e-5)
        strength.set(0.375)
        torch.testing.assert_close(model(x) - base, 0.375 * delta, atol=3e-7, rtol=1e-5)
        strength.restore()
        assert strength.value == 1.0
        assert torch.equal(model(x), positive)
    _assert_scales_restored(snapshot)
    for name, param in model.named_parameters():
        if name in frozen:
            assert torch.equal(param, frozen[name]), name


def test_adapter_strength_requires_an_adapter():
    with pytest.raises(RuntimeError, match="no PEFT or LyCORIS adapter"):
        AdapterStrength(_TinyModel())


def _manual_prediction(model, x, sign, scaling):
    """LoRA's function using its tensors directly, without reading mutable strength."""
    layer = model.projection
    adapter = next(iter(layer.lora_A))
    base = torch.nn.functional.linear(x, layer.base_layer.weight, layer.base_layer.bias)
    a = torch.nn.functional.linear(x, layer.lora_A[adapter].weight)
    delta = torch.nn.functional.linear(a, layer.lora_B[adapter].weight)
    return base + sign * scaling * delta


def _reference_targets(reference, guidance):
    # Deliberately independent of norm_like, slider_targets and mean_mse.
    plus, minus = [], []
    for p, n, q in zip(reference["positive"], reference["neutral"], reference["negative"], strict=True):
        for sign, output in ((1, plus), (-1, minus)):
            raw = n + sign * guidance * (p - q)
            output.append((raw - raw.mean()) / (raw.std() + 1e-8) * n.std() + n.mean())
    return plus, minus


def _reference_mse(predictions, targets):
    return torch.stack([(p - t).square().mean()
                        for p, t in zip(predictions, targets, strict=True)]).mean()


def _reference_objective(cfg, predict, reference):
    plus, minus = _reference_targets(reference, cfg.guidance_strength)
    loss = (_reference_mse(predict("neutral", 1), plus)
            + _reference_mse(predict("neutral", -1), minus)) / 3
    if cfg.use_anchor:
        loss = loss + cfg.anchor_strength / 6 * (
            _reference_mse(predict("anchor", 1), reference["anchor"])
            + _reference_mse(predict("anchor", -1), reference["anchor"])
        )
    return loss


def _lora_grads(model):
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert params and all("lora_" in name for name in params)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
    return {name: p.grad.detach().clone() for name, p in params.items()}


@pytest.mark.parametrize("anchor,anchor_strength,num_passes", [
    ("", 1.0, 2), ("a tree", 0.7, 4), ("a tree", 0.0, 2),
])
def test_slider_step_checkpointed_gradients_match_independent_objective(anchor, anchor_strength, num_passes):
    cfg = SliderConfig(enabled=True, positive_prompt="happy", negative_prompt="sad",
                       anchor_class=anchor, anchor_strength=anchor_strength, guidance_strength=2.3)
    model = _adapted()
    layer = model.projection
    scaling = next(iter(layer.scaling.values()))
    snapshot = _scale_snapshot(model)
    strength = AdapterStrength(model)
    x = torch.randn(2, 3, 8)
    contexts = {role: torch.randn(1, 3, 8) * 0.4 for role in cfg.prompts()}
    forward_calls = []
    counts = Counter()

    @contextmanager
    def early_context():
        counts["enter"] += 1
        counts["active"] += 1
        try:
            yield
        finally:
            counts["active"] -= 1
            counts["exit"] += 1

    def forward(role):
        forward_calls.append((role, strength.value, torch.is_grad_enabled(), counts["active"]))
        return checkpoint(model, x + contexts[role], use_reentrant=False)

    def early_backward(term):
        assert counts["active"] == 1
        counts["backward"] += 1
        term.backward()

    model.zero_grad(set_to_none=True)
    loss, stats = slider_step(cfg, strength, forward, early_backward, early_context)
    assert loss.requires_grad and torch.isfinite(loss)
    assert strength.value == -1.0
    assert counts == Counter(enter=num_passes - 1, exit=num_passes - 1, backward=num_passes - 1, active=0)
    assert [role for role, sign, grad, _ in forward_calls if not grad and sign == 0] == list(cfg.prompts())
    signed = [(role, sign, active) for role, sign, grad, active in forward_calls if grad]
    expected_roles = [("neutral", 1.0)]
    if cfg.use_anchor:
        expected_roles.append(("anchor", 1.0))
    expected_roles.append(("neutral", -1.0))
    if cfg.use_anchor:
        expected_roles.append(("anchor", -1.0))
    assert [(role, sign) for role, sign, _ in signed] == expected_roles
    assert [active for _, _, active in signed] == [1] * (num_passes - 1) + [0]
    loss.backward()
    strength.restore()
    assert strength.value == 1.0
    _assert_scales_restored(snapshot)
    actual_grads = _lora_grads(model)
    assert any(g.abs().sum() > 0 for g in actual_grads.values())

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        reference = {role: _manual_prediction(model, x + ctx, 0, scaling) for role, ctx in contexts.items()}
    predict = lambda role, sign: _manual_prediction(model, x + contexts[role], sign, scaling)
    objective = _reference_objective(cfg, predict, reference)
    torch.testing.assert_close(loss.detach(), objective.detach(), atol=2e-7, rtol=2e-6)
    assert set(stats) == {"slider_pos", "slider_neg"} | ({"slider_anchor"} if cfg.use_anchor else set())
    assert all(not value.requires_grad for value in stats.values())
    torch.testing.assert_close(sum(stats.values()), objective.detach(), atol=2e-7, rtol=2e-6)
    objective.backward()
    expected_grads = _lora_grads(model)
    for name, expected in expected_grads.items():
        torch.testing.assert_close(actual_grads[name], expected, atol=2e-6, rtol=2e-5, msg=name)


class _NonlinearModel(_TinyModel):
    """An activation after the adapter saves a sign-dependent tensor for backward."""

    def forward(self, x):
        return torch.tanh(self.projection(x) * 3)


def test_per_pass_backward_is_what_keeps_checkpointing_exact():
    # Doing both signed forwards and one backward would let checkpoint recomputation replay the
    # +1 pass at strength -1. A single Linear cannot show that (its scalar multiply saves no
    # tensor); an activation after the adapter does.
    cfg = SliderConfig(enabled=True, positive_prompt="happy", negative_prompt="sad")
    model = _NonlinearModel()
    apply_adapter(model, AdapterConfig(kind="lora", rank=2, alpha=1.7, components=["image_attn"]))
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.normal_(0, 0.5)
    strength = AdapterStrength(model)
    x = torch.randn(2, 3, 8)
    contexts = {role: torch.randn(1, 3, 8) for role in cfg.prompts()}

    def grads_of(forward):
        model.zero_grad(set_to_none=True)
        loss, _ = slider_step(cfg, strength, forward, lambda term: term.backward())
        loss.backward()
        strength.restore()
        return _lora_grads(model)

    exact = grads_of(lambda role: model(x + contexts[role]))
    checkpointed = grads_of(lambda role: checkpoint(model, x + contexts[role], use_reentrant=False))
    for name, expected in exact.items():
        torch.testing.assert_close(checkpointed[name], expected, atol=2e-6, rtol=2e-5, msg=name)

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        strength.set(0.0)
        reference = {role: per_sample(model(x + ctx)) for role, ctx in contexts.items()}
    plus, minus = slider_targets(reference["positive"], reference["neutral"], reference["negative"],
                                 cfg.guidance_strength)
    strength.set(1.0)
    naive = mean_mse(checkpoint(model, x + contexts["neutral"], use_reentrant=False), plus) / 3
    strength.set(-1.0)
    naive = naive + mean_mse(checkpoint(model, x + contexts["neutral"], use_reentrant=False), minus) / 3
    naive.backward()
    strength.restore()
    naive_grads = _lora_grads(model)
    assert any(not torch.allclose(naive_grads[name], exact[name], atol=1e-4, rtol=1e-3) for name in exact)


@pytest.mark.parametrize("fail_at", [1.0, -1.0])
def test_slider_step_restores_strength_if_signed_forward_raises(fail_at):
    model = _adapted()
    strength = AdapterStrength(model)
    snapshot = _scale_snapshot(model)
    cfg = SliderConfig(enabled=True, positive_prompt="happy", negative_prompt="sad")
    x = torch.randn(2, 3, 8)
    def forward(role):
        if torch.is_grad_enabled() and strength.value == fail_at:
            raise RuntimeError("signed forward failed")
        return model(x)
    with pytest.raises(RuntimeError, match="signed forward failed"):
        slider_step(cfg, strength, forward, lambda term: term.backward())
    assert strength.value == 1.0
    _assert_scales_restored(snapshot)


@pytest.mark.parametrize("packed", [False, True])
def test_per_sample_fp32_conversion_preserves_samples_and_gradients(packed):
    values = [torch.randn(2, 3, dtype=torch.float64, requires_grad=True),
              torch.randn(2, 5 if packed else 3, dtype=torch.float64, requires_grad=True)]
    pred = values if packed else torch.stack(values)
    result = per_sample(pred)
    assert len(result) == 2
    for actual, expected in zip(result, values, strict=True):
        assert actual.dtype == torch.float32
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected.float())
    sum(p.sum() for p in result).backward()
    assert all(torch.equal(value.grad, torch.ones_like(value)) for value in values)


def test_targets_match_ai_toolkit_and_neutral_statistics_per_sample():
    positive = [torch.randn(3, 5), torch.randn(2, 7)]
    neutral = [torch.randn(3, 5) * 0.3 - 4, torch.randn(2, 7) * 3 + 9]
    negative = [torch.randn(3, 5), torch.randn(2, 7)]
    guidance = 3.0
    plus, minus = slider_targets(positive, neutral, negative, guidance)
    for p, n, q, actual_plus, actual_minus in zip(positive, neutral, negative, plus, minus, strict=True):
        enhance_positive_target = n + guidance * (p - q)
        erase_negative_target = n - guidance * (q - p)
        torch.testing.assert_close(enhance_positive_target, erase_negative_target, rtol=0, atol=0)
        for actual, raw in ((actual_plus, enhance_positive_target), (actual_minus, n - guidance * (p - q))):
            expected = (raw - raw.mean()) / (raw.std() + 1e-8) * n.std() + n.mean()
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(actual, norm_like(raw, n), atol=0, rtol=0)
            torch.testing.assert_close(actual.mean(), n.mean(), atol=1e-5, rtol=0)
            torch.testing.assert_close(actual.std(), n.std(), atol=1e-5, rtol=0)
    assert not torch.allclose(plus[0], minus[0])


def test_mean_mse_weights_packed_images_equally():
    pred = [torch.full((1, 2, 1, 2, 2), 2.0, requires_grad=True),
            torch.full((1, 2, 1, 3, 5), 4.0, requires_grad=True)]
    targets = [torch.zeros_like(p) for p in pred]
    expected = torch.stack([(p - target).square().mean() for p, target in zip(pred, targets, strict=True)]).mean()
    actual = mean_mse(pred, targets)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.item() == 10.0
    token_weighted = sum(p.square().sum() for p in pred) / sum(p.numel() for p in pred)
    assert not torch.isclose(actual, token_weighted)
    actual.backward()
    for p in pred:
        torch.testing.assert_close(p.grad, p.detach() / p.numel(), atol=0, rtol=0)


def test_slider_passes_roles_weights_and_targets():
    cfg = SliderConfig(enabled=True, positive_prompt="happy", negative_prompt="sad",
                       anchor_class="tree", anchor_strength=1.2)
    reference = {role: [torch.randn(2, 4), torch.randn(3, 4)] for role in cfg.prompts()}
    passes = slider_passes(cfg, reference)
    assert [(p.strength, p.prompt, p.stat) for p in passes] == [
        (1.0, "neutral", "slider_pos"), (1.0, "anchor", "slider_anchor"),
        (-1.0, "neutral", "slider_neg"), (-1.0, "anchor", "slider_anchor"),
    ]
    assert [p.weight for p in passes] == pytest.approx([1 / 3, 0.2, 1 / 3, 0.2])
    assert passes[1].targets is reference["anchor"] and passes[3].targets is reference["anchor"]
    cfg.anchor_strength = 0
    assert len(slider_passes(cfg, reference)) == 2


class _CPUAccelerator:
    """Accelerate's image-scaling contract, with observable no_sync scopes."""

    device = torch.device("cpu")
    gradient_accumulation_steps = 3

    def __init__(self):
        self.backward_values = []
        self.no_sync_count = 0
        self.in_no_sync = False

    def backward(self, loss):
        self.backward_values.append((loss.detach().clone(), self.in_no_sync))
        (loss / self.gradient_accumulation_steps).backward()

    @contextmanager
    def no_sync(self, model):
        assert isinstance(model, nn.Module)
        self.no_sync_count += 1
        self.in_no_sync = True
        try:
            yield
        finally:
            self.in_no_sync = False

    def unwrap_model(self, model, **kwargs):
        return model


class _TinyTransformer(_TinyModel):
    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        hidden, mask = encoder_hidden_states
        assert hidden.shape[0] == mask.shape[0] == len(timestep)
        context = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        assert not hidden.requires_grad and not mask.requires_grad
        def project(x, ctx, t):
            x = x.movedim(1, -1) + ctx[:, None, None, None, :] + t[:, None, None, None, None] * 0.1
            return checkpoint(self.projection, x, use_reentrant=False).movedim(-1, 1)
        if isinstance(hidden_states, list):
            result = [project(x, context[i:i + 1], timestep[i:i + 1]) for i, x in enumerate(hidden_states)]
        else:
            result = project(hidden_states, context, timestep)
        return (result,)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("anchor", [False, True])
def test_trainer_slider_step_and_prompt_encoding(tmp_path, monkeypatch, packed, anchor):
    from trainer.training.flow_rng import FlowRNG
    from trainer.training.train import Trainer
    import trainer.training.train as module
    data = _config_data(tmp_path)
    data["train"]["gradient_accumulation_steps"] = 3
    if anchor:
        data["slider"]["anchor_class"] = "a tree"
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = _load(tmp_path, data)
    # Windows guards real packed attention backends at config load. This CPU
    # transformer has no attention backend; exercise the packed trainer path directly.
    trainer.cfg.train.pack_resolutions = packed
    trainer.dtype = torch.float32
    trainer.accelerator = _CPUAccelerator()
    trainer.phase = None
    trainer.flow_rng = FlowRNG(42, 0, 1, "cpu")
    trainer.transformer = _TinyTransformer()
    apply_adapter(trainer.transformer, AdapterConfig(kind="lora", rank=2, alpha=1.7, components=["image_attn"]))
    with torch.no_grad():
        for p in trainer.transformer.parameters():
            if p.requires_grad:
                p.normal_(0, 0.2)
    encoded = []
    def encode(prompts):
        assert len(prompts) == 1
        encoded.extend(prompts)
        length = len(prompts[0]) % 4 + 1
        hidden = torch.randn(1, length, 8, requires_grad=True)
        mask = torch.ones(1, length, dtype=torch.bool)
        return hidden, mask
    trainer._encode = encode
    trainer.slider = ConceptSlider(trainer.cfg.slider, trainer)
    scaling = next(iter(trainer.transformer.projection.scaling.values()))
    prepared = []
    prepare = module.prepare_training_flow_batch
    def capture_flow(*args, **kwargs):
        result = prepare(*args, **kwargs)
        prepared.append((result[0].detach().clone(), result[1].detach().clone()))
        return result
    monkeypatch.setattr(module, "prepare_training_flow_batch", capture_flow)
    expected_prompts = ["a cheerful person", " ", "a gloomy person"] + (["a tree"] if anchor else [])
    assert Counter(encoded) == Counter(expected_prompts)
    for role, (hidden, mask) in trainer.slider.contexts.items():
        expanded, expanded_mask = trainer.slider.context(role, 2)
        assert expanded.shape == (2, *hidden.shape[1:]) and expanded_mask.shape == (2, *mask.shape[1:])
        assert expanded.is_contiguous() and expanded_mask.is_contiguous()
        assert not hidden.requires_grad and not mask.requires_grad
        assert torch.equal(expanded[0], hidden[0]) and torch.equal(expanded[1], hidden[0])
    xs = [torch.randn(1, 8, 1, 2, 3), torch.randn(1, 8, 1, 3 if packed else 2, 4 if packed else 3)]
    batch = {"latents": xs if packed else torch.cat(xs), "captions": ["unused one", "unused two"]}
    # Repeated steps must reuse fixed contexts, including the substituted empty class.
    for _ in range(2):
        trainer.transformer.zero_grad(set_to_none=True)
        trainer.accelerator.backward_values.clear()
        prepared.clear()
        before = trainer.accelerator.no_sync_count
        loss, extra = trainer._step(batch)
        assert extra is None and loss.requires_grad and torch.isfinite(loss)
        assert trainer.slider.strength.value == -1.0
        trainer.accelerator.backward(loss * len(batch["captions"]))
        trainer.slider.after_backward()
        assert trainer.slider.strength.value == 1.0
        grads = _lora_grads(trainer.transformer)
        assert all(g.abs().sum() > 0 for g in grads.values())
        passes = 4 if anchor else 2
        backwards = trainer.accelerator.backward_values
        assert len(backwards) == passes
        torch.testing.assert_close(backwards[0][0], len(batch["captions"]) * trainer.last_stats["slider_pos"])
        assert [sync for _, sync in backwards] == [True] * (passes - 1) + [False]
        assert trainer.accelerator.no_sync_count - before == passes - 1
        expected_stats = {"slider_pos", "slider_neg"} | ({"slider_anchor"} if anchor else set())
        assert expected_stats <= trainer.last_stats.keys()
        torch.testing.assert_close(sum(trainer.last_stats[key] for key in expected_stats), loss.detach())

        # Independent objective on the real step's captured noise/timesteps. This
        # also checks equal image weighting for differently sized packed samples.
        def predict(role, sign):
            context, mask = trainer.slider.context(role, len(batch["captions"]))
            context = (context * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            predictions = []
            for i, (noisy, times) in enumerate(prepared):
                ctx = context[i:i + 1] if packed else context
                x = (noisy.movedim(1, -1) + ctx[:, None, None, None, :]
                     + times[:, None, None, None, None] * 0.1)
                out = _manual_prediction(trainer.transformer, x, sign, scaling).movedim(-1, 1)
                predictions.extend(out.unbind(0))
            return predictions
        with torch.no_grad():
            reference = {role: predict(role, 0) for role in trainer.cfg.slider.prompts()}
        trainer.transformer.zero_grad(set_to_none=True)
        objective = _reference_objective(trainer.cfg.slider, predict, reference)
        torch.testing.assert_close(loss.detach(), objective.detach(), atol=2e-7, rtol=2e-6)
        (objective * len(batch["captions"]) / trainer.accelerator.gradient_accumulation_steps).backward()
        for name, expected in _lora_grads(trainer.transformer).items():
            torch.testing.assert_close(grads[name], expected, atol=2e-6, rtol=2e-5, msg=name)
    assert Counter(encoded) == Counter(expected_prompts)


@pytest.mark.parametrize("anchor", ["", "a tree", "a cheerful person"])
def test_text_cache_build_uses_only_unique_slider_prompts(tmp_path, monkeypatch, anchor):
    from trainer.training.train import Trainer
    import trainer.training.train as module
    from trainer.data.text_cache import validate_static_captions
    data = _config_data(tmp_path)
    data["slider"]["anchor_class"] = anchor
    # Caption options left over from a normal config are accepted and unused, variations included.
    data["train"].update(cache_text_embeddings=True, text_cache_batch_size=2, caption_variations=2)
    data["dataset"]["caption"] = {"shuffle_tags": True}
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = _load(tmp_path, data)
    trainer.dtype = torch.float32
    trainer.accelerator = SimpleNamespace(device="cpu", is_main_process=False)
    class ForbiddenCaption:
        def __getattr__(self, name):
            pytest.fail(f"slider cache inspected caption option {name}")
    class ForbiddenDataset:
        @property
        def entries(self):
            pytest.fail("slider cache inspected dataset captions")
    trainer.cfg.dataset.caption = ForbiddenCaption()
    trainer.dataset = ForbiddenDataset()
    validate_static_captions(trainer.cfg)
    forbidden = Mock(side_effect=AssertionError("caption path must not run for a slider"))
    monkeypatch.setattr("trainer.data.text_cache.validate_static_captions", forbidden)
    monkeypatch.setattr("trainer.data.caption.build_caption", forbidden)
    monkeypatch.setattr(trainer, "_build_variation_cache", forbidden)
    load = Mock(return_value=SimpleNamespace(text_encoder=nn.Identity()))
    monkeypatch.setattr(module, "load_components", load)
    encoded = []
    def encode(components, captions, device, max_length):
        encoded.extend(captions)
        lengths = torch.tensor([len(c) for c in captions])
        hidden = lengths.float()[:, None, None].expand(-1, int(lengths.max()), 8)
        return hidden, torch.arange(hidden.shape[1])[None] < lengths[:, None]
    monkeypatch.setattr(module, "encode_prompts", encode)
    trainer._build_text_cache()
    expected = ConceptSlider.cached_prompts(trainer.cfg.slider)
    independent = {"a cheerful person", " ", "a gloomy person"} | ({anchor} if anchor else set())
    assert expected == sorted(independent)
    assert Counter(encoded) == Counter(expected)
    assert set(trainer.text_cache.embeddings) == independent
    for prompt, embedding in trainer.text_cache.embeddings.items():
        assert embedding.shape == (len(prompt), 8)
        assert torch.equal(embedding, torch.full_like(embedding, len(prompt)))
        assert not embedding.requires_grad and embedding.device.type == "cpu"
    forbidden.assert_not_called()
    load.assert_called_once()
    assert load.call_args.kwargs["load_transformer"] is False
    assert load.call_args.kwargs["load_vae"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_gui_slider_schema_and_normal_save_roundtrip(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from trainer.gui import bridge
    from trainer.gui.schema import SPEC, layout_keys
    slider_keys = {key for key in bridge.schema() if key.startswith("slider.")}
    assert slider_keys == {"slider." + f.name for f in fields(SliderConfig)}
    assert slider_keys <= set(layout_keys())
    assert slider_keys <= SPEC.keys()
    data = _config_data(tmp_path)
    data["slider"].update(enabled=enabled, target_class="a person", anchor_class="a tree",
                          guidance_strength=4.25, anchor_strength=0.625)
    flat = bridge.defaults() | bridge.flatten(data)
    path = tmp_path / "gui-slider.toml"
    bridge.write_toml(path, flat)
    restored = bridge.defaults() | bridge.flatten(bridge.read_toml(path))
    loaded = load_config(path)
    for key, expected in data["slider"].items():
        assert restored["slider." + key] == expected
        assert getattr(loaded.slider, key) == expected
