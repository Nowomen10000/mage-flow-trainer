"""LyCORIS LoRA registered under each target, so DDP and checkpointing see its tensors."""

from dataclasses import asdict
from functools import wraps

import torch


ALGORITHMS = ("locon", "lokr", "dora")
DECOMPOSED = {"dora"}


def dense_weight(module):
    if hasattr(module, "sdnq_dequantizer"):
        return module.sdnq_dequantizer(
            module.weight,
            module.scale,
            zero_point=module.zero_point,
            svd_up=module.svd_up,
            svd_down=module.svd_down,
            skip_quantized_matmul=True,
        ).detach()
    return module.weight.detach()


class SDNQWeightAccess:
    def _current_weight(self):
        return dense_weight(self.org_module[0])

    @property
    def org_weight(self):
        return self._current_weight()


_ADAPTER_CLASSES = None


def install_allocator_safe_timing():
    """Avoid pinned LyCORIS's graph-capture/free abort with cudaMallocAsync.

    Patch only its tuner's timing entry point, not the kernels or torch.compile.
    Native allocation retains upstream graph timing; already cached choices work.
    """
    from lycoris.kernels.tuner import measure

    original = measure.time_candidates
    if getattr(original, '_bluflow_allocator_guard', False):
        return
    announced = False

    @wraps(original)
    def time_candidates(makers, sets, dev, rounds=4, use_graph=True, prune=2.0, errors=None):
        nonlocal announced
        if dev.kind == 'cuda' and torch.cuda.memory.get_allocator_backend() == 'cudaMallocAsync':
            use_graph = False
            if not announced:
                print('LyCORIS: using CUDA-event tuning without graph capture for cudaMallocAsync; '
                      'fused adapter kernels remain enabled.', flush=True)
                announced = True
        return original(makers, sets, dev, rounds=rounds, use_graph=use_graph,
                        prune=prune, errors=errors)

    time_candidates._bluflow_allocator_guard = True
    measure.time_candidates = time_candidates


class ContiguousLoConInput:
    def bypass_forward_diff(self, x, scale=1):
        # Pinned LyCORIS's atomic bypass backward allocates empty_like(x), but
        # indexes that gradient using x's strides. Sliced/broadcast inputs can
        # therefore write outside the allocation (verified with compute-sanitizer).
        # Keep the frozen base unchanged; normalize only the adapter input.
        # Already contiguous activations incur no copy.
        return super().bypass_forward_diff(x.contiguous(), scale=scale)


def algorithm_classes():
    global _ADAPTER_CLASSES
    if _ADAPTER_CLASSES is None:
        from lycoris.modules import LoConModule, LokrModule

        install_allocator_safe_timing()
        base = dict(locon=LoConModule, lokr=LokrModule, dora=LoConModule)
        _ADAPTER_CLASSES = {
            k: type("Mage" + k.title(),
                    (SDNQWeightAccess, ContiguousLoConInput, cls)
                    if k in ("locon", "dora") else (SDNQWeightAccess, cls), {})
            for k, cls in base.items()
        }
    return _ADAPTER_CLASSES


def apply_lycoris_lora(model, cfg):
    from .params import adapter_target_names, classify

    cfg._validate_lora_targets()
    targets = [
        (name, model.get_submodule(name))
        for name in adapter_target_names(
            model, cfg.components, moss_train_text_refiner=cfg.moss_train_text_refiner
        )
    ]
    model.requires_grad_(False)
    for name, module in targets:
        if classify(name + ".weight") == "adaln":
            raise ValueError("AdaLN must remain frozen for LoRA")
        # Bypass keeps SDNQ's frozen-base forward intact; only the low-rank
        # activation path is dispatched through LyCORIS kernels.
        wd = cfg.lycoris_algo in DECOMPOSED
        source = module
        if wd:
            # Materialize one target only to initialize the magnitude; release it immediately.
            source = torch.nn.Linear(
                module.in_features, module.out_features, bias=False, device="meta"
            )
            source.weight = torch.nn.Parameter(
                dense_weight(module), requires_grad=False
            )
        adapter = algorithm_classes()[cfg.lycoris_algo](
            name,
            source,
            lora_dim=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            bypass_mode=cfg.lycoris_bypass and not wd,
            weight_decompose=wd,
            wd_on_out=cfg.lycoris_wd_on_output,
            factor=cfg.lokr_factor,
            decompose_both=cfg.lokr_decompose_both,
        )
        adapter.org_module = [module]
        adapter.org_forward = module.forward
        adapter.bypass_mode = cfg.lycoris_bypass and not wd
        del source
        alpha = (
            adapter.alpha.detach().clone().float()
            if hasattr(adapter, "alpha")
            else None
        )
        adapter.to(device=module.weight.device, dtype=getattr(torch, cfg.dtype))
        # Preserve the algorithm's effective alpha (LoKr can normalize its scale).
        if alpha is not None:
            adapter.alpha = alpha.to(module.weight.device)
        module.add_module("lycoris_adapter", adapter)
        adapter.apply_to()
    model._lycoris_config = asdict(cfg)
    return model


def lycoris_state_dict(model):
    """Native dotted Mage-Flow targets with LyCORIS down/up/alpha tensor names."""
    return {
        "diffusion_model." + name.removesuffix(".lycoris_adapter") + "." + key: value
        for name, module in model.named_modules()
        if name.endswith(".lycoris_adapter")
        for key, value in module.state_dict().items()
    }


def load_lycoris_state_dict(model, state):
    expected = lycoris_state_dict(model)
    if set(state) != set(expected):
        raise ValueError(
            "LyCORIS checkpoint targets do not match this adapter configuration"
        )
    for name, module in model.named_modules():
        if name.endswith(".lycoris_adapter"):
            prefix = "diffusion_model." + name.removesuffix(".lycoris_adapter") + "."
            module.load_state_dict(
                {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
                strict=True,
            )
