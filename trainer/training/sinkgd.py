"""SinkGD as an SDNQ optimizer: stateless SR-Sinkhorn gradient normalization.

Paper: https://arxiv.org/abs/2502.06742 (Gradient Multi-Normalization for Efficient LLM Training).

For a weight matrix with gradient G (m x n) each step does

    X <- G
    repeat L times:
        X <- sqrt(n) * X / rownorm(X)      # every row has l2 norm sqrt(n)
        X <- sqrt(m) * X / colnorm(X)      # every column has l2 norm sqrt(m)
    W <- W - lr * X

The output has RMS ~1 per element (like sign descent), so the LR scale is Adam-like.

Building on `SDNQOptimizer` rather than a bare `torch.optim.Optimizer` is what makes this work with
SDNQ weights. The base class's `get_param_grad` dequantizes an `SDNQTensor` master weight to fp32,
and `update_param_` writes the result back with stochastic rounding, optional Kahan summation and
decoupled weight decay -- so full finetuning in SDNQ *training* mode and LoRA over a frozen SDNQ
base both go through the same path as SDNQ's AdamW/Lion/Muon. SinkGD itself keeps no momentum or
second-moment buffers; the only state is the Kahan residual when `use_kahan` is on.

Non-matrix parameters need a convention, because row/column norms only exist for 2-D tensors:
  * >2-D (conv kernels, LyCORIS factors) are viewed as (shape[0], -1), like Adv-Optm does;
  * 1-D (norm gains, DoRA magnitudes, biases) are rescaled to RMS 1, i.e. normalized SGD.
This keeps the optimizer fully stateless, where the paper falls back to Adam for those tensors.
"""
from __future__ import annotations

import math

import torch
from sdnq.optim.optimizer import SDNQOptimizer


@torch.no_grad()
def sr_sinkhorn(grad: torch.Tensor, iters: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """Row/column l2 normalization, `iters` rounds. Returns a new fp32 tensor of grad's shape.

    eps matters: a zero-initialized LoRA B makes A's gradient exactly zero on step 1, and a zero
    row or column would otherwise divide by zero. With the clamp those entries stay zero.
    """
    if type(iters) is not int or iters < 1:
        raise ValueError("sinkhorn_iterations must be a positive integer")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Sinkhorn epsilon must be finite and positive")
    if grad.is_sparse:
        raise ValueError("SinkGD does not support sparse gradients")
    if not grad.numel():
        return grad.float().clone()
    shape = grad.shape
    x = grad.to(dtype=torch.float32)
    if x.ndim == 0:
        return x.sign()
    if x.ndim == 1:
        return x * (math.sqrt(x.numel()) / x.norm().clamp_min(eps))
    x = x.reshape(shape[0], -1)
    m, n = x.shape
    for _ in range(iters):
        x = x * (math.sqrt(n) / x.norm(dim=1, keepdim=True).clamp_min(eps))
        x = x * (math.sqrt(m) / x.norm(dim=0, keepdim=True).clamp_min(eps))
    return x.reshape(shape)


class SinkGD(SDNQOptimizer):
    _extra_group_keys = {"sinkhorn_iterations"}  # noqa: RUF012
    _keep_in_fp32_keys = set()  # noqa: RUF012
    _group_keys = set.union(SDNQOptimizer._base_group_keys, _extra_group_keys)

    def __init__(self, params, **kwargs):
        # Defaults that differ from SDNQ's AdamW-shaped ones. The SR-Sinkhorn output has RMS 1, so
        # SDNQ's default "clip" final norm mode (clamp to +-1) would truncate it, and the default
        # gradient clamp would distort the input. Both are kept as ordinary overridable kwargs.
        kwargs.setdefault("final_norm_mode", "none")
        kwargs.setdefault("clip_threshold", (1e4, 1e-3, 1e-3))
        kwargs.setdefault("weight_decay", 0.0)
        kwargs.setdefault("betas", ())
        kwargs.setdefault("sinkhorn_iterations", 5)
        unknown = set(kwargs) - self._group_keys
        if unknown:
            raise ValueError(f"Unknown SinkGD options: {sorted(unknown)}")
        if isinstance(params, torch.Tensor):
            raise TypeError("SinkGD parameters must be an iterable, not a single tensor")
        params = list(params)
        if not params:
            raise ValueError("SinkGD got an empty parameter list")
        param_groups = ([dict(group) for group in params] if isinstance(params[0], dict)
                        else [{"params": params}])
        for group in param_groups:
            group = self.apply_group_defaults(group, **kwargs)
            group["sinkhorn_iterations"] = self.get_default_kwarg(
                group, kwargs, "sinkhorn_iterations", 5)
            if type(group["sinkhorn_iterations"]) is not int or group["sinkhorn_iterations"] < 1:
                raise ValueError("sinkhorn_iterations must be a positive integer")
            if set(group) != self._group_keys:
                raise ValueError(f"Unknown SinkGD group options: {sorted(set(group) - self._group_keys)}")
            if not math.isfinite(group["lr"]) or group["lr"] < 0:
                raise ValueError("SinkGD lr must be finite and nonnegative")
            if not math.isfinite(group["weight_decay"]) or group["weight_decay"] < 0:
                raise ValueError("SinkGD weight_decay must be finite and nonnegative")
            if group["betas"]:
                raise ValueError("SinkGD does not use betas/momentum")
        super().__init__(param_groups, {})

    @torch.no_grad()
    def init_state(self, param: torch.Tensor, group: dict, state: dict) -> dict:
        return state  # stateless; SDNQOptimizer adds the Kahan buffer itself when requested

    @torch.no_grad()
    def get_param_update(self, param_fp32: torch.FloatTensor, grad: torch.FloatTensor,
                         group: dict, state: dict) -> torch.FloatTensor:
        return sr_sinkhorn(grad, group["sinkhorn_iterations"])
