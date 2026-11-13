"""Concept sliders, ported from ostris/ai-toolkit's `ConceptSliderTrainer`.

A slider is an ordinary adapter trained so that strength +1 pushes images toward one prompt and
strength -1 pushes them toward the opposite one, while 0 leaves the base model alone. The dataset
images only supply realistic noisy latents; their captions are not used.

Each step, with the adapter contributing nothing and no gradients, the model is run on the step's
noisy latents with the fixed prompts:

    pos     = v(x, positive_prompt)
    neutral = v(x, target_class)
    neg     = v(x, negative_prompt)
    anchor  = v(x, anchor_class)            (optional)

and the two ends of the slider get the targets

    T+ = neutral + g * (pos - neg)          T- = neutral - g * (pos - neg)

each rescaled to `neutral`'s mean and standard deviation. Then the adapter at +1, prompted with
`target_class`, regresses onto T+; at -1 onto T-. With an anchor, the anchor prompt's prediction at
both ends regresses onto the base model's own anchor prediction, which keeps the slider from
dragging an unrelated concept along.

Differences from the ai-toolkit trainer, all deliberate:

* ai-toolkit computes `enhance` and `erase` losses that are algebraically the same target, so its
  per-sign loss is `(2 * mse(class) + anchor_strength * mse(anchor)) / 3`. That weighting is kept
  so learning rates carry over; the duplicate forward is not.
* The renormalisation runs per sample instead of over the whole batch. Samples in one batch sit at
  different timesteps, and pooling their statistics mixes noise levels. At batch size 1 the two
  are identical.
* Reference predictions run one prompt at a time at the training batch's own shape instead of as
  one batch of 3-4x the size. Peak memory stays at the training forward's, and compiled graphs and
  SDNQ's quantized-matmul kernels see no new shapes.
* An empty `anchor_class` means no anchor. (ai-toolkit's UI sends an empty string, which there
  anchors the empty prompt.)

**Why every signed pass gets its own backward.** The sign lives in the adapter, not in the
autograd graph. Gradient checkpointing re-runs the forward during backward, so a pass whose
backward ran after the sign flipped would recompute the wrong function. Each pass therefore
finishes its backward before the next one changes the sign; the final pass's backward is the
trainer's ordinary one, and `after_backward()` restores the adapter afterwards. Under DDP the early
passes run inside `no_sync`, so the step still reduces gradients once.

**SDNQ.** Only the adapter branch is scaled: PEFT's per-layer `scaling` and LyCORIS's
`multiplier`. Quantized base layers (SDNQ frozen weights, the frozen INT8 matmul path, ConvRot)
are never merged into, unmerged or otherwise written. DoRA is refused because its magnitude
normalisation makes neither "0 = off" nor "-1 = mirror image" hold.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, ContextManager, Sequence

import torch


@dataclass
class SliderConfig:
    enabled: bool = False
    # The class the slider acts on, e.g. "person". The adapter is always trained with this prompt.
    # Empty is allowed and means the unconditional (empty) prompt.
    target_class: str = ""
    # The two ends of the slider: +1 moves toward positive_prompt, -1 toward negative_prompt.
    positive_prompt: str = ""
    negative_prompt: str = ""
    # Optional concept to hold still while the slider moves. Empty disables the anchor.
    anchor_class: str = ""
    # g in T = neutral +/- g * (pos - neg). ai-toolkit's default.
    guidance_strength: float = 3.0
    # Weight of the anchor term relative to the class term (ai-toolkit's weighting).
    anchor_strength: float = 1.0

    def __post_init__(self):
        for name in ("target_class", "positive_prompt", "negative_prompt", "anchor_class"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"[slider] {name} must be a string")
        for name in ("guidance_strength", "anchor_strength"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"[slider] {name} must be a finite number, got {value!r}")
            setattr(self, name, float(value))
        if not self.enabled:
            return
        if self.positive_prompt.strip() == self.negative_prompt.strip():
            raise ValueError(
                "[slider] positive_prompt and negative_prompt are identical, so the slider "
                "direction is zero. Give the two ends of the slider, e.g. 'person who is happy' "
                "and 'person who is sad'."
            )
        if self.guidance_strength == 0:
            raise ValueError("[slider] guidance_strength must be non-zero")
        if self.anchor_strength < 0:
            raise ValueError("[slider] anchor_strength must be >= 0")

    @property
    def use_anchor(self) -> bool:
        return bool(self.anchor_class.strip()) and self.anchor_strength > 0

    def prompts(self) -> dict[str, str]:
        """Role -> prompt for every prompt the slider encodes."""
        out = {
            "positive": self.positive_prompt,
            "neutral": self.target_class,
            "negative": self.negative_prompt,
        }
        if self.use_anchor:
            out["anchor"] = self.anchor_class
        return out


def validate_slider_options(cfg) -> None:
    """Cross-section checks for `[slider]`; called from `validate_model_options`."""
    if not cfg.slider.enabled:
        return
    adapter = cfg.adapter
    if adapter.kind == "none":
        raise ValueError(
            "[slider] trains an adapter whose strength is flipped between +1 and -1, so it needs "
            "adapter.kind = 'lora', 'lokr' or 'lycoris_lora'; a full finetune has no strength to flip"
        )
    if adapter.kind == "lycoris_lora" and adapter.lycoris_algo == "dora":
        raise ValueError(
            "[slider] does not support DoRA: its magnitude normalisation means strength 0 does not "
            "switch the adapter off and -1 is not the mirror of +1. Use lycoris_algo = 'locon' or "
            "'lokr', or adapter.kind = 'lora'"
        )
    if cfg.optimizer.gradient_release:
        raise ValueError(
            "[slider] runs one backward per slider end; optimizer.gradient_release would update "
            "the weights between them. Disable gradient_release"
        )
    # Dataset caption options (caption modes, shuffling, train.caption_variations) are simply
    # unused: slider steps never read dataset captions and the text cache holds only the slider
    # prompts, so leaving them set in a reused config is harmless and is not refused.
    unsupported = [
        name for name, on in (
            ("preserve", cfg.preserve.enabled),
            ("self_flow", cfg.self_flow.enabled),
            ("perceptual", cfg.perceptual.enabled),
            ("adversarial", cfg.adversarial.enabled),
            ("paired_texture", bool(cfg.paired_texture.scale)),
            ("flow.hf_scale", bool(cfg.flow.hf_scale)),
            ("rti", cfg.rti.enabled),
            ("texture curriculum phases", any(p.mode == "texture" for p in cfg.curriculum.phases)),
        ) if on
    ]
    if unsupported:
        raise ValueError(
            f"[slider] replaces the flow-matching target with slider targets, so these "
            f"dataset-target losses and modes cannot be combined with it: {', '.join(unsupported)}"
        )


# --------------------------------------------------------------------------- adapter strength


class AdapterStrength:
    """Signed strength for every adapter layer in a model.

    0 makes each adapter contribute exactly nothing (its branch is multiplied by zero), +1 is the
    trained adapter as saved, -1 is its mirror image. Only the adapter branch is touched; base
    layers are left exactly as they are, which is what keeps quantized SDNQ bases safe.
    """

    def __init__(self, model: torch.nn.Module):
        from peft.tuners.tuners_utils import BaseTunerLayer

        self._peft: list[tuple[torch.nn.Module, dict]] = []
        self._lycoris: list[tuple[torch.nn.Module, float]] = []
        for name, module in model.named_modules():
            if isinstance(module, BaseTunerLayer) and isinstance(getattr(module, "scaling", None), dict):
                if getattr(module, "merged", False):
                    raise RuntimeError(f"[slider] adapter at {name} is merged into its base layer")
                self._peft.append((module, dict(module.scaling)))
            elif name.endswith("lycoris_adapter") and hasattr(module, "multiplier"):
                self._lycoris.append((module, float(module.multiplier)))
        if not self._peft and not self._lycoris:
            raise RuntimeError("[slider] found no PEFT or LyCORIS adapter layers to scale")
        self.value = 1.0

    def __len__(self) -> int:
        return len(self._peft) + len(self._lycoris)

    def set(self, value: float) -> None:
        value = float(value)
        for module, base in self._peft:
            for adapter, scale in base.items():
                module.scaling[adapter] = scale * value
        for module, base in self._lycoris:
            module.multiplier = base * value
        self.value = value

    def restore(self) -> None:
        self.set(1.0)


# --------------------------------------------------------------------------- slider math


Prediction = torch.Tensor | Sequence[torch.Tensor]


def per_sample(pred: Prediction) -> list[torch.Tensor]:
    """A batched tensor or a packed list -> one FP32 tensor per sample."""
    if isinstance(pred, torch.Tensor):
        return [p.float() for p in pred.unbind(0)]
    return [p.float() for p in pred]


def norm_like(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """`tensor` rescaled to `target`'s mean and standard deviation (ai-toolkit `norm_like_tensor`)."""
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8) * target.std() + target.mean()


def slider_targets(
    positive: list[torch.Tensor],
    neutral: list[torch.Tensor],
    negative: list[torch.Tensor],
    guidance: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-sample targets for strength +1 and -1."""
    plus, minus = [], []
    for p, n, q in zip(positive, neutral, negative, strict=True):
        direction = guidance * (p - q)
        plus.append(norm_like(n + direction, n))
        minus.append(norm_like(n - direction, n))
    return plus, minus


def mean_mse(pred: Prediction, targets: list[torch.Tensor]) -> torch.Tensor:
    """Mean over samples of each sample's MSE, so every image weighs the same in a packed batch."""
    preds = per_sample(pred)
    return torch.stack([
        torch.nn.functional.mse_loss(p, t) for p, t in zip(preds, targets, strict=True)
    ]).mean()


@dataclass
class SliderPass:
    strength: float
    prompt: str          # role key into the encoded contexts
    targets: list[torch.Tensor]
    weight: float
    stat: str


def slider_passes(cfg: SliderConfig, reference: dict[str, list[torch.Tensor]]) -> list[SliderPass]:
    """The signed passes of one step, in execution order.

    Weights reproduce ai-toolkit's `0.5 * (loss(+1) + loss(-1))` with
    `loss(s) = (2 * mse(class) + anchor_strength * mse(anchor)) / 3`.
    """
    plus, minus = slider_targets(
        reference["positive"], reference["neutral"], reference["negative"], cfg.guidance_strength
    )
    out = [SliderPass(1.0, "neutral", plus, 1.0 / 3.0, "slider_pos")]
    if cfg.use_anchor:
        out.append(SliderPass(1.0, "anchor", reference["anchor"], cfg.anchor_strength / 6.0, "slider_anchor"))
    out.append(SliderPass(-1.0, "neutral", minus, 1.0 / 3.0, "slider_neg"))
    if cfg.use_anchor:
        out.append(SliderPass(-1.0, "anchor", reference["anchor"], cfg.anchor_strength / 6.0, "slider_anchor"))
    return out


def slider_step(
    cfg: SliderConfig,
    strength: AdapterStrength,
    forward: Callable[[str], Prediction],
    early_backward: Callable[[torch.Tensor], None],
    early_context: Callable[[], ContextManager] = nullcontext,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one slider step.

    `forward(role)` runs the model on this step's noisy latents with the prompt for `role`
    (`positive`, `neutral`, `negative` or `anchor`), at whatever strength is currently set.
    Every pass but the last is backpropagated here through `early_backward`, inside
    `early_context()` (the caller's DDP `no_sync`). The returned loss carries the last pass's graph
    for the caller's own backward and the value of the whole step for logging. The adapter is left
    at the last pass's strength; call `strength.restore()` once that backward has run.
    """
    try:
        strength.set(0.0)
        with torch.no_grad():
            reference = {role: [p.detach() for p in per_sample(forward(role))]
                         for role in cfg.prompts()}
        passes = slider_passes(cfg, reference)
        done = None
        # Detached tensors, not floats: `.item()` here would add a device sync per pass.
        stats: dict[str, torch.Tensor] = {}
        for i, p in enumerate(passes):
            last = i == len(passes) - 1
            strength.set(p.strength)
            with nullcontext() if last else early_context():
                term = p.weight * mean_mse(forward(p.prompt), p.targets)
                if not last:
                    early_backward(term)
            value = term.detach()
            stats[p.stat] = value if p.stat not in stats else stats[p.stat] + value
            if last:
                loss = term if done is None else term + done
            else:
                done = value if done is None else done + value
        return loss, stats
    except BaseException:
        strength.restore()
        raise


class ConceptSlider:
    """Trainer-side glue: encodes the fixed prompts once and owns the adapter strength."""

    def __init__(self, cfg: SliderConfig, trainer):
        from .train import substitute_empty

        self.cfg = cfg
        self.trainer = trainer
        self.strength = AdapterStrength(trainer._unwrap(trainer.transformer))
        prompts = cfg.prompts()
        with torch.no_grad():
            # One prompt per call so each keeps its own length; no padding is shared between them.
            self.contexts = {
                role: tuple(t.detach() for t in trainer._encode(substitute_empty([prompt])))
                for role, prompt in prompts.items()
            }

    @staticmethod
    def cached_prompts(cfg: SliderConfig) -> list[str]:
        """Exact strings `_encode` will look up, for building a static text cache."""
        from .train import substitute_empty

        return sorted(set(substitute_empty(list(cfg.prompts().values()))))

    def context(self, role: str, n: int):
        hidden, mask = self.contexts[role]
        return hidden.expand(n, *hidden.shape[1:]).contiguous(), mask.expand(n, *mask.shape[1:]).contiguous()

    def after_backward(self) -> None:
        self.strength.restore()

    def describe(self) -> str:
        c = self.cfg
        anchor = f", anchor {c.anchor_class!r} x{c.anchor_strength:g}" if c.use_anchor else ""
        return (f"slider   {c.negative_prompt!r} <- -1 | {c.target_class!r} | +1 -> {c.positive_prompt!r}, "
                f"guidance {c.guidance_strength:g}{anchor}; {len(self.strength)} adapter layer(s)")
