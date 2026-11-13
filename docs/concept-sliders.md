# Concept sliders

A concept slider is an ordinary adapter trained so that strength **+1** pushes images toward one
prompt, **-1** toward the opposite prompt, and **0** leaves the base model alone. Load it like any
other LoRA and move its strength between -1 and +1 (or past them) to dial the concept.

This is a port of ostris/ai-toolkit's Concept Slider trainer
(`extensions_built_in/concept_slider`). It works with every model family and adapter backend the
trainer supports, except DoRA, and on SDNQ-quantized bases, including the MOSS `balanced` and
`fast` INT8 presets.

## Quick start

Add a `[slider]` section to an adapter config:

```toml
[train]
cache_text_embeddings = true   # encodes only the 3-4 slider prompts, then frees the encoder

[adapter]
kind = "lora"                  # or "lokr", or "lycoris_lora" with locon/lokr (not dora)

[slider]
enabled = true
target_class = "person"
positive_prompt = "person who is happy"
negative_prompt = "person who is sad"
# anchor_class = "dog"         # optional: a concept the slider must not change
# guidance_strength = 3.0
# anchor_strength = 1.0
```

In the GUI the same settings are under **Method → Concept Slider**. A full example is
[`configs/moss-image-concept-slider.toml`](../configs/moss-image-concept-slider.toml).

**Dataset.** Images only supply realistic noisy latents, so use a few dozen to a few hundred
images of the target class. Their captions are ignored: caption files, caption modes, shuffling
and caption variations have no effect in slider mode. Images need no `.txt` files at all: slider
mode implies `dataset.allow_missing_captions`, so latent caching (including the GUI's cache step
before training) and latents-only datasets accept uncaptioned images.

## Settings

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Train a slider instead of fitting the dataset. |
| `target_class` | `""` | The concept the slider acts on. The adapter is always trained with this prompt. Empty means the unconditional prompt. |
| `positive_prompt` | `""` | Where strength +1 moves the target class. |
| `negative_prompt` | `""` | Where strength -1 moves it. Must differ from `positive_prompt`. |
| `anchor_class` | `""` | Optional concept held at the base model at both ends. Empty disables the anchor. |
| `guidance_strength` | `3.0` | How far each end is pushed (`g` below). |
| `anchor_strength` | `1.0` | Weight of the anchor term. `0` disables the anchor. |

The log line gains `pos`, `neg` and, with an anchor, `anchor`: the weighted terms of `loss` for
each slider end.

## How a step works

For the step's noisy latents `x` (noise, timesteps, curriculum range and bucketing are exactly
those of a normal step), with the adapter at strength 0 and no gradients:

```
pos     = v(x, positive_prompt)
neutral = v(x, target_class)
neg     = v(x, negative_prompt)
anchor  = v(x, anchor_class)                     (if enabled)

T+ = renorm(neutral + g * (pos - neg))           T- = renorm(neutral - g * (pos - neg))
```

`renorm` rescales a target to `neutral`'s mean and standard deviation. Then the adapter, prompted
with `target_class`, regresses onto `T+` at strength +1 and onto `T-` at strength -1. With an
anchor, the anchor prompt's prediction at both strengths regresses onto the base `anchor`
prediction. The loss is ai-toolkit's:

```
loss = 0.5 * (L(+1) + L(-1)),   L(s) = (2 * mse(class, T_s) + anchor_strength * mse(anchor_s, anchor)) / 3
```

so learning rates from ai-toolkit slider recipes carry over.

**Cost.** A step runs 3 reference forwards (4 with an anchor) without gradients, plus one forward
and backward per slider end (two more with an anchor). Reference forwards run one prompt at a time
at the training batch's shape, so peak memory stays at a normal step's and compiled graphs and
SDNQ INT8 kernels see no new shapes. Measured on MOSS-Image Base, RTX 4080 SUPER, `balanced`
preset, LyCORIS LoCon, compiled, 512px, batch 1: a slider step took about 1.5 s against 0.6 s for
a plain LoRA step (about 2.5x), at the same 8.9 GiB peak. An anchor adds one reference forward and
two forward/backward passes, so expect roughly twice the slider cost with one. `torch.compile`
builds separate graphs for strength 0, +1 and -1 on the first step.

**Why each slider end has its own backward.** The sign lives in the adapter, not in the autograd
graph. Gradient checkpointing re-runs block forwards during backward, so a pass whose backward ran
after the sign flipped would recompute the wrong function. Each pass finishes its backward before
the next flips the sign; under DDP the early passes run under `no_sync`, so gradients are still
reduced once per step.

## Compatibility

- **Adapters:** PEFT LoRA and LoKr, LyCORIS LoCon and LoKr (bypass or not). Strength scales only
  the adapter branch (PEFT `scaling`, LyCORIS `multiplier`); it is never merged into the stored
  base. Keep LyCORIS in its default bypass mode: weight mode adds `±delta` to BF16 weights on the
  fly, so a small delta rounds differently at +1 and -1.
- **SDNQ and other quantized bases:** frozen SDNQ storage, the MOSS `balanced`/`fast` frozen INT8
  matmul presets and ConvRot all keep their base layers untouched; slider mode changes nothing
  about how they quantize, store or multiply.
- **Not supported**, refused at load: full finetuning (`adapter.kind = "none"`), DoRA (its
  magnitude normalisation makes strength 0 not "off" and -1 not the mirror of +1),
  `optimizer.gradient_release` (it would update weights between the two slider ends),
  concept preservation, Self-Flow, decoded perceptual and eye adversarial losses, paired texture
  loss, HF loss, RTI and texture curriculum phases.
- **Ignored**, accepted so a normal config can be reused: caption files and every
  `[dataset.caption]` option, and `train.caption_variations` (no variation cache is built).
- Packed resolutions, multi-resolution buckets, staged resolution training, dual timestep
  noising, gradient accumulation and Linux DDP work as usual.

## Differences from ai-toolkit

- ai-toolkit computes `enhance` and `erase` losses that are the same target; the weighting is kept,
  the duplicate forward is not.
- Target renormalisation is per sample rather than over the whole batch: samples in one batch sit
  at different timesteps, and pooling their statistics mixes noise levels. At batch size 1 the two
  are identical.
- Reference predictions run per prompt instead of as one 3-4x batch.
- An empty `anchor_class` disables the anchor. ai-toolkit's UI sends an empty string by default,
  which there anchors the empty prompt; set `anchor_class` explicitly if you want an anchor.
