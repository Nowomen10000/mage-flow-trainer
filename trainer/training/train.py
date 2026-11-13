"""Mage-Flow training: full finetuning and PEFT adapters, SDNQ, Accelerate and bucketed datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import replace
from itertools import islice, accumulate
from pathlib import Path

from .timing import TrainingTiming, utc_now

_SESSION_STARTED_AT = utc_now()
_SESSION_STARTED_COUNTER = time.perf_counter()

# Must precede `import torch`: the caching allocator parses this once, at first CUDA use.
# Bucketing hands the allocator a different tensor shape per bucket (55 of them on a two-tier
# texture run), which is the textbook way to fragment it -- measured 3.35 GB of reserved-but-
# unallocated on a shape-varying probe, versus 0.74 GB with expandable segments. That gap is the
# difference between fitting and OOM at the batch sizes this trainer targets on Linux.
# The pinned Windows wheel omits support. Preserve either explicit allocator env spelling.
from .allocator import configure_allocator
configure_allocator()

# Triton loads this lazily in compiler workers too. Rebuild corrupt group indexes
# without discarding healthy kernels or overriding a user's custom cache manager.
os.environ.setdefault("TRITON_CACHE_MANAGER",
                      "trainer.training.triton_cache:RecoveringFileCacheManager")

# SDNQ may initialize Triton while importing. Choose the Windows C launcher
# compiler before that happens; MSVC remains available for Inductor's C++ code.
from ..tools.windows_toolchain import configure_windows_triton_compiler
configure_windows_triton_compiler()

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import MageFlowDataset, BucketBatchSampler, collate
from ..modeling.loader import model_load_kwargs, load_components, encode_prompts
from .config import Config, load_config
from .flow import flow_loss, hf_loss, prepare_training_flow_batch, effective_timesteps
from .optim import build_optimizer, build_scheduler, estimate_optimizer_bytes
from .params import (
    apply_adapter,
    build_adapter_param_groups,
    build_param_groups,
    count_parameters,
)
from .preserve import ConceptPreserver
from .quant import (
    quantize_module,
    quantized_layer_report,
    resolve_quantized_matmul,
    tokens_for_bucket,
    text_encoder_quant_config,
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _peft_state_dict(module) -> dict[str, torch.Tensor]:
    """Adapter tensors only. `get_peft_model_state_dict` is not usable here because diffusers'
    `add_adapter` injects layers without wrapping the module in a PeftModel."""
    return {k: v for k, v in module.state_dict().items() if "lora_" in k or "lokr_" in k}


def _emit(obj, msg: str) -> None:
    """Print without tearing the progress bar.

    Module-level and `getattr`-guarded on purpose: `_prune_checkpoints` is exercised by the gate
    against a duck-typed stub that has no `bar`, and a bound method would make that stub carry
    display state it has nothing to do with.
    """
    if getattr(obj, "bar", None) is not None:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _fmt_duration(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


UNCONDITIONAL = " "


def substitute_empty(captions: list[str], filler: str = UNCONDITIONAL) -> list[str]:
    """Replace captions that would tokenize to nothing. See `Trainer._encode` for why."""
    return [c if c.strip() else filler for c in captions]


def require_cuda() -> None:
    """Refuse to start on CPU.

    Torch does not fail on a bad `CUDA_VISIBLE_DEVICES`; it reports zero devices, Accelerate falls
    back to gloo, and a 2B model trains on the CPU at roughly a thousandth of the speed while
    printing a completely normal-looking startup summary. That is the worst failure mode available
    -- it looks like it is working.

    Observed with `CUDA_VISIBLE_DEVICES=all`, which is not a device list. The message names the
    variable and its value because that is nearly always what is wrong.
    """
    import os

    if torch.cuda.is_available():
        return
    if os.environ.get("MAGE_FLOW_ALLOW_CPU"):
        print("WARNING: no CUDA device; continuing because MAGE_FLOW_ALLOW_CPU is set.", flush=True)
        return

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    detail = ""
    if cvd is not None:
        detail = (f"\nCUDA_VISIBLE_DEVICES is set to {cvd!r}. It must be a comma-separated list of "
                  f"device indices (e.g. '0' or '0,1'); anything else -- including 'all' and an "
                  f"empty string -- hides every GPU.")
    raise RuntimeError(
        "no CUDA device is visible, so this would train on the CPU." + detail +
        "\nSet MAGE_FLOW_ALLOW_CPU=1 to override."
    )


def _text_cache_batches(items, size):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


class Trainer:
    def __init__(self, cfg: Config, config_path: str | Path | None = None):
        self.timing = TrainingTiming(_SESSION_STARTED_AT, _SESSION_STARTED_COUNTER)
        from .config import validate_model_options
        validate_model_options(cfg)
        from ..tools.windows_toolchain import prepare_windows_toolchain, check_windows_triton_compiler, check_windows_cpp_language
        prepare_windows_toolchain(required=bool(cfg.train.compile or cfg.train.compile_text_encoder))
        if cfg.train.compile or cfg.train.compile_text_encoder:
            check_windows_cpp_language(install_missing=True)
        check_windows_triton_compiler()
        require_cuda()
        self.cfg = cfg
        self.config_path = Path(config_path) if config_path else None
        self.dtype = _DTYPES[cfg.train.dtype]

        # `mixed_precision="no"` is explicit, not a default being restated. Left unset, Accelerate
        # reads `~/.cache/huggingface/accelerate/default_config.yaml`, and a `mixed_precision: bf16`
        # there silently wraps every forward in autocast -- but ONLY under `accelerate launch`, so
        # `python -m trainer.training.train` and the GUI (which always launches through accelerate)
        # would run different numerics from the same config file.
        #
        # Modules are already loaded in train.dtype. SDNQ supplies its own compiled
        # backward, and an extra autocast wrapper can give its inputs and grad_output
        # different dtypes. This previously failed on the first backward under
        # accelerate launch; explicit mixed_precision="no" keeps both launch paths aligned.
        # INT8 training uses quantized storage with BF16 logical weights/gradients when
        # train.dtype is BF16, plus FP32 intermediates in the optimizer and loss.
        #
        # Precision is `train.dtype`'s job. Making that the single control is what keeps the CLI
        # and the GUI running the same arithmetic.
        env_mp = os.environ.get("ACCELERATE_MIXED_PRECISION")
        if env_mp and env_mp.lower() != "no":
            print(f"note: ignoring mixed_precision={env_mp!r} from the accelerate environment; "
                  f"this trainer sets precision through train.dtype (={cfg.train.dtype}) and "
                  f"autocast breaks SDNQ's training-mode backward.", flush=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            log_with=None,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),
                             self._process_group_kwargs()],
            mixed_precision="no",
            dynamo_plugin=self._dynamo_plugin(),
        )
        if (cfg.quant.mode == "training" or cfg.quant.backend == "convrot" or cfg.quant.frozen_matmul) and self.accelerator.num_processes > 1:
            from .distributed import configure_quantized_ddp
            self.accelerator.ddp_handler = configure_quantized_ddp()
        from .distributed import configure_moss_checkpoint_ddp
        configure_moss_checkpoint_ddp(cfg.train, self.accelerator.num_processes)
        from .attention_check import check_attention_backend
        check_attention_backend(cfg.train.attention_backend, self.accelerator.device, self.dtype)
        from .int8_calibration import prepare_training
        prepare_training(cfg, self.accelerator)
        # Rank-local global RNG still serves dropout and other model operations.
        # DDP synchronizes initial model/adapter weights. Flow draws use separate
        # streams below; stochastic optimizer updates use shared isolated RNG.
        set_seed(cfg.train.seed, device_specific=True)
        from .flow_rng import FlowRNG
        self.flow_rng = FlowRNG(cfg.train.seed, self.accelerator.process_index,
                                self.accelerator.num_processes, self.accelerator.device)

        self.out_dir = Path(cfg.train.output_dir) / cfg.train.run_name
        if cfg.train.resume_mode == "weights":
            from .adapter_loading import validate_new_phase_output
            from accelerate.utils import broadcast_object_list
            # Check once, before any rank creates the run/config. Otherwise a
            # slower rank can mistake rank zero's new files for an older run.
            error = [None]
            if self.accelerator.is_main_process:
                try:
                    validate_new_phase_output(cfg)
                except (ValueError, OSError) as exc:
                    error[0] = str(exc)
            broadcast_object_list(error)
            if error[0]:
                raise ValueError(error[0])
        if self.accelerator.is_main_process:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # The config that produced these checkpoints, copied verbatim so its comments survive.
            # Checkpoints outlive the configs directory -- a file gets edited for the next
            # experiment and the run that produced a LoRA becomes unreconstructable. Copied rather
            # than re-serialised for exactly that reason: a round trip through the dataclasses
            # would emit the resolved values and drop every comment explaining them.
            if self.config_path and self.config_path.is_file():
                dest = self.out_dir / f"{cfg.train.run_name}.toml"
                if self.config_path.resolve() != dest.resolve():
                    shutil.copy2(self.config_path, dest)

        # None when no curriculum is configured, which is the default and is exactly today's
        # behaviour: `t_range=None` reaches the sampler and no range mapping happens at all.
        self.phase = None
        # NCCL has no Windows build, so torch falls back to gloo, which does not support the GPU
        # collectives DDP needs here. Caught with a sentence rather than left to surface as a
        # backend error from inside `accelerator.prepare`.
        if self.accelerator.num_processes > 1 and sys.platform == "win32":
            raise RuntimeError(
                f"multi-GPU training is not available on Windows: DDP requires NCCL, which is "
                f"Linux-only, and gloo cannot carry the GPU collectives. Got "
                f"{self.accelerator.num_processes} processes.\n"
                f"Run one process and pick a device with CUDA_VISIBLE_DEVICES=<index>."
            )
        # Refused rather than warned: a texture curriculum under DDP produced progressively
        # worsening anatomy in two runs, already worse than single-GPU at matched step 480. A
        # warning printed at startup scrolls away in the first few seconds of a 25-minute run,
        # which is exactly how those two were lost. Checked here rather than in `load_config`
        # because only the Accelerator knows the process count.
        if (self.accelerator.num_processes > 1
                and any(p.mode == "texture" for p in cfg.curriculum.phases)
                and not cfg.train.allow_multi_gpu_texture):
            raise ValueError(
                f"texture curricula are not supported on {self.accelerator.num_processes} "
                f"processes.\n\nTwo runs here degraded anatomy progressively against a single-GPU "
                f"run at MATCHED optimizer steps, so this is not the halved step count and more "
                f"epochs do not fix it. The cause is not identified -- ranks seeding identically "
                f"was found and fixed, but was never shown to be the mechanism.\n\n"
                f"Run on one process (CUDA_VISIBLE_DEVICES=<one gpu>, no accelerate launch), or "
                f"set train.allow_multi_gpu_texture = true to re-test deliberately against a "
                f"single-GPU control."
            )
        if cfg.optimizer.gradient_release and self.accelerator.num_processes != 1:
            raise ValueError("Optimi gradient release is single-GPU only. DDP can update "
                             "parameters before gradient synchronization; use the isolated "
                             "trainer.tools.probe_gradient_release diagnostic to investigate.")
        self._build_data()
        self._build_model()
        self._build_optimizer()
        self._prepare()
        self.paired_texture = None
        if cfg.paired_texture.scale:
            from .paired_texture import PairedTextureLoss
            self.paired_texture = PairedTextureLoss(cfg.paired_texture)
        self.perceptual = None
        if cfg.perceptual.enabled:
            from .perceptual import DecodedPerceptualLoss
            self.perceptual = DecodedPerceptualLoss(cfg.train, cfg.perceptual, self.accelerator.device)
            if self.perceptual.eye_regions is not None:
                covered = sum(str(e.latent_path.resolve()) in self.perceptual.eye_regions
                              for e in self.dataset.entries)
                if not covered:
                    raise ValueError('No dataset latents match the eye annotation manifest')
                self.accelerator.print(f'eye regions {covered}/{len(self.dataset.entries)} samples annotated; '
                                       'remaining samples receive ordinary flow loss only')
            if cfg.perceptual.dedicated_low_noise:
                import random
                self.eye_order = [i for i,e in enumerate(self.dataset.entries)
                                  if str(e.latent_path.resolve()) in self.perceptual.eye_regions]
                random.Random(cfg.train.seed + 1000003).shuffle(self.eye_order)
                self.eye_micro = 0
        self.adversarial = None
        if cfg.adversarial.enabled:
            from .eye_adversarial import EyeAdversarialLoss
            self.adversarial = EyeAdversarialLoss(self.perceptual,cfg.adversarial,self.accelerator.device)
        self.self_flow_ema = None
        if cfg.self_flow.enabled:
            from .self_flow import AdapterEMA
            sf = cfg.self_flow
            self.self_flow_ema = AdapterEMA(self.accelerator.unwrap_model(self.transformer), 8,
                sf.ema_device, sf.decay, getattr(torch, sf.ema_dtype), sf.stochastic_rounding,
                full_finetune=True, adaln_fp32=sf.adaln_fp32)
            self.accelerator.register_for_checkpointing(self.self_flow_ema)
            self.accelerator.print(f'Self-Flow enabled: block 4 -> 8, EMA {sf.ema_dtype} on {sf.ema_device}, '
                                  f'FP32 AdaLN={sf.adaln_fp32}, storage {self.self_flow_ema.nbytes / 2**30:.2f} GiB')

        # Built after `_prepare` so it scales the adapter layers the prepared (DDP/compiled) model
        # actually runs, and encodes its prompts while the text encoder or text cache is ready.
        self.slider = None
        if cfg.slider.enabled:
            from .slider import ConceptSlider
            self.slider = ConceptSlider(cfg.slider, self)

        self.successful_updates = 0
        self.skipped_updates = 0
        self.consecutive_nonfinite = 0
        self.global_step = 0
        self.start_epoch = 0
        self.resume_skip_batches = 0
        if cfg.train.resume_from and cfg.train.resume_mode == "exact":
            self._resume(cfg.train.resume_from)

    def _dynamo_plugin(self):
        return None


    # ---------------------------------------------------------------- data

    def _build_data(self) -> None:
        cfg = self.cfg
        cfg.resolution_schedule.validate(cfg)
        dataset_cfg = replace(cfg.dataset, tier_collapse='repeat') if cfg.resolution_schedule.enabled else cfg.dataset
        if cfg.slider.enabled and not dataset_cfg.allow_missing_captions:
            # Slider steps never read dataset captions, so uncaptioned latents are not a mistake.
            dataset_cfg = replace(dataset_cfg, allow_missing_captions=True)
        self.dataset = MageFlowDataset(dataset_cfg, caption_seed=cfg.train.seed)
        from ..data.cache_provenance import validate_dataset_caches
        validate_dataset_caches(self.dataset, cfg.train, cfg.dataset.cache_validation)
        from ..data.packed import NativeResolutionBatchSampler, collate_native
        if cfg.resolution_schedule.enabled:
            from ..data.staged import StagedBatchSampler
            self.sampler = StagedBatchSampler(
                self.dataset, cfg.train.batch_size, cfg.resolution_schedule, cfg.train.max_steps,
                seed=cfg.train.seed, rank=self.accelerator.process_index,
                world=self.accelerator.num_processes, accumulation=cfg.train.gradient_accumulation_steps)
            self.stage_index = -1
            self.stage_cursor = 0
            self.stage_images = [0] * len(cfg.resolution_schedule.stages)
            self.stage_attempts = [0] * len(cfg.resolution_schedule.stages)
            self.loader = None
            self.total_steps = cfg.train.max_steps
            self.steps_per_epoch = self.total_batches = 0
        else:
            sampler_cls = NativeResolutionBatchSampler if cfg.train.pack_resolutions else BucketBatchSampler
            self.sampler = sampler_cls(
                self.dataset,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                drop_last=False,
                seed=cfg.train.seed,
            )
            self.loader = DataLoader(
                self.dataset,
                batch_sampler=self.sampler,
                collate_fn=collate_native if cfg.train.pack_resolutions else collate,
                num_workers=cfg.train.num_workers,
                pin_memory=True,
                # Workers must be re-forked each epoch: the sampler's epoch is read at __iter__ time,
                # and a persistent worker would keep serving batches built from a stale epoch.
                persistent_workers=False,
            )

            # `self.sampler` is unsharded; Accelerate's BatchSamplerShard splits it across ranks and
            # pads with even_batches, so each rank sees ceil(total / num_processes) batches. The
            # trailing partial accumulation group still steps (end_of_dataloader sync), hence ceil()
            # again -- nothing is dropped.
            total_batches = len(self.sampler)
            per_rank = math.ceil(total_batches / self.accelerator.num_processes)
            steps_per_epoch = math.ceil(per_rank / cfg.train.gradient_accumulation_steps)

            self.total_batches = total_batches
            self.total_steps = cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
            self.steps_per_epoch = steps_per_epoch
            if cfg.curriculum:
                self.sampler.set_curriculum(
                    cfg.curriculum, cfg.train.epochs, cfg.dataset.texture,
                    total_steps=self.total_steps,
                    gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
                    num_processes=self.accelerator.num_processes,
                )

        # Progress display. The bar is main-process-only: under DDP every rank runs this loop, and
        # N ranks writing \r to the same terminal is unreadable.
        mode = cfg.train.progress
        if mode == "auto":
            mode = "bar" if sys.stdout.isatty() else "plain"
        self.progress_mode = mode if self.accelerator.is_main_process else "off"
        self.bar: tqdm | None = None
        # (wall clock, global_step) taken *after* the first step of the run, so the ETA average
        # excludes torch.compile and the first dataloader fork -- both one-off costs that would
        # otherwise inflate every estimate for the rest of the run.
        self._eta_anchor: tuple[float, int] | None = None

    # --------------------------------------------------------------- model

    def _build_model(self) -> None:
        cfg = self.cfg
        self.source_checkpoint = None
        if self.accelerator.is_main_process:
            from ..modeling.checkpoint_metadata import checkpoint_identity
            source = cfg.train.transformer_path or Path(cfg.train.model_path) / ('model.safetensors' if cfg.train.model_family == 'flux3' else 'transformer')
            self.accelerator.print(f"Hashing source transformer: {source}")
            self.source_checkpoint = checkpoint_identity(source)
        self.text_cache = None
        if cfg.train.cache_text_embeddings:
            # Keep text encoding from consuming the training Torch RNG stream.
            # This does not promise bitwise determinism of compiled CUDA kernels.
            with torch.random.fork_rng(devices=[self.accelerator.device]):
                self._build_text_cache()
        needs_te = not cfg.train.cache_text_embeddings
        # Normally latents are cached and the VAE is 243MB of dead weight. Under
        # `dataset.source = "encode"` it is the whole point: pixels arrive per step and have to be
        # encoded before anything else can happen.
        needs_vae = cfg.dataset.source == "encode"

        self.components = load_components(
            cfg.train.model_path,
            dtype=self.dtype,
            **model_load_kwargs(cfg.train),
            load_text_encoder=needs_te,
            load_vae=needs_vae,
            load_tokenizers=needs_te,
        )
        self.transformer = self.components.transformer
        self.text_encoder = self.components.text_encoder

        # Cached and live encoding share the same Mage-VAE path.
        self.latent_encoder = None
        if needs_vae:
            from ..data.cache import LatentCacher
            self.latent_encoder = LatentCacher(
                self.components.vae, device=self.accelerator.device, dtype=self.dtype, flux2_vae=cfg.train.flux2_vae
            )
            self.components.vae.requires_grad_(False)

        # The hf loss tokenizes exactly as the model patchifies, so the patch size is read off the
        # model rather than configured -- a mismatch would weight the wrong spatial extent.
        # patch_size is (T, H, W); T is 1 for images and the spatial dims are square.
        self.hf_patch = None
        if cfg.flow.hf_scale > 0.0:
            ph = pw = self.transformer.patch_size
            if ph != pw:
                raise ValueError(f"hf loss assumes a square patch, got {ph}x{pw}")
            self.hf_patch = ph

        if cfg.rti.enabled:
            self.rti_interface = self.transformer.configure_rti(
                cfg.rti.dense_prefix_blocks, cfg.rti.dense_suffix_blocks, cfg.rti.size_buckets)
            self.rti_budget = cfg.rti.schedule()
            self.keep_fraction, self.rti_phase = self.rti_budget.resolve(0)
        else:
            self.rti_interface = self.rti_budget = None
            self.keep_fraction, self.rti_phase = 1.0, None
        self._quantize()

        self.adapter_initialization = None
        if cfg.is_lora:
            apply_adapter(
                self.transformer, cfg.adapter
            )
            if cfg.train.resume_mode == "weights":
                from .adapter_loading import load_adapter_weights
                self.adapter_initialization = load_adapter_weights(self.transformer, cfg.train.resume_from)
                self.accelerator.print(
                    f"Loaded adapter weights from {cfg.train.resume_from}; base unchanged. "
                    "Starting a new phase at step 0 with fresh optimizer, schedule and RNG.")
            self.param_report = build_adapter_param_groups(
                self.transformer,
                cfg.component_lr,
                default_lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                block_lr=cfg.block_lr,
            )
            groups = self.param_report.groups
        else:
            self.param_report = build_param_groups(
                self.transformer,
                cfg.component_lr,
                default_lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
            )
            groups = self.param_report.groups
            if self.text_encoder is not None:
                self.text_encoder.requires_grad_(False)

        self.transformer.configure_execution(
            cfg.train.gradient_checkpointing, cfg.train.checkpoint_blocks,
            cfg.train.compile, cfg.train.compile_dynamic, cfg.train.attention_backend,
            activation_offload=cfg.train.activation_offload,
            activation_offload_budget_mb=cfg.train.activation_offload_budget_mb,
            activation_offload_pin_memory=cfg.train.activation_offload_pin_memory,
            activation_offload_prefetch=cfg.train.activation_offload_prefetch,
            **({"moss_checkpoint_inside_compile": cfg.train.moss_checkpoint_inside_compile}
               if cfg.train.model_family == "moss_image" else {}))
        self.groups = groups
        if cfg.self_flow.enabled:
            from types import MethodType
            from .self_flow import distributed_probe_forward
            if len(self.transformer.transformer_blocks) < 8:
                raise ValueError('Self-Flow requires at least eight transformer blocks')
            d = self.transformer.inner_dim
            self.transformer.self_flow_projector = torch.nn.Sequential(
                torch.nn.Linear(d, 2*d), torch.nn.SiLU(), torch.nn.Linear(2*d, d)
            ).to(device=self.accelerator.device, dtype=self.dtype)
            self.groups.append(dict(params=list(self.transformer.self_flow_projector.parameters()),
                                    lr=cfg.optimizer.lr, weight_decay=0.0))
            self.transformer._probe_original_forward = self.transformer.forward
            self.transformer.forward = MethodType(distributed_probe_forward, self.transformer)

        # The reference is "this model with the adapter switched off", which only exists when there
        # is an adapter to switch off. A full finetune has no frozen copy to compare against, and
        # holding one would double the trunk's memory.
        self._micro = 0
        self.last_preserve = None
        self.preserver = None
        if cfg.preserve.enabled:
            if not cfg.is_lora:
                raise ValueError(
                    "[preserve] needs an adapter to disable for its reference pass, but "
                    "adapter.kind = 'none'. Concept preservation is not available for full "
                    "finetunes."
                )
            self.preserver = ConceptPreserver(cfg.preserve, self)

    def _quantize(self) -> None:
        """Quantize before adapters are injected, so LoRA layers wrap quantized bases rather than
        being quantized themselves."""
        qcfg = self.cfg.quant
        device = self.accelerator.device
        te_cfg = text_encoder_quant_config(qcfg)
        if te_cfg is not None and self.text_encoder is not None:
            self.text_encoder = quantize_module(self.text_encoder, te_cfg, device, self.dtype, False)
            self.components.text_encoder = self.text_encoder
        if qcfg.mode == "none":
            self.quant_info = None
            return

        device = self.accelerator.device
        # One token count per sample, so `auto` weights buckets by how much of the run they are,
        # not by which is biggest. Under a multi-resolution ladder those differ sharply.
        tokens = [
            tokens_for_bucket(b)
            for b, idx in self.sampler.bucket_indices.items()
            for _ in idx
        ]
        max_tokens = max(tokens)
        use_qmm = resolve_quantized_matmul(qcfg, tokens)

        # The crossover only governs frozen weights, so a straddle warning in training mode would
        # point at a threshold that no longer decides anything.
        self.transformer = quantize_module(self.transformer, qcfg, device, self.dtype, use_qmm)
        nq, ntotal, _ = quantized_layer_report(self.transformer)
        wmean = sum(t * t for t in tokens) / sum(tokens)
        self.quant_info = (nq, ntotal, use_qmm, max_tokens, wmean)

    def _all_modules_params(self):
        for m in (self.transformer, self.text_encoder):
            if m is not None:
                yield from m.parameters()

    # ----------------------------------------------------------- optimizer

    def _build_optimizer(self) -> None:
        cfg = self.cfg
        self.optimizer = build_optimizer(self.groups, cfg.optimizer)
        # Accelerate's scheduler wrapper advances the inner scheduler `num_processes` times per
        # `step()` call (accelerate/scheduler.py: it assumes the dataloader batch was multiplied by
        # world size). Our horizon is counted in optimizer steps, so without this the schedule
        # burns through `num_processes`x too fast -- on 2 GPUs a cosine decay reached its floor at
        # the halfway point and trained the rest of the run at lr 0.
        scale = self.accelerator.num_processes
        schedule = replace(cfg.schedule, warmup_steps=cfg.schedule.warmup_steps * scale)
        self.scheduler = build_scheduler(self.optimizer, schedule, self.total_steps * scale)

    def _prepare(self):
        raw_optimizer = self.optimizer
        if (self.cfg.quant.mode == "training" or self.cfg.quant.backend == "convrot" or self.cfg.quant.frozen_matmul) and self.accelerator.num_processes > 1:
            from .distributed import sync_quantized_model
            sync_quantized_model(self.transformer)
        if self.text_encoder is not None:
            device = "cpu" if self.cfg.train.offload_text_encoder else self.accelerator.device
            self.text_encoder.to(device).eval()
            if self.cfg.train.compile_text_encoder:
                language_model = self.text_encoder.model.language_model
                for block in language_model.layers:
                    block.compile(dynamic=True)
                self.accelerator.print("Loaded Text Encoder: compiled decoder blocks (dynamic=True); first encoding includes compile time")
            if self.cfg.train.text_encoder_embedding_only:
                self.accelerator.print("Loaded Text Encoder: embedding-only output; first encoding verifies wrapper equivalence")
        if self.cfg.resolution_schedule.enabled:
            # Staged batches are already sharded and contain complete accumulation
            # groups. Accelerate must not shard or pad them a second time.
            self.transformer, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.transformer, self.optimizer, self.scheduler)
        else:
            self.transformer, self.optimizer, self.scheduler, self.loader = self.accelerator.prepare(
                self.transformer, self.optimizer, self.scheduler, self.loader)
        if self.cfg.optimizer.gradient_release:
            from .optim import enable_gradient_release
            self._gradient_release_holder = enable_gradient_release(
                raw_optimizer, self.accelerator.num_processes)


    # ----------------------------------------------------------- one step

    def _build_text_cache(self):
        """Encode static captions before loading the transformer, then release Qwen3-VL."""
        import gc
        import random
        from ..data.caption import build_caption
        from ..data.text_cache import TextEmbeddingCache, validate_static_captions
        cfg = self.cfg
        if cfg.slider.enabled:
            # Slider steps never read dataset captions, so only the slider's own prompts are
            # encoded; the encoder is released before the transformer loads, as usual.
            from .slider import ConceptSlider
            captions = ConceptSlider.cached_prompts(cfg.slider)
        else:
            validate_static_captions(cfg)
            if cfg.train.caption_variations:
                return self._build_variation_cache()
            captions = sorted({build_caption(e.tags, e.nl, cfg.dataset.caption, random.Random(0))
                               for e in self.dataset.entries})
            if any(p.mode == "texture" for p in cfg.curriculum.phases):
                captions = sorted(set(captions) | {cfg.dataset.texture.trigger})
        if not captions:
            raise ValueError("No captions available for text embedding caching")
        components = load_components(cfg.train.model_path, self.dtype, **model_load_kwargs(cfg.train),
            load_transformer=False, load_vae=False)
        q = text_encoder_quant_config(cfg.quant)
        if q is not None:
            components.text_encoder = quantize_module(components.text_encoder, q,
                self.accelerator.device, self.dtype, False)
        components.text_encoder.to(self.accelerator.device).eval()
        cache = TextEmbeddingCache()
        batches = _text_cache_batches(captions, cfg.train.text_cache_batch_size)
        for batch in tqdm(batches, total=math.ceil(len(captions) / cfg.train.text_cache_batch_size),
                          desc="Caching text batches", disable=not self.accelerator.is_main_process):
            hidden, mask = encode_prompts(components, batch, self.accelerator.device,
                                         cfg.train.max_text_tokens)
            hidden, mask = hidden.cpu(), mask.cpu()
            for i, caption in enumerate(batch):
                cache.add(caption, hidden[i:i + 1], mask[i:i + 1])
        del hidden, mask, components
        gc.collect()
        torch.cuda.empty_cache()
        self.text_cache = cache

    @staticmethod
    def _process_group_kwargs():
        from datetime import timedelta
        from accelerate.utils import InitProcessGroupKwargs

        return InitProcessGroupKwargs(timeout=timedelta(hours=1))

    def _build_variation_cache(self):
        import gc
        from contextlib import nullcontext
        from accelerate.utils import broadcast_object_list
        from ..data.caption_variations import CaptionVariationCache, encoder_fingerprint
        cfg = self.cfg
        default_root = Path(cfg.dataset.effective_subsets()[0].path)
        path = cfg.train.caption_cache_path or str(
            default_root / "cached_text_embeddings" / "caption_variations.sqlite")
        from filelock import FileLock
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        accelerator = self.accelerator
        # Rank 0 holds the job-level lock until all ranks finish. Workers in this
        # job coordinate through SQLite's short transactions, not this file lock.
        lock = FileLock(str(path) + ".lock") if accelerator.is_main_process else nullcontext()
        with lock:
            key = [encoder_fingerprint(cfg, path) if accelerator.is_main_process else None]
            if accelerator.num_processes > 1:
                broadcast_object_list(key)
            cache = CaptionVariationCache(path, key[0])
            if accelerator.is_main_process:
                cache.prepare(self.dataset.entries, cfg.dataset.caption, cfg.train.seed,
                              cfg.train.caption_variations,
                              extra_captions=([cfg.dataset.texture.trigger] if any(
                                  p.mode == "texture" for p in cfg.curriculum.phases) else []))
                self.accelerator.print(f"caption cache {cache.statistics}")
            accelerator.wait_for_everyone()
            if not accelerator.is_main_process:
                cache.prepare(self.dataset.entries, cfg.dataset.caption, cfg.train.seed,
                              cfg.train.caption_variations, write=False)
            iterator = iter(cache.pending(accelerator.process_index, accelerator.num_processes))
            first = next(iterator, None)
            if first is not None:
                from itertools import chain
                components = load_components(cfg.train.model_path, self.dtype, **model_load_kwargs(cfg.train),
                    load_transformer=False, load_vae=False)
                q = text_encoder_quant_config(cfg.quant)
                if q is not None:
                    components.text_encoder = quantize_module(components.text_encoder, q,
                        self.accelerator.device, self.dtype, False)
                components.text_encoder.to(self.accelerator.device).eval()
                with cache.writer() as db:
                    batches = _text_cache_batches(chain([first], iterator), cfg.train.text_cache_batch_size)
                    for batch in tqdm(batches,
                                      desc=f"Caching caption batches rank {accelerator.process_index}",
                                      position=accelerator.process_index):
                        hidden, mask = encode_prompts(components, [caption for _, caption in batch],
                                                     self.accelerator.device, cfg.train.max_text_tokens)
                        hidden, mask = hidden.cpu(), mask.cpu()
                        for i, (key, _) in enumerate(batch):
                            cache.add(key, hidden[i:i + 1], mask[i:i + 1], db=db)
                        # Commit the batch before the next GPU forward pass.
                        db.commit()
                db.close()
                del hidden, mask, components
                gc.collect()
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and next(cache.pending(), None) is not None:
                raise RuntimeError("Distributed caption cache is incomplete")
        cache.close()
        self.dataset.caption_variation_cache = cache
        self.text_cache = cache
        self.accelerator.print(f"text cache {path}: {cfg.train.caption_variations} slots/image; encoder removed")

    def _encode(self, captions):
        if self.text_cache is not None:
            return self.text_cache.get(captions, self.accelerator.device, self.dtype)
        offload = self.cfg.train.offload_text_encoder
        if offload:
            self.text_encoder.to(self.accelerator.device)
        try:
            return encode_prompts(self.components, captions, self.accelerator.device,
                                  self.cfg.train.max_text_tokens,
                                  embedding_only=self.cfg.train.text_encoder_embedding_only)
        finally:
            if offload:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()


    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels -> normalised latents, in chunks, with nothing left alive afterwards.

        Chunked so the encode's transient activations are bounded by `vae_encode_chunk` rather than
        by the training batch size. Because this runs under `no_grad` and everything but the (tiny)
        latents is freed before the transformer forward begins, the step's peak is
        `max(train_peak, encode_peak)` -- not their sum. That is the whole reason encode-on-the-fly
        does not force the batch size down to a couple of images.
        """
        chunk = max(1, self.cfg.train.vae_encode_chunk)
        out = [
            self.latent_encoder.encode_tensor(pixels[i:i + chunk])
            for i in range(0, pixels.shape[0], chunk)
        ]
        return torch.cat(out).float()

    def _step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        if getattr(self, "slider", None) is not None:
            return self._step_slider(batch)
        if self.self_flow_ema is not None:
            return self._step_self_flow(batch)
        if self.cfg.train.pack_resolutions:
            return self._step_packed(batch)
        device = self.accelerator.device
        if "pixels" in batch:
            latents = self._encode_pixels(batch["pixels"].to(device, self.dtype))
        else:
            latents = batch["latents"].to(device, torch.float32)

        stats: dict = {}
        noisy, timesteps, target, second_t, token_mask = prepare_training_flow_batch(
            latents, self.cfg.flow, stats=stats, t_range=self.phase.t_range if self.phase else None,
            **self.flow_rng.kwargs(),
        )
        self.last_stats = stats
        encoder_hidden_states = self._encode(batch["captions"])

        pred = self.transformer(
            hidden_states=noisy.to(self.dtype),
            timestep=self._conditioning_timesteps(timesteps),
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            **({"second_timestep": second_t.to(self.dtype), "timestep_mask": token_mask}
               if second_t is not None else {}),
        )[0]

        # (B,1,h,w) -> (B,1,1,h,w) to broadcast over the latent's channel and time axes. Texture
        # batches supervise only a feathered sub-region; the rest is real context that conditions
        # the forward pass but carries no target.
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, torch.float32).unsqueeze(2)

        loss = flow_loss(pred, target, mask=mask)
        if self.paired_texture is not None:
            if mask is not None:
                raise ValueError('Paired texture loss does not support masked batches')
            loss, term = self.paired_texture.apply(loss,pred,latents,noisy,timesteps)
            self.last_stats['texture'] = term.item()
        loss = loss + self._preserve_term()
        if self.hf_patch is None:
            return loss, None

        # Reported separately and already scaled, so the log decomposes as total = mse + hf.
        # The mask goes through here too: weighting unsupervised context by its detail content
        # would put a target back on exactly the region the mask exists to exclude.
        hf = self.cfg.flow.hf_scale * hf_loss(
            pred, noisy, latents, effective_timesteps(timesteps, second_t, token_mask),
            self.hf_patch, self.cfg.flow.hf_exponent, mask=mask, **self.cfg.flow.hf_gate_kwargs()
        )
        return loss + hf, hf

    def _conditioning_timesteps(self, timesteps):
        # MOSS scales normalized sigma by 1000 before computing sinusoids.
        # BF16 rounding here would condition on a different time than noising.
        dtype = torch.float32 if self.cfg.train.model_family == "moss_image" else self.dtype
        return timesteps.to(dtype)

    def _step_slider(self, batch: dict) -> tuple[torch.Tensor, None]:
        """Concept-slider step: the batch supplies noisy latents, the base model supplies targets.

        Latents, noise, timesteps (curriculum t-range and dual timestep included) and the packed
        layout are prepared exactly as in `_step` / `_step_packed`; every slider forward reuses
        them, so the reference and signed predictions see identical inputs. See `slider.py` for
        the objective and for why each signed pass gets its own backward.
        """
        from .slider import slider_step
        if batch.get("mask") is not None:
            raise ValueError("[slider] does not support masked batches")
        device = self.accelerator.device
        packed = self.cfg.train.pack_resolutions
        if "pixels" in batch:
            latents = ([self._encode_pixels(x.to(device, self.dtype)) for x in batch["pixels"]] if packed
                       else self._encode_pixels(batch["pixels"].to(device, self.dtype)))
        else:
            latents = ([x.to(device, torch.float32) for x in batch["latents"]] if packed
                       else batch["latents"].to(device, torch.float32))

        stats: dict = {}
        if packed:
            prepared = [prepare_training_flow_batch(x, self.cfg.flow, **self.flow_rng.kwargs()) for x in latents]
            noisy, times, _, second_times, token_masks = zip(*prepared)
            hidden = [x.to(self.dtype) for x in noisy]
            call = {"timestep": self._conditioning_timesteps(torch.cat(times))}
            if self.cfg.flow.dual_timestep:
                call.update(second_timestep=torch.cat(second_times).to(self.dtype), timestep_mask=token_masks)
        else:
            noisy, timesteps, _, second_t, token_mask = prepare_training_flow_batch(
                latents, self.cfg.flow, stats=stats, t_range=self.phase.t_range if self.phase else None,
                **self.flow_rng.kwargs(),
            )
            hidden = noisy.to(self.dtype)
            call = {"timestep": self._conditioning_timesteps(timesteps), "return_dict": False}
            if second_t is not None:
                call.update(second_timestep=second_t.to(self.dtype), timestep_mask=token_mask)
        self.last_stats = stats

        samples = len(batch["captions"])
        contexts = {}

        def forward(role: str):
            if role not in contexts:
                contexts[role] = self.slider.context(role, samples)
            return self.transformer(hidden_states=hidden, encoder_hidden_states=contexts[role], **call)[0]

        acc = self.accelerator
        loss, slider_stats = slider_step(
            self.cfg.slider, self.slider.strength, forward,
            # Same per-image scaling as the trainer's own backward (gradient release is refused).
            early_backward=lambda term: acc.backward(term * samples),
            # Early passes accumulate locally; the final pass's backward reduces once under DDP.
            early_context=lambda: acc.no_sync(self.transformer),
        )
        self.last_stats.update(slider_stats)
        return loss, None

    def _step_self_flow(self, batch):
        from .self_flow import forward_probe
        from .flow import sample_timesteps
        if batch.get('mask') is not None:
            raise ValueError('Self-Flow requires unmasked cached latents')
        latents = [x.to(self.accelerator.device, torch.float32) for x in batch['latents']]
        context = self._encode(batch['captions'])
        noises, times, masks, noisy, teacher_images, clean_times = [], [], [], [], [], []
        for latent in latents:
            noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype,
                                generator=self.flow_rng.noise)
            t = sample_timesteps(self.cfg.flow, 2, *latent.shape[-2:], latent.device,
                                 generator=self.flow_rng.timesteps)
            ids = (torch.rand(latent.shape[-2:], device=latent.device,
                              generator=self.flow_rng.mask) < self.cfg.flow.dual_timestep_mask_ratio).long()
            per_token_t = t[ids][None, None, None]
            noises.append(noise)
            times.append(t)
            masks.append(ids)
            noisy.append(((1-per_token_t)*latent + per_token_t*noise).to(self.dtype))
            clean_t = t.min().reshape(1)
            clean_times.append(clean_t)
            teacher_images.append(((1-clean_t)*latent + clean_t*noise).to(self.dtype))
        single = len(latents) == 1
        model = self.accelerator.unwrap_model(self.transformer)
        model.eval()
        with torch.no_grad():
            _, teacher = forward_probe(model, teacher_images[0] if single else teacher_images,
                                       torch.cat(clean_times), context, stop_at=8, ema=self.self_flow_ema)
        model.train()
        prediction, projected = self.transformer(noisy[0] if single else noisy, torch.cat(times), context,
                                                 token_ids=masks[0] if single else masks,
                                                 probe=True, project_features=True)
        if single:
            prediction, projected, teacher = [prediction], [projected], [teacher]
        # Equal image weighting for both objectives, not weighting by token count.
        alignment = torch.stack([1-torch.nn.functional.cosine_similarity(
            p.float(), t.float(), dim=-1, eps=1e-12).mean() for p,t in zip(projected, teacher)]).mean()
        fm = torch.stack([flow_loss(p, n-x) for p,n,x in zip(prediction, noises, latents)]).mean()
        self.last_stats = {'self_flow_alignment': alignment.detach().item()}
        return fm + self.cfg.self_flow.weight*alignment, None

    def _step_packed(self, batch):
        device = self.accelerator.device
        latents = ([self._encode_pixels(x.to(device,self.dtype)) for x in batch["pixels"]]
                   if "pixels" in batch else [x.to(device,torch.float32) for x in batch["latents"]])
        prepared = [prepare_training_flow_batch(x,self.cfg.flow, **self.flow_rng.kwargs()) for x in latents]
        noisy,times,targets,second_times,token_masks = zip(*prepared)
        ordinary_count = len(latents)
        dedicated = self.cfg.perceptual.enabled and self.cfg.perceptual.dedicated_low_noise
        captions = list(batch['captions'])
        paths = list(batch['paths'])
        if dedicated:
            from .perceptual import dedicated_flow_batch
            ordinal = self.eye_micro*self.accelerator.num_processes+self.accelerator.process_index
            sample = self.dataset[(self.eye_order[ordinal % len(self.eye_order)],
                                   self.eye_micro // len(self.loader))]
            extra = sample['latents'].unsqueeze(0).to(device,torch.float32)
            extra_noisy, extra_t, extra_target = dedicated_flow_batch(
                extra, self.cfg.train.seed, ordinal, self.cfg.perceptual.max_t)
            latents.append(extra)
            noisy, times, targets = (*noisy, extra_noisy), (*times, extra_t), (*targets, extra_target)
            captions.append(sample['caption']); paths.append(sample['path'])
            self.eye_micro += 1
        context = self._encode(captions)
        predictions = self.transformer(hidden_states=[x.to(self.dtype) for x in noisy],
            timestep=self._conditioning_timesteps(torch.cat(times)), encoder_hidden_states=context,
            **({"second_timestep": torch.cat(second_times).to(self.dtype),
                "timestep_mask": token_masks} if self.cfg.flow.dual_timestep else {}),
            **({"keep_fraction": self.keep_fraction} if self.rti_interface is not None else {}))[0]
        # Equal sample weighting: a large image must not gain extra influence merely
        # because it contributes more tokens to the packed attention kernel.
        loss = torch.stack([flow_loss(y,t) for y,t in
                            zip(predictions[:ordinary_count],targets[:ordinary_count])]).mean()
        if dedicated:
            # Keep ordinary loss unchanged; extra example has one full-batch sample's weight.
            loss = loss + flow_loss(predictions[-1],targets[-1])/self.cfg.train.batch_size
        self.last_stats = {}
        if self.paired_texture is not None:
            if batch.get('mask') is not None:
                raise ValueError('Paired texture loss does not support masked batches')
            terms = [self.paired_texture.apply(loss.new_zeros(()),y,x,n,t)[0]
                     for y,x,n,t in zip(predictions[:ordinary_count],latents[:ordinary_count],
                                        noisy[:ordinary_count],times[:ordinary_count])]
            texture = torch.stack(terms).mean()
            loss = loss + texture
            self.last_stats['texture'] = texture.detach().item()
        if self.adversarial is not None:
            loss = loss + self.adversarial.loss(predictions[-1],noisy[-1],latents[-1],times[-1],paths[-1])
            self.last_stats.update(self.adversarial.last_stats)
        if self.perceptual is not None:
            if dedicated:
                loss = loss + self.perceptual.loss(predictions[-1:],noisy[-1:],latents[-1:],times[-1:],paths[-1:])
            else:
                loss = loss + self.perceptual.loss(predictions, noisy, latents, times, paths)
            if self.perceptual.last_raw is not None:
                self.last_stats['lpips'] = self.perceptual.last_raw
        loss = loss + self._preserve_term()
        if self.hf_patch is None:
            return loss,None
        hf = self.cfg.flow.hf_scale * torch.stack([
            hf_loss(y,n,x,effective_timesteps(t,s,m),self.hf_patch,self.cfg.flow.hf_exponent,
                    **self.cfg.flow.hf_gate_kwargs())
            for y,n,x,t,s,m in zip(predictions,noisy,latents,times,second_times,token_masks)]).mean()
        return loss+hf,hf

    def _preserve_term(self) -> torch.Tensor | float:
        """Concept-preservation loss, or 0.0 when it is off or this is not an evaluation step.

        Gated on a micro-step counter rather than on `global_step`, which only advances on sync
        boundaries: under gradient accumulation every micro-step of a group shares one
        `global_step`, so gating on it would fire the probe `gradient_accumulation_steps` times per
        optimizer step instead of once.
        """
        self.last_preserve = None
        if self.preserver is None:
            return 0.0
        self._micro += 1
        term = self.preserver.loss(self._micro)
        if term is None:
            return 0.0
        self.last_preserve = term.detach()
        return term

    # ------------------------------------------------------------ curriculum

    def _set_phase(self) -> None:
        if not self.cfg.curriculum:
            return
        prev = self.phase
        self.phase = self.cfg.curriculum.resolve(self.global_step / max(1, self.total_steps))
        if prev is not self.phase and self.accelerator.is_main_process:
            _emit(self, f"phase   {self.phase.label()}  @ step {self.global_step}")

    def _set_rti_budget(self) -> None:
        if self.rti_budget is not None:
            self.keep_fraction, self.rti_phase = self.rti_budget.resolve(self.global_step)

    def _apply_lr_mul(self) -> None:
        """Scale the scheduler's current LR by the active stage/phase multiplier.

        Set absolutely from `get_last_lr()` rather than multiplied in place: this runs on every
        iteration but the scheduler only advances on sync boundaries, so an in-place multiply
        would compound `lr_mul` once per accumulation micro-step -- silently turning lr_mul=0.5
        into 0.5^accum.
        """
        if self.cfg.resolution_schedule.enabled:
            multiplier = self.cfg.resolution_schedule.stages[self.stage_index].lr_mul
        else:
            if not self.phase or self.phase.lr_mul == 1.0:
                return
            multiplier = self.phase.lr_mul
        last = getattr(self.scheduler, "get_last_lr", None)
        if last is None:
            return
        for pg, lr in zip(self.optimizer.param_groups, last()):
            pg["lr"] = lr * multiplier

    # ---------------------------------------------------------------- run

    def _training_passes(self):
        """Yield complete update groups. Dataset cycles never control staged duration."""
        cfg = self.cfg
        if not cfg.resolution_schedule.enabled:
            for epoch in range(self.start_epoch, cfg.train.epochs):
                if self.global_step >= self.total_steps:
                    break
                self.sampler.set_epoch(epoch)
                loader = self.loader
                if epoch == self.start_epoch and self.resume_skip_batches:
                    loader = self.accelerator.skip_first_batches(loader, self.resume_skip_batches)
                yield epoch, loader
            return
        from ..data.packed import collate_native
        ends = list(accumulate(self.sampler.budgets))
        while self.global_step < self.total_steps:
            stage = next(i for i, end in enumerate(ends) if self.global_step < end)
            if stage != self.stage_index:
                self.stage_index, self.stage_cursor = stage, 0
                resolution = cfg.resolution_schedule.stages[stage].resolution
                effective = cfg.train.batch_size[resolution] * self.accelerator.num_processes * cfg.train.gradient_accumulation_steps
                if self.accelerator.is_main_process:
                    _emit(self, f'stage    {stage + 1}/{len(ends)}: {resolution}px, '
                               f'{self.sampler.budgets[stage]} successful updates, effective batch {effective}, '
                               f'LR x{cfg.resolution_schedule.stages[stage].lr_mul:g}')
            self.sampler.configure(stage, self.stage_cursor, ends[stage] - self.global_step)
            # A private worker RNG prevents loader construction from advancing model
            # RNG on transitions or resume. Captions are seeded per sample/cycle.
            generator = torch.Generator().manual_seed(cfg.train.seed + stage)
            loader = DataLoader(self.dataset, batch_sampler=self.sampler, collate_fn=collate_native,
                                num_workers=cfg.train.num_workers, pin_memory=True, generator=generator,
                                multiprocessing_context="spawn" if cfg.train.num_workers else None)
            self.loader = loader
            yield stage, loader
            # If nonfinite attempts were skipped, refill only this stage's remaining
            # successful-update budget, continuing after the consumed data cursor.

    def _resolution_state(self):
        if not self.cfg.resolution_schedule.enabled:
            return None
        return dict(identity=self.sampler.identity, stage=self.stage_index, cursor=self.stage_cursor,
                    images=self.stage_images, attempts=self.stage_attempts,
                    lr_multipliers=[s.lr_mul for s in self.cfg.resolution_schedule.stages])

    def train(self) -> None:
        cfg = self.cfg
        acc = self.accelerator
        self._report()
        offloader = getattr(self._unwrap(self.transformer), '_activation_offloader', None)
        if offloader is not None:
            _emit(self, f"activation offload: {offloader.budget_bytes / 2**20:.0f} MiB live payload/rank, "
                       f"pin={offloader.pin_memory}, prefetch={offloader.prefetch}; excess inputs stay on GPU")

        if cfg.optimizer.gradient_release:
            _emit(self, "optim    gradient release enabled (single GPU; updates during backward)")
        trainable = [p for p in self._all_modules_params() if p.requires_grad]
        done = False
        from .step_safety import global_sample_scale, scale_gradients, gradient_norm, globally_finite
        local_samples = 0
        finite_losses = torch.ones((), dtype=torch.bool, device=acc.device)
        loss_sum = torch.zeros((), device=acc.device)
        hf_sum = torch.zeros((), device=acc.device)
        hf_present = False

        if self.progress_mode == "bar":
            self.bar = tqdm(
                total=self.total_steps, initial=self.global_step, unit="step",
                dynamic_ncols=True, smoothing=0.05,
            )

        for epoch, loader in self._training_passes():
            self.transformer.train()
            t0 = time.time()
            for batch in loader:
                if cfg.resolution_schedule.enabled:
                    self.stage_cursor += 1
                    self.stage_images[self.stage_index] += len(batch['captions']) * acc.num_processes
                # Resolved before the step so the phase's t_range reaches `_step`. Progress is a
                # pure function of the step count, so every DDP rank picks the same phase without
                # communicating, and a resumed run lands in the right phase on its own.
                self._set_phase()
                self._set_rti_budget()
                with acc.accumulate(self.transformer):
                    loss, hf = self._step(batch)
                    if cfg.optimizer.gradient_release:
                        self._apply_lr_mul()  # Hook updates happen inside backward.
                    samples = len(batch["captions"])
                    local_samples += samples
                    finite_losses &= torch.isfinite(loss.detach())
                    loss_sum += loss.detach().float() * samples
                    if hf is not None:
                        hf_sum += hf.detach().float() * samples
                        hf_present = True
                    if cfg.optimizer.gradient_release and not bool(finite_losses.item()):
                        raise FloatingPointError("Nonfinite loss before gradient-release backward")
                    # Accelerate divides by accumulation_steps; DDP averages ranks.
                    # Restore an equal-per-image mean across the entire update below.
                    try:
                        acc.backward(loss if cfg.optimizer.gradient_release else loss * samples)
                    finally:
                        if getattr(self, "slider", None) is not None:
                            # The last slider pass ran at strength -1 and its backward (which may
                            # recompute checkpointed blocks) has now finished, or raised; back to +1.
                            self.slider.after_backward()
                    update_ok = False
                    if acc.sync_gradients:
                        if not cfg.optimizer.gradient_release:
                            scale = global_sample_scale(local_samples, cfg.train.gradient_accumulation_steps, acc.device)
                            scale_gradients(trainable, scale)
                            self.last_grad_norm = gradient_norm(trainable, acc.device).detach()
                            update_ok = globally_finite(finite_losses, self.last_grad_norm)
                        else:
                            update_ok = True  # Per-parameter guards run before optimizer hooks.
                        if update_ok:
                            if not cfg.optimizer.gradient_release:
                                if cfg.optimizer.max_grad_norm > 0:
                                    torch.nn.utils.clip_grad_norm_(trainable, cfg.optimizer.max_grad_norm)
                                self._apply_lr_mul()
                                if cfg.resolution_schedule.enabled:
                                    self._last_update_lrs = [group['lr'] for group in self.optimizer.param_groups]
                                if acc.num_processes > 1:
                                    from .distributed import quantized_optimizer_rng
                                    with quantized_optimizer_rng(acc.device, cfg.train.seed, self.global_step):
                                        self.optimizer.step()
                                else:
                                    self.optimizer.step()
                            self.scheduler.step()
                            self.successful_updates += 1
                            self.consecutive_nonfinite = 0
                        else:
                            self.skipped_updates += 1
                            self.consecutive_nonfinite += 1
                            from .step_safety import nonfinite_names
                            bad_names = nonfinite_names(self._unwrap(self.transformer).named_parameters())
                            reports = [None] * acc.num_processes
                            if acc.num_processes > 1:
                                torch.distributed.all_gather_object(reports, bad_names)
                            else:
                                reports[0] = bad_names
                            _emit(self, f"Nonfinite gradients by rank (first eight names): {reports}")
                            _emit(self, f"Skipped nonfinite update {self.global_step + 1}; "
                                       f"consecutive={self.consecutive_nonfinite}")
                        self.optimizer.zero_grad(set_to_none=True)

                if not acc.sync_gradients:
                    continue

                if update_ok and self.self_flow_ema is not None and not acc.optimizer_step_was_skipped:
                    from .distributed import quantized_optimizer_rng
                    with quantized_optimizer_rng(acc.device, cfg.train.seed + 7919, self.global_step):
                        self.self_flow_ema.update()
                # Log the same global image-weighted loss that supplies the gradients.
                totals = torch.stack((loss_sum, loss_sum.new_tensor(local_samples), hf_sum,
                                      loss_sum.new_tensor(hf_present)))
                if acc.num_processes > 1:
                    torch.distributed.all_reduce(totals)
                loss = totals[0] / totals[1]
                hf = totals[2] / totals[1] if totals[3].item() else None
                hf_sum.zero_()
                hf_present = False
                local_samples = 0
                finite_losses.fill_(True)
                loss_sum.zero_()
                if self.consecutive_nonfinite >= cfg.train.nonfinite_patience:
                    raise FloatingPointError("Repeated nonfinite updates; stopped before saving invalid training")

                if cfg.resolution_schedule.enabled:
                    self.stage_attempts[self.stage_index] += 1
                    if not update_ok:
                        continue
                self.global_step += 1
                if self.bar is not None:
                    self.bar.update(1)
                if self._eta_anchor is None:
                    self._eta_anchor = (time.time(), self.global_step)
                if self.global_step % cfg.train.log_every == 0:
                    self._log(loss, hf, epoch, t0)
                    t0 = time.time()

                periodic = cfg.train.save_every_steps and self.global_step % cfg.train.save_every_steps == 0
                stage_end = (cfg.resolution_schedule.enabled and cfg.resolution_schedule.save_stage_end
                             and self.global_step == sum(self.sampler.budgets[:self.stage_index + 1]))
                if periodic or (stage_end and self.global_step < self.total_steps):
                    self.save(f"step{self.global_step:06d}")

                if self.global_step >= self.total_steps:
                    done = True
                    break

            if not cfg.resolution_schedule.enabled and cfg.train.save_every_epochs and (epoch + 1) % cfg.train.save_every_epochs == 0:
                self.save(f"epoch{epoch + 1:03d}")
            if done:
                break

        if self.bar is not None:
            self.bar.close()
            self.bar = None

        if not cfg.train.skip_final_save:
            self.save("final")
        if self.perceptual is not None:
            p = self.perceptual
            stats = acc.gather(torch.tensor([[p.micro, p.applied, p.raw_sum]],
                                           device=acc.device, dtype=torch.float64)).tolist()
            acc.print(f'perceptual summary per rank [microbatches, applied, raw_loss_sum]: {stats}')
        if self.adversarial is not None:
            from safetensors.torch import save_file
            import hashlib
            state = {k:v.detach().cpu().contiguous() for k,v in self.adversarial.discriminator.state_dict().items()}
            digest = hashlib.sha256(b''.join(v.numpy().tobytes() for v in state.values())).hexdigest()
            gathered = [None]*acc.num_processes
            if acc.num_processes > 1:
                torch.distributed.all_gather_object(gathered,digest)
            else:
                gathered = [digest]
            if len(set(gathered)) != 1:
                raise RuntimeError('Discriminator replicas diverged')
            acc.print(f'adversarial discriminator synchronized: {digest}; updates/rank={self.adversarial.micro}')
            if acc.is_main_process:
                path = Path(cfg.train.output_dir)/cfg.train.run_name/'eye-discriminator.safetensors'
                save_file(state,str(path))
        if offloader is not None:
            stats = acc.gather(torch.tensor([[offloader.peak_bytes / 2**20,
                offloader.offloaded_tensors, offloader.budget_fallbacks]], device=acc.device))
            acc.print(f'activation offload per rank [peak host MiB, copies, budget fallbacks]: {stats.tolist()}')
        acc.end_training()

    def _log(
        self, loss: torch.Tensor, hf: torch.Tensor | None, epoch: int, t0: float
    ) -> None:
        acc = self.accelerator
        value = acc.gather(loss.detach().float().repeat(1)).mean().item()
        # gather is collective, so it must run on every rank or the non-main ranks deadlock.
        hf_value = (
            acc.gather(hf.detach().float().repeat(1)).mean().item() if hf is not None else None
        )

        # Peak memory must be gathered, not read locally: ranks are NOT symmetric. Rank 0 carries
        # NCCL buffers and does the checkpoint export, so reporting only its number both overstates
        # the typical rank and hides which rank is actually the batch-size ceiling.
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(acc.device)
            local_memory = [torch.cuda.max_memory_allocated(acc.device),
                            torch.cuda.memory_allocated(acc.device),
                            torch.cuda.memory_reserved(acc.device), total - free]
        else:
            local_memory = [0, 0, 0, 0]
        memory = acc.gather(torch.tensor([local_memory], device=acc.device,
                                         dtype=torch.float64)).tolist()

        if not acc.is_main_process or self.progress_mode == "off":
            return
        # Every group's LR, not just group 0's. With `[component_lr]` set, printing `get_last_lr()[0]`
        # showed only the first component -- so a run with self_attn 4e-5 / cross_attn 1e-5 /
        # mlp 8e-6 reported a flat 4e-5 and looked exactly like a run where per-component LR had
        # silently not applied. It had; only the log was blind. Slash-joined to match the `peak
        # A/B GB` convention already used for per-rank memory, and collapsed to one number when the
        # groups agree, so the common case reads as before. The startup report names the groups in
        # this same order.
        # The scheduler has already advanced to the next unscaled rate. Staged
        # logs must show the penalized LR that produced the reported update.
        lrs = (self._last_update_lrs if self.cfg.resolution_schedule.enabled
               else self.scheduler.get_last_lr())
        lr = "/".join(f"{v:.2e}" for v in lrs) if len(set(lrs)) > 1 else f"{lrs[0]:.2e}"
        dt = (time.time() - t0) / max(1, self.cfg.train.log_every)
        def memory_field(column):
            return "/".join(f"{row[column] / 2**30:.2f}" for row in memory) + "GiB"
        mem = (f"peak_alloc {memory_field(0)}  allocated {memory_field(1)}  "
               f"reserved {memory_field(2)}  device_used {memory_field(3)}")
        # total = mse + hf, so the two printed numbers add up to the first.
        parts = f"loss {value:.4f}"
        if 'texture' in getattr(self, 'last_stats', {}):
            parts += f"  texture {self.last_stats['texture']:.5f}"
        if 'd_loss' in getattr(self, 'last_stats', {}):
            parts += f"  D {self.last_stats['d_loss']:.4f}"
            if 'g_loss' in self.last_stats:
                parts += f"  adv {self.last_stats['g_loss']:.4f}"
        if 'lpips' in getattr(self, 'last_stats', {}):
            parts += f"  lpips {self.last_stats['lpips']:.4f}"
        if 'self_flow_alignment' in getattr(self, 'last_stats', {}):
            parts += f"  sf {self.last_stats['self_flow_alignment']:.4f}"
        if 'slider_pos' in getattr(self, 'last_stats', {}):
            # Weighted terms of `loss` for strength +1 / -1 (and the anchor, summed over both).
            parts += (f"  pos {float(self.last_stats['slider_pos']):.4f}"
                      f"  neg {float(self.last_stats['slider_neg']):.4f}")
            if 'slider_anchor' in self.last_stats:
                parts += f"  anchor {float(self.last_stats['slider_anchor']):.4f}"
        if hf_value is not None:
            parts += f" (mse {value - hf_value:.4f} + hf {hf_value:.4f})"
        if getattr(self, "last_grad_norm", None) is not None:
            grad_norm = self.last_grad_norm.item()
            clip_factor = (min(1.0, self.cfg.optimizer.max_grad_norm / (grad_norm + 1e-6))
                           if self.cfg.optimizer.max_grad_norm > 0 else 1.0)
            parts += f"  grad {grad_norm:.3g}  clip {clip_factor:.3f}"
        # Fraction of rows optimal transport actually reordered. 0.00 means it is doing nothing --
        # either the batch is 1, or the pairing happened to be optimal already.
        if "ot_moved" in getattr(self, "last_stats", {}):
            parts += f"  ot {self.last_stats['ot_moved']:.2f}"
        # Already weighted, so it is the actual contribution to `loss`, not the raw cosine gap.
        # Printed only on steps the probe ran; a blank means "not evaluated", not "zero".
        if self.last_preserve is not None:
            parts += f"  pres {self.last_preserve.item():.4f}"

        if self.bar is not None:
            # tqdm already draws step/total, percentage, elapsed and its own ETA, so the postfix
            # carries only what it cannot know.
            self.bar.set_description_str(f"stage{epoch + 1}" if self.cfg.resolution_schedule.enabled else f"e{epoch}")
            self.bar.set_postfix_str(f"{parts}  lr {lr}  {mem}", refresh=True)
            return

        eta = self._eta()
        print(
            f"{'stage' + str(epoch + 1) if self.cfg.resolution_schedule.enabled else 'e' + str(epoch)} step {self.global_step}/{self.total_steps} ({self.progress_pct():.0f}%)  "
            f"{parts}  lr {lr}  {dt:.2f}s/it  {mem}"
            + (f"  eta {eta}" if eta else ""),
            flush=True,
        )

        if self.cfg.resolution_schedule.enabled:
            stage = self.stage_index
            info = dict(stage=stage + 1, resolution=self.cfg.resolution_schedule.stages[stage].resolution,
                        completed=self.global_step - sum(self.sampler.budgets[:stage]),
                        total=self.sampler.budgets[stage], images=self.stage_images[stage],
                        attempts=self.stage_attempts[stage],
                        lr_mul=self.cfg.resolution_schedule.stages[stage].lr_mul,
                        effective_batch=self.cfg.train.batch_size[self.cfg.resolution_schedule.stages[stage].resolution]
                        * acc.num_processes * self.cfg.train.gradient_accumulation_steps)
            _emit(self, 'BLUFLOW_STAGE ' + json.dumps(info))

    # ---------------------------------------------------------------- progress

    def progress_pct(self) -> float:
        return 100.0 * self.global_step / max(1, self.total_steps)

    def _eta(self) -> str | None:
        """Remaining wall clock, from the mean step time since the anchor.

        Deliberately a cumulative mean rather than the instantaneous `dt`: with bucketing -- and
        far more so with multi-resolution -- step cost swings with the bucket that happened to come
        up, so an ETA off the last interval would jump by several-fold between log lines. The mean
        over many steps sees the bucket mix in its true proportion. Returns None until at least one
        step has elapsed past the anchor.
        """
        if self._eta_anchor is None:
            return None
        t_anchor, s_anchor = self._eta_anchor
        done = self.global_step - s_anchor
        remaining = self.total_steps - self.global_step
        if done <= 0 or remaining <= 0:
            return None
        return _fmt_duration((time.time() - t_anchor) / done * remaining)

    def _report(self) -> None:
        if not self.accelerator.is_main_process:
            return
        cfg = self.cfg
        trainable, total = count_parameters(
            self.transformer,
        )
        print(f"\ndataset  {len(self.dataset)} samples, {len(self.sampler.bucket_indices)} buckets")
        subsets = getattr(self.dataset, "subset_report", [])
        if len(subsets) > 1:
            # Worth printing even though it is configured: a subset's share of the epoch is
            # `entries x num_repeats`, not its file count, and that is the number that decides how
            # hard a regularization set actually pulls.
            sample_total = max(1, len(self.dataset))
            for sub, n in subsets:
                flag = "" if sub.texture else "  [fullres only -- keeps its captions]"
                print(f"         {Path(sub.path).name:<24} {n:>6} entries  {100 * n / sample_total:4.1f}% "
                      f"of the {'dataset' if cfg.resolution_schedule.enabled else 'epoch'}  x{sub.num_repeats}{flag}")
        if self.dataset.tier_report is not None:
            summary = self.dataset.tier_report.summary()
            if summary:
                print(summary)
        if cfg.resolution_schedule.enabled:
            print(f"steps    {self.total_steps} successful updates, {self.accelerator.num_processes} process(es)")
            for i, (stage, budget) in enumerate(zip(cfg.resolution_schedule.stages, self.sampler.budgets)):
                print(f"         stage {i + 1}: {stage.resolution}px, {budget} updates, batch/GPU {cfg.train.batch_size[stage.resolution]}, LR x{stage.lr_mul:g}")
        else:
            print(f"steps    {self.total_steps} total, {self.steps_per_epoch}/epoch, "
                  f"{self.accelerator.num_processes} process(es)")
        n_proc = self.accelerator.num_processes
        if n_proc > 1 and not cfg.resolution_schedule.enabled:
            # The trap this exists for: the same config on N GPUs takes N times FEWER optimizer
            # steps, because each rank consumes whole batches. The number is right there in the
            # line above and still reads as "it is running", so it is spelled out.
            print(f"         ^ {n_proc} GPUs means {n_proc}x FEWER optimizer steps than the same "
                  f"config on one ({self.total_steps * n_proc} single-GPU), at {n_proc}x the "
                  f"effective batch. Multiply `epochs` by {n_proc} to match a single-GPU run.")
            if any(p.mode == "texture" for p in cfg.curriculum.phases):
                # Only reachable with the override on -- the constructor refuses otherwise.
                print("         ^ WARNING: texture mode under DDP, override ON. This produced "
                      "progressively worsening anatomy twice. You are re-testing an unproven fix; "
                      "compare against a single-GPU control at matched steps.")
        print(f"mode     {'LoRA/' + cfg.adapter.kind if cfg.is_lora else 'full finetune'}")
        if getattr(self, "slider", None) is not None:
            print(self.slider.describe())
        print(f"params   {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")
        if self.param_report is not None:
            print(self.param_report.summary())
        if cfg.block_lr.enabled:
            print(cfg.block_lr.summary())
        if self.quant_info is not None:
            nq, ntotal, use_qmm, max_tokens, wmean = self.quant_info
            how = "auto" if cfg.quant.use_quantized_matmul == "auto" else "explicit"
            if cfg.quant.frozen_matmul:
                rotation = "Hadamard group 256" if cfg.quant.use_hadamard else "no rotation"
                print(f"quant    {cfg.quant.preset}: frozen INT8 storage + INT8 matmul, "
                      f"{nq}/{ntotal} Linear quantized; {rotation}; BF16 adapters/attention")
                from .int8_calibration import has_profile
                from .frozen_matmul import uses_fixed_ada_tile
                if has_profile(self.accelerator.device):
                    print("         calibrated INT8 kernels; per-shape autotuning disabled for profiled projections")
                elif uses_fixed_ada_tile(self.accelerator.device):
                    print("         Ada INT8 tile 128x128x128; per-shape matmul autotuning disabled")
            elif cfg.quant.backend == "convrot":
                print(f"quant    ConvRot frozen/int8, {nq}/{ntotal} Linear quantized, "
                      f"skip={cfg.quant.skip_policy}; INT8 matmul ON, rotation group 256")
            else:
                print(f"quant    {cfg.quant.mode}/{cfg.quant.weights_dtype}, {nq}/{ntotal} Linear "
                      f"quantized, skip={cfg.quant.skip_policy}")
                print(f"         quantized matmul {'ON' if use_qmm else 'OFF'} ({how}; auto defaults to off)")
        f = cfg.flow
        flow_bits = [f.timestep_sample_method]
        if f.timestep_sample_method == "logit_normal":
            flow_bits.append(f"sigmoid_scale {f.sigmoid_scale}")
        elif f.timestep_sample_method == "moss":
            flow_bits.append(f"Kohaku static recipe mu={f.moss_mu:g}, std={f.moss_std:g}, "
                             f"cutoff={f.moss_cutoff:g}, uniform={f.moss_uniform_mix:g}, "
                             f"p_full={f.moss_p_full:g}, grid={f.moss_steps}")
        flow_bits.append("flux_shift" if f.flux_shift else f"shift {f.shift if f.shift else 'none'}")
        if f.dual_timestep:
            teacher_label = 'Self-Flow EMA teacher' if cfg.self_flow.enabled else 'no teacher'
            flow_bits.append(f"dual timestep ON (mask={f.dual_timestep_mask_ratio:g}; {teacher_label})")
        if cfg.rti.enabled:
            print(f"rti      keep {self.keep_fraction:.3f}, {cfg.rti.dense_prefix_blocks} dense prefix / "
                  f"{cfg.rti.dense_suffix_blocks} dense suffix, {cfg.rti.size_buckets} size buckets")
        if f.use_ot:
            # Batch size 1 makes OT a no-op, and the per-bucket map can produce one silently.
            ones = [b for b in self.sampler.bucket_indices
                    if self.sampler.batch_size_for(b) < 2]
            note = f" -- NO-OP in {len(ones)} bucket(s) at batch size 1" if ones else ""
            flow_bits.append(f"optimal transport ON{note}")
        print(f"flow     {', '.join(flow_bits)}")
        if cfg.train.compile:
            print(f"compile  {cfg.train.compile} (dynamic={cfg.train.compile_dynamic}, "
                  f"regional={cfg.train.compile_regional}); first steps include compile time")
            if cfg.train.model_family == "moss_image" and cfg.train.gradient_checkpointing:
                boundary = ("inside compile (experimental)" if cfg.train.moss_checkpoint_inside_compile
                            else "outside compile (lower VRAM)")
                print(f"         MOSS checkpoint boundary: {boundary}")
        sched = f"{cfg.schedule.kind}, warmup {cfg.schedule.warmup_steps}, floor {cfg.schedule.min_lr_ratio}"
        if cfg.schedule.kind == "rex":
            sched += f", d={cfg.schedule.d}"
        elif cfg.schedule.kind == "rerex":
            sched += (f", global_d={cfg.schedule.global_d}, local_d={cfg.schedule.local_d}, "
                      f"{cfg.schedule.num_segments} segments, weight_power={cfg.schedule.weight_power}")
        print(f"sched    {sched}")
        if self.hf_patch is not None:
            print(f"hf loss  scale {cfg.flow.hf_scale}, exponent {cfg.flow.hf_exponent}, "
                  f"patch {self.hf_patch}")
        if cfg.curriculum:
            print(cfg.curriculum.report(self.total_steps)
                  + f"\n        mapping {cfg.flow.phase_mapping}")
        if any(p.mode == "texture" for p in cfg.curriculum.phases):
            # Printed because texture mode degrades *silently*: with no slack between source and
            # canvas the crop is forced, energy selection chooses from one candidate, and the run
            # is indistinguishable from one that is actually choosing. Header-only reads, so this
            # costs milliseconds even on thousands of files.
            from PIL import Image as _Image

            from ..data.texture import describe_feasibility
            sizes = []
            # Texture-exempt subsets are never cropped, so including them would describe the crop
            # freedom of images that will never be cropped -- and a regularization set of small
            # flat squares would drag every "fits %" down and invent a "no room to choose" warning
            # about a canvas that is in fact fine for everything it will actually see.
            for p in {e.path for e in self.dataset.entries if e.texture_ok}:
                try:
                    with _Image.open(p) as im:
                        sizes.append(im.size)
                except OSError:
                    continue
            print(describe_feasibility(sizes, cfg.dataset.texture))
        est = estimate_optimizer_bytes(trainable, cfg.optimizer) / 1e9
        if cfg.optimizer.kind == "sinkgd":
            print(f"optim    standalone SinkGD, {cfg.optimizer.sinkhorn_iterations} normalization passes; "
                  "no momentum, vector RMS normalization")
        if cfg.optimizer.kind in ("adafactor", "came"):
            norm_mode = self.optimizer.param_groups[0]["norm_mode"]
            print(f"optim    update normalization={norm_mode}")
            if cfg.optimizer.kind == "adafactor":
                group = self.optimizer.param_groups[0]
                print(f"optim    first moment={'ON' if group['use_first_moment'] else 'OFF'} "
                      f"(second beta={group['betas'][1]})")
            if cfg.optimizer.kind == "adafactor" and cfg.adapter.kind != "none" and norm_mode == "relative":
                print("note: Adafactor relative normalization can nearly stall zero-initialized LoRA "
                      "projections at small learning rates. Consider optimizer.norm_mode='rms_clip' "
                      "and tune the learning rate; max_grad_norm is separate.")
        print(f"optim    {cfg.optimizer.kind}, state ~{est:.1f}GB "
              f"(quantized={cfg.optimizer.quantize_state}, offload={cfg.optimizer.offload_state})\n",
              flush=True)

    # -------------------------------------------------------- checkpoints

    def _unwrap(self, module):
        """Strip DDP and torch.compile wrappers before reading a state dict.

        `keep_torch_compile=False` matters: Accelerate defaults it to True, and `compile_regions`
        stamps `_orig_mod` on the root object it returns -- which, because regional compilation
        runs after the DDP wrap, is the DDP module. `extract_model_from_parallel` then sees that
        stamp on the way out and re-attaches the wrapper it had just removed.
        """
        return self.accelerator.unwrap_model(module, keep_torch_compile=False)

    def save(self, tag: str) -> None:
        acc = self.accelerator
        acc.wait_for_everyone()

        # `save_state` is collective -- every rank must call it -- so it happens before the
        # main-process-only export below, not after.
        if self.cfg.train.save_optimizer_state:
            self.save_full_state(tag)

        if not acc.is_main_process:
            return

        cfg = self.cfg
        # sd-scripts' convention: one file per checkpoint, named `<run>-stepNNNNNN.safetensors`,
        # sitting directly in the run directory. A folder per checkpoint made every LoRA in a run
        # share the same filename, so twenty of them could not live in one ComfyUI models folder
        # without being renamed by hand.
        #
        # Only exports that are genuinely ONE file get flattened. A full finetune without
        # `save_native` writes `transformer/` and `text_conditioner/` as diffusers directories, so
        # it keeps a directory -- that is the format's shape, not a naming choice.
        stem = cfg.train.run_name if tag == "final" else f"{cfg.train.run_name}-{tag}"
        single_file = cfg.is_lora or cfg.train.save_native
        dest = self.out_dir if single_file else self.out_dir / stem
        dest.mkdir(parents=True, exist_ok=True)

        # `keep_torch_compile=False` is load-bearing, not tidiness. Accelerate's default is True,
        # and under regional compilation that makes `unwrap_model` re-attach the compiled wrapper
        # it just removed -- returning keys with `module.` at the root and `_orig_mod` inside every
        # block. The export then failed with 560 unmapped keys, at the first checkpoint rather than
        # at startup. `convert.strip_wrappers` catches it too; this stops it happening at all.
        transformer = self._unwrap(self.transformer)
        from ..modeling.export import export_checkpoint
        from ..modeling.checkpoint_metadata import optimizer_snapshot

        # Batch size may vary by resolution tier. Keep that mapping in metadata
        # rather than pretending all updates have a single effective batch size.
        # These are configured full-batch sizes; partial/mixed-tier accumulation
        # groups can contain different actual sample counts.
        batch_scale = cfg.train.gradient_accumulation_steps * acc.num_processes
        batch_size = cfg.train.batch_size
        effective_batch_size = (
            {tier: size * batch_scale for tier, size in batch_size.items()}
            if isinstance(batch_size, dict) else batch_size * batch_scale
        )
        runtime = optimizer_snapshot(self.optimizer)
        runtime['block_lr_multipliers'] = cfg.block_lr.expanded()
        timing = self.timing.snapshot()
        runtime.update(timing)
        if cfg.resolution_schedule.enabled:
            runtime['resolution_schedule_state'] = self._resolution_state()
        runtime.update(
            save_tag=tag,
            successful_updates=self.successful_updates, skipped_updates=self.skipped_updates,
            source_checkpoint=self.source_checkpoint,
            world_size=acc.num_processes,
            effective_batch_size=effective_batch_size,
        )
        if getattr(self, "adapter_initialization", None) is not None:
            runtime['adapter_initialization'] = self.adapter_initialization
        offloader = getattr(transformer, '_activation_offloader', None)
        if offloader is not None:
            runtime['activation_offload_peak_host_bytes_rank0'] = offloader.peak_bytes
        if self.rti_interface is not None:
            runtime.update(rti_keep_fraction=self.keep_fraction, rti_phase=self.rti_phase)
        n = export_checkpoint(transformer, dest, stem, cfg, self.dtype, self.global_step,
                              runtime=runtime)

        written = dest / f"{stem}.safetensors" if single_file else dest
        # `state.json` lives beside the optimizer state, which is the only thing that reads it.
        # A flat single-file checkpoint carries its step in the safetensors metadata instead, so an
        # inference-only export is one file with nothing to keep in sync alongside it.
        if cfg.train.save_optimizer_state:
            (self.out_dir / f"{stem}-state" / "state.json").write_text(
                json.dumps({"global_step": self.global_step, "tensors": n,
                            "successful_updates": self.successful_updates, "skipped_updates": self.skipped_updates,
                            "source_checkpoint": self.source_checkpoint,
                            "adapter_initialization": getattr(self, "adapter_initialization", None),
                            "training_timing": timing,
                            "flow_rng_version": self.flow_rng.version,
                            "block_lr_multipliers": cfg.block_lr.expanded(),
                            "resolution_schedule_state": self._resolution_state()}, indent=2)
            )
        _emit(self, f"saved {written} ({n} tensors)")
        if tag == "final":
            # GUI queue snapshots only after a successful final checkpoint write.
            _emit(self, "BLUFLOW_FINAL_SAVE " + json.dumps({
                "output_dir": str(self.out_dir.resolve()), "step": self.global_step,
            }))
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        """Keep only the newest `keep_last_n` periodic checkpoints.

        Deliberately narrow, because this deletes. A candidate must carry THIS run's name and a
        periodic tag (`-epochNNN` / `-stepNNNNNN`); `final` -- which is the bare run name, with no
        tag -- and anything else in the directory are never candidates. Ordering is by the recorded
        step rather than mtime, so a resumed run cannot delete the wrong one, and epoch-tagged and
        step-tagged checkpoints order correctly against each other.

        Everything belonging to one checkpoint is removed together: the file, its `_te` sibling,
        and its `-state` directory. Directory-form checkpoints (a full finetune without
        `save_native`, and anything written before the layout was flattened) are still recognised,
        so `keep_last_n` does not quietly stop pruning an existing run.
        """
        keep = self.cfg.train.keep_last_n
        if not keep or keep < 1:
            return

        run = re.escape(self.cfg.train.run_name)
        # `<run>-step000060` / `<run>-epoch005`, and the bare `step000060` of the old layout.
        tagged = re.compile(rf"(?:{run}-)?(epoch|step)(\d+)$")
        groups: dict[tuple[str, int], list[Path]] = {}
        for entry in self.out_dir.iterdir():
            name = entry.name
            if entry.is_file():
                if entry.suffix not in (".safetensors", ".json"):
                    continue
                name = entry.stem.removesuffix("_te")
            elif name.endswith("-state"):
                name = name[: -len("-state")]
            m = tagged.fullmatch(name)
            if not m:
                continue
            groups.setdefault((m.group(1), int(m.group(2))), []).append(entry)

        # An epoch tag counts epochs and a step tag counts steps, so the two are not comparable as
        # raw numbers. `state.json` carries the true global_step where it exists; otherwise the
        # safetensors metadata does; otherwise fall back to the tag number, which is correct within
        # a single tag kind and is the only case that can mix.
        def ordinal(kind: str, num: int, paths: list[Path]) -> int:
            for p in paths:
                state = (p / "state.json") if p.is_dir() else None
                if state is not None and state.exists():
                    try:
                        return int(json.loads(state.read_text())["global_step"])
                    except (ValueError, KeyError, TypeError):
                        pass
                if p.is_file():
                    try:
                        from safetensors import safe_open
                        with safe_open(p, framework="pt") as f:
                            step = (f.metadata() or {}).get("step")
                        if step is not None:
                            return int(step)
                    except Exception:
                        pass
            return num if kind == "step" else num * max(1, self.steps_per_epoch)

        ranked = sorted((ordinal(k, n, ps), (k, n), ps) for (k, n), ps in groups.items())
        for _, _, paths in ranked[:-keep]:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
                _emit(self, f"pruned {p} (keep_last_n={keep})")

    def save_full_state(self, tag: str) -> None:
        """Optimizer + scheduler + weights, for resuming. Separate from `save` because the
        exported checkpoint deliberately drops optimizer state -- it is for inference -- and a
        resume that silently restarted the moments would quietly change the training dynamics.

        Sits in `<run>-<tag>-state/` beside the flat checkpoint rather than inside it: accelerate
        writes a directory, and a directory cannot live inside a `.safetensors` file.
        """
        stem = self.cfg.train.run_name if tag == "final" else f"{self.cfg.train.run_name}-{tag}"
        dest = self.out_dir / f"{stem}-state" / "accelerator"
        # SDNQTensor is a wrapper without ordinary tensor storage. Native
        # torch serialization preserves its quantized payload and metadata.
        self.accelerator.save_state(str(dest), safe_serialization=self.cfg.quant.mode != 'training')
        # Accelerate custom-object files are main-rank saves; these streams differ
        # per rank, so each process writes its own state explicitly.
        self.flow_rng.save(dest)
        self.accelerator.wait_for_everyone()

    def _resume(self, path: str) -> None:
        p = Path(path)
        # Accept the checkpoint FILE as well as its state directory. Under the flat layout the
        # thing a user has in hand is `<run>-step000060.safetensors`, and pointing `resume_from` at
        # it is the obvious move -- so it resolves to the sibling `-state` directory rather than
        # erroring about a state.json that was never meant to sit next to it.
        if p.is_file() and p.suffix == ".safetensors":
            sibling = p.with_name(p.stem.removesuffix("_te") + "-state")
            if not sibling.exists():
                raise FileNotFoundError(
                    f"{p} has no optimizer state beside it (expected {sibling}). It was written "
                    f"with save_optimizer_state disabled, so it can be used for inference but not "
                    f"resumed."
                )
            p = sibling

        state_file = p / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"no state.json in {p}; not a checkpoint directory")

        acc_dir = p / "accelerator"
        if not acc_dir.exists():
            raise FileNotFoundError(
                f"{p} has no saved optimizer state (expected {acc_dir}). It was written with "
                f"save_full_state disabled, so it can be used for inference but not resumed."
            )
        saved_state = json.loads(state_file.read_text())
        self.cfg.block_lr.check_resume(saved_state.get('block_lr_multipliers'))
        staged = saved_state.get('resolution_schedule_state')
        if bool(staged) != self.cfg.resolution_schedule.enabled:
            raise ValueError('Full-state resume must preserve staged-resolution mode; use weights-only to start a new plan')
        if staged:
            if staged['identity'] != self.sampler.identity:
                raise ValueError('Staged resume plan/data/batch size/GPU count changed; use weights-only for a new plan')
            multipliers = [s.lr_mul for s in self.cfg.resolution_schedule.stages]
            if staged.get('lr_multipliers', [1.0] * len(multipliers)) != multipliers:
                raise ValueError('Staged resume LR multipliers changed; use weights-only for a new plan')
            self.stage_index, self.stage_cursor = staged['stage'], staged['cursor']
            self.stage_images, self.stage_attempts = staged['images'], staged['attempts']
        self.timing.resume(saved_state.get("training_timing"))
        self.adapter_initialization = saved_state.get("adapter_initialization")
        self.flow_rng.load(acc_dir, required='flow_rng_version' in saved_state)
        from sdnq.training import SDNQTensor
        from sdnq.dequantizer import SDNQDequantizer
        with torch.serialization.safe_globals([SDNQTensor, SDNQDequantizer]):
            self.accelerator.load_state(str(acc_dir))

        self.global_step = saved_state["global_step"]
        self.skipped_updates = saved_state.get("skipped_updates", 0)
        self.successful_updates = saved_state.get("successful_updates", self.global_step - self.skipped_updates)
        if saved_state.get('source_checkpoint') is not None:
            self.source_checkpoint = saved_state['source_checkpoint']
        elif self.source_checkpoint is not None:
            # Old resume states cannot establish the original training source.
            self.source_checkpoint = dict(self.source_checkpoint,
                role='initialization_before_legacy_resume',
                original_training_source_verified=False, resume_from=str(p.resolve()))
        if self.cfg.resolution_schedule.enabled:
            self.start_epoch = self.resume_skip_batches = 0
            self._micro = self.global_step * self.cfg.train.gradient_accumulation_steps
            self.accelerator.print(f'resumed at successful update {self.global_step}; staged data cursor {self.stage_cursor}')
            return
        self.start_epoch = self.global_step // max(1, self.steps_per_epoch)
        self.resume_skip_batches = ((self.global_step % max(1, self.steps_per_epoch))
                                    * self.cfg.train.gradient_accumulation_steps)
        self._micro = self.global_step * self.cfg.train.gradient_accumulation_steps
        self.accelerator.print(f"resumed at step {self.global_step} (epoch {self.start_epoch})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Mage-Flow")
    ap.add_argument("config", help="path to a TOML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and run one step (one per resolution stage in staged mode), then exit without saving")
    ap.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    ap.add_argument("--no-save", action="store_true", help="benchmark without writing checkpoints")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.dry_run:
        if cfg.resolution_schedule.enabled:
            count = len(cfg.resolution_schedule.stages)
            cfg.train.max_steps = count
            cfg.resolution_schedule.stages = [replace(s, share=1 / count) for s in cfg.resolution_schedule.stages]
        else:
            cfg.train.max_steps = 1
            cfg.train.epochs = 1
    if args.dry_run or args.no_save:
        cfg.resolution_schedule.save_stage_end = False
        cfg.train.save_every_epochs = None
        cfg.train.save_every_steps = None
        cfg.train.skip_final_save = True

    cfg.resolution_schedule.validate(cfg)
    Trainer(cfg, config_path=args.config).train()


if __name__ == "__main__":
    main()
