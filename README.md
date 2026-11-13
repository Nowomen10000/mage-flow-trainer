# BluFlow-Trainer

**MOSS-Image LoRA, LyCORIS, and full-model training, with a desktop GUI and CLI.**
Train with BF16 weights or SDNQ quantization, cache Qwen text embeddings to free
VRAM, and use aspect-ratio buckets or multiple resolution tiers in the same run.

Formerly `mage-flow-trainer`. Existing TOMLs, caches, launch commands, and the
`trainer` Python module remain compatible. The repository now lives at
[bluvoll/BluFlow-Trainer](https://github.com/bluvoll/BluFlow-Trainer).

[Start training](#start-training-moss-image) · [Model files](#moss-model-files) ·
[Multi-resolution](#datasets-and-multi-resolution) · [GUI and queue](#gui-and-training-queue) ·
[Support tiers](#model-support-tiers) · [Documentation](#documentation)

## What it supports

- **MOSS-Image Base/SFT:** PEFT LoRA, LyCORIS LoCon/LoKr/DoRA, and full finetuning.
- **Memory controls:** SDNQ INT8/UINT8 weight storage, independent text-encoder
  quantization, gradient checkpointing, and optional optimizer-state quantization/offload.
- **Cached conditioning:** reusable image latents and SQLite caption embeddings,
  including precomputed tag/caption variations. The text encoder is released before training.
- **Flexible batching:** aspect-ratio buckets, multi-resolution datasets with
  per-tier batch sizes, or packed `torch_varlen` attention across resolutions.
- **Training controls:** `torch.compile`, single-GPU training, Linux DDP,
  timestep-distribution previews, and optional HF/paired-texture losses.
- **Concept sliders:** train an adapter that moves a concept toward a positive prompt at
  +1 and a negative prompt at -1 (port of ai-toolkit's Concept Slider), including on
  SDNQ-quantized bases. See the [concept slider guide](docs/concept-sliders.md).
- **GUI workflow:** editable TOMLs, dataset subsets, live metrics, a training queue,
  and MOSS fused/separated QKV export.

## Start training MOSS-Image

### 1. Install and launch

Clone the repository, then run the installer for your platform:

```bash
git clone https://github.com/bluvoll/BluFlow-Trainer.git
cd BluFlow-Trainer
```

| Platform | Install | Launch GUI |
|---|---|---|
| Linux | `./install.sh` | `./start-gui.sh` |
| Windows | `install.bat` | `start-gui.bat` |

The installers create `venv/`. The tested stack uses **PyTorch 2.10.0 + CUDA 12.8**;
pinned packages and Git revisions are recorded in `uv.lock` and `requirements.txt`.
**SDNQ is pinned to 0.2.9**, validated on Linux RTX 4090s with ten-epoch MOSS
LoCon comparisons, Kahan, text caching, finetuning smokes and two-GPU DDP.
The integration handles the new kernel interface, deterministic DDP storage order,
older optimizer states, and a remaining upstream Kahan/weight-decay interaction.
Kahan stays optional and off by default; better update precision does not guarantee
better visual results. Native Windows validation below was performed on 0.2.4.

To launch from an already configured environment, use `python -m trainer.gui`.
Run CLI commands below with that environment activated, or substitute its Python
executable (`venv/bin/python` on Linux, `venv\Scripts\python.exe` on Windows).

### 2. Choose a preset

| Goal | Preset | Base-weight storage |
|---|---|---|
| Train a LoCon adapter | [MOSS LoCon](configs/moss-image-locon-experimental.toml) | Frozen SDNQ UINT8 |
| Finetune the transformer | [MOSS full finetune](configs/moss-image-finetune-experimental.toml) | Training-mode SDNQ UINT8 |

Both presets use **1024px**, batch size 1, cached text, checkpointing, and compilation.
Their filenames retain the historical `experimental` suffix for compatibility;
**MOSS itself has first-class support**. Treat their learning rate and duration as
starting settings to adjust for your dataset and optimizer.

### 3. Configure and train

1. Load a MOSS preset in the GUI and choose a unique run name. Select
   **MOSS-Image (Tier 1)** and set the model package or separate component files.
2. Set the dataset folder containing images and matching `.txt` captions. Choose
   resolution and batch size, then use **Cache latents** with the selected MOSS VAE.
3. Choose the adapter or finetuning method. SDNQ uses **frozen** mode for adapters
   and **training** mode for full finetuning. Adapters leave modulation/AdaLN frozen.
4. Keep **Cache text embeddings** enabled. Start with text-cache batch size 1;
   enable text-encoder quantization if the Qwen encoder does not fit. Set caption
   augmentation before building the cache.
5. Select GPU(s) and press **Start Training**. The trainer builds missing text
   embeddings automatically. Multi-GPU training is supported on Linux;
   the Windows GUI uses one selected GPU.

For the CLI, copy either preset to `configs/my-moss-run.toml`, edit its paths and
training settings, then run from the repository directory:

```bash
python -m trainer.tools.cache_latents cache-config configs/my-moss-run.toml --batch-size 1
python -m trainer.training.train configs/my-moss-run.toml
```

`cache-config` uses the TOML's model/VAE, subsets, and resolution tiers. The
presets read cached latents during training, so complete latent caching first.
See the [MOSS guide](docs/moss-image-experimental.md) for loading and training details.

## MOSS model files

MOSS uses a **5.082B transformer**, **Qwen3.5-9B** text conditioning, and its packaged
**FLUX.2 VAE with a small decoder**. Its /8 encoder output is packed and normalized
into the transformer's /16 latent layout. Use the VAE matching your base model;
Mage-VAE and VSF3 caches are not interchangeable with MOSS caches.

Point `train.model_path` at a package containing `transformer/`, `text_encoder/`,
`tokenizer/`, and `vae/`, or select separate files under **Model / Output**:

```toml
[train]
model_family = "moss_image"
transformer_path = "/path/to/moss_image_base_bf16.safetensors"
text_encoder_path = "/path/to/qwen3.5_9b_bf16.safetensors"
vae_path = "/path/to/flux2-vae-smalldecoder.safetensors"
```

The standard MOSS architecture configs and tokenizer are bundled, so separate
component files do not require a full Diffusers directory. A model package or
explicit `tokenizer_path` can supply the tokenizer instead.

Use **native BF16 transformer weights** as the source for SDNQ quantization.
Unquantized fused-QKV exports are also accepted and split during loading.
Prequantized inference transformer exports are not training sources.
The text loader accepts supported standalone Qwen quantizations, including
Comfy INT8/ConvRot, but dequantizes them during loading. Enable the trainer's
text-encoder quantization separately to reduce resident encoder memory.
See [supported loaders and formats](docs/moss-image-experimental.md#start).

## Continuing an adapter and SinkGD

A finished LoRA/LoCon can start a new training phase without optimizer-state
files. In **Checkpoints**, set **Resume from** to the adapter `.safetensors`,
choose **Weights only (new phase)**, and use a new run name. Match the original
base, adapter rank, alpha, algorithm and targets. Native separated and BluFlow's
fused MOSS exports are supported. The adapter is loaded directly; the base stays
unchanged, and the new optimizer, schedule and counters start fresh.

CLI/TOML: `train.resume_mode="weights"` with `train.resume_from` set to the file.
The default remains `"exact"` for full saved-state resume.

**SinkGD** is also available in the optimizer selector (`optimizer.kind="sinkgd"`).
This standalone SDNQ-based variant defaults to five Sinkhorn passes, zero weight
decay and no momentum. It supports BF16/SDNQ master-weight updates and optional
Kahan compensation. It differs from Advanced Optimizers' `SinkSGD_adv` and does
not require that package. Its image quality and best LR still need evaluation.
See the [training guide](docs/training-guide.md#optimizer-families) for settings.

## MOSS checkpointing and Windows attention

MOSS defaults to checkpointing **around each compiled block**, which reduced
peak VRAM by about **1.35 GB** in our matched 1024px batch-7 test. Both boundaries
retain the corrected RMSNorm gradients. With DDP, the default disables compiler
partitioning at gradient buckets; DDP still synchronizes weights and gradients.

Enable **Preferences → Enable advanced experimental settings**, then find the
alternative boundary under **Experimental → Experimental MOSS checkpointing →
Checkpoint inside compile**. It can change
speed and numerical results and used more VRAM in that test. For the CLI:

```toml
[train]
compile = "default"
gradient_checkpointing = true
moss_checkpoint_inside_compile = false  # default; true opts into the experiment
```

Existing TOMLs without this setting use the lower-memory default. Text-refinement
blocks remain eager; this switch changes the main blocks' checkpoint boundary.

On native Windows, **MOSS packed training is available with `flash_attn_2`**
and a compatible Flash Attention wheel. Select that backend, then enable
**Pack native resolutions**. Compilation and ordinary multi-resolution tiers can
be used together. Startup tests the selected backend's GPU forward **and backward**
before loading models or building text embeddings; importing the package alone
is not enough. SDPA remains the option without an extra wheel, with packing off.

Validated on an RTX 4090 with Python 3.12, Torch 2.10.0+cu128, SDNQ 0.2.4 and
Flash Attention 2.8.3 from the community
[v0.7.13 Windows release](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.7.13):

```bat
venv\Scripts\python.exe -m pip install --no-deps "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-win_amd64.whl"
```

`install.bat` installs this release automatically after the base dependencies,
selecting a SHA256-pinned wheel for the venv's Python 3.11, 3.12 or 3.13. New
Windows installs prefer Python 3.12; existing venvs keep their interpreter.
The installer checks the Torch/CUDA versions and verifies that the native
extension imports. Training startup still checks GPU forward and backward.
For an existing install, run `venv\Scripts\python.exe -m trainer.tools.install_windows_flash`.
The manual command above is specific to Python 3.12 and the pinned Torch stack.
Other GPU architectures need their own forward/backward validation.

```toml
[train]
attention_backend = "flash_attn_2"
pack_resolutions = true
batch_size = 6
compile = "default"
```

The Windows smoke test completed six compiled Balanced LoRA updates with mixed
512/768/1024 resolution tiers and six images per batch. Separate kernel checks
covered BF16/FP16 outputs and gradients against SDPA, compiled execution, and
isolation between packed samples. These tests do not establish long-run convergence.

`torch_varlen` still selects PyTorch's own implementation and remains disabled
on Windows. Installing the external wheel does not change that backend; select
`flash_attn_2` explicitly for MOSS. On supported platforms, `torch_varlen` needs
working GPU forward **and backward** kernels, not just the Python module.
Native Windows/MSVC builds disable Flash Attention in both
[PyTorch 2.10](https://github.com/pytorch/pytorch/blob/v2.10.0/CMakeLists.txt) and
[2.14.1](https://github.com/pytorch/pytorch/blob/v2.14.1/CMakeLists.txt).
A version upgrade alone therefore does not enable PyTorch's native varlen kernels.
Use the external MOSS backend above, SDPA, or a compatible Linux/WSL2 environment.
SDPA remains valid for MOSS, including multi-resolution
training, but varied caption lengths can cause more recompilation than varlen.
The Windows attention backend uses the same repository Torch and SDNQ pins.

The pinned Torch 2.10 Windows build does not support `expandable_segments`.
The trainer enables it by default on Linux only. `install.sh` also generates
`start-gui.sh` with `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
clearing `PYTORCH_ALLOC_CONF` to avoid conflicting settings. This applies before
the GUI starts and is inherited by caching, calibration, and queued training.
Direct CLI training preserves explicit allocator settings.
On Windows, newly generated launchers preserve `PYTORCH_ALLOC_CONF` and the legacy
`PYTORCH_CUDA_ALLOC_CONF` environment variable. Without an override, PyTorch uses
**native**, our compatibility default.

In **Preferences → CUDA memory allocator**, choose:

- **Default**: respect the launch environment, otherwise use native.
- **Native**: explicitly use native allocation, including when an older launcher
  forces async allocation.
- **cudaMallocAsync**: optional memory savings. Some Windows setups crash with
  this allocator during compiled training; switch back to Native if affected.

The choice is saved in the machine-local GUI preferences, not training TOMLs.
Explicit Native/Async choices override inherited allocator settings for newly
started training, caching, and calibration processes, including queued jobs.
Already running processes are unaffected. Linux allocation remains unchanged.
Async is a separate allocator, not an implementation of expandable segments.
See [PyTorch allocator settings](https://docs.pytorch.org/docs/2.10/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf).

**Existing Windows launchers:** rerun `install.bat` to regenerate them, or remove
these two lines from the old `start-gui.bat`:

```bat
set "PYTORCH_CUDA_ALLOC_CONF="
set "PYTORCH_ALLOC_CONF=backend:cudaMallocAsync"
```

Selecting **Native** in Preferences also overrides those old launcher settings
for GUI jobs without reinstalling anything. For direct CLI training, set the
allocator before starting Python (GUI preferences do not apply to direct CLI):

```powershell
$env:PYTORCH_ALLOC_CONF = "backend:native"
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
venv\Scripts\python.exe -m trainer.training.train configs\your.toml
```

Replace `backend:native` with `backend:cudaMallocAsync` to opt into async allocation.

In a 12-step Windows RTX 4090 smoke test (MOSS, balanced LoRA, batch 6,
512/768/1024 packed resolutions, Flash Attention 2, compile), final CUDA-reported
device usage was **19.21 GiB native versus 15.29 GiB async**. A warm-cache native
repeat reproduced the memory result. Mean step times excluding step 1 were
3.25 s native and 3.36 s async; this is a short smoke test, not a long-run guarantee.
Native `roundup_power2_divisions:4` increased usage to 19.50 GiB in that test.
`backend:native,max_split_size_mb:128` also completed, but final reserved memory
rose to 22.85 GiB and CUDA-reported device usage reached 23.99 GiB. It is not a
recommended default for this workload. This native-only option prevents splitting
large blocks; it does not compact existing allocations, and async ignores it.

Windows training automatically finds installed C++ Build Tools using Microsoft's
`vswhere` and loads the x64 compiler environment, including include/library
paths, for the training process. No system PATH changes or special terminal
are needed when discovery succeeds. GUI runs and queued configs check this
**before caching**; CLI training checks before loading models.
Triton's C launchers use its bundled TinyCC by default, while Inductor keeps
MSVC for C++. This avoids older MSVC toolsets rejecting Triton's generated
C with `C2059: syntax error: '}'`. An explicit `CC` environment variable is
respected. Preflight compiles and loads a small C extension to check the
launcher compiler and Python development files before starting expensive work;
this does not certify GPU kernel support. If a custom `CC` selects an older
MSVC and fails this check, unset it or update that toolset.

Localized Visual Studio installations also need the **English language pack**
for PyTorch's C++ compiler detection. Preflight checks for these resources before
caching or model loading. When they are missing, it starts Visual Studio Installer
to add English and verifies the installation before continuing. When present,
it selects English compiler messages and proceeds normally. This check applies
to every language; your Windows and Visual Studio UI languages stay unchanged.
You can also repair ahead of time with **Preferences → Install English C++ language pack** in
the trainer, or run:

```bat
venv\Scripts\python.exe -m trainer.tools.windows_toolchain --install-english
```

The repair opens the installed Visual Studio Installer for the selected
compiler's installation and requests administrator approval. It adds English
resources without removing other languages. If installation is cancelled or
requires a reboot, resolve that before starting training again.

If the tools are missing, install Visual Studio Build Tools with **Desktop
development with C++**, including **MSVC x64/x86 tools** and a **Windows 10/11
SDK**, then restart the trainer. The full Visual Studio IDE is unnecessary.
The error includes the installer link. If automatic discovery fails, launch from
**x64 Native Tools Command Prompt** and check `where cl`; see Microsoft's
[developer environment instructions](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line).
Triton's bundled C compiler does not replace the C++ compiler for CPU code
generated by Inductor, even during GPU training.

The installer already includes **triton-windows 3.6.0.post26** for Torch 2.10;
keep that matched pair rather than installing the newest Triton release. See
[Triton's Windows compatibility and compiler notes](https://github.com/triton-lang/triton-windows).
Installing a separate `flash-attn` wheel does **not** enable PyTorch's built-in
`torch_varlen` kernel. MOSS currently uses SDPA or `torch_varlen`; using the
external Flash Attention package requires a separate backend integration.

## MOSS LoRA quantization presets

In the GUI, open **Quantization → MOSS adapter preset**:

| Preset | Frozen base computation | Choose it for |
|---|---|---|
| **Standard** | UINT8 storage, BF16 matmul | The established storage-only baseline; speed is secondary. |
| **Balanced** | INT8 storage and matmul, Hadamard rotation (group 256) | Faster training with rotation intended to limit quantization error. |
| **Fast** | INT8 storage and matmul, no rotation | The fastest measured option; it may learn slightly differently. |
| **Custom** | Your explicit quantization settings | Existing configs, unquantized training, or full finetuning. |

All three produced usable, closely matched LoRAs in our MOSS tests. **These names
are not a proven visual-quality ranking**; results can vary by dataset. Standard
retains BF16 matmul. Balanced/Fast also quantize matmul inputs and the frozen
base's input-gradient computation; adapters and attention remain BF16.

Balanced/Fast default to **INT8 kernel calibration → Auto (calibrate once)**. Before
loading models, the trainer validates and benchmarks SDNQ kernel tiles for MOSS
forward/backward projections on each selected GPU, then saves a local profile.
DDP calibrates ranks sequentially. Later launches reuse matching profiles; changing
token counts selects a nearby measured tile instead of retuning 18 candidates.
New shapes can still incur JIT compilation. This selects kernel launch settings;
it does not change your quantization preset, batch size, or learning rate.

First calibration can take several minutes. Use **Quantization → Calibrate INT8
kernels** to refresh results while the selected GPUs are idle, or run:

```bash
python -m trainer.tools.calibrate_int8 configs/my-run.toml --all-devices --force
```

`quant.matmul_tuning = "cached"` uses existing profiles without benchmarking;
`"off"` uses SDNQ's upstream hardware-aware kernel selection. SDNQ 0.2.9
prunes candidates using GPU cache/shared-memory limits; calibration also applies
that pruning before measuring valid tiles.
Explicit `SDNQ_TRITON_MM_*` overrides take precedence. Profiles are invalidated
by GPU/software/kernel changes. See [calibration details](docs/training-guide.md#int8-kernel-calibration).
Standard, full finetuning, and text-encoder quantization are unaffected.

On an RTX 4090, a matched 1024px Tangobat LoCon test (rank/alpha 16, batch 1,
accumulation 2, cached text, checkpointing and compile) averaged **3.152 / 2.590 /
2.520 seconds per optimizer update** for Standard/Balanced/Fast. These are 639
updates from epochs 2–10, excluding the first epoch. Fast was about **2.8% higher
throughput than Balanced**. The older Standard run slowed midway, so its roughly
25% throughput comparison is less controlled. These are separate runs, not a
hardware-independent speed guarantee.

For CLI training, set the preset in your existing TOML and launch normally:

```toml
[quant]
preset = "balanced"  # "standard", "balanced", "fast", or "custom"
```

```bash
python -m trainer.training.train configs/my-moss-locon.toml
```

The presets require **MOSS BF16 adapter training**. Balanced/Fast support
**PEFT LoRA or bypass LyCORIS LoCon**, with **BF16 adapter dtype and random
initialization**. For PEFT, select **LoRA (PEFT)** in the GUI, set adapter dtype to
**bfloat16**, and choose Balanced or Fast. The equivalent TOML is:

```toml
[train]
model_family = "moss_image"
dtype = "bfloat16"

[adapter]
kind = "lora"
dtype = "bfloat16"
init = "random"
```

For LyCORIS, use `kind = "lycoris_lora"`, `lycoris_algo = "locon"`, and
`lycoris_bypass = true` instead. LoKr and DoRA are not supported by Balanced/Fast.

LoCon adapter inputs are made contiguous to avoid an out-of-bounds gradient write
in the pinned LyCORIS atomic kernel with strided inputs. This guard preserves the
SDNQ base-model kernels. Both INT8 calibration and LyCORIS kernel tuning avoid
CUDA graph capture with `cudaMallocAsync`, which can abort while freeing captured
allocations. They use CUDA-event timing instead, retaining fused training kernels
and model compilation. These guards were verified on Linux; Windows confirmation
is pending. If a LyCORIS-specific crash persists, PEFT LoRA can use the same
Balanced/Fast preset.

Use native BF16 source transformer weights. SDNQ **0.2.9** performs quantization;
Balanced/Fast do **not** require ComfyUI or Comfy Kitchen. They support frozen-base
adapter training, **not full finetuning**. The GUI and CLI reject unsupported
combinations before loading the model.

Presets override mode, backend, weight dtype, matmul, rotation, group size,
stochastic rounding, and dynamic loss threshold; the GUI locks those fields.
Switch to Custom to edit them. Omitted `preset` means Custom, so existing TOMLs
keep their settings and `use_quantized_matmul = "auto"` still means off.
The preset and resolved settings are saved in checkpoint metadata.

Checkpoint `training_state` also records UTC `training_started_at`,
`session_started_at`, and `checkpoint_saved_at` timestamps, plus
`session_elapsed_seconds` and cumulative `training_elapsed_seconds`.
The save timestamp and durations are captured immediately before weight export;
they include process initialization, loading, compilation, and earlier saves,
but not the current export's duration or gaps between resumed sessions.
Exact resumes preserve cumulative time through the optimizer state; weights-only
initialization starts a new timer. Older resume states start counting from the
new session and set `timing_history_complete` to false.

Text-encoder and optimizer quantization remain separate. To keep cached text
embeddings from changing storage dtype when switching presets, select an explicit
text-encoder storage dtype rather than **Inherit**. Changes to other inherited
encoder quantization settings can still require a cache rebuild. No preset
automatically enables text-encoder quantization.

## Memory and text caching

For adapter training, frozen SDNQ storage reduces base-weight memory while the
adapter remains trainable. BF16 base weights are supported when they fit.
Full finetuning also needs gradients and optimizer state; training-mode SDNQ,
checkpointing, and cached conditioning are useful on 24 GB GPUs. Optimizer-state
quantization and CPU offload are separate settings, and offload consumes system RAM.

**Text caching is a separate memory phase.** Qwen3.5-9B text weights alone occupy
roughly 15 GiB in BF16. Unloading it after caching helps training memory, but does
not make the initial encoding step fit a smaller GPU. For lower caching memory,
use **Quantize the text encoder**, **uint8**, and **Text cache batch size = 1**.
Starting from original BF16 weights avoids stacking another quantization pass on
an already lossy source. Quantization can change the resulting embeddings.

```toml
[train]
cache_text_embeddings = true
caption_variations = 1
text_cache_batch_size = 1

[quant]
quantize_text_encoder = true
text_encoder_weights_dtype = "uint8"
```

These are overrides to a preset, not a complete config. `caption_variations > 0`
uses a persistent SQLite cache; it controls augmentation attempts per image,
not a guarantee of distinct captions. Increase it when using caption/tag
augmentation. Zero selects fixed-caption RAM caching and requires augmentation off.
Exact caption matches share embeddings. A complete compatible SQLite cache skips
loading the encoder on later runs; changing encoder settings can require new entries.

MOSS BF16 text embeddings have **4,096 features per token: 8 KiB/token**, or
**4 MiB per unique 512-token caption**, before database overhead. Stored FP32
latents occupy about **2 MiB per 1024² image resolution**. Plan disk space for
unique captions and resolution variants, not just image count.
See [caption-cache behavior](docs/caption-variation-cache.md); its older Mage
storage examples use a different embedding width.

A short **MOSS full-finetuning DDP test** on two RTX 4090s at **512px, batch 1 per
GPU** completed ten updates at **15.72 GiB allocated / 16.09 GiB reserved per GPU**,
using UINT8 SDNQ, Adafactor, cached short captions, checkpointing, and compile,
without CPU optimizer offload. This repeated one image and excluded checkpoint
saving; it is not a memory guarantee for the 1024px presets or longer captions.
See [MOSS validation](docs/moss-image-experimental.md#initial-validation-rtx-4090).

DDP replicates the model on each GPU; two 24 GB cards do not form a single 48 GB
memory pool. Historical [Mage VRAM](docs/readme-vram-benchmark.md) and
[batch-size](docs/batch-size-vram-benchmark.md) measurements apply to Mage, not MOSS.

## Datasets and multi-resolution

A single dataset folder or multiple subsets can supply images and captions.
Adding subset rows in the GUI disables the main folder and repeat count;
subsets replace that source. Each subset can have its own repeat count.

For multiple resolution tiers with different batch sizes, remove the preset's
single `dataset.resolution` field and apply these overrides:

```toml
[dataset]
resolutions = [512, 768, 1024]
max_bucket_reso = 1536

[train]
batch_size = { "512" = 4, "768" = 2, "1024" = 1 }
pack_resolutions = false
```

The batch sizes are examples, not measured capacity limits. Resolutions are
**area budgets**, not fixed square sizes; `max_bucket_reso` independently caps
each side. Batch-size keys must match the declared tiers. Cache every tier before
training. Each tier normally adds another view of each source; images too small
to produce a distinct larger bucket are handled by the tier-collapse setting.

**Multi-resolution datasets and packed attention are separate choices.**
`pack_resolutions = true` batches different shapes together and requires one
integer `batch_size`, rather than a per-tier map. The MOSS presets enable packing
with `torch_varlen`; set it to false for the per-tier example above. Packing,
batch size, and accumulation affect optimizer updates per epoch—compare runs by
updates and samples seen as well as epochs.

### Staged resolution LoRA training (experimental)

[Staged MOSS preset](configs/moss-image-lora-staged.toml) trains one resolution tier
at a time, with its own per-GPU batch size. It supports PEFT LoRA and LyCORIS
LoCon, cached latents, and packed varlen attention. This mode uses **successful
optimizer updates**, never epochs: `train.max_steps` is required; `epochs` and
`save_every_epochs` are rejected. Existing non-staged runs are unchanged.

```toml
[resolution_schedule]
enabled = true
save_stage_end = true
stages = [
  {resolution = 512, share = 0.20, lr_mul = 0.8},
  {resolution = 768, share = 0.20, lr_mul = 0.9},
  {resolution = 1024, share = 0.60, lr_mul = 1.0},
]
```

With `max_steps = 560`, `dataset.resolutions = [512, 768, 1024]`, and
`batch_size = {512 = 24, 768 = 12, 1024 = 7}`, this allocates 112/112/336
updates. Shares sum to one; rounding preserves the total and rejects empty
stages. The GUI's **Staged Resolutions** table previews the exact budgets and
effective batch sizes. Configure batches under **Batching** and matching tiers
under **Dataset**. Windows requires a compatible `flash_attn_2` installation.

Packing combines aspect ratios only within the active tier. Shuffled sample
streams wrap to fill complete batches; repeats and subsets retain their dataset
weights. Small sources remain eligible at later tiers without forced upscaling,
reusing identical latent files. Cache all tiers before starting. Shares divide
updates, not wall-clock time or image exposure.

The GUI's **LR %** column scales the scheduled LR per stage: 80% at 512, 90% at
768, and 100% at 1024 in the preset. `lr_mul` must be greater than zero and at
most one; older TOMLs without it keep 100%. This applies to every parameter
group without compounding across updates. Logs show the LR actually used.
Warmup and decay still apply: a decaying schedule does not restart or regain its
configured peak when the 1024 stage begins.

The optimizer and LR schedule continue across stages, with one warmup. New
shapes can compile at transitions. Skipped nonfinite attempts do not advance the
stage or scheduler; repeated nonfinite updates still stop training. Metrics show
stage progress, images consumed (including skipped attempts), and effective batch.
Optional stage saves use normal step checkpoint filenames; the final stage uses
the normal final save. `save_every_steps` also works.

Launch the preset through the GUI or `python -m trainer.training.train
configs/moss-image-lora-staged.toml`. `--dry-run` performs one successful update
per stage without saving; `--no-save` disables periodic, stage-boundary, and final
checkpoints.

Full-state resume requires the same stage plan (including LR multipliers), data manifest, batch sizes,
accumulation, and GPU count; enable optimizer-state saving for this. Weights-only
continuation starts a fresh plan and schedule. Texture/timestep curricula, OT,
Self-Flow, gradient release, preservation, and decoded perceptual/adversarial
experiments cannot be combined with staged resolutions in this first version.

### MOSS block learning rates (experimental)

Under **Preferences → Enable advanced experimental settings**, open
**Experimental → Experimental MOSS block learning rates**. The range table sets
training LR multipliers for PEFT LoRA or LyCORIS LoCon joint blocks **0–41**;
both ends are inclusive. Defaults are disabled and ×1.0 throughout.

```toml
[block_lr]
enabled = true
ranges = [
  {start = 0, end = 24, lr_mul = 0.75},
  {start = 25, end = 32, lr_mul = 1.0},
  {start = 33, end = 41, lr_mul = 1.0},
]
```

For a separate comparison, change the last multiplier to `1.5`. These are
experimental settings, not established quality presets. Unlisted blocks stay
at ×1.0; ranges cannot overlap. Zero freezes the adapters in that range.
Text-refiner adapters keep their existing component LR.

Multipliers compose with component LRs, the LR scheduler, and resolution-stage
or curriculum multipliers. They do not change rank, alpha, or inference weights.
Logs report the ranges, and checkpoints retain the configuration for TOML
recovery. Full-state resume requires the same effective block multipliers;
use weights-only continuation to start a different LR plan.

## Timestep sampling and detail losses

**Method → Flow Matching** shows a live timestep probability plot. Adjust the
sampler, sigmoid scale, or shift to preview its distribution. `t=0` is clean and
`t=1` is pure noise; the plot shows sampling frequency, not loss weighting.

The presets use **logit-normal sampling with shift 1**. The optional `moss`
sampler implements Kohaku's static MOSS recipe:

```toml
[flow]
timestep_sample_method = "moss"
shift = 1.0
sigmoid_scale = 1.0
flux_shift = false
```

Selecting it locks sigmoid scale and shift in the GUI to prevent a second
transformation of that distribution; conflicting explicit TOML values are rejected.
Its defaults are `mu=-1.09`, `std=0.95`, `cutoff=0.95`, a 0.01 uniform mixture,
and a 1000-step grid. This is the static recipe, not a staged pretraining schedule.
MOSS's inference shift of 5 does not prescribe the training distribution.

Optional **HF loss with SNR gating** and **paired texture loss** are documented
in the [detail-loss guide](docs/detail-losses.md). They change the objective;
leave them off when establishing a baseline or comparing trainers.

## Saving and using MOSS adapters

**Training → Checkpoints → MOSS Q/K/V saving format** defaults to **fused** for
full finetunes and plain LoRA/LoCon, for compatibility with inference loaders that
expect fused QKV. Separated export remains available; LoKr/DoRA retain their
native layout. Fused LoRA/LoCon files can be larger while representing the same
updates. Convert an existing adapter without overwriting it:

```bash
python lora_qkv_convert.py model.safetensors --format fused
```

Running `python lora_qkv_convert.py` without arguments opens its small GUI.
The footer's **Utils** menu opens the alpha/strength editor, MOSS Q/K/V converter,
LoRA rank resizer, and training TOML recovery. **Console** is available in the top tab bar, while
**Preferences** lives beside the run controls.

Q/K/V conversion works in both directions. Separated output removes exactly unused
columns, recovering the compact ranks of BluFlow's block-diagonal fused exports
without changing their strength. Arbitrary dense fused adapters may grow when split.

Rank resizing uses SVD on CPU and creates a new file. Smaller ranks approximate the
original update and can lose detail; alpha scaling is preserved. For fused MOSS
joint Q/K/V, the chosen rank applies to each projection, and the output is fused
again. Plain linear LoRA and LoCon with a 1×1 up projection are supported; DoRA,
LoKr and LoHa resizing is rejected. These commands also work outside the GUI:

```bash
python -m trainer.tools.lora_alpha_edit
python lora_qkv_convert.py model.safetensors --format separated
python lora_resize.py model.safetensors --rank 8
python lora_resize.py model.safetensors --method sv_fro --threshold 0.99
```

The resize window defaults to **sv_fro**, which selects a rank per projection to
retain the requested Frobenius norm. It uses sd-scripts' convention: `0.99` means
at least **98.01% squared energy**, before output rounding. There is no extra rank
cap unless enabled; existing ranks are never increased. An optional `--rank N`
caps sv_fro ranks, and the result reports any modules that miss the target because
of that cap. **Fixed rank** remains available; CLI calls without `--method` retain
the original fixed-rank behavior. Retention is a weight-space measure, not a
guarantee of visual fidelity.

**Utils → Recover training TOML from checkpoint** reads BluFlow's embedded
`training_config` without loading weights. It restores the original recipe,
including texture curriculum phases, crop settings, captions and dataset subsets.
Saved defaults are retained, resume instructions are cleared, and existing files
are never overwritten. Open the recovered TOML in the trainer and review local
model/dataset paths, output naming and GPU selection before starting a fresh run.
The CLI equivalent is:

```bash
python -m trainer.tools.recover_config adapter.safetensors -o recovered.toml
```

Files without readable BluFlow training metadata cannot recover a complete recipe.
Settings from edited adapters describe their original training, not necessarily
their current rank or strength. Unknown or incompatible settings are preserved
and reported for review; optimizer state and dataset files are not recovered.

See the [QKV export guide](docs/moss-image-experimental.md#qkv-saving-format-and-existing-loras).

**Method → Adapter → Train MOSS text-refinement adapters** controls adaptation of
caption features inside the transformer. It defaults to on for existing configs;
disable it to train only the main blocks. It does not train Qwen or modify cached
embeddings. Keep the setting unchanged when resuming a run.

## GUI and training queue

<a href="docs/images/trainer-gui.png"><img src="docs/images/trainer-gui.png" alt="BluFlow-Trainer GUI layout with model, batching, compilation, checkpoint, and training controls" width="960"></a>

*Earlier GUI screenshot; model labels and controls have evolved.*

The config picker lists TOMLs **newest first**. **Autosave**, beside the run name,
is on by default and saves when starting training—not while editing. With it off,
Start uses a temporary TOML; explicit Save and Save As still work. Settings sections
can collapse individually, with Expand all / Collapse all on each tab.

Use **Preferences** to choose a TOML save folder, or launch with
`python -m trainer.gui --config-dir "path/to/my configs"`. Without a saved folder,
the GUI uses `configs/`. Preferences retain the legacy `mage-flow-trainer/gui.json`
location under the OS configuration directory so the rename preserves your settings.

**Save & add to queue** validates, saves, and snapshots the current editor settings,
even when Autosave is off. **Run Queue → Add TOMLs** snapshots existing files.
Later source-file edits do not change queued settings. Give each run a distinct
output directory/run name; relative paths resolve from the trainer directory.

If Run Queue contains TOMLs and you press **Start Training** for the current
editor, a confirmation asks whether to start only the loaded TOML. **Yes** runs
that configuration and keeps the queue intact. **No** (or closing the dialog)
starts nothing and switches to **Run Queue**.

You can add runs while training and reorder or remove pending entries; the active
job is protected. Errors stop the queue by default. Enable **Continue queue after
errors** before starting to skip failed configs. Failed caching skips that config's
training, and **Stop** always cancels the remaining jobs. Select a failed entry to
read its error or **Retry selected** to retry its saved settings. The queue and
snapshots last for the current GUI session.

After each queued run successfully writes its final checkpoint, the GUI saves
`metrics-final.png` in that run's output directory. The image includes the latest
metrics even when another tab is selected, before the next run resets the graphs.
Image export failures are logged and do not stop the queue.

**Live Metrics** includes total elapsed time for the current training process,
including loading, text caching, compilation, and saves. Separate latent-cache
jobs and earlier training sessions are excluded.

Memory readings use **GiB** (1024³ bytes). **Peak allocated** is PyTorch's highest
allocated memory since startup; **allocated** is its current live allocation;
**reserved** includes the allocator's reusable cache; **device used** is current
device-wide usage reported by CUDA, including other processes and driver overhead.
Reserved minus allocated is not all wasted memory: much of it can be reused by
the next step. Per-rank readings are slash-separated. The memory graph shows peak
allocated memory; the header shows the current readings. Older GB logs are
converted to GiB when displayed.

### Updates and dependency repair

The launcher checks the current branch's configured upstream, including forks,
and offers a **fast-forward-only** update. Local tracked edits or divergent commits
require manual updating; it does not reset or stash them. The old GitHub repository
URL currently redirects to BluFlow-Trainer.

The updater checks dependency-file contents **and installed package versions/Git
revisions**. Accepting repair restores the repository pins in either direction:
older LyCORIS is updated, and newer unvalidated SDNQ is **downgraded to 0.2.9**.
It uses the active venv's pip or locked uv sync, verifies the result, and leaves
unrelated additional packages alone. Windows package updates wait for the GUI
process to close so Qt DLLs are released before installation.

Choosing No keeps the existing environment. Failed installs stop restart and log
to `.mageflow-trainer-update.log` inside the venv. `--no-update-check` skips Git
checking; `--no-dependency-check` skips package checking. Matching environments
are not reinstalled on each launch, and global Python is not modified.

## Model support tiers

| Tier | Models | Maintenance policy |
|---|---|---|
| **1 — First-class** | **MOSS-Image Base/SFT** | Primary development, correctness testing, and bug-fix priority. Highest level of support. |
| **2 — Promising integrations** | Reserved; **Krea2 is a candidate, not implemented** | Selected future flow models, below MOSS priority. |
| **3 — Limited support** | **Mage-Flow, FLUX.3 Image Klein Base** | Available today with lower maintenance priority; support may be discontinued or removed. |

Tier changes do not alter existing checkpoint formats or training settings.

- **Mage-Flow:** uses its own transformer, Qwen3-VL conditioning, and matching
  VAE. See the [training guide](docs/training-guide.md),
  [benchmark preset](configs/mageflow-readme-benchmark.toml), and
  [finetuning measurements](docs/bisque-finetune-smoke.md). These are Mage-specific.
- **FLUX.3 Image Klein Base:** text-to-image adapters and full finetuning only;
  reference-image editing and distilled flow-map checkpoints are unsupported.
  Use its own package and VAE. See the [Flux3 guide](docs/flux3-experimental.md),
  [LoCon preset](configs/flux3-locon-experimental.toml), and
  [finetuning preset](configs/flux3-finetune-experimental.toml).

### Experimental features

MOSS's first-class status does not make every optional backend established.
**ConvRot INT8** is an experimental MOSS BF16 LyCORIS LoCon backend, using native
BF16 source weights and frozen quantization. It does not support full finetuning
or other adapter algorithms. Install the optional `convrot` dependencies to use it;
see the [MOSS guide](docs/moss-image-experimental.md) for restrictions.

The GUI's **Preferences → Enable advanced experimental settings** toggle reveals
research controls and is remembered across launches; it does not enable their losses.
Configs with active experiments always reveal their controls regardless of this preference.
**Self-Flow, dual-timestep noising, RTI, decoded
perceptual/eye LPIPS, and eye adversarial loss are not supported for MOSS.**
Their separate Mage workflows are described in the [Self-Flow](docs/self-flow-training.md),
[decoded-loss](docs/decoded-perceptual-loss.md), and
[adversarial-loss](docs/eye-adversarial-loss.md) guides. Mage's
[compressed modulation](docs/compressed-modulation.md) and
[compiled loaded-text-encoder benchmark](docs/text-encoder-large-batch-benchmark.md)
are also separate from MOSS training.

## Documentation

- [MOSS-Image model loading, training, validation, and export](docs/moss-image-experimental.md)
- [Caption variation cache and invalidation](docs/caption-variation-cache.md)
- [HF and paired texture losses](docs/detail-losses.md)
- [Concept sliders](docs/concept-sliders.md)
- [SDNQ update-rounding diagnosis](docs/sdnq-integer-update-rounding.md)
- [General configuration and legacy Mage workflows](docs/training-guide.md)
- [FLUX.3 integration and limitations](docs/flux3-experimental.md)

The combined trainer is **GPL-3.0**. Vendored components retain their respective
notices; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
