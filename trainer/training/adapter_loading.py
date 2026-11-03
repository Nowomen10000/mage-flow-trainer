"""Strict weights-only continuation of exported adapters; never merge into the base."""
from pathlib import Path
import json
import re

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def validate_weights_source(cfg):
    if not cfg.is_lora or cfg.adapter.init != "random":
        raise ValueError("Weights-only continuation requires an adapter and adapter.init='random'")
    source = Path(cfg.train.resume_from)
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise ValueError("Weights-only continuation needs an adapter .safetensors file")


def validate_new_phase_output(cfg):
    # A fresh phase restarts epoch/step numbering. Avoid overwriting or pruning
    # the original run, including when the source is an intermediate export.
    out_dir = Path(cfg.train.output_dir) / cfg.train.run_name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Weights-only continuation requires a new run_name or empty output directory")


def _native_keys(state, expected):
    """Map native or flattened Comfy names using actual model targets, not guesses."""
    aliases = {}
    targets = set()
    for key in expected:
        # Target paths can contain underscores; only split at a known tensor name.
        match = re.match(r'^(.*)\.(lora_[AB]\.|lora_(?:up|down)\.|lokr_|dora_scale|alpha)', key)
        if match:
            targets.add(match.group(1))
    targets |= {re.sub(r'\.proj\.[qkv]$', '.proj.qkv', t) for t in list(targets)
                if re.search(r'\.proj\.[qkv]$', t)}
    for target in targets:
        bare = target.removeprefix('diffusion_model.')
        for alias in (target, bare, 'lora_unet_' + bare.replace('.', '_')):
            aliases[alias] = target
    ordered = sorted(aliases, key=len, reverse=True)
    result = {}
    for key, value in state.items():
        # Native paths may contain dots; match longest target first.
        target = next((a for a in ordered if key.startswith(a + '.')), None)
        native = aliases[target] + key[len(target):] if target else key
        if native in result:
            raise ValueError(f"Duplicate adapter tensor after name conversion: {native}")
        result[native] = value
    return result


def _unfuse_for_training(state, expected):
    """Invert our block-diagonal export exactly, without increasing training rank.

    General fused adapters may be loaded if each native projection uses their
    full rank. A block-diagonal export can instead recover its original factors.
    Any other rank reduction is rejected, never approximated with SVD.
    """
    state = dict(state)
    for key in list(state):
        if not re.search(r'\.layers\.\d+\.proj\.qkv\.(lora_down|lora_A)\.weight$', key):
            continue
        down_suffix = '.lora_down.weight' if key.endswith('.lora_down.weight') else '.lora_A.weight'
        up_suffix = '.lora_up.weight' if down_suffix == '.lora_down.weight' else '.lora_B.weight'
        prefix = key.removesuffix('qkv' + down_suffix)
        fused = prefix + 'qkv'
        keys = {fused + s for s in (down_suffix, up_suffix, '.alpha')}
        if {k for k in state if k.startswith(fused + '.')} != keys:
            raise ValueError(f"{fused}: expected complete plain fused LoRA factors")
        if any(k.startswith(prefix + p + '.') for k in state for p in ('q', 'k', 'v')):
            raise ValueError(f"{fused}: ambiguous fused and separated QKV")
        down, up, alpha = (state[fused + s] for s in (down_suffix, up_suffix, '.alpha'))
        native = [prefix + p for p in ('q', 'k', 'v')]
        if not all(t + down_suffix in expected and t + up_suffix in expected for t in native):
            raise ValueError(f"{fused}: QKV targets do not match the configured adapter")
        ranks = [expected[t + down_suffix].shape[0] for t in native]
        rows = [expected[t + up_suffix].shape[0] for t in native]
        if down.ndim != 2 or up.ndim != 2 or up.shape != (sum(rows), down.shape[0]):
            raise ValueError(f"{fused}: invalid fused factor shapes")
        compact = down.shape[0] == sum(ranks)
        if not compact and not all(r == down.shape[0] for r in ranks):
            raise ValueError(f"{fused}: rank mismatch; cannot losslessly reduce this fused adapter")
        row_start = col_start = 0
        for target, rank, nrows in zip(native, ranks, rows):
            part = up[row_start:row_start + nrows]
            if compact:
                if torch.count_nonzero(part[:, :col_start]) or torch.count_nonzero(part[:, col_start + rank:]):
                    raise ValueError(f"{fused}: nonzero off-diagonal factors; cannot recover original rank")
                d = down[col_start:col_start + rank]
                u = part[:, col_start:col_start + rank]
            else:
                d, u = down, part
            state[target + down_suffix] = d.contiguous()
            state[target + up_suffix] = u.contiguous()
            state[target + '.alpha'] = alpha.double() * (rank / down.shape[0])
            row_start += nrows
            col_start += rank
        for k in keys:
            del state[k]
    return state


def load_adapter_weights(model, path):
    """Validate the complete file before copying anything; optimizer/RNG untouched."""
    lycoris = hasattr(model, '_lycoris_config')
    if lycoris:
        from .lycoris import lycoris_state_dict, load_lycoris_state_dict
        expected = lycoris_state_dict(model)
    else:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        expected = {'diffusion_model.' + k: v for k, v in get_peft_model_state_dict(model).items()}
        config = model.peft_config['default']
        alpha = float(getattr(config, 'lora_alpha', getattr(config, 'alpha', 1)))
        for key in list(expected):
            if '.lora_A.' in key or '.lokr_' in key:
                target = key.split('.lora_A.')[0] if '.lora_A.' in key else key.split('.lokr_')[0]
                expected[target + '.alpha'] = torch.tensor(alpha)
    if not expected:
        raise ValueError("No adapters found for weights-only continuation")
    with safe_open(path, framework='pt', device='cpu') as reader:
        metadata = reader.metadata() or {}
    if metadata.get('model') and metadata['model'] != getattr(model, 'model_family', 'mage_flow'):
        raise ValueError("Adapter model family differs from the configured base")
    saved = json.loads(metadata.get('adapter_config', '{}'))
    if lycoris:
        current = model._lycoris_config
        for option in ('kind', 'lycoris_algo'):
            if option in saved and saved[option] != current[option]:
                raise ValueError(f"Adapter {option} differs from the saved configuration")
        if current['lycoris_algo'] == 'dora' and saved.get('lycoris_wd_on_output', True) != current['lycoris_wd_on_output']:
            raise ValueError("Adapter DoRA decomposition direction differs from the saved configuration")
    elif saved.get('use_rslora') or saved.get('use_dora') or saved.get('alpha_pattern') or saved.get('rank_pattern'):
        raise ValueError("Weights-only loading does not support external PEFT scaling/rank overrides")
    state = _unfuse_for_training(_native_keys(load_file(path, device='cpu'), expected), expected)
    if set(state) != set(expected):
        missing, extra = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
        raise ValueError(f"Adapter targets/algorithm mismatch: missing {missing[:4]}, unexpected {extra[:4]}. "
                         "Match adapter kind, components, and MOSS text-refiner selection.")
    for key, value in state.items():
        reference = expected[key]
        if value.shape != reference.shape:
            raise ValueError(f"{key}: shape {tuple(value.shape)} != {tuple(reference.shape)}; match adapter rank/algorithm")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{key}: adapter weights must be finite floating-point tensors")
        if key.endswith('.alpha') and not torch.allclose(value.double(), reference.detach().cpu().double(), rtol=1e-6, atol=1e-7):
            raise ValueError(f"{key}: alpha mismatch ({value.item()} vs {reference.item()}); match the saved alpha")
        if not torch.isfinite(value.to(dtype=reference.dtype)).all():
            raise ValueError(f"{key}: values overflow the configured adapter dtype")
    if lycoris:
        # Do not change alpha: LyCORIS also caches its derived multiplier.
        state.update({k: v for k, v in expected.items() if k.endswith('.alpha')})
        load_lycoris_state_dict(model, state)
    else:
        set_peft_model_state_dict(model, {k.removeprefix('diffusion_model.'): v
                                         for k, v in state.items() if not k.endswith('.alpha')})
    from ..modeling.checkpoint_metadata import checkpoint_identity
    return dict(checkpoint_identity(path), model_type='adapter', source_step=metadata.get('step'), mode='weights',
                optimizer_restored=False, schedule_restarted=True)
