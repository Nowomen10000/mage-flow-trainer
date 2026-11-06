"""Pre-training INT8 kernel calibration. No model weights or global RNG are used.

A profile fixes launch tiles by (N, K, weight layout) and a few token anchors.
Unseen M values reuse the nearest anchor; kernels can still JIT compile for M.
Profiles are local JSON, never executable code. Only current candidate tiles load.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
from importlib.metadata import version

SCHEMA = 1
TOKEN_ANCHORS = (512, 4096, 16384)
_ACTIVE = {}


def has_kernel_overrides():
    return any(k.startswith('SDNQ_TRITON_MM_') for k in os.environ)


def cache_root():
    if value := os.environ.get('BLUFLOW_INT8_CACHE_DIR'):
        return Path(value).expanduser()
    base = os.environ.get('LOCALAPPDATA') if os.name == 'nt' else os.environ.get('XDG_CACHE_HOME')
    return (Path(base) if base else Path.home() / '.cache') / 'bluflow-trainer' / 'int8-kernels'


def candidate_tiles():
    from sdnq.kernels import triton_scaled_mm as kernel
    configs = getattr(kernel, 'matmul_configs', None)
    if configs is None:
        configs = [*kernel.autotune_configs, *kernel.small_autotune_configs]
    return [dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages) for c in configs]


def eligible_tiles(candidates, a, b, sa, sb):
    """Use upstream's hardware/shape pruning before compiling calibration tiles."""
    from sdnq.kernels import triton_scaled_mm as kernel
    if hasattr(kernel, 'min_block_size'):
        return candidates  # Keep the measured 0.2.4 path unchanged.
    import torch
    import triton
    configs = [triton.Config({k: v for k, v in tile.items() if k not in ('num_warps', 'num_stages')},
                            num_warps=tile['num_warps'], num_stages=tile['num_stages'])
               for tile in candidates]
    # prune_configs only needs C's element size, not an output allocation.
    args = dict(a_ptr=a, b_ptr=b, c_ptr=torch.empty((), device=a.device, dtype=torch.bfloat16),
                scale_a_ptr=sa, scale_b_ptr=sb, bias_ptr=None, bias_ndim=0,
                M=a.shape[0], N=b.shape[1], K=a.shape[1],
                scale_a_is_tensorwise=sa.numel() == 1, scale_b_is_tensorwise=sb.numel() == 1)
    pruned = kernel.prune_configs(configs, args)
    tiles = [dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages) for c in pruned]
    return [tile for tile in tiles if tile in candidates]


def moss_shapes(model_path):
    from ..modeling.moss_loader import component_config
    from ..modeling.moss_image import MOSSParams
    from sdnq.kernel_wrappers import use_contiguous_int8_mm
    data = json.loads(component_config(model_path, 'transformer').read_text(encoding='utf-8'))
    d, f = data.get('dim', MOSSParams.dim), data.get('ffn_dim', MOSSParams.ffn_dim)
    # Forward [tokens,in] @ weight.T; input gradient [tokens,out] @ weight.
    # SDNQ check_mats normalizes BOTH paths to its selected weight layout.
    return sorted(set([(d, d, use_contiguous_int8_mm), (2*f, d, use_contiguous_int8_mm),
                       (d, f, use_contiguous_int8_mm), (d, 2*f, use_contiguous_int8_mm), (f, d, use_contiguous_int8_mm)]))


def _driver_version():
    try:
        lib = ctypes.CDLL('nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1')
        value = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(value)) == 0:
            return value.value
    except (OSError, AttributeError):
        pass
    return None


def _driver_build():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                       timeout=10, text=True, stderr=subprocess.DEVNULL).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def fingerprint(device, shapes, tokens):
    import torch
    import sdnq.kernels.triton_scaled_mm as kernel
    props = torch.cuda.get_device_properties(device)
    return dict(schema=SCHEMA, gpu=props.name, uuid=str(getattr(props, 'uuid', 'unknown')),
                capability=list(torch.cuda.get_device_capability(device)),
                multiprocessors=props.multi_processor_count, memory=props.total_memory,
                system=platform.system(), machine=platform.machine(), torch=str(torch.__version__),
                cuda=torch.version.cuda, driver=_driver_version(), driver_build=_driver_build(), triton=version('triton-windows' if os.name == 'nt' else 'triton'),
                sdnq=version('sdnq'), dtype='bfloat16', shapes=[list(s) for s in shapes], tokens=list(tokens),
                kernel_hash=hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
                calibration_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                candidates=candidate_tiles())


def profile_path(identity, root=None):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return (Path(root) if root else cache_root()) / (digest + '.json')


def read_profile(path, identity):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data['identity'] != identity or not isinstance(data['entries'], list):
            return None
        seen = set()
        for e in data['entries']:
            key = (e['n'], e['k'], e['contiguous'], e['m'])
            if key in seen or [e['n'], e['k'], e['contiguous']] not in identity['shapes'] or e['m'] not in identity['tokens']:
                return None
            seen.add(key)
            if e['tile'] != 'torch' and e['tile'] not in identity['candidates']:
                return None
            if not isinstance(e['ms'], (int, float)) or not math.isfinite(e['ms']) or e['ms'] <= 0:
                return None
        # Every matrix/layout needs at least one validated launch; large anchors
        # can be omitted if this card lacks the benchmark memory headroom.
        if not all(any([e['n'], e['k'], e['contiguous']] == s for e in data['entries']) for s in identity['shapes']):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_profile(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as f:
            handle = Path(f.name)
            json.dump(data, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(handle, path)
    finally:
        if handle is not None:
            handle.unlink(missing_ok=True)


def activate(device, profile):
    import torch
    device = torch.device(device)
    _ACTIVE[device.index if device.index is not None else torch.cuda.current_device()] = profile


def has_profile(device):
    import torch
    device = torch.device(device)
    return device.type == "cuda" and (device.index if device.index is not None else torch.cuda.current_device()) in _ACTIVE


def clear_active():
    _ACTIVE.clear()


def select_tile(device, m, n, k, contiguous):
    import torch
    if device.type != 'cuda' or has_kernel_overrides():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    profile = _ACTIVE.get(index)
    if not profile:
        return None
    entries = [e for e in profile['entries'] if (e['n'], e['k'], e['contiguous']) == (n, k, contiguous)]
    if not entries:
        return None
    return min(entries, key=lambda e: abs(math.log2(max(m, 1) / e['m'])))['tile']


def launch_scaled_mm(a, b, sa, sb, dtype, tile):
    """SDNQ's original kernel/math, with an explicit tile and no autotuner."""
    import torch
    if tile == 'torch':
        # cuBLAS INT8 requires aligned dimensions. Pad tokens only; calibrated
        # MOSS N/K are aligned already. Keep INT32 accumulation and FP32 scales.
        padded = torch.nn.functional.pad(a, (0, 0, 0, (-a.shape[0]) % 8))
        result = torch._int_mm(padded, b.contiguous())[:a.shape[0]].float()
        return result.mul_(sa.reshape(-1, 1)).mul_(sb.reshape(1, -1)).to(dtype)
    from sdnq.kernels import triton_scaled_mm as kernel
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=dtype)
    grid = (((m + tile['BLOCK_SIZE_M'] - 1) // tile['BLOCK_SIZE_M']) *
            ((n + tile['BLOCK_SIZE_N'] - 1) // tile['BLOCK_SIZE_N']),)
    if hasattr(kernel, 'min_block_size'):  # SDNQ 0.2.4 ABI
        block = kernel.min_block_size
        kernel.sdnq_scaled_mm_kernel.fn[grid](
            a, b, out, None, sa, sb, m, n, k, int(b.is_contiguous()), 0,
            (m + block - 1) // block, (n + block - 1) // block,
            (k + block - 1) // block, str(a.dtype), str(dtype), **tile)
    else:
        # New SDNQ kernels also specialize scale layout and descriptor options.
        # Calibration handles row-wise INT8 only; retain INT32 accumulation and
        # ordinary loads so this path works on Ampere/Ada as well as newer GPUs.
        block = kernel.block_size_divisor
        kernel.sdnq_scaled_mm_kernel.fn[grid](
            a, b, out, None, sa, sb, m, n, k, 0, int(b.is_contiguous()),
            int(sa.numel() == 1), int(sb.numel() == 1), 0, 0,
            str(a.dtype), str(dtype), (m + block - 1) // block,
            (n + block - 1) // block, (k + block - 1) // block, block, **tile)
    return out


def _sample_indices(size):
    return sorted({0, max(0, size-2), size-1, size//2, *[min(x, size-1) for x in (1, 31, 63, 64, 127, 128, 255, 256, 511, 512)]})


def _reference(a, b, sa, sb):
    # Independent CPU INT64 oracle, sampling tile edges and the matrix tail.
    # Full model-sized CPU GEMMs would make calibration impractically slow.
    import torch
    rows, cols = _sample_indices(a.shape[0]), _sample_indices(b.shape[1])
    accum = a[rows].cpu().to(torch.int64) @ b[:, cols].cpu().to(torch.int64)
    result = accum.float() * sa.reshape(-1)[rows].cpu()[:, None]
    result = (result * sb.reshape(-1)[cols].cpu()[None, :]).to(torch.bfloat16)
    return rows, cols, result


def _measure(fn, device):
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    # Capture repeated launches to exclude Python dispatch and allocation gaps
    # from timings, especially for small matrices. Compilation happened above.
    graph = torch.cuda.CUDAGraph()
    repeats = 5
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            out = fn()
    graph.replay()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / repeats)
    return statistics.median(times)


def _fatal_cuda_error(exc):
    return any(s in str(exc).lower() for s in ('illegal memory access', 'device-side assert', 'unspecified launch failure'))


def calibrate(device, identity, log=print):
    import torch
    entries, rejected = [], []
    with torch.cuda.device(device):
        generator = torch.Generator(device=device).manual_seed(781032)
        # Smallest first makes it possible to omit only large memory-heavy probes.
        for n, k, contiguous in identity['shapes']:
            for m in sorted(identity['tokens']):
                free, _ = torch.cuda.mem_get_info(device)
                # Inputs, output, integer fallback output and a conservative temporary allowance.
                required = k*n*2 + m*k*2 + m*n*8 + 64*2**20
                if required > min(2*2**30, free * .5):
                    log(f'INT8 calibration: skip M={m} N={n} K={k}: insufficient benchmark headroom', flush=True)
                    continue
                log(f'INT8 calibration: testing M={m} N={n} K={k} contiguous={contiguous}', flush=True)
                a = torch.randint(-128, 128, (m+1, k), device=device, dtype=torch.int8, generator=generator)
                b = torch.randint(-128, 128, (k, n) if contiguous else (n, k), device=device, dtype=torch.int8, generator=generator)
                if not contiguous:
                    b = b.T
                sa = torch.rand(m+1, device=device, generator=generator) * .01
                sb = torch.rand(n, device=device, generator=generator) * .01
                a[0].zero_()
                oracle = _reference(a, b, sa, sb)
                winners = []
                # Probe odd M+1 as well as the representative M. No autotuning calls.
                tiles = eligible_tiles(identity['candidates'], a, b, sa, sb)
                for tile in [*tiles, 'torch']:
                    if tile == 'torch' and winners:
                        break
                    try:
                        result = launch_scaled_mm(a, b, sa, sb, torch.bfloat16, tile)
                        rows, cols, expected = oracle
                        actual = result[rows][:, cols].cpu()
                        del result
                        if not torch.equal(actual, expected):
                            raise ValueError('sampled INT64 correctness check failed')
                        # The aligned representative must pass too; M+1 alone
                        # exercises different masks and launch-grid boundaries.
                        keep = [i for i, r in enumerate(rows) if r < m]
                        aligned = launch_scaled_mm(a[:m], b, sa[:m], sb, torch.bfloat16, tile)
                        if not torch.equal(aligned[[rows[i] for i in keep]][:, cols].cpu(), expected[keep]):
                            raise ValueError('aligned INT64 correctness check failed')
                        del aligned
                        elapsed = _measure(lambda: launch_scaled_mm(a[:m], b, sa[:m], sb, torch.bfloat16, tile), device)
                        if not math.isfinite(elapsed) or elapsed <= 0:
                            raise ValueError('invalid timing')
                        winners.append((elapsed, tile))
                    except Exception as exc:
                        if _fatal_cuda_error(exc):
                            raise RuntimeError('CUDA context failed during calibration; restart the process') from exc
                        rejected.append(dict(m=m, n=n, k=k, contiguous=contiguous, tile=tile,
                                             reason=f'{type(exc).__name__}: {exc}'[:600]))
                        torch.cuda.empty_cache()
                if winners:
                    elapsed, tile = min(winners, key=lambda pair: pair[0])
                    entries.append(dict(m=m, n=n, k=k, contiguous=contiguous, tile=tile, ms=elapsed))
                    label = 'PyTorch INT8 fallback' if tile == 'torch' else str(tile)
                    log(f'INT8 calibration: M={m} N={n} K={k} contiguous={contiguous}: {elapsed:.3f} ms, {label}', flush=True)
                else:
                    log(f'INT8 calibration: no validated kernel for M={m} N={n} K={k}', flush=True)
                del a, b, sa, sb
        if not all(any((e['n'], e['k'], e['contiguous']) == tuple(s) for e in entries) for s in identity['shapes']):
            raise RuntimeError('INT8 calibration could not validate all projection layouts. '
                               'Free GPU memory and retry, or use the Standard preset.')
    return dict(identity=identity, entries=entries, rejected=rejected,
                validation='CPU INT64 sampled rows/columns including tile edges; aligned and odd token counts',
                timing='median of 7 CUDA-event graph replays, 5 launches each, after compilation and warmup')


def prepare(device, shapes, *, mode='auto', force=False, root=None, tokens=TOKEN_ANCHORS, log=print):
    """Isolate global RNG, including SDNQ's first-import capability probes."""
    import torch
    device = torch.device(device)
    if device.type != 'cuda' or torch.version.hip:
        raise ValueError('INT8 calibration currently supports NVIDIA CUDA only')
    with torch.cuda.device(device), torch.random.fork_rng(devices=[device]):
        return _prepare(device, shapes, mode=mode, force=force, root=root, tokens=tokens, log=log)


def _prepare(device, shapes, *, mode='auto', force=False, root=None, tokens=TOKEN_ANCHORS, log=print):
    """Return/activate a valid profile, or retain existing behavior when disabled."""
    import torch
    from sdnq import kernel_wrappers
    from filelock import FileLock
    if mode not in ('auto', 'cached', 'off'):
        raise ValueError('matmul_tuning must be auto, cached, or off')
    device = torch.device(device)
    if device.type != 'cuda' or torch.version.hip:
        raise ValueError('INT8 calibration currently supports NVIDIA CUDA only')
    index = device.index if device.index is not None else torch.cuda.current_device()
    _ACTIVE.pop(index, None)
    if mode == 'off' and not force:
        return None
    if has_kernel_overrides() or not kernel_wrappers.use_triton_scaled_mm:
        log('INT8 calibration skipped: explicit SDNQ overrides or Triton scaled matmul disabled.', flush=True)
        return None
    shapes = sorted(set(tuple(s) for s in shapes))
    tokens = sorted(set(tokens))
    if not shapes or not tokens or any(type(m) is not int or m < 8 or m > 65536 for m in tokens):
        raise ValueError('Calibration needs projection shapes and token anchors between 8 and 65536')
    if any(n < 8 or k < 8 or n % 8 or k % 8 for n, k, _ in shapes):
        raise ValueError('Calibration requires MOSS projection dimensions divisible by 8')
    with torch.cuda.device(device):
        identity = fingerprint(device, shapes, tokens)
        path = profile_path(identity, root)
        if not force and (cached := read_profile(path, identity)):
            activate(device, cached)
            log(f'INT8 calibration: loaded {path}', flush=True)
            return cached
        if mode == 'cached' and not force:
            log('INT8 calibration: no matching cache; retaining the existing kernel selection.', flush=True)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        log(f'INT8 calibration on {identity["gpu"]}: first use may take several minutes. Cache: {path}', flush=True)
        with FileLock(str(path) + '.lock', timeout=120):
            # Another process may have completed the same profile while we waited.
            data = None if force else read_profile(path, identity)
            if data is None:
                data = calibrate(device, identity, log)
                write_profile(path, data)
        activate(device, data)
        torch.cuda.empty_cache()
        return data


def prepare_training(cfg, accelerator):
    """Coordinate setup before models/caches are loaded or training collectives run."""
    if not cfg.quant.frozen_matmul:
        clear_active()
        return
    from accelerate.utils import broadcast_object_list
    # One rank at a time avoids simultaneous compiler/benchmark contention.
    # Broadcast failures too, so other ranks do not hang waiting for a failed tuner.
    for rank in range(accelerator.num_processes):
        error = [None]
        if accelerator.process_index == rank:
            try:
                prepare(accelerator.device, moss_shapes(cfg.train.model_path), mode=cfg.quant.matmul_tuning)
            except Exception as exc:
                error[0] = f'INT8 calibration failed on rank {rank}: {exc}'
        broadcast_object_list(error, from_process=rank)
        if error[0]:
            raise RuntimeError(error[0])
