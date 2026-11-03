"""Benchmark frozen MOSS INT8 kernels without loading models or training."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--device', type=int, default=0, help='Index among CUDA_VISIBLE_DEVICES')
    parser.add_argument('--all-devices', action='store_true', help='Calibrate visible GPUs sequentially')
    parser.add_argument('--force', action='store_true', help='Replace matching calibration results')
    parser.add_argument('--cache-dir', type=Path, help='Override BLUFLOW_INT8_CACHE_DIR')
    args = parser.parse_args()
    from trainer.tools.windows_toolchain import configure_windows_triton_compiler
    configure_windows_triton_compiler()
    from trainer.training.config import load_config
    cfg = load_config(args.config)
    if cfg.train.model_family != 'moss_image' or not cfg.quant.frozen_matmul:
        parser.error('Select MOSS Balanced/Fast or custom frozen INT8 matmul before calibrating')
    from trainer.tools.windows_toolchain import prepare_windows_toolchain, check_windows_triton_compiler
    prepare_windows_toolchain(required=True)
    check_windows_triton_compiler()
    import torch
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required')
    devices = list(range(torch.cuda.device_count())) if args.all_devices else [args.device]
    if any(i < 0 or i >= torch.cuda.device_count() for i in devices):
        parser.error('Device index is outside the visible GPU list')
    # Select the device before SDNQ imports and its backend capability detection.
    torch.cuda.set_device(devices[0])
    from trainer.training.int8_calibration import moss_shapes, prepare
    shapes = moss_shapes(cfg.train.model_path)
    completed = 0
    for i in devices:
        torch.cuda.set_device(i)
        result = prepare(torch.device('cuda', i), shapes, mode='auto', force=args.force, root=args.cache_dir)
        completed += result is not None
    print(f'INT8 calibration: {completed}/{len(devices)} GPUs ready. Training reuses matching profiles.', flush=True)


if __name__ == '__main__':
    main()
